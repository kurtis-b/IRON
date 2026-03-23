# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from iron.common import AIEContext
from iron.operators.encoder.op import AIEBERTEncoder

from .model import EncoderEmbeddings
from .utils import assign


class EncoderBackboneOperatorRunlist:
    """Encoder backbone wrapper for the stitched encoder-operator runlist path."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.embeddings = EncoderEmbeddings(config)
        self.num_hidden_layers = int(config.model_config.num_hidden_layers)
        self.encoder_context = AIEContext(use_runlist=True)
        self.encoder_layer = AIEBERTEncoder(
            seq_len=seq_len,
            hidden_size=int(config.model_config.hidden_size),
            intermediate_size=int(config.model_config.intermediate_size),
            num_heads=int(config.model_config.num_attention_heads),
            use_pip_ffn=False,
            use_pip_addnorm=False,
            use_pip_mha=False,
            use_pip_an_ffn=False,
            use_static_runtime_weights=False,
            context=self.encoder_context,
        )
        self.encoder_layer.lazy_kernel_loading = False
        self.layer_weights = []
        self.dtype = config.aie_config.dtype

    def eval(self):
        self.embeddings.eval()
        return self

    def _assign_layer_weights(self, layer_idx):
        weights = self.layer_weights[layer_idx]
        self.encoder_layer.q_weight = weights["q_weight"]
        self.encoder_layer.k_weight = weights["k_weight"]
        self.encoder_layer.v_weight = weights["v_weight"]
        self.encoder_layer.attn_output_weight = weights["attn_output_weight"]
        self.encoder_layer.ffn_up_weight = weights["ffn_up_weight"]
        self.encoder_layer.ffn_down_weight = weights["ffn_down_weight"]
        self.encoder_layer.ln1_weight = weights["ln1_weight"]
        self.encoder_layer.ln2_weight = weights["ln2_weight"]

    def forward_with_stage_timings(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        embedding_start = time.perf_counter()
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        embedding_end = time.perf_counter()

        hidden_states = embedding_output.to(self.dtype).squeeze(0)
        runlist_start = time.perf_counter()
        for layer_idx in range(self.num_hidden_layers):
            self._assign_layer_weights(layer_idx)
            hidden_states = self.encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        runlist_end = time.perf_counter()

        dispatch_count = len(self.encoder_layer.runlist) * self.num_hidden_layers
        unique_instruction_binary_count = len(self.encoder_layer.kernels)
        output = hidden_states.unsqueeze(0)

        return output, {
            "embedding_sec": embedding_end - embedding_start,
            "operator_runlist_sec": runlist_end - runlist_start,
            "host_preprocess_sec": 0.0,
            "device_sync_sec": 0.0,
            "npu_dispatch_count": dispatch_count,
            "npu_unique_instruction_binary_count": (unique_instruction_binary_count),
        }

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        output, _ = self.forward_with_stage_timings(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return output

    def assign_backbone_weights(self, combined_weights):
        self.embeddings.word_embeddings.weight = assign(
            self.embeddings.word_embeddings.weight,
            combined_weights["embeddings.word_embeddings.weight"],
            "embeddings.word_embeddings.weight",
        )
        self.embeddings.position_embeddings.weight = assign(
            self.embeddings.position_embeddings.weight,
            combined_weights["embeddings.position_embeddings.weight"],
            "embeddings.position_embeddings.weight",
        )
        if self.embeddings.token_type_embeddings is not None:
            self.embeddings.token_type_embeddings.weight = assign(
                self.embeddings.token_type_embeddings.weight,
                combined_weights["embeddings.token_type_embeddings.weight"],
                "embeddings.token_type_embeddings.weight",
            )
        self.embeddings.LayerNorm.bias = assign(
            self.embeddings.LayerNorm.bias,
            combined_weights["embeddings.LayerNorm.bias"],
            "embeddings.LayerNorm.bias",
        )
        self.embeddings.LayerNorm.weight = assign(
            self.embeddings.LayerNorm.weight,
            combined_weights["embeddings.LayerNorm.weight"],
            "embeddings.LayerNorm.weight",
        )

        dtype = self.dtype
        self.layer_weights = []
        for layer_idx in range(self.num_hidden_layers):
            self.layer_weights.append(
                {
                    "q_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.attention.self.query.weight"
                    ].to(dtype),
                    "k_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.attention.self.key.weight"
                    ].to(dtype),
                    "v_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.attention.self.value.weight"
                    ].to(dtype),
                    "attn_output_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.attention.output.dense.weight"
                    ].to(dtype),
                    "ffn_up_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.intermediate.dense.weight"
                    ].to(dtype),
                    "ffn_down_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.output.dense.weight"
                    ].to(dtype),
                    "ln1_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"
                    ].to(dtype),
                    "ln2_weight": combined_weights[
                        f"encoder.layer.{layer_idx}.output.LayerNorm.weight"
                    ].to(dtype),
                }
            )
        if self.layer_weights:
            self._assign_layer_weights(0)
