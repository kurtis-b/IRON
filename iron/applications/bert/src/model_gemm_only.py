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

from .block.transformer_gemm_only import BertEncoderGemmOnly
from .model import EncoderEmbeddings
from .utils import assign


class EncoderBackboneGemmOnly:
    """Minimal encoder backbone used by the gemm_only NPU benchmark."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.embeddings = EncoderEmbeddings(config)
        self.encoder_context = AIEContext(use_runlist=False)
        self.encoder = BertEncoderGemmOnly(
            config,
            seq_len=seq_len,
            context=self.encoder_context,
        )
        self.dtype = config.aie_config.dtype

    def eval(self):
        self.embeddings.eval()
        self.encoder.eval()
        return self

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
        encoder_output, encoder_timings = self.encoder.forward_with_stage_timings(
            embedding_output.to(self.dtype),
            attention_mask=attention_mask,
        )
        return encoder_output.to(embedding_output.dtype), {
            "embedding_sec": embedding_end - embedding_start,
            **encoder_timings,
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
        self.encoder.assign_weights(combined_weights)
