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
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference used:
# https://medium.com/@alexmriggio/bert-for-sequence-classification-from-scratch-code-and-theory-fb88053800fa

import torch
import torch.nn as nn
from ..utils import assign
from .feed_forward import BertFeedForward
from .mha import BertAttention
from operators import AIEBERTEncoder


class BertLayer(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.attention = BertAttention(config, seq_len=seq_len)
        self.ffn = BertFeedForward(config, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        attention_mask,
    ) -> tuple[torch.Tensor]:
        self_attention_output, _ = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_output
        ffn_output = self.ffn(attention_output)
        return ffn_output


class BertEncoder(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        offload_individual_operator = [
            config.aie_config.use_aie_gemm == True,
            config.aie_config.use_aie_gelu == True,
            config.aie_config.use_aie_softmax == True,
            config.aie_config.use_aie_layernorm == True,
            config.aie_config.use_aie_elementwise_add == True,
            config.aie_config.use_aie_elementwise_mul == True,
            config.aie_config.use_aie_transpose == True,
            config.aie_config.use_aie_ffn == True,
            config.aie_config.use_aie_addandnorm == True,
        ]
        assert (
            self.config.aie_config.use_aie_bert_encoder
            and not any(offload_individual_operator)
            or not self.config.aie_config.use_aie_bert_encoder
        ), "Cannot mix Encoder runlist with individual AIE operators."
        offload_individual_ffn_operator = [
            # Don't check for aie gemm here since it's used in attention too
            config.aie_config.use_aie_gelu
            == True,
        ]
        assert (
            self.config.aie_config.use_aie_ffn
            and not any(offload_individual_ffn_operator)
            or not self.config.aie_config.use_aie_ffn
        ), "Cannot mix pipelined FFN with individual AIE operators."
        offload_individual_addandnorm_operator = [
            config.aie_config.use_aie_layernorm == True,
            config.aie_config.use_aie_elementwise_add == True,
        ]
        assert (
            self.config.aie_config.use_aie_addandnorm
            and not any(offload_individual_addandnorm_operator)
            or not self.config.aie_config.use_aie_addandnorm
        ), "Cannot mix pipelined Add & Norm with individual AIE operators."
        if config.aie_config.use_aie_bert_encoder:
            self.layer = [
                AIEBERTEncoder(
                    seq_len=seq_len,
                    hidden_size=config.model_config.hidden_size,
                    intermediate_size=config.model_config.intermediate_size,
                    num_heads=config.model_config.num_attention_heads,
                )
                for i in range(config.model_config.num_hidden_layers)
            ]
        else:
            self.layer = nn.ModuleList(
                [
                    BertLayer(config, seq_len=seq_len)
                    for i in range(config.model_config.num_hidden_layers)
                ]
            )

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
            )

        return hidden_states

    def assign_weights(self, combined_weights):
        if self.config.aie_config.use_aie_bert_encoder:
            for l in range(self.config.model_config.num_hidden_layers):
                self.layer[l].q_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.query.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].k_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.key.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].v_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.value.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].attn_output_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.output.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ln1_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ffn_up_weight = combined_weights[
                    f"bert.encoder.layer.{l}.intermediate.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ffn_down_weight = combined_weights[
                    f"bert.encoder.layer.{l}.output.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ln2_weight = combined_weights[
                    f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                ].to(self.config.aie_config.dtype)
        else:
            # Operators here could be offloaded to AIE
            for l in range(self.config.model_config.num_hidden_layers):
                self.layer[l].attention.output.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.weight"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.bias"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                )
                self.layer[l].attention.self.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.query.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.query.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.key.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.key.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.value.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.value.bias"
                    ].to(self.config.aie_config.dtype),
                )
                self.layer[l].ffn.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.weight"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.bias"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                )
