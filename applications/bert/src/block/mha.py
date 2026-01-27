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

import math
import torch
import torch.nn as nn
from ..utils import assign
from operators import AIEGEMM
from operators import AIESoftmax
from operators import AIEElementwiseAdd
from operators import AIELayerNorm
from operators import AIEAddAndNorm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (
            config.model_config.hidden_size % config.model_config.num_attention_heads
            != 0
            and not hasattr(config, "embedding_size")
        ):
            raise ValueError(
                f"The hidden size ({config.model_config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.model_config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.model_config.num_attention_heads
        self.attention_head_size = (
            config.model_config.hidden_size // config.model_config.num_attention_heads
        )
        self.scaling = self.attention_head_size**-0.5

        if config.aie_config.use_aie_gemm:
            aie_gemm_config = {
                "num_aie_columns": 8,
                "tile_m": 64,
                "tile_k": 96,
                "tile_n": 48,
                "use_static_weight": True,
                "emulate_bf16_mmul_with_bfp16": True,
                "prio_accuracy": False,
            }
            self.query = AIEGEMM(
                M=512,
                K=config.model_config.hidden_size,
                N=config.model_config.hidden_size,
                **aie_gemm_config,
            )
            self.key = AIEGEMM(
                M=512,
                K=config.model_config.hidden_size,
                N=config.model_config.hidden_size,
                **aie_gemm_config,
            )
            self.value = AIEGEMM(
                M=512,
                K=config.model_config.hidden_size,
                N=config.model_config.hidden_size,
                **aie_gemm_config,
            )
        else:
            self.query = nn.Linear(
                config.model_config.hidden_size,
                config.model_config.hidden_size,
                dtype=config.aie_config.dtype,
            )
            self.key = nn.Linear(
                config.model_config.hidden_size,
                config.model_config.hidden_size,
                dtype=config.aie_config.dtype,
            )
            self.value = nn.Linear(
                config.model_config.hidden_size,
                config.model_config.hidden_size,
                dtype=config.aie_config.dtype,
            )
        if config.aie_config.use_aie_gemm:
            aie_gemm_config = {
                "num_aie_columns": 8,
                "tile_m": 64,
                "tile_k": 64,
                "tile_n": 64,
                "use_static_weight": False,  # Attention score calculations don't use weights
                "emulate_bf16_mmul_with_bfp16": True,
                "prio_accuracy": False,
            }
            self.attn_weights = AIEGEMM(
                M=512, K=self.attention_head_size, N=512, **aie_gemm_config
            )
            aie_gemm_config["tile_n"] = 16  # min tile for n is 2t in kernel
            aie_gemm_config["num_aie_columns"] = 4  # Can only use 4 since N=64
            self.attn_score = AIEGEMM(
                M=512, K=512, N=self.attention_head_size, **aie_gemm_config
            )
        self.use_aie_gemm = config.aie_config.use_aie_gemm

        if self.config.aie_config.use_aie_softmax:
            self.softmax = AIESoftmax(
                num_aie_columns=8,
                num_channels=2,
                rows=512,
                cols=512,
            )
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.use_aie_gemm:
            attn_weights = []
            for i in range(self.num_attention_heads):
                attn_weights.append(
                    self.attn_weights(
                        query_layer[:, i, :, :].squeeze(0),
                        key_layer[:, i, :, :].squeeze(0).transpose(0, 1),
                    )
                )
            attn_weights = torch.stack(attn_weights, dim=0)
            attn_weights = attn_weights * self.scaling
        else:
            attn_weights = (
                torch.matmul(query_layer, key_layer.transpose(2, 3)) * self.scaling
            )

        if attention_mask is not None:
            attention_mask = (
                (attention_mask > 0)
                .unsqueeze(1)
                .repeat(1, attention_mask.size(1), 1)
                .unsqueeze(1)
            )
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        # TODO: Handle the attention masking in softmax--currently in "-inf" can lead to
        # NaNs in the result
        attn_weights = self.softmax(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0)

        if self.use_aie_gemm:
            # Compute attention score for each head separately using AIE GEMM
            attn_output = []
            for i in range(self.num_attention_heads):
                attn_output.append(
                    self.attn_score(
                        attn_weights[:, i, :, :].squeeze(0),
                        value_layer[:, i, :, :].squeeze(0),
                    )
                )
            attn_output = torch.stack(attn_output, dim=0)
        else:
            attn_output = torch.matmul(attn_weights, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights

    def assign_weights(self, l, query_w, query_b, key_w, key_b, value_w, value_b):
        if self.config.aie_config.use_aie_gemm:
            self.query.weight = query_w
            self.key.weight = key_w
            self.value.weight = value_w
            # TODO: Need to implement bias assignment
        else:
            assign(
                self.query.weight,
                query_w,
                f"bert.encoder.layer.{l}.attention.self.query.weight",
            )
            assign(
                self.query.bias,
                query_b,
                f"bert.encoder.layer.{l}.attention.self.query.bias",
            )
            assign(
                self.key.weight,
                key_w,
                f"bert.encoder.layer.{l}.attention.self.key.weight",
            )
            assign(
                self.key.bias, key_b, f"bert.encoder.layer.{l}.attention.self.key.bias"
            )
            assign(
                self.value.weight,
                value_w,
                f"bert.encoder.layer.{l}.attention.self.value.weight",
            )
            assign(
                self.value.bias,
                value_b,
                f"bert.encoder.layer.{l}.attention.self.value.bias",
            )


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.aie_config.use_aie_gemm:
            aie_gemm_config = {
                "num_aie_columns": 8,
                "tile_m": 64,
                "tile_k": 96,
                "tile_n": 48,
                "use_static_weight": True,
                "emulate_bf16_mmul_with_bfp16": True,
                "prio_accuracy": False,
            }
            self.dense = AIEGEMM(
                M=512,
                K=config.model_config.hidden_size,
                N=config.model_config.hidden_size,
                **aie_gemm_config,
            )
        else:
            self.dense = nn.Linear(
                config.model_config.hidden_size,
                config.model_config.hidden_size,
                dtype=config.aie_config.dtype,
            )
        if config.aie_config.use_aie_addandnorm:
            self.aie_add_and_norm = AIEAddAndNorm(
                size=512 * config.model_config.hidden_size,
                num_aie_columns=8,
                tile_size=config.model_config.hidden_size,
            )
        else:
            if config.aie_config.use_aie_layernorm:
                self.LayerNorm = AIELayerNorm(
                    size=512 * config.model_config.hidden_size,
                    # eps=config.model_config.layer_norm_eps,
                    num_aie_columns=8,
                    num_channels=2,
                    tile_size=config.model_config.hidden_size,
                    weights=torch.ones(config.model_config.hidden_size),
                )
            else:
                self.LayerNorm = nn.LayerNorm(
                    config.model_config.hidden_size,
                    eps=config.model_config.layer_norm_eps,
                    dtype=config.aie_config.dtype,
                )
            self.use_aie_elementwise_add = config.aie_config.use_aie_elementwise_add
            if self.use_aie_elementwise_add:
                eltwise_add_tile_size = (512 * config.model_config.hidden_size) // 16
                self.aie_elementwise_add = AIEElementwiseAdd(
                    size=512 * config.model_config.hidden_size,
                    num_aie_columns=8,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size
                    ),
                )
        self.dropout = nn.Dropout(config.model_config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.config.aie_config.use_aie_addandnorm:
            hidden_states = self.aie_add_and_norm(hidden_states, input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
            if self.use_aie_elementwise_add:
                hidden_states = self.aie_elementwise_add(hidden_states, input_tensor)
            else:
                hidden_states = hidden_states + input_tensor
        return hidden_states

    def assign_weights(self, l, dense_w, dense_b, layernorm_w, layernorm_b):
        if self.config.aie_config.use_aie_gemm:
            self.dense.weight = dense_w
            # TODO: Need to implement bias assignment
        else:
            assign(
                self.dense.weight,
                dense_w,
                f"bert.encoder.layer.{l}.attention.output.dense.weight",
            )
            assign(
                self.dense.bias,
                dense_b,
                f"bert.encoder.layer.{l}.attention.output.dense.bias",
            )
        if self.config.aie_config.use_aie_addandnorm:
            self.aie_add_and_norm.weight = layernorm_w
            # TODO: Need to implement bias assignment
        else:
            if self.config.aie_config.use_aie_layernorm:
                self.LayerNorm.weight = layernorm_w
                # TODO: Need to implement bias assignment
            else:
                assign(
                    self.LayerNorm.weight,
                    layernorm_w,
                    f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma",
                )
                assign(
                    self.LayerNorm.bias,
                    layernorm_b,
                    f"bert.encoder.layer.{l}.attention.output.LayerNorm.beta",
                )


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attn_weights = self.self(hidden_states, attention_mask)
        attention_output = self.output(attention_output, hidden_states)

        return attention_output, attn_weights
