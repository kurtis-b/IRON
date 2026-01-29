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
from operators import AIEGELU
from operators import AIEElementwiseAdd
from operators import AIELayerNorm
from operators import AIEFFN
from operators import AIEAddAndNorm


class BertFeedForward(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        if config.aie_config.use_aie_ffn:
            aie_ffn_config = {
                "b_col_maj": False,
                "c_col_maj": False,
                "emulate_bf16_mmul_with_bfp16": True,
                "n_a_tiles_distributed": 4,
                "n_b_tiles_distributed": 4,
                "stage_only": None,
                "gelu_stage": 1,
                "use_static_weight": True,
            }
            self.ffn = AIEFFN(
                M=seq_len,
                K=config.model_config.hidden_size,
                N=config.model_config.intermediate_size,
                tile_m=64,
                tile_k=48,
                tile_n=96,
                down_proj_depth=8,
                num_aie_columns=8,
                **aie_ffn_config,
            )
        else:
            if config.aie_config.use_aie_gemm:
                aie_gemm_config = {
                    "num_aie_columns": 8,
                    "tile_m": 64,
                    "tile_k": 48,
                    "tile_n": 96,
                    "use_static_weight": True,
                    "emulate_bf16_mmul_with_bfp16": True,
                    "prio_accuracy": False,
                }
                self.dense_up = AIEGEMM(
                    M=seq_len,
                    K=config.model_config.hidden_size,
                    N=config.model_config.intermediate_size,
                    **aie_gemm_config,
                )
            else:
                self.dense_up = nn.Linear(
                    config.model_config.hidden_size,
                    config.model_config.intermediate_size,
                    dtype=config.aie_config.dtype,
                )
            if config.aie_config.use_aie_gelu:
                gelu_tile_size = (seq_len * config.model_config.intermediate_size) // 16
                self.gelu = AIEGELU(
                    size=seq_len * config.model_config.intermediate_size,
                    num_aie_columns=8,
                    num_channels=2,
                    tile_size=min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
                )
            else:
                self.gelu = nn.GELU()
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
                self.dense_down = AIEGEMM(
                    M=seq_len,
                    K=config.model_config.intermediate_size,
                    N=config.model_config.hidden_size,
                    **aie_gemm_config,
                )
            else:
                self.dense_down = nn.Linear(
                    config.model_config.intermediate_size,
                    config.model_config.hidden_size,
                    dtype=config.aie_config.dtype,
                )
        if config.aie_config.use_aie_addandnorm:
            self.aie_addnorm = AIEAddAndNorm(
                size=seq_len * config.model_config.hidden_size,
                num_aie_columns=8,
                tile_size=config.model_config.hidden_size,
            )
        else:
            if config.aie_config.use_aie_layernorm:
                self.LayerNorm = AIELayerNorm(
                    size=seq_len * config.model_config.hidden_size,
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
                eltwise_add_tile_size = (
                    seq_len * config.model_config.hidden_size
                ) // 16
                self.aie_elementwise_add = AIEElementwiseAdd(
                    size=seq_len * config.model_config.hidden_size,
                    num_aie_columns=8,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size
                    ),
                )
        self.dropout = nn.Dropout(config.model_config.hidden_dropout_prob)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        if self.config.aie_config.use_aie_ffn:
            hidden_states = self.ffn(hidden_states)
        else:
            hidden_states = self.dense_up(hidden_states)
            hidden_states = self.gelu(hidden_states)
            hidden_states = self.dense_down(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.config.aie_config.use_aie_addandnorm:
            hidden_states = self.aie_addnorm(hidden_states, input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
            if self.use_aie_elementwise_add:
                hidden_states = self.aie_elementwise_add(hidden_states, input_tensor)
            else:
                hidden_states = hidden_states + input_tensor
        return hidden_states

    def assign_weights(
        self,
        l,
        dense_up_w,
        dense_up_b,
        dense_down_w,
        dense_down_b,
        layernorm_w,
        layernorm_b,
    ):
        if self.config.aie_config.use_aie_ffn:
            self.ffn.weight_up_proj = dense_up_w
            self.ffn.weight_down_proj = dense_down_w
            # TODO: Need to implement bias assignment
        else:
            if self.config.aie_config.use_aie_gemm:
                self.dense_up.weight = dense_up_w
                self.dense_down.weight = dense_down_w
                # TODO: Need to implement bias assignment
            else:
                assign(
                    self.dense_up.weight,
                    dense_up_w,
                    f"bert.encoder.layer.{l}.intermediate.dense.weight",
                )
                assign(
                    self.dense_up.bias,
                    dense_up_b,
                    f"bert.encoder.layer.{l}.intermediate.dense.bias",
                )
                assign(
                    self.dense_down.weight,
                    dense_down_w,
                    f"bert.encoder.layer.{l}.output.dense.weight",
                )
                assign(
                    self.dense_down.bias,
                    dense_down_b,
                    f"bert.encoder.layer.{l}.output.dense.bias",
                )
        if self.config.aie_config.use_aie_addandnorm:
            self.aie_addnorm.weight = layernorm_w
            # TODO: Need to implement bias assignment
        else:
            if self.config.aie_config.use_aie_layernorm:
                self.LayerNorm.weight = layernorm_w
                # TODO: Need to implement bias assignment
            else:
                assign(
                    self.LayerNorm.weight,
                    layernorm_w,
                    f"bert.encoder.layer.{l}.output.LayerNorm.gamma",
                )
                assign(
                    self.LayerNorm.bias,
                    layernorm_b,
                    f"bert.encoder.layer.{l}.output.LayerNorm.beta",
                )
