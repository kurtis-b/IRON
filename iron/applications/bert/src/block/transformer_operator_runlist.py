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

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.gelu.op import AIEGELU
from iron.operators.gemm.op import AIEGEMM
from iron.operators.softmax.op import AIESoftmax

from ..utils import assign


def _balanced_tile_size(total_size, num_aie_columns, num_channels):
    per_channel_tile = max(16, total_size // (num_aie_columns * num_channels))
    return min(math.gcd(4096, per_channel_tile), per_channel_tile)


class BertEncoderOperatorRunlistLayer(nn.Module):
    DISPATCH_COUNT = 13
    UNIQUE_INSTRUCTION_BINARY_COUNT = 9

    def __init__(self, config, seq_len=512, context=None):
        super().__init__()
        hidden_size = int(config.model_config.hidden_size)
        intermediate_size = int(config.model_config.intermediate_size)
        num_heads = int(config.model_config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})"
            )

        runtime_dtype = config.aie_config.dtype
        if runtime_dtype != torch.bfloat16:
            raise ValueError(
                "operator_runlist currently supports torch.bfloat16 runtime only"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.seq_len = int(seq_len)
        self.runtime_dtype = runtime_dtype
        self.layer_norm_eps = float(
            getattr(config.model_config, "layer_norm_eps", 1.0e-12)
        )
        self.scale = self.attention_head_size**-0.5

        gemm_common = {
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
        eltwise_tile = _balanced_tile_size(
            self.seq_len * self.hidden_size,
            num_aie_columns=8,
            num_channels=2,
        )
        gelu_tile = _balanced_tile_size(
            self.seq_len * self.intermediate_size,
            num_aie_columns=8,
            num_channels=2,
        )
        attn_tile = _balanced_tile_size(
            self.seq_len * self.seq_len * self.num_attention_heads,
            num_aie_columns=8,
            num_channels=2,
        )

        self.query_proj = AIEGEMM(
            M=self.seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.key_proj = AIEGEMM(
            M=self.seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.value_proj = AIEGEMM(
            M=self.seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.attn_scores = AIEGEMM(
            M=self.seq_len,
            K=self.attention_head_size,
            N=self.seq_len,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            num_aie_columns=8,
            batch_A=(num_heads, 1),
            batch_B=(num_heads, 1),
            batch_C=(num_heads, 0),
            context=context,
            **gemm_common,
        )
        self.attn_scale = AIEElementwiseMul(
            size=self.seq_len * self.seq_len * self.num_attention_heads,
            num_aie_columns=8,
            num_channels=2,
            tile_size=attn_tile,
            scalar_broadcast=self.scale,
            context=context,
        )
        self.attn_softmax = AIESoftmax(
            rows=self.seq_len * self.num_attention_heads,
            cols=self.seq_len,
            num_aie_columns=8,
            num_channels=2,
            context=context,
        )
        self.attn_output = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.attention_head_size,
            tile_m=64,
            tile_k=64,
            tile_n=16,
            num_aie_columns=4,
            batch_A=(num_heads, 0),
            batch_B=(num_heads, 1),
            batch_C=(num_heads, 1),
            context=context,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=self.seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.add1 = AIEElementwiseAdd(
            size=self.seq_len * hidden_size,
            num_aie_columns=8,
            num_channels=2,
            tile_size=eltwise_tile,
            context=context,
        )
        self.ffn_up = AIEGEMM(
            M=self.seq_len,
            K=hidden_size,
            N=intermediate_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=48,
            tile_n=96,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.gelu = AIEGELU(
            size=self.seq_len * intermediate_size,
            num_aie_columns=8,
            num_channels=2,
            tile_size=gelu_tile,
            context=context,
        )
        self.ffn_down = AIEGEMM(
            M=self.seq_len,
            K=intermediate_size,
            N=hidden_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=context,
            **gemm_common,
        )
        self.add2 = AIEElementwiseAdd(
            size=self.seq_len * hidden_size,
            num_aie_columns=8,
            num_channels=2,
            tile_size=eltwise_tile,
            context=context,
        )
        self.ln1_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=runtime_dtype),
            requires_grad=False,
        )
        self.ln2_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=runtime_dtype),
            requires_grad=False,
        )

    def _split_heads(self, projected):
        return projected.view(
            self.seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            raise RuntimeError(
                "operator_runlist benchmark path currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError("operator_runlist benchmark path supports batch size 1")

        hidden_states = hidden_states.squeeze(0).to(self.runtime_dtype)
        qkv_start = time.perf_counter()
        query_layer = self.query_proj(hidden_states)
        key_layer = self.key_proj(hidden_states)
        value_layer = self.value_proj(hidden_states)
        qkv_end = time.perf_counter()

        host_preprocess_sec = 0.0
        runlist_start = time.perf_counter()

        host_start = time.perf_counter()
        query_heads = self._split_heads(query_layer).contiguous()
        key_heads = self._split_heads(key_layer).permute(2, 1, 0).contiguous()
        value_heads = self._split_heads(value_layer).contiguous()
        host_preprocess_sec += time.perf_counter() - host_start

        attn_scores = self.attn_scores(query_heads, key_heads)
        attn_scaled = self.attn_scale(attn_scores.unsqueeze(0)).squeeze(0)
        attn_probs = self.attn_softmax(
            attn_scaled.reshape(
                self.num_attention_heads * self.seq_len,
                self.seq_len,
            )
        ).view(self.num_attention_heads, self.seq_len, self.seq_len)
        attn_context = self.attn_output(attn_probs, value_heads)

        host_start = time.perf_counter()
        attn_context = attn_context.contiguous().view(self.seq_len, self.hidden_size)
        host_preprocess_sec += time.perf_counter() - host_start

        attention_output = self.out_proj(attn_context)
        attention_added = self.add1(
            attention_output.view(1, -1),
            hidden_states.view(1, -1),
        ).view(self.seq_len, self.hidden_size)
        attention_output = F.layer_norm(
            attention_added,
            (self.hidden_size,),
            weight=self.ln1_weight,
            bias=None,
            eps=self.layer_norm_eps,
        )

        ffn_up = self.ffn_up(attention_output)
        ffn_activated = self.gelu(ffn_up)
        ffn_down = self.ffn_down(ffn_activated)
        output_added = self.add2(
            ffn_down.view(1, -1),
            attention_output.view(1, -1),
        ).view(self.seq_len, self.hidden_size)
        output = F.layer_norm(
            output_added,
            (self.hidden_size,),
            weight=self.ln2_weight,
            bias=None,
            eps=self.layer_norm_eps,
        )

        run_end = time.perf_counter()
        return output.unsqueeze(0), {
            "qkv_projection_sec": qkv_end - qkv_start,
            "operator_runlist_sec": run_end - runlist_start,
            "host_preprocess_sec": host_preprocess_sec,
            "device_sync_sec": 0.0,
            "npu_dispatch_count": self.DISPATCH_COUNT,
            "npu_unique_instruction_binary_count": (
                self.UNIQUE_INSTRUCTION_BINARY_COUNT
            ),
        }

    def forward(self, hidden_states, attention_mask=None):
        output, _ = self.forward_with_stage_timings(
            hidden_states,
            attention_mask=attention_mask,
        )
        return output

    def assign_weights(self, layer_idx, combined_weights, dtype):
        self.query_proj.weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.query.weight"
        ].to(dtype)
        self.key_proj.weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.key.weight"
        ].to(dtype)
        self.value_proj.weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.value.weight"
        ].to(dtype)
        self.out_proj.weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.output.dense.weight"
        ].to(dtype)
        self.ffn_up.weight = combined_weights[
            f"encoder.layer.{layer_idx}.intermediate.dense.weight"
        ].to(dtype)
        self.ffn_down.weight = combined_weights[
            f"encoder.layer.{layer_idx}.output.dense.weight"
        ].to(dtype)
        self.ln1_weight = assign(
            self.ln1_weight,
            combined_weights[
                f"encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"
            ].to(dtype),
            f"encoder.layer.{layer_idx}.attention.output.LayerNorm.weight",
        )
        self.ln2_weight = assign(
            self.ln2_weight,
            combined_weights[f"encoder.layer.{layer_idx}.output.LayerNorm.weight"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.output.LayerNorm.weight",
        )


class BertEncoderOperatorRunlist(nn.Module):
    def __init__(self, config, seq_len=512, context=None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                BertEncoderOperatorRunlistLayer(
                    config,
                    seq_len=seq_len,
                    context=context,
                )
                for _ in range(config.model_config.num_hidden_layers)
            ]
        )

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        qkv_projection_sec = 0.0
        operator_runlist_sec = 0.0
        host_preprocess_sec = 0.0
        device_sync_sec = 0.0
        npu_dispatch_count = 0
        npu_unique_instruction_binary_count = 0
        for layer_module in self.layer:
            hidden_states, stage_timings = layer_module.forward_with_stage_timings(
                hidden_states,
                attention_mask,
            )
            qkv_projection_sec += stage_timings["qkv_projection_sec"]
            operator_runlist_sec += stage_timings["operator_runlist_sec"]
            host_preprocess_sec += stage_timings["host_preprocess_sec"]
            device_sync_sec += stage_timings["device_sync_sec"]
            npu_dispatch_count += stage_timings["npu_dispatch_count"]
            npu_unique_instruction_binary_count = max(
                npu_unique_instruction_binary_count,
                stage_timings["npu_unique_instruction_binary_count"],
            )
        return hidden_states, {
            "qkv_projection_sec": qkv_projection_sec,
            "operator_runlist_sec": operator_runlist_sec,
            "host_preprocess_sec": host_preprocess_sec,
            "device_sync_sec": device_sync_sec,
            "npu_dispatch_count": npu_dispatch_count,
            "npu_unique_instruction_binary_count": (
                npu_unique_instruction_binary_count
            ),
        }

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

    def assign_weights(self, combined_weights, dtype):
        for layer_idx, layer in enumerate(self.layer):
            layer.assign_weights(layer_idx, combined_weights, dtype)
