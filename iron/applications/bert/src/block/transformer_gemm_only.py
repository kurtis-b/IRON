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

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.operators.gemm.op import AIEGEMM

from ..utils import assign


def _layer_norm_no_bias(hidden_states, weight, eps):
    return F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=weight,
        bias=None,
        eps=eps,
    )


class BertEncoderGemmOnlyLayer(nn.Module):
    UNIQUE_INSTRUCTION_BINARY_COUNT = 6
    UNIQUE_XCLBIN_COUNT = 1

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
            raise ValueError("gemm_only currently supports torch.bfloat16 runtime only")

        gemm_common = {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }

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
        self.gemm_dispatch_count = 4 + (2 * self.num_attention_heads)

        self.qkv_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size * 3,
            use_static_weight=True,
            context=context,
            **gemm_common,
        )
        self.attn_scores = AIEGEMM(
            M=self.seq_len,
            K=self.attention_head_size,
            N=self.seq_len,
            context=context,
            **gemm_common,
        )
        self.attn_output = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.attention_head_size,
            context=context,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            context=context,
            **gemm_common,
        )
        self.ffn_up = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.intermediate_size,
            use_static_weight=True,
            context=context,
            **gemm_common,
        )
        self.ffn_down = AIEGEMM(
            M=self.seq_len,
            K=self.intermediate_size,
            N=self.hidden_size,
            use_static_weight=True,
            context=context,
            **gemm_common,
        )
        self.context = self.qkv_proj.context
        self.shared_runtime_xclbin = self.qkv_proj.get_runtime_xclbin_artifact(
            prefix="bert_gemm_only_runtime_"
        )
        self._bind_shared_runtime_xclbin()

        self.ln1_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=runtime_dtype),
            requires_grad=False,
        )
        self.ln2_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=runtime_dtype),
            requires_grad=False,
        )

    def _bind_shared_runtime_xclbin(self):
        runtime_xclbin = self.shared_runtime_xclbin
        runtime_kernel_name = runtime_xclbin.kernel_name
        stage_prefixes = {
            "qkv_proj": "bert_gemm_only_qkv_",
            "attn_scores": "bert_gemm_only_scores_",
            "attn_output": "bert_gemm_only_output_",
            "out_proj": "bert_gemm_only_out_proj_",
            "ffn_up": "bert_gemm_only_ffn_up_",
            "ffn_down": "bert_gemm_only_ffn_down_",
        }
        for stage_name, prefix in stage_prefixes.items():
            op = getattr(self, stage_name)
            insts = op.get_insts_artifact(prefix=prefix, xclbin_input=runtime_xclbin)
            op.bind_artifacts(
                runtime_xclbin,
                insts,
                runtime_xclbin_artifact=runtime_xclbin,
                runtime_kernel_name=runtime_kernel_name,
            )

    def _split_heads(self, projected):
        return projected.view(
            self.seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )

    def _run_attn_scores(self, query_heads, key_heads):
        return torch.stack(
            [
                self.attn_scores(query_heads[head_idx], key_heads[head_idx])
                for head_idx in range(self.num_attention_heads)
            ],
            dim=0,
        )

    def _run_attn_output(self, attn_probs, value_heads):
        return torch.stack(
            [
                self.attn_output(attn_probs[head_idx], value_heads[head_idx])
                for head_idx in range(self.num_attention_heads)
            ],
            dim=1,
        )

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            raise RuntimeError(
                "gemm_only benchmark path currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError("gemm_only benchmark path supports batch size 1")

        hidden_states = hidden_states.squeeze(0).to(self.runtime_dtype)
        qkv_start = time.perf_counter()
        qkv = self.qkv_proj(hidden_states)
        qkv_end = time.perf_counter()
        query_layer, key_layer, value_layer = qkv.split(self.hidden_size, dim=-1)

        host_preprocess_sec = 0.0
        host_postprocess_sec = 0.0
        npu_gemm_sec = qkv_end - qkv_start

        host_start = time.perf_counter()
        query_heads = self._split_heads(query_layer).permute(1, 0, 2).contiguous()
        key_heads = self._split_heads(key_layer).permute(1, 2, 0).contiguous()
        value_heads = self._split_heads(value_layer).permute(1, 0, 2).contiguous()
        host_preprocess_sec += time.perf_counter() - host_start

        npu_start = time.perf_counter()
        attn_scores = self._run_attn_scores(query_heads, key_heads)
        npu_gemm_sec += time.perf_counter() - npu_start

        host_start = time.perf_counter()
        attn_probs = torch.softmax(
            attn_scores.to(torch.float32) * self.scale,
            dim=-1,
        ).to(self.runtime_dtype)
        host_postprocess_sec += time.perf_counter() - host_start

        npu_start = time.perf_counter()
        attn_context = self._run_attn_output(attn_probs, value_heads)
        npu_gemm_sec += time.perf_counter() - npu_start

        host_start = time.perf_counter()
        attn_context = attn_context.contiguous().view(
            self.seq_len,
            self.hidden_size,
        )
        host_preprocess_sec += time.perf_counter() - host_start

        npu_start = time.perf_counter()
        attention_output = self.out_proj(attn_context)
        npu_gemm_sec += time.perf_counter() - npu_start

        host_start = time.perf_counter()
        attention_output = _layer_norm_no_bias(
            attention_output + hidden_states,
            self.ln1_weight,
            self.layer_norm_eps,
        )
        host_postprocess_sec += time.perf_counter() - host_start

        npu_start = time.perf_counter()
        ffn_up = self.ffn_up(attention_output)
        npu_gemm_sec += time.perf_counter() - npu_start

        host_start = time.perf_counter()
        ffn_activated = F.gelu(ffn_up)
        host_postprocess_sec += time.perf_counter() - host_start

        npu_start = time.perf_counter()
        ffn_down = self.ffn_down(ffn_activated)
        npu_gemm_sec += time.perf_counter() - npu_start

        host_start = time.perf_counter()
        output = _layer_norm_no_bias(
            ffn_down + attention_output,
            self.ln2_weight,
            self.layer_norm_eps,
        )
        host_postprocess_sec += time.perf_counter() - host_start

        return output.unsqueeze(0), {
            "qkv_projection_sec": qkv_end - qkv_start,
            "host_preprocess_sec": host_preprocess_sec,
            "npu_gemm_sec": npu_gemm_sec,
            "host_postprocess_sec": host_postprocess_sec,
            "device_sync_sec": 0.0,
            "npu_dispatch_count": self.gemm_dispatch_count,
            "npu_unique_instruction_binary_count": self.UNIQUE_INSTRUCTION_BINARY_COUNT,
        }

    def forward(self, hidden_states, attention_mask=None):
        output, _ = self.forward_with_stage_timings(
            hidden_states,
            attention_mask=attention_mask,
        )
        return output

    def assign_weights(self, layer_idx, combined_weights, dtype):
        query_weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.query.weight"
        ].to(dtype)
        key_weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.key.weight"
        ].to(dtype)
        value_weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.self.value.weight"
        ].to(dtype)
        self.qkv_proj.weight = assign(
            self.qkv_proj.weight,
            torch.cat([query_weight, key_weight, value_weight], dim=0),
            f"encoder.layer.{layer_idx}.attention.self.qkv.weight",
        )
        self.out_proj.weight = assign(
            self.out_proj.weight,
            combined_weights[
                f"encoder.layer.{layer_idx}.attention.output.dense.weight"
            ].to(dtype),
            f"encoder.layer.{layer_idx}.attention.output.dense.weight",
        )
        self.ffn_up.weight = assign(
            self.ffn_up.weight,
            combined_weights[f"encoder.layer.{layer_idx}.intermediate.dense.weight"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.intermediate.dense.weight",
        )
        self.ffn_down.weight = assign(
            self.ffn_down.weight,
            combined_weights[f"encoder.layer.{layer_idx}.output.dense.weight"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.output.dense.weight",
        )
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


class BertEncoderGemmOnly(nn.Module):
    def __init__(self, config, seq_len=512, context=None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                BertEncoderGemmOnlyLayer(config, seq_len=seq_len, context=context)
                for _ in range(config.model_config.num_hidden_layers)
            ]
        )

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        qkv_projection_sec = 0.0
        host_preprocess_sec = 0.0
        npu_gemm_sec = 0.0
        host_postprocess_sec = 0.0
        device_sync_sec = 0.0
        npu_dispatch_count = 0
        npu_unique_instruction_binary_count = 0
        for layer_module in self.layer:
            hidden_states, stage_timings = layer_module.forward_with_stage_timings(
                hidden_states,
                attention_mask,
            )
            qkv_projection_sec += stage_timings["qkv_projection_sec"]
            host_preprocess_sec += stage_timings["host_preprocess_sec"]
            npu_gemm_sec += stage_timings["npu_gemm_sec"]
            host_postprocess_sec += stage_timings["host_postprocess_sec"]
            device_sync_sec += stage_timings["device_sync_sec"]
            npu_dispatch_count += stage_timings["npu_dispatch_count"]
            npu_unique_instruction_binary_count = max(
                npu_unique_instruction_binary_count,
                stage_timings["npu_unique_instruction_binary_count"],
            )
        return hidden_states, {
            "qkv_projection_sec": qkv_projection_sec,
            "host_preprocess_sec": host_preprocess_sec,
            "npu_gemm_sec": npu_gemm_sec,
            "host_postprocess_sec": host_postprocess_sec,
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

    def assign_weights(self, combined_weights):
        dtype = self.config.aie_config.dtype
        for layer_idx, layer in enumerate(self.layer):
            layer.assign_weights(layer_idx, combined_weights, dtype)
