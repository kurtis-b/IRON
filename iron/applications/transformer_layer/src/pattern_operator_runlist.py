# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.gelu.op import AIEGELU
from iron.operators.gemm.op import AIEGEMM
from iron.operators.softmax.op import AIESoftmax

from .layer_spec import TransformerLayerSpec
from .utils import require_keys


def _balanced_tile_size(
    total_size: int, num_aie_columns: int, num_channels: int
) -> int:
    per_channel_tile = max(16, total_size // (num_aie_columns * num_channels))
    return min(math.gcd(4096, per_channel_tile), per_channel_tile)


def _layer_norm_no_bias(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=weight,
        bias=None,
        eps=eps,
    )


class OperatorRunlistPattern(nn.Module):
    pattern_label = "operator_runlist"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "operator_runlist thesis pattern currently requires use_bias=False"
            )
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError(
                "operator_runlist thesis pattern currently requires dtype=bfloat16"
            )

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        head_dim = spec.attention_head_size
        gemm_common = {
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
        eltwise_tile = _balanced_tile_size(spec.seq_len * hidden, 8, 2)
        gelu_tile = _balanced_tile_size(spec.seq_len * spec.intermediate_size, 8, 2)
        attn_tile = _balanced_tile_size(spec.seq_len * spec.seq_len * heads, 8, 2)

        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self.query_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.key_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.value_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.attn_scores = AIEGEMM(
            M=spec.seq_len,
            K=head_dim,
            N=spec.seq_len,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            num_aie_columns=8,
            batch_A=(heads, 1),
            batch_B=(heads, 1),
            batch_C=(heads, 0),
            context=self.context,
            **gemm_common,
        )
        self.attn_scale = AIEElementwiseMul(
            size=spec.seq_len * spec.seq_len * heads,
            num_aie_columns=8,
            num_channels=2,
            tile_size=attn_tile,
            scalar_broadcast=head_dim**-0.5,
            context=self.context,
        )
        self.attn_softmax = AIESoftmax(
            rows=spec.seq_len * heads,
            cols=spec.seq_len,
            num_aie_columns=8,
            num_channels=2,
            context=self.context,
        )
        self.attn_output = AIEGEMM(
            M=spec.seq_len,
            K=spec.seq_len,
            N=head_dim,
            tile_m=64,
            tile_k=64,
            tile_n=16,
            num_aie_columns=4,
            batch_A=(heads, 0),
            batch_B=(heads, 1),
            batch_C=(heads, 1),
            context=self.context,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.add1 = AIEElementwiseAdd(
            size=spec.seq_len * hidden,
            num_aie_columns=8,
            num_channels=2,
            tile_size=eltwise_tile,
            context=self.context,
        )
        self.ffn_up = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=spec.intermediate_size,
            use_static_weight=True,
            tile_m=64,
            tile_k=48,
            tile_n=96,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.gelu = AIEGELU(
            size=spec.seq_len * spec.intermediate_size,
            num_aie_columns=8,
            num_channels=2,
            tile_size=gelu_tile,
            context=self.context,
        )
        self.ffn_down = AIEGEMM(
            M=spec.seq_len,
            K=spec.intermediate_size,
            N=hidden,
            use_static_weight=True,
            tile_m=64,
            tile_k=96,
            tile_n=48,
            num_aie_columns=8,
            context=self.context,
            **gemm_common,
        )
        self.add2 = AIEElementwiseAdd(
            size=spec.seq_len * hidden,
            num_aie_columns=8,
            num_channels=2,
            tile_size=eltwise_tile,
            context=self.context,
        )
        self.ln1_weight = nn.Parameter(
            torch.ones(hidden, dtype=spec.torch_dtype), requires_grad=False
        )
        self.ln2_weight = nn.Parameter(
            torch.ones(hidden, dtype=spec.torch_dtype), requires_grad=False
        )

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        require_keys(
            weights,
            [
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln1_weight",
                "ln2_weight",
            ],
        )
        self.query_proj.weight = weights["q_proj_weight"].contiguous()
        self.key_proj.weight = weights["k_proj_weight"].contiguous()
        self.value_proj.weight = weights["v_proj_weight"].contiguous()
        self.out_proj.weight = weights["out_proj_weight"].contiguous()
        self.ffn_up.weight = weights["ffn_up_weight"].contiguous()
        self.ffn_down.weight = weights["ffn_down_weight"].contiguous()
        self.ln1_weight.data.copy_(weights["ln1_weight"])
        self.ln2_weight.data.copy_(weights["ln2_weight"])

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        return projected.view(
            self.spec.seq_len,
            self.spec.num_attention_heads,
            self.spec.attention_head_size,
        )

    def forward_with_stage_timings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "operator_runlist thesis pattern currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError(
                "operator_runlist thesis pattern currently supports batch_size=1"
            )

        hidden_states = hidden_states.squeeze(0).to(self.spec.torch_dtype)
        start = time.perf_counter()
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        qkv_end = time.perf_counter()

        query_heads = self._split_heads(query).permute(1, 0, 2).contiguous()
        key_heads = self._split_heads(key).permute(1, 2, 0).contiguous()
        value_heads = self._split_heads(value).permute(1, 0, 2).contiguous()
        attn_scores = self.attn_scores(query_heads, key_heads)
        attn_scores = self.attn_scale(attn_scores)
        attn_probs = self.attn_softmax(attn_scores)
        attn_context = (
            self.attn_output(attn_probs, value_heads)
            .contiguous()
            .view(
                self.spec.seq_len,
                self.spec.hidden_size,
            )
        )
        attention_output = self.out_proj(attn_context)
        attention_output = _layer_norm_no_bias(
            self.add1(attention_output, hidden_states),
            self.ln1_weight,
            self.spec.layer_norm_eps,
        )
        ffn_hidden = self.ffn_up(attention_output)
        ffn_hidden = self.gelu(ffn_hidden)
        ffn_output = self.ffn_down(ffn_hidden)
        output = _layer_norm_no_bias(
            self.add2(ffn_output, attention_output),
            self.ln2_weight,
            self.spec.layer_norm_eps,
        ).unsqueeze(0)
        end = time.perf_counter()
        return output, {
            "qkv_projection_sec": qkv_end - start,
            "pattern_sec": end - start,
        }

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            hidden_states, attention_mask=attention_mask
        )
        return output
