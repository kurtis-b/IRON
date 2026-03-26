# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext
from iron.operators.gemm.op import AIEGEMM

from .layer_spec import TransformerLayerSpec
from .utils import require_keys


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


class GemmOnlyPattern(nn.Module):
    pattern_label = "gemm_only"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "gemm_only thesis pattern currently requires use_bias=False"
            )
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError(
                "gemm_only thesis pattern currently requires dtype=bfloat16"
            )

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        head_dim = spec.attention_head_size
        gemm_common = {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }

        self.spec = spec
        self.context = AIEContext(use_runlist=False)
        self.qkv_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden * 3,
            use_static_weight=True,
            context=self.context,
            **gemm_common,
        )
        self.attn_scores = AIEGEMM(
            M=spec.seq_len,
            K=head_dim,
            N=spec.seq_len,
            context=self.context,
            **gemm_common,
        )
        self.attn_output = AIEGEMM(
            M=spec.seq_len,
            K=spec.seq_len,
            N=head_dim,
            context=self.context,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            context=self.context,
            **gemm_common,
        )
        self.ffn_up = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=spec.intermediate_size,
            use_static_weight=True,
            context=self.context,
            **gemm_common,
        )
        self.ffn_down = AIEGEMM(
            M=spec.seq_len,
            K=spec.intermediate_size,
            N=hidden,
            use_static_weight=True,
            context=self.context,
            **gemm_common,
        )
        self.qkv_proj_weight = None
        self.scale = head_dim**-0.5
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
        self.qkv_proj.weight = torch.cat(
            [
                weights["q_proj_weight"],
                weights["k_proj_weight"],
                weights["v_proj_weight"],
            ],
            dim=0,
        ).contiguous()
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
                "gemm_only thesis pattern currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError(
                "gemm_only thesis pattern currently supports batch_size=1"
            )

        hidden_states = hidden_states.squeeze(0).to(self.spec.torch_dtype)
        start = time.perf_counter()
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.spec.hidden_size, dim=-1)
        qkv_end = time.perf_counter()

        query_heads = self._split_heads(query).permute(1, 0, 2).contiguous()
        key_heads = self._split_heads(key).permute(1, 2, 0).contiguous()
        value_heads = self._split_heads(value).permute(1, 0, 2).contiguous()

        attn_scores = torch.stack(
            [
                self.attn_scores(query_heads[h], key_heads[h])
                for h in range(self.spec.num_attention_heads)
            ],
            dim=0,
        )
        attn_probs = torch.softmax(
            attn_scores.to(torch.float32) * self.scale, dim=-1
        ).to(self.spec.torch_dtype)
        attn_context = (
            torch.stack(
                [
                    self.attn_output(attn_probs[h], value_heads[h])
                    for h in range(self.spec.num_attention_heads)
                ],
                dim=1,
            )
            .contiguous()
            .view(self.spec.seq_len, self.spec.hidden_size)
        )
        attention_output = self.out_proj(attn_context)
        attention_output = _layer_norm_no_bias(
            attention_output + hidden_states,
            self.ln1_weight,
            self.spec.layer_norm_eps,
        )
        ffn_up = self.ffn_up(attention_output)
        ffn_down = self.ffn_down(F.gelu(ffn_up))
        output = _layer_norm_no_bias(
            ffn_down + attention_output,
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
