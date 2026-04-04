# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.common import AIEOperatorConstraintError
from iron.operators.gemm.op import AIEGEMM

from iron.applications.transformer_layer_new.pattern.gemm_only.op import (
    AIETransformerGemmOnly,
    default_gemm_only_operator_config,
    resolve_gemm_only_operator_config,
)


def default_offload_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
):
    return default_gemm_only_operator_config(
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=num_aie_columns,
    )


def resolve_offload_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
    operator_config=None,
):
    return resolve_gemm_only_operator_config(
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=num_aie_columns,
        operator_config=operator_config,
    )


class AIETransformerOffload(AIETransformerGemmOnly):
    """Legacy shared-xclbin GEMM-only offload baseline for transformer_layer_new."""

    pattern_label = "offload"

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        *,
        num_aie_columns=8,
        ln1_weight=None,
        ln2_weight=None,
        operator_config=None,
        context=None,
    ):
        super().__init__(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_aie_columns=num_aie_columns,
            ln1_weight=ln1_weight,
            ln2_weight=ln2_weight,
            operator_config=operator_config,
            context=context,
        )

        gemm_common = dict(self.operator_config["shared_gemm"])
        gemm_common["context"] = self.context
        runtime_seq_len = self.query_block_size
        self.q_proj = AIEGEMM(
            M=runtime_seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            **gemm_common,
        )
        self.k_proj = AIEGEMM(
            M=runtime_seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            **gemm_common,
        )
        self.v_proj = AIEGEMM(
            M=runtime_seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            **gemm_common,
        )
        self.gemm_ops = [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            *self.gemm_ops,
        ]
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None

    def _validate_required_weights(self) -> None:
        super()._validate_required_weights()
        required_weights = {
            "q_weight": self.q_weight,
            "k_weight": self.k_weight,
            "v_weight": self.v_weight,
        }
        missing = [name for name, value in required_weights.items() if value is None]
        if missing:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: missing required qkv weights "
                + ", ".join(missing)
            )

    def _bind_static_weights(self) -> None:
        super()._bind_static_weights()
        self.q_proj.weight = self.q_weight.T.contiguous()
        self.k_proj.weight = self.k_weight.T.contiguous()
        self.v_proj.weight = self.v_weight.T.contiguous()

    def get_benchmark_metadata(self) -> dict[str, object]:
        metadata = super().get_benchmark_metadata()
        block_count = (
            self.seq_len + self.query_block_size - 1
        ) // self.query_block_size
        metadata["npu_dispatch_count"] = ((2 * self.num_heads) + 6) * block_count
        return metadata

    def forward_precomputed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            q=q,
            k=k,
            v=v,
            residual=residual,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload currently requires attention_mask=None"
            )
        if not self._runtime_ready:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: prepare_runtime() must be called before forward"
            )
        if tuple(hidden_states.shape) != (self.seq_len, self.hidden_size):
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: expected hidden_states shape "
                f"({self.seq_len}, {self.hidden_size})"
            )

        hidden_states = hidden_states.contiguous()
        projected_q = []
        projected_k = []
        projected_v = []
        for block_start in range(0, self.seq_len, self.query_block_size):
            block_end = min(block_start + self.query_block_size, self.seq_len)
            hidden_block = hidden_states[block_start:block_end]
            projected_q.append(self.q_proj(hidden_block))
            projected_k.append(self.k_proj(hidden_block))
            projected_v.append(self.v_proj(hidden_block))
        q = torch.cat(projected_q, dim=0)
        k = torch.cat(projected_k, dim=0)
        v = torch.cat(projected_v, dim=0)
        precomputed = {
            "q": q.view(self.seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous(),
            "k": k.view(self.seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous(),
            "v": v.view(self.seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous(),
            "residual": hidden_states,
        }
        return self.forward_precomputed(
            q=precomputed["q"],
            k=precomputed["k"],
            v=precomputed["v"],
            residual=precomputed["residual"],
            attention_mask=attention_mask,
        )
