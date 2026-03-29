# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.operators.qkv_proj.design import qkv_proj_design
from iron.operators.gemm.op import AIEGEMM


class _ProjectionWeightView:
    def __init__(self, block: "AIEQKVProj", projection_index: int) -> None:
        self._block = block
        self._projection_index = projection_index

    @property
    def weight(self) -> torch.Tensor:
        return self._block._projection_weight(self._projection_index)

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self._block._set_projection_weight(self._projection_index, value)

    @property
    def insts_artifact(self):
        return self._block.qkv_proj.insts_artifact

    @property
    def runtime_xclbin_artifact(self):
        return self._block.qkv_proj.runtime_xclbin_artifact

    @property
    def xclbin_artifact(self):
        return self._block.qkv_proj.xclbin_artifact


class AIEQKVProj:
    """Shared-runtime Q/K/V projection block built from one fused GEMM."""

    @staticmethod
    def _to_head_major(
        tensor: torch.Tensor,
        *,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        head_dim = hidden_size // num_heads
        return tensor.view(seq_len, num_heads, head_dim).permute(1, 0, 2).contiguous()

    @staticmethod
    def _combined_hidden_size(hidden_size: int) -> int:
        return hidden_size * 3

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        context,
        num_heads: int,
        topology_id: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context = context
        topology = qkv_proj_design(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            topology_id=topology_id,
        )
        self.topology_id = str(topology["topology_id"])
        self.topology_family = str(topology["topology_family"])
        self.parallel_seq = int(topology["parallel_seq"])
        self.parallel_heads = int(topology["parallel_heads"])
        self.parallel_head_dim = int(topology["parallel_head_dim"])
        self.tile_m = int(topology["tile_m"])
        self.tile_k = int(topology["tile_k"])
        self.tile_n = int(topology["tile_n"])
        gemm_common = {
            "tile_m": self.tile_m,
            "tile_k": self.tile_k,
            "tile_n": self.tile_n,
            "num_aie_columns": int(topology["num_aie_columns"]),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
            "use_static_weight": True,
            "force_batched_design": True,
            "context": context,
        }
        self.qkv_proj = AIEGEMM(
            M=seq_len,
            K=hidden_size,
            N=self._combined_hidden_size(hidden_size),
            **gemm_common,
        )
        self.q_proj = _ProjectionWeightView(self, 0)
        self.k_proj = _ProjectionWeightView(self, 1)
        self.v_proj = _ProjectionWeightView(self, 2)

    def _projection_slice(self, projection_index: int) -> slice:
        start = projection_index * self.hidden_size
        end = start + self.hidden_size
        return slice(start, end)

    def _projection_weight(self, projection_index: int) -> torch.Tensor:
        return self.qkv_proj.weight[self._projection_slice(projection_index)]

    def _set_projection_weight(
        self, projection_index: int, value: torch.Tensor
    ) -> None:
        expected_shape = (self.hidden_size, self.hidden_size)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"AIEQKVProj: expected projection weight shape {expected_shape}, got {tuple(value.shape)}"
            )
        self.qkv_proj.weight[self._projection_slice(projection_index)] = (
            value.contiguous()
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(hidden_states).contiguous()
        q_matrix, k_matrix, v_matrix = torch.split(qkv, self.hidden_size, dim=-1)

        q = self._to_head_major(
            q_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        k = self._to_head_major(
            k_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        v = self._to_head_major(
            v_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        return q, k, v
