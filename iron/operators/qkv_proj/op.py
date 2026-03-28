# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.operators.qkv_proj.design import qkv_proj_design
from iron.operators.gemm.op import AIEGEMM


class AIEQKVProj:
    """Shared-runtime Q/K/V projection block built from three GEMMs."""

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
        self.q_proj = AIEGEMM(M=seq_len, K=hidden_size, N=hidden_size, **gemm_common)
        self.k_proj = AIEGEMM(M=seq_len, K=hidden_size, N=hidden_size, **gemm_common)
        self.v_proj = AIEGEMM(M=seq_len, K=hidden_size, N=hidden_size, **gemm_common)
        self._bind_shared_artifacts()

    def _bind_shared_artifacts(self) -> None:
        shared_xclbin = self.q_proj.get_runtime_xclbin_artifact(
            prefix=f"qkv_proj_{self.seq_len}x{self.hidden_size}_runtime_"
        )
        shared_xclbin.kernel_name = "qkv_proj_runtime"
        for workload_name, gemm_op in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
        ):
            insts_artifact = gemm_op.get_insts_artifact(
                prefix=(f"qkv_proj_{self.seq_len}x{self.hidden_size}_{workload_name}_"),
                xclbin_input=shared_xclbin,
                kernel_name=shared_xclbin.kernel_name,
            )
            gemm_op.bind_artifacts(
                shared_xclbin,
                insts_artifact,
                runtime_xclbin_artifact=shared_xclbin,
                runtime_kernel_name=shared_xclbin.kernel_name,
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._to_head_major(
            self.q_proj(hidden_states).contiguous(),
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        k = self._to_head_major(
            self.k_proj(hidden_states).contiguous(),
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        v = self._to_head_major(
            self.v_proj(hidden_states).contiguous(),
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        return q, k, v
