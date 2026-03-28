# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.operators.ffn_addnorm.op import AIEFFNAN

_BLOCK3_TOPOLOGIES = {
    (768, 3072): [
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        }
    ],
    (1024, 4096): [
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        }
    ],
}


def _block3_topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


class AIEAddNormFFNAddNorm:
    """
    Thesis-v2 Block 3 operator.

    This block is backed by the pipelined thesis-branch AN+FFN design, which
    pipelines the first Add & Norm stage into the Up projection cores.
    """

    @classmethod
    def enumerate_topologies(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[dict[str, int | str]]:
        try:
            topologies = _BLOCK3_TOPOLOGIES[(hidden_size, intermediate_size)]
        except KeyError as exc:
            raise ValueError(
                "Block 3 currently supports only the retained thesis families 768/3072 and 1024/4096"
            ) from exc
        return [
            {
                **config,
                "topology_id": _block3_topology_id(config),
                "topology_family": "pipelined_addnorm_ffn_addnorm",
            }
            for config in topologies
        ]

    @classmethod
    def select_default_topology(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
    ) -> dict[str, int | str]:
        del seq_len
        topologies = cls.enumerate_topologies(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return max(
            topologies,
            key=lambda topology: (
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
            ),
        )

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        context,
        topology_id: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.context = context
        config = self.select_default_topology(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            seq_len=seq_len,
        )
        if topology_id is not None:
            config = next(
                candidate
                for candidate in self.enumerate_topologies(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                if candidate["topology_id"] == topology_id
            )
        self.topology_id = str(config["topology_id"])
        self.topology_family = str(config["topology_family"])
        self.parallel_seq = int(config["parallel_seq"])
        self.parallel_int_dim = int(config["parallel_int_dim"])
        self.tile_m = int(config["tile_m"])
        self.tile_k = int(config["tile_k"])
        self.tile_n = int(config["tile_n"])
        self.down_proj_depth = int(config["down_proj_depth"])
        self.gelu_stage = int(config["gelu_stage"])
        self.block = AIEFFNAN(
            M=int(config["compile_rows"]),
            K=hidden_size,
            N=intermediate_size,
            use_static_weight=True,
            tile_m=self.tile_m,
            tile_k=self.tile_k,
            tile_n=self.tile_n,
            down_proj_depth=self.down_proj_depth,
            num_aie_columns=int(config["num_aie_columns"]),
            ln2_weight=torch.ones(hidden_size, dtype=torch.bfloat16),
            emulate_bf16_mmul_with_bfp16=True,
            nA_tiles_distributed=self.parallel_seq,
            nB_tiles_distributed=self.parallel_int_dim,
            gelu_stage=self.gelu_stage,
            context=context,
        )

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.block.weight_up_proj = weights["ffn_up_weight"].contiguous()
        self.block.weight_down_proj = weights["ffn_down_weight"].contiguous()
        self.block.ln2_weight = weights["ln2_weight"].contiguous()

    def forward(
        self, attention_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.block(attention_output, residual)

    def __call__(
        self, attention_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(attention_output, residual)

    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "npu_dispatch_count": len(self.block.runlist),
            "npu_unique_instruction_binary_count": 1,
            "npu_unique_xclbin_count": 1,
        }
