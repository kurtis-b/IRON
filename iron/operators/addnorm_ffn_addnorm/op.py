# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.operators.ffn_addnorm.op import AIEFFNAN


def _select_block3_config(hidden_size: int, intermediate_size: int) -> dict[str, int]:
    if hidden_size == 768 and intermediate_size == 3072:
        return {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "nA_tiles_distributed": 4,
            "nB_tiles_distributed": 3,
            "gelu_stage": 1,
        }
    if hidden_size == 1024 and intermediate_size == 4096:
        return {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "nA_tiles_distributed": 4,
            "nB_tiles_distributed": 2,
            "gelu_stage": 1,
        }
    raise ValueError(
        "Block 3 currently supports only the retained thesis families 768/3072 and 1024/4096"
    )


class AIEAddNormFFNAddNorm:
    """
    Thesis-v2 Block 3 operator.

    This block is backed by the pipelined thesis-branch AN+FFN design, which
    pipelines the first Add & Norm stage into the Up projection cores.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        context,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.context = context
        config = _select_block3_config(hidden_size, intermediate_size)
        self.block = AIEFFNAN(
            M=config["compile_rows"],
            K=hidden_size,
            N=intermediate_size,
            use_static_weight=True,
            tile_m=config["tile_m"],
            tile_k=config["tile_k"],
            tile_n=config["tile_n"],
            down_proj_depth=config["down_proj_depth"],
            num_aie_columns=config["num_aie_columns"],
            ln2_weight=torch.ones(hidden_size, dtype=torch.bfloat16),
            emulate_bf16_mmul_with_bfp16=True,
            nA_tiles_distributed=config["nA_tiles_distributed"],
            nB_tiles_distributed=config["nB_tiles_distributed"],
            gelu_stage=config["gelu_stage"],
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
