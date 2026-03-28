# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.operators.addnorm_ffn_addnorm.design import addnorm_ffn_addnorm_design
from iron.operators.ffn_addnorm.op import AIEFFNAN


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
        topology_id: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.context = context
        config = addnorm_ffn_addnorm_design(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topology_id=topology_id,
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

    @property
    def weight_up_proj(self):
        return self.block.weight_up_proj

    @weight_up_proj.setter
    def weight_up_proj(self, value):
        self.block.weight_up_proj = value

    @property
    def weight_down_proj(self):
        return self.block.weight_down_proj

    @weight_down_proj.setter
    def weight_down_proj(self, value):
        self.block.weight_down_proj = value

    @property
    def ln2_weight(self):
        return self.block.ln2_weight

    @ln2_weight.setter
    def ln2_weight(self, value):
        self.block.ln2_weight = value

    def forward(
        self, attention_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.block(attention_output, residual)
