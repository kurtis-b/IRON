# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.common import AIEOperatorConstraintError
from iron.common.utils import torch_to_numpy
from iron.operators.elementwise_add.op import AIEElementwiseAdd


def _build_causal_mask(
    *,
    query_block_size: int,
    seq_len: int,
    num_heads: int,
    q_start: int,
    masked_fill_value: float,
) -> torch.Tensor:
    q_positions = torch.arange(q_start, q_start + query_block_size).unsqueeze(1)
    kv_positions = torch.arange(seq_len).unsqueeze(0)
    causal_mask = kv_positions > q_positions
    mask = torch.zeros((query_block_size, seq_len), dtype=torch.bfloat16)
    mask[causal_mask] = masked_fill_value
    return mask.unsqueeze(0).repeat(num_heads, 1, 1).reshape(-1)


class AIECausalMask(AIEElementwiseAdd):
    """Apply a static causal mask by adding a precomputed triangular tensor."""

    def __init__(
        self,
        *,
        query_block_size: int,
        seq_len: int,
        num_heads: int,
        q_start: int = 0,
        masked_fill_value: float = -10000.0,
        num_aie_columns: int,
        num_channels: int,
        tile_size: int,
        context=None,
        skip_add_to_list: bool = False,
    ):
        self.query_block_size = query_block_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.q_start = q_start
        self.masked_fill_value = masked_fill_value
        self.mask_tensor = _build_causal_mask(
            query_block_size=query_block_size,
            seq_len=seq_len,
            num_heads=num_heads,
            q_start=q_start,
            masked_fill_value=masked_fill_value,
        )
        super().__init__(
            size=query_block_size * seq_len * num_heads,
            num_aie_columns=num_aie_columns,
            num_channels=num_channels,
            tile_size=tile_size,
            context=context,
            skip_add_to_list=skip_add_to_list,
        )

    def set_up_runtime(self):
        self.add_buffer("input1", self.size)
        self.add_buffer(
            "input2", self.size, static_data=torch_to_numpy(self.mask_tensor)
        )
        self.add_buffer("output", self.size)
        self.add_kernel(
            "causal_mask",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("causal_mask", "input1", "input2", "output")

    def forward(self, x):
        if x.numel() != self.size:
            raise AIEOperatorConstraintError(
                "AIECausalMask: incompatible tensor shape(s)"
            )
        return super().forward(x.reshape(-1), self.mask_tensor.reshape(-1))
