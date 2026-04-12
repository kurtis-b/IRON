# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .op import _build_causal_mask


def generate_golden_reference(
    *,
    query_block_size: int,
    seq_len: int,
    num_heads: int,
    q_start: int = 0,
    masked_fill_value: float = -10000.0,
    seed: int = 42,
):
    torch.manual_seed(seed)
    input_tensor = torch.randn(
        query_block_size * seq_len * num_heads,
        dtype=torch.bfloat16,
    )
    mask = _build_causal_mask(
        query_block_size=query_block_size,
        seq_len=seq_len,
        num_heads=num_heads,
        q_start=q_start,
        masked_fill_value=masked_fill_value,
    )
    return {
        "input": input_tensor,
        "mask": mask,
        "output": input_tensor + mask,
    }
