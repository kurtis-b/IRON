# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch


def generate_golden_reference(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_size = num_heads * head_dim
    val_range = 4
    hidden_states = torch.rand(seq_len, hidden_size, dtype=torch.bfloat16) * val_range
    q_proj_weight = (
        torch.rand(hidden_size, hidden_size, dtype=torch.bfloat16) * val_range
    )
    k_proj_weight = (
        torch.rand(hidden_size, hidden_size, dtype=torch.bfloat16) * val_range
    )
    v_proj_weight = (
        torch.rand(hidden_size, hidden_size, dtype=torch.bfloat16) * val_range
    )

    def to_head_major(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(seq_len, num_heads, head_dim).permute(1, 0, 2).contiguous()

    q = to_head_major(torch.matmul(hidden_states, q_proj_weight.T.contiguous()))
    k = to_head_major(torch.matmul(hidden_states, k_proj_weight.T.contiguous()))
    v = to_head_major(torch.matmul(hidden_states, v_proj_weight.T.contiguous()))

    return {
        "hidden_states": hidden_states,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "q": q,
        "k": k,
        "v": v,
    }
