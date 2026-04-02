# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.common.utils import torch_dtype_map


def generate_golden_reference(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "bf16",
    seed: int = 42,
    debug_mode: int = -1,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    if debug_mode == 0:
        hidden_states = torch.arange(seq_len * hidden_size, dtype=dtype_torch).reshape(
            seq_len, hidden_size
        )
        residual = torch.zeros(seq_len, hidden_size, dtype=dtype_torch)
        ln1_weight = torch.ones(hidden_size, dtype=dtype_torch)
        ln2_weight = torch.ones(hidden_size, dtype=dtype_torch)
    elif debug_mode == 1:
        hidden_states = torch.zeros(seq_len, hidden_size, dtype=dtype_torch)
        residual = torch.arange(seq_len * hidden_size, dtype=dtype_torch).reshape(
            seq_len, hidden_size
        )
        ln1_weight = torch.ones(hidden_size, dtype=dtype_torch)
        ln2_weight = torch.ones(hidden_size, dtype=dtype_torch)
    else:
        hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype_torch) * val_range
        residual = torch.rand(seq_len, hidden_size, dtype=dtype_torch) * val_range
        ln1_weight = torch.rand(hidden_size, dtype=dtype_torch) * val_range
        ln2_weight = torch.rand(hidden_size, dtype=dtype_torch) * val_range

    preadd = hidden_states + residual
    if debug_mode in (0, 1):
        ln1_out = preadd.clone()
    else:
        ln1_out = torch.nn.functional.layer_norm(
            preadd,
            normalized_shape=(hidden_size,),
            weight=ln1_weight,
            bias=None,
        )

    if debug_mode in (0, 1):
        up_weight = torch.eye(hidden_size, intermediate_size, dtype=dtype_torch)
        down_weight = torch.eye(intermediate_size, hidden_size, dtype=dtype_torch)
    else:
        up_weight = (
            torch.randn(hidden_size, intermediate_size, dtype=dtype_torch) * val_range
        )
        down_weight = (
            torch.randn(intermediate_size, hidden_size, dtype=dtype_torch) * val_range
        )

    up_proj = torch.matmul(ln1_out, up_weight)
    gelu = (
        up_proj.clone() if debug_mode in (0, 1) else torch.nn.functional.gelu(up_proj)
    )
    down_proj = torch.matmul(gelu, down_weight)

    if debug_mode == 0:
        output = down_proj.clone()
    elif debug_mode == 1:
        output = preadd.clone()
    else:
        output = torch.nn.functional.layer_norm(
            down_proj + preadd,
            normalized_shape=(hidden_size,),
            weight=ln2_weight,
            bias=None,
        )

    return {
        "hidden_states": hidden_states,
        "residual": residual,
        "ffn_up_weight": up_weight,
        "ffn_down_weight": down_weight,
        "ln1_weight": ln1_weight,
        "ln2_weight": ln2_weight,
        "output": output,
    }
