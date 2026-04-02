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
) -> dict[str, object]:
    del debug_mode  # Block 3 uses the functional reference path only.

    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype_torch) * val_range
    residual = torch.rand(seq_len, hidden_size, dtype=dtype_torch) * val_range
    ln1_weight = torch.rand(hidden_size, dtype=dtype_torch) * val_range
    ln2_weight = torch.rand(hidden_size, dtype=dtype_torch) * val_range
    up_weight = (
        torch.randn(hidden_size, intermediate_size, dtype=dtype_torch) * val_range
    )
    down_weight = (
        torch.randn(intermediate_size, hidden_size, dtype=dtype_torch) * val_range
    )

    preadd = hidden_states + residual
    addnorm1 = torch.nn.functional.layer_norm(
        preadd,
        normalized_shape=(hidden_size,),
        weight=ln1_weight,
        bias=None,
    )
    up_proj = torch.matmul(addnorm1, up_weight)
    gelu = torch.nn.functional.gelu(up_proj)
    down_proj = torch.matmul(gelu, down_weight)
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
        "preadd": preadd,
        "ln1_out": addnorm1,
        "down_proj": down_proj,
        "output": output,
    }
