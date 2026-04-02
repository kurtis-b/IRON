# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from iron.common.utils import torch_dtype_map


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    dtype="bf16",
    seed=42,
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    input_tensor = torch.rand(seq_len, hidden_size, dtype=dtype_torch) * val_range
    input_b = torch.rand(hidden_size, 3 * hidden_size, dtype=dtype_torch) * val_range

    projected = torch.matmul(input_tensor, input_b)

    return {
        "input": input_tensor,
        "input_b": input_b,
        "output_q": projected[:, :hidden_size].contiguous(),
        "output_k": projected[:, hidden_size : 2 * hidden_size].contiguous(),
        "output_v": projected[:, 2 * hidden_size :].contiguous(),
    }
