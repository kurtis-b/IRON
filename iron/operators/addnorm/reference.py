# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.utils import torch_dtype_map


def generate_golden_reference(rows: int, cols: int, dtype="bf16", seed=42):
    torch.manual_seed(seed)
    val_range = 4
    input1_tensor = torch.rand(rows, cols, dtype=torch_dtype_map[dtype]) * val_range
    input2_tensor = torch.rand(rows, cols, dtype=torch_dtype_map[dtype]) * val_range
    weights = torch.rand(cols, dtype=torch_dtype_map[dtype]) * val_range

    # Compute addition followed by layer norm.
    preadd_tensor = input1_tensor + input2_tensor
    output_tensor = torch.nn.functional.layer_norm(
        preadd_tensor, normalized_shape=(cols,), weight=weights, bias=None
    )

    return {
        "input1": input1_tensor,
        "input2": input2_tensor,
        "weight": weights,
        "output": output_tensor,
    }
