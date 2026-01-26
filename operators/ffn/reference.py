# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from operators.common.utils import torch_dtype_map


def generate_golden_reference(
    M: int,
    K: int,
    N: int,
    dtype="bf16",
    seed=42,
    b_col_maj=False,
    c_col_maj=False,
    debug_mode=False,
):
    """
    Generate golden reference for BERT FFN block. Using uniform distribution [0, 4) to populate tensors.

    A BERT FFN block consists of:
    1. Up-projection: input @ W1 -> (M, K) @ (K, N) = (M, N)
    2. GeLU activation: gelu(intermediate) -> (M, N)
    3. Down-projection: gelu_out @ W2 -> (M, N) @ (N, K) = (M, K)

    Args:
        M: Sequence length
        K: Hidden size (input/output dimension)
        N: Intermediate size (typically 4*K for BERT)
        dtype: Data type for tensors
        seed: Random seed for reproducibility
        b_col_maj: Whether weight matrix is column-major
        c_col_maj: Whether output matrix is column-major
        debug_mode: If True, use row/col indices as input and identity matrices for weights

    Returns:
        Dictionary containing:
            - input: Input tensor (M, K)
            - input_b: Up-projection weight (K, N) or (N, K) if b_col_maj
            - output: Final output after FFN block (M, K) or (K, M) if c_col_maj
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Generate input tensor (M, K)
    if debug_mode:
        input_tensor = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
    else:
        input_tensor = torch.rand(M, K, dtype=dtype_torch) * val_range

    # Generate up-projection weight (K, N)
    if debug_mode:
        up_weight = torch.eye(K, N, dtype=dtype_torch)
    else:
        up_weight = torch.rand(K, N, dtype=dtype_torch) * val_range

    # Up-projection: (M, K) @ (K, N) = (M, N)
    intermediate = torch.matmul(input_tensor, up_weight)

    # GeLU activation
    gelu_output = torch.nn.functional.gelu(intermediate)

    # Generate down-projection weight (N, K)
    if debug_mode:
        down_weight = torch.eye(N, K, dtype=dtype_torch)
    else:
        down_weight = torch.rand(N, K, dtype=dtype_torch) * val_range

    # Down-projection: (M, N) @ (N, K) = (M, K)
    output = torch.matmul(gelu_output, down_weight)

    # Handle column-major layouts if requested
    if b_col_maj:
        up_weight = up_weight.T
        down_weight = down_weight.T
    if c_col_maj:
        output = output.T

    return {
        "input": input_tensor,
        "input_b_up": up_weight,
        "input_b_down": down_weight,
        "output": output,
    }
