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
    debug_mode=False,
):
    """
    Generate golden reference for BERT Add & Norm -> FFN -> Add & Norm. Using uniform distribution [0, 4) to populate tensors.

    A BERT Add & Norm consists of:
    1. Layer Norm: layer_norm(input1, weight) -> (M, K)
    2. Addition: layer_norm_out + input2 -> (M, K)

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
        debug_mode: If True, use row/col indices as input and identity matrices for weights

    Returns:
        Dictionary containing:
            - input: Input tensor for first layer norm (M, K)
            - input_residual: Input tensor for first residual addition (M, K)
            - input_b_up: Up-projection weight (K, N)
            - input_b_down: Down-projection weight (N, K)
            - output: Final output after FFN block (M, K)
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Generate input tensor (M, K)
    if debug_mode:
        input_tensor = torch.zeros(M, K, dtype=dtype_torch)
        ln1_weights = torch.zeros(K, dtype=dtype_torch)
        ln2_weights = torch.zeros(K, dtype=dtype_torch)
    else:
        input_tensor = torch.rand(M, K, dtype=dtype_torch) * val_range
        ln1_weights = torch.rand(K, dtype=dtype_torch) * val_range
        ln2_weights = torch.rand(K, dtype=dtype_torch) * val_range

    layer_norm1_output = torch.nn.functional.layer_norm(
        input_tensor, normalized_shape=(K,), weight=ln1_weights, bias=None
    )

    # Generate input for residual addition (M, K)
    if debug_mode:
        input_residual = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
    else:
        input_residual = torch.rand(M, K, dtype=dtype_torch) * val_range

    add1_output = layer_norm1_output + input_residual

    # Generate up-projection weight (K, N)
    if debug_mode:
        up_weight = torch.eye(K, N, dtype=dtype_torch)
    else:
        up_weight = torch.rand(K, N, dtype=dtype_torch) * val_range

    # Up-projection: (M, K) @ (K, N) = (M, N)
    up_proj_output = torch.matmul(add1_output, up_weight)

    # GeLU activation
    gelu_output = torch.nn.functional.gelu(up_proj_output)

    # Generate down-projection weight (N, K)
    if debug_mode:
        down_weight = torch.eye(N, K, dtype=dtype_torch)
    else:
        down_weight = torch.rand(N, K, dtype=dtype_torch) * val_range

    # Down-projection: (M, N) @ (N, K) = (M, K)
    down_proj_output = torch.matmul(gelu_output, down_weight)

    # Final layer norm
    layer_norm2_output = torch.nn.functional.layer_norm(
        down_proj_output, normalized_shape=(K,), weight=ln2_weights, bias=None
    )

    # Final addition with residual
    output = layer_norm2_output + add1_output

    return {
        "input": input_tensor,
        "input_residual": input_residual,
        "input_b_up": up_weight,
        "input_b_down": down_weight,
        "weight1": ln1_weights,
        "weight2": ln2_weights,
        "output": output,
    }
