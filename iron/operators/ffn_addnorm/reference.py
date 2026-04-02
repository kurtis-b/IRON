# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.utils import torch_dtype_map


def generate_golden_reference(
    M: int,
    K: int,
    N: int,
    dtype="bf16",
    seed=42,
    debug_mode=-1,
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
        debug_mode:
            - If 0, check indexes through FFN path,
            - if 1, check indexes through residual connection path,
            - else, random data

    Returns:
        Dictionary containing:
            - input: Input tensor for FFN (M, K)
            - input_residual: Input tensor for residual addition (M, K)
            - input_b_up: Up-projection weight (K, N)
            - input_b_down: Down-projection weight (N, K)
            - output: Final output after FFN block (M, K)
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Generate input tensor (M, K)
    if debug_mode == 0:
        input_tensor = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
        ln2_weights = torch.ones(K, dtype=dtype_torch)
    elif debug_mode == 1:
        input_tensor = torch.zeros(M, K, dtype=dtype_torch)
        ln2_weights = torch.ones(K, dtype=dtype_torch)
    else:
        input_tensor = torch.rand(M, K, dtype=dtype_torch) * val_range
        ln2_weights = torch.rand(K, dtype=dtype_torch) * val_range

    # Generate input for residual addition (M, K)
    if debug_mode == 0:
        input_residual = torch.zeros(M, K, dtype=dtype_torch)
    elif debug_mode == 1:
        # Use a range only across columns since the values get really large, which could affect the layer norm calculations
        input_residual = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
    else:
        input_residual = torch.rand(M, K, dtype=dtype_torch) * val_range

    # Generate up-projection weight (K, N)
    if debug_mode == 1 or debug_mode == 0:
        up_weight = torch.eye(K, N, dtype=dtype_torch)
    else:
        up_weight = torch.randn(K, N, dtype=dtype_torch) * val_range

    # Up-projection: (M, K) @ (K, N) = (M, N)
    up_proj_output = torch.matmul(input_tensor, up_weight)

    # GeLU activation
    if debug_mode == 1 or debug_mode == 0:
        gelu_output = up_proj_output.clone()
    else:
        # print(f"Up Projection Output: {up_proj_output}")
        gelu_output = torch.nn.functional.gelu(up_proj_output)

    # Generate down-projection weight (N, K)
    if debug_mode == 1 or debug_mode == 0:
        down_weight = torch.eye(N, K, dtype=dtype_torch)
    else:
        down_weight = torch.randn(N, K, dtype=dtype_torch) * val_range

    # Down-projection: (M, N) @ (N, K) = (M, K)
    down_proj_output = torch.matmul(gelu_output, down_weight)

    # Final layer norm
    if debug_mode == 0:
        # The kernel passes through the input only, so skip layer norm and residual addition for this mode
        output = down_proj_output.clone()
    elif debug_mode == 1:
        # The kernel passes through the residual only, so skip layer norm and add with down_proj_output
        output = input_residual.clone()
    else:
        layer_norm2_output = torch.nn.functional.layer_norm(
            down_proj_output, normalized_shape=(K,), weight=ln2_weights, bias=None
        )
        # for i in range(M):
        #     print(f"Layer Norm 2 Output - Row {i} sum: {layer_norm2_output[i].sum()}")
        #     print(
        #         f"Layer Norm 2 Output - Row {i} sum of squares: {(layer_norm2_output[i] ** 2).sum()}"
        #     )
        # print(f"Layer Norm 2 Input: {down_proj_output}")
        # print(f"Layer Norm 2 Output: {layer_norm2_output}")
        # Final addition with residual
        output = layer_norm2_output + input_residual

    return {
        "input": input_tensor,
        "input_residual": input_residual,
        "input_b_up": up_weight,
        "input_b_down": down_weight,
        "weight2": ln2_weights,
        "output": output,
    }
