# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from operators.common.utils import torch_dtype_map

DEBUG_FFN_PATH = 0
DEBUG_RESIDUAL_PATH = 1


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
            - input_and_residual: Combined input buffer [input; input_residual] (2M, K)
            - input_b_up: Up-projection weight (K, N)
            - input_b_down: Down-projection weight (N, K)
            - output: Final output after FFN block (M, K)
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]
    is_debug_mode = debug_mode in (DEBUG_FFN_PATH, DEBUG_RESIDUAL_PATH)

    if debug_mode == DEBUG_FFN_PATH:
        input_tensor = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
    elif debug_mode == DEBUG_RESIDUAL_PATH:
        input_tensor = torch.zeros(M, K, dtype=dtype_torch)
    else:
        input_tensor = torch.rand(M, K, dtype=dtype_torch) * val_range

    if debug_mode == DEBUG_FFN_PATH:
        input_residual = torch.zeros(M, K, dtype=dtype_torch)
    elif debug_mode == DEBUG_RESIDUAL_PATH:
        # Use a range so index flow through the residual path is easy to validate.
        input_residual = torch.arange(M * K, dtype=dtype_torch).reshape(M, K)
    else:
        input_residual = torch.rand(M, K, dtype=dtype_torch) * val_range

    ln2_weights = (
        torch.ones(K, dtype=dtype_torch)
        if is_debug_mode
        else torch.rand(K, dtype=dtype_torch) * val_range
    )
    up_weight = (
        torch.eye(K, N, dtype=dtype_torch)
        if is_debug_mode
        else torch.randn(K, N, dtype=dtype_torch) * val_range
    )
    down_weight = (
        torch.eye(N, K, dtype=dtype_torch)
        if is_debug_mode
        else torch.randn(N, K, dtype=dtype_torch) * val_range
    )

    up_proj_output = torch.matmul(input_tensor, up_weight)
    gelu_output = (
        up_proj_output.clone()
        if is_debug_mode
        else torch.nn.functional.gelu(up_proj_output)
    )
    down_proj_output = torch.matmul(gelu_output, down_weight)

    if debug_mode == DEBUG_FFN_PATH:
        # Kernel pass-through mode for AddNorm input.
        output = down_proj_output.clone()
    elif debug_mode == DEBUG_RESIDUAL_PATH:
        # Kernel pass-through mode for AddNorm residual input.
        output = input_residual.clone()
    else:
        layer_norm2_output = torch.nn.functional.layer_norm(
            down_proj_output, normalized_shape=(K,), weight=ln2_weights, bias=None
        )
        output = layer_norm2_output + input_residual

    input_and_residual = torch.cat((input_tensor, input_residual), dim=0)

    return {
        "input": input_tensor,
        "input_residual": input_residual,
        "input_and_residual": input_and_residual,
        "input_b_up": up_weight,
        "input_b_down": down_weight,
        "weight2": ln2_weights,
        "output": output,
    }
