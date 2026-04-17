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
    b_col_maj=False,
    c_col_maj=False,
    batch_A=(1, 0),
    batch_B=(1, 0),
    batch_C=(1, 0),
):
    """Generate golden reference for GEMM operation.

    Args:
        M: Number of rows in matrix A
        K: Number of columns in A / rows in B
        N: Number of columns in matrix B
        dtype: Data type for tensors
        seed: Random seed for reproducibility
        b_col_maj: Whether B matrix is column-major
        c_col_maj: Whether output C matrix is column-major
        batch_A: Tuple of (batch size, batch stride dim) for matrix A
        batch_B: Tuple of (batch size, batch stride dim) for matrix B
        batch_C: Tuple of (batch size, batch stride dim) for output matrix C

    Returns:
        Dictionary with 'input', 'input_b', and 'output' tensors
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    batch_size_A, batch_stride_dim_A = batch_A
    batch_size_B, batch_stride_dim_B = batch_B
    batch_size_C, batch_stride_dim_C = batch_C

    # Determine if this is a batched operation
    is_batched = batch_size_C > 1

    if is_batched:
        # Generate batched inputs based on batch stride dimensions
        if batch_stride_dim_A == 0:
            input_a = torch.randn(batch_size_A, M, K, dtype=dtype_torch) * val_range
        else:
            input_a = torch.randn(M, batch_size_A, K, dtype=dtype_torch) * val_range

        if batch_stride_dim_B == 0:
            input_b = torch.rand(batch_size_B, K, N, dtype=dtype_torch) * val_range
        else:
            input_b = torch.rand(K, batch_size_B, N, dtype=dtype_torch) * val_range

        # Compute batched matrix multiplication
        # For batch_stride_dim == 0: (batch, M, K) @ (batch, K, N) -> (batch, M, N)
        # For batch_stride_dim == 1: need to move batch dim for matmul
        if batch_stride_dim_A == 0:
            input_a_perm = input_a  # (batch, M, K)
        else:
            input_a_perm = input_a.permute(1, 0, 2)  # (batch, M, K)

        if batch_stride_dim_B == 0:
            input_b_perm = input_b  # (batch, K, N)
        else:
            input_b_perm = input_b.permute(1, 0, 2)  # (batch, K, N)

        output_perm = torch.matmul(input_a_perm, input_b_perm)  # (batch, M, N)

        if batch_stride_dim_C == 1:
            output = output_perm.permute(1, 0, 2)  # (M, batch, N)
        else:
            output = output_perm  # (batch, M, N)

        # Combine into 2D matrices across the batch stride dimension
        if batch_stride_dim_A == 0:
            # Combine along first dimension: (batch, M, K) -> (batch*M, K)
            input_a = input_a.reshape(batch_size_A * M, K)
        else:
            # Combine along last dimension: (M, batch, K) -> (M, K*batch)
            input_a = input_a.reshape(M, K * batch_size_A)

        if b_col_maj:
            if batch_stride_dim_B == 0:
                # Combine along first dimension: (batch, K, N) -> (batch, N, K) -> (batch*N, K)
                input_b = input_b.transpose(-2, -1)
                input_b = input_b.reshape(batch_size_B * N, K)
            else:
                # Combine along last dimension: (K, batch, N) -> (N, batch, K) -> (N, K*batch)
                input_b = input_b.transpose(0, 2)
                input_b = input_b.reshape(N, K * batch_size_B)
        else:
            if batch_stride_dim_B == 0:
                # Combine along first dimension: (batch, K, N) -> (batch*K, N)
                input_b = input_b.reshape(batch_size_B * K, N)
            else:
                # Combine along last dimension: (K, batch, N) -> (K, N*batch)
                input_b = input_b.reshape(K, N * batch_size_B)

        if c_col_maj:
            # Transpose last two dimensions for each batch
            if batch_stride_dim_C == 0:
                # Combine along first dimension: (batch, M, N) -> (batch, N, M) -> (batch*N, M)
                output = output.transpose(-2, -1)
                output = output.reshape(batch_size_C * N, M)
            else:
                # Combine along last dimension: (M, batch, N) -> (N, batch, M) -> (N, M*batch)
                output = output.transpose(0, 2)
                output = output.reshape(N, M * batch_size_C)
        else:
            if batch_stride_dim_C == 0:
                # Combine along first dimension: (batch, M, N) -> (batch*M, N)
                output = output.reshape(batch_size_C * M, N)
            else:
                # Combine along last dimension: (M, batch, N) -> (M, N*batch)
                output = output.reshape(M, N * batch_size_C)
    else:
        # Generate non-batched inputs
        input_a = torch.randn(M, K, dtype=dtype_torch) * val_range
        input_b = torch.rand(K, N, dtype=dtype_torch) * val_range
        if False:
            # The following inputs are useful for debugging;
            # the A matrix becomes a matrix where each element encodes its row and column index,
            # and the B matrix is an identity matrix.
            col_digits = len(str(K - 1)) if K > 0 else 1
            factor = 10 ** (col_digits + 1)
            row_indices = torch.arange(M, dtype=torch.int64).unsqueeze(1)
            col_indices = torch.arange(K, dtype=torch.int64).unsqueeze(0)
            input_a = (row_indices * factor + col_indices).to(dtype=dtype_torch)
            input_b = torch.zeros(K, N, dtype=dtype_torch)
            diag_dim = min(K, N)
            input_b[:diag_dim, :diag_dim] = torch.eye(diag_dim, dtype=dtype_torch)
        output = torch.matmul(input_a, input_b)

        if b_col_maj:
            input_b = input_b.T
        if c_col_maj:
            output = output.T

    return {"input": input_a, "input_b": input_b, "output": output}
