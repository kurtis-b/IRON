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
    partition_N=1,
    batch_A=(1, 0),
    batch_B=(1, 0),
    batch_C=(1, 0),
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    if batch_C[0] > 1:
        batch_size_A, batch_stride_dim_A = batch_A
        batch_size_B, batch_stride_dim_B = batch_B
        batch_size_C, batch_stride_dim_C = batch_C

        if batch_stride_dim_A == 0:
            input_a = torch.randn(batch_size_A, M, K, dtype=dtype_torch) * val_range
        else:
            input_a = torch.randn(M, batch_size_A, K, dtype=dtype_torch) * val_range

        if batch_stride_dim_B == 0:
            input_b = torch.rand(batch_size_B, K, N, dtype=dtype_torch) * val_range
        else:
            input_b = torch.rand(K, batch_size_B, N, dtype=dtype_torch) * val_range

        input_a_perm = input_a if batch_stride_dim_A == 0 else input_a.permute(1, 0, 2)
        input_b_perm = input_b if batch_stride_dim_B == 0 else input_b.permute(1, 0, 2)
        output_perm = torch.matmul(input_a_perm, input_b_perm)
        output = (
            output_perm.permute(1, 0, 2) if batch_stride_dim_C == 1 else output_perm
        )

        if batch_stride_dim_A == 0:
            input_a = input_a.reshape(batch_size_A * M, K)
        else:
            input_a = input_a.reshape(M, K * batch_size_A)

        if b_col_maj:
            if batch_stride_dim_B == 0:
                input_b = input_b.transpose(-2, -1).reshape(batch_size_B * N, K)
            else:
                input_b = input_b.transpose(0, 2).reshape(N, K * batch_size_B)
        else:
            if batch_stride_dim_B == 0:
                input_b = input_b.reshape(batch_size_B * K, N)
            else:
                input_b = input_b.reshape(K, N * batch_size_B)

        if c_col_maj:
            if batch_stride_dim_C == 0:
                output = output.transpose(-2, -1).reshape(batch_size_C * N, M)
            else:
                output = output.transpose(0, 2).reshape(N, M * batch_size_C)
        else:
            if batch_stride_dim_C == 0:
                output = output.reshape(batch_size_C * M, N)
            else:
                output = output.reshape(M, N * batch_size_C)

        return {"input": input_a, "input_b": input_b, "output": output}

    if partition_N != 1 and N % partition_N != 0:
        raise ValueError(f"N ({N}) must be divisible by partition_N ({partition_N})")

    input_a = torch.randn(M, K, dtype=dtype_torch) * val_range
    input_b_full = torch.rand(K, N, dtype=dtype_torch) * val_range
    output_full = torch.matmul(input_a, input_b_full)

    if b_col_maj:
        input_b_full = input_b_full.T
    if c_col_maj:
        output_full = output_full.T

    if partition_N == 1:
        return {"input": input_a, "input_b": input_b_full, "output": output_full}

    input_b = []
    output = []
    for i in range(partition_N):
        col_start = i * (N // partition_N)
        col_end = (i + 1) * (N // partition_N)
        if b_col_maj:
            input_b.append(input_b_full[col_start:col_end, :])
        else:
            input_b.append(input_b_full[:, col_start:col_end])
        if c_col_maj:
            output.append(output_full[col_start:col_end, :])
        else:
            output.append(output_full[:, col_start:col_end])

    return {"input": input_a, "input_b": input_b, "output": output}
