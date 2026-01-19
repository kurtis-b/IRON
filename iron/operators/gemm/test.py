#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import logging


from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        params = [
            #   M,     K,     N, num_aie_columns, b_col_maj, c_col_maj,   m,   k,   n, prio_accuracy, emulate_bf16, trace_size, batch_size, batch_stride_dim_A, batch_stride_dim_B, batch_stride_dim_C
            (512, 768, 768, 8, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
            (512, 64, 512, 8, False, False, 64, 64, 64, False, True, 0, 12, 1, 1, 0),
            (512, 512, 64, 4, False, False, 64, 64, 16, False, True, 0, 12, 0, 1, 1),
            (512, 768, 3072, 8, False, False, 64, 48, 96, False, True, 0, 1, 0, 0, 0),
            (512, 3072, 768, 8, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
        ]
        extensive_params = [
            (512, 768, 768, 4, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
            (512, 64, 512, 4, False, False, 64, 64, 64, False, True, 0, 12, 1, 1, 0),
            (512, 512, 64, 2, False, False, 64, 64, 16, False, True, 0, 12, 0, 1, 1),
            (512, 768, 3072, 4, False, False, 64, 48, 96, False, True, 0, 1, 0, 0, 0),
            (512, 3072, 768, 4, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
        ]
    else:
        # fmt: off
        params = [
            #   M,     K,     N, num_aie_columns, b_col_maj, c_col_maj,   m,   k,   n, prio_accuracy, emulate_bf16, trace_size, batch_size, batch_stride_dim_A, batch_stride_dim_B, batch_stride_dim_C
            (2048,  2048,  2048,               1,     False,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  2048,               2,      True,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  2048,               8,      True,      True,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            ( 384,  1536,  1792,               4,      True,     False,  32,  48,  64, True,          False,        0,           1,         0,                  0,                  0),
            (1792,   896,  1152,               8,     False,      True,  64,  32,  48, True,          False,        0,           1,         0,                  0,                  0),
            ( 896,  1792,   640,               8,     False,      True,  32,  64,  80, True,          False,        0,           1,         0,                  0,                  0),
            ( 192,   384,    64,               4,     False,     False,  48,  96,  16, True,          False,        0,           1,         0,                  0,                  0),
            ( 192,   384,    64,               4,      True,      True,  48,  96,  16, True,          False,        0,           1,         0,                  0,                  0),
            ( 512,   768,   768,               8,     False,     False,  64,  96,  48, False,          True,        0,           1,         0,                  0,                  0),
            ( 512,    64,   512,               8,     False,     False,  64,  64,  64, False,          True,        0,          12,         1,                  1,                  0),
            ( 512,   512,    64,               4,     False,     False,  64,  64,  16, False,          True,        0,          12,         0,                  1,                  1),
            ( 512,   768,  3072,               8,     False,     False,  64,  48,  96, False,          True,        0,           1,         0,                  0,                  0),
            ( 512,  3072,   768,               8,     False,     False,  64,  96,  48, False,          True,        0,           1,         0,                  0,                  0),
        ]
        extensive_params = [
            (2048,  2048,  2048,               8,     False,     False,  32,  32, 128, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  8192,               2,     False,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  8192,  2048,               2,     False,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  2048,               2,     False,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  8192,               2,     False,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  2048,               8,      True,     False, 128,  32,  32, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  8192,               2,      True,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  8192,  2048,               2,      True,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  2048,               2,      True,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  8192,               2,      True,     False,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  2048,               2,     False,      True,   8,  16,  32, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  2048,  8192,               2,     False,      True,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,  8192,  2048,               2,     False,      True,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  2048,               2,     False,      True,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
            (2048,    64,  8192,               2,     False,      True,  64,  64,  64, True,          False,        0,           1,         0,                  0,                  0),
        ]
        # fmt: on

    if extensive:
        params = extensive_params

    names = []
    for (
        M,
        K,
        N,
        num_aie_columns,
        b_col_maj,
        c_col_maj,
        m,
        k,
        n,
        prio_accuracy,
        emulate_bf16,
        trace_size,
        batch_size,
        batch_stride_dim_A,
        batch_stride_dim_B,
        batch_stride_dim_C,
    ) in params:
        name = f"gemm_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}cols"
        if b_col_maj:
            name += "_bcolmaj"
        if c_col_maj:
            name += "_ccolmaj"
        if trace_size > 0:
            name += f"_{trace_size}trace"
        name += f"_prioacc{prio_accuracy}_emubf16{emulate_bf16}"
        name += f"_batch{batch_size}_stridedims{batch_stride_dim_A}{batch_stride_dim_B}{batch_stride_dim_C}"
        names.append(name)

    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Combine params with marks - extensive params get pytest.mark.extensive
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,prio_accuracy,emulate_bf16,trace_size,batch_size,batch_stride_dim_A,batch_stride_dim_B,batch_stride_dim_C",
    all_params,
)
def test_gemm(
    M,
    K,
    N,
    num_aie_columns,
    b_col_maj,
    c_col_maj,
    m,
    k,
    n,
    prio_accuracy,
    emulate_bf16,
    trace_size,
    batch_size,
    batch_stride_dim_A,
    batch_stride_dim_B,
    batch_stride_dim_C,
    aie_context,
):
    logging.debug(
        f"Testing GEMM with M={M}, K={K}, N={N}, m={m}, k={k}, n={n}, num_aie_columns={num_aie_columns}, b_col_maj={b_col_maj}, c_col_maj={c_col_maj}, prio_accuracy={prio_accuracy}, emulate_bf16={emulate_bf16}, batch_size={batch_size}, batch_stride_dim_A={batch_stride_dim_A}, batch_stride_dim_B={batch_stride_dim_B}, batch_stride_dim_C={batch_stride_dim_C}"
    )
    # Create batch tuples for A, B, and C
    # For simplicity, use the same batch configuration for all matrices
    # Batch stride dim example: 0 means batch in the M dimension for A
    # and 1 means batch in the K dimension
    batch_A = (batch_size, batch_stride_dim_A)
    batch_B = (batch_size, batch_stride_dim_B)
    batch_C = (batch_size, batch_stride_dim_C)

    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        batch_A=batch_A,
        batch_B=batch_B,
        batch_C=batch_C,
    )

    aie_gemm_config = {
        "prio_accuracy": prio_accuracy,
        "emulate_bf16_mmul_with_bfp16": emulate_bf16,
    }
    operator = AIEGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        batch_A=batch_A,
        batch_B=batch_B,
        batch_C=batch_C,
        context=aie_context,
        **aie_gemm_config,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "B": golden_ref["input_b"].flatten(),
    }
    output_buffers = {"C": golden_ref["output"].flatten()}

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=0.1,
            abs_tol=0.5,
            warmup_iters=10,
            timed_iters=100,
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.1, abs_tol=0.5
        )

    # Use batch_size of C for total operations
    gflops = (2.0 * M * K * N * batch_size) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(M * N * batch_size * error_threshold)

    if errors:
        print(
            "({} errors out of {} max allowable)".format(
                len(errors["C"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"
