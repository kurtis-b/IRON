#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.gemm.op import AIEGEMM
from operators.gemm.reference import generate_golden_reference
from operators.common.test_utils import run_test


def generate_test_params(extensive=False):
    M_list = [2048] if not extensive else [2048]
    K_list = [2048] if not extensive else [2048, 8192, 64]
    N_list = [2048] if not extensive else [2048, 8192]
    batch_list = (
        [(1, 0), (4, 0), (4, 1)]
        if not extensive
        else [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (8, 0), (8, 1)]
    )  # (batch_size, batch_stride_dim)
    m, k, n = 64, 64, 64
    num_aie_columns = 8
    col_maj = [(False, False), (True, False), (False, True)]
    trace_size = 0

    params = []
    names = []

    for b_col_maj, c_col_maj in col_maj:
        for M in M_list:
            for K in K_list:
                for N in N_list:
                    for batch_C in batch_list:
                        if N == 8192 and K == 8192:
                            continue  # Untested combination because huge & slow, unused in our application
                        batch_size, batch_stride_dim = batch_C
                        batch_str = (
                            f"_batch{batch_size}_{batch_stride_dim}"
                            if batch_size > 1
                            else ""
                        )
                        params.append(
                            (
                                M,
                                K,
                                N,
                                num_aie_columns,
                                b_col_maj,
                                c_col_maj,
                                m,
                                k,
                                n,
                                trace_size,
                                batch_size,
                                batch_stride_dim,
                            )
                        )
                        names.append(
                            f"gemm_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}_cols_{int(b_col_maj)}_bcolmaj_{int(c_col_maj)}_ccolmaj{batch_str}_{trace_size}"
                        )

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
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,trace_size,batch_size,batch_stride_dim",
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
    trace_size,
    batch_size,
    batch_stride_dim,
    aie_context,
):
    # Create batch tuples for A, B, and C
    # For simplicity, use the same batch configuration for all matrices
    # Batch stride dim example: 0 means batch in the M dimension for A
    # and 1 means batch in the K dimension
    batch_A = (args.batch_size, args.batch_stride_dim)
    batch_B = (args.batch_size, args.batch_stride_dim)
    batch_C = (args.batch_size, args.batch_stride_dim)

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

    operator = AIEGEMM(
        M=M,
        K=K,
        N=N,
        num_aie_columns=num_aie_columns,
        prio_accuracy=True,
        emulate_bf16_mmul_with_bfp16=False,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        batch_A=batch_A,
        batch_B=batch_B,
        batch_C=batch_C,
        context=aie_context,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "B": golden_ref["input_b"].flatten(),
    }
    output_buffers = {"C": golden_ref["output"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )

    gflops = (2.0 * M * K * N * batch_size) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, f"Test failed with errors: {errors}"
