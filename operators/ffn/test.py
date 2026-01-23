#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.ffn.op import AIEFFN
from operators.ffn.reference import generate_golden_reference
from operators.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        params = [
            #   M,     K,     N,    num_aie_columns, b_col_maj, c_col_maj,   m,   k,   n,   prio_accuracy, emulate_bf16, trace_size, down_proj_depth, n_a_tiles_distributed, n_b_tiles_distributed
            (64, 48, 96, 2, False, False, 64, 48, 96, False, True, 0, 1, 1, 1),
            (64 * 4, 48, 96, 2, False, False, 64, 48, 96, False, True, 0, 1, 1, 1),
            (64, 48 * 4, 96, 2, False, False, 64, 48, 96, False, True, 0, 1, 1, 1),
            (64, 48, 96 * 4, 2, False, False, 64, 48, 96, False, True, 0, 1, 1, 1),
            # (512, 768, 3072, 2, False, False, 64, 48, 96, False, True, 0, 1, 1, 1),
            # (512, 768, 3072, 8, False, False, 64, 48, 96, False, True, 0, 8, 8, 2),
        ]
        extensive_params = []
    else:
        params = []
        extensive_params = []

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
        down_proj_depth,
        n_a_tiles_distributed,
        n_b_tiles_distributed,
    ) in params:
        name = f"gemm_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}cols"
        if b_col_maj:
            name += "_bcolmaj"
        if c_col_maj:
            name += "_ccolmaj"
        if trace_size > 0:
            name += f"_{trace_size}trace"
        name += f"_prioacc{prio_accuracy}_emubf16{emulate_bf16}"
        name += f"_dprojdepth{down_proj_depth}_nA{n_a_tiles_distributed}_nB{n_b_tiles_distributed}"
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
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,prio_accuracy,emulate_bf16,trace_size,down_proj_depth,n_a_tiles_distributed,n_b_tiles_distributed",
    all_params,
)
def test_ffn(
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
    down_proj_depth,
    n_a_tiles_distributed,
    n_b_tiles_distributed,
    aie_context,
):
    logging.debug(
        f"Testing GEMM with M={M}, K={K}, N={N}, m={m}, k={k}, n={n}, num_aie_columns={num_aie_columns}, b_col_maj={b_col_maj}, c_col_maj={c_col_maj}, prio_accuracy={prio_accuracy}, emulate_bf16={emulate_bf16}, down_proj_depth={down_proj_depth}, n_a_tiles_distributed={n_a_tiles_distributed}, n_b_tiles_distributed={n_b_tiles_distributed}"
    )

    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )

    aie_ffn_config = {
        "prio_accuracy": prio_accuracy,
        "emulate_bf16_mmul_with_bfp16": emulate_bf16,
        "n_a_tiles_distributed": n_a_tiles_distributed,
        "n_b_tiles_distributed": n_b_tiles_distributed,
    }
    operator = AIEFFN(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        down_proj_depth=down_proj_depth,
        num_aie_columns=num_aie_columns,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        context=aie_context,
        **aie_ffn_config,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "B_Up": golden_ref["input_b_up"].flatten(),
        "B_Down": golden_ref["input_b_down"].flatten(),
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

    # 2 GEMMs are executed, with a MAC per element
    gflops = 2 * (2.0 * M * K * N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(M * N * error_threshold)

    if errors:
        print(
            "({} errors out of {} max allowable)".format(
                len(errors["C"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"
