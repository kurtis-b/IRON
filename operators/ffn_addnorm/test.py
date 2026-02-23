#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.ffn_addnorm.op import AIEFFNAN
from operators.ffn_addnorm.reference import generate_golden_reference
from operators.common.test_utils import run_test

TEST_BERT = True
INCLUDE_SIMPLE_TESTS = True
DEBUG_MODE = -1
"""
Debug mode 0: 
    Input to FFN indexes, residual connection is set to 0's. 
    GEMM weights are identity matrices. Layer norm weights are all 1's.
    The fused layer norm add kernels will pass through inputs.
Debug mode 1: 
    Input to FFN is set to 0's, first residual connection are indexes. 
    GEMM weights are identity matrices. Layer norm weights are all 1's.
    The fused layer norm add kernels will pass through residual connections.
Debug mode < 0 or > 1:
    Random data for all inputs, normal operation
"""


def generate_test_params(extensive=False):
    if TEST_BERT:
        params_simple = []
        if INCLUDE_SIMPLE_TESTS:
            params_simple = [
                #   M,     K,     N,    num_aie_columns,   m,   k,   n, trace_size, down_proj_depth, nA_tiles_distributed, nB_tiles_distributed, stage_only, gelu_stage
                # GeLU fused with up projection
                ## Baselines
                ### No compute
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, -1, 0),
                ### Only up projection + GeLU
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 0, 0),
                ### Only down projection
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 1, 0),
                ### Only second add & layer norm
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 2, 0),
                ## All compute executed
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, None, 0),
                ## M scaled up from baseline
                (32 * 8, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, None, 0),
                # K scaled up from baseline, for now down_proj_depth is required to be scaled so that the full K is processed
                (32, 96 * 2, 96, 2, 32, 96, 96, 0, 2, 1, 1, None, 0),
                (32, 96 * 8, 96, 2, 32, 96, 96, 0, 8, 1, 1, None, 0),
                ## N scaled up from baseline
                (32, 96, 96 * 8, 2, 32, 96, 96, 0, 1, 1, 1, None, 0),
                ## M scaled up with mathing scaling with nA_tiles_distributed (duplicates pipeline with more A streams)
                (32 * 8, 96, 96, 4, 32, 96, 96, 0, 1, 2, 1, None, 0),
                (32 * 8, 96, 96, 8, 32, 96, 96, 0, 1, 4, 1, None, 0),
                ## N scaled up with matching scaling with nB_tiles_distributed (duplicates pipeline with more B_Up/B_Down streams)
                (32, 96, 96 * 2, 8, 32, 96, 96, 0, 1, 1, 2, None, 0),
                (32, 96, 96 * 8, 8, 32, 96, 96, 0, 1, 1, 4, None, 0),
                # BERT workload
                (512, 768, 3072, 2, 32, 96, 96, 0, 8, 1, 1, None, 0),
                # Bottleneck testing
                ## Only up projection + GeLU
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 0, 0),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 0, 0),
                ## Only down projection
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 1, 0),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 1, 0),
                ## Only second add & layer norm
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 2, 0),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 2, 0),
                # GeLU fused with down projection
                ## Baselines
                ### No compute
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, -1, 1),
                ### Only up projection + GeLU
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 0, 1),
                ### Only down projection
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 1, 1),
                ### Only second add & layer norm
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, 2, 1),
                ### All compute executed
                (32, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, None, 1),
                ## M scaled up from baseline
                (32 * 8, 96, 96, 2, 32, 96, 96, 0, 1, 1, 1, None, 1),
                ## K scaled up from baseline, for now down_proj_depth is required to be scaled so that the full K is processed
                (32, 96 * 8, 96, 2, 32, 96, 96, 0, 8, 1, 1, None, 1),
                ## N scaled up from baseline
                (32, 96, 96 * 8, 2, 32, 96, 96, 0, 1, 1, 1, None, 1),
                ## M scaled up with mathing scaling with nA_tiles_distributed (duplicates pipeline with more A streams)
                (32 * 8, 96, 96, 4, 32, 96, 96, 0, 1, 2, 1, None, 1),
                (32 * 8, 96, 96, 8, 32, 96, 96, 0, 1, 4, 1, None, 1),
                ## N scaled up with matching scaling with nB_tiles_distributed (duplicates pipeline with more B_Up/B_Down streams)
                (32, 96, 96 * 2, 8, 32, 96, 96, 0, 1, 1, 2, None, 1),
                (32, 96, 96 * 8, 8, 32, 96, 96, 0, 1, 1, 4, None, 1),
                # BERT workload
                (512, 768, 3072, 2, 32, 96, 96, 0, 8, 1, 1, None, 1),
                # Bottleneck testing
                ## Only up projection + GeLU
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 0, 1),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 0, 1),
                ## Only down projection
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 1, 1),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 1, 1),
                ## Only second add & layer norm
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, 2, 1),
                (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, 2, 1),
            ]
        params = params_simple + [
            #   M,     K,     N,    num_aie_columns,   m,   k,   n, trace_size, down_proj_depth, nA_tiles_distributed, nB_tiles_distributed, stage_only, gelu_stage
            # GeLU fused with up projection
            (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, None, 0),
            (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, None, 0),
            # GeLU fused with down projection
            (512, 768, 3072, 8, 32, 96, 64, 0, 8, 2, 6, None, 1),
            (512, 768, 3072, 8, 32, 96, 64, 0, 8, 4, 3, None, 1),
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
        m,
        k,
        n,
        trace_size,
        down_proj_depth,
        nA_tiles_distributed,
        nB_tiles_distributed,
        stage_only,
        gelu_stage,
    ) in params:
        name = f"an_ffn_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}cols"
        if trace_size > 0:
            name += f"_{trace_size}trace"
        name += f"_dprojdepth{down_proj_depth}_nA{nA_tiles_distributed}_nB{nB_tiles_distributed}"
        if stage_only is not None:
            name += f"_stageonly{stage_only}"
        name += f"_gelustage{gelu_stage}"
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
    "M,K,N,num_aie_columns,m,k,n,trace_size,down_proj_depth,nA_tiles_distributed,nB_tiles_distributed,stage_only,gelu_stage",
    all_params,
)
def test_ffn(
    M,
    K,
    N,
    num_aie_columns,
    m,
    k,
    n,
    trace_size,
    down_proj_depth,
    nA_tiles_distributed,
    nB_tiles_distributed,
    stage_only,
    gelu_stage,
    aie_context,
):
    logging.debug(
        f"Testing GEMM with M={M}, K={K}, N={N}, m={m}, k={k}, n={n}, "
        f"num_aie_columns={num_aie_columns}, down_proj_depth={down_proj_depth}, "
        f"nA_tiles_distributed={nA_tiles_distributed}, "
        f"nB_tiles_distributed={nB_tiles_distributed}, "
        f"stage_only={stage_only}, gelu_stage={gelu_stage}"
    )

    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        debug_mode=DEBUG_MODE,
    )

    aie_ffn_config = {
        "emulate_bf32_mmul_with_bfp32": True,
        "nA_tiles_distributed": nA_tiles_distributed,
        "nB_tiles_distributed": nB_tiles_distributed,
        "stage_only": stage_only,
        "gelu_stage": gelu_stage,
    }
    operator = AIEFFNAN(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        down_proj_depth=down_proj_depth,
        num_aie_columns=num_aie_columns,
        ln2_weight=golden_ref["weight2"],
        debug_mode=DEBUG_MODE,
        context=aie_context,
        **aie_ffn_config,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "R": golden_ref["input_residual"].flatten(),
        "B_Up": golden_ref["input_b_up"].flatten(),
        "B_Down": golden_ref["input_b_down"].flatten(),
    }
    output_buffers = {"C": golden_ref["output"].flatten()}

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=4.0e-2,
            abs_tol=1.5e-1,
            warmup_iters=10,
            timed_iters=100,
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
        )

    # 2 GEMMs are executed, with a MAC per element
    gflops = 2 * (2.0 * M * K * N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.005
    max_acceptable_errors = int(M * K * error_threshold)

    if stage_only is not None:
        print(f"Stage only mode: {stage_only}, skipping error check.")
    else:
        if (
            errors
        ):  # If only one stage is performing the computation, skip error check since the output will always be wrong
            logging.info(
                "({} errors out of {} max allowable)".format(
                    len(errors["C"]), max_acceptable_errors
                )
            )
            assert (
                len(errors["C"]) <= max_acceptable_errors
            ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"
