#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import logging
from aie.helpers.taplib import TensorAccessPattern

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.design_batched import (
    DMA_BD_MAX_STRIDE,
    _expand_dma_tap_for_stride_limit,
    my_matmul as batched_gemm_design,
)
from iron.operators.gemm.design import (
    my_matmul as gemm_design,
    plan_gemm_fill_drain_tasks,
)
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
            # (512, 768, 768, 4, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
            # (512, 64, 512, 4, False, False, 64, 64, 64, False, True, 0, 12, 1, 1, 0),
            # (512, 512, 64, 2, False, False, 64, 64, 16, False, True, 0, 12, 0, 1, 1),
            # (512, 768, 3072, 4, False, False, 64, 48, 96, False, True, 0, 1, 0, 0, 0),
            # (512, 3072, 768, 4, False, False, 64, 96, 48, False, True, 0, 1, 0, 0, 0),
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


def test_expand_dma_tap_for_stride_limit_collapses_unit_dims_without_splitting():
    tap = TensorAccessPattern(
        (64, 196608),
        offset=0,
        sizes=[32, 1, 64, 64],
        strides=[512, 12582912, 196608, 1],
    )

    planned_taps, was_split = _expand_dma_tap_for_stride_limit(tap)

    assert was_split is False
    assert len(planned_taps) == 1
    assert list(planned_taps[0].sizes) == [32, 64, 64]
    assert list(planned_taps[0].strides) == [512, 196608, 1]
    assert max(planned_taps[0].strides) <= DMA_BD_MAX_STRIDE


def test_expand_dma_tap_for_stride_limit_splits_even_oversized_dims():
    tap = TensorAccessPattern(
        (49152, 16384),
        offset=0,
        sizes=[2, 32, 256, 64],
        strides=[4194304, 512, 16384, 1],
    )

    planned_taps, was_split = _expand_dma_tap_for_stride_limit(tap)

    assert was_split is True
    assert len(planned_taps) == 2
    assert [planned_taps[0].offset, planned_taps[1].offset] == [0, 4194304]
    for planned_tap in planned_taps:
        assert max(planned_tap.strides) <= DMA_BD_MAX_STRIDE


def test_batched_gemm_design_avoids_oversized_dma_strides_for_large_attn_scores():
    module = batched_gemm_design(
        dev="npu2",
        M=4096,
        K=64,
        N=16384,
        m=64,
        k=64,
        n=64,
        n_aie_cols=8,
        dtype_in_str="bf16",
        dtype_out_str="bf16",
        b_col_maj=0,
        c_col_maj=0,
        use_scalar=False,
        emulate_bf16_mmul_with_bfp16=True,
        prio_accuracy=False,
        trace_size=0,
        archive=None,
        generate_taps=False,
        batch_A=(12, 1),
        batch_B=(12, 1),
        batch_C=(12, 0),
        input_a_buffer_shape=(16384, 768),
    )

    module_text = str(module)
    assert "stride = 12582912" not in module_text
    assert "stride = 4194304" not in module_text


def _tap_signature(tap: TensorAccessPattern) -> tuple[object, ...]:
    return (
        tuple(tap.tensor_dims),
        int(tap.offset),
        tuple(int(value) for value in tap.sizes),
        tuple(int(value) for value in tap.strides),
    )


def test_plan_gemm_fill_drain_tasks_matches_current_runtime_taps():
    plan = plan_gemm_fill_drain_tasks(
        M=512,
        K=768,
        N=768,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        b_col_maj=False,
        c_col_maj=False,
        separate_c_tiles=False,
    )

    runtime_A_taps, runtime_B_taps, runtime_C_taps = gemm_design(
        dev="npu2",
        M=512,
        K=768,
        N=768,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        dtype_in_str="bf16",
        dtype_out_str="bf16",
        b_col_maj=False,
        c_col_maj=False,
        use_scalar=False,
        emulate_bf16_mmul_with_bfp16=True,
        prio_accuracy=False,
        separate_c_tiles=False,
        trace_size=0,
        archive=None,
        generate_taps=True,
    )

    assert [_tap_signature(tap) for tap in runtime_A_taps] == [
        _tap_signature(tap) for tap in plan.A_taps
    ]
    assert [_tap_signature(tap) for tap in runtime_B_taps] == [
        _tap_signature(tap) for tap in plan.B_taps
    ]
    assert [_tap_signature(tap) for tap in runtime_C_taps] == [
        _tap_signature(tap) for tap in plan.C_taps
    ]


def test_plan_gemm_fill_drain_tasks_records_compute_tile_consumers():
    plan = plan_gemm_fill_drain_tasks(
        M=512,
        K=768,
        N=768,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        b_col_maj=False,
        c_col_maj=False,
        separate_c_tiles=False,
    )

    assert plan.a_fills
    assert plan.b_fills
    assert plan.c_drains
    assert plan.a_fills[0].compute_tiles[0] == (0, 0)
    assert plan.b_fills[0].compute_tiles[0] == (0, 0)
    assert plan.c_drains[0].compute_tiles[-1] == (3, 0)
