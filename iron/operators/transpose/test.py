#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


import pytest
from iron.operators.transpose.op import AIETranspose
from iron.operators.transpose.design import _expand_dma_tap_for_bd_limits
from iron.operators.transpose.reference import generate_golden_reference
from iron.common import AIEOperatorConstraintError
from iron.common.test_utils import run_test
from aie.helpers.taplib.tap import TensorAccessPattern


def test_expand_dma_tap_for_bd_limits_splits_outer_rank_dimension():
    tap = TensorAccessPattern(
        (16384, 1024),
        offset=0,
        sizes=[128, 2, 64, 64],
        strides=[65536, 64, 1024, 1],
    )

    planned_taps, was_split = _expand_dma_tap_for_bd_limits(tap)

    assert was_split is True
    assert len(planned_taps) == 128
    assert all(len(planned.sizes) == 3 for planned in planned_taps)
    assert all(planned.sizes == [2, 64, 64] for planned in planned_taps)
    assert planned_taps[0].offset == 0
    assert planned_taps[1].offset == 65536


def test_expand_dma_tap_for_bd_limits_recursively_splits_large_outer_rank():
    tap = TensorAccessPattern(
        (16384, 1024),
        offset=0,
        sizes=[256, 4, 64, 32],
        strides=[65536, 32, 1024, 1],
    )

    planned_taps, was_split = _expand_dma_tap_for_bd_limits(tap)

    assert was_split is True
    assert len(planned_taps) == 256
    assert all(len(planned.sizes) == 3 for planned in planned_taps)
    assert all(planned.sizes == [4, 64, 32] for planned in planned_taps)
    assert planned_taps[0].offset == 0
    assert planned_taps[1].offset == 65536


def test_transpose_rejects_per_column_tiles_that_do_not_cover_column_partition():
    with pytest.raises(
        AIEOperatorConstraintError,
        match="per-column column partition must be divisible by n",
    ):
        AIETranspose(
            M=512,
            N=768,
            num_aie_columns=8,
            num_channels=2,
            m=64,
            n=64,
            s=8,
        )


def generate_test_params(extensive=False):
    params = []
    names = []
    max_aie_columns = 8
    input_lengths = [2048] if not extensive else [64, 2048]
    n_list = [64] if not extensive else [64, 128, 256, 512]
    s_list = [8]
    m = 64
    n = 64

    for M in input_lengths:
        for N in n_list:
            for s in s_list:
                for num_aie_columns in range(1, max_aie_columns + 1):
                    for num_channels in [1, 2]:
                        row_part = M // num_channels
                        col_part = N // num_aie_columns
                        if row_part % m != 0 or col_part % n != 0:
                            continue
                        check_length = (
                            row_part * col_part * num_channels * num_aie_columns
                        )
                        length = M * N
                        if check_length != length:
                            continue
                        names.append(
                            f"transpose_{M}_M_{N}_N_{num_aie_columns}_cols_{num_channels}_channels_{m}_m_{n}_n_{s}_s"
                        )
                        params.append((M, N, num_aie_columns, num_channels, m, n, s))

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
)
@pytest.mark.parametrize("M,N,aie_columns,channels,m,n,s", all_params)
def test_transpose(M, N, aie_columns, channels, m, n, s, aie_context):
    golden_ref = generate_golden_reference(rows=M, cols=N)

    operator = AIETranspose(
        M=M,
        N=N,
        num_aie_columns=aie_columns,
        num_channels=channels,
        m=m,
        n=n,
        s=s,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
