#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path


from iron.operators.gelu.op import AIEGELU
from iron.operators.gelu.reference import generate_golden_reference
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    max_aie_columns = 8
    num_channels_choices = [1, 2]
    input_lengths = [2048] if not extensive else [1024, 4096, 8192]

    params = []
    names = []
    for input_length in input_lengths:
        for num_aie_columns in range(1, max_aie_columns + 1):
            for num_channels in num_channels_choices:
                total_cores = num_aie_columns * num_channels
                tile_size = input_length // total_cores
                if tile_size > 8192:
                    tile_size = 8192
                check_length = tile_size * total_cores
                if check_length == input_length:
                    names.append(
                        f"gelu_{num_aie_columns}_cols_{num_channels}_channels_{input_length}_tile_{tile_size}"
                    )
                    params.append(
                        (input_length, num_aie_columns, num_channels, tile_size)
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


def generate_test_params_bert(extensive=False):
    params = []
    names = []

    params.extend(
        [
            (1572864, 8, 2, 4096),
            (1572864, 4, 2, 4096),
            (1572864, 2, 2, 4096),
        ]
    )
    names.extend(
        [
            f"gelu_8_cols_2_channels_1572864_tile_4096",
            f"gelu_4_cols_2_channels_1572864_tile_4096",
            f"gelu_2_cols_2_channels_1572864_tile_4096",
        ]
    )

    return params, names


regular_params_bert, regular_names_bert = generate_test_params_bert(extensive=False)

bert_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params_bert, regular_names_bert)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,num_aie_columns,num_channels,tile_size",
    all_params if not TEST_BERT else bert_params,
)
def test_gelu(input_length, num_aie_columns, num_channels, tile_size, aie_context):
    golden_ref = generate_golden_reference(input_length=input_length)

    operator = AIEGELU(
        size=input_length,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6, warmup_iters=10, timed_iters=100
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
        )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
