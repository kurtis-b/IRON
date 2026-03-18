#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.common.test_utils import run_test
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.elementwise_mul.reference import generate_golden_reference


def generate_test_params(extensive=False):
    max_aie_columns = 8
    num_channels = 2
    input_lengths = [2048] if not extensive else [1024, 4096, 8192]

    params = []
    names = []
    for input_length in input_lengths:
        for num_aie_columns in range(1, max_aie_columns + 1):
            tile_size = input_length // num_aie_columns
            if tile_size > 4096:
                tile_size = 4096
            if tile_size * num_aie_columns != input_length:
                continue
            names.append(
                f"eltwise_mul_{num_aie_columns}_cols_{num_channels}_channels_{input_length}_tile_{tile_size}"
            )
            params.append(
                (input_length, num_aie_columns, num_channels, tile_size, None)
            )

    # The scalar-broadcast path has a stricter divisibility contract.
    # Keep the old BERT-sized surfaces that are known-valid.
    broadcast_cases = [
        (3145728, 8, num_channels, 4096, 0.125),
        (3145728, 4, num_channels, 4096, 0.125),
        (3145728, 2, num_channels, 4096, 0.125),
    ]
    for case in broadcast_cases:
        input_length, num_aie_columns, num_channels_case, tile_size, scalar = case
        names.append(
            f"eltwise_mul_{num_aie_columns}_cols_{num_channels_case}_channels_{input_length}_tile_{tile_size}_{scalar}_broadcast"
        )
        params.append(case)
    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)
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
@pytest.mark.parametrize(
    "input_length,num_aie_columns,num_channels,tile_size,scalar_broadcast",
    all_params,
)
def test_elementwise_mul(
    input_length,
    num_aie_columns,
    num_channels,
    tile_size,
    scalar_broadcast,
    aie_context,
):
    golden_ref = generate_golden_reference(
        input_length=input_length, scalar_broadcast=scalar_broadcast
    )

    operator = AIEElementwiseMul(
        size=input_length,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        scalar_broadcast=scalar_broadcast,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["A"]}
    if scalar_broadcast is None:
        input_buffers["input2"] = golden_ref["B"]
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
