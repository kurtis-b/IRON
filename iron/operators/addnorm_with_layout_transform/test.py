#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.addnorm_with_layout_transform.op import AIEAddAndNorm
from iron.operators.addnorm_with_layout_transform.reference import (
    generate_golden_reference,
)
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        params = [
            # M, K, m, k, s, num_aie_columns
            (512, 768, 4, 192, 8, 8),
            (512, 768, 4, 192, 8, 4),
            (512, 768, 4, 192, 8, 2),
            (512, 768, 4, 192, 8, 1),
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
        m,
        k,
        s,
        num_aie_columns,
    ) in params:
        name = f"addnorm_{num_aie_columns}cols_{M}x{K}_tile_{m}x{k}_sub{s}"
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
)
@pytest.mark.parametrize(
    "M,K,m,k,s,num_aie_columns",
    all_params,
)
def test_layer_norm(
    M,
    K,
    m,
    k,
    s,
    num_aie_columns,
    aie_context,
):

    golden_ref = generate_golden_reference(M=M, K=K, m=m, k=k, s=s)

    operator = AIEAddAndNorm(
        M=M,
        K=K,
        m=m,
        k=k,
        s=s,
        num_aie_columns=num_aie_columns,
        weights=golden_ref["weight"],
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["input1"], "input2": golden_ref["input2"]}
    output_buffers = {"output": golden_ref["output"]}

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=0.1,
            abs_tol=0.1,
            warmup_iters=10,
            timed_iters=100,
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.1, abs_tol=0.1
        )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    error_threshold = 0.005
    max_acceptable_errors = int(M * K * error_threshold)

    if errors:
        logging.info(
            "({} errors out of {} max allowable)".format(
                len(errors["output"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["output"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['output'])} errors (max allowable: {max_acceptable_errors})"
