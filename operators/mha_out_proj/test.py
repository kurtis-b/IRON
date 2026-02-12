#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.mha_out_proj.op import AIEMHAOutProj
from operators.mha_out_proj.reference import generate_golden_reference
from operators.common.test_utils import run_test


def generate_test_params(extensive=False):
    params = [
        # seq_len, head_dim, num_heads, seq_tile, emb_tile, parallel_heads, o_proj_acc_depth
        (64, 64, 3, 64, 64, 1, 1),
        # (512, 64, 12, 64, 96, 1, 1),
        # (512, 64, 12, 64, 96, 2, 1),
        # (512, 64, 12, 64, 96, 1, 2),
        # (512, 64, 12, 64, 96, 4, 1),
        # (512, 64, 12, 64, 96, 1, 4),
        # (512, 64, 12, 64, 96, 8, 1),
        # (512, 64, 12, 64, 96, 1, 8),
    ]
    extensive_params = []

    if extensive:
        params = extensive_params

    names = []
    for (
        seq_len,
        head_dim,
        num_heads,
        seq_tile,
        emb_tile,
        parallel_heads,
        o_proj_acc_depth,
    ) in params:
        name = f"mha_{num_heads}heads_{seq_len}seq_{head_dim}hdim_{seq_tile}seqtile_{emb_tile}embtile_{parallel_heads}heads_{o_proj_acc_depth}acc"
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
    "seq_len,head_dim,num_heads,seq_tile,emb_tile,parallel_heads,o_proj_acc_depth",
    all_params,
)
def test_mha(
    seq_len,
    head_dim,
    num_heads,
    seq_tile,
    emb_tile,
    parallel_heads,
    o_proj_acc_depth,
    aie_context,
):
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        seq_tile=seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=o_proj_acc_depth,
        context=aie_context,
    )

    input_buffers = {
        "Q": golden_ref["Q"].flatten(),
        "K": golden_ref["K"].flatten(),
        "V": golden_ref["V"].flatten(),
        "W_O": golden_ref["W_O"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
    )

    error_threshold = 0.005
    max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    print(
        "({} errors out of {} max allowable)".format(
            len(errors["O"]), max_acceptable_errors
        )
    )

    assert (
        len(errors["O"]) <= max_acceptable_errors
    ), f"Test failed with {len(errors['O'])} errors (max allowable: {max_acceptable_errors})"
