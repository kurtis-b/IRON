#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.mha_out_proj.design import mha_out_proj_topologies
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.mha_out_proj.reference import generate_golden_reference
from iron.common.test_utils import run_test

DEBUG_MODE = 0


def generate_test_params():
    workloads = [
        (64, 64, 1),
        (512, 64, 1),
        (64, 64, 12),
        (512, 64, 12),
        (64, 64, 16),
        (512, 64, 16),
        (1984, 64, 16),
        (2048, 64, 16),
    ]
    params = []
    for seq_len, head_dim, num_heads in workloads:
        for topology in mha_out_proj_topologies(
            num_heads=num_heads,
            head_dim=head_dim,
        ):
            topology_id = str(topology["topology_id"])
            params.append(
                pytest.param(
                    seq_len,
                    head_dim,
                    num_heads,
                    topology_id,
                    id=(
                        f"mha_out_proj_{num_heads}heads_{seq_len}seq_{head_dim}hdim_"
                        f"{topology_id}"
                    ),
                )
            )

    return params


regular_params = generate_test_params()


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,head_dim,num_heads,topology_id",
    regular_params,
)
def test_mha_out_proj(
    seq_len,
    head_dim,
    num_heads,
    topology_id,
    aie_context,
):
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
        debug=DEBUG_MODE,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id=topology_id,
        debug=DEBUG_MODE,
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
    assert operator.topology_id == topology_id
    assert operator.topology_family == "fused_mha_out_proj"
    if errors:
        print(
            "({} errors out of {} max allowable)".format(
                len(errors["O"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["O"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['O'])} errors (max allowable: {max_acceptable_errors})"
