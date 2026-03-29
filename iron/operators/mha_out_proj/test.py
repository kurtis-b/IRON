#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.mha_out_proj.design import (
    mha_out_proj_theoretical_topologies,
    mha_out_proj_topologies,
)
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


def test_theoretical_block2_topologies_cover_supported_surface():
    for seq_len, num_heads in ((64, 1), (512, 12), (2048, 16)):
        supported_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_topologies(
                num_heads=num_heads,
                head_dim=64,
            )
        }
        theoretical_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_theoretical_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=64,
            )
        }
        assert supported_ids <= theoretical_ids


def test_theoretical_block2_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in mha_out_proj_theoretical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    }
    assert "q32_kv64_e128_ps1_ph1_acc1" in topology_ids
    assert "q32_kv64_e128_ps1_ph2_acc1" in topology_ids
    assert "q32_kv64_e128_ps1_ph1_acc2" in topology_ids
    assert "q64_kv64_e128_ps1_ph1_acc1" in topology_ids


def test_theoretical_block2_topologies_are_unique_and_contract_valid():
    topologies = mha_out_proj_theoretical_topologies(
        seq_len=2048,
        num_heads=16,
        head_dim=64,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_heads = int(topology["parallel_heads"])
        q_seq_tile = int(topology["q_seq_tile"])
        kv_seq_tile = int(topology["kv_seq_tile"])
        emb_tile = int(topology["emb_tile"])
        o_proj_acc_depth = int(topology["o_proj_acc_depth"])

        assert parallel_seq in (1, 2, 4, 6, 8)
        assert 2048 % (parallel_seq * q_seq_tile) == 0
        assert 2048 % kv_seq_tile == 0
        assert 16 % parallel_heads == 0
        assert parallel_seq * parallel_heads <= 8
        assert 1024 % (emb_tile * o_proj_acc_depth) == 0
