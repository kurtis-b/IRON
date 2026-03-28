#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.addnorm_ffn_addnorm.design import addnorm_ffn_addnorm_design
from iron.operators.addnorm_ffn_addnorm.op import AIEAddNormFFNAddNorm
from iron.operators.addnorm_ffn_addnorm.reference import generate_golden_reference


def generate_test_params():
    workloads = [
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
    ]
    params = [
        (
            seq_len,
            hidden_size,
            intermediate_size,
            str(
                addnorm_ffn_addnorm_design(
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )["topology_id"]
            ),
        )
        for seq_len, hidden_size, intermediate_size in workloads
    ]
    names = [
        f"block3_{seq_len}x{hidden_size}x{intermediate_size}_{topology_id}"
        for seq_len, hidden_size, intermediate_size, topology_id in params
    ]
    return params, names


regular_params, regular_names = generate_test_params()


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id",
    [
        pytest.param(*params, id=name)
        for params, name in zip(regular_params, regular_names)
    ],
)
def test_block3_topologies_construct_and_match_reference_contract(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    aie_context,
):
    golden = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=7,
    )

    operator = AIEAddNormFFNAddNorm(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=topology_id,
        context=aie_context,
    )

    assert operator.topology_id == topology_id
    assert operator.topology_family == "pipelined_addnorm_ffn_addnorm"
    assert golden["hidden_states"].shape == (seq_len, hidden_size)
    assert golden["residual"].shape == (seq_len, hidden_size)
    assert golden["ffn_up_weight"].shape == (hidden_size, intermediate_size)
    assert golden["ffn_down_weight"].shape == (intermediate_size, hidden_size)
    assert golden["output"].shape == (seq_len, hidden_size)
