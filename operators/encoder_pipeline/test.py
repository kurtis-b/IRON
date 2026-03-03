#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.constants import (
    DEBUG_FULL,
    ADDNORM_DEBUG_DISABLED,
)
from operators.encoder_pipeline.reference import generate_golden_reference
from operators.common.test_utils import run_test

DEBUG_MODE = DEBUG_FULL
ADDNORM_DEBUG_MODE = ADDNORM_DEBUG_DISABLED
ADDNORM1_DEBUG_MODE = None
ADDNORM2_DEBUG_MODE = None


def generate_test_params(extensive=False):
    # Unified tuple:
    # heads, seq_len, d, seq_tile, kv_seq_tile, emb_tile,
    # parallel_heads, o_proj_acc_depth, down_proj_depth, intermediate_size
    regular_params = [
        (3, 64, 64, 32, 64, 96, 1, 2, 2, 768),
        (12, 64, 64, 32, 64, 96, 1, 8, 8, 3072),
        (12, 128, 64, 32, 64, 96, 1, 8, 8, 3072),
        (12, 512, 64, 32, 64, 96, 1, 8, 8, 3072),
        (12, 2048, 64, 32, 64, 96, 1, 8, 8, 3072),
        (3, 64, 64, 32, 64, 96, 3, 2, 2, 768),
        (12, 64, 64, 32, 64, 96, 6, 8, 8, 3072),
        (12, 512, 64, 32, 64, 96, 6, 8, 8, 3072),
        (12, 2048, 64, 32, 64, 96, 6, 8, 8, 3072),
        (12, 1024, 64, 32, 64, 96, 1, 8, 8, 3072),
        (16, 512, 64, 32, 64, 128, 1, 8, 8, 4096),
        (16, 1024, 64, 32, 64, 128, 1, 8, 8, 4096),
        (16, 2048, 64, 32, 64, 128, 1, 8, 8, 4096),
    ]
    extensive_params = []

    params = extensive_params if extensive else regular_params
    names = []
    for (
        heads,
        seq_len,
        d,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        o_proj_acc_depth,
        down_proj_depth,
        intermediate_size,
    ) in params:
        names.append(
            f"encoder_{heads}heads_{seq_len}seq_{d}hdim_{seq_tile}seqtile_{kv_seq_tile}kvtile_"
            f"{emb_tile}embtile_{parallel_heads}pheads_{o_proj_acc_depth}acc_{down_proj_depth}dproj_"
            f"{intermediate_size}ffn"
        )
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
    "heads,seq_len,d,seq_tile,kv_seq_tile,emb_tile,parallel_heads,o_proj_acc_depth,down_proj_depth,intermediate_size",
    all_params,
)
def test_encoder_pipeline(
    heads,
    seq_len,
    d,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_heads,
    o_proj_acc_depth,
    down_proj_depth,
    intermediate_size,
    aie_context,
):
    golden_ref = generate_golden_reference(
        heads=heads,
        seq_len=seq_len,
        d=d,
        intermediate_size=intermediate_size,
        seed=42,
        debug=DEBUG_MODE,
        addnorm_debug_mode=ADDNORM_DEBUG_MODE,
        addnorm1_debug_mode=ADDNORM1_DEBUG_MODE,
        addnorm2_debug_mode=ADDNORM2_DEBUG_MODE,
    )

    operator = AIEEncoderPipeline(
        num_heads=heads,
        seq_len=seq_len,
        d=d,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=o_proj_acc_depth,
        down_proj_depth=down_proj_depth,
        ffn_intermediate_size=intermediate_size,
        nB_tiles_distributed=1,
        debug=DEBUG_MODE,
        addnorm_debug_mode=ADDNORM_DEBUG_MODE,
        addnorm1_debug_mode=ADDNORM1_DEBUG_MODE,
        addnorm2_debug_mode=ADDNORM2_DEBUG_MODE,
        ln1_weight=golden_ref["ln1_weight"],
        ln2_weight=golden_ref["ln2_weight"],
        context=aie_context,
    )

    input_buffers = {
        "QKV": golden_ref["QKV"].flatten(),
        "W_O": golden_ref["W_O"].flatten(),
        "OR": golden_ref["OR"].flatten(),
        "B_Up": golden_ref["B_Up"].flatten(),
        "B_Down": golden_ref["B_Down"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    error_threshold = 0.005
    max_acceptable_errors = int(seq_len * d * heads * error_threshold)
    if errors:
        print(
            "({} errors out of {} max allowable)".format(
                len(errors["O"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["O"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['O'])} errors (max allowable: {max_acceptable_errors})"
