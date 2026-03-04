#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.reference import generate_golden_reference
from operators.common.test_utils import run_test

DEBUG_MODE = -1


def generate_test_params(extensive=False):
    # Unified tuple:
    # seq_len, d, heads, intermediate_size, q_seq_tile, kv_seq_tile,
    # emb_tile, parallel_heads, parallel_ffn, proj_acc_depth
    regular_params = [
        (64, 64, 3, 768, 32, 64, 96, 1, 1, 2),
        (64, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
        (128, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
        (512, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
        (2048, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
        (64, 64, 3, 768, 32, 64, 96, 3, 2, 2),
        (64, 64, 12, 3072, 32, 64, 96, 6, 3, 8),
        (512, 64, 12, 3072, 32, 64, 96, 6, 3, 8),
        (2048, 64, 12, 3072, 32, 64, 96, 6, 3, 8),
        (1024, 64, 12, 3072, 32, 64, 96, 6, 3, 8),
        (512, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
        (1024, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
        (2048, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
    ]
    extensive_params = []

    params = extensive_params if extensive else regular_params
    names = []
    for (
        seq_len,
        d,
        heads,
        intermediate_size,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        parallel_ffn,
        proj_acc_depth,
    ) in params:
        names.append(
            f"encoder_{seq_len}seq_{d}hdim_{heads}heads_{intermediate_size}ffn_{q_seq_tile}qseqtile_{kv_seq_tile}kvtile_"
            f"{emb_tile}embtile_{parallel_heads}pheads_{parallel_ffn}pffn_{proj_acc_depth}pacc"
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
    "seq_len,d,heads,intermediate_size,q_seq_tile,kv_seq_tile,emb_tile,parallel_heads,parallel_ffn,proj_acc_depth",
    all_params,
)
def test_encoder_pipeline(
    seq_len,
    d,
    heads,
    intermediate_size,
    q_seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_heads,
    parallel_ffn,
    proj_acc_depth,
    aie_context,
):
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=d,
        heads=heads,
        intermediate_size=intermediate_size,
        seed=42,
        debug=DEBUG_MODE,
    )

    operator = AIEEncoderPipeline(
        seq_len=seq_len,
        d=d,
        num_heads=heads,
        seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=proj_acc_depth,
        down_proj_depth=proj_acc_depth,
        ffn_intermediate_size=intermediate_size,
        nB_tiles_distributed=parallel_ffn,
        debug=DEBUG_MODE,
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
