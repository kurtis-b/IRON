#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.mha_out_proj.op import AIEMHAOutProj
from operators.mha_out_proj.reference import generate_golden_reference
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        params = [
            # seq_len, head_dim, num_heads, parallel_seq, q_seq_tile, kv_seq_tile, emb_tile, parallel_heads, o_proj_acc_depth
            (64, 64, 3, 1, 32, 64, 64, 1, 1),
            (128, 64, 12, 1, 32, 64, 64, 1, 1),
            (512, 64, 12, 1, 32, 64, 64, 1, 6),
            (512, 64, 12, 1, 32, 64, 64, 6, 1),
            (128, 64, 12, 4, 32, 64, 64, 1, 1),
            (512, 64, 12, 2, 32, 64, 64, 1, 6),
            (512, 64, 12, 8, 32, 64, 64, 1, 1),
        ]
        extensive_params = []
    else:
        params = []
        extensive_params = []

    if extensive:
        params = extensive_params

    names = []
    for (
        seq_len,
        head_dim,
        num_heads,
        parallel_seq,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        o_proj_acc_depth,
    ) in params:
        names.append(
            f"mha_out_proj_{seq_len}seq_{head_dim}hdim_{num_heads}heads_"
            f"ps{parallel_seq}_{q_seq_tile}q_{kv_seq_tile}kv_{emb_tile}e_"
            f"ph{parallel_heads}_acc{o_proj_acc_depth}"
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
    "seq_len,head_dim,num_heads,parallel_seq,q_seq_tile,kv_seq_tile,emb_tile,parallel_heads,o_proj_acc_depth",
    all_params,
)
def test_mha_out_proj(
    seq_len,
    head_dim,
    num_heads,
    parallel_seq,
    q_seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_heads,
    o_proj_acc_depth,
    aie_context,
):
    logging.debug(
        "Testing MHA out projection with seq_len=%s head_dim=%s num_heads=%s "
        "parallel_seq=%s q_seq_tile=%s kv_seq_tile=%s emb_tile=%s "
        "parallel_heads=%s o_proj_acc_depth=%s",
        seq_len,
        head_dim,
        num_heads,
        parallel_seq,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        o_proj_acc_depth,
    )

    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        parallel_seq=parallel_seq,
        q_seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=o_proj_acc_depth,
        context=aie_context,
    )

    input_buffers = {
        "Q": golden_ref["input_q"].flatten(),
        "K": golden_ref["input_k"].flatten(),
        "V": golden_ref["input_v"].flatten(),
        "W_O": golden_ref["input_w_o"].flatten(),
    }
    output_buffers = {"O": golden_ref["output"].flatten()}

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=4.0e-2,
            abs_tol=1.5e-1,
            warmup_iters=10,
            timed_iters=100,
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=4.0e-2,
            abs_tol=1.5e-1,
        )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    max_acceptable_errors = int(seq_len * head_dim * num_heads * 0.005)
    output_errors = len(errors.get("O", []))
    if output_errors:
        logging.info(
            "(%s errors out of %s max allowable) for O",
            output_errors,
            max_acceptable_errors,
        )
    assert (
        output_errors <= max_acceptable_errors
    ), f"Test failed for O with {output_errors} errors (max allowable: {max_acceptable_errors})"
