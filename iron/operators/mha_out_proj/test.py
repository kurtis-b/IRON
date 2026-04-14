#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.mha_out_proj.op import AIEMHAOutProj
from operators.mha_out_proj.reference import generate_golden_reference
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        base_params_by_causal = {
            False: [
                # seq_len, head_dim, num_heads, q_seq_tile, kv_seq_tile, emb_tile, parallel_seq, parallel_heads, o_proj_acc_depth
                (64, 64, 3, 64, 64, 64, 1, 1, 1),
                (64, 64, 12, 64, 64, 64, 1, 1, 1),
                (64, 64, 12, 64, 64, 64, 1, 1, 6),
                (128, 64, 12, 64, 64, 64, 1, 1, 1),
                (128, 64, 12, 64, 64, 64, 1, 1, 2),
                (128, 64, 12, 64, 64, 64, 2, 1, 1),
                (256, 64, 12, 64, 64, 64, 1, 1, 1),
                (256, 64, 12, 64, 64, 64, 1, 1, 4),
                (512, 64, 12, 64, 64, 64, 1, 1, 6),
                (512, 64, 12, 64, 64, 64, 1, 6, 1),
                (512, 64, 12, 64, 64, 64, 2, 1, 6),
                (512, 64, 12, 64, 64, 64, 8, 1, 1),
                (512, 64, 12, 64, 64, 64, 8, 1, 6),
                (512, 64, 12, 32, 64, 96, 8, 1, 8),
                # TinyBERT-6L-512
                (512, 64, 8, 64, 64, 64, 8, 1, 8),
                (512, 64, 8, 64, 64, 64, 8, 1, 8),
            ],
            True: [
                (64, 64, 3, 64, 64, 64, 1, 1, 1),
                (64, 64, 12, 64, 64, 96, 1, 6, 8),
                (128, 64, 12, 64, 64, 64, 1, 1, 1),
                (128, 64, 12, 64, 64, 64, 2, 1, 1),
                (128, 64, 16, 32, 32, 128, 2, 1, 8),
                (512, 64, 12, 64, 64, 64, 1, 1, 6),
                (512, 64, 12, 64, 64, 64, 1, 6, 1),
                (512, 64, 12, 64, 64, 64, 2, 1, 6),
                (512, 64, 12, 64, 64, 64, 8, 1, 1),
                # TinyBERT-6L-512
                (512, 64, 8, 64, 64, 64, 8, 1, 8),
            ],
        }
        params = []
        for is_causal in (False, True):
            for base_param in base_params_by_causal[is_causal]:
                params.append((*base_param, is_causal))
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
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_seq,
        parallel_heads,
        o_proj_acc_depth,
        is_causal,
    ) in params:
        causal_suffix = "causal" if is_causal else "noncausal"
        names.append(
            f"mha_out_proj_{seq_len}seq_{head_dim}hdim_{num_heads}heads_"
            f"{q_seq_tile}q_{kv_seq_tile}kv_{emb_tile}e_ps{parallel_seq}_"
            f"ph{parallel_heads}_acc{o_proj_acc_depth}_{causal_suffix}"
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
    "seq_len,head_dim,num_heads,q_seq_tile,kv_seq_tile,emb_tile,parallel_seq,parallel_heads,o_proj_acc_depth,is_causal",
    all_params,
)
def test_mha_out_proj(
    seq_len,
    head_dim,
    num_heads,
    q_seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_seq,
    parallel_heads,
    o_proj_acc_depth,
    is_causal,
    aie_context,
):
    logging.debug(
        "Testing MHA out projection with seq_len=%s head_dim=%s num_heads=%s "
        "q_seq_tile=%s kv_seq_tile=%s emb_tile=%s parallel_seq=%s "
        "parallel_heads=%s o_proj_acc_depth=%s is_causal=%s",
        seq_len,
        head_dim,
        num_heads,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_seq,
        parallel_heads,
        o_proj_acc_depth,
        is_causal,
    )

    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        is_causal=is_causal,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        q_seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=o_proj_acc_depth,
        is_causal=is_causal,
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


@pytest.mark.parametrize("o_proj_acc_depth", [1, 2, 4])
def test_mha_out_proj_causal_low_acc_reuses_identical_col_groups(
    o_proj_acc_depth,
    aie_context,
):
    seq_len = 512
    head_dim = 64
    num_heads = 12
    q_seq_tile = 32
    kv_seq_tile = 32
    emb_tile = 96
    parallel_seq = 8
    parallel_heads = 1
    hidden = num_heads * head_dim

    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        is_causal=True,
    )
    repeated_block = torch.zeros((hidden, emb_tile), dtype=torch.bfloat16)
    for i in range(emb_tile):
        repeated_block[i, i] = 1
    repeated_w_o = torch.cat([repeated_block] * (hidden // emb_tile), dim=1)

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        q_seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        o_proj_acc_depth=o_proj_acc_depth,
        is_causal=True,
        context=aie_context,
    )

    operator.context.compile_all()
    operator.context.prepare_runtime()
    output = operator(
        golden_ref["input_q"],
        golden_ref["input_k"],
        golden_ref["input_v"],
        repeated_w_o,
    ).to(torch.float32)

    base_chunk = output[:, :emb_tile]
    for start in range(emb_tile, hidden, emb_tile):
        chunk = output[:, start : start + emb_tile]
        diff = (chunk - base_chunk).abs()
        assert int((diff > 0.01).sum().item()) == 0, (
            f"causal repeated col-group mismatch at start={start} "
            f"for acc={o_proj_acc_depth}, max_abs={float(diff.max().item())}"
        )
