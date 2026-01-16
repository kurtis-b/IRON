# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.test_batched_attn_score.op import AIEAttnScores
from operators.test_batched_attn_score.reference import generate_golden_reference
from operators.common.test_utils import run_test, verify_buffer


def generate_test_params():
    params = [(512, 768, 12, False)]
    names = [
        f"attn_scores_{seq}x{emb}x{h}x{use_sep_gemms}"
        for seq, emb, h, use_sep_gemms in params
    ]
    return params, names


regular_params, regular_names = generate_test_params()

all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("seq_len,embedding_dim,num_heads,use_sep_gemms", all_params)
def test_bert_encoder(seq_len, embedding_dim, num_heads, use_sep_gemms, aie_context):
    golden_ref = generate_golden_reference(
        seq_len, embedding_dim, num_heads, use_sep_gemms=use_sep_gemms
    )

    operator = AIEAttnScores(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        num_heads=num_heads,
        use_sep_gemms=use_sep_gemms,
        context=aie_context,
    )

    if use_sep_gemms:
        input_buffers = {
            **{f"q_{i}": golden_ref[f"q_{i}"] for i in range(num_heads)},
            **{f"k_{i}": golden_ref[f"k_{i}"] for i in range(num_heads)},
        }
        output_buffers = {
            **{
                f"attn_scores_{i}": golden_ref[f"attn_scores_{i}"]
                for i in range(num_heads)
            },
        }
    else:
        input_buffers = {
            "q": golden_ref["q"],
            "k": golden_ref["k"],
        }
        output_buffers = {
            "attn_scores": golden_ref["attn_scores"],
        }
    intermediate_buffers = {}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        intermediate_buffers,
        rel_tol=0.05,
        abs_tol=0.5,
    )

    # Use batch_size of C for total operations
    total_macs = seq_len * (embedding_dim // num_heads) * seq_len * num_heads
    total_ops = total_macs * 2  # 2 operations per MAC
    gflops = total_ops / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    # print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * embedding_dim * error_threshold)

    if use_sep_gemms:
        total_errors = 0
        for i in range(num_heads):
            buf_name = f"attn_scores_{i}"
            total_errors += len(errors[buf_name])
        assert (
            total_errors <= max_acceptable_errors
        ), f"Test failed with {total_errors} errors (max allowable: {max_acceptable_errors})"
    else:
        assert (
            len(errors["attn_scores"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['attn_scores'])} errors (max allowable: {max_acceptable_errors})"
