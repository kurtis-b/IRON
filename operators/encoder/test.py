# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.encoder.op import AIEBERTEncoder
from operators.encoder.reference import generate_golden_reference
from operators.common.test_utils import run_test, verify_buffer


def generate_test_params():
    params = [(512, 768, 3072, 12)]
    names = [f"bert_encoder_{seq}x{emb}x{ffn}x{h}" for seq, emb, ffn, h in params]
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
@pytest.mark.parametrize("seq_len,embedding_dim,ffn_dim,num_heads", all_params)
def test_bert_encoder(seq_len, embedding_dim, ffn_dim, num_heads, aie_context):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIEBERTEncoder(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        context=aie_context,
    )

    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ln1_weight = golden_ref["weights"]["ln1_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]
    operator.ln2_weight = golden_ref["weights"]["ln2_weight"]

    input_buffers = {
        "input": golden_ref["input"],
    }
    output_buffers = {
        "output": golden_ref["output"],
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
    total_macs = seq_len * embedding_dim * embedding_dim * 4
    total_macs += seq_len * (embedding_dim // num_heads) * seq_len * num_heads * 2
    total_macs += seq_len * embedding_dim * ffn_dim * 2
    total_ops = total_macs * 2  # 2 operations per MAC
    gflops = total_ops / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    # print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * embedding_dim * error_threshold)

    assert (
        len(errors["output"]) <= max_acceptable_errors
    ), f"Test failed with {len(errors['output'])} errors (max allowable: {max_acceptable_errors})"
