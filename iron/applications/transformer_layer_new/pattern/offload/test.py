# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from iron.applications.transformer_layer_new.pattern.offload.op import (
    AIETransformerOffload,
)
from iron.applications.transformer_layer_new.pattern.reference import (
    generate_golden_reference,
)


def generate_test_params(extensive=False):
    params = [
        # seq_len,embedding_dim,ffn_dim,num_heads
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
        (512, 768, 3072, 12),
    ]
    extensive_params = []

    if extensive:
        params = extensive_params

    names = []
    for seq_len, embedding_dim, ffn_dim, num_heads in params:
        names.append(f"transformer_{seq_len}x{embedding_dim}x{ffn_dim}x{num_heads}")

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
@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads",
    all_params,
)
def test_transformer_layer(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
    aie_context,
):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIETransformerOffload(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        context=aie_context,
    )

    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    operator.prepare_runtime()

    operator.forward(golden_ref["input"])

    start = time.perf_counter()
    output = operator.forward(golden_ref["input"])
    elapsed = time.perf_counter() - start
    latency_us = elapsed * 1e6

    expected = golden_ref["output"]
    mismatches = ~torch.isclose(
        output.float(),
        expected.float(),
        rtol=0.05,
        atol=0.5,
    )
    mismatch_indices = torch.nonzero(mismatches.reshape(-1), as_tuple=False).flatten()
    for idx in mismatch_indices[:10].tolist():
        print(
            f"Mismatch in output[{idx}]: expected "
            f"{float(expected.reshape(-1)[idx]):.6f}, got {float(output.reshape(-1)[idx]):.6f}"
        )

    total_macs = seq_len * embedding_dim * embedding_dim * 4
    total_macs += seq_len * (embedding_dim // num_heads) * seq_len * num_heads * 2
    total_macs += seq_len * embedding_dim * ffn_dim * 2
    total_ops = total_macs * 2
    gflops = total_ops / elapsed / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    max_acceptable_errors = int(seq_len * embedding_dim * 0.05)
    output_errors = len(mismatch_indices)
    assert (
        output_errors <= max_acceptable_errors
    ), f"Test failed with {output_errors} errors (max allowable: {max_acceptable_errors})"


def test_offload_runtime_lifecycle_uses_non_runlist_context_and_registers_only_children(
    aie_context,
):
    golden_ref = generate_golden_reference(64, 768, 3072, 12)
    operator = AIETransformerOffload(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        context=aie_context,
    )

    assert operator.context.use_runlist is False
    assert operator not in operator.context.operators
    assert len(operator.context.operators) == len(operator.gemm_ops)
