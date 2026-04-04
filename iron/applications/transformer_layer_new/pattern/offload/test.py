# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import pytest
import torch

from iron.applications.transformer_layer_new.pattern.offload.op import (
    AIETransformerOffload,
)
from iron.applications.transformer_layer_new.pattern.reference import (
    derive_offload_inputs,
    generate_golden_reference,
)


@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads",
    [
        pytest.param(64, 768, 3072, 12, id="offload_64x768x3072x12"),
    ],
)
def test_offload_pattern_matches_reference(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIETransformerOffload(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
    )
    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    operator.prepare_runtime()

    start = time.perf_counter()
    output = operator.forward(golden_ref["input"])
    latency_us = (time.perf_counter() - start) * 1e6
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
    print(f"\nLatency (us): {latency_us:.1f}\n")

    max_acceptable_errors = int(seq_len * embedding_dim * 0.05)
    assert len(mismatch_indices) <= max_acceptable_errors


def test_offload_forward_wrapper_matches_precomputed():
    golden_ref = generate_golden_reference(64, 768, 3072, 12)
    inputs = derive_offload_inputs(golden_ref, num_heads=12)

    precomputed_operator = AIETransformerOffload(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
    )
    wrapper_operator = AIETransformerOffload(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
    )

    for operator in (precomputed_operator, wrapper_operator):
        operator.q_weight = golden_ref["weights"]["q_weight"]
        operator.k_weight = golden_ref["weights"]["k_weight"]
        operator.v_weight = golden_ref["weights"]["v_weight"]
        operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
        operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
        operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]
        operator.prepare_runtime()

    precomputed_output = precomputed_operator.forward_precomputed(**inputs)
    wrapper_output = wrapper_operator.forward(golden_ref["input"])

    assert torch.allclose(
        precomputed_output.float(),
        wrapper_output.float(),
        rtol=0.05,
        atol=0.5,
    )


def test_offload_runtime_lifecycle_uses_single_non_runlist_context():
    golden_ref = generate_golden_reference(64, 768, 3072, 12)
    operator = AIETransformerOffload(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
    )

    assert operator.context.use_runlist is False
    assert operator not in operator.context.operators
    assert tuple(name for name, _ in operator.gemm_ops[:3]) == (
        "q_proj",
        "k_proj",
        "v_proj",
    )
    assert len(operator.gemm_ops) == 8
    assert len(operator.context.operators) == len(operator.gemm_ops)


def test_offload_prepare_runtime_is_idempotent(monkeypatch):
    golden_ref = generate_golden_reference(64, 768, 3072, 12)
    operator = AIETransformerOffload(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
    )
    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    operator.prepare_runtime()

    compile_calls = []
    prepare_calls = []

    original_compile = operator.context.compile_all
    original_prepare = operator.context.prepare_runtime

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return original_compile(*args, **kwargs)

    def wrapped_prepare(*args, **kwargs):
        prepare_calls.append(kwargs.get("describe_runtime", True))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(operator.context, "compile_all", wrapped_compile)
    monkeypatch.setattr(operator.context, "prepare_runtime", wrapped_prepare)

    operator.prepare_runtime()

    assert compile_calls == []
    assert prepare_calls == []


def test_offload_supports_seq_len_16384_via_query_blocking():
    operator = AIETransformerOffload(
        seq_len=16384,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
    )

    assert operator.query_block_size == 256
    assert operator.uses_query_blocking is True
    assert operator.attn_scores.partition_N == 4
    assert operator.q_proj.M == 256
    assert operator.k_proj.M == 256
    assert operator.v_proj.M == 256
