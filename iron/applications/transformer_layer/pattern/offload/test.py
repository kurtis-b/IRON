# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from iron.applications.transformer_layer.pattern.offload.op import (
    AIETransformerOffload,
    resolve_offload_operator_config,
)
from iron.applications.transformer_layer.pattern.offload.reference import (
    generate_golden_reference,
)
from iron.common.test_utils import run_test


def _configure_weights(operator, golden_ref):
    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]


def test_offload_runtime_sync_metadata():
    operator = AIETransformerOffload(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
    )

    assert operator.host_output_buffer_names == ("output",)


def test_offload_short_seq_uses_smaller_shared_tile_m():
    seq64 = resolve_offload_operator_config(64, 768, 3072, 12)
    seq128 = resolve_offload_operator_config(128, 768, 3072, 12)
    seq256 = resolve_offload_operator_config(256, 768, 3072, 12)

    assert {config["tile_m"] for config in seq64.values()} == {16}
    assert {config["tile_k"] for config in seq64.values()} == {64}
    assert {config["tile_n"] for config in seq64.values()} == {64}

    assert {config["tile_m"] for config in seq128.values()} == {32}
    assert {config["tile_k"] for config in seq128.values()} == {64}
    assert {config["tile_n"] for config in seq128.values()} == {64}

    assert {config["tile_m"] for config in seq256.values()} == {64}
    assert {config["tile_k"] for config in seq256.values()} == {64}
    assert {config["tile_n"] for config in seq256.values()} == {64}


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_heads",
    (
        (512, 2048, 8),
        (768, 3072, 12),
        (1024, 4096, 16),
    ),
)
@pytest.mark.parametrize("seq_len", (64, 512, 2048, 16384))
def test_offload_shared_supertile_uses_minimum_k_and_n_workloads(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
):
    resolved = resolve_offload_operator_config(
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
    )

    assert {config["tile_k"] for config in resolved.values()} == {64}
    assert {config["tile_n"] for config in resolved.values()} == {64}


def test_offload_rejects_per_operator_supertile_overrides():
    with pytest.raises(
        ValueError,
        match="shared tile_m/tile_k/tile_n",
    ):
        resolve_offload_operator_config(
            16384,
            768,
            3072,
            12,
            operator_config={"q_proj": {"tile_k": 96, "tile_n": 48}},
        )


def test_offload_artifacts_share_one_xclbin(aie_context):
    operator = AIETransformerOffload(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        context=aie_context,
    )

    operator.set_up_artifacts()

    inst_paths = {
        str(getattr(operator, f"{name}_insts").path)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_scores",
            "attn_output",
            "output_proj",
            "up_proj",
            "down_proj",
        )
    }
    assert len(inst_paths) == 8
    assert operator.shared_gemm_xclbin is not None
    assert operator.shared_gemm_xclbin.kernel_name == "offload_gemm"


def test_offload_long_seq_reuses_query_block_insts(aie_context):
    operator = AIETransformerOffload(
        seq_len=16384,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        context=aie_context,
    )

    operator.set_up_artifacts()

    assert operator.query_block_size < operator.seq_len
    assert operator.query_block_count == operator.seq_len // operator.query_block_size
    assert operator.operator_config["attn_scores"]["M"] == operator.query_block_size
    assert operator.operator_config["attn_output"]["M"] == operator.query_block_size


@pytest.mark.metrics(
    Latency=r"Latency \\(us\\): (?P<value>[\\d\\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\\d\\.e\\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "workload_variant,seq_len,hidden_size,intermediate_size,num_heads",
    (
        ("encoder_bert", 512, 768, 3072, 12),
        ("decoder_gpt2", 512, 768, 3072, 12),
        ("encoder_bert", 512, 1024, 4096, 16),
    ),
)
def test_transformer_layer_offload_smoke(
    workload_variant,
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    aie_context,
):
    golden_ref = generate_golden_reference(
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        workload_variant=workload_variant,
    )

    operator = AIETransformerOffload(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        workload_variant=workload_variant,
        context=aie_context,
    )
    _configure_weights(operator, golden_ref)

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"input": golden_ref["input"]},
        {"output": golden_ref["output"]},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    total_macs = seq_len * hidden_size * hidden_size * 4
    total_macs += seq_len * (hidden_size // num_heads) * seq_len * num_heads * 2
    total_macs += seq_len * hidden_size * intermediate_size * 2
    total_ops = total_macs * 2
    gflops = total_ops / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * hidden_size * error_threshold)
    output_errors = len(errors.get("output", []))
    assert output_errors <= max_acceptable_errors
