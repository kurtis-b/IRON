# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import csv
from pathlib import Path
from types import SimpleNamespace

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pipeline import (
    validate_npu_parity as structured_validate_npu_parity_module,
)
from iron.applications.transformer_layer.src.pipeline import (
    error_stats as structured_error_stats,
)
from iron.applications.transformer_layer.src.pipeline.validate_npu_parity import (
    PARITY_FIELD_ORDER as structured_parity_field_order,
)
from iron.applications.transformer_layer.src.pipeline.validate_npu_parity import (
    run_parity_cli as structured_run_parity_cli,
)
from iron.applications.transformer_layer.src.pipeline import (
    run_parity_suite as structured_run_parity_suite,
)
from iron.applications.transformer_layer.src.pipeline import (
    validate_pattern_parity as structured_validate_pattern_parity,
)
from iron.applications.transformer_layer.validate_npu_parity import (
    error_stats,
    run_parity_suite,
    validate_pattern_parity,
)


def test_validate_npu_parity_restructure_preserves_legacy_imports_and_helpers():
    assert error_stats is structured_error_stats
    assert validate_pattern_parity is structured_validate_pattern_parity
    assert run_parity_suite is structured_run_parity_suite

    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.5, 1.0], dtype=torch.float32)
    stats = error_stats(reference, candidate)
    assert stats == {"max_abs_diff": 1.0, "mean_abs_diff": 0.75}

    spec = TransformerLayerSpec(seq_len=64)
    rows = run_parity_suite(
        execution_modes=[],
        seq_lens=[spec.seq_len],
        base_spec=spec,
        seed=0,
    )
    assert rows == []


def test_run_parity_cli_forwards_block_topology_overrides(monkeypatch):
    captured = {}

    def fake_run_parity_suite(
        *, execution_modes, seq_lens, base_spec, seed, study_id=None
    ):
        captured["execution_modes"] = execution_modes
        captured["seq_lens"] = seq_lens
        captured["base_spec"] = base_spec
        captured["seed"] = seed
        return []

    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "run_parity_suite",
        fake_run_parity_suite,
    )

    structured_run_parity_cli(
        SimpleNamespace(
            execution_mode="dataflow",
            seq_lens="64,128",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
            block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
            block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
            seed=7,
            output_csv=None,
        )
    )

    assert captured["execution_modes"] == ["dataflow"]
    assert captured["seq_lens"] == [64, 128]
    assert captured["base_spec"].block1_topology_id == "m64_k64_n16_ps1_ph1_pd1"
    assert captured["base_spec"].block2_topology_id == "q32_kv64_e96_ps1_ph1_acc1"
    assert captured["base_spec"].block3_topology_id == "m32_k96_n64_ps4_pi3_d8_g1"


def test_validate_pattern_parity_includes_requested_block_topology_families(
    monkeypatch,
):
    class FakeReferenceTransformerLayer:
        def __init__(self, spec):
            self.spec = spec

        def assign_weights(self, weights):
            self.weights = weights

        def __call__(self, layer_inputs):
            return torch.zeros((1,), dtype=torch.float32)

    class FakePattern:
        def assign_weights(self, weights):
            self.weights = weights

        def prepare_benchmark_inputs(self, layer_inputs):
            self.layer_inputs = layer_inputs

        def __call__(self, layer_inputs):
            return torch.zeros((1,), dtype=torch.float32)

    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "ReferenceTransformerLayer",
        FakeReferenceTransformerLayer,
    )
    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "build_pattern",
        lambda execution_mode, spec: FakePattern(),
    )
    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "make_synthetic_layer_weights",
        lambda spec, seed: {},
    )
    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "make_synthetic_layer_inputs",
        lambda spec, seed: object(),
    )

    row = structured_validate_pattern_parity(
        execution_mode="dataflow",
        spec=TransformerLayerSpec(
            seq_len=64,
            block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
            block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
            block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
        ),
        seed=7,
    )

    assert row["block1_topology_family"] == "shared_runtime_qkv_proj"
    assert row["block2_topology_family"] == "fused_mha_out_proj"
    assert row["block3_topology_family"] == "pipelined_addnorm_ffn_addnorm"


def test_run_parity_cli_writes_stable_topology_columns(tmp_path: Path, monkeypatch):
    def fake_run_parity_suite(
        *, execution_modes, seq_lens, base_spec, seed, study_id=None
    ):
        return [
            {
                "study_id": "study",
                "execution_mode": "dataflow",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block1_topology_family": "shared_runtime_qkv_proj",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block2_topology_family": "fused_mha_out_proj",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "block3_topology_family": "pipelined_addnorm_ffn_addnorm",
                "batch_size": 1,
                "dtype": "bfloat16",
                "weights_source": "synthetic",
                "seed": 7,
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
            }
        ]

    monkeypatch.setattr(
        structured_validate_npu_parity_module,
        "run_parity_suite",
        fake_run_parity_suite,
    )

    output_csv = tmp_path / "parity.csv"
    structured_run_parity_cli(
        SimpleNamespace(
            execution_mode="dataflow",
            seq_lens="64",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
            block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
            block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
            seed=7,
            output_csv=str(output_csv),
        )
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    assert header == structured_parity_field_order


def test_write_parity_rows_csv_accepts_manifest_case_columns(tmp_path: Path):
    output_csv = tmp_path / "parity.csv"
    structured_validate_npu_parity_module.write_parity_rows_csv(
        output_csv,
        [
            {
                "study_id": "study",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "execution_mode": "dataflow",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block1_topology_family": "shared_runtime_qkv_proj",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block2_topology_family": "fused_mha_out_proj",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "block3_topology_family": "pipelined_addnorm_ffn_addnorm",
                "batch_size": 1,
                "dtype": "bfloat16",
                "weights_source": "synthetic",
                "seed": 7,
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
            }
        ],
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["study_case_id"] == "baseline_768"
    assert rows[0]["study_case_label"] == "baseline_768"
