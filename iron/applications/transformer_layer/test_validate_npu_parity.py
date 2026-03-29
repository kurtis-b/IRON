# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from types import SimpleNamespace

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pipeline import (
    validate_npu_parity as structured_validate_npu_parity_module,
)
from iron.applications.transformer_layer.src.pipeline import (
    error_stats as structured_error_stats,
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
