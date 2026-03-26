from pathlib import Path

import pytest
import torch

from iron.applications.transformer_layer.npu_inference import (
    SUPPORTED_EXECUTION_MODES,
    benchmark_pattern,
    build_pattern,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


class _FakePattern:
    def __init__(self, spec):
        self.spec = spec
        self.weights = None

    def assign_weights(self, weights):
        self.weights = weights

    def __call__(self, hidden_states):
        return hidden_states


@pytest.mark.parametrize("execution_mode", SUPPORTED_EXECUTION_MODES)
def test_build_pattern_dispatches(monkeypatch, execution_mode):
    spec = TransformerLayerSpec(seq_len=64)
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.EncoderPipelinePattern",
        _FakePattern,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.GemmOnlyPattern",
        _FakePattern,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.OperatorRunlistPattern",
        _FakePattern,
    )

    pattern = build_pattern(execution_mode, spec)

    assert isinstance(pattern, _FakePattern)
    assert pattern.spec == spec


def test_benchmark_pattern_emits_expected_schema(monkeypatch, tmp_path):
    spec = TransformerLayerSpec(seq_len=64)
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.build_pattern",
        lambda execution_mode, spec: _FakePattern(spec),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.make_synthetic_layer_weights",
        lambda spec, seed: {"dummy": torch.ones(1)},
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.make_synthetic_hidden_states",
        lambda spec, seed: torch.ones(
            (spec.batch_size, spec.seq_len, spec.hidden_size), dtype=spec.torch_dtype
        ),
    )
    output_csv = tmp_path / "results.csv"

    rows = benchmark_pattern(
        execution_mode="encoder_pipeline",
        spec=spec,
        warmup_runs=1,
        runs_per_sample=2,
        output_csv=str(output_csv),
        seed=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["execution_mode"] == "encoder_pipeline"
    assert row["backend"] == "npu"
    assert row["throughput_flops_per_sec"] is not None
    assert row["measured_inference_count"] == 2
    assert output_csv.exists()
