import pytest
import torch

from iron.applications.transformer_layer.gpu_inference import (
    benchmark_gpu_layer,
    resolve_amd_gpu_device,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_benchmark_gpu_layer_emits_shared_schema_on_cpu_for_unit_test(tmp_path):
    output_csv = tmp_path / "gpu.csv"
    spec = TransformerLayerSpec(seq_len=64)

    rows = benchmark_gpu_layer(
        spec=spec,
        warmup_runs=1,
        runs_per_sample=2,
        output_csv=str(output_csv),
        seed=0,
        device=torch.device("cpu"),
        power_backend="none",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["backend"] == "gpu"
    assert row["execution_mode"] == "amd_gpu_reference"
    assert row["pattern_label"] == "amd_gpu_reference"
    assert row["gpu_device"] == "cpu"
    assert row["power_backend"] == "none"
    assert row["flops_per_joule"] is None
    assert row["gflops_per_joule"] is None
    assert row["power_sample_count"] is None
    assert output_csv.exists()


def test_resolve_amd_gpu_device_requires_rocm_visible_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_amd_gpu_device("cuda:0")
