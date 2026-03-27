import pytest

from iron.applications.transformer_layer.src.result_schema import normalize_result_row


def test_normalize_result_row_accepts_minimum_required_fields():
    row = normalize_result_row(
        {
            "study_id": "synthetic_transformer_layer",
            "backend": "npu",
            "execution_mode": "encoder_pipeline",
            "seq_len": 128,
            "batch_size": 1,
            "dtype": "bfloat16",
            "weights_source": "synthetic",
            "warmup_runs": 5,
            "runs_per_sample": 20,
            "measured_inference_count": 20,
            "timed_total_sec": 1.0,
            "avg_latency_ms": 50.0,
        }
    )
    assert row["execution_mode"] == "encoder_pipeline"
    assert row["avg_latency_ms"] == 50.0
    assert "topology_id" in row
    assert "power_backend" in row
    assert "flops_per_joule" in row
    assert "gflops_per_joule" in row


def test_normalize_result_row_rejects_missing_required_fields():
    with pytest.raises(KeyError):
        normalize_result_row({"study_id": "missing"})
