import csv
import json

from iron.applications.transformer_layer.gpu_compare_best_npu import (
    benchmark_best_npu_vs_gpu,
    load_compare_config,
    select_best_npu_rows,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_select_best_npu_rows_picks_lowest_latency_per_case_and_seq():
    rows = [
        {
            "backend": "npu",
            "run_status": "completed",
            "study_case_id": "baseline_768",
            "seq_len": "64",
            "execution_mode": "encoder_pipeline",
            "avg_latency_ms": "10.0",
        },
        {
            "backend": "npu",
            "run_status": "completed",
            "study_case_id": "baseline_768",
            "seq_len": "64",
            "execution_mode": "gemm_only",
            "avg_latency_ms": "12.0",
        },
        {
            "backend": "npu",
            "run_status": "completed",
            "study_case_id": "dense_4b_class",
            "seq_len": "64",
            "execution_mode": "operator_runlist",
            "avg_latency_ms": "8.0",
        },
    ]

    best = select_best_npu_rows(rows)

    assert len(best) == 2
    assert best[0]["execution_mode"] == "encoder_pipeline"
    assert best[1]["execution_mode"] == "operator_runlist"


def test_load_compare_config_resolves_relative_paths(tmp_path):
    config = tmp_path / "igpu.json"
    config.write_text(
        json.dumps(
            {
                "reference_npu_csv": "../results/suite.csv",
                "output_csv": "../results/igpu.csv",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_compare_config(config)

    assert loaded["reference_npu_csv"].endswith("/results/suite.csv")
    assert loaded["output_csv"].endswith("/results/igpu.csv")
    assert loaded["execution_mode"] == "amd_igpu_reference"


def test_benchmark_best_npu_vs_gpu_carries_case_and_reference_fields(
    monkeypatch, tmp_path
):
    suite_csv = tmp_path / "suite.csv"
    output_csv = tmp_path / "igpu.csv"
    with suite_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "run_status",
                "study_case_id",
                "study_case_label",
                "seq_len",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "attention_head_size",
                "batch_size",
                "dtype",
                "use_bias",
                "weights_source",
                "source_model_name",
                "source_layer_index",
                "execution_mode",
                "avg_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "backend": "npu",
                "run_status": "completed",
                "study_case_id": "dense_4b_class",
                "study_case_label": "dense_4b_class",
                "seq_len": 64,
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_attention_heads": 32,
                "attention_head_size": 80,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": "",
                "source_layer_index": "",
                "execution_mode": "operator_runlist",
                "avg_latency_ms": 9.0,
            }
        )

    benchmark_calls = []

    def fake_benchmark_gpu_layer(**kwargs):
        benchmark_calls.append(kwargs)
        spec: TransformerLayerSpec = kwargs["spec"]
        return [
            {
                "study_id": "gpu_compare",
                "backend": "gpu",
                "execution_mode": "amd_gpu_reference",
                "pattern_label": "amd_gpu_reference",
                "seq_len": spec.seq_len,
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "num_attention_heads": spec.num_attention_heads,
                "attention_head_size": spec.attention_head_size,
                "batch_size": spec.batch_size,
                "dtype": spec.dtype,
                "use_bias": spec.use_bias,
                "weights_source": spec.weights_source,
                "source_model_name": spec.source_model_name,
                "source_layer_index": spec.source_layer_index,
                "warmup_runs": kwargs["warmup_runs"],
                "runs_per_sample": kwargs["runs_per_sample"],
                "measured_inference_count": kwargs["runs_per_sample"],
                "timed_total_sec": 1.0,
                "avg_latency_ms": 5.0,
                "throughput_flops_per_sec": 100.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": None,
                "roofline_bound_ops_per_sec": None,
                "backend_pct_of_peak": None,
                "roofline_pct": None,
                "avg_power_w": None,
                "max_power_w": None,
                "energy_j": None,
                "power_sample_count": None,
                "run_status": "completed",
                "failure_component": None,
                "failure_category": None,
                "failure_message": None,
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.gpu_compare_best_npu.benchmark_gpu_layer",
        fake_benchmark_gpu_layer,
    )

    rows = benchmark_best_npu_vs_gpu(
        {
            "study_id": "gpu_compare_embedding_igpu",
            "reference_npu_csv": str(suite_csv),
            "output_csv": str(output_csv),
            "execution_mode": "amd_igpu_reference",
            "pattern_label": "amd_igpu_reference",
            "device": "cuda:1",
            "power_backend": "none",
            "power_sample_interval_sec": 0.2,
            "warmup_runs": 10,
            "runs_per_sample": 50,
        }
    )

    assert len(benchmark_calls) == 1
    assert benchmark_calls[0]["spec"].hidden_size == 2560
    assert benchmark_calls[0]["warmup_runs"] == 10
    assert rows[0]["study_case_id"] == "dense_4b_class"
    assert rows[0]["reference_npu_execution_mode"] == "operator_runlist"
    assert rows[0]["execution_mode"] == "amd_igpu_reference"
