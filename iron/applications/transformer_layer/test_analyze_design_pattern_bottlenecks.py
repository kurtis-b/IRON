import json

from iron.applications.transformer_layer.analyze_design_pattern_bottlenecks import (
    analyze_results,
    main,
)
from iron.applications.transformer_layer.benchmark_common import write_results_csv


def _write_sample_suite(path):
    write_results_csv(
        path,
        [
            {
                "study_id": "design_patterns_main",
                "backend": "npu",
                "execution_mode": "encoder_pipeline",
                "pattern_label": "encoder_pipeline",
                "seq_len": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 2,
                "measured_inference_count": 2,
                "timed_total_sec": 0.02,
                "avg_latency_ms": 10.0,
                "compile_setup_time_ms": 50.0,
                "avg_encoder_pipeline_latency_ms": 8.5,
                "npu_dispatch_count": 1,
                "npu_unique_instruction_binary_count": 1,
                "topology_id": "seq32_kv64__ps1_ph1_pffn1",
                "topology_family": "seq32_kv64",
                "parallel_seq": 1,
                "parallel_heads": 1,
                "parallel_ffn": 1,
                "compute_tile_count": 16,
                "compute_tile_utilization_fraction": 0.5,
                "process_model": "in_process",
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
            },
            {
                "study_id": "design_patterns_main",
                "backend": "npu",
                "execution_mode": "gemm_only",
                "pattern_label": "gemm_only",
                "seq_len": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 2,
                "measured_inference_count": 2,
                "timed_total_sec": 0.04,
                "avg_latency_ms": 20.0,
                "compile_setup_time_ms": 80.0,
                "avg_npu_gemm_latency_ms": 10.0,
                "avg_host_preprocess_latency_ms": 1.0,
                "avg_host_postprocess_latency_ms": 6.0,
                "avg_device_sync_latency_ms": 0.0,
                "npu_dispatch_count": 28,
                "npu_unique_instruction_binary_count": 4,
                "process_model": "in_process",
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
            },
        ],
    )


def test_analyze_results_derives_pattern_specific_summaries(tmp_path):
    suite_csv = tmp_path / "suite.csv"
    _write_sample_suite(suite_csv)

    summary_rows, aggregates, text_summary = analyze_results(suite_csv)

    assert len(summary_rows) == 2
    assert summary_rows[0]["dominant_component"] == "encoder_pipeline"
    assert summary_rows[0]["dominant_component_fraction"] == 0.85
    assert "seq32_kv64__ps1_ph1_pffn1" in summary_rows[0]["bottleneck_summary"]
    assert aggregates["gemm_only"]["dominant_component_counts"] == {"npu_gemm": 1}
    assert "encoder_pipeline: encoder_pipeline dominates 1/1 rows" in text_summary


def test_cli_writes_csv_json_and_text_outputs(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    summary_csv = tmp_path / "summary.csv"
    summary_json = tmp_path / "summary.json"
    summary_text = tmp_path / "summary.txt"
    _write_sample_suite(suite_csv)

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_design_pattern_bottlenecks.py",
            "--input-csv",
            str(suite_csv),
            "--summary-csv",
            str(summary_csv),
            "--summary-json",
            str(summary_json),
            "--summary-text",
            str(summary_text),
        ],
    )

    main()

    assert summary_csv.exists()
    assert summary_json.exists()
    assert summary_text.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["encoder_pipeline"]["row_count"] == 1
