from pathlib import Path

from iron.applications.transformer_layer.benchmark_common import (
    write_dict_rows_csv,
    write_results_csv,
)
from iron.applications.transformer_layer.plot_design_pattern_results import (
    _best_npu_rows,
    generate_plots,
)


def _write_fixture_suite(path: Path):
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
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 10.0,
                "throughput_flops_per_sec": 100.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": 200.0,
                "roofline_bound_ops_per_sec": 150.0,
                "backend_pct_of_peak": 0.5,
                "roofline_pct": 0.66,
                "avg_power_w": 20.0,
                "max_power_w": 22.0,
                "energy_j": 2.0,
                "power_sample_count": 3,
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
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 15.0,
                "throughput_flops_per_sec": 90.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": 200.0,
                "roofline_bound_ops_per_sec": 150.0,
                "backend_pct_of_peak": 0.45,
                "roofline_pct": 0.60,
                "avg_power_w": 25.0,
                "max_power_w": 28.0,
                "energy_j": 3.0,
                "power_sample_count": 3,
            },
            {
                "study_id": "design_patterns_main",
                "backend": "npu",
                "execution_mode": "operator_runlist",
                "pattern_label": "operator_runlist",
                "seq_len": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 18.0,
                "throughput_flops_per_sec": 80.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": 200.0,
                "roofline_bound_ops_per_sec": 150.0,
                "backend_pct_of_peak": 0.40,
                "roofline_pct": 0.53,
                "avg_power_w": 30.0,
                "max_power_w": 31.0,
                "energy_j": 4.0,
                "power_sample_count": 3,
            },
        ],
    )


def _write_fixture_bottlenecks(path: Path):
    write_dict_rows_csv(
        path,
        [
            {
                "study_id": "design_patterns_main",
                "execution_mode": "encoder_pipeline",
                "seq_len": 64,
                "avg_latency_ms": 10.0,
                "dominant_component": "encoder_pipeline",
                "dominant_component_fraction": 0.8,
            },
            {
                "study_id": "design_patterns_main",
                "execution_mode": "gemm_only",
                "seq_len": 64,
                "avg_latency_ms": 15.0,
                "dominant_component": "npu_gemm",
                "dominant_component_fraction": 0.7,
            },
            {
                "study_id": "design_patterns_main",
                "execution_mode": "operator_runlist",
                "seq_len": 64,
                "avg_latency_ms": 18.0,
                "dominant_component": "operator_runlist",
                "dominant_component_fraction": 0.9,
            },
        ],
    )


def _write_fixture_gpu(path: Path):
    write_results_csv(
        path,
        [
            {
                "study_id": "gpu_compare",
                "backend": "gpu",
                "execution_mode": "amd_gpu_reference",
                "pattern_label": "amd_gpu_reference",
                "seq_len": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 5.0,
                "throughput_flops_per_sec": 150.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": None,
                "roofline_bound_ops_per_sec": None,
                "backend_pct_of_peak": None,
                "roofline_pct": None,
                "avg_power_w": 17.0,
                "max_power_w": 18.0,
                "energy_j": 1.0,
                "power_sample_count": 2,
            }
        ],
    )


def _write_fixture_embedding_suite(path: Path):
    write_results_csv(
        path,
        [
            {
                "study_id": "embedding_scale",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "backend": "npu",
                "execution_mode": "encoder_pipeline",
                "pattern_label": "encoder_pipeline",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 10.0,
                "throughput_flops_per_sec": 100.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": 200.0,
                "roofline_bound_ops_per_sec": 150.0,
                "backend_pct_of_peak": 0.5,
                "roofline_pct": 0.66,
                "avg_power_w": 20.0,
                "max_power_w": 22.0,
                "energy_j": 2.0,
                "power_sample_count": 3,
                "run_status": "completed",
            },
            {
                "study_id": "embedding_scale",
                "study_case_id": "dense_8b_class",
                "study_case_label": "dense_8b_class",
                "backend": "npu",
                "execution_mode": "gemm_only",
                "pattern_label": "gemm_only",
                "seq_len": 64,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "attention_head_size": 128,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 12.0,
                "throughput_flops_per_sec": 120.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "backend_peak_ops_per_sec": 200.0,
                "roofline_bound_ops_per_sec": 150.0,
                "backend_pct_of_peak": 0.6,
                "roofline_pct": 0.8,
                "avg_power_w": 22.0,
                "max_power_w": 24.0,
                "energy_j": 2.2,
                "power_sample_count": 3,
                "run_status": "completed",
            },
        ],
    )


def _write_fixture_igpu(path: Path):
    write_results_csv(
        path,
        [
            {
                "study_id": "gpu_compare_embedding_igpu",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "backend": "gpu",
                "execution_mode": "amd_igpu_reference",
                "pattern_label": "amd_igpu_reference",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "source_model_name": None,
                "source_layer_index": None,
                "warmup_runs": 1,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 1.0,
                "avg_latency_ms": 5.0,
                "throughput_flops_per_sec": 150.0,
                "estimated_flops_per_inference": 10.0,
                "estimated_bytes_per_inference": 5.0,
                "operational_intensity_flops_per_byte": 2.0,
                "avg_power_w": 17.0,
                "max_power_w": 18.0,
                "energy_j": 1.0,
                "power_sample_count": 2,
                "run_status": "completed",
            }
        ],
    )


def test_best_npu_rows_selects_lowest_latency_per_sequence_length():
    rows = [
        {
            "backend": "npu",
            "execution_mode": "encoder_pipeline",
            "pattern_label": "encoder_pipeline",
            "seq_len": "64",
            "avg_latency_ms": "10.0",
        },
        {
            "backend": "npu",
            "execution_mode": "gemm_only",
            "pattern_label": "gemm_only",
            "seq_len": "64",
            "avg_latency_ms": "12.0",
        },
    ]

    best = _best_npu_rows(rows)

    assert len(best) == 1
    assert best[0]["pattern_label"] == "best_npu(encoder_pipeline)"


def test_generate_plots_writes_expected_svg_set(tmp_path):
    suite_csv = tmp_path / "suite.csv"
    bottleneck_csv = tmp_path / "bottlenecks.csv"
    gpu_csv = tmp_path / "gpu.csv"
    output_dir = tmp_path / "plots"
    _write_fixture_suite(suite_csv)
    _write_fixture_bottlenecks(bottleneck_csv)
    _write_fixture_gpu(gpu_csv)

    sections = generate_plots(
        input_csv=suite_csv,
        output_dir=output_dir,
        bottleneck_csv=bottleneck_csv,
        gpu_compare_csv=gpu_csv,
    )

    assert "latency_by_sequence_length.svg" in sections["main"]
    assert "bottleneck_breakdown.svg" in sections["main"]
    assert "best_npu_vs_amd_gpu_latency.svg" in sections["gpu_compare"]
    assert (output_dir / "index.html").exists()
    assert (output_dir / "latency_by_sequence_length.svg").exists()
    assert (output_dir / "best_npu_vs_amd_gpu_latency.svg").exists()
    assert (output_dir / "percent_of_roofline_by_sequence_length.svg").exists()
    latency_svg = (output_dir / "latency_by_sequence_length.svg").read_text(
        encoding="utf-8"
    )
    compare_svg = (output_dir / "best_npu_vs_amd_gpu_latency.svg").read_text(
        encoding="utf-8"
    )
    roofline_svg = (
        output_dir / "percent_of_roofline_by_sequence_length.svg"
    ).read_text(encoding="utf-8")
    assert "<rect" in latency_svg
    assert "<polyline" not in latency_svg
    assert 'opacity="0.9"' in latency_svg
    assert "<rect" in compare_svg
    assert "<polyline" not in compare_svg
    assert 'opacity="0.9"' in compare_svg
    assert "<circle" in roofline_svg
    assert 'opacity="0.9"' not in roofline_svg
    assert "Transformer Layer Thesis Plots" in (output_dir / "index.html").read_text(
        encoding="utf-8"
    )


def test_generate_plots_supports_hidden_size_axis_and_igpu_compare(tmp_path):
    suite_csv = tmp_path / "embedding_suite.csv"
    gpu_csv = tmp_path / "igpu.csv"
    output_dir = tmp_path / "embedding_plots"
    _write_fixture_embedding_suite(suite_csv)
    _write_fixture_igpu(gpu_csv)

    sections = generate_plots(
        input_csv=suite_csv,
        output_dir=output_dir,
        gpu_compare_csv=gpu_csv,
        x_axis="hidden_size",
    )

    assert any("latency_by_hidden_size" in name for name in sections["main"])
    assert any(
        "best_npu_vs_amd_igpu_latency" in name for name in sections["gpu_compare"]
    )
    assert (output_dir / "index.html").exists()
