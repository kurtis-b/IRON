import csv
from pathlib import Path

from iron.applications.transformer_layer.automated_benchmark import (
    _failure_result_row,
    main,
)
from iron.applications.transformer_layer.benchmark_common import load_study_manifest
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_manifest_driven_automation_writes_suite_and_parity(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    parity_csv = tmp_path / "parity.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline","gemm_only"],'
            '"seq_lens":[64,128],'
            '"warmup_runs":1,'
            '"runs_per_sample":2,'
            f'"output_csv":"{suite_csv}",'
            '"parity":{"enabled":true,"execution_modes":["encoder_pipeline"],"seq_lens":[64],'
            f'"output_csv":"{parity_csv}"'
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    benchmark_calls = []

    def fake_benchmark_pattern(**kwargs):
        benchmark_calls.append(kwargs)
        spec = kwargs["spec"]
        return [
            {
                "study_id": "design_patterns_main",
                "backend": "npu",
                "execution_mode": kwargs["execution_mode"],
                "pattern_label": kwargs["execution_mode"],
                "seq_len": spec.seq_len,
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
                "avg_latency_ms": 10.0,
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
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        fake_benchmark_pattern,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark._run_parity_checks",
        lambda **kwargs: [
            {
                "study_id": "design_patterns_main",
                "execution_mode": "encoder_pipeline",
                "seq_len": 64,
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
        ],
    )

    main()

    assert len(benchmark_calls) == 4
    assert suite_csv.exists()
    assert parity_csv.exists()


def test_manifest_override_flags_take_precedence(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":2,'
            f'"output_csv":"{suite_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    calls = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--execution-modes",
            "gemm_only",
            "--seq-lens",
            "128",
            "--warmup-runs",
            "3",
            "--runs-per-sample",
            "4",
        ],
    )

    main()

    assert len(calls) == 1
    assert calls[0]["execution_mode"] == "gemm_only"
    assert calls[0]["spec"].seq_len == 128
    assert calls[0]["warmup_runs"] == 3
    assert calls[0]["runs_per_sample"] == 4


def test_manifest_can_request_annotated_output(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    annotated_csv = tmp_path / "suite_annotated.csv"
    peak_json = tmp_path / "peak.json"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            f'"output_csv":"{suite_csv}",'
            f'"peak_reference":"{peak_json}",'
            f'"annotated_output_csv":"{annotated_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        lambda **kwargs: [
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
                "avg_latency_ms": 1.0,
                "throughput_flops_per_sec": 20.0,
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
            }
        ],
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.annotate_results_csv",
        lambda input_csv, peak_reference_path, output_csv: annotated_csv.write_text(
            "annotated\n", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--skip-parity-check",
        ],
    )

    main()

    assert suite_csv.exists()
    assert annotated_csv.exists()


def test_load_study_manifest_resolves_roofline_paths(tmp_path):
    manifest_dir = tmp_path / "study"
    manifest_dir.mkdir()
    manifest = manifest_dir / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            '"output_csv":"../results/suite.csv",'
            '"peak_reference":"../config/peak.json",'
            '"annotated_output_csv":"../results/suite_annotated.csv"'
            "}"
        ),
        encoding="utf-8",
    )

    loaded = load_study_manifest(manifest)

    assert loaded["peak_reference"].endswith("/config/peak.json")
    assert loaded["annotated_output_csv"].endswith("/results/suite_annotated.csv")


def test_load_study_manifest_resolves_debug_log_path(tmp_path):
    manifest_dir = tmp_path / "study"
    manifest_dir.mkdir()
    manifest = manifest_dir / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            '"debug_log_csv":"../results/debug_log.csv"'
            "}"
        ),
        encoding="utf-8",
    )

    loaded = load_study_manifest(manifest)

    assert loaded["debug_log_csv"].endswith("/results/debug_log.csv")


def test_automation_writes_debug_log_for_completed_case(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    debug_log_csv = tmp_path / "debug.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            f'"output_csv":"{suite_csv}",'
            f'"debug_log_csv":"{debug_log_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        lambda **kwargs: [
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
                "avg_latency_ms": 1.0,
                "npu_dispatch_count": 3,
                "npu_unique_instruction_binary_count": 2,
                "throughput_flops_per_sec": 20.0,
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
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--skip-parity-check",
        ],
    )

    main()

    debug_log_text = debug_log_csv.read_text(encoding="utf-8")
    assert "study_started" in debug_log_text
    assert "benchmark_case_completed" in debug_log_text
    assert "study_completed" in debug_log_text


def test_automation_writes_debug_log_for_failed_case(monkeypatch, tmp_path):
    debug_log_csv = tmp_path / "debug.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"design_patterns_main",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            f'"debug_log_csv":"{debug_log_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    def _boom(**kwargs):
        raise RuntimeError("unsupported topology for requested placement")

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        _boom,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--skip-parity-check",
        ],
    )

    try:
        main()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    debug_log_text = debug_log_csv.read_text(encoding="utf-8")
    assert "benchmark_case_failed" in debug_log_text
    assert "unsupported_topology_or_placement" in debug_log_text
    assert "study_failed" in debug_log_text


def test_automation_applies_sampling_schedule(monkeypatch, tmp_path):
    suite_csv = tmp_path / "suite.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"sampling_schedule",'
            '"layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null},'
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64,128],'
            '"warmup_runs":1,'
            '"runs_per_sample":2,'
            '"sampling_schedule":{"64":{"warmup_runs":8,"runs_per_sample":40}},'
            f'"output_csv":"{suite_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_benchmark_pattern(**kwargs):
        calls.append(kwargs)
        spec = kwargs["spec"]
        return [
            {
                "study_id": "sampling_schedule",
                "backend": "npu",
                "execution_mode": kwargs["execution_mode"],
                "pattern_label": kwargs["execution_mode"],
                "seq_len": spec.seq_len,
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
                "avg_latency_ms": 10.0,
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
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        fake_benchmark_pattern,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--skip-parity-check",
        ],
    )

    main()

    assert len(calls) == 2
    assert calls[0]["spec"].seq_len == 64
    assert calls[0]["warmup_runs"] == 8
    assert calls[0]["runs_per_sample"] == 40
    assert calls[1]["spec"].seq_len == 128
    assert calls[1]["warmup_runs"] == 1
    assert calls[1]["runs_per_sample"] == 2


def test_automation_multi_case_continue_on_error_writes_failure_rows(
    monkeypatch, tmp_path
):
    suite_csv = tmp_path / "suite.csv"
    support_csv = tmp_path / "support.csv"
    manifest = tmp_path / "study.json"
    manifest.write_text(
        (
            "{"
            '"study_id":"embedding_scale",'
            '"study_cases":['
            '{"case_id":"baseline_768","case_label":"baseline_768","layer_spec":{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null}},'
            '{"case_id":"dense_4b_class","case_label":"dense_4b_class","layer_spec":{"hidden_size":2560,"intermediate_size":10240,"num_attention_heads":32,"batch_size":1,"seq_len":64,"dtype":"bfloat16","activation":"gelu","use_bias":false,"layer_norm_eps":1e-12,"attention_mask_mode":"none","weights_source":"synthetic","source_model_name":null,"source_layer_index":null}}'
            "],"
            '"execution_modes":["encoder_pipeline"],'
            '"seq_lens":[64],'
            '"warmup_runs":1,'
            '"runs_per_sample":1,'
            '"continue_on_error":true,'
            f'"output_csv":"{suite_csv}",'
            f'"support_matrix_csv":"{support_csv}"'
            "}"
        ),
        encoding="utf-8",
    )

    def fake_benchmark_pattern(**kwargs):
        spec = kwargs["spec"]
        if spec.hidden_size == 2560:
            raise RuntimeError("unsupported topology for requested placement")
        return [
            {
                "study_id": "embedding_scale",
                "backend": "npu",
                "execution_mode": kwargs["execution_mode"],
                "pattern_label": kwargs["execution_mode"],
                "seq_len": spec.seq_len,
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
                "avg_latency_ms": 10.0,
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
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.automated_benchmark.benchmark_pattern",
        fake_benchmark_pattern,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "automated_benchmark.py",
            "--study-manifest",
            str(manifest),
            "--skip-parity-check",
        ],
    )

    main()

    with suite_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    statuses = {(row["study_case_id"], row["run_status"]) for row in rows}
    assert ("baseline_768", "completed") in statuses
    assert ("dense_4b_class", "unsupported") in statuses

    with support_csv.open(newline="", encoding="utf-8") as handle:
        support_rows = list(csv.DictReader(handle))
    assert len(support_rows) == 2


def test_failure_result_row_marks_unsupported_pattern_surface():
    row = _failure_result_row(
        study_id="embedding_scale",
        case={
            "case_id": "dense_4b_class",
            "case_label": "dense_4b_class",
        },
        spec=TransformerLayerSpec(
            hidden_size=2560,
            intermediate_size=10240,
            num_attention_heads=32,
            seq_len=64,
        ),
        execution_mode="encoder_pipeline",
        warmup_runs=0,
        runs_per_sample=1,
        exc=ValueError(
            "encoder_pipeline thesis pattern currently supports only the 768/3072/12 and 1024/4096/16 families"
        ),
    )

    assert row["run_status"] == "unsupported"
    assert row["failure_category"] == "unsupported_pattern_surface"


def test_failure_result_row_marks_dma_limit_as_unsupported():
    row = _failure_result_row(
        study_id="embedding_scale",
        case={
            "case_id": "dense_8b_class",
            "case_label": "dense_8b_class",
        },
        spec=TransformerLayerSpec(
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            seq_len=512,
        ),
        execution_mode="gemm_only",
        warmup_runs=0,
        runs_per_sample=1,
        exc=RuntimeError(
            "MLIR compilation failed: 'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range."
        ),
    )

    assert row["run_status"] == "unsupported"
    assert row["failure_category"] == "unsupported_dma_descriptor_limits"
