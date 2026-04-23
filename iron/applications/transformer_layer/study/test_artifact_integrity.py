#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer.study.artifact_integrity import (
    validate_canonical_results_root,
    validate_paper_results_root,
    validate_p0_results_root,
)
from iron.applications.transformer_layer.study.end_to_end.validation import (
    REFERENCE_TOLERANCE_VALIDATION_MODE,
    validation_policy_manifest,
)
from iron.applications.transformer_layer.study.end_to_end.run import (
    RESULTS_CSV_FIELDNAMES as END_TO_END_RESULTS_FIELDNAMES,
)
from iron.applications.transformer_layer.study.host_comparison.run import (
    RESULTS_CSV_FIELDNAMES as HOST_COMPARISON_RESULTS_FIELDNAMES,
)
from iron.applications.transformer_layer.study.publish_results_root import (
    publish_results_root,
)
from iron.applications.transformer_layer.study.regenerate_plots import (
    main as regenerate_plots_main,
)
from iron.applications.transformer_layer.study.results_manifest import (
    load_results_root_manifest,
    write_results_root_manifest,
)


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_valid_results_root(tmp_path: Path) -> Path:
    results_root = tmp_path / "results"
    block_dir = results_root / "block"
    end_to_end_dir = results_root / "end_to_end"
    host_dir = results_root / "host_comparison"
    memory_tile_staging_dir = results_root / "memory_tile_staging"
    memcpy_dir = results_root / "memcpy_bandwidth"
    resource_usage_dir = results_root / "resource_usage"
    roofline_dir = results_root / "roofline"

    _write_text(block_dir / "results.csv", "study_id\n")

    _write_csv(
        end_to_end_dir / "results_all_power.csv",
        END_TO_END_RESULTS_FIELDNAMES,
        [
            {
                "study_id": "end_to_end",
                "campaign_id": "canonical",
                "repeat_index": 0,
                "matched_run_id": "canonical:baseline_768:hybrid:seq64:repeat0",
                "study_case_id": "baseline_768",
                "study_case_label": "BERT-Base",
                "workload_variant": "encoder_bert",
                "backend": "npu",
                "execution_mode": "hybrid",
                "pattern_label": "Hybrid",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "batch_size": 1,
                "dtype": "bf16",
                "use_bias": False,
                "weights_source": "synthetic",
                "joint_search_policy": "isolated_operator_screen_then_top_3_estimated_joint_full_patterns",
                "joint_candidate_count_total": 2,
                "joint_candidates_evaluated": 2,
                "selected_joint_rank": 1,
                "selection_provenance": "exhaustive_joint_search_best",
                "candidate_inventory_json": json.dumps(
                    {"qkv_proj": ["cand0", "cand1"]},
                    sort_keys=True,
                ),
                "candidate_count_by_operator_json": json.dumps(
                    {"qkv_proj": 2},
                    sort_keys=True,
                ),
                "warmup_runs": 1,
                "runs_per_sample": 10,
                "measured_inference_count": 10,
                "latency_sample_count": 10,
                "timed_total_sec": 0.1,
                "avg_latency_ms": 10.0,
                "min_latency_ms": 9.0,
                "max_latency_ms": 11.0,
                "compile_setup_time_ms": 1.0,
                "host_qkv_precompute_ms": "",
                "effective_gflops_per_sec": 100.0,
                "power_backend": "turbostat_pkgwatt",
                "power_boundary": "package",
                "power_estimation_method": "delta_package_power",
                "baseline_policy": "quiescent_package_power",
                "baseline_avg_power_w": 5.0,
                "active_avg_power_w": 15.0,
                "sensor_source": "turbostat:PkgWatt",
                "temperature_source": "unavailable",
                "avg_power_w": 10.0,
                "min_power_w": 9.0,
                "max_power_w": 11.0,
                "power_sample_count": 8,
                "raw_avg_power_w": 10.5,
                "raw_min_power_w": 9.0,
                "raw_max_power_w": 12.0,
                "raw_power_sample_count": 9,
                "power_std_w": 0.4,
                "raw_power_std_w": 0.6,
                "power_outlier_sample_count": 1,
                "power_outlier_filter_applied": True,
                "raw_package_avg_power_w": 15.0,
                "raw_package_min_power_w": 14.0,
                "raw_package_max_power_w": 16.0,
                "effective_gflops_per_sec_per_watt": 10.0,
                "npu_dispatch_count": 8,
                "npu_unique_instruction_binary_count": 8,
                "npu_unique_xclbin_count": 8,
                "process_model": "in_process",
                "validation_mode": REFERENCE_TOLERANCE_VALIDATION_MODE,
                "validation_error_count": 0,
                "run_status": "passed",
                "failure_message": "",
                "selected_candidate_ids_json": json.dumps(
                    {"qkv_proj": "cand0"},
                    sort_keys=True,
                ),
                "selected_config_json": json.dumps(
                    {"qkv_proj": {"tile_m": 16}},
                    sort_keys=True,
                ),
                "is_best": True,
            }
        ],
    )
    _write_text(end_to_end_dir / "tuning_all_power.csv", "study_id\n")
    _write_text(end_to_end_dir / "correctness_spot_checks.csv", "study_id\n")
    _write_text(end_to_end_dir / "latency_variation.csv", "study_id\n")
    _write_text(end_to_end_dir / "fairness_repeatability.csv", "study_id\n")
    _write_text(end_to_end_dir / "offload_breakdown.csv", "study_id\n")
    _write_text(end_to_end_dir / "runlist_launch_ablation.csv", "study_id\n")
    _write_text(end_to_end_dir / "staging_ablation.csv", "study_id\n")
    _write_text(end_to_end_dir / "search_validation.csv", "study_id\n")
    _write_text(
        end_to_end_dir / "search_validation_failure_taxonomy.csv",
        "execution_mode,failure_kind,count\n",
    )
    _write_text(end_to_end_dir / "error_distribution.csv", "study_id\n")
    _write_text(
        end_to_end_dir / "hybrid_selected_blocks_vs_pattern_latency.svg", "<svg/>"
    )
    _write_text(
        end_to_end_dir / "campaign_manifest.json",
        json.dumps(
            {
                "campaign_id": "canonical",
                "git_sha": "deadbeef",
                "python_version": "3.12.3",
                "platform": "Linux-test",
                "hostname": "host-a",
                "tool_paths": {"python3": "/usr/bin/python3"},
                "command_line": ["python3", "-m", "end_to_end.run"],
                "joint_search_policy": "isolated_operator_screen_then_top_3_estimated_joint_full_patterns",
                "candidate_policy_by_mode": {
                    "hybrid": {},
                    "runlist": {},
                    "offload": {},
                },
                "effective_candidate_inventory_by_case": {
                    "baseline_768:seq64": {
                        "hybrid": {
                            "candidate_ids_by_operator": {
                                "qkv_proj": ["cand0", "cand1"]
                            },
                            "candidate_count_by_operator": {"qkv_proj": 2},
                        }
                    }
                },
                "study_case_ids": ["baseline_768"],
                "validation_policy": validation_policy_manifest(),
            }
        ),
    )

    _write_csv(
        host_dir / "results.csv",
        HOST_COMPARISON_RESULTS_FIELDNAMES,
        [
            {
                "campaign_id": "canonical",
                "repeat_index": 0,
                "matched_run_id": "canonical:baseline_768:paired:seq64:repeat0",
                "workload_variant": "encoder_bert",
                "study_case_id": "baseline_768",
                "seq_len": 64,
                "metric": "effective_gflops_per_sec",
                "power_boundary_policy_family": "package",
                "igpu_power_boundary": "package",
                "npu_power_boundary": "package",
                "igpu_power_estimation_method": "delta_package_power",
                "npu_power_estimation_method": "delta_package_power",
                "igpu_baseline_policy": "quiescent_package_power",
                "npu_baseline_policy": "quiescent_package_power",
                "igpu": 80.0,
                "igpu_rocm_smi": "",
                "hybrid": 100.0,
                "runlist": 90.0,
                "offload": 70.0,
            },
            {
                "campaign_id": "canonical",
                "repeat_index": 0,
                "matched_run_id": "canonical:baseline_768:paired:seq64:repeat0",
                "workload_variant": "encoder_bert",
                "study_case_id": "baseline_768",
                "seq_len": 64,
                "metric": "effective_gflops_per_sec_per_watt",
                "power_boundary_policy_family": "package",
                "igpu_power_boundary": "package",
                "npu_power_boundary": "package",
                "igpu_power_estimation_method": "delta_package_power",
                "npu_power_estimation_method": "delta_package_power",
                "igpu_baseline_policy": "quiescent_package_power",
                "npu_baseline_policy": "quiescent_package_power",
                "igpu": "",
                "igpu_rocm_smi": 8.0,
                "hybrid": 10.0,
                "runlist": 9.0,
                "offload": 7.0,
            },
        ],
    )
    _write_text(host_dir / "fairness_repeatability.csv", "study_id\n")
    _write_text(host_dir / "effective_gflops_comparison.svg", "<svg/>")
    _write_text(host_dir / "effective_gflops_per_watt_comparison.svg", "<svg/>")
    _write_text(memory_tile_staging_dir / "results.csv", "study_id\n")
    _write_text(memcpy_dir / "results.csv", "study_id\n")
    _write_text(
        resource_usage_dir / "dataflow_block_best_configs.csv",
        "family_id,block_kind\n",
    )
    _write_text(resource_usage_dir / "hybrid_selected_ops.csv", "execution_mode\n")
    _write_text(resource_usage_dir / "runlist_selected_ops.csv", "execution_mode\n")
    _write_text(resource_usage_dir / "offload_selected_ops.csv", "execution_mode\n")
    _write_text(roofline_dir / "kernel_points.csv", "study_case_id\n")
    _write_text(roofline_dir / "implementation_points.csv", "study_case_id\n")
    _write_text(
        results_root / "analysis" / "crossover_report.csv",
        "study_case_id,execution_mode,seq_len\n",
    )
    _write_text(
        host_dir / "campaign_manifest.json",
        json.dumps(
            {
                "campaign_id": "canonical",
                "git_sha": "deadbeef",
                "python_version": "3.12.3",
                "platform": "Linux-test",
                "hostname": "host-a",
                "tool_paths": {"python3": "/usr/bin/python3"},
                "command_line": ["python3", "-m", "host_comparison.run"],
                "validation_policy": validation_policy_manifest(),
            }
        ),
    )
    write_results_root_manifest(results_root)
    return results_root


def test_validate_canonical_results_root_accepts_consistent_delta_power_bundle(
    tmp_path,
):
    results_root = _build_valid_results_root(tmp_path)

    validate_canonical_results_root(results_root)


def test_validate_canonical_results_root_rejects_mixed_per_watt_methods(tmp_path):
    results_root = _build_valid_results_root(tmp_path)
    host_results = results_root / "host_comparison" / "results.csv"
    rows = _load_host_rows(host_results)
    rows[1]["igpu_power_estimation_method"] = "direct_package_power"
    _write_csv(host_results, HOST_COMPARISON_RESULTS_FIELDNAMES, rows)

    try:
        validate_canonical_results_root(results_root)
    except ValueError as exc:
        assert "mixes power estimation methods" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected mixed power methods to fail validation")


def test_results_root_manifest_is_written_with_campaign_metadata(tmp_path):
    results_root = _build_valid_results_root(tmp_path)

    manifest = load_results_root_manifest(results_root)

    assert manifest is not None
    assert manifest["campaign_id"] == "canonical"
    assert manifest["git_sha"] == "deadbeef"
    assert manifest["study_manifest_files"] == {
        "end_to_end": "end_to_end/campaign_manifest.json",
        "host_comparison": "host_comparison/campaign_manifest.json",
    }
    assert manifest["plan_kind"] == ""
    assert manifest["evaluation_profile"] == ""


def test_results_root_manifest_records_plan_metadata_from_automation_state(tmp_path):
    results_root = _build_valid_results_root(tmp_path)
    automation_dir = results_root / "automation"
    automation_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        automation_dir / "state.json",
        json.dumps(
            {
                "run_id": "paper_run",
                "artifact_profile": "paper",
                "plan_kind": "paper_p0",
                "plan_layout": "paper",
                "run_user": "runner",
                "jobs": [],
            }
        ),
    )

    write_results_root_manifest(results_root)
    manifest = load_results_root_manifest(results_root)

    assert manifest is not None
    assert manifest["plan_kind"] == "paper_p0"
    assert manifest["evaluation_profile"] == "paper"
    assert manifest["automation"]["artifact_profile"] == "paper"


def test_publish_results_root_copies_validated_artifacts(tmp_path):
    source_root = _build_valid_results_root(tmp_path / "source")
    target_root = tmp_path / "published"

    published_root = publish_results_root(
        source_root=source_root,
        target_root=target_root,
    )

    assert published_root == target_root.resolve()
    validate_p0_results_root(target_root)
    manifest = load_results_root_manifest(target_root)
    assert manifest is not None
    assert manifest["source_results_root"] == str(source_root.resolve())
    assert (target_root / "analysis" / "crossover_report.csv").exists()
    assert (target_root / "end_to_end" / "results_all_power.csv").exists()
    assert (target_root / "host_comparison" / "results.csv").exists()


def test_publish_results_root_can_publish_paper_profile(tmp_path):
    source_root = _build_valid_results_root(tmp_path / "source")
    target_root = tmp_path / "published_paper"

    published_root = publish_results_root(
        source_root=source_root,
        target_root=target_root,
        artifact_profile="paper",
    )

    assert published_root == target_root.resolve()
    validate_paper_results_root(target_root)
    manifest = load_results_root_manifest(target_root)
    assert manifest is not None
    assert manifest["evaluation_profile"] == "paper"


def test_publish_results_root_refuses_existing_target_without_force(tmp_path):
    source_root = _build_valid_results_root(tmp_path / "source")
    target_root = tmp_path / "published"
    target_root.mkdir(parents=True)

    try:
        publish_results_root(source_root=source_root, target_root=target_root)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected publish_results_root to reject existing target")


def test_regenerate_plots_refuses_legacy_dataflow_artifacts(monkeypatch, tmp_path):
    results_root = _build_valid_results_root(tmp_path)
    _write_text(
        results_root / "end_to_end" / "legacy_dataflow_plot.svg",
        "<svg>Dataflow</svg>",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_block_plots",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_end_to_end_summary_plots",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_memory_tile_staging_plots",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_host_comparison_plots",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_offload_partitioning_plots",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.regenerate_plots.regenerate_correctness_plots",
        lambda results_root: None,
    )

    try:
        regenerate_plots_main(["--results-root", str(results_root)])
    except ValueError as exc:
        assert "legacy 'dataflow' labels" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected legacy dataflow artifacts to be rejected")


def _load_host_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_validate_paper_results_root_accepts_bundle_without_appendix_outputs(tmp_path):
    results_root = _build_valid_results_root(tmp_path)
    for relative_path in (
        ("end_to_end", "correctness_spot_checks.csv"),
        ("end_to_end", "latency_variation.csv"),
        ("end_to_end", "fairness_repeatability.csv"),
        ("host_comparison", "fairness_repeatability.csv"),
    ):
        results_root.joinpath(*relative_path).unlink()

    validate_paper_results_root(results_root)


def test_validate_paper_results_root_rejects_missing_required_file(tmp_path):
    results_root = _build_valid_results_root(tmp_path)
    missing_path = results_root / "resource_usage" / "hybrid_selected_ops.csv"
    missing_path.unlink()

    try:
        validate_paper_results_root(results_root)
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected paper validation to fail")
