#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path
import types

import torch

from iron.applications.transformer_layer.study.analysis.crossover_report import (
    build_rows as build_crossover_rows,
)
from iron.applications.transformer_layer.study.correctness.common import error_summary
from iron.applications.transformer_layer.study.correctness.error_distribution import (
    parse_args as error_distribution_parse_args,
)
from iron.applications.transformer_layer.study.cpu_baseline.run import (
    main as cpu_baseline_main,
)
from iron.applications.transformer_layer.study.offload_partitioning.run_query_block_sweep import (
    _selected_query_block_sizes,
    parse_args as offload_partitioning_parse_args,
)
from iron.applications.transformer_layer.study.reviewer_quickcheck.run import (
    main as reviewer_quickcheck_main,
)
from iron.applications.transformer_layer.study.runlist_launch_ablation.run import (
    _annotate_speedups,
    parse_args as runlist_launch_ablation_parse_args,
)
from iron.applications.transformer_layer.study.search_validation.run import (
    parse_args as search_validation_parse_args,
)
from iron.applications.transformer_layer.study.search_validation.failure_taxonomy import (
    build_taxonomy,
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_error_summary_reports_histogram_and_thresholds():
    summary = error_summary(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        torch.tensor([0.0, 0.5, 1.5], dtype=torch.float32),
        abs_tol=0.25,
        rel_tol=0.10,
        histogram_bins=4,
    )

    assert summary["abs_error_max"] == 0.5
    assert summary["abs_threshold_fraction"] > 0.0
    assert "abs_histogram_edges_json" in summary
    assert "abs_histogram_counts_json" in summary


def test_build_taxonomy_groups_passed_and_selection_delta():
    rows = [
        {"execution_mode": "hybrid", "run_status": "passed", "selected_matches_baseline": "True"},
        {"execution_mode": "hybrid", "run_status": "passed", "selected_matches_baseline": "False"},
        {"execution_mode": "runlist", "run_status": "failed_exception", "selected_matches_baseline": "False"},
    ]

    taxonomy = build_taxonomy(rows)

    assert {"execution_mode": "hybrid", "failure_kind": "passed", "count": 1} in taxonomy
    assert {"execution_mode": "hybrid", "failure_kind": "selection_delta", "count": 1} in taxonomy
    assert {"execution_mode": "runlist", "failure_kind": "failed_exception", "count": 1} in taxonomy


def test_annotate_speedups_uses_separate_launch_baseline():
    rows = [
        {
            "study_case_id": "baseline_768",
            "repeat_index": 0,
            "workload_variant": "encoder_bert",
            "seq_len": 256,
            "submission_model": "separate_launch",
            "avg_latency_ms": 10.0,
            "run_status": "passed",
        },
        {
            "study_case_id": "baseline_768",
            "repeat_index": 0,
            "workload_variant": "encoder_bert",
            "seq_len": 256,
            "submission_model": "runlist",
            "avg_latency_ms": 5.0,
            "run_status": "passed",
        },
    ]

    _annotate_speedups(rows)

    assert rows[0]["speedup_vs_separate_launch"] == 1.0
    assert rows[1]["speedup_vs_separate_launch"] == 2.0


def test_reviewer_quickcheck_delegates_to_execution_smoke(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def record_subprocess(argv, check):
        calls["env"] = (tuple(argv), check)
        return None

    def record_unattended(argv):
        calls["unattended"] = tuple(argv)
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.reviewer_quickcheck.run.subprocess.run",
        record_subprocess,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.reviewer_quickcheck.run.unattended_reboot.main",
        record_unattended,
    )

    env_script = tmp_path / "env.sh"
    env_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    exit_code = reviewer_quickcheck_main(
        [
            "--run-id",
            "quickcheck",
            "--env-check-script",
            str(env_script),
            "--results-root",
            str(tmp_path / "results"),
            "--artifact-profile",
            "paper",
        ]
    )

    assert exit_code == 0
    assert calls["env"] == ((str(env_script),), True)
    assert calls["unattended"][0] == "execution-smoke-test"
    assert "--artifact-profile" in calls["unattended"]
    assert (
        calls["unattended"][calls["unattended"].index("--artifact-profile") + 1]
        == "paper"
    )


def test_helper_cli_parsers_accept_sampling_overrides():
    runlist_args = runlist_launch_ablation_parse_args(
        ["--warmup-runs", "1", "--runs-per-sample", "3"]
    )
    search_args = search_validation_parse_args(
        ["--warmup-runs", "1", "--runs-per-sample", "3"]
    )
    error_args = error_distribution_parse_args(
        ["--warmup-runs", "1", "--runs-per-sample", "3"]
    )

    assert (runlist_args.warmup_runs, runlist_args.runs_per_sample) == (1, 3)
    assert (search_args.warmup_runs, search_args.runs_per_sample) == (1, 3)
    assert (error_args.warmup_runs, error_args.runs_per_sample) == (1, 3)


def test_offload_partitioning_cli_and_representative_policy(monkeypatch):
    args = offload_partitioning_parse_args(
        [
            "--warmup-runs",
            "1",
            "--runs-per-sample",
            "3",
            "--query-block-policy",
            "representative",
        ]
    )

    assert args.query_block_policy == "representative"
    assert (args.warmup_runs, args.runs_per_sample) == (1, 3)

    selected_row = types.SimpleNamespace(
        selected_config={"attn_scores": {"query_block_size": 8}},
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.offload_partitioning.run_query_block_sweep._legal_query_block_sizes",
        lambda row: (1, 2, 4, 8, 16, 32),
    )

    assert _selected_query_block_sizes(
        selected_row,
        query_block_policy="representative",
    ) == (1, 4, 8, 32)


def test_cpu_baseline_defaults_host_backends_to_cpu(monkeypatch):
    calls: dict[str, object] = {}

    def record_host_comparison(argv):
        calls["argv"] = list(argv)
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.cpu_baseline.run.host_comparison_main",
        record_host_comparison,
    )

    exit_code = cpu_baseline_main(["--family", "baseline_768"])

    assert exit_code == 0
    assert calls["argv"] == ["--family", "baseline_768", "--host-backends", "cpu"]


def test_crossover_report_joins_roofline_and_memcpy(tmp_path):
    end_to_end_path = tmp_path / "end_to_end.csv"
    roofline_path = tmp_path / "implementation_points.csv"
    memcpy_path = tmp_path / "memcpy.csv"
    _write_csv(
        end_to_end_path,
        (
            "campaign_id",
            "repeat_index",
            "matched_run_id",
            "study_case_id",
            "study_case_label",
            "workload_variant",
            "backend",
            "execution_mode",
            "seq_len",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "attention_head_size",
            "selected_candidate_ids_json",
            "selected_config_json",
            "joint_search_policy",
            "joint_candidate_count_total",
            "joint_candidates_evaluated",
            "selection_provenance",
            "candidate_inventory_json",
            "candidate_count_by_operator_json",
            "validation_mode",
            "run_status",
            "avg_latency_ms",
            "effective_gflops_per_sec",
        ),
        [
            {
                "campaign_id": "canonical",
                "repeat_index": 0,
                "matched_run_id": "m",
                "study_case_id": "baseline_768",
                "study_case_label": "BERT-Base",
                "workload_variant": "encoder_bert",
                "backend": "npu",
                "execution_mode": "hybrid",
                "seq_len": 256,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "selected_candidate_ids_json": "{\"qkv_proj\":\"cand0\"}",
                "selected_config_json": "{\"qkv_proj\":{}}",
                "joint_search_policy": "policy",
                "joint_candidate_count_total": 1,
                "joint_candidates_evaluated": 1,
                "selection_provenance": "exhaustive_joint_search_best",
                "candidate_inventory_json": "{\"qkv_proj\":[\"cand0\"]}",
                "candidate_count_by_operator_json": "{\"qkv_proj\":1}",
                "validation_mode": "reference_tolerance_validation",
                "run_status": "passed",
                "avg_latency_ms": 4.0,
                "effective_gflops_per_sec": 100.0,
            }
        ],
    )
    _write_csv(
        roofline_path,
        (
            "study_case_id",
            "family_label",
            "workload_variant",
            "execution_mode",
            "seq_len",
            "compute_tiles_used",
            "shim_tiles_used",
            "run_status",
            "omission_note",
            "avg_latency_ms",
            "effective_gflops_per_sec",
            "implementation_flop_count",
            "implementation_byte_count",
            "operational_intensity_flops_per_byte",
            "selected_candidate_ids_json",
            "plot_label",
        ),
        [
            {
                "study_case_id": "baseline_768",
                "family_label": "BERT-Base",
                "workload_variant": "encoder_bert",
                "execution_mode": "hybrid",
                "seq_len": 256,
                "compute_tiles_used": 8,
                "shim_tiles_used": 2,
                "run_status": "passed",
                "omission_note": "",
                "avg_latency_ms": 4.0,
                "effective_gflops_per_sec": 100.0,
                "implementation_flop_count": 1,
                "implementation_byte_count": 1,
                "operational_intensity_flops_per_byte": 2.0,
                "selected_candidate_ids_json": "{}",
                "plot_label": "Hybrid@256",
            }
        ],
    )
    _write_csv(
        memcpy_path,
        ("run_status", "bandwidth_gbps"),
        [{"run_status": "passed", "bandwidth_gbps": 50.0}],
    )

    rows = build_crossover_rows(
        end_to_end_input=end_to_end_path,
        roofline_input=roofline_path,
        memcpy_input=memcpy_path,
    )

    assert len(rows) == 1
    assert rows[0]["peak_memcpy_bandwidth_gbps"] == 50.0
    assert rows[0]["dominant_limiter"] in {"bandwidth", "memory_pressure", "parallelism", "compute"}
