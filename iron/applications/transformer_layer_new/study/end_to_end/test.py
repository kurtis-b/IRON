#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from iron.applications.transformer_layer_new.study.end_to_end import modes
from iron.applications.transformer_layer_new.study.end_to_end.cases import (
    EXECUTION_MODES,
    MODE_OPERATORS,
    EndToEndWorkload,
    candidate_table_for_case,
    load_default_candidate_payloads,
)
from iron.applications.transformer_layer_new.study.end_to_end.modes import (
    benchmark_mode,
)
from iron.applications.transformer_layer_new.study.end_to_end.run import (
    iteration_schedule,
)
from iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks import (
    build_rows as build_correctness_rows,
)
from iron.applications.transformer_layer_new.study.end_to_end.run_fairness_repeatability import (
    build_rows as build_fairness_rows,
)
from iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation import (
    build_rows as build_latency_rows,
    summarize_latency_samples,
)
from iron.applications.transformer_layer_new.study.end_to_end.run_staging_ablation import (
    build_rows as build_staging_rows,
)
from iron.applications.transformer_layer_new.study.end_to_end.select import (
    select_result_rows,
)


def _result_row(
    execution_mode: str,
    *,
    seq_len: str = "64",
    study_case_id: str = "baseline_768",
    avg_latency_ms: str = "5.0",
    power_backend: str = "none",
    backend: str = "npu",
) -> dict[str, str]:
    return {
        "study_id": "end_to_end",
        "study_case_id": study_case_id,
        "study_case_label": study_case_id,
        "backend": backend,
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": seq_len,
        "hidden_size": "768",
        "intermediate_size": "3072",
        "num_attention_heads": "12",
        "attention_head_size": "64",
        "batch_size": "1",
        "dtype": "bf16",
        "use_bias": "False",
        "weights_source": "synthetic",
        "warmup_runs": "1",
        "runs_per_sample": "100" if int(seq_len) <= 256 else "10",
        "measured_inference_count": "10",
        "timed_total_sec": "0.5",
        "avg_latency_ms": avg_latency_ms,
        "compile_setup_time_ms": "1.0",
        "host_qkv_precompute_ms": "",
        "effective_gflops_per_sec": "1000.0",
        "power_backend": power_backend,
        "avg_power_w": "12.0",
        "effective_gflops_per_sec_per_watt": "80.0",
        "npu_dispatch_count": "8",
        "npu_unique_instruction_binary_count": "8",
        "npu_unique_xclbin_count": "8",
        "process_model": "in_process",
        "validation_error_count": "0",
        "run_status": "passed",
        "failure_message": "",
        "selected_candidate_ids_json": json.dumps(
            {"candidate": execution_mode},
            sort_keys=True,
        ),
        "selected_config_json": json.dumps(
            {"operator": {"tile_m": 16}},
            sort_keys=True,
        ),
        "is_best": "False",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_candidate_table_covers_only_dataflow_and_runlist():
    payloads = load_default_candidate_payloads()
    assert set(payloads) == {"dataflow", "runlist"}

    table = candidate_table_for_case("baseline_768", 64, payloads=payloads)
    assert set(table) == set(EXECUTION_MODES)
    for execution_mode in EXECUTION_MODES:
        assert set(table[execution_mode]) == set(MODE_OPERATORS[execution_mode])


def test_iteration_schedule_uses_100_timed_iterations_through_256():
    assert iteration_schedule(64) == (1, 100)
    assert iteration_schedule(128) == (1, 100)
    assert iteration_schedule(256) == (1, 100)
    assert iteration_schedule(512) == (1, 10)
    assert iteration_schedule(4096) == (1, 5)
    assert iteration_schedule(8192) == (1, 2)


def test_select_result_rows_parses_selected_config_and_filters_modes():
    rows = [
        _result_row("dataflow", seq_len="64"),
        _result_row("runlist", seq_len="64"),
        _result_row("dataflow", seq_len="64", backend="gpu"),
    ]

    selected_rows = select_result_rows(rows)

    assert len(selected_rows) == 2
    assert tuple(row.execution_mode for row in selected_rows) == ("dataflow", "runlist")
    assert selected_rows[0].selected_config == {"operator": {"tile_m": 16}}


def test_benchmark_mode_capture_latencies_and_forced_reference(monkeypatch):
    fake_output = torch.ones((2, 2), dtype=torch.float32)
    include_output_calls: list[bool] = []

    class FakeContext:
        def compile_all(self):
            return None

        def prepare_runtime(self):
            return None

        def reset_runtime(self):
            return None

    class FakeOperator:
        def __init__(self):
            self.context = FakeContext()

        def forward(self, _input):
            return fake_output

    def fake_generate_golden_reference(*args, **kwargs):
        include_output_calls.append(bool(kwargs["include_output"]))
        return {
            "input": torch.ones((2, 2), dtype=torch.float32),
            "output": fake_output,
            "weights": {},
        }

    monkeypatch.setattr(
        modes, "generate_golden_reference", fake_generate_golden_reference
    )
    monkeypatch.setattr(
        modes, "_build_operator", lambda *args, **kwargs: FakeOperator()
    )
    monkeypatch.setattr(modes, "_metadata_for_operator", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        modes,
        "_measure_power",
        lambda *args, **kwargs: {"power_backend": "none", "avg_power_w": None},
    )

    result = benchmark_mode(
        "dataflow",
        EndToEndWorkload(64, 768, 3072, 12),
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        power_backend="none",
        include_reference_output=True,
        capture_latencies=True,
    )

    assert include_output_calls == [True]
    assert result["run_status"] == "passed"
    assert len(result["latency_samples_ms"]) == 2


def test_correctness_rows_only_include_spot_check_sequences(monkeypatch, tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(
        results_input,
        [
            _result_row("dataflow", seq_len="512"),
            _result_row("runlist", seq_len="2048"),
            _result_row("dataflow", seq_len="4096"),
        ],
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks.benchmark_mode",
        lambda *args, **kwargs: {
            "avg_latency_ms": 3.5,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_correctness_rows(
        results_input=results_input,
        family_filter="all",
        mode_filter="all",
        seed=42,
    )

    assert len(rows) == 2
    assert {row["seq_len"] for row in rows} == {512, 2048}
    assert {row["validation_mode"] for row in rows} == {
        "exact_reference",
        "numerical_spot_check",
    }


def test_summarize_latency_samples_reports_expected_statistics():
    summary = summarize_latency_samples([1.0, 2.0, 3.0])

    assert summary["sample_count"] == 3
    assert summary["mean_latency_ms"] == 2.0
    assert summary["min_latency_ms"] == 1.0
    assert summary["max_latency_ms"] == 3.0


def test_latency_variation_rows_capture_sample_statistics(monkeypatch, tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(results_input, [_result_row("dataflow", seq_len="64")])

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation.benchmark_mode",
        lambda *args, **kwargs: {
            "latency_samples_ms": [1.0, 2.0, 3.0],
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_latency_rows(
        results_input=results_input,
        family_filter="all",
        mode_filter="all",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
    )

    assert len(rows) == 1
    assert rows[0]["runs_per_sample"] == 100
    assert rows[0]["sample_count"] == 3
    assert rows[0]["mean_latency_ms"] == 2.0


def test_staging_ablation_rows_join_staging_and_end_to_end_results(tmp_path):
    end_to_end_results = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    _write_csv(
        end_to_end_results,
        [_result_row("dataflow", seq_len="64", avg_latency_ms="8.0")],
    )
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "64",
                "block_kind": "ffn",
                "source_staging_depth": "1",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
                "is_best_depth": "False",
                "speedup_vs_depth1": "1.0",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "64",
                "block_kind": "ffn",
                "source_staging_depth": "1",
                "staging_depth": "2",
                "avg_latency_ms": "2.0",
                "run_status": "passed",
                "is_best_depth": "True",
                "speedup_vs_depth1": "2.0",
            },
        ],
    )

    rows = build_staging_rows(
        end_to_end_results=end_to_end_results,
        staging_results=staging_results,
        family_filter="all",
    )

    assert len(rows) == 1
    assert rows[0]["speedup_vs_source_depth"] == 2.0
    assert rows[0]["dataflow_end_to_end_latency_ms"] == 8.0


def test_fairness_rows_report_two_modes_and_new_schedule(tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(
        results_input,
        [
            _result_row("dataflow", seq_len="64", power_backend="none"),
            _result_row("runlist", seq_len="64", power_backend="turbostat_pkgwatt"),
        ],
    )

    rows = build_fairness_rows(results_input)

    assert {row["execution_mode"] for row in rows} == {"dataflow", "runlist"}
    schedule = json.loads(rows[0]["iteration_schedule_json"])
    assert schedule["64-256"]["runs_per_sample"] == 100
    assert rows[0]["selected_candidate_source"]
