#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from iron.applications.transformer_layer_new.study.igpu.run import (
    build_rows_for_group,
    resolve_sampling,
)
from iron.applications.transformer_layer_new.study.igpu.run_fairness_repeatability import (
    build_rows as build_fairness_rows,
)
from iron.applications.transformer_layer_new.study.igpu.select import (
    REFERENCE_EXECUTION_MODES,
    group_reference_rows,
)


def _reference_row(
    execution_mode: str,
    *,
    avg_latency_ms: str = "5.0",
    effective_gflops_per_sec: str = "12800.0",
    effective_gflops_per_sec_per_watt: str = "1066.7",
    run_status: str = "passed",
    backend: str = "npu",
    study_case_id: str = "baseline_768",
    seq_len: str = "64",
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
        "runs_per_sample": "100",
        "measured_inference_count": "100",
        "timed_total_sec": "0.5",
        "avg_latency_ms": avg_latency_ms,
        "effective_gflops_per_sec": effective_gflops_per_sec,
        "power_backend": "turbostat_pkgwatt",
        "avg_power_w": "12.0",
        "effective_gflops_per_sec_per_watt": effective_gflops_per_sec_per_watt,
        "process_model": "in_process",
        "validation_error_count": "0",
        "run_status": run_status,
        "failure_message": "",
        "selected_candidate_ids_json": json.dumps(
            {"candidate": execution_mode}, sort_keys=True
        ),
        "selected_config_json": json.dumps({"tile_m": 16}, sort_keys=True),
        "is_best": "False",
    }


def test_group_reference_rows_keeps_only_two_passing_npu_patterns():
    rows = [
        _reference_row("dataflow", avg_latency_ms="3.0"),
        _reference_row("runlist", avg_latency_ms="4.0"),
        _reference_row("dataflow", avg_latency_ms="5.0", backend="gpu"),
        _reference_row("dataflow", run_status="failed_validation", seq_len="128"),
    ]

    groups = group_reference_rows(rows)

    assert len(groups) == 1
    group = groups[0]
    assert group.study_case_id == "baseline_768"
    assert group.seq_len == 64
    assert (
        tuple(row["execution_mode"] for row in group.rows) == REFERENCE_EXECUTION_MODES
    )


def test_resolve_sampling_uses_100_timed_iterations_for_short_sequences():
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
        ]
    )[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 100


def test_resolve_sampling_falls_back_to_new_short_sequence_policy():
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
        ]
    )[0]
    zero_schedule_group = group.__class__(
        study_case_id=group.study_case_id,
        study_case_label=group.study_case_label,
        seq_len=group.seq_len,
        hidden_size=group.hidden_size,
        intermediate_size=group.intermediate_size,
        num_attention_heads=group.num_attention_heads,
        attention_head_size=group.attention_head_size,
        batch_size=group.batch_size,
        dtype=group.dtype,
        use_bias=group.use_bias,
        weights_source=group.weights_source,
        warmup_runs=0,
        runs_per_sample=0,
        rows=group.rows,
    )

    warmup_runs, runs_per_sample = resolve_sampling(
        zero_schedule_group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 100


def test_build_rows_for_group_aggregates_dataflow_and_runlist(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row(
                "dataflow",
                effective_gflops_per_sec="100.0",
                effective_gflops_per_sec_per_watt="10.0",
            ),
            _reference_row(
                "runlist",
                effective_gflops_per_sec="200.0",
                effective_gflops_per_sec_per_watt="20.0",
            ),
        ]
    )[0]

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.igpu.run.benchmark_igpu_group",
        lambda *args, **kwargs: {
            "effective_gflops_per_sec": 800.0,
            "effective_gflops_per_sec_per_watt": 80.0,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_rows_for_group(
        group,
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        device_name="cuda:0",
        power_backend="none",
        power_sample_interval_sec=0.2,
    )

    assert rows == [
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec",
            "igpu": 800.0,
            "dataflow": 100.0,
            "runlist": 200.0,
        },
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": 80.0,
            "dataflow": 10.0,
            "runlist": 20.0,
        },
    ]


def test_build_rows_for_group_blanks_igpu_values_on_gpu_failure(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
        ]
    )[0]

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.igpu.run.benchmark_igpu_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ROCm unavailable")),
    )

    rows = build_rows_for_group(
        group,
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        device_name="cuda:0",
        power_backend="rocm-smi",
        power_sample_interval_sec=0.2,
    )

    assert len(rows) == 2
    assert all(row["igpu"] is None for row in rows)
    assert rows[0]["dataflow"] == 12800.0
    assert rows[1]["runlist"] == 1066.7


def test_fairness_rows_report_two_reference_modes_and_new_schedule():
    rows = build_fairness_rows(device="cuda:0", power_backend="rocm-smi")

    assert len(rows) == 1
    assert json.loads(rows[0]["reference_execution_modes_json"]) == [
        "dataflow",
        "runlist",
    ]
    schedule = json.loads(rows[0]["iteration_schedule_json"])
    assert schedule["64-256"]["runs_per_sample"] == 100
