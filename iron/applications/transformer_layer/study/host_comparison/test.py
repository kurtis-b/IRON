#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

import torch

from iron.applications.transformer_layer.study.host_comparison.remeasure_power_only import (
    main as remeasure_power_only_main,
)
from iron.applications.transformer_layer.study.host_comparison.run import (
    RESULTS_CSV_FIELDNAMES,
    _forward_reference,
    _normalized_existing_row,
    RocmSMIPowerMonitor,
    build_rows_for_group,
    configure_cpu_runtime_for_max_physical_cores,
    generate_synthetic_reference,
    load_existing_rows,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
    resolve_power_sampling_policy,
    resolve_sampling,
)
from iron.applications.transformer_layer.study.host_comparison.run_fairness_repeatability import (
    build_rows as build_fairness_rows,
)
from iron.applications.transformer_layer.study.host_comparison.select import (
    REFERENCE_EXECUTION_MODES,
    default_reference_results_path,
    group_reference_rows,
)


def _reference_row(
    *,
    avg_latency_ms: str = "5.0",
    effective_gflops_per_sec: str = "12800.0",
    effective_gflops_per_sec_per_watt: str = "1066.7",
    run_status: str = "passed",
    backend: str = "npu",
    execution_mode: str = "hybrid",
    study_case_id: str = "tinybert_512",
    seq_len: str = "64",
) -> dict[str, str]:
    hidden_size = "512" if study_case_id == "tinybert_512" else "768"
    intermediate_size = "2048" if study_case_id == "tinybert_512" else "3072"
    num_heads = "8" if study_case_id == "tinybert_512" else "12"
    return {
        "study_id": "end_to_end",
        "study_case_id": study_case_id,
        "study_case_label": study_case_id,
        "workload_variant": "encoder_bert",
        "backend": backend,
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
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


def test_group_reference_rows_keeps_only_passing_reference_npu_rows():
    rows = [
        _reference_row(),
        _reference_row(
            execution_mode="offload",
            avg_latency_ms="6.0",
            effective_gflops_per_sec="6400.0",
            effective_gflops_per_sec_per_watt="533.3",
        ),
        _reference_row(backend="gpu"),
        _reference_row(run_status="failed_validation", seq_len="128"),
        _reference_row(execution_mode="runlist"),
    ]

    groups = group_reference_rows(rows)

    assert len(groups) == 1
    group = groups[0]
    assert group.study_case_id == "tinybert_512"
    assert group.workload_variant == "encoder_bert"
    assert group.seq_len == 64
    assert (
        tuple(row["execution_mode"] for row in group.rows) == REFERENCE_EXECUTION_MODES
    )


def test_default_reference_results_path_prefers_powered_results():
    assert default_reference_results_path().name == "results_all_power.csv"


def test_resolve_sampling_uses_100_timed_iterations_for_short_sequences():
    group = group_reference_rows([_reference_row()])[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 100


def test_resolve_sampling_uses_10_timed_iterations_for_8192_and_16384_sequences():
    group = group_reference_rows([_reference_row(seq_len="8192")])[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 10


def test_resolve_sampling_keeps_4096_at_5_timed_iterations():
    group = group_reference_rows([_reference_row(seq_len="4096")])[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 5


def test_rocm_power_policy_targets_at_least_twenty_samples():
    (
        min_measurement_duration_sec,
        min_sample_count,
        min_interval_sec,
        target_sample_count,
    ) = resolve_power_sampling_policy(power_backend="rocm-smi", seq_len=512)
    avg_iteration_sec = 0.0003
    power_probe_runs = resolve_power_probe_runs(
        avg_iteration_sec=avg_iteration_sec,
        baseline_runs=100,
        min_measurement_duration_sec=min_measurement_duration_sec,
    )
    estimated_window_sec = avg_iteration_sec * float(power_probe_runs)
    sample_interval_sec = resolve_power_sample_interval_sec(
        requested_interval_sec=0.2,
        estimated_timed_window_sec=estimated_window_sec,
        min_sample_count=target_sample_count,
        min_interval_sec=min_interval_sec,
    )

    assert min_measurement_duration_sec >= 2.0
    assert min_sample_count >= 16
    assert target_sample_count >= 24
    assert estimated_window_sec / sample_interval_sec >= float(target_sample_count)


def test_rocm_long_sequence_power_policy_extends_probe_window():
    (
        min_measurement_duration_sec,
        min_sample_count,
        min_interval_sec,
        target_sample_count,
    ) = resolve_power_sampling_policy(power_backend="rocm-smi", seq_len=8192)

    assert min_measurement_duration_sec >= 3.0
    assert min_sample_count >= 20
    assert min_interval_sec == 0.05
    assert target_sample_count >= 30


def test_turbostat_long_sequence_power_policy_extends_probe_window():
    (
        min_measurement_duration_sec,
        min_sample_count,
        min_interval_sec,
        target_sample_count,
    ) = resolve_power_sampling_policy(
        power_backend="turbostat_pkgwatt",
        seq_len=8192,
    )

    assert min_measurement_duration_sec >= 3.0
    assert min_sample_count >= 20
    assert min_interval_sec == 0.1
    assert target_sample_count >= 24


def test_rocm_smi_power_monitor_stats_filter_outliers_when_variance_drops():
    monitor = RocmSMIPowerMonitor()
    monitor.samples_w = [10.0, 10.1, 9.9, 10.2, 10.0, 10.1, 9.8, 10.0, 10.2, 22.0]

    stats = monitor.stats(elapsed_sec=2.0)

    assert stats["power_outlier_filter_applied"] is True
    assert stats["power_outlier_sample_count"] == 1
    assert stats["raw_power_sample_count"] == 10
    assert stats["power_sample_count"] == 9
    assert float(stats["raw_avg_power_w"]) > float(stats["avg_power_w"])
    assert float(stats["raw_power_std_w"]) > float(stats["power_std_w"])


def test_rocm_smi_power_monitor_records_sample_failures():
    monitor = RocmSMIPowerMonitor()

    monitor._record_sample_failure("sample parse failed")
    monitor._record_sample_failure("ignored")

    assert monitor.sample_error_count() == 2
    assert (
        monitor.failure_summary() == "sample parse failed (2 sampling failures total)"
    )


def test_configure_cpu_runtime_for_max_physical_cores(monkeypatch):
    recorded: dict[str, int] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run._physical_core_count_for_current_affinity",
        lambda: 12,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run.torch.set_num_threads",
        lambda value: recorded.setdefault("set_num_threads", int(value)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run.torch.get_num_threads",
        lambda: 12,
    )

    assert configure_cpu_runtime_for_max_physical_cores() == 12
    assert recorded["set_num_threads"] == 12


def test_build_rows_for_group_aggregates_igpu_and_npu_reference_modes(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row(
                effective_gflops_per_sec="100.0",
                effective_gflops_per_sec_per_watt="10.0",
            ),
            _reference_row(
                execution_mode="runlist",
                avg_latency_ms="7.0",
                effective_gflops_per_sec="90.0",
                effective_gflops_per_sec_per_watt="9.0",
            ),
            _reference_row(
                execution_mode="offload",
                avg_latency_ms="6.0",
                effective_gflops_per_sec="120.0",
                effective_gflops_per_sec_per_watt="12.0",
            ),
        ]
    )[0]

    def _benchmark(*args, **kwargs):
        return {
            "effective_gflops_per_sec": 800.0,
            "effective_gflops_per_sec_per_watt": 80.0,
            "avg_latency_ms": 4.0,
            "min_latency_ms": 3.5,
            "max_latency_ms": 4.5,
            "latency_sample_count": 100,
            "power_backend": "rocm-smi",
            "avg_power_w": 10.0,
            "min_power_w": 9.0,
            "max_power_w": 12.0,
            "raw_avg_power_w": 11.0,
            "raw_min_power_w": 9.0,
            "raw_max_power_w": 20.0,
            "raw_power_sample_count": 10,
            "power_std_w": 0.8,
            "raw_power_std_w": 3.1,
            "power_outlier_sample_count": 1,
            "energy_j": 5.0,
            "power_sample_count": 7,
            "extra_power_stats": {},
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run.benchmark_host_group",
        _benchmark,
    )

    rows = build_rows_for_group(
        group,
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        host_backends=("igpu",),
        igpu_device_name="cuda:0",
        igpu_power_backend="none",
        igpu_power_sample_interval_sec=0.2,
    )

    expected_rows = [
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec",
            "igpu": 800.0,
            "igpu_rocm_smi": None,
            "hybrid": 100.0,
            "runlist": 90.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "avg_latency_ms",
            "igpu": 4.0,
            "igpu_rocm_smi": None,
            "hybrid": 5.0,
            "runlist": 7.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "min_latency_ms",
            "igpu": 3.5,
            "igpu_rocm_smi": None,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "max_latency_ms",
            "igpu": 4.5,
            "igpu_rocm_smi": None,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "latency_sample_count",
            "igpu": 100,
            "igpu_rocm_smi": None,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": None,
            "igpu_rocm_smi": 80.0,
            "hybrid": 10.0,
            "runlist": 9.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "avg_power_w",
            "igpu": None,
            "igpu_rocm_smi": 10.0,
            "hybrid": 12.0,
            "runlist": 12.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "min_power_w",
            "igpu": None,
            "igpu_rocm_smi": 9.0,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "max_power_w",
            "igpu": None,
            "igpu_rocm_smi": 12.0,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "power_sample_count",
            "igpu": None,
            "igpu_rocm_smi": 7,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "raw_avg_power_w",
            "igpu": None,
            "igpu_rocm_smi": 11.0,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "raw_min_power_w",
            "igpu": None,
            "igpu_rocm_smi": 9.0,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "raw_max_power_w",
            "igpu": None,
            "igpu_rocm_smi": 20.0,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "raw_power_sample_count",
            "igpu": None,
            "igpu_rocm_smi": 10,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "power_std_w",
            "igpu": None,
            "igpu_rocm_smi": 0.8,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "raw_power_std_w",
            "igpu": None,
            "igpu_rocm_smi": 3.1,
            "hybrid": None,
            "runlist": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "power_outlier_sample_count",
            "igpu": None,
            "igpu_rocm_smi": 1,
            "hybrid": None,
            "runlist": None,
        },
    ]
    mode_values = {
        "runlist": {
            "effective_gflops_per_sec": 90.0,
            "avg_latency_ms": 7.0,
            "effective_gflops_per_sec_per_watt": 9.0,
            "avg_power_w": 12.0,
        },
        "offload": {
            "effective_gflops_per_sec": 120.0,
            "avg_latency_ms": 6.0,
            "effective_gflops_per_sec_per_watt": 12.0,
            "avg_power_w": 12.0,
        },
    }
    for row in expected_rows:
        row["runlist"] = mode_values["runlist"].get(str(row["metric"]))
        row["offload"] = mode_values["offload"].get(str(row["metric"]))

    assert rows == expected_rows


def test_build_rows_for_group_blanks_missing_igpu_backend(monkeypatch):
    group = group_reference_rows([_reference_row()])[0]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run.benchmark_host_group",
        lambda *args, **kwargs: {
            "effective_gflops_per_sec": 300.0,
            "effective_gflops_per_sec_per_watt": 30.0,
            "avg_latency_ms": 1.0,
            "min_latency_ms": 0.9,
            "max_latency_ms": 1.1,
            "latency_sample_count": 2,
            "power_backend": "rocm-smi",
            "avg_power_w": 20.0,
            "min_power_w": 19.0,
            "max_power_w": 21.0,
            "power_sample_count": 3,
            "extra_power_stats": {},
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_rows_for_group(
        group,
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        host_backends=tuple(),
        igpu_device_name="cuda:0",
        igpu_power_backend="none",
        igpu_power_sample_interval_sec=0.2,
    )

    assert rows[0]["igpu"] is None


def test_build_rows_for_group_reuses_matching_existing_rows(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row(),
            _reference_row(
                execution_mode="runlist",
                avg_latency_ms="7.0",
                effective_gflops_per_sec="3200.0",
                effective_gflops_per_sec_per_watt="266.7",
            ),
            _reference_row(
                execution_mode="offload",
                avg_latency_ms="6.0",
                effective_gflops_per_sec="6400.0",
                effective_gflops_per_sec_per_watt="533.3",
            ),
        ]
    )[0]

    call_count = {"benchmark_host_group": 0}

    def record_benchmark(*args, **kwargs):
        call_count["benchmark_host_group"] += 1
        return {
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.run.benchmark_host_group",
        record_benchmark,
    )

    existing_rows = {
        ("encoder_bert", "tinybert_512", 64, "effective_gflops_per_sec"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec",
            "igpu": "800.0",
            "igpu_rocm_smi": "",
            "hybrid": "1.0",
            "runlist": "2.0",
        },
        (
            "encoder_bert",
            "tinybert_512",
            64,
            "effective_gflops_per_sec_per_watt",
        ): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": "",
            "igpu_rocm_smi": "80.0",
            "hybrid": "3.0",
            "runlist": "4.0",
        },
        ("encoder_bert", "tinybert_512", 64, "avg_latency_ms"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_latency_ms",
            "igpu": "4.0",
            "igpu_rocm_smi": "",
            "hybrid": "5.0",
            "runlist": "7.0",
        },
        ("encoder_bert", "tinybert_512", 64, "min_latency_ms"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_latency_ms",
            "igpu": "3.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "max_latency_ms"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_latency_ms",
            "igpu": "4.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "latency_sample_count"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "latency_sample_count",
            "igpu": "100",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "avg_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_power_w",
            "igpu": "",
            "igpu_rocm_smi": "10.0",
            "hybrid": "30.0",
            "runlist": "40.0",
        },
        ("encoder_bert", "tinybert_512", 64, "min_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_power_w",
            "igpu": "",
            "igpu_rocm_smi": "9.0",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "max_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_power_w",
            "igpu": "",
            "igpu_rocm_smi": "12.0",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "power_sample_count"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "7",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "raw_avg_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_avg_power_w",
            "igpu": "",
            "igpu_rocm_smi": "11.0",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "raw_min_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_min_power_w",
            "igpu": "",
            "igpu_rocm_smi": "9.0",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "raw_max_power_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_max_power_w",
            "igpu": "",
            "igpu_rocm_smi": "20.0",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "raw_power_sample_count"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_power_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "10",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "power_std_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_std_w",
            "igpu": "",
            "igpu_rocm_smi": "0.8",
            "hybrid": "",
            "runlist": "",
        },
        ("encoder_bert", "tinybert_512", 64, "raw_power_std_w"): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_power_std_w",
            "igpu": "",
            "igpu_rocm_smi": "3.1",
            "hybrid": "",
            "runlist": "",
        },
        (
            "encoder_bert",
            "tinybert_512",
            64,
            "power_outlier_sample_count",
        ): {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_outlier_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "1",
            "hybrid": "",
            "runlist": "",
        },
    }
    mode_values = {
        "runlist": {
            "effective_gflops_per_sec": "3200.0",
            "effective_gflops_per_sec_per_watt": "266.7",
            "avg_latency_ms": "7.0",
            "avg_power_w": "12.0",
        },
        "offload": {
            "effective_gflops_per_sec": "6400.0",
            "effective_gflops_per_sec_per_watt": "533.3",
            "avg_latency_ms": "6.0",
            "avg_power_w": "12.0",
        },
    }
    for existing_row in existing_rows.values():
        existing_row["offload"] = mode_values["offload"].get(
            str(existing_row["metric"]), ""
        )

    rows = build_rows_for_group(
        group,
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        host_backends=("igpu",),
        igpu_device_name="cuda:0",
        igpu_power_backend="rocm-smi",
        igpu_power_sample_interval_sec=0.2,
        existing_rows=existing_rows,
    )
    assert call_count["benchmark_host_group"] == 0
    expected_rows = [
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec",
            "igpu": "800.0",
            "igpu_rocm_smi": "",
            "hybrid": "12800.0",
            "runlist": "3200.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": "",
            "igpu_rocm_smi": "80.0",
            "hybrid": "1066.7",
            "runlist": "266.7",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_latency_ms",
            "igpu": "4.0",
            "igpu_rocm_smi": "",
            "hybrid": "5.0",
            "runlist": "7.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_latency_ms",
            "igpu": "3.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_latency_ms",
            "igpu": "4.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "latency_sample_count",
            "igpu": "100",
            "igpu_rocm_smi": "",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_power_w",
            "igpu": "",
            "igpu_rocm_smi": "10.0",
            "hybrid": "12.0",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_power_w",
            "igpu": "",
            "igpu_rocm_smi": "9.0",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_power_w",
            "igpu": "",
            "igpu_rocm_smi": "12.0",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "7",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_avg_power_w",
            "igpu": "",
            "igpu_rocm_smi": "11.0",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_min_power_w",
            "igpu": "",
            "igpu_rocm_smi": "9.0",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_max_power_w",
            "igpu": "",
            "igpu_rocm_smi": "20.0",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_power_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "10",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_std_w",
            "igpu": "",
            "igpu_rocm_smi": "0.8",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "raw_power_std_w",
            "igpu": "",
            "igpu_rocm_smi": "3.1",
            "hybrid": "",
            "runlist": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_outlier_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "1",
            "hybrid": "",
            "runlist": "",
        },
    ]
    for expected_row in expected_rows:
        expected_row["runlist"] = mode_values["runlist"].get(
            str(expected_row["metric"]), ""
        )
        expected_row["offload"] = mode_values["offload"].get(
            str(expected_row["metric"]), ""
        )

    def normalize_row_values(row: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in row.items():
            if value in ("", None):
                normalized[key] = None
                continue
            try:
                normalized[key] = float(str(value))
            except ValueError:
                normalized[key] = str(value)
        return normalized

    comparable_rows = {str(row["metric"]): normalize_row_values(row) for row in rows}
    expected_rows_by_metric = {
        str(row["metric"]): normalize_row_values(row) for row in expected_rows
    }
    assert comparable_rows == expected_rows_by_metric


def test_build_fairness_rows_emits_igpu_metadata():
    rows = build_fairness_rows(
        igpu_device="cuda:0",
        igpu_power_backend="rocm-smi",
    )

    assert [row["backend"] for row in rows] == ["igpu"]
    assert all(
        json.loads(row["reference_execution_modes_json"])
        == ["hybrid", "runlist", "offload"]
        for row in rows
    )
    schedule = json.loads(rows[0]["iteration_schedule_json"])
    assert schedule["8192-16384"]["runs_per_sample"] == 10


def test_normalized_existing_row_keeps_current_schema():
    normalized = _normalized_existing_row(
        {
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec",
            "igpu": "800.0",
            "hybrid": "12800.0",
        }
    )

    assert normalized == {
        "workload_variant": "encoder_bert",
        "study_case_id": "tinybert_512",
        "seq_len": 64,
        "metric": "effective_gflops_per_sec",
        "igpu": "800.0",
        "igpu_rocm_smi": "",
        "hybrid": "12800.0",
        "runlist": "",
        "offload": "",
    }


def test_load_existing_rows_repairs_missing_igpu_per_watt_from_existing_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "workload_variant,study_case_id,seq_len,metric,igpu,igpu_rocm_smi,hybrid,runlist,offload",
                "encoder_bert,tinybert_512,64,effective_gflops_per_sec,800.0,,100.0,90.0,120.0",
                "encoder_bert,tinybert_512,64,effective_gflops_per_sec_per_watt,,,10.0,9.0,12.0",
                "encoder_bert,tinybert_512,64,avg_power_w,,10.0,12.0,12.0,12.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_existing_rows((csv_path,))

    repaired_row = rows[
        ("encoder_bert", "tinybert_512", 64, "effective_gflops_per_sec_per_watt")
    ]
    assert repaired_row["igpu_rocm_smi"] == 80.0


def test_remeasure_power_only_fails_without_selected_reference_groups(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "results.csv"
    output_path.write_text(
        "\n".join(
            [
                "workload_variant,study_case_id,seq_len,metric,igpu,igpu_rocm_smi,hybrid,runlist,offload",
                "encoder_bert,baseline_768,64,effective_gflops_per_sec,700.0,,100.0,90.0,80.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    original_text = output_path.read_text(encoding="utf-8")

    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_reference_row().keys()))
        writer.writeheader()
        writer.writerow(_reference_row())

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.remeasure_power_only.write_plots",
        lambda *args, **kwargs: None,
    )

    assert (
        remeasure_power_only_main(
            [
                "--reference-input",
                str(reference_path),
                "--output",
                str(output_path),
                "--family",
                "baseline_768",
            ]
        )
        == 1
    )
    assert output_path.read_text(encoding="utf-8") == original_text


def test_remeasure_power_only_preserves_untargeted_existing_rows(tmp_path, monkeypatch):
    output_path = tmp_path / "results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "workload_variant": "encoder_bert",
                    "study_case_id": "tinybert_512",
                    "seq_len": "64",
                    "metric": "effective_gflops_per_sec",
                    "igpu": "800.0",
                    "igpu_rocm_smi": "",
                    "hybrid": "100.0",
                    "runlist": "90.0",
                    "offload": "80.0",
                },
                {
                    "workload_variant": "encoder_bert",
                    "study_case_id": "tinybert_512",
                    "seq_len": "64",
                    "metric": "avg_latency_ms",
                    "igpu": "4.0",
                    "igpu_rocm_smi": "",
                    "hybrid": "5.0",
                    "runlist": "6.0",
                    "offload": "7.0",
                },
                {
                    "workload_variant": "encoder_bert",
                    "study_case_id": "tinybert_512",
                    "seq_len": "64",
                    "metric": "avg_power_w",
                    "igpu": "",
                    "igpu_rocm_smi": "10.0",
                    "hybrid": "12.0",
                    "runlist": "12.0",
                    "offload": "12.0",
                },
                {
                    "workload_variant": "encoder_bert",
                    "study_case_id": "baseline_768",
                    "seq_len": "64",
                    "metric": "effective_gflops_per_sec",
                    "igpu": "700.0",
                    "igpu_rocm_smi": "",
                    "hybrid": "200.0",
                    "runlist": "180.0",
                    "offload": "160.0",
                },
            ]
        )

    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_reference_row().keys()))
        writer.writeheader()
        writer.writerow(_reference_row())

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.remeasure_power_only.write_plots",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.host_comparison.remeasure_power_only.benchmark_host_group_power_only",
        lambda *args, **kwargs: {
            "run_status": "passed",
            "failure_message": "",
            "avg_power_w": "15.0",
            "min_power_w": "14.0",
            "max_power_w": "16.0",
            "power_sample_count": "9",
            "raw_avg_power_w": "15.5",
            "raw_min_power_w": "14.0",
            "raw_max_power_w": "17.0",
            "raw_power_sample_count": "10",
            "power_std_w": "0.5",
            "raw_power_std_w": "0.7",
            "power_outlier_sample_count": "1",
            "effective_gflops_per_sec_per_watt": "53.3",
        },
    )

    assert (
        remeasure_power_only_main(
            [
                "--reference-input",
                str(reference_path),
                "--output",
                str(output_path),
                "--family",
                "tinybert_512",
            ]
        )
        == 0
    )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    baseline_row = next(
        row
        for row in rows
        if row["study_case_id"] == "baseline_768"
        and row["metric"] == "effective_gflops_per_sec"
    )
    assert baseline_row["igpu"] == "700.0"
    tinybert_power_row = next(
        row
        for row in rows
        if row["study_case_id"] == "tinybert_512" and row["metric"] == "avg_power_w"
    )
    assert tinybert_power_row["igpu_rocm_smi"] == "15.0"


def test_decoder_forward_reference_matches_shared_transformer_reference():
    reference = generate_synthetic_reference(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        workload_variant="decoder_gpt2",
        dtype="bf16",
        seed=42,
        include_output=True,
    )

    output = _forward_reference(
        reference["input"],
        reference["weights"],
        num_attention_heads=12,
        workload_variant="decoder_gpt2",
    )

    assert output.shape == reference["output"].shape
    assert output.dtype == reference["output"].dtype
    assert torch.equal(output, reference["output"])


def test_decoder_forward_reference_accepts_precomputed_causal_mask():
    reference = generate_synthetic_reference(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        workload_variant="decoder_gpt2",
        dtype="bf16",
        seed=42,
        include_output=True,
    )
    causal_mask = torch.triu(
        torch.ones((64, 64), dtype=torch.bool),
        diagonal=1,
    )

    output = _forward_reference(
        reference["input"],
        reference["weights"],
        num_attention_heads=12,
        workload_variant="decoder_gpt2",
        causal_mask=causal_mask,
    )

    assert output.shape == reference["output"].shape
    assert output.dtype == reference["output"].dtype
    assert torch.equal(output, reference["output"])
