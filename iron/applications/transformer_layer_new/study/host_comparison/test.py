#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import torch

from iron.applications.transformer_layer_new.study.host_comparison.run import (
    _forward_reference,
    _normalized_existing_row,
    build_rows_for_group,
    configure_cpu_runtime_for_max_physical_cores,
    generate_synthetic_reference,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
    resolve_power_sampling_policy,
    resolve_sampling,
)
from iron.applications.transformer_layer_new.study.host_comparison.run_fairness_repeatability import (
    build_rows as build_fairness_rows,
)
from iron.applications.transformer_layer_new.study.host_comparison.select import (
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


def test_group_reference_rows_keeps_only_passing_hybrid_npu_rows():
    rows = [
        _reference_row(),
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


def test_resolve_sampling_uses_5_timed_iterations_for_long_sequences():
    group = group_reference_rows([_reference_row(seq_len="8192")])[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 1
    assert runs_per_sample == 5


def test_rocm_power_policy_targets_at_least_ten_samples():
    (
        min_measurement_duration_sec,
        min_interval_sec,
        target_sample_count,
    ) = resolve_power_sampling_policy(power_backend="rocm-smi")
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

    assert min_measurement_duration_sec >= 1.2
    assert target_sample_count >= 10
    assert estimated_window_sec / sample_interval_sec >= float(target_sample_count)


def test_configure_cpu_runtime_for_max_physical_cores(monkeypatch):
    recorded: dict[str, int] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run._physical_core_count_for_current_affinity",
        lambda: 12,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.torch.set_num_threads",
        lambda value: recorded.setdefault("set_num_threads", int(value)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.torch.get_num_threads",
        lambda: 12,
    )

    assert configure_cpu_runtime_for_max_physical_cores() == 12
    assert recorded["set_num_threads"] == 12


def test_build_rows_for_group_aggregates_igpu_and_hybrid(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row(
                effective_gflops_per_sec="100.0",
                effective_gflops_per_sec_per_watt="10.0",
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
            "energy_j": 5.0,
            "power_sample_count": 7,
            "extra_power_stats": {},
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.benchmark_host_group",
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

    assert rows == [
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec",
            "igpu": 800.0,
            "igpu_rocm_smi": None,
            "hybrid": 100.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "avg_latency_ms",
            "igpu": 4.0,
            "igpu_rocm_smi": None,
            "hybrid": 5.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "min_latency_ms",
            "igpu": 3.5,
            "igpu_rocm_smi": None,
            "hybrid": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "max_latency_ms",
            "igpu": 4.5,
            "igpu_rocm_smi": None,
            "hybrid": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "latency_sample_count",
            "igpu": 100,
            "igpu_rocm_smi": None,
            "hybrid": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": None,
            "igpu_rocm_smi": 80.0,
            "hybrid": 10.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "avg_power_w",
            "igpu": None,
            "igpu_rocm_smi": 10.0,
            "hybrid": 12.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "min_power_w",
            "igpu": None,
            "igpu_rocm_smi": 9.0,
            "hybrid": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "max_power_w",
            "igpu": None,
            "igpu_rocm_smi": 12.0,
            "hybrid": None,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "power_sample_count",
            "igpu": None,
            "igpu_rocm_smi": 7,
            "hybrid": None,
        },
    ]


def test_build_rows_for_group_blanks_missing_igpu_backend(monkeypatch):
    group = group_reference_rows([_reference_row()])[0]

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.benchmark_host_group",
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
    group = group_reference_rows([_reference_row()])[0]

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark_host_group should not be called")

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.benchmark_host_group",
        fail_benchmark,
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
        existing_rows={
            ("encoder_bert", "tinybert_512", 64, "effective_gflops_per_sec"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "effective_gflops_per_sec",
                "igpu": "800.0",
                "igpu_rocm_smi": "",
                "hybrid": "12800.0",
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
                "hybrid": "1066.7",
            },
            ("encoder_bert", "tinybert_512", 64, "avg_latency_ms"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "avg_latency_ms",
                "igpu": "4.0",
                "igpu_rocm_smi": "",
                "hybrid": "5.0",
            },
            ("encoder_bert", "tinybert_512", 64, "min_latency_ms"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "min_latency_ms",
                "igpu": "3.5",
                "igpu_rocm_smi": "",
                "hybrid": "",
            },
            ("encoder_bert", "tinybert_512", 64, "max_latency_ms"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "max_latency_ms",
                "igpu": "4.5",
                "igpu_rocm_smi": "",
                "hybrid": "",
            },
            ("encoder_bert", "tinybert_512", 64, "latency_sample_count"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "latency_sample_count",
                "igpu": "100",
                "igpu_rocm_smi": "",
                "hybrid": "",
            },
            ("encoder_bert", "tinybert_512", 64, "avg_power_w"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "avg_power_w",
                "igpu": "",
                "igpu_rocm_smi": "10.0",
                "hybrid": "12.0",
            },
            ("encoder_bert", "tinybert_512", 64, "min_power_w"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "min_power_w",
                "igpu": "",
                "igpu_rocm_smi": "9.0",
                "hybrid": "",
            },
            ("encoder_bert", "tinybert_512", 64, "max_power_w"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "max_power_w",
                "igpu": "",
                "igpu_rocm_smi": "12.0",
                "hybrid": "",
            },
            ("encoder_bert", "tinybert_512", 64, "power_sample_count"): {
                "workload_variant": "encoder_bert",
                "study_case_id": "tinybert_512",
                "seq_len": "64",
                "metric": "power_sample_count",
                "igpu": "",
                "igpu_rocm_smi": "7",
                "hybrid": "",
            },
        },
    )

    assert rows == [
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec",
            "igpu": "800.0",
            "igpu_rocm_smi": "",
            "hybrid": "12800.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": "",
            "igpu_rocm_smi": "80.0",
            "hybrid": "1066.7",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_latency_ms",
            "igpu": "4.0",
            "igpu_rocm_smi": "",
            "hybrid": "5.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_latency_ms",
            "igpu": "3.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_latency_ms",
            "igpu": "4.5",
            "igpu_rocm_smi": "",
            "hybrid": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "latency_sample_count",
            "igpu": "100",
            "igpu_rocm_smi": "",
            "hybrid": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "avg_power_w",
            "igpu": "",
            "igpu_rocm_smi": "10.0",
            "hybrid": "12.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "min_power_w",
            "igpu": "",
            "igpu_rocm_smi": "9.0",
            "hybrid": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "max_power_w",
            "igpu": "",
            "igpu_rocm_smi": "12.0",
            "hybrid": "",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "power_sample_count",
            "igpu": "",
            "igpu_rocm_smi": "7",
            "hybrid": "",
        },
    ]


def test_build_fairness_rows_emits_igpu_metadata():
    rows = build_fairness_rows(
        igpu_device="cuda:0",
        igpu_power_backend="rocm-smi",
    )

    assert [row["backend"] for row in rows] == ["igpu"]
    assert all(
        json.loads(row["reference_execution_modes_json"]) == ["hybrid"] for row in rows
    )
    schedule = json.loads(rows[0]["iteration_schedule_json"])
    assert schedule["8192-16384"]["runs_per_sample"] == 5


def test_normalized_existing_row_upgrades_legacy_schema():
    normalized = _normalized_existing_row(
        {
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec",
            "igpu": "800.0",
            "dataflow": "12800.0",
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
    }


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
