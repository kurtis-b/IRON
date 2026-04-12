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
            "power_backend": "rocm-smi",
            "extra_power_stats": {
                "turbostat_pkgwatt": {
                    "avg_power_w": 20.0,
                }
            },
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
            "igpu_turbostat_pkgwatt": None,
            "hybrid": 100.0,
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": 64,
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": None,
            "igpu_rocm_smi": 80.0,
            "igpu_turbostat_pkgwatt": 40.0,
            "hybrid": 10.0,
        },
    ]


def test_build_rows_for_group_blanks_missing_igpu_backend(monkeypatch):
    group = group_reference_rows([_reference_row()])[0]

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.host_comparison.run.benchmark_host_group",
        lambda *args, **kwargs: {
            "effective_gflops_per_sec": 300.0,
            "effective_gflops_per_sec_per_watt": 30.0,
            "power_backend": "rocm-smi",
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
                "igpu_turbostat_pkgwatt": "",
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
                "igpu_turbostat_pkgwatt": "40.0",
                "hybrid": "1066.7",
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
            "igpu_turbostat_pkgwatt": "",
            "hybrid": "12800.0",
        },
        {
            "workload_variant": "encoder_bert",
            "study_case_id": "tinybert_512",
            "seq_len": "64",
            "metric": "effective_gflops_per_sec_per_watt",
            "igpu": "",
            "igpu_rocm_smi": "80.0",
            "igpu_turbostat_pkgwatt": "40.0",
            "hybrid": "1066.7",
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
        "igpu_turbostat_pkgwatt": "",
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
