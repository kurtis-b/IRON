#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

from iron.applications.transformer_layer_new.study.reconfiguration_overhead.modes import (
    dispatch_count_for_mode,
    resolve_mode_operator_config,
)
from iron.applications.transformer_layer_new.study.reconfiguration_overhead.power import (
    parse_turbostat_pkgwatt_samples,
    resolve_requested_power_backend,
)
from iron.applications.transformer_layer_new.study.reconfiguration_overhead.plot_latency_comparison import (
    render_plot,
)
from iron.applications.transformer_layer_new.study.reconfiguration_overhead.run import (
    main,
)
from iron.applications.transformer_layer_new.study.reconfiguration_overhead.select import (
    select_reference_rows,
)
from iron.applications.transformer_layer_new.study.reconfiguration_overhead.cases import (
    ReconfigurationWorkload,
)


def _offload_source_config() -> dict[str, dict[str, object]]:
    return {
        "shared_gemm": {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 8,
            "b_col_maj": False,
            "c_col_maj": False,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
    }


def _runlist_source_config() -> dict[str, dict[str, object]]:
    return {
        "qkvo_proj": {
            "M": 256,
            "K": 768,
            "N": 768,
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "k_transpose": {"ignored": True},
        "attn_scores": {
            "M": 256,
            "K": 64,
            "N": 256,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "num_aie_columns": 8,
            "batch_A": [12, 1],
            "batch_B": [12, 1],
            "batch_C": [12, 0],
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "attn_output": {
            "M": 256,
            "K": 256,
            "N": 64,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 4,
            "batch_A": [12, 0],
            "batch_B": [12, 1],
            "batch_C": [12, 1],
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "up_proj": {
            "M": 256,
            "K": 768,
            "N": 3072,
            "tile_m": 64,
            "tile_k": 48,
            "tile_n": 96,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "gelu": {"ignored": True},
        "down_proj": {
            "M": 256,
            "K": 3072,
            "N": 768,
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
    }


def _reference_row(
    execution_mode: str,
    *,
    avg_latency_ms: str = "5.0",
    run_status: str = "passed",
    study_case_id: str = "baseline_768",
    seq_len: str = "256",
    selected_config_json: str | None = None,
    selected_candidate_ids_json: str | None = None,
) -> dict[str, str]:
    if selected_config_json is None:
        selected_config_json = json.dumps(
            (
                _offload_source_config()
                if execution_mode == "offload"
                else _runlist_source_config()
            ),
            sort_keys=True,
        )
    if selected_candidate_ids_json is None:
        selected_candidate_ids_json = json.dumps(
            (
                {"shared_gemm": "shared_0"}
                if execution_mode == "offload"
                else {
                    "qkvo_proj": "qkvo_0",
                    "attn_scores": "attn_scores_0",
                    "attn_output": "attn_output_0",
                    "up_proj": "up_0",
                    "down_proj": "down_0",
                    "gelu": "ignored",
                }
            ),
            sort_keys=True,
        )
    return {
        "study_id": "end_to_end",
        "study_case_id": study_case_id,
        "study_case_label": study_case_id,
        "backend": "npu",
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
        "runs_per_sample": "10",
        "measured_inference_count": "10",
        "timed_total_sec": "0.5",
        "avg_latency_ms": avg_latency_ms,
        "compile_setup_time_ms": "12.0",
        "host_qkv_precompute_ms": "0.0",
        "tokens_per_sec": "512.0",
        "power_backend": "none",
        "avg_power_w": "0.0",
        "tokens_per_sec_per_watt": "0.0",
        "npu_dispatch_count": "0",
        "npu_unique_instruction_binary_count": "0",
        "npu_unique_xclbin_count": "0",
        "process_model": "in_process",
        "validation_error_count": "0",
        "run_status": run_status,
        "failure_message": "",
        "selected_candidate_ids_json": selected_candidate_ids_json,
        "selected_config_json": selected_config_json,
        "is_best": "False",
    }


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reconfig_result_row(
    *,
    family: str,
    seq_len: int,
    execution_mode: str,
    avg_latency_ms: float,
    npu_unique_xclbin_count: int,
) -> dict[str, str]:
    return {
        "study_id": "reconfiguration_overhead",
        "study_case_id": family,
        "study_case_label": family,
        "backend": "npu",
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "source_end_to_end_execution_mode": (
            "offload" if execution_mode == "offload_gemm_sequence" else "runlist"
        ),
        "seq_len": str(seq_len),
        "hidden_size": "768" if family == "baseline_768" else "1024",
        "intermediate_size": "3072" if family == "baseline_768" else "4096",
        "num_attention_heads": "12" if family == "baseline_768" else "16",
        "attention_head_size": "64",
        "batch_size": "1",
        "dtype": "bf16",
        "use_bias": "False",
        "weights_source": "synthetic",
        "warmup_runs": "1",
        "runs_per_sample": "2",
        "measured_inference_count": "2",
        "timed_total_sec": "0.1",
        "avg_latency_ms": str(avg_latency_ms),
        "compile_setup_time_ms": "0.0",
        "power_backend": "none",
        "avg_power_w": "0.0",
        "max_power_w": "0.0",
        "energy_j": "0.0",
        "power_sample_count": "0",
        "npu_dispatch_count": "0",
        "npu_unique_instruction_binary_count": "0",
        "npu_unique_xclbin_count": str(npu_unique_xclbin_count),
        "process_model": "in_process",
        "validation_error_count": "0",
        "run_status": "passed",
        "failure_message": "",
        "selected_candidate_ids_json": "{}",
        "selected_config_json": "{}",
    }


def test_select_reference_rows_filters_modes_and_extracts_gemm_configs():
    rows = [
        _reference_row("offload"),
        _reference_row("runlist"),
        _reference_row("dataflow"),
        _reference_row("offload", run_status="failed_validation"),
        _reference_row("runlist", selected_config_json=json.dumps({}, sort_keys=True)),
    ]

    selections = select_reference_rows(rows)

    assert [selection.execution_mode for selection in selections] == [
        "offload_gemm_sequence",
        "runlist_gemm_sequence",
    ]
    assert tuple(selections[0].selected_config.keys()) == ("shared_gemm",)
    assert tuple(selections[1].selected_config.keys()) == (
        "qkvo_proj",
        "attn_scores",
        "attn_output",
        "up_proj",
        "down_proj",
    )


def test_render_plot_writes_outputs(tmp_path):
    rows = [
        _reconfig_result_row(
            family="baseline_768",
            seq_len=256,
            execution_mode="offload_gemm_sequence",
            avg_latency_ms=8.0,
            npu_unique_xclbin_count=1,
        ),
        _reconfig_result_row(
            family="baseline_768",
            seq_len=256,
            execution_mode="runlist_gemm_sequence",
            avg_latency_ms=16.0,
            npu_unique_xclbin_count=5,
        ),
        _reconfig_result_row(
            family="baseline_1024",
            seq_len=256,
            execution_mode="offload_gemm_sequence",
            avg_latency_ms=12.0,
            npu_unique_xclbin_count=1,
        ),
        _reconfig_result_row(
            family="baseline_1024",
            seq_len=256,
            execution_mode="runlist_gemm_sequence",
            avg_latency_ms=18.0,
            npu_unique_xclbin_count=5,
        ),
    ]

    svg_path = tmp_path / "reconfig.svg"
    png_path = tmp_path / "reconfig.png"
    render_plot(rows, output_svg_path=svg_path, output_png_path=png_path)

    assert svg_path.exists()
    assert png_path.exists()
    svg = svg_path.read_text(encoding="utf-8")
    assert "Reconfiguration Overhead: Runlist vs Offload" in svg
    assert "Offload (1 xclbin)" in svg
    assert "Runlist (5 xclbins)" in svg


def test_select_reference_rows_chooses_fastest_duplicate():
    rows = [
        _reference_row("runlist", avg_latency_ms="8.0"),
        _reference_row("runlist", avg_latency_ms="4.0"),
    ]

    selections = select_reference_rows(rows)

    assert len(selections) == 1
    assert selections[0].selected_candidate_ids["qkvo_proj"] == "qkvo_0"


def test_resolve_mode_operator_config_uses_blocked_shapes_for_long_seq():
    workload = ReconfigurationWorkload(
        seq_len=16384,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
    )

    offload_config = resolve_mode_operator_config(
        "offload_gemm_sequence",
        workload,
        {
            "shared_gemm": _offload_source_config()["shared_gemm"],
        },
    )
    runlist_config = resolve_mode_operator_config(
        "runlist_gemm_sequence",
        workload,
        {
            "qkvo_proj": {
                "tile_m": 64,
                "tile_k": 96,
                "tile_n": 48,
                "num_aie_columns": 8,
            },
            "attn_scores": {
                "tile_m": 64,
                "tile_k": 64,
                "tile_n": 64,
                "num_aie_columns": 8,
            },
            "attn_output": {
                "tile_m": 64,
                "tile_k": 64,
                "tile_n": 16,
                "num_aie_columns": 4,
            },
            "up_proj": {
                "tile_m": 64,
                "tile_k": 48,
                "tile_n": 96,
                "num_aie_columns": 8,
            },
            "down_proj": {
                "tile_m": 64,
                "tile_k": 96,
                "tile_n": 48,
                "num_aie_columns": 8,
            },
        },
    )

    assert offload_config["q_proj"]["M"] == 256
    assert offload_config["attn_scores"]["partition_N"] == 4
    assert runlist_config["q_proj"]["M"] == 256
    assert runlist_config["attn_scores"]["N"] == 16384
    assert runlist_config["attn_scores"]["batch_A"] == (16, 1)
    assert runlist_config["attn_output"]["batch_C"] == (16, 1)
    assert runlist_config["ffn_down"]["K"] == 4096


def test_dispatch_count_for_mode_uses_expected_blocking():
    workload = ReconfigurationWorkload(
        seq_len=16384,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
    )

    assert dispatch_count_for_mode("offload_gemm_sequence", workload) == 2432
    assert dispatch_count_for_mode("runlist_gemm_sequence", workload) == 512


def test_main_skips_when_selected_configs_missing(tmp_path):
    reference_input = tmp_path / "end_to_end.csv"
    rows = [
        _reference_row("offload", selected_config_json=json.dumps({}, sort_keys=True)),
        _reference_row("runlist", selected_config_json=json.dumps({}, sort_keys=True)),
    ]
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_path = tmp_path / "results.csv"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--output",
            str(output_path),
            "--power-backend",
            "none",
        ]
    )

    assert exit_code == 0
    assert _read_csv_rows(output_path) == []


def test_main_writes_long_form_rows_from_available_reference_rows(
    monkeypatch, tmp_path
):
    reference_input = tmp_path / "end_to_end.csv"
    rows = [
        _reference_row("offload"),
        _reference_row("runlist"),
    ]
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def fake_benchmark_mode(
        execution_mode,
        workload,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
        operator_config,
        power_sample_interval_sec,
    ):
        return {
            "measured_inference_count": runs_per_sample,
            "timed_total_sec": 0.2,
            "avg_latency_ms": 2.0,
            "compile_setup_time_ms": 10.0,
            "power_backend": "none",
            "avg_power_w": None,
            "max_power_w": None,
            "energy_j": None,
            "power_sample_count": None,
            "npu_dispatch_count": 8,
            "npu_unique_instruction_binary_count": 5,
            "npu_unique_xclbin_count": 5,
            "process_model": "in_process",
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.reconfiguration_overhead.run.benchmark_mode",
        fake_benchmark_mode,
    )

    output_path = tmp_path / "results.csv"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--output",
            str(output_path),
            "--power-backend",
            "none",
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert len(rows) == 2
    assert {row["execution_mode"] for row in rows} == {
        "offload_gemm_sequence",
        "runlist_gemm_sequence",
    }
    assert all(row["warmup_runs"] == "1" for row in rows)
    assert all(row["runs_per_sample"] == "2" for row in rows)
    assert all(row["avg_latency_ms"] == "2.0" for row in rows)
    runlist_row = next(
        row for row in rows if row["execution_mode"] == "runlist_gemm_sequence"
    )
    resolved_config = json.loads(runlist_row["selected_config_json"])
    assert resolved_config["q_proj"]["M"] == 256
    assert resolved_config["attn_scores"]["N"] == 256


def test_main_filters_reference_rows_to_intended_sequence_ladder(monkeypatch, tmp_path):
    reference_input = tmp_path / "end_to_end.csv"
    rows = [
        _reference_row("offload", seq_len="64"),
        _reference_row("runlist", seq_len="64"),
        _reference_row("offload", seq_len="256"),
        _reference_row("runlist", seq_len="256"),
        _reference_row("offload", seq_len="2048"),
        _reference_row("runlist", seq_len="2048"),
        _reference_row("offload", seq_len="16384"),
        _reference_row("runlist", seq_len="16384"),
    ]
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def fake_benchmark_mode(
        execution_mode,
        workload,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
        operator_config,
        power_sample_interval_sec,
    ):
        del execution_mode, warmup_runs, runs_per_sample, seed, power_backend
        del operator_config, power_sample_interval_sec
        return {
            "measured_inference_count": 1,
            "timed_total_sec": 0.1,
            "avg_latency_ms": float(workload.seq_len),
            "compile_setup_time_ms": 1.0,
            "power_backend": "none",
            "avg_power_w": None,
            "max_power_w": None,
            "energy_j": None,
            "power_sample_count": None,
            "npu_dispatch_count": 8,
            "npu_unique_instruction_binary_count": 5,
            "npu_unique_xclbin_count": 5,
            "process_model": "in_process",
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.reconfiguration_overhead.run.benchmark_mode",
        fake_benchmark_mode,
    )

    output_path = tmp_path / "results.csv"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--output",
            str(output_path),
            "--power-backend",
            "none",
            "--warmup-iters",
            "1",
            "--timed-iters",
            "1",
        ]
    )

    assert exit_code == 0
    output_rows = _read_csv_rows(output_path)
    assert len(output_rows) == 6
    assert {row["seq_len"] for row in output_rows} == {"256", "2048", "16384"}


def test_parse_turbostat_pkgwatt_samples_reads_values():
    samples = parse_turbostat_pkgwatt_samples("PkgWatt\n12.5\n13.0\n")
    assert samples == [12.5, 13.0]


def test_resolve_requested_power_backend_auto_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.reconfiguration_overhead.power.shutil.which",
        lambda _: None,
    )
    assert resolve_requested_power_backend("auto") == "none"
