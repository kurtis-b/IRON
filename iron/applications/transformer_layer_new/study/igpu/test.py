#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

from iron.applications.transformer_layer_new.study.igpu.run import (
    build_rows_for_group,
    main,
    parse_rocm_smi_average_power_w,
    resolve_sampling,
)
from iron.applications.transformer_layer_new.study.igpu.select import (
    REFERENCE_EXECUTION_MODES,
    group_reference_rows,
)


def _reference_row(
    execution_mode: str,
    *,
    avg_latency_ms: str = "5.0",
    tokens_per_sec: str = "12800.0",
    tokens_per_sec_per_watt: str = "1066.7",
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
        "warmup_runs": "10",
        "runs_per_sample": "100",
        "measured_inference_count": "100",
        "timed_total_sec": "0.5",
        "avg_latency_ms": avg_latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "power_backend": "turbostat_pkgwatt",
        "avg_power_w": "12.0",
        "tokens_per_sec_per_watt": tokens_per_sec_per_watt,
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


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_group_reference_rows_keeps_only_three_passing_npu_patterns():
    rows = [
        _reference_row("dataflow", avg_latency_ms="3.0"),
        _reference_row("runlist", avg_latency_ms="4.0"),
        _reference_row("offload", avg_latency_ms="5.0"),
        _reference_row("dataflow", run_status="failed_validation", seq_len="128"),
        _reference_row("amd_igpu_reference", backend="gpu"),
    ]

    groups = group_reference_rows(rows)

    assert len(groups) == 1
    group = groups[0]
    assert group.study_case_id == "baseline_768"
    assert group.seq_len == 64
    assert (
        tuple(row["execution_mode"] for row in group.rows) == REFERENCE_EXECUTION_MODES
    )


def test_resolve_sampling_prefers_reference_schedule_without_override():
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
            _reference_row("offload"),
        ]
    )[0]

    warmup_runs, runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )

    assert warmup_runs == 10
    assert runs_per_sample == 100


def test_resolve_sampling_falls_back_when_reference_counts_are_zero():
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
            _reference_row("offload"),
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

    assert warmup_runs == 10
    assert runs_per_sample == 48


def test_build_rows_for_group_aggregates_all_reference_rows(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row(
                "dataflow",
                tokens_per_sec="100.0",
                tokens_per_sec_per_watt="10.0",
            ),
            _reference_row(
                "dataflow",
                tokens_per_sec="300.0",
                tokens_per_sec_per_watt="30.0",
            ),
            _reference_row(
                "runlist",
                tokens_per_sec="200.0",
                tokens_per_sec_per_watt="20.0",
            ),
            _reference_row(
                "offload",
                tokens_per_sec="400.0",
                tokens_per_sec_per_watt="40.0",
            ),
        ]
    )[0]

    def fake_benchmark_igpu_group(
        group,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        device_name,
        power_backend,
        power_sample_interval_sec,
    ):
        return {
            "tokens_per_sec": 800.0,
            "tokens_per_sec_per_watt": 80.0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.igpu.run.benchmark_igpu_group",
        fake_benchmark_igpu_group,
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
            "metric": "tps",
            "igpu": 800.0,
            "dataflow": 200.0,
            "runlist": 200.0,
            "offload": 400.0,
        },
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "metric": "tps_per_watt",
            "igpu": 80.0,
            "dataflow": 20.0,
            "runlist": 20.0,
            "offload": 40.0,
        },
    ]


def test_build_rows_for_group_blanks_igpu_values_on_gpu_failure(monkeypatch):
    group = group_reference_rows(
        [
            _reference_row("dataflow"),
            _reference_row("runlist"),
            _reference_row("offload"),
        ]
    )[0]

    def fake_benchmark_igpu_group(*args, **kwargs):
        raise RuntimeError("ROCm device unavailable")

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.igpu.run.benchmark_igpu_group",
        fake_benchmark_igpu_group,
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
    assert {row["metric"] for row in rows} == {"tps", "tps_per_watt"}
    assert all(row["igpu"] is None for row in rows)
    assert rows[0]["dataflow"] == 12800.0
    assert rows[1]["offload"] == 1066.7


def test_main_skips_when_reference_case_data_is_missing(tmp_path):
    reference_input = tmp_path / "end_to_end.csv"
    reference_input.write_text(
        ",".join(_reference_row("dataflow").keys())
        + "\n"
        + ",".join(_reference_row("dataflow").values())
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "igpu.csv"
    tps_plot_path = tmp_path / "tps.svg"
    tps_per_watt_plot_path = tmp_path / "tps_per_watt.svg"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--family",
            "baseline_1024",
            "--seq-len",
            "64",
            "--output",
            str(output_path),
            "--tps-plot",
            str(tps_plot_path),
            "--tps-per-watt-plot",
            str(tps_per_watt_plot_path),
            "--power-backend",
            "none",
        ]
    )

    assert exit_code == 0
    assert _read_csv_rows(output_path) == []
    assert "<svg" in tps_plot_path.read_text(encoding="utf-8")
    assert "No data available" in tps_plot_path.read_text(encoding="utf-8")
    assert "<svg" in tps_per_watt_plot_path.read_text(encoding="utf-8")


def test_main_writes_clean_csv_and_svg_plots(monkeypatch, tmp_path):
    reference_input = tmp_path / "end_to_end.csv"
    fieldnames = list(_reference_row("dataflow").keys())
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            _reference_row(
                "dataflow",
                tokens_per_sec="100.0",
                tokens_per_sec_per_watt="10.0",
            )
        )
        writer.writerow(
            _reference_row(
                "runlist",
                tokens_per_sec="200.0",
                tokens_per_sec_per_watt="20.0",
            )
        )
        writer.writerow(
            _reference_row(
                "offload",
                tokens_per_sec="300.0",
                tokens_per_sec_per_watt="30.0",
            )
        )

    def fake_benchmark_igpu_group(
        group,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        device_name,
        power_backend,
        power_sample_interval_sec,
    ):
        return {
            "tokens_per_sec": 32000.0,
            "tokens_per_sec_per_watt": 1600.0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.igpu.run.benchmark_igpu_group",
        fake_benchmark_igpu_group,
    )

    output_path = tmp_path / "igpu.csv"
    tps_plot_path = tmp_path / "tps.svg"
    tps_per_watt_plot_path = tmp_path / "tps_per_watt.svg"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--family",
            "baseline_768",
            "--seq-len",
            "64",
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
            "--output",
            str(output_path),
            "--tps-plot",
            str(tps_plot_path),
            "--tps-per-watt-plot",
            str(tps_per_watt_plot_path),
            "--power-backend",
            "none",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert rows == [
        {
            "study_case_id": "baseline_768",
            "seq_len": "64",
            "metric": "tps",
            "igpu": "32000.0",
            "dataflow": "100.0",
            "runlist": "200.0",
            "offload": "300.0",
        },
        {
            "study_case_id": "baseline_768",
            "seq_len": "64",
            "metric": "tps_per_watt",
            "igpu": "1600.0",
            "dataflow": "10.0",
            "runlist": "20.0",
            "offload": "30.0",
        },
    ]
    tps_svg = tps_plot_path.read_text(encoding="utf-8")
    tps_per_watt_svg = tps_per_watt_plot_path.read_text(encoding="utf-8")
    assert "TPS Comparison" in tps_svg
    assert 'class="bar"' in tps_svg
    assert "TPS/W Comparison" in tps_per_watt_svg
    assert 'class="bar"' in tps_per_watt_svg


def test_parse_rocm_smi_average_power_w_reads_package_power():
    parsed = parse_rocm_smi_average_power_w(
        json.dumps(
            {
                "card0": {
                    "Average Graphics Package Power (W)": "12.5W",
                }
            }
        ),
        card_label="card0",
    )
    assert parsed == 12.5
