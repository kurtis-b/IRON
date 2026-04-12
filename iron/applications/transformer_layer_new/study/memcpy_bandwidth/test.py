#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import pytest

from iron.applications.transformer_layer_new.study.memcpy_bandwidth.cases import (
    iter_cases,
)
from iron.applications.transformer_layer_new.study.memcpy_bandwidth.run import (
    CSV_FIELDNAMES,
    main,
    mark_peak_rows,
    reusable_existing_row,
    write_peak_metric_plot,
    write_rows,
)


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _result(
    *,
    size_elements: int,
    num_cores: int,
    num_channels: int,
    bypass: bool,
    tile_size: int,
    latency_us: float | None,
    bandwidth_gbps: float | None,
    run_status: str = "passed",
    failure_message: str = "",
) -> dict[str, object]:
    return {
        "study_id": "memcpy_bandwidth",
        "case_id": (
            f"memcpy_{size_elements}_cores{num_cores}_ch{num_channels}_"
            f"{'bypass' if bypass else 'kernel'}"
        ),
        "size_elements": size_elements,
        "size_bytes": size_elements * 2,
        "total_moved_bytes": size_elements * 4,
        "num_cores": num_cores,
        "num_channels": num_channels,
        "bypass": bypass,
        "tile_size": tile_size,
        "warmup_iters": 1,
        "timed_iters": 10,
        "latency_us": latency_us,
        "bandwidth_gbps": bandwidth_gbps,
        "validation_error_count": 0,
        "run_status": run_status,
        "failure_message": failure_message,
        "is_size_peak": False,
        "is_overall_peak": False,
    }


def test_iter_cases_uses_fixed_size_and_shim_tile_scaled_cores():
    invalid_cases = iter_cases(
        size_filter="8388608",
        num_cores_filter="1",
        num_channels_filter="2",
        bypass_filter="all",
    )
    assert invalid_cases == ()

    valid_cases = iter_cases(
        size_filter="8388608",
        num_cores_filter="8",
        num_channels_filter="2",
        bypass_filter="all",
    )
    assert len(valid_cases) == 2
    assert {case.num_channels for case in valid_cases} == {2}
    assert {case.tile_size for case in valid_cases} == {4096}

    max_core_cases = iter_cases(
        size_filter="8388608",
        num_cores_filter="16",
        num_channels_filter="2",
        bypass_filter="all",
    )
    assert len(max_core_cases) == 2
    assert {case.tile_size for case in max_core_cases} == {4096}

    invalid_one_channel_cases = iter_cases(
        size_filter="8388608",
        num_cores_filter="16",
        num_channels_filter="1",
        bypass_filter="all",
    )
    assert invalid_one_channel_cases == ()

    invalid_core_cases = iter_cases(
        size_filter="8388608",
        num_cores_filter="6",
        num_channels_filter="2",
        bypass_filter="all",
    )
    assert invalid_core_cases == ()


def test_mark_peak_rows_flags_size_peaks_and_overall_peak():
    rows = [
        _result(
            size_elements=1024,
            num_cores=1,
            num_channels=1,
            bypass=False,
            tile_size=1024,
            latency_us=9.0,
            bandwidth_gbps=1.0,
        ),
        _result(
            size_elements=1024,
            num_cores=2,
            num_channels=1,
            bypass=False,
            tile_size=512,
            latency_us=6.0,
            bandwidth_gbps=1.5,
        ),
        _result(
            size_elements=2048,
            num_cores=2,
            num_channels=1,
            bypass=True,
            tile_size=1024,
            latency_us=5.0,
            bandwidth_gbps=3.0,
        ),
    ]

    mark_peak_rows(rows)

    assert rows[0]["is_size_peak"] is False
    assert rows[1]["is_size_peak"] is True
    assert rows[1]["is_overall_peak"] is False
    assert rows[2]["is_size_peak"] is True
    assert rows[2]["is_overall_peak"] is True


def test_write_rows_writes_required_columns(tmp_path):
    output_path = tmp_path / "results.csv"
    write_rows(
        output_path,
        [
            _result(
                size_elements=1024,
                num_cores=1,
                num_channels=1,
                bypass=False,
                tile_size=1024,
                latency_us=10.0,
                bandwidth_gbps=0.5,
            )
        ],
    )

    rows = _read_csv_rows(output_path)
    assert len(rows) == 1
    assert tuple(rows[0].keys()) == CSV_FIELDNAMES
    assert rows[0]["size_elements"] == "1024"
    assert rows[0]["bandwidth_gbps"] == "0.5"


def test_write_peak_metric_plot_writes_svg_labels(tmp_path):
    output_path = tmp_path / "peak_bandwidth_by_size.svg"
    rows = [
        {
            **_result(
                size_elements=1024,
                num_cores=2,
                num_channels=2,
                bypass=False,
                tile_size=1024,
                latency_us=10.0,
                bandwidth_gbps=0.5,
            ),
            "is_size_peak": True,
        },
        {
            **_result(
                size_elements=2048,
                num_cores=2,
                num_channels=2,
                bypass=True,
                tile_size=1024,
                latency_us=8.0,
                bandwidth_gbps=1.0,
            ),
            "is_size_peak": True,
        },
    ]

    write_peak_metric_plot(
        output_path,
        rows,
        metric_key="bandwidth_gbps",
        title="Bandwidth by Shim Tile Count",
        y_axis_label="Effective Bandwidth (GB/s)",
        bar_color="#0072b2",
    )

    svg = output_path.read_text(encoding="utf-8")
    assert "Bandwidth by Shim Tile Count" in svg
    assert "Shim Tile Count" in svg
    assert "Effective Bandwidth (GB/s)" in svg
    assert "1" in svg
    assert "2" in svg


def test_main_writes_csv_and_plots_from_monkeypatched_benchmarks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    def fake_benchmark_case(case, *, warmup_iters, timed_iters):
        assert warmup_iters == 10
        assert timed_iters == 4
        return {
            "latency_us": float(case.size_elements) / float(case.num_cores * 100.0),
            "bandwidth_gbps": float(case.num_cores) + (0.5 if case.bypass else 0.0),
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.benchmark_case",
        fake_benchmark_case,
    )

    output_path = tmp_path / "results.csv"
    bandwidth_plot_path = tmp_path / "bandwidth_by_shim_tiles.svg"

    exit_code = main(
        [
            "--size",
            "8388608",
            "--num-cores",
            "8",
            "--num-channels",
            "2",
            "--bypass",
            "all",
            "--timed-iters",
            "4",
            "--output",
            str(output_path),
            "--bandwidth-plot",
            str(bandwidth_plot_path),
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert len(rows) == 2
    assert rows[0]["run_status"] == "passed"
    assert rows[0]["is_overall_peak"] == "False"
    assert rows[-1]["is_overall_peak"] == "True"
    assert bandwidth_plot_path.exists()


def test_main_defaults_to_bypass_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    def fake_benchmark_case(case, *, warmup_iters, timed_iters):
        assert warmup_iters == 10
        assert timed_iters == 500
        return {
            "latency_us": float(case.size_elements) / float(case.num_cores * 100.0),
            "bandwidth_gbps": float(case.num_cores) + (0.5 if case.bypass else 0.0),
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.benchmark_case",
        fake_benchmark_case,
    )

    output_path = tmp_path / "results.csv"
    exit_code = main(
        [
            "--size",
            "8388608",
            "--num-cores",
            "8",
            "--num-channels",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert len(rows) == 1
    assert {row["bypass"] for row in rows} == {"True"}


def test_main_emits_failed_exception_rows_without_aborting(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    def fake_benchmark_case(case, *, warmup_iters, timed_iters):
        del warmup_iters, timed_iters
        if case.bypass:
            raise RuntimeError("boom")
        return {
            "latency_us": 5.0,
            "bandwidth_gbps": 2.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.benchmark_case",
        fake_benchmark_case,
    )

    output_path = tmp_path / "results.csv"
    exit_code = main(
        [
            "--size",
            "8388608",
            "--num-cores",
            "8",
            "--num-channels",
            "2",
            "--bypass",
            "all",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert len(rows) == 2
    assert [row["run_status"] for row in rows] == ["passed", "failed_exception"]
    assert rows[0]["is_size_peak"] == "True"
    assert rows[1]["failure_message"] == "boom"


def test_reusable_existing_row_matches_case_and_sampling():
    case = iter_cases(
        size_filter="8388608",
        num_cores_filter="8",
        num_channels_filter="2",
        bypass_filter="true",
    )[0]
    row = reusable_existing_row(
        {
            case.case_id: {
                "case_id": case.case_id,
                "warmup_iters": "10",
                "timed_iters": "500",
                "run_status": "passed",
            }
        },
        case=case,
        warmup_iters=10,
        timed_iters=500,
    )

    assert row is not None
    assert row["case_id"] == case.case_id


def test_main_reuses_matching_existing_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    def fail_benchmark_case(*args, **kwargs):
        raise AssertionError("benchmark_case should not be called")

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.benchmark_case",
        fail_benchmark_case,
    )

    output_path = tmp_path / "results.csv"
    write_rows(
        output_path,
        [
            {
                **_result(
                    size_elements=8388608,
                    num_cores=8,
                    num_channels=2,
                    bypass=True,
                    tile_size=4096,
                    latency_us=10.0,
                    bandwidth_gbps=12.5,
                ),
                "warmup_iters": 10,
                "timed_iters": 500,
            }
        ],
    )

    exit_code = main(
        [
            "--size",
            "8388608",
            "--num-cores",
            "8",
            "--num-channels",
            "2",
            "--bypass",
            "true",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert len(rows) == 1
    assert rows[0]["bandwidth_gbps"] == "12.5"


def test_main_rejects_out_of_surface_cli_choices(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("benchmark_case should not run when no cases match")

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memcpy_bandwidth.run.benchmark_case",
        fail_if_called,
    )

    output_path = tmp_path / "results.csv"
    with pytest.raises(SystemExit):
        main(
            [
                "--size",
                "8388608",
                "--num-cores",
                "1",
                "--num-channels",
                "2",
                "--bypass",
                "all",
                "--output",
                str(output_path),
            ]
        )
