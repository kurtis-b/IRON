#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..end_to_end.select import default_results_path, load_result_rows, select_result_rows

RESULTS_CSV_FIELDNAMES = (
    "study_case_id",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "avg_latency_ms",
    "effective_gflops_per_sec",
    "compute_tiles_used",
    "shim_tiles_used",
    "measured_bandwidth_gbps",
    "peak_memcpy_bandwidth_gbps",
    "operational_intensity_flops_per_byte",
    "dominant_limiter",
    "crossover_explanation",
)


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "analysis" / "crossover_report.csv"


def default_roofline_input() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "roofline" / "implementation_points.csv"


def default_memcpy_input() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "memcpy_bandwidth" / "results.csv"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _roofline_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        if str(row.get("run_status") or "") != "passed":
            continue
        key = (
            str(row.get("study_case_id") or ""),
            str(row.get("execution_mode") or ""),
            int(float(str(row.get("seq_len") or 0))),
        )
        index[key] = row
    return index


def _peak_memcpy_bandwidth(rows: list[dict[str, str]]) -> float | None:
    values = [
        float(row["bandwidth_gbps"])
        for row in rows
        if str(row.get("run_status") or "") == "passed"
        and str(row.get("bandwidth_gbps") or "").strip()
    ]
    if not values:
        return None
    return max(values)


def _crossover_explanation(
    *,
    measured_bandwidth_gbps: float | None,
    peak_memcpy_bandwidth_gbps: float | None,
    operational_intensity: float | None,
    compute_tiles_used: int | None,
) -> tuple[str, str]:
    if measured_bandwidth_gbps is not None and peak_memcpy_bandwidth_gbps:
        bandwidth_fraction = measured_bandwidth_gbps / peak_memcpy_bandwidth_gbps
        if bandwidth_fraction >= 0.75:
            return (
                "bandwidth",
                f"Measured bandwidth is {bandwidth_fraction:.2f} of the memcpy roofline.",
            )
    if operational_intensity is not None and operational_intensity < 8.0:
        return (
            "memory_pressure",
            f"Operational intensity is only {operational_intensity:.2f} flop/byte.",
        )
    if compute_tiles_used is not None and compute_tiles_used <= 4:
        return (
            "parallelism",
            f"Only {compute_tiles_used} compute tiles are active in the selected implementation.",
        )
    return ("compute", "The selected implementation appears compute-limited.")


def build_rows(
    *,
    end_to_end_input: Path,
    roofline_input: Path,
    memcpy_input: Path,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(load_result_rows(end_to_end_input))
    roofline_rows = _roofline_index(_load_csv_rows(roofline_input))
    peak_memcpy_bandwidth_gbps = _peak_memcpy_bandwidth(_load_csv_rows(memcpy_input))
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        roofline_row = roofline_rows.get(
            (selected_row.study_case_id, selected_row.execution_mode, selected_row.seq_len)
        )
        operational_intensity = (
            None
            if roofline_row is None
            else _optional_float(roofline_row.get("operational_intensity_flops_per_byte"))
        )
        effective_gflops_per_sec = (
            None
            if roofline_row is None
            else _optional_float(roofline_row.get("effective_gflops_per_sec"))
        )
        measured_bandwidth_gbps = (
            None
            if effective_gflops_per_sec is None
            or operational_intensity in (None, 0.0)
            else effective_gflops_per_sec / operational_intensity
        )
        compute_tiles_used = (
            None if roofline_row is None else _optional_int(roofline_row.get("compute_tiles_used"))
        )
        shim_tiles_used = (
            None if roofline_row is None else _optional_int(roofline_row.get("shim_tiles_used"))
        )
        dominant_limiter, explanation = _crossover_explanation(
            measured_bandwidth_gbps=measured_bandwidth_gbps,
            peak_memcpy_bandwidth_gbps=peak_memcpy_bandwidth_gbps,
            operational_intensity=operational_intensity,
            compute_tiles_used=compute_tiles_used,
        )
        rows.append(
            {
                "study_case_id": selected_row.study_case_id,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "avg_latency_ms": selected_row.row.get("avg_latency_ms", ""),
                "effective_gflops_per_sec": (
                    effective_gflops_per_sec
                    if effective_gflops_per_sec is not None
                    else selected_row.row.get("effective_gflops_per_sec", "")
                ),
                "compute_tiles_used": compute_tiles_used,
                "shim_tiles_used": shim_tiles_used,
                "measured_bandwidth_gbps": measured_bandwidth_gbps,
                "peak_memcpy_bandwidth_gbps": peak_memcpy_bandwidth_gbps,
                "operational_intensity_flops_per_byte": operational_intensity,
                "dominant_limiter": dominant_limiter,
                "crossover_explanation": explanation,
            }
        )
    return rows


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join end-to-end, roofline, and memcpy studies into a crossover explanation table."
    )
    parser.add_argument("--end-to-end-input", type=Path, default=default_results_path())
    parser.add_argument("--roofline-input", type=Path, default=default_roofline_input())
    parser.add_argument("--memcpy-input", type=Path, default=default_memcpy_input())
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(
        end_to_end_input=args.end_to_end_input.expanduser(),
        roofline_input=args.roofline_input.expanduser(),
        memcpy_input=args.memcpy_input.expanduser(),
    )
    write_rows(args.output.expanduser(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
