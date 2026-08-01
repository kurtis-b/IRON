#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    WORKLOAD_VARIANTS,
    SEQUENCE_LADDER,
    default_candidates_path,
    mode_operators,
)
from .run import iteration_schedule
from .select import default_results_path, load_result_rows

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "workload_variant",
    "execution_mode",
    "operator_candidate_count_summary_json",
    "iteration_schedule_json",
    "observed_power_backends_json",
    "observed_selected_row_count",
    "latency_variation_row_count",
    "latency_variation_sample_count_min",
    "latency_variation_sample_count_max",
    "latency_variation_mean_cv_pct",
    "latency_variation_max_cv_pct",
    "validation_policy",
    "latency_variation_policy",
    "selected_candidate_source",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "fairness_repeatability.csv"
    )


def default_latency_variation_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "latency_variation.csv"
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _latency_variation_summary(
    rows: list[dict[str, str]],
    *,
    workload_variant: str,
    execution_mode: str,
) -> dict[str, object]:
    matching_rows = [
        row
        for row in rows
        if str(row.get("execution_mode") or "") == execution_mode
        and str(row.get("workload_variant") or "") == workload_variant
        and str(row.get("run_status") or "") == "passed"
    ]
    sample_counts: list[int] = []
    cv_values: list[float] = []
    for row in matching_rows:
        sample_count = _optional_int(row.get("sample_count"))
        mean_latency_ms = _optional_float(row.get("mean_latency_ms"))
        stddev_latency_ms = _optional_float(row.get("stddev_latency_ms"))
        if sample_count is not None:
            sample_counts.append(sample_count)
        if mean_latency_ms and stddev_latency_ms is not None and mean_latency_ms > 0.0:
            cv_values.append((stddev_latency_ms / mean_latency_ms) * 100.0)

    return {
        "latency_variation_row_count": len(matching_rows),
        "latency_variation_sample_count_min": (
            min(sample_counts) if sample_counts else ""
        ),
        "latency_variation_sample_count_max": (
            max(sample_counts) if sample_counts else ""
        ),
        "latency_variation_mean_cv_pct": (
            math.fsum(cv_values) / float(len(cv_values)) if cv_values else ""
        ),
        "latency_variation_max_cv_pct": max(cv_values) if cv_values else "",
    }


def _candidate_payload_path(execution_mode: str) -> Path:
    return default_candidates_path(execution_mode)


def _candidate_count_summary(
    execution_mode: str,
    workload_variant: str,
) -> dict[str, dict[str, int]]:
    payload = json.loads(
        _candidate_payload_path(execution_mode).read_text(encoding="utf-8")
    )
    variant_families = [
        family_id
        for family_id in FAMILY_IDS
        if FAMILY_SPECS[family_id].workload_variant == workload_variant
    ]
    summary: dict[str, dict[str, int]] = {
        operator_name: {"min": 10**9, "max": 0}
        for operator_name in mode_operators(execution_mode, workload_variant)
    }
    for family_id in variant_families:
        family_payload = payload[family_id]
        for seq_key in ("all", *(str(seq_len) for seq_len in SEQUENCE_LADDER)):
            seq_payload = family_payload.get(seq_key, {})
            for operator_name, counts in summary.items():
                candidate_count = len(
                    seq_payload.get(
                        operator_name,
                        family_payload.get("all", {}).get(operator_name, []),
                    )
                )
                counts["min"] = min(counts["min"], candidate_count)
                counts["max"] = max(counts["max"], candidate_count)
    return summary


def _iteration_schedule_summary() -> dict[str, dict[str, int]]:
    return {
        "64-256": {
            "warmup_runs": iteration_schedule(64)[0],
            "runs_per_sample": iteration_schedule(64)[1],
        },
        "512-2048": {
            "warmup_runs": iteration_schedule(512)[0],
            "runs_per_sample": iteration_schedule(512)[1],
        },
        "4096": {
            "warmup_runs": iteration_schedule(4096)[0],
            "runs_per_sample": iteration_schedule(4096)[1],
        },
        "8192-16384": {
            "warmup_runs": iteration_schedule(8192)[0],
            "runs_per_sample": iteration_schedule(8192)[1],
        },
    }


def build_rows(
    results_input: Path,
    latency_variation_input: Path | None = None,
) -> list[dict[str, object]]:
    observed_rows = load_result_rows(results_input) if results_input.exists() else []
    latency_variation_rows = _load_csv_rows(
        default_latency_variation_path()
        if latency_variation_input is None
        else latency_variation_input
    )
    rows: list[dict[str, object]] = []
    for workload_variant in WORKLOAD_VARIANTS:
        for execution_mode in EXECUTION_MODES:
            mode_rows = [
                row
                for row in observed_rows
                if row.get("backend") == "npu"
                and str(row.get("execution_mode") or "") == execution_mode
                and (
                    str(row.get("workload_variant") or "").strip() == workload_variant
                    or (
                        not str(row.get("workload_variant") or "").strip()
                        and row.get("study_case_id") in FAMILY_SPECS
                        and FAMILY_SPECS[str(row.get("study_case_id"))].workload_variant
                        == workload_variant
                    )
                )
            ]
            power_backends = sorted(
                {
                    str(row.get("power_backend") or "")
                    for row in mode_rows
                    if str(row.get("power_backend") or "")
                }
            )
            latency_summary = _latency_variation_summary(
                latency_variation_rows,
                workload_variant=workload_variant,
                execution_mode=execution_mode,
            )
            rows.append(
                {
                    "study_id": "end_to_end_fairness_repeatability",
                    "workload_variant": workload_variant,
                    "execution_mode": execution_mode,
                    "operator_candidate_count_summary_json": json.dumps(
                        _candidate_count_summary(execution_mode, workload_variant),
                        sort_keys=True,
                    ),
                    "iteration_schedule_json": json.dumps(
                        _iteration_schedule_summary(),
                        sort_keys=True,
                    ),
                    "observed_power_backends_json": json.dumps(power_backends),
                    "observed_selected_row_count": len(mode_rows),
                    **latency_summary,
                    "validation_policy": (
                        "Main end-to-end runner uses exact reference validation through "
                        "seq_len=512 and finite-output validation above that threshold; "
                        "paper correctness spot checks rerun validation at 512, 2048, and 8192."
                    ),
                    "latency_variation_policy": (
                        "Latency-variation runs reuse the selected end-to-end warmup and "
                        "timed-iteration schedule unless explicitly overridden; this summary "
                        "reports observed coefficient-of-variation statistics when those rows exist."
                    ),
                    "selected_candidate_source": (
                        "selected end-to-end results input CSV "
                        "selected_candidate_ids_json and selected_config_json"
                    ),
                }
            )
    return rows


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a fairness/repeatability summary for the end-to-end NPU study."
    )
    parser.add_argument(
        "--results-input",
        type=Path,
        default=default_results_path(),
    )
    parser.add_argument(
        "--latency-variation-input",
        type=Path,
        default=default_latency_variation_path(),
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="end-to-end fairness repeatability",
    ):
        rows = build_rows(
            args.results_input.expanduser(),
            args.latency_variation_input.expanduser(),
        )
        write_rows(output_path, rows)
        LOGGER.info("Wrote %d fairness rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
