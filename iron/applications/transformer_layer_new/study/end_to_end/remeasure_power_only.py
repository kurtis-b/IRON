#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    WORKLOAD_VARIANTS,
    effective_gflops_per_sec_per_watt,
    get_case,
)
from .modes import benchmark_mode_power_only
from .power import SUPPORTED_POWER_BACKENDS

LOGGER = logging.getLogger(__name__)

POWER_RESULT_FIELDS = (
    "power_backend",
    "avg_power_w",
    "min_power_w",
    "max_power_w",
    "power_sample_count",
    "effective_gflops_per_sec_per_watt",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "results_all_power.csv"
    )


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected(
    row: dict[str, str],
    *,
    workload_variant: str,
    family: str,
    seq_len: str,
    mode: str,
) -> bool:
    if str(row.get("backend") or "") != "npu":
        return False
    if str(row.get("run_status") or "") != "passed":
        return False
    if (
        workload_variant != "all"
        and str(row.get("workload_variant") or "") != workload_variant
    ):
        return False
    if family != "all" and str(row.get("study_case_id") or "") != family:
        return False
    if seq_len != "all" and int(float(str(row.get("seq_len") or 0))) != int(seq_len):
        return False
    if mode != "all" and str(row.get("execution_mode") or "") != mode:
        return False
    return True


def updated_row_with_power_measurement(
    existing_row: dict[str, str],
    power_result: dict[str, object],
) -> dict[str, object]:
    updated: dict[str, object] = dict(existing_row)
    updated["power_backend"] = power_result.get(
        "power_backend",
        existing_row.get("power_backend", ""),
    )
    updated["avg_power_w"] = power_result.get("avg_power_w")
    updated["min_power_w"] = power_result.get("min_power_w")
    updated["max_power_w"] = power_result.get("max_power_w")
    updated["power_sample_count"] = power_result.get("power_sample_count")
    updated["effective_gflops_per_sec_per_watt"] = effective_gflops_per_sec_per_watt(
        _optional_float(existing_row.get("effective_gflops_per_sec")),
        _optional_float(power_result.get("avg_power_w")),
    )
    return updated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remeasure end-to-end NPU power only and rewrite GFLOPS/W"
    )
    parser.add_argument(
        "--workload-variant",
        choices=[*WORKLOAD_VARIANTS, "all"],
        default="all",
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
    )
    parser.add_argument(
        "--seq-len",
        choices=[*(str(value) for value in SEQUENCE_LADDER), "all"],
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=[*EXECUTION_MODES, "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--power-backend",
        choices=list(SUPPORTED_POWER_BACKENDS),
        default="auto",
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
        study_name="end-to-end power-only refresh",
    ):
        rows = load_rows(output_path)
        if not rows:
            LOGGER.warning("No end-to-end rows found at %s", output_path)
            return 0

        refreshed = 0
        preserved = 0
        for index, row in enumerate(rows):
            if not _selected(
                row,
                workload_variant=args.workload_variant,
                family=args.family,
                seq_len=args.seq_len,
                mode=args.mode,
            ):
                continue
            selected_config_text = str(row.get("selected_config_json") or "")
            if not selected_config_text:
                LOGGER.warning(
                    "Skipping %s/%s/%s: missing selected_config_json",
                    row.get("study_case_id"),
                    row.get("execution_mode"),
                    row.get("seq_len"),
                )
                preserved += 1
                continue
            try:
                selected_config = json.loads(selected_config_text)
            except json.JSONDecodeError as exc:
                LOGGER.warning(
                    "Skipping %s/%s/%s: invalid selected_config_json: %s",
                    row.get("study_case_id"),
                    row.get("execution_mode"),
                    row.get("seq_len"),
                    exc,
                )
                preserved += 1
                continue
            case = get_case(
                str(row["study_case_id"]),
                int(float(str(row["seq_len"]))),
            )
            power_result = benchmark_mode_power_only(
                str(row["execution_mode"]),
                case.workload,
                warmup_runs=int(float(str(row.get("warmup_runs") or 0))),
                runs_per_sample=int(float(str(row.get("runs_per_sample") or 0))),
                seed=int(args.seed),
                power_backend=str(args.power_backend),
                operator_config=selected_config,
                avg_latency_ms=_optional_float(row.get("avg_latency_ms")),
                timed_total_sec=float(str(row.get("timed_total_sec") or 0.0)),
                effective_gflops_per_sec_value=_optional_float(
                    row.get("effective_gflops_per_sec")
                ),
            )
            if str(power_result.get("run_status") or "") != "passed":
                LOGGER.warning(
                    "Keeping existing row for %s/%s/%s: %s",
                    row.get("study_case_id"),
                    row.get("execution_mode"),
                    row.get("seq_len"),
                    power_result.get("failure_message") or power_result["run_status"],
                )
                preserved += 1
                continue
            rows[index] = updated_row_with_power_measurement(row, power_result)
            refreshed += 1
            LOGGER.info(
                "Updated power-only row for %s/%s/%s",
                row.get("study_case_id"),
                row.get("execution_mode"),
                row.get("seq_len"),
            )

        write_rows(output_path, rows)
        LOGGER.info(
            "Wrote %s rows to %s (%s refreshed, %s preserved)",
            len(rows),
            output_path,
            refreshed,
            preserved,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
