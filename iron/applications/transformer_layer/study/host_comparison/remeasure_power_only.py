#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from ..run_lock import default_lock_path, hold_study_lock
from ..end_to_end.cases import FAMILY_IDS, SEQUENCE_LADDER, WORKLOAD_VARIANTS
from .run import (
    DIRECT_COMPARISON_METRICS,
    POWER_COMPARISON_METRICS,
    _comparison_row,
    _normalized_existing_row,
    _reference_metric_means,
    _resolve_plot_path,
    _row_key,
    benchmark_host_group_power_only,
    default_effective_gflops_per_watt_plot_path,
    default_effective_gflops_plot_path,
    default_output_path,
    RESULTS_CSV_FIELDNAMES,
    resolve_sampling,
    write_plots,
)
from .select import (
    default_reference_results_path,
    group_reference_rows,
    load_reference_rows,
)

LOGGER = logging.getLogger(__name__)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def _existing_group_value(
    existing_rows: dict[tuple[str, str, int, str], dict[str, object]],
    *,
    workload_variant: str,
    study_case_id: str,
    seq_len: int,
    metric: str,
    column: str,
) -> float | None:
    row = existing_rows.get((workload_variant, study_case_id, seq_len, metric))
    if row is None:
        return None
    return _optional_float(row.get(column))


def build_power_only_rows_for_group(
    *,
    group,
    existing_rows: dict[tuple[str, str, int, str], dict[str, object]],
    seed: int,
    igpu_device_name: str,
    igpu_power_backend: str,
    igpu_power_sample_interval_sec: float,
) -> list[dict[str, object]]:
    reference_metric_values = {
        metric: _reference_metric_means(group, metric)
        for metric in (*DIRECT_COMPARISON_METRICS, *POWER_COMPARISON_METRICS)
    }
    existing_direct_metrics = {
        metric: _existing_group_value(
            existing_rows,
            workload_variant=group.workload_variant,
            study_case_id=group.study_case_id,
            seq_len=group.seq_len,
            metric=metric,
            column="igpu",
        )
        for metric in DIRECT_COMPARISON_METRICS
    }
    existing_power_metrics = {
        metric: _existing_group_value(
            existing_rows,
            workload_variant=group.workload_variant,
            study_case_id=group.study_case_id,
            seq_len=group.seq_len,
            metric=metric,
            column="igpu_rocm_smi",
        )
        for metric in POWER_COMPARISON_METRICS
    }

    resolved_warmup_runs, resolved_runs_per_sample = resolve_sampling(
        group,
        warmup_runs=None,
        runs_per_sample=None,
    )
    power_result = benchmark_host_group_power_only(
        group,
        warmup_runs=resolved_warmup_runs,
        runs_per_sample=resolved_runs_per_sample,
        seed=seed,
        device_name=igpu_device_name,
        power_backend=igpu_power_backend,
        power_sample_interval_sec=igpu_power_sample_interval_sec,
        existing_avg_latency_ms=existing_direct_metrics["avg_latency_ms"],
        existing_effective_gflops_per_sec=existing_direct_metrics[
            "effective_gflops_per_sec"
        ],
    )

    if str(power_result.get("run_status") or "") != "passed":
        LOGGER.warning(
            "Keeping existing iGPU power rows for %s seq_len=%s: %s",
            group.study_case_id,
            group.seq_len,
            power_result.get("failure_message") or power_result["run_status"],
        )
        updated_power_metrics = dict(existing_power_metrics)
    else:
        updated_power_metrics = {
            metric: _optional_float(power_result.get(metric))
            for metric in POWER_COMPARISON_METRICS
        }

    rows = [
        _comparison_row(
            group=group,
            metric=metric,
            igpu_value=existing_direct_metrics[metric],
            reference_values=reference_metric_values[metric],
        )
        for metric in DIRECT_COMPARISON_METRICS
    ]
    rows.extend(
        _comparison_row(
            group=group,
            metric=metric,
            igpu_value=None,
            igpu_rocm_smi_value=updated_power_metrics[metric],
            reference_values=reference_metric_values[metric],
        )
        for metric in POWER_COMPARISON_METRICS
    )
    return rows


def _selected_group(
    group,
    *,
    workload_variant: str,
    family: str,
    seq_len: str,
) -> bool:
    if workload_variant != "all" and group.workload_variant != workload_variant:
        return False
    if family != "all" and group.study_case_id != family:
        return False
    if seq_len != "all" and int(group.seq_len) != int(seq_len):
        return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remeasure host-comparison iGPU power only and rewrite GFLOPS/W"
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
        "--reference-input", type=Path, default=default_reference_results_path()
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--igpu-device", default="cuda:0")
    parser.add_argument("--igpu-power-backend", default="rocm-smi")
    parser.add_argument("--igpu-power-sample-interval-sec", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--effective-gflops-plot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--effective-gflops-per-watt-plot",
        type=Path,
        default=None,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    reference_input = args.reference_input.expanduser()
    output_path = args.output.expanduser()
    effective_gflops_plot_path = (
        default_effective_gflops_plot_path(output_path)
        if args.effective_gflops_plot is None
        else _resolve_plot_path(args.effective_gflops_plot.expanduser())
    )
    effective_gflops_per_watt_plot_path = (
        default_effective_gflops_per_watt_plot_path(output_path)
        if args.effective_gflops_per_watt_plot is None
        else _resolve_plot_path(args.effective_gflops_per_watt_plot.expanduser())
    )

    with hold_study_lock(
        default_lock_path(output_path),
        study_name="host comparison power-only refresh",
    ):
        existing_row_list = load_rows(output_path)
        if not existing_row_list:
            LOGGER.warning("No host comparison rows found at %s", output_path)
            return 0
        if not reference_input.exists():
            LOGGER.error(
                "Reference end_to_end results not found at %s", reference_input
            )
            return 1
        groups = group_reference_rows(load_reference_rows(reference_input))
        selected_groups = tuple(
            group
            for group in groups
            if _selected_group(
                group,
                workload_variant=args.workload_variant,
                family=args.family,
                seq_len=args.seq_len,
            )
        )
        if not selected_groups:
            LOGGER.error(
                "No matching reference rows found in %s for workload_variant=%s family=%s seq_len=%s",
                reference_input,
                args.workload_variant,
                args.family,
                args.seq_len,
            )
            return 1
        existing_rows = {
            _row_key(normalized): normalized
            for row in existing_row_list
            if (normalized := _normalized_existing_row(row)) is not None
        }
        updated_row_map: dict[tuple[str, str, int, str], dict[str, object]] = dict(
            existing_rows
        )
        for group in selected_groups:
            for row in build_power_only_rows_for_group(
                group=group,
                existing_rows=existing_rows,
                seed=int(args.seed),
                igpu_device_name=str(args.igpu_device),
                igpu_power_backend=str(args.igpu_power_backend),
                igpu_power_sample_interval_sec=float(
                    args.igpu_power_sample_interval_sec
                ),
            ):
                updated_row_map[_row_key(row)] = row
        updated_rows = list(updated_row_map.values())
        write_rows(output_path, updated_rows)
        write_plots(
            updated_rows,
            effective_gflops_plot_path=effective_gflops_plot_path,
            effective_gflops_per_watt_plot_path=effective_gflops_per_watt_plot_path,
        )
        LOGGER.info(
            "Wrote %s host comparison rows to %s",
            len(updated_rows),
            output_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
