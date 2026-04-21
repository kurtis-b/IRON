#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from ..campaign import (
    DEFAULT_REPEAT_COUNT,
    matched_run_id,
    summarize_repeat_values,
)
from ..run_lock import default_lock_path, hold_study_lock
from .cases import EXECUTION_MODES, FAMILY_IDS, WORKLOAD_VARIANTS, get_case
from .modes import benchmark_mode, benchmark_mode_power_only
from .power import PERSISTED_POWER_RESULT_FIELDS, SUPPORTED_POWER_BACKENDS
from .run import iteration_schedule
from .select import default_results_path, load_result_rows, select_result_rows

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "matched_run_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "latency_sample_count",
    "min_latency_ms",
    "max_latency_ms",
    "effective_gflops_per_sec",
    "power_backend",
    *PERSISTED_POWER_RESULT_FIELDS,
    "effective_gflops_per_sec_per_watt",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
    "repeat_count",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_stddev_ms",
    "latency_iqr_ms",
    "latency_min_repeat_ms",
    "latency_max_repeat_ms",
    "latency_ci95_ms",
    "power_mean_w",
    "power_median_w",
    "power_stddev_w",
    "power_iqr_w",
    "power_min_repeat_w",
    "power_max_repeat_w",
    "power_ci95_w",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "fairness_repeatability.csv"
    )


def _resolved_sampling(selected_row) -> tuple[int, int]:
    scheduled_warmup_runs, scheduled_runs_per_sample = iteration_schedule(
        selected_row.seq_len
    )
    warmup_runs = (
        scheduled_warmup_runs
        if selected_row.warmup_runs in (None, 0)
        else int(selected_row.warmup_runs)
    )
    runs_per_sample = (
        scheduled_runs_per_sample
        if selected_row.runs_per_sample in (None, 0)
        else int(selected_row.runs_per_sample)
    )
    return warmup_runs, runs_per_sample


def _repeat_stats(values: list[float], *, stem: str, suffix: str) -> dict[str, object]:
    stats = summarize_repeat_values(values, prefix=stem)
    return {
        "repeat_count": stats["repeat_count"],
        f"{stem}_mean{suffix}": stats[f"{stem}_mean"],
        f"{stem}_median{suffix}": stats[f"{stem}_median"],
        f"{stem}_stddev{suffix}": stats[f"{stem}_stddev"],
        f"{stem}_iqr{suffix}": stats[f"{stem}_iqr"],
        f"{stem}_min_repeat{suffix}": stats[f"{stem}_min"],
        f"{stem}_max_repeat{suffix}": stats[f"{stem}_max"],
        f"{stem}_ci95{suffix}": stats[f"{stem}_ci95"],
    }


def _apply_group_stats(rows: list[dict[str, object]]) -> None:
    latency_values = [
        float(row["avg_latency_ms"])
        for row in rows
        if row.get("run_status") == "passed"
        and row.get("avg_latency_ms") not in (None, "")
    ]
    power_values = [
        float(row["avg_power_w"])
        for row in rows
        if row.get("run_status") == "passed"
        and row.get("avg_power_w") not in (None, "")
    ]
    latency_stats = _repeat_stats(latency_values, stem="latency", suffix="_ms")
    power_stats = _repeat_stats(power_values, stem="power", suffix="_w")
    for row in rows:
        row.update(latency_stats)
        row.update(power_stats)


def build_rows(
    results_input: Path,
    *,
    workload_variant_filter: str = "all",
    family_filter: str = "all",
    mode_filter: str = "all",
    seq_len_filter: str = "all",
    seed: int = 42,
    power_backend: str = "auto",
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    benchmark_latency_fn=None,
    benchmark_power_fn=None,
    benchmark_validation_fn=None,
) -> list[dict[str, object]]:
    if benchmark_latency_fn is None:
        benchmark_latency_fn = benchmark_mode
    if benchmark_power_fn is None:
        benchmark_power_fn = benchmark_mode_power_only
    if benchmark_validation_fn is None:
        benchmark_validation_fn = benchmark_mode

    selected_rows = select_result_rows(
        load_result_rows(results_input),
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter=mode_filter,
    )
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        warmup_runs, runs_per_sample = _resolved_sampling(selected_row)
        selected_candidate_ids_json = json.dumps(
            selected_row.selected_candidate_ids,
            sort_keys=True,
        )
        selected_config_json = json.dumps(
            selected_row.selected_config,
            sort_keys=True,
        )
        group_rows: list[dict[str, object]] = []
        requested_power_backend = str(
            selected_row.row.get("power_backend") or power_backend
        )
        for repeat_index in range(int(repeat_count)):
            matched_id = matched_run_id(
                campaign_id=selected_row.campaign_id,
                study_case_id=selected_row.study_case_id,
                execution_mode=selected_row.execution_mode,
                seq_len=selected_row.seq_len,
                repeat_index=repeat_index,
            )
            latency_result = benchmark_latency_fn(
                selected_row.execution_mode,
                case.workload,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                power_backend="none",
                operator_config=selected_row.selected_config,
                include_reference_output=False,
                scope_suffix=f"fairness_latency_repeat_{repeat_index}",
            )
            validation_result = benchmark_validation_fn(
                selected_row.execution_mode,
                case.workload,
                warmup_runs=0,
                runs_per_sample=1,
                seed=seed,
                power_backend="none",
                operator_config=selected_row.selected_config,
                include_reference_output=True,
                scope_suffix=f"fairness_validation_repeat_{repeat_index}",
            )
            power_result = benchmark_power_fn(
                selected_row.execution_mode,
                case.workload,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                power_backend=requested_power_backend,
                operator_config=selected_row.selected_config,
                avg_latency_ms=latency_result.get("avg_latency_ms"),
                timed_total_sec=float(latency_result.get("timed_total_sec") or 0.0),
                effective_gflops_per_sec_value=latency_result.get(
                    "effective_gflops_per_sec"
                ),
                scope_suffix=f"fairness_power_repeat_{repeat_index}",
            )

            run_status = "passed"
            failure_message = ""
            for candidate_result in (
                latency_result,
                power_result,
                validation_result,
            ):
                candidate_status = str(candidate_result.get("run_status") or "")
                if candidate_status != "passed":
                    run_status = candidate_status
                    failure_message = str(candidate_result.get("failure_message") or "")
                    break

            row = {
                "study_id": "end_to_end_fairness_repeatability",
                "campaign_id": selected_row.campaign_id,
                "repeat_index": repeat_index,
                "matched_run_id": matched_id,
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "hidden_size": selected_row.hidden_size,
                "intermediate_size": selected_row.intermediate_size,
                "num_attention_heads": selected_row.num_attention_heads,
                "attention_head_size": selected_row.attention_head_size,
                "warmup_runs": warmup_runs,
                "runs_per_sample": runs_per_sample,
                "avg_latency_ms": latency_result.get("avg_latency_ms"),
                "latency_sample_count": latency_result.get("latency_sample_count"),
                "min_latency_ms": latency_result.get("min_latency_ms"),
                "max_latency_ms": latency_result.get("max_latency_ms"),
                "effective_gflops_per_sec": latency_result.get(
                    "effective_gflops_per_sec"
                ),
                "power_backend": power_result.get("power_backend"),
                "effective_gflops_per_sec_per_watt": power_result.get(
                    "effective_gflops_per_sec_per_watt"
                ),
                "validation_error_count": validation_result.get(
                    "validation_error_count", ""
                ),
                "run_status": run_status,
                "failure_message": failure_message,
                "selected_candidate_ids_json": selected_candidate_ids_json,
                "selected_config_json": selected_config_json,
            }
            for field in PERSISTED_POWER_RESULT_FIELDS:
                row[field] = power_result.get(field)
            group_rows.append(row)
        _apply_group_stats(group_rows)
        rows.extend(group_rows)
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
        description="Run paired repeatability measurements for selected end-to-end NPU configs."
    )
    parser.add_argument("--results-input", type=Path, default=default_results_path())
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
    parser.add_argument("--seq-len", default="all")
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
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
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
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            mode_filter=str(args.mode),
            seq_len_filter=str(args.seq_len),
            seed=int(args.seed),
            power_backend=str(args.power_backend),
            repeat_count=int(args.repeat_count),
        )
        write_rows(output_path, rows)
        LOGGER.info(
            "Wrote %d fairness-repeatability rows to %s", len(rows), output_path
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
