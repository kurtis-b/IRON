#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from ..campaign import DEFAULT_REPEAT_COUNT, summarize_repeat_values
from ..run_lock import default_lock_path, hold_study_lock
from .run import (
    PERSISTED_POWER_RESULT_FIELDS,
    SUPPORTED_IGPU_POWER_BACKENDS,
    benchmark_host_group,
    benchmark_host_group_power_only,
    resolve_sampling,
)
from .select import (
    REFERENCE_EXECUTION_MODES,
    default_reference_results_path,
    group_reference_rows,
    load_reference_rows,
)

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "matched_run_id",
    "workload_variant",
    "study_case_id",
    "study_case_label",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "warmup_runs",
    "runs_per_sample",
    "device",
    "power_backend",
    "reference_execution_modes_json",
    "avg_latency_ms",
    "latency_sample_count",
    "min_latency_ms",
    "max_latency_ms",
    "effective_gflops_per_sec",
    *PERSISTED_POWER_RESULT_FIELDS,
    "effective_gflops_per_sec_per_watt",
    "validation_error_count",
    "run_status",
    "failure_message",
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
        / "host_comparison"
        / "fairness_repeatability.csv"
    )


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
    *,
    reference_input: Path,
    igpu_device: str,
    igpu_power_backend: str,
    igpu_power_sample_interval_sec: float = 0.2,
    family_filter: str = "all",
    seq_len_filter: str = "all",
    seed: int = 42,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    benchmark_latency_fn=None,
    benchmark_power_fn=None,
) -> list[dict[str, object]]:
    if benchmark_latency_fn is None:
        benchmark_latency_fn = benchmark_host_group
    if benchmark_power_fn is None:
        benchmark_power_fn = benchmark_host_group_power_only

    groups = group_reference_rows(
        load_reference_rows(reference_input),
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
    )
    rows: list[dict[str, object]] = []
    for group in groups:
        warmup_runs, runs_per_sample = resolve_sampling(
            group,
            warmup_runs=None,
            runs_per_sample=None,
        )
        group_rows: list[dict[str, object]] = []
        for repeat_index in range(int(repeat_count)):
            latency_result = benchmark_latency_fn(
                group,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                device_name=igpu_device,
                power_backend="none",
                power_sample_interval_sec=igpu_power_sample_interval_sec,
            )
            power_result = benchmark_power_fn(
                group,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                device_name=igpu_device,
                power_backend=igpu_power_backend,
                power_sample_interval_sec=igpu_power_sample_interval_sec,
                existing_avg_latency_ms=latency_result.get("avg_latency_ms"),
                existing_effective_gflops_per_sec=latency_result.get(
                    "effective_gflops_per_sec"
                ),
            )
            run_status = "passed"
            failure_message = ""
            for candidate_result in (latency_result, power_result):
                candidate_status = str(candidate_result.get("run_status") or "")
                if candidate_status != "passed":
                    run_status = candidate_status
                    failure_message = str(candidate_result.get("failure_message") or "")
                    break
            row = {
                "study_id": "host_comparison_fairness_repeatability",
                "campaign_id": group.campaign_id,
                "repeat_index": repeat_index,
                "matched_run_id": f"{group.matched_run_id}:host_repeat_{repeat_index}",
                "workload_variant": group.workload_variant,
                "study_case_id": group.study_case_id,
                "study_case_label": group.study_case_label,
                "seq_len": group.seq_len,
                "hidden_size": group.hidden_size,
                "intermediate_size": group.intermediate_size,
                "num_attention_heads": group.num_attention_heads,
                "attention_head_size": group.attention_head_size,
                "warmup_runs": warmup_runs,
                "runs_per_sample": runs_per_sample,
                "device": igpu_device,
                "power_backend": power_result.get("power_backend"),
                "reference_execution_modes_json": json.dumps(
                    list(REFERENCE_EXECUTION_MODES)
                ),
                "avg_latency_ms": latency_result.get("avg_latency_ms"),
                "latency_sample_count": latency_result.get("latency_sample_count"),
                "min_latency_ms": latency_result.get("min_latency_ms"),
                "max_latency_ms": latency_result.get("max_latency_ms"),
                "effective_gflops_per_sec": latency_result.get(
                    "effective_gflops_per_sec"
                ),
                "effective_gflops_per_sec_per_watt": power_result.get(
                    "effective_gflops_per_sec_per_watt"
                ),
                "validation_error_count": latency_result.get(
                    "validation_error_count", ""
                ),
                "run_status": run_status,
                "failure_message": failure_message,
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
        description="Run paired repeatability measurements for the host comparison study."
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=default_reference_results_path(),
    )
    parser.add_argument("--igpu-device", default="cuda:0")
    parser.add_argument(
        "--igpu-power-backend",
        choices=list(SUPPORTED_IGPU_POWER_BACKENDS),
        default="rocm-smi",
    )
    parser.add_argument("--igpu-power-sample-interval-sec", type=float, default=0.2)
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--seed", type=int, default=42)
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
        study_name="host comparison fairness repeatability",
    ):
        rows = build_rows(
            reference_input=args.reference_input.expanduser(),
            igpu_device=str(args.igpu_device),
            igpu_power_backend=str(args.igpu_power_backend),
            igpu_power_sample_interval_sec=float(args.igpu_power_sample_interval_sec),
            family_filter=str(args.family),
            seq_len_filter=str(args.seq_len),
            seed=int(args.seed),
            repeat_count=int(args.repeat_count),
        )
        write_rows(output_path, rows)
        LOGGER.info(
            "Wrote %d host fairness-repeatability rows to %s",
            len(rows),
            output_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
