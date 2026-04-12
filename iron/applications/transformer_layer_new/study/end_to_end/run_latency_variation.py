#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from ..npu_runtime_checks import warn_if_npu_power_mode_not_turbo
from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_SPECS,
    FAMILY_IDS,
    WORKLOAD_VARIANTS,
    SEQUENCE_LADDER,
    get_case,
)
from .modes import benchmark_mode, benchmark_mode_subprocess
from .run import iteration_schedule
from .select import default_results_path, load_result_rows, select_result_rows

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
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
    "sample_count",
    "mean_latency_ms",
    "stddev_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
)
MODE_COLORS = {
    "hybrid": "#1f6f8b",
    "runlist": "#e07a5f",
}
MODE_MARKERS = {
    "hybrid": "o",
    "runlist": "s",
}
MODE_LABELS = {
    "hybrid": "Hybrid",
    "runlist": "Runlist",
}
FAMILY_LABELS = {
    "tinybert_512": "TinyBERT",
    "baseline_768": "BERT-Base",
    "baseline_1024": "BERT-Large",
    "gpt2_small_768": "GPT-2 Small",
    "gpt2_medium_1024": "GPT-2 Medium",
}


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "latency_variation.csv"
    )


def default_plot_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "latency_variation_by_pattern.svg"
    )


def default_resume_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2]
        / "results_final"
        / "end_to_end"
        / output_path.name
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def _resume_execution_mode(value: object) -> str:
    execution_mode = str(value or "")
    return "hybrid" if execution_mode == "dataflow" else execution_mode


def _resume_workload_variant(row: dict[str, str]) -> str:
    workload_variant = str(row.get("workload_variant") or "")
    if workload_variant:
        return workload_variant
    study_case_id = str(row.get("study_case_id") or "")
    if study_case_id in FAMILY_SPECS:
        return FAMILY_SPECS[study_case_id].workload_variant
    return ""


def _row_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        int(float(str(row.get("seq_len") or 0))),
    )


def load_existing_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, str, str, int], dict[str, object]]:
    rows: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("execution_mode") or ""):
                    continue
                rows[_row_key(row)] = dict(row)
    return rows


def reusable_existing_row(
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]],
    *,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    seq_len: int,
    warmup_runs: int,
    runs_per_sample: int,
    selected_candidate_ids_json: str,
    selected_config_json: str,
) -> dict[str, object] | None:
    row = existing_rows.get((study_case_id, workload_variant, execution_mode, seq_len))
    if row is None:
        return None
    if str(row.get("run_status") or "") != "passed":
        return None
    if int(float(str(row.get("warmup_runs") or 0))) != int(warmup_runs):
        return None
    if int(float(str(row.get("runs_per_sample") or 0))) != int(runs_per_sample):
        return None
    if str(row.get("selected_candidate_ids_json") or "") != selected_candidate_ids_json:
        return None
    if str(row.get("selected_config_json") or "") != selected_config_json:
        return None
    return dict(row)


def _resolved_sampling(
    selected_row,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
) -> tuple[int, int]:
    scheduled_warmup_runs, scheduled_runs_per_sample = iteration_schedule(
        selected_row.seq_len
    )
    resolved_warmup_runs = (
        scheduled_warmup_runs
        if selected_row.warmup_runs in (None, 0)
        else int(selected_row.warmup_runs)
    )
    resolved_runs_per_sample = (
        scheduled_runs_per_sample
        if selected_row.runs_per_sample in (None, 0)
        else int(selected_row.runs_per_sample)
    )
    if warmup_runs is not None:
        resolved_warmup_runs = int(warmup_runs)
    if runs_per_sample is not None:
        resolved_runs_per_sample = int(runs_per_sample)
    return resolved_warmup_runs, resolved_runs_per_sample


def summarize_latency_samples(
    latency_samples_ms: list[float],
) -> dict[str, float | int | None]:
    if not latency_samples_ms:
        return {
            "sample_count": 0,
            "mean_latency_ms": None,
            "stddev_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
        }
    return {
        "sample_count": len(latency_samples_ms),
        "mean_latency_ms": statistics.fmean(latency_samples_ms),
        "stddev_latency_ms": (
            statistics.pstdev(latency_samples_ms)
            if len(latency_samples_ms) > 1
            else 0.0
        ),
        "min_latency_ms": min(latency_samples_ms),
        "max_latency_ms": max(latency_samples_ms),
    }


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    mode_filter: str,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]] | None = None,
    benchmark_fn=None,
    checkpoint_fn=None,
) -> list[dict[str, object]]:
    if benchmark_fn is None:
        benchmark_fn = benchmark_mode
    selected_rows = select_result_rows(
        load_result_rows(results_input),
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        mode_filter=mode_filter,
    )
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        resolved_warmup_runs, resolved_runs_per_sample = _resolved_sampling(
            selected_row,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
        )
        selected_candidate_ids_json = json.dumps(
            selected_row.selected_candidate_ids,
            sort_keys=True,
        )
        selected_config_json = json.dumps(
            selected_row.selected_config,
            sort_keys=True,
        )
        reused_row = reusable_existing_row(
            {} if existing_rows is None else existing_rows,
            study_case_id=selected_row.study_case_id,
            workload_variant=selected_row.workload_variant,
            execution_mode=selected_row.execution_mode,
            seq_len=selected_row.seq_len,
            warmup_runs=resolved_warmup_runs,
            runs_per_sample=resolved_runs_per_sample,
            selected_candidate_ids_json=selected_candidate_ids_json,
            selected_config_json=selected_config_json,
        )
        if reused_row is not None:
            rows.append(
                {field: reused_row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )
            if checkpoint_fn is not None:
                checkpoint_fn(rows)
            continue
        LOGGER.info(
            "Running latency variation benchmark for %s seq_len=%s mode=%s",
            selected_row.study_case_id,
            selected_row.seq_len,
            selected_row.execution_mode,
        )
        result = benchmark_fn(
            selected_row.execution_mode,
            case.workload,
            warmup_runs=resolved_warmup_runs,
            runs_per_sample=resolved_runs_per_sample,
            seed=seed,
            power_backend="none",
            operator_config=selected_row.selected_config,
            capture_latencies=True,
        )
        latency_samples_ms = [
            float(value) for value in result.get("latency_samples_ms", []) or []
        ]
        rows.append(
            {
                "study_id": "end_to_end_latency_variation",
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "hidden_size": selected_row.hidden_size,
                "intermediate_size": selected_row.intermediate_size,
                "num_attention_heads": selected_row.num_attention_heads,
                "attention_head_size": selected_row.attention_head_size,
                "warmup_runs": resolved_warmup_runs,
                "runs_per_sample": resolved_runs_per_sample,
                **summarize_latency_samples(latency_samples_ms),
                "validation_error_count": result.get("validation_error_count", ""),
                "run_status": result.get("run_status", ""),
                "failure_message": result.get("failure_message", ""),
                "selected_candidate_ids_json": selected_candidate_ids_json,
                "selected_config_json": selected_config_json,
            }
        )
        if checkpoint_fn is not None:
            checkpoint_fn(rows)
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


def render_plot(rows: list[dict[str, object]]) -> plt.Figure:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )
    fig, axes = plt.subplots(
        1,
        len(FAMILY_IDS),
        figsize=(8 * len(FAMILY_IDS), 8),
        sharey=True,
    )
    if len(FAMILY_IDS) == 1:
        axes = [axes]

    for ax, family_id in zip(axes, FAMILY_IDS, strict=True):
        family_rows = [
            row
            for row in rows
            if str(row.get("study_case_id") or "") == family_id
            and str(row.get("run_status") or "") == "passed"
        ]
        for execution_mode in EXECUTION_MODES:
            mode_rows = [
                row
                for row in family_rows
                if str(row.get("execution_mode") or "") == execution_mode
            ]
            if not mode_rows:
                continue
            mode_rows = sorted(mode_rows, key=lambda row: int(row["seq_len"]))
            x_values = [int(row["seq_len"]) for row in mode_rows]
            y_values = [float(row["mean_latency_ms"]) for row in mode_rows]
            y_errors = [float(row["stddev_latency_ms"]) for row in mode_rows]
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                color=MODE_COLORS[execution_mode],
                marker=MODE_MARKERS[execution_mode],
                linewidth=2.6,
                markersize=8,
                capsize=4,
            )

        sample_row = family_rows[0] if family_rows else None
        title = family_id
        if sample_row is not None:
            title = FAMILY_LABELS[str(sample_row["study_case_id"])]
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQUENCE_LADDER)
        ax.set_xticklabels([str(value) for value in SEQUENCE_LADDER], rotation=0)
        ax.set_xlabel("Sequence Length", fontsize=15)
        ax.set_ylabel("Latency (ms)", fontsize=15)
        ax.set_title(title, loc="left", fontsize=18, pad=6)
        ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)
        if ax is not axes[0]:
            ax.set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODE_COLORS[execution_mode],
            marker=MODE_MARKERS[execution_mode],
            linewidth=2.6,
            markersize=8,
            label=MODE_LABELS[execution_mode],
        )
        for execution_mode in EXECUTION_MODES
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.87, 0.5),
        fontsize=13,
    )
    fig.suptitle(
        "End-to-End Latency with Run-to-Run Variation",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 0.84, 0.92])
    return fig


def write_plot(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_plot(rows)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark selected end-to-end NPU configs and capture run-to-run latency variation."
    )
    parser.add_argument(
        "--results-input",
        type=Path,
        default=default_results_path(),
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
        "--mode",
        choices=[*EXECUTION_MODES, "all"],
        default="all",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--plot-output", type=Path, default=default_plot_path())
    parser.add_argument("--resume-input", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    warn_if_npu_power_mode_not_turbo(
        LOGGER,
        study_name="end-to-end latency variation",
    )
    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="end-to-end latency variation",
    ):
        resume_paths: tuple[Path, ...] = tuple()
        if not args.no_resume:
            if args.resume_input is not None:
                resume_paths = (args.resume_input.expanduser(),)
            else:
                resume_paths = default_resume_paths(output_path)
        existing_rows = load_existing_rows(resume_paths)
        if resume_paths and existing_rows:
            LOGGER.info(
                "Reusing %d latency-variation rows from %s",
                len(existing_rows),
                ", ".join(str(path) for path in resume_paths),
            )
        rows = build_rows(
            results_input=args.results_input.expanduser(),
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            mode_filter=str(args.mode),
            warmup_runs=args.warmup_iters,
            runs_per_sample=args.timed_iters,
            seed=int(args.seed),
            existing_rows=existing_rows,
            benchmark_fn=benchmark_mode_subprocess,
            checkpoint_fn=lambda current_rows: write_rows(output_path, current_rows),
        )
        write_rows(output_path, rows)
        write_plot(args.plot_output.expanduser(), rows)
        LOGGER.info("Wrote %d latency-variation rows to %s", len(rows), output_path)
        LOGGER.info("Wrote latency-variation plot to %s", args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
