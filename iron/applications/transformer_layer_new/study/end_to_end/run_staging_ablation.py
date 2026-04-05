#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from .cases import FAMILY_IDS, SEQUENCE_LADDER, get_case
from .modes import benchmark_mode
from .run import iteration_schedule
from .select import (
    default_results_path as default_end_to_end_results_path,
    load_result_rows,
    select_result_rows,
)

LOGGER = logging.getLogger(__name__)

STAGING_BLOCK_KINDS: tuple[str, ...] = ("mha_out_proj", "ffn")
BLOCK_LABELS = {
    "mha_out_proj": "Dataflow Pattern with MHA + Output Projection Staging Ablation",
    "ffn": "Dataflow Pattern with FFN Staging Ablation",
}
FAMILY_LABELS = {
    "baseline_768": "Hidden Size = 768 / FFN Dim = 3072 / Heads = 12",
    "baseline_1024": "Hidden Size = 1024 / FFN Dim = 4096 / Heads = 16",
}
SEQ_COLORS = {
    seq_len: color
    for seq_len, color in zip(
        SEQUENCE_LADDER,
        sns.color_palette("crest", n_colors=len(SEQUENCE_LADDER)),
        strict=True,
    )
}
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "execution_mode",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "block_kind",
    "source_staging_depth",
    "staging_depth",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "compile_setup_time_ms",
    "effective_gflops_per_sec",
    "speedup_vs_source_depth",
    "speedup_vs_depth1",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
    "is_best_depth",
)


def default_staging_results_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "memory_tile_staging"
        / "results.csv"
    )


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "staging_ablation.csv"
    )


def default_plot_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "staging_ablation.svg"
    )


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


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


def _depth_key(block_kind: str) -> str:
    if block_kind == "mha_out_proj":
        return "o_proj_acc_depth"
    return "down_proj_depth"


def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        return tuple()
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


def _supported_staging_depths_from_config(
    selected_row, *, block_kind: str
) -> tuple[int, ...]:
    block_config = selected_row.selected_config.get(block_kind, {})
    if block_kind == "mha_out_proj":
        emb_tile = _optional_int(block_config.get("emb_tile"))
        if emb_tile in (None, 0) or selected_row.hidden_size % int(emb_tile) != 0:
            return tuple()
        return _divisors(selected_row.hidden_size // int(emb_tile))

    tile_k = _optional_int(block_config.get("tile_k"))
    if tile_k in (None, 0) or selected_row.hidden_size % int(tile_k) != 0:
        return tuple()
    return _divisors(selected_row.hidden_size // int(tile_k))


def _source_staging_depth_from_selected_config(
    selected_row, *, block_kind: str
) -> int | None:
    return _optional_int(
        selected_row.selected_config.get(block_kind, {}).get(_depth_key(block_kind))
    )


def _load_staging_depth_overrides(
    staging_results: Path,
) -> dict[tuple[str, int, str], tuple[int, ...]]:
    if not staging_results.exists():
        return {}

    grouped_depths: dict[tuple[str, int, str], set[int]] = {}
    for row in load_result_rows(staging_results):
        if row.get("run_status") != "passed":
            continue
        family_id = str(row.get("family_id") or "")
        seq_len = _optional_int(row.get("seq_len"))
        block_kind = str(row.get("block_kind") or "")
        staging_depth = _optional_int(row.get("staging_depth"))
        if (
            not family_id
            or seq_len is None
            or block_kind not in STAGING_BLOCK_KINDS
            or staging_depth is None
        ):
            continue
        grouped_depths.setdefault((family_id, seq_len, block_kind), set()).add(
            staging_depth
        )
    return {key: tuple(sorted(depths)) for key, depths in grouped_depths.items()}


def _candidate_staging_depths(
    selected_row,
    *,
    block_kind: str,
    staging_depth_overrides: dict[tuple[str, int, str], tuple[int, ...]],
) -> tuple[int, ...]:
    supported_depths = _supported_staging_depths_from_config(
        selected_row,
        block_kind=block_kind,
    )
    if not supported_depths:
        return tuple()

    source_depth = _source_staging_depth_from_selected_config(
        selected_row,
        block_kind=block_kind,
    )
    override_depths = staging_depth_overrides.get(
        (selected_row.study_case_id, selected_row.seq_len, block_kind),
        tuple(),
    )
    if override_depths:
        depth_set = {depth for depth in override_depths if depth in supported_depths}
        if source_depth is not None and source_depth in supported_depths:
            depth_set.add(source_depth)
        if 1 in supported_depths:
            depth_set.add(1)
        return tuple(sorted(depth_set))
    return supported_depths


def _config_with_staging_depth(
    selected_row,
    *,
    block_kind: str,
    staging_depth: int,
) -> dict[str, dict[str, object]]:
    config = copy.deepcopy(selected_row.selected_config)
    config.setdefault(block_kind, {})
    config[block_kind][_depth_key(block_kind)] = int(staging_depth)
    return config


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        row["is_best_depth"] = False
        key = (str(row["study_case_id"]), int(row["seq_len"]), str(row["block_kind"]))
        grouped_rows.setdefault(key, []).append(row)

    for group_rows in grouped_rows.values():
        successful_rows = [
            row
            for row in group_rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] not in ("", None)
        ]
        if not successful_rows:
            continue
        best_row = min(successful_rows, key=lambda row: float(row["avg_latency_ms"]))
        best_row["is_best_depth"] = True


def annotate_speedups(rows: list[dict[str, object]]) -> None:
    depth1_latency_by_group: dict[tuple[str, int, str], float] = {}
    source_latency_by_group: dict[tuple[str, int, str], float] = {}

    for row in rows:
        row["speedup_vs_source_depth"] = ""
        row["speedup_vs_depth1"] = ""
        if row["run_status"] != "passed" or row["avg_latency_ms"] in ("", None):
            continue
        group = (str(row["study_case_id"]), int(row["seq_len"]), str(row["block_kind"]))
        latency = float(row["avg_latency_ms"])
        if int(row["staging_depth"]) == 1:
            depth1_latency_by_group[group] = latency
        if int(row["staging_depth"]) == int(row["source_staging_depth"]):
            source_latency_by_group[group] = latency

    for row in rows:
        if row["run_status"] != "passed" or row["avg_latency_ms"] in ("", None):
            continue
        group = (str(row["study_case_id"]), int(row["seq_len"]), str(row["block_kind"]))
        current_latency = float(row["avg_latency_ms"])
        depth1_latency = depth1_latency_by_group.get(group)
        source_latency = source_latency_by_group.get(group)
        if depth1_latency is not None and depth1_latency > 0:
            row["speedup_vs_depth1"] = depth1_latency / current_latency
        if source_latency is not None and source_latency > 0:
            row["speedup_vs_source_depth"] = source_latency / current_latency


def build_rows(
    *,
    results_input: Path,
    staging_results: Path,
    family_filter: str,
    seq_len_filter: str,
    block_filter: str,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    benchmark_fn: Callable[..., dict[str, object]] = benchmark_mode,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        load_result_rows(results_input),
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter="dataflow",
    )
    staging_depth_overrides = _load_staging_depth_overrides(staging_results)

    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        resolved_warmup_runs, resolved_runs_per_sample = _resolved_sampling(
            selected_row,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
        )
        for block_kind in STAGING_BLOCK_KINDS:
            if block_filter != "all" and block_kind != block_filter:
                continue

            source_depth = _source_staging_depth_from_selected_config(
                selected_row,
                block_kind=block_kind,
            )
            if source_depth is None:
                continue
            candidate_depths = _candidate_staging_depths(
                selected_row,
                block_kind=block_kind,
                staging_depth_overrides=staging_depth_overrides,
            )
            if not candidate_depths:
                continue

            for staging_depth in candidate_depths:
                operator_config = _config_with_staging_depth(
                    selected_row,
                    block_kind=block_kind,
                    staging_depth=staging_depth,
                )
                LOGGER.info(
                    "Running end-to-end staging ablation for %s seq_len=%s block=%s depth=%s",
                    selected_row.study_case_id,
                    selected_row.seq_len,
                    block_kind,
                    staging_depth,
                )
                result = benchmark_fn(
                    "dataflow",
                    case.workload,
                    warmup_runs=resolved_warmup_runs,
                    runs_per_sample=resolved_runs_per_sample,
                    seed=seed,
                    power_backend="none",
                    operator_config=operator_config,
                )
                rows.append(
                    {
                        "study_id": "end_to_end_staging_ablation",
                        "study_case_id": selected_row.study_case_id,
                        "study_case_label": selected_row.study_case_label,
                        "execution_mode": "dataflow",
                        "seq_len": selected_row.seq_len,
                        "hidden_size": selected_row.hidden_size,
                        "intermediate_size": selected_row.intermediate_size,
                        "num_attention_heads": selected_row.num_attention_heads,
                        "attention_head_size": selected_row.attention_head_size,
                        "block_kind": block_kind,
                        "source_staging_depth": source_depth,
                        "staging_depth": staging_depth,
                        "warmup_runs": resolved_warmup_runs,
                        "runs_per_sample": resolved_runs_per_sample,
                        "avg_latency_ms": result.get("avg_latency_ms"),
                        "compile_setup_time_ms": result.get("compile_setup_time_ms"),
                        "effective_gflops_per_sec": result.get(
                            "effective_gflops_per_sec"
                        ),
                        "validation_error_count": result.get(
                            "validation_error_count",
                            "",
                        ),
                        "run_status": result.get("run_status", ""),
                        "failure_message": result.get("failure_message", ""),
                        "selected_candidate_ids_json": json.dumps(
                            selected_row.selected_candidate_ids,
                            sort_keys=True,
                        ),
                        "selected_config_json": json.dumps(
                            operator_config,
                            sort_keys=True,
                        ),
                    }
                )

    annotate_speedups(rows)
    mark_best_rows(rows)
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
    plt.rcParams["svg.fonttype"] = "none"

    successful_rows = [
        row
        for row in rows
        if row.get("run_status") == "passed"
        and row.get("avg_latency_ms") not in ("", None)
    ]
    if not successful_rows:
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_axis_off()
        fig.text(
            0.5,
            0.57,
            "End-to-End Dataflow Latency by Block Staging Depth",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.42,
            "No data available",
            ha="center",
            va="center",
            fontsize=18,
        )
        fig.patch.set_facecolor("#f7f5f2")
        return fig

    fig, axes = plt.subplots(
        len(STAGING_BLOCK_KINDS),
        len(FAMILY_IDS),
        figsize=(20, 12),
        sharex=False,
        sharey="row",
    )
    if len(STAGING_BLOCK_KINDS) == 1:
        axes = [axes]

    for row_index, block_kind in enumerate(STAGING_BLOCK_KINDS):
        for col_index, family_id in enumerate(FAMILY_IDS):
            ax = axes[row_index][col_index]
            panel_rows = [
                row
                for row in successful_rows
                if str(row.get("block_kind")) == block_kind
                and str(row.get("study_case_id")) == family_id
            ]
            panel_depths = sorted(
                {
                    int(row["staging_depth"])
                    for row in panel_rows
                    if _optional_int(row.get("staging_depth")) is not None
                }
            )
            for seq_len in SEQUENCE_LADDER:
                seq_rows = [row for row in panel_rows if int(row["seq_len"]) == seq_len]
                if not seq_rows:
                    continue
                seq_rows.sort(key=lambda row: int(row["staging_depth"]))
                ax.plot(
                    [int(row["staging_depth"]) for row in seq_rows],
                    [float(row["avg_latency_ms"]) for row in seq_rows],
                    label=str(seq_len),
                    color=SEQ_COLORS[seq_len],
                    marker="o",
                    linewidth=2.4,
                    markersize=7,
                )

            ax.set_yscale("log")
            ax.set_xticks(panel_depths)
            ax.set_xlabel("Staging Depth", fontsize=14)
            ax.set_ylabel("End-to-End Latency (ms, log scale)", fontsize=14)
            ax.set_title(
                f"{BLOCK_LABELS[block_kind]}\n{FAMILY_LABELS[family_id]}",
                loc="left",
                fontsize=16,
                pad=10,
            )
            ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
            ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.3)
            ax.tick_params(axis="both", labelsize=11)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SEQ_COLORS[seq_len],
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=str(seq_len),
        )
        for seq_len in SEQUENCE_LADDER
        if any(int(row["seq_len"]) == seq_len for row in successful_rows)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(5, len(legend_handles) or 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        title="Context Length (tokens)",
        fontsize=12,
        title_fontsize=13,
    )
    fig.suptitle(
        "End-to-End Dataflow Latency by Block Staging Depth",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    return fig


def write_plot(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_plot(rows)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end dataflow staging-depth ablations for selected NPU configs."
    )
    parser.add_argument(
        "--results-input",
        "--end-to-end-results",
        dest="results_input",
        type=Path,
        default=default_end_to_end_results_path(),
    )
    parser.add_argument(
        "--staging-results",
        type=Path,
        default=default_staging_results_path(),
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
        "--block",
        choices=[*STAGING_BLOCK_KINDS, "all"],
        default="all",
    )
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--plot-output", type=Path, default=default_plot_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    rows = build_rows(
        results_input=args.results_input.expanduser(),
        staging_results=args.staging_results.expanduser(),
        family_filter=str(args.family),
        seq_len_filter=str(args.seq_len),
        block_filter=str(args.block),
        warmup_runs=args.warmup_runs,
        runs_per_sample=args.runs_per_sample,
        seed=int(args.seed),
    )
    write_rows(args.output.expanduser(), rows)
    write_plot(args.plot_output.expanduser(), rows)
    LOGGER.info("Wrote %d staging-ablation rows to %s", len(rows), args.output)
    LOGGER.info("Wrote staging-ablation plot to %s", args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
