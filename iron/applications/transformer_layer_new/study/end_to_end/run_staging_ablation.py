#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from .cases import FAMILY_IDS, SEQUENCE_LADDER
from .select import default_results_path as default_end_to_end_results_path

LOGGER = logging.getLogger(__name__)

STAGING_BLOCK_KINDS: tuple[str, ...] = ("mha_out_proj", "ffn")
BLOCK_LABELS = {
    "mha_out_proj": "MHA Out Proj",
    "ffn": "FFN",
}
BLOCK_COLORS = {
    "mha_out_proj": "#1f6f8b",
    "ffn": "#81b29a",
}
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "family_id",
    "seq_len",
    "block_kind",
    "source_staging_depth",
    "best_staging_depth",
    "source_avg_latency_ms",
    "best_avg_latency_ms",
    "speedup_vs_source_depth",
    "speedup_vs_depth1",
    "dataflow_end_to_end_latency_ms",
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


def _optional_bool(value: object) -> bool | None:
    if value in (None, "", "None"):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rows(
    *,
    end_to_end_results: Path,
    staging_results: Path,
    family_filter: str,
) -> list[dict[str, object]]:
    dataflow_latency_by_group: dict[tuple[str, int], float] = {}
    for row in _load_csv_rows(end_to_end_results):
        if row.get("backend") != "npu":
            continue
        if row.get("execution_mode") != "dataflow":
            continue
        if row.get("run_status") != "passed":
            continue
        family_id = str(row.get("study_case_id") or "")
        seq_len = _optional_int(row.get("seq_len"))
        latency_ms = _optional_float(row.get("avg_latency_ms"))
        if not family_id or seq_len is None or latency_ms is None:
            continue
        dataflow_latency_by_group[(family_id, seq_len)] = latency_ms

    grouped_staging_rows: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for row in _load_csv_rows(staging_results):
        family_id = str(row.get("family_id") or "")
        if family_filter != "all" and family_id != family_filter:
            continue
        if row.get("run_status") != "passed":
            continue
        seq_len = _optional_int(row.get("seq_len"))
        block_kind = str(row.get("block_kind") or "")
        avg_latency_ms = _optional_float(row.get("avg_latency_ms"))
        if seq_len is None or not block_kind or avg_latency_ms is None:
            continue
        grouped_staging_rows.setdefault((family_id, seq_len, block_kind), []).append(
            row
        )

    rows: list[dict[str, object]] = []
    for key in sorted(grouped_staging_rows):
        family_id, seq_len, block_kind = key
        source_depth = None
        best_row = None
        source_row = None
        for row in grouped_staging_rows[key]:
            current_source_depth = _optional_int(row.get("source_staging_depth"))
            if source_depth is None and current_source_depth is not None:
                source_depth = current_source_depth
            if _optional_bool(row.get("is_best_depth")) is True:
                best_row = row
            if (
                current_source_depth is not None
                and _optional_int(row.get("staging_depth")) == current_source_depth
            ):
                source_row = row
        if best_row is None:
            best_row = min(
                grouped_staging_rows[key],
                key=lambda row: float(
                    _optional_float(row.get("avg_latency_ms")) or 0.0
                ),
            )
        if source_row is None:
            continue

        source_latency_ms = float(
            _optional_float(source_row.get("avg_latency_ms")) or 0.0
        )
        best_latency_ms = float(_optional_float(best_row.get("avg_latency_ms")) or 0.0)
        if source_latency_ms <= 0 or best_latency_ms <= 0:
            continue
        rows.append(
            {
                "study_id": "end_to_end_staging_ablation",
                "family_id": family_id,
                "seq_len": seq_len,
                "block_kind": block_kind,
                "source_staging_depth": source_depth,
                "best_staging_depth": _optional_int(best_row.get("staging_depth")),
                "source_avg_latency_ms": source_latency_ms,
                "best_avg_latency_ms": best_latency_ms,
                "speedup_vs_source_depth": source_latency_ms / best_latency_ms,
                "speedup_vs_depth1": _optional_float(best_row.get("speedup_vs_depth1")),
                "dataflow_end_to_end_latency_ms": dataflow_latency_by_group.get(
                    (family_id, seq_len)
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


def render_plot(rows: list[dict[str, object]]) -> plt.Figure:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )
    fig, axes = plt.subplots(1, len(FAMILY_IDS), figsize=(20, 8), sharey=True)
    if len(FAMILY_IDS) == 1:
        axes = [axes]

    for ax, family_id in zip(axes, FAMILY_IDS, strict=True):
        family_rows = [
            row for row in rows if str(row.get("family_id") or "") == family_id
        ]
        x_positions = list(range(len(SEQUENCE_LADDER)))
        bar_width = 0.28
        secondary_ax = ax.twinx()

        for block_index, block_kind in enumerate(STAGING_BLOCK_KINDS):
            block_rows = {
                int(row["seq_len"]): float(row["speedup_vs_source_depth"])
                for row in family_rows
                if str(row.get("block_kind") or "") == block_kind
                and row.get("speedup_vs_source_depth") not in (None, "", "None")
            }
            ax.bar(
                [
                    x_pos - (bar_width / 2.0) + (block_index * bar_width)
                    for x_pos in x_positions
                ],
                [block_rows.get(seq_len, 0.0) for seq_len in SEQUENCE_LADDER],
                width=bar_width * 0.92,
                color=BLOCK_COLORS[block_kind],
                edgecolor="white",
                linewidth=0.7,
                label=BLOCK_LABELS[block_kind],
            )

        latency_map = {}
        for row in family_rows:
            latency_ms = _optional_float(row.get("dataflow_end_to_end_latency_ms"))
            if latency_ms is not None:
                latency_map[int(row["seq_len"])] = latency_ms
        secondary_ax.plot(
            x_positions,
            [latency_map.get(seq_len) for seq_len in SEQUENCE_LADDER],
            color="#3d405b",
            marker="o",
            linewidth=2.2,
            markersize=6,
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seq_len) for seq_len in SEQUENCE_LADDER], rotation=0)
        ax.set_xlabel("Context Length (tokens)", fontsize=15)
        ax.set_ylabel("Block Speedup vs Source Depth", fontsize=15)
        secondary_ax.set_ylabel("Dataflow End-to-End Latency (ms)", fontsize=15)
        ax.set_title(family_id, loc="left", fontsize=18, pad=12)
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.tick_params(axis="both", labelsize=12)
        secondary_ax.tick_params(axis="y", labelsize=12)

    legend_handles = [
        Patch(
            facecolor=BLOCK_COLORS[block_kind],
            edgecolor="none",
            label=BLOCK_LABELS[block_kind],
        )
        for block_kind in STAGING_BLOCK_KINDS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#3d405b",
            marker="o",
            linewidth=2.2,
            markersize=6,
            label="Dataflow End-to-End Latency",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=13,
    )
    fig.suptitle(
        "Staging Ablation and End-to-End Dataflow Latency",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def write_plot(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_plot(rows)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize staging-depth effects alongside end-to-end dataflow latency."
    )
    parser.add_argument(
        "--end-to-end-results",
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
        end_to_end_results=args.end_to_end_results.expanduser(),
        staging_results=args.staging_results.expanduser(),
        family_filter=str(args.family),
    )
    write_rows(args.output.expanduser(), rows)
    write_plot(args.plot_output.expanduser(), rows)
    LOGGER.info("Wrote %d staging-ablation rows to %s", len(rows), args.output)
    LOGGER.info("Wrote staging-ablation plot to %s", args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
