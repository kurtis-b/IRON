#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from matplotlib import pyplot as plt


BREAKDOWN_FIELDS = (
    ("npu_gemm_time_sec", "NPU GEMM", "#1f6f8b"),
    ("host_attention_time_sec", "Host Attention", "#b85c38"),
    ("host_elementwise_time_sec", "Host Elementwise", "#6c9a3b"),
    ("transfer_time_sec", "Transfers", "#7b8cde"),
)


def default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "offload_breakdown.csv"
    )


def default_output_path() -> Path:
    return default_input_path().with_suffix(".svg")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_plot(rows: list[dict[str, str]]) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["svg.fonttype"] = "none"
    successful_rows = [
        row
        for row in rows
        if str(row.get("run_status") or "") == "passed"
        and str(row.get("query_block_size") or "").strip()
    ]
    if not successful_rows:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_axis_off()
        fig.text(0.5, 0.5, "No offload partitioning data available", ha="center")
        return fig

    case_labels = []
    grouped_rows: list[dict[str, str]] = []
    for row in sorted(
        successful_rows,
        key=lambda item: (
            str(item.get("study_case_id") or ""),
            int(float(str(item.get("seq_len") or 0))),
            int(float(str(item.get("query_block_size") or 0))),
        ),
    ):
        case_labels.append(
            f"{row['study_case_id']}\nL={row['seq_len']}\nQ={row['query_block_size']}"
        )
        grouped_rows.append(row)

    fig, ax = plt.subplots(figsize=(max(12, len(grouped_rows) * 1.4), 7))
    bottoms = [0.0 for _ in grouped_rows]
    for field_name, label, color in BREAKDOWN_FIELDS:
        values = [
            float(str(row.get(field_name) or 0.0))
            for row in grouped_rows
        ]
        ax.bar(case_labels, values, bottom=bottoms, color=color, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_ylabel("Seconds")
    ax.set_title("Offload Query-Block Runtime Breakdown")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stacked runtime-breakdown plot for the offload query-block sweep."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _load_rows(args.input.expanduser())
    fig = render_plot(rows)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
