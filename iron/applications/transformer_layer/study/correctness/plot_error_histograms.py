#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from matplotlib import pyplot as plt


def default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "error_distribution.csv"
    )


def default_output_path() -> Path:
    return default_input_path().with_name("error_histograms.svg")


def render_plot(rows: list[dict[str, str]]) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["svg.fonttype"] = "none"
    successful_rows = [
        row for row in rows if str(row.get("run_status") or "") == "passed"
    ]
    if not successful_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        fig.text(0.5, 0.5, "No correctness histogram data available", ha="center")
        return fig

    fig, axes = plt.subplots(
        len(successful_rows),
        1,
        figsize=(12, max(4, len(successful_rows) * 3.5)),
        squeeze=False,
    )
    for axis, row in zip(axes.flatten(), successful_rows, strict=True):
        edges = json.loads(str(row.get("abs_histogram_edges_json") or "[]"))
        counts = json.loads(str(row.get("abs_histogram_counts_json") or "[]"))
        if len(edges) < 2 or not counts:
            axis.set_axis_off()
            continue
        widths = [edges[index + 1] - edges[index] for index in range(len(edges) - 1)]
        axis.bar(edges[:-1], counts, width=widths, align="edge", color="#1f6f8b")
        axis.set_title(
            f"{row['study_case_id']} {row['execution_mode']} L={row['seq_len']}",
            loc="left",
        )
        axis.set_xlabel("Absolute Error")
        axis.set_ylabel("Count")
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render absolute-error histograms from the correctness error-distribution CSV."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.input.expanduser().open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig = render_plot(rows)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
