#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ..plot_families import (
    PLOT_FAMILY_GRID_SHAPE,
    PLOT_FAMILY_ORDER,
    ordered_plot_families,
    plot_family_label,
)

MODE_ORDER = ["hybrid", "runlist", "offload"]
SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MODE_LABELS = {
    "hybrid": "Coarse runlist",
    "runlist": "Runlist",
    "offload": "Offload",
}
MODE_COLORS = {
    "hybrid": "#1f6f8b",
    "runlist": "#e07a5f",
    "offload": "#6c9a3b",
}
MODE_MARKERS = {
    "hybrid": "o",
    "runlist": "s",
    "offload": "^",
}
LONG_SEQUENCE_SUFFIX_TEMPLATE = "after_{threshold}_tokens"


def default_results_csv() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "results_all_power.csv"
    )


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "end_to_end"


def _sequence_suffix(min_seq_len_exclusive: int | None = None) -> str:
    if min_seq_len_exclusive is None:
        return ""
    return f"_{LONG_SEQUENCE_SUFFIX_TEMPLATE.format(threshold=min_seq_len_exclusive)}"


def _sequence_order(
    df: pd.DataFrame,
    min_seq_len_exclusive: int | None = None,
) -> list[int]:
    present = {int(seq_len) for seq_len in df["seq_len"].tolist()}
    ordered = [
        seq_len
        for seq_len in SEQ_ORDER
        if seq_len in present
        and (min_seq_len_exclusive is None or seq_len > int(min_seq_len_exclusive))
    ]
    extras = sorted(set(present) - set(ordered))
    return ordered + extras


def variant_stem(
    metric: str,
    variant: str,
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> str:
    stems = {
        "throughput": {
            "standard": "effective_gflops_per_second_by_pattern",
            "slides": "effective_gflops_per_second_by_pattern_slides",
        },
        "latency": {
            "standard": "latency_by_pattern",
            "slides": "latency_by_pattern_slides",
        },
    }
    stem = f"{stems[metric][variant]}{_sequence_suffix(min_seq_len_exclusive)}"
    if metric == "latency" and y_scale == "linear":
        return f"{stem}_linear"
    return stem


def load_plot_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = df[df["run_status"] == "passed"].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["effective_gflops_per_sec"] = df["effective_gflops_per_sec"].astype(float)
    df["avg_latency_ms"] = df["avg_latency_ms"].astype(float)
    df = df[df["execution_mode"].isin(MODE_ORDER)].copy()
    df["execution_mode"] = pd.Categorical(
        df["execution_mode"], MODE_ORDER, ordered=True
    )
    df["study_case_id"] = pd.Categorical(
        df["study_case_id"], PLOT_FAMILY_ORDER, ordered=True
    )
    df = df.sort_values(["study_case_id", "execution_mode", "seq_len"]).reset_index(
        drop=True
    )

    return df


def render_plot(
    df: pd.DataFrame,
    *,
    metric: str = "throughput",
    variant: str = "standard",
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> plt.Figure:
    if min_seq_len_exclusive is not None:
        df = df[df["seq_len"] > int(min_seq_len_exclusive)].copy()
    seq_order = _sequence_order(df, min_seq_len_exclusive)
    metric_column = {
        "throughput": "effective_gflops_per_sec",
        "latency": "avg_latency_ms",
    }[metric]
    ylabel = {
        "throughput": "Effective Throughput (GFLOP/s)",
        "latency": "Latency (ms, log scale)" if y_scale == "log" else "Latency (ms)",
    }[metric]
    title = {
        "throughput": "Effective Throughput Comparison Across Execution Boundaries",
        "latency": "Latency Comparison Across Execution Boundaries",
    }[metric]
    if min_seq_len_exclusive is not None:
        title = f"{title} (>{int(min_seq_len_exclusive)} Tokens)"
    if variant == "slides":
        context = "poster"
        figsize = (20, 11.5)
        family_title_size = 20
        axis_label_size = 18
        tick_label_size = 14
        legend_font_size = 16
        suptitle_size = 30
        title_y = 0.985
    else:
        context = "talk"
        figsize = (18, 10)
        family_title_size = 18
        axis_label_size = 15
        tick_label_size = 12
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.975

    sns.set_theme(
        style="whitegrid",
        context=context,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )

    present_family_ids = set(df["study_case_id"].astype(str).tolist())
    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=figsize,
        sharey=True,
    )
    axes_grid = axes_obj

    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        if family.family_id not in present_family_ids:
            ax.set_axis_off()
            continue

        family_df = df[df["study_case_id"] == family.family_id].copy()

        for mode in MODE_ORDER:
            mode_df = family_df[family_df["execution_mode"] == mode].copy()
            ax.plot(
                mode_df["seq_len"],
                mode_df[metric_column],
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linewidth=2.6,
                markersize=8 if variant == "standard" else 9,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_order)
        ax.set_xticklabels([str(seq_len) for seq_len in seq_order], rotation=0)
        if metric == "latency" and y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Sequence Length", fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)
        ax.set_title(
            plot_family_label(family.family_id),
            loc="left",
            fontsize=family_title_size,
            pad=6,
        )
        ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if family.col_index != 0:
            ax.set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            linewidth=2.6,
            markersize=8 if variant == "standard" else 9,
            label=MODE_LABELS[mode],
        )
        for mode in MODE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.915, 0.5),
        fontsize=legend_font_size,
    )
    fig.suptitle(
        title,
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.03, 0.89, 0.95])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an effective GFLOP/s comparison across execution boundaries."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results_csv(),
        help="Path to end-to-end results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for rendered chart files",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem, without extension",
    )
    parser.add_argument(
        "--variant",
        choices=["standard", "slides"],
        default="standard",
        help="Chart layout preset",
    )
    parser.add_argument(
        "--metric",
        choices=["throughput", "latency"],
        default="throughput",
        help="Metric to render",
    )
    parser.add_argument(
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scaling mode for latency plots",
    )
    parser.add_argument(
        "--min-seq-len-exclusive",
        type=int,
        default=None,
        help="Only include sequence lengths strictly greater than this value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_plot_rows(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_plot(
        df,
        metric=args.metric,
        variant=args.variant,
        y_scale=args.y_scale,
        min_seq_len_exclusive=args.min_seq_len_exclusive,
    )
    stem = args.stem or variant_stem(
        args.metric,
        args.variant,
        args.y_scale,
        args.min_seq_len_exclusive,
    )
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
