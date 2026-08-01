#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from ..plot_families import (
    PLOT_FAMILY_GRID_SHAPE,
    ordered_plot_families,
    plot_family_label,
)

SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BLOCK_ORDER = ["QKV Proj", "MHAO", "Add + Norm (x2)", "FFN"]
BLOCK_COLORS = {
    "QKV Proj": "#1f6f8b",
    "MHAO": "#e07a5f",
    "Add + Norm (x2)": "#3d405b",
    "FFN": "#81b29a",
}
PATTERN_COLOR = "#d8d4cf"
PATTERN_EDGE = "#2b2b2b"
BLOCK_KIND_MAP = {
    "qkv_proj": "QKV Proj",
    "mha_out_proj": "MHAO",
    "add_norm1": "Add + Norm (x2)",
    "add_norm2": "Add + Norm (x2)",
    "ln1": "Add + Norm (x2)",
    "add_norm": "Add + Norm (x2)",
    "ffn": "FFN",
    "add": "Add + Norm (x2)",
}
LONG_SEQUENCE_SUFFIX_TEMPLATE = "after_{threshold}_tokens"


def default_results_csv() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "results_all_power.csv"
    )


def default_tuning_csv() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "tuning_all_power.csv"
    )


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "end_to_end"


def _sequence_suffix(min_seq_len_exclusive: int | None = None) -> str:
    if min_seq_len_exclusive is None:
        return ""
    return f"_{LONG_SEQUENCE_SUFFIX_TEMPLATE.format(threshold=min_seq_len_exclusive)}"


def _sequence_order(
    pattern_df: pd.DataFrame,
    min_seq_len_exclusive: int | None = None,
) -> list[int]:
    present = {int(seq_len) for seq_len in pattern_df["seq_len"].tolist()}
    ordered = [
        seq_len
        for seq_len in SEQ_ORDER
        if seq_len in present
        and (min_seq_len_exclusive is None or seq_len > int(min_seq_len_exclusive))
    ]
    extras = sorted(set(present) - set(ordered))
    return ordered + extras


def variant_stem(
    variant: str,
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> str:
    stem = {
        "standard": "dataflow_selected_blocks_vs_pattern_latency",
        "slides": "dataflow_selected_blocks_vs_pattern_latency_slides",
    }[variant]
    stem = f"{stem}{_sequence_suffix(min_seq_len_exclusive)}"
    if y_scale == "linear":
        return f"{stem}_linear"
    return stem


def load_plot_rows(
    results_csv: Path,
    tuning_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.read_csv(results_csv)
    tuning_df = pd.read_csv(tuning_csv)

    pattern_df = results_df[
        (results_df["execution_mode"] == "hybrid")
        & (results_df["run_status"] == "passed")
    ].copy()
    pattern_df["seq_len"] = pattern_df["seq_len"].astype(int)
    pattern_df["avg_latency_ms"] = pattern_df["avg_latency_ms"].astype(float)
    pattern_df = pattern_df.sort_values(["study_case_id", "seq_len"]).reset_index(
        drop=True
    )

    chosen_blocks = tuning_df[
        (tuning_df["execution_mode"] == "hybrid")
        & (tuning_df["is_operator_best"].astype(str) == "True")
        & (tuning_df["run_status"] == "passed")
    ].copy()
    chosen_blocks["seq_len"] = chosen_blocks["seq_len"].astype(int)
    chosen_blocks["avg_latency_ms"] = chosen_blocks["avg_latency_ms"].astype(float)
    chosen_blocks["block_label"] = chosen_blocks["internal_operator"].map(
        BLOCK_KIND_MAP
    )
    chosen_blocks = chosen_blocks.dropna(subset=["block_label"])

    block_df = (
        chosen_blocks.groupby(
            ["study_case_id", "seq_len", "block_label"], as_index=False
        )["avg_latency_ms"]
        .sum()
        .copy()
    )

    return pattern_df, block_df


def render_plot(
    pattern_df: pd.DataFrame,
    block_df: pd.DataFrame,
    *,
    variant: str = "standard",
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> plt.Figure:
    if min_seq_len_exclusive is not None:
        threshold = int(min_seq_len_exclusive)
        pattern_df = pattern_df[pattern_df["seq_len"] > threshold].copy()
        block_df = block_df[block_df["seq_len"] > threshold].copy()
    seq_order = _sequence_order(pattern_df, min_seq_len_exclusive)
    if variant == "slides":
        context = "poster"
        figsize = (20, 11.5)
        family_title_size = 20
        axis_label_size = 18
        tick_label_size = 14
        legend_font_size = 16
        suptitle_size = 30
        title_y = 0.985
        factor_font_size = 11
    else:
        context = "talk"
        figsize = (18, 10)
        family_title_size = 18
        axis_label_size = 15
        tick_label_size = 12
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.975
        factor_font_size = 9

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

    present_family_ids = set(pattern_df["study_case_id"].astype(str).tolist())
    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=figsize,
        sharey=True,
    )
    axes_grid = axes_obj
    bar_width = 0.34
    x_positions = list(range(len(seq_order)))

    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        if family.family_id not in present_family_ids:
            ax.set_axis_off()
            continue

        family_patterns = pattern_df[
            pattern_df["study_case_id"] == family.family_id
        ].copy()
        family_blocks = block_df[block_df["study_case_id"] == family.family_id].copy()

        pattern_map = {
            int(row.seq_len): float(row.avg_latency_ms)
            for row in family_patterns.itertuples(index=False)
        }
        block_map = {
            (int(row.seq_len), str(row.block_label)): float(row.avg_latency_ms)
            for row in family_blocks.itertuples(index=False)
        }

        bottoms = [0.0] * len(seq_order)
        for block_label in BLOCK_ORDER:
            heights = [
                block_map.get((seq_len, block_label), 0.0) for seq_len in seq_order
            ]
            ax.bar(
                [x - (bar_width / 2) for x in x_positions],
                heights,
                width=bar_width,
                bottom=bottoms,
                color=BLOCK_COLORS[block_label],
                edgecolor="white",
                linewidth=0.7,
            )
            bottoms = [
                bottom + height for bottom, height in zip(bottoms, heights, strict=True)
            ]

        pattern_heights = [pattern_map.get(seq_len, 0.0) for seq_len in seq_order]
        ax.bar(
            [x + (bar_width / 2) for x in x_positions],
            pattern_heights,
            width=bar_width,
            color=PATTERN_COLOR,
            edgecolor=PATTERN_EDGE,
            linewidth=1.2,
        )

        pair_maxima = [
            max(stack_total, pattern_total)
            for stack_total, pattern_total in zip(bottoms, pattern_heights, strict=True)
        ]
        for x_pos, stack_total, pattern_total, pair_max in zip(
            x_positions,
            bottoms,
            pattern_heights,
            pair_maxima,
            strict=True,
        ):
            if stack_total <= 0 or pattern_total <= 0:
                continue
            factor = pattern_total / stack_total
            ax.text(
                x_pos + (bar_width / 2),
                pattern_total * 1.12,
                f"{factor:.2f}x",
                ha="center",
                va="bottom",
                fontsize=factor_font_size,
                color="#4a4a4a",
                fontweight="bold",
            )

        if y_scale == "log":
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax, max(pair_maxima) * 1.45))
        else:
            ax.set_ylim(0, max(pair_maxima) * 1.18 if pair_maxima else 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seq_len) for seq_len in seq_order], rotation=0)
        ax.set_xlabel("Sequence Length", fontsize=axis_label_size)
        ax.set_ylabel(
            "Latency (ms, log scale)" if y_scale == "log" else "Latency (ms)",
            fontsize=axis_label_size,
        )
        ax.set_title(
            plot_family_label(family.family_id),
            loc="left",
            fontsize=family_title_size,
            pad=6,
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.35)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if family.col_index != 0:
            ax.set_ylabel("")

    legend_handles = [
        Patch(facecolor=BLOCK_COLORS[label], edgecolor="none", label=label)
        for label in BLOCK_ORDER
    ]
    legend_handles.append(
        Patch(
            facecolor=PATTERN_COLOR,
            edgecolor=PATTERN_EDGE,
            label="Coarse Runlist End-to-End",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.915, 0.5),
        fontsize=legend_font_size,
    )
    fig.suptitle(
        "Aggregate Latency of Coarse Kernels Compared to Coarse Runlist End-to-End Latency"
        + (
            f" (>{int(min_seq_len_exclusive)} Tokens)"
            if min_seq_len_exclusive is not None
            else ""
        ),
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.03, 0.89, 0.95])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stacked comparison of selected coarse-kernel latencies versus full-path latency."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results_csv(),
        help="Path to end-to-end results CSV",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=default_tuning_csv(),
        help="Path to end-to-end tuning CSV",
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
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scaling mode",
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
    pattern_df, block_df = load_plot_rows(args.results, args.tuning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_plot(
        pattern_df,
        block_df,
        variant=args.variant,
        y_scale=args.y_scale,
        min_seq_len_exclusive=args.min_seq_len_exclusive,
    )
    stem = args.stem or variant_stem(
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
