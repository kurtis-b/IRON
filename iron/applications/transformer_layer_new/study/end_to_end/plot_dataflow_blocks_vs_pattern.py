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

from .cases import FAMILY_IDS

FAMILY_ORDER = list(FAMILY_IDS)
SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BLOCK_ORDER = ["QKV Proj", "MHA Out Proj", "Add + Norm (x2)", "FFN"]
BLOCK_COLORS = {
    "QKV Proj": "#1f6f8b",
    "MHA Out Proj": "#e07a5f",
    "Add + Norm (x2)": "#3d405b",
    "FFN": "#81b29a",
}
PATTERN_COLOR = "#d8d4cf"
PATTERN_EDGE = "#2b2b2b"
BLOCK_KIND_MAP = {
    "qkv_proj": "QKV Proj",
    "mha_out_proj": "MHA Out Proj",
    "add_norm1": "Add + Norm (x2)",
    "add_norm2": "Add + Norm (x2)",
    "ffn": "FFN",
}


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


def variant_stem(variant: str) -> str:
    return {
        "standard": "dataflow_selected_blocks_vs_pattern_latency",
        "slides": "dataflow_selected_blocks_vs_pattern_latency_slides",
    }[variant]


def family_label(pattern_rows: pd.DataFrame) -> str:
    sample = pattern_rows.iloc[0]
    return (
        f"Head Dim = {int(sample['attention_head_size'])} / "
        f"Num Heads = {int(sample['num_attention_heads'])} / "
        f"FFN Dim = {int(sample['intermediate_size'])}"
    )


def load_plot_rows(
    results_csv: Path, tuning_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.read_csv(results_csv)
    tuning_df = pd.read_csv(tuning_csv)

    pattern_df = results_df[
        (results_df["execution_mode"] == "dataflow")
        & (results_df["run_status"] == "passed")
    ].copy()
    pattern_df["seq_len"] = pattern_df["seq_len"].astype(int)
    pattern_df["avg_latency_ms"] = pattern_df["avg_latency_ms"].astype(float)
    pattern_df = pattern_df.sort_values(["study_case_id", "seq_len"]).reset_index(
        drop=True
    )

    chosen_blocks = tuning_df[
        (tuning_df["execution_mode"] == "dataflow")
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
) -> plt.Figure:
    if variant == "slides":
        context = "poster"
        figsize = (22, 8.5)
        family_title_size = 20
        axis_label_size = 18
        tick_label_size = 14
        legend_font_size = 16
        suptitle_size = 30
        title_y = 0.99
        factor_font_size = 11
    else:
        context = "talk"
        figsize = (20, 8)
        family_title_size = 18
        axis_label_size = 15
        tick_label_size = 12
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.98
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

    family_ids = [
        family_id
        for family_id in FAMILY_ORDER
        if family_id in set(pattern_df["study_case_id"].astype(str).tolist())
    ]
    if variant == "slides" and len(family_ids) == 3:
        fig, axes_grid = plt.subplots(2, 2, figsize=(22, 12), sharey=True)
        axes = [axes_grid[0][0], axes_grid[0][1], axes_grid[1][0]]
        legend_ax = axes_grid[1][1]
        legend_ax.set_axis_off()
    else:
        fig, axes_obj = plt.subplots(
            1, max(1, len(family_ids)), figsize=figsize, sharey=True
        )
        axes = [axes_obj] if len(family_ids) == 1 else list(axes_obj)
        legend_ax = None
    bar_width = 0.34
    x_positions = list(range(len(SEQ_ORDER)))

    for ax, family_id in zip(axes, family_ids, strict=True):
        family_patterns = pattern_df[pattern_df["study_case_id"] == family_id].copy()
        family_blocks = block_df[block_df["study_case_id"] == family_id].copy()

        pattern_map = {
            int(row.seq_len): float(row.avg_latency_ms)
            for row in family_patterns.itertuples(index=False)
        }
        block_map = {
            (int(row.seq_len), str(row.block_label)): float(row.avg_latency_ms)
            for row in family_blocks.itertuples(index=False)
        }

        bottoms = [0.0] * len(SEQ_ORDER)
        for block_label in BLOCK_ORDER:
            heights = [
                block_map.get((seq_len, block_label), 0.0) for seq_len in SEQ_ORDER
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

        pattern_heights = [pattern_map.get(seq_len, 0.0) for seq_len in SEQ_ORDER]
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

        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, max(ymax, max(pair_maxima) * 1.45))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seq_len) for seq_len in SEQ_ORDER], rotation=0)
        ax.set_xlabel("Context Length (tokens)", fontsize=axis_label_size)
        ax.set_ylabel("Latency (ms, log scale)", fontsize=axis_label_size)
        ax.set_title(
            family_label(family_patterns),
            loc="left",
            fontsize=family_title_size,
            pad=12,
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.35)
        ax.tick_params(axis="both", labelsize=tick_label_size)

    legend_handles = [
        Patch(facecolor=BLOCK_COLORS[label], edgecolor="none", label=label)
        for label in BLOCK_ORDER
    ]
    legend_handles.append(
        Patch(facecolor=PATTERN_COLOR, edgecolor=PATTERN_EDGE, label="Dataflow Pattern")
    )
    if legend_ax is not None:
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            fontsize=legend_font_size,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=legend_font_size,
        )
    fig.suptitle(
        "Aggregate Latency of Blocks Compared to Dataflow Pattern Latency",
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stacked comparison of selected dataflow block latencies versus full-pattern latency."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern_df, block_df = load_plot_rows(args.results, args.tuning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_plot(pattern_df, block_df, variant=args.variant)
    stem = args.stem or variant_stem(args.variant)
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
