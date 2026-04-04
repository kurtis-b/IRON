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

FAMILY_ORDER = ["baseline_768", "baseline_1024"]
MODE_ORDER = ["dataflow", "runlist", "offload"]
SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MODE_LABELS = {
    "dataflow": "Dataflow",
    "runlist": "Runlist",
    "offload": "Offload",
}
MODE_COLORS = {
    "dataflow": "#1f6f8b",
    "runlist": "#e07a5f",
    "offload": "#81b29a",
}
MODE_MARKERS = {
    "dataflow": "o",
    "runlist": "s",
    "offload": "D",
}


def default_results_csv() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "results_all_power.csv"
    )


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "end_to_end"


def variant_stem(variant: str) -> str:
    return {
        "standard": "tokens_per_second_by_pattern",
        "slides": "tokens_per_second_by_pattern_slides",
    }[variant]


def family_label(results_rows: pd.DataFrame) -> str:
    sample = results_rows.iloc[0]
    return (
        f"Head Dim = {int(sample['attention_head_size'])} / "
        f"Num Heads = {int(sample['num_attention_heads'])} / "
        f"FFN Dim = {int(sample['intermediate_size'])}"
    )


def load_plot_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = df[df["run_status"] == "passed"].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["tokens_per_sec"] = df["tokens_per_sec"].astype(float)
    df = df[df["execution_mode"].isin(MODE_ORDER)].copy()
    df["execution_mode"] = pd.Categorical(
        df["execution_mode"], MODE_ORDER, ordered=True
    )
    df["study_case_id"] = pd.Categorical(
        df["study_case_id"], FAMILY_ORDER, ordered=True
    )
    df = df.sort_values(["study_case_id", "execution_mode", "seq_len"]).reset_index(
        drop=True
    )

    expected_keys = {
        (family_id, mode, seq_len)
        for family_id in FAMILY_ORDER
        for mode in MODE_ORDER
        for seq_len in SEQ_ORDER
    }
    seen_keys = {
        (str(row.study_case_id), str(row.execution_mode), int(row.seq_len))
        for row in df.itertuples(index=False)
    }
    missing = expected_keys - seen_keys
    if missing:
        missing_summary = ", ".join(
            f"{family_id}:{mode}:{seq_len}"
            for family_id, mode, seq_len in sorted(missing)
        )
        raise ValueError(f"Missing end-to-end TPS rows for: {missing_summary}")

    return df


def render_plot(df: pd.DataFrame, *, variant: str = "standard") -> plt.Figure:
    if variant == "slides":
        context = "poster"
        figsize = (22, 8.5)
        family_title_size = 20
        axis_label_size = 18
        tick_label_size = 14
        legend_font_size = 16
        suptitle_size = 30
        title_y = 0.99
    else:
        context = "talk"
        figsize = (20, 8)
        family_title_size = 18
        axis_label_size = 15
        tick_label_size = 12
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.98

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

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, family_id in zip(axes, FAMILY_ORDER, strict=True):
        family_df = df[df["study_case_id"] == family_id].copy()

        for mode in MODE_ORDER:
            mode_df = family_df[family_df["execution_mode"] == mode].copy()
            ax.plot(
                mode_df["seq_len"],
                mode_df["tokens_per_sec"],
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linewidth=2.6,
                markersize=8 if variant == "standard" else 9,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQ_ORDER)
        ax.set_xticklabels([str(seq_len) for seq_len in SEQ_ORDER], rotation=0)
        ax.set_xlabel("Context Length (tokens)", fontsize=axis_label_size)
        ax.set_ylabel("Throughput (tokens/s)", fontsize=axis_label_size)
        ax.set_title(
            family_label(family_df),
            loc="left",
            fontsize=family_title_size,
            pad=12,
        )
        ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_label_size)

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
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=legend_font_size,
    )
    fig.suptitle(
        "Throughput Comparison of Dataflow, Runlist, and Offload Patterns",
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a TPS comparison across execution patterns."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_plot_rows(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_plot(df, variant=args.variant)
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
