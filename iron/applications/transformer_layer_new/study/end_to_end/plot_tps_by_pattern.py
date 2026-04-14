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

FAMILY_ORDER = [
    "tinybert_512",
    "baseline_768",
    "baseline_1024",
    "gpt2_small_768",
    "gpt2_medium_1024",
]
FAMILY_LABELS = {
    "tinybert_512": "TinyBERT",
    "baseline_768": "BERT-Base",
    "baseline_1024": "BERT-Large",
    "gpt2_small_768": "GPT-2 Small",
    "gpt2_medium_1024": "GPT-2 Medium",
}
MODE_ORDER = ["hybrid", "runlist"]
SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MODE_LABELS = {
    "hybrid": "Hybrid",
    "runlist": "Runlist",
}
MODE_COLORS = {
    "hybrid": "#1f6f8b",
    "runlist": "#e07a5f",
}
MODE_MARKERS = {
    "hybrid": "o",
    "runlist": "s",
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


def variant_stem(metric: str, variant: str, y_scale: str = "log") -> str:
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
    stem = stems[metric][variant]
    if metric == "latency" and y_scale == "linear":
        return f"{stem}_linear"
    return stem


def family_label(results_rows: pd.DataFrame) -> str:
    sample = results_rows.iloc[0]
    return FAMILY_LABELS.get(str(sample["study_case_id"]), str(sample["study_case_id"]))


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
        df["study_case_id"], FAMILY_ORDER, ordered=True
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
) -> plt.Figure:
    metric_column = {
        "throughput": "effective_gflops_per_sec",
        "latency": "avg_latency_ms",
    }[metric]
    ylabel = {
        "throughput": "Effective Throughput (GFLOP/s)",
        "latency": "Latency (ms, log scale)" if y_scale == "log" else "Latency (ms)",
    }[metric]
    title = {
        "throughput": "Effective Throughput Comparison of Hybrid and Runlist Patterns",
        "latency": "Latency Comparison of Hybrid and Runlist Patterns",
    }[metric]
    if variant == "slides":
        context = "poster"
        figsize = (34, 8.5)
        family_title_size = 20
        axis_label_size = 18
        tick_label_size = 14
        legend_font_size = 16
        suptitle_size = 30
        title_y = 0.965
    else:
        context = "talk"
        figsize = (30, 8)
        family_title_size = 18
        axis_label_size = 15
        tick_label_size = 12
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.955

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
        if family_id in set(df["study_case_id"].astype(str).tolist())
    ]
    fig, axes_obj = plt.subplots(
        1, max(1, len(family_ids)), figsize=figsize, sharey=True
    )
    axes = [axes_obj] if len(family_ids) == 1 else list(axes_obj)
    legend_ax = None

    for ax, family_id in zip(axes, family_ids, strict=True):
        family_df = df[df["study_case_id"] == family_id].copy()

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
        ax.set_xticks(SEQ_ORDER)
        ax.set_xticklabels([str(seq_len) for seq_len in SEQ_ORDER], rotation=0)
        if metric == "latency" and y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Sequence Length", fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)
        ax.set_title(
            family_label(family_df),
            loc="left",
            fontsize=family_title_size,
            pad=6,
        )
        ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if ax is not axes[0]:
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
            loc="center left",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.91, 0.5),
            fontsize=legend_font_size,
        )
    fig.suptitle(
        title,
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.03, 0.89, 0.92])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an effective GFLOP/s comparison across execution patterns."
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
    )
    stem = args.stem or variant_stem(args.metric, args.variant, args.y_scale)
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
