"""
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

BLOCK_ORDER = ["qkv_proj", "mha_out_proj", "addnorm", "ffn"]
BLOCK_LABELS = {
    "qkv_proj": "QKV Proj",
    "mha_out_proj": "MHAO",
    "addnorm": "Add + Norm",
    "ffn": "FFN",
}
FAMILY_ORDER = ["tinybert_512", "baseline_768", "baseline_1024"]
FAMILY_LABELS = {
    "tinybert_512": "TinyBERT",
    "baseline_768": "BERT-Base",
    "baseline_1024": "BERT-Large",
}
PALETTE = {
    "QKV Proj": "#1f6f8b",
    "MHAO": "#e07a5f",
    "Add + Norm": "#3d405b",
    "FFN": "#81b29a",
}


def variant_stem(variant: str, y_scale: str = "log") -> str:
    stem = {
        "standard": "best_latency_by_block",
        "slides": "best_latency_by_block_slides",
    }[variant]
    if y_scale == "linear":
        return f"{stem}_linear"
    return stem


def load_best_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = df[df["run_status"] == "passed"].copy()
    df = df[df["block_kind"].isin(BLOCK_ORDER)].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["avg_latency_ms"] = df["avg_latency_ms"].astype(float)

    best = (
        df.groupby(
            ["family_id", "family_label", "seq_len", "block_kind"], as_index=False
        )["avg_latency_ms"]
        .min()
        .copy()
    )
    best["block_label"] = best["block_kind"].map(BLOCK_LABELS)
    best["family_id"] = pd.Categorical(best["family_id"], FAMILY_ORDER, ordered=True)
    best["block_kind"] = pd.Categorical(best["block_kind"], BLOCK_ORDER, ordered=True)
    best = best.sort_values(["family_id", "seq_len", "block_kind"]).reset_index(
        drop=True
    )
    return best


def plot_best_latency(
    best: pd.DataFrame,
    title: str,
    variant: str = "standard",
    y_scale: str = "log",
):
    if variant == "slides":
        context = "poster"
        figsize = (22, 8.5)
        nrows, ncols = 1, max(1, len(FAMILY_ORDER))
        legend_bbox = (0.88, 0.5)
        title_y = 0.99
        family_title_size = 18
        y_label_size = 18
        x_label_size = 18
        tick_label_size = 14
        legend_font_size = 14
    else:
        context = "talk"
        figsize = (24, 8)
        nrows, ncols = 1, max(1, len(FAMILY_ORDER))
        legend_bbox = (0.88, 0.5)
        title_y = 1.06
        family_title_size = 16
        y_label_size = 16
        x_label_size = 16
        tick_label_size = 12
        legend_font_size = 13

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

    fig, axes_obj = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=variant == "standard",
        sharey=True,
    )
    axes = [axes_obj] if not hasattr(axes_obj, "flatten") else list(axes_obj.flatten())

    for ax, family_id in zip(axes, FAMILY_ORDER, strict=True):
        family_df = best[best["family_id"] == family_id].copy()
        family_df["seq_label"] = family_df["seq_len"].astype(str)
        sns.barplot(
            data=family_df,
            x="seq_label",
            y="avg_latency_ms",
            hue="block_label",
            hue_order=[BLOCK_LABELS[k] for k in BLOCK_ORDER],
            palette=PALETTE,
            errorbar=None,
            ax=ax,
        )
        if y_scale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(
            "Latency (ms, log scale)" if y_scale == "log" else "Latency (ms)",
            fontsize=y_label_size,
        )
        ax.set_xlabel(
            "" if variant == "standard" else "Sequence Length",
            fontsize=x_label_size,
        )
        ax.set_title(
            FAMILY_LABELS[family_id], loc="left", fontsize=family_title_size, pad=10
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.4)
        ax.legend_.remove()
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if ax is not axes[0]:
            ax.set_ylabel("")

    if variant == "standard":
        axes[-1].set_xlabel("Sequence Length", fontsize=x_label_size)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=1,
        loc="center left",
        bbox_to_anchor=legend_bbox,
        frameon=False,
        fontsize=legend_font_size,
    )
    fig.suptitle(
        title,
        fontsize=24 if variant == "standard" else 28,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.03, 0.84, 0.94])
    return fig


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[5]
    default_results = (
        repo_root
        / "iron"
        / "applications"
        / "transformer_layer_new"
        / "results"
        / "block"
        / "results.csv"
    )
    default_output_dir = default_results.parent
    parser = argparse.ArgumentParser(
        description="Render a block-study chart of best latency by sequence length."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results,
        help="Path to block-study results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for the rendered chart files",
    )
    parser.add_argument(
        "--variant",
        choices=["standard", "slides"],
        default="standard",
        help="Chart layout preset",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem, without extension",
    )
    parser.add_argument(
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scaling mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = load_best_rows(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    title = "Latency Comparison of Blocks for Transformer Layer"
    fig = plot_best_latency(best, title, variant=args.variant, y_scale=args.y_scale)
    stem = args.stem or variant_stem(args.variant, args.y_scale)
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
