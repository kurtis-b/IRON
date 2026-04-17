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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BLOCK_LABELS = {
    "qkv_proj": "QKV Proj",
    "mha_out_proj": "MHAO",
    "mha_out_proj_causal": "MHAO",
    "addnorm": "Add + Norm",
    "layer_norm": "Add + Norm",
    "elementwise_add": "Add + Norm",
    "ffn": "FFN",
}
LEGEND_ORDER = [
    "QKV Proj",
    "MHAO",
    "Add + Norm",
    "FFN",
]
SECTION_CONFIGS = [
    {
        "row_title": "Encoder Blocks",
        "family_ids": ["tinybert_512", "baseline_768", "baseline_1024"],
        "family_labels": {
            "tinybert_512": "TinyBERT",
            "baseline_768": "BERT-Base",
            "baseline_1024": "BERT-Large",
        },
        "block_order": ["qkv_proj", "mha_out_proj", "addnorm", "ffn"],
    },
    {
        "row_title": "Decoder Blocks",
        "family_ids": ["baseline_768", "baseline_1024"],
        "family_labels": {
            "baseline_768": "GPT-2 Small",
            "baseline_1024": "GPT-2 Medium",
        },
        "block_order": [
            "qkv_proj",
            "mha_out_proj_causal",
            "layer_norm",
            "elementwise_add",
            "addnorm",
            "ffn",
        ],
    },
]
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
    family_ids = {
        family_id for config in SECTION_CONFIGS for family_id in config["family_ids"]
    }
    block_order = {
        block_kind for config in SECTION_CONFIGS for block_kind in config["block_order"]
    }
    df = pd.read_csv(results_csv)
    df = df[df["run_status"] == "passed"].copy()
    df = df[df["family_id"].isin(family_ids)].copy()
    df = df[df["block_kind"].isin(block_order)].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["avg_latency_ms"] = df["avg_latency_ms"].astype(float)

    best = (
        df.groupby(["family_id", "seq_len", "block_kind"], as_index=False)[
            "avg_latency_ms"
        ]
        .min()
        .copy()
    )
    best["block_label"] = best["block_kind"].map(BLOCK_LABELS)
    best = best.dropna(subset=["block_label"]).reset_index(drop=True)
    return best


def plot_best_latency(
    best: pd.DataFrame,
    title: str,
    *,
    variant: str = "standard",
    y_scale: str = "log",
):
    if variant == "slides":
        context = "poster"
        figsize = (34, 8.5)
        row_title_size = 20
        family_title_size = 18
        y_label_size = 18
        x_label_size = 18
        tick_label_size = 14
        legend_font_size = 15
        suptitle_size = 30
        title_y = 0.97
    else:
        context = "talk"
        figsize = (30, 8)
        row_title_size = 17
        family_title_size = 16
        y_label_size = 15
        x_label_size = 15
        tick_label_size = 11
        legend_font_size = 13
        suptitle_size = 24
        title_y = 0.96

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

    fig, axes_obj = plt.subplots(1, 5, figsize=figsize, sharey=True)
    axes = list(axes_obj)

    axis_index = 0
    hue_order = ["QKV Proj", "MHAO", "Add + Norm", "FFN"]
    for config in SECTION_CONFIGS:
        for family_id in config["family_ids"]:
            ax = axes[axis_index]
            axis_index += 1
            family_df = best[
                (best["family_id"] == family_id)
                & (best["block_kind"].isin(config["block_order"]))
            ].copy()
            family_df = (
                family_df.groupby(
                    ["family_id", "seq_len", "block_label"], as_index=False
                )["avg_latency_ms"]
                .sum()
                .sort_values(["seq_len", "block_label"])
                .reset_index(drop=True)
            )
            family_df["seq_label"] = pd.Categorical(
                family_df["seq_len"].astype(str),
                [
                    str(seq_len)
                    for seq_len in SEQ_ORDER
                    if seq_len in set(family_df["seq_len"].tolist())
                ],
                ordered=True,
            )

            sns.barplot(
                data=family_df,
                x="seq_label",
                y="avg_latency_ms",
                hue="block_label",
                hue_order=hue_order,
                palette=PALETTE,
                errorbar=None,
                ax=ax,
            )
            if y_scale == "log":
                ax.set_yscale("log")
            ax.set_ylabel(
                (
                    "Latency (ms, log scale)"
                    if axis_index == 1 and y_scale == "log"
                    else "Latency (ms)" if axis_index == 1 else ""
                ),
                fontsize=y_label_size,
            )
            ax.set_xlabel("Sequence Length", fontsize=x_label_size)
            ax.set_title(
                config["family_labels"][family_id],
                loc="left",
                fontsize=family_title_size,
                pad=8,
            )
            ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
            ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.4)
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.tick_params(axis="both", labelsize=tick_label_size)

    legend_labels = [
        label for label in LEGEND_ORDER if label in set(best["block_label"].tolist())
    ]
    legend_handles = [
        Patch(facecolor=PALETTE[label], edgecolor="none", label=label)
        for label in legend_labels
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.905, 0.5),
        frameon=False,
        fontsize=legend_font_size,
    )
    fig.suptitle(
        title,
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0.03, 0.89, 0.91])

    encoder_axes = axes[: len(SECTION_CONFIGS[0]["family_ids"])]
    decoder_axes = axes[len(SECTION_CONFIGS[0]["family_ids"]) :]
    top_y = max(ax.get_position().y1 for ax in axes) + 0.018

    def section_center(section_axes):
        return (
            section_axes[0].get_position().x0 + section_axes[-1].get_position().x1
        ) / 2

    fig.text(
        section_center(encoder_axes),
        top_y,
        SECTION_CONFIGS[0]["row_title"],
        fontsize=row_title_size,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    fig.text(
        section_center(decoder_axes),
        top_y,
        SECTION_CONFIGS[1]["row_title"],
        fontsize=row_title_size,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    separator_x = (
        encoder_axes[-1].get_position().x1 + decoder_axes[0].get_position().x0
    ) / 2
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    fig.add_artist(
        Line2D(
            [separator_x, separator_x],
            [y0, y1],
            transform=fig.transFigure,
            color="#c7c1b7",
            linewidth=1.5,
        )
    )
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
    fig = plot_best_latency(
        best,
        "Latency Comparison of Encoder and Decoder Blocks",
        variant=args.variant,
        y_scale=args.y_scale,
    )
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
