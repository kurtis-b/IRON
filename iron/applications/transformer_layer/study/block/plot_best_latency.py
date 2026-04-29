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
from matplotlib.patches import Patch

from ..plot_families import (
    PLOT_FAMILY_GRID_SHAPE,
    PLOT_ROW_LABELS,
    ordered_plot_families,
    plot_family_label,
)

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
BLOCK_ORDER_BY_VARIANT = {
    "encoder_bert": ["qkv_proj", "mha_out_proj", "addnorm", "ffn"],
    "decoder_gpt2": [
        "qkv_proj",
        "mha_out_proj_causal",
        "layer_norm",
        "elementwise_add",
        "addnorm",
        "ffn",
    ],
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
    family_ids = {family.family_id for family in ordered_plot_families()}
    block_order = {
        block_kind
        for block_kinds in BLOCK_ORDER_BY_VARIANT.values()
        for block_kind in block_kinds
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
        figsize = (20, 11.5)
        row_title_size = 20
        family_title_size = 18
        y_label_size = 18
        x_label_size = 18
        tick_label_size = 14
        legend_font_size = 15
        suptitle_size = 30
        title_y = 0.985
    else:
        context = "talk"
        figsize = (18, 10)
        row_title_size = 17
        family_title_size = 16
        y_label_size = 15
        x_label_size = 15
        tick_label_size = 11
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

    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=figsize,
        sharey=True,
    )
    axes_grid = axes_obj

    present_family_ids = set(best["family_id"].astype(str).tolist())
    hue_order = ["QKV Proj", "MHAO", "Add + Norm", "FFN"]
    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        if family.family_id not in present_family_ids:
            ax.set_axis_off()
            continue

        family_df = best[
            (best["family_id"] == family.family_id)
            & (best["block_kind"].isin(BLOCK_ORDER_BY_VARIANT[family.workload_variant]))
        ].copy()
        if family_df.empty:
            ax.set_axis_off()
            continue

        family_df = (
            family_df.groupby(["family_id", "seq_len", "block_label"], as_index=False)[
                "avg_latency_ms"
            ]
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
            "Latency (ms, log scale)" if y_scale == "log" else "Latency (ms)",
            fontsize=y_label_size,
        )
        ax.set_xlabel("Sequence Length", fontsize=x_label_size)
        ax.set_title(
            plot_family_label(family.family_id),
            loc="left",
            fontsize=family_title_size,
            pad=8,
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.4)
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if family.col_index != 0:
            ax.set_ylabel("")

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
    fig.tight_layout(rect=[0, 0.03, 0.89, 0.95])

    for row_index, row_label in enumerate(PLOT_ROW_LABELS):
        row_axes = [
            axes_grid[row_index][col_index]
            for col_index in range(PLOT_FAMILY_GRID_SHAPE[1])
            if axes_grid[row_index][col_index].axison
        ]
        if not row_axes:
            continue
        top_y = max(ax.get_position().y1 for ax in row_axes) + 0.012
        center_x = (row_axes[0].get_position().x0 + row_axes[-1].get_position().x1) / 2
        fig.text(
            center_x,
            top_y,
            row_label,
            fontsize=row_title_size,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    return fig


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[5]
    default_results = (
        repo_root
        / "iron"
        / "applications"
        / "transformer_layer"
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
