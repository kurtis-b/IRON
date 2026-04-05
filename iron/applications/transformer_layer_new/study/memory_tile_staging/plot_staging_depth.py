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

from iron.applications.transformer_layer_new.study.block.cases import (
    FAMILY_IDS,
    SEQUENCE_LADDER,
)

from .select import STAGING_BLOCK_KINDS

BLOCK_LABELS = {
    "mha_out_proj": "Dataflow Block MHA + Output Projection",
    "ffn": "Dataflow Block FFN",
}
FAMILY_LABELS_BY_BLOCK = {
    "mha_out_proj": {
        "baseline_768": "Head Dim = 64 / Num Heads = 12",
        "baseline_1024": "Head Dim = 64 / Num Heads = 16",
    },
    "ffn": {
        "baseline_768": "Head Dim = 64 / Num Heads = 12 / FFN Dim = 3072",
        "baseline_1024": "Head Dim = 64 / Num Heads = 16 / FFN Dim = 4096",
    },
}
SEQ_COLORS = {
    seq_len: color
    for seq_len, color in zip(
        SEQUENCE_LADDER,
        sns.color_palette("crest", n_colors=len(SEQUENCE_LADDER)),
        strict=True,
    )
}
PLOT_SEQUENCE_LENGTHS = tuple(
    seq_len for seq_len in SEQUENCE_LADDER if 256 <= seq_len <= 8192
)


def default_results_csv() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "memory_tile_staging"
        / "results.csv"
    )


def default_output_dir() -> Path:
    return default_results_csv().parent


def variant_stem(block_kind: str, metric: str) -> str:
    metric_stem = {
        "latency": "latency_by_staging_depth",
        "speedup": "speedup_by_staging_depth",
    }[metric]
    return f"{block_kind}_{metric_stem}"


def load_plot_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if df.empty:
        return df
    df["seq_len"] = df["seq_len"].astype(int)
    df["staging_depth"] = df["staging_depth"].astype(int)
    df["avg_latency_ms"] = pd.to_numeric(df["avg_latency_ms"], errors="coerce")
    df["speedup_vs_depth1"] = pd.to_numeric(df["speedup_vs_depth1"], errors="coerce")
    return df


def render_plot(
    rows: pd.DataFrame,
    *,
    block_kind: str,
    metric: str,
) -> plt.Figure:
    metric_column = {
        "latency": "avg_latency_ms",
        "speedup": "speedup_vs_depth1",
    }[metric]
    metric_label = {
        "latency": "Latency (ms, log scale)",
        "speedup": "Speedup vs Depth 1",
    }[metric]
    title = {
        "latency": f"{BLOCK_LABELS[block_kind]} Latency by Memory-Tile Staging Depth",
        "speedup": f"{BLOCK_LABELS[block_kind]} Speedup by Memory-Tile Staging Depth",
    }[metric]

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )
    plt.rcParams["svg.fonttype"] = "none"

    block_rows = rows[rows["block_kind"] == block_kind].copy()
    block_rows = block_rows[block_rows["seq_len"].isin(PLOT_SEQUENCE_LENGTHS)].copy()
    if metric == "latency":
        block_rows = block_rows[
            (block_rows["run_status"] == "passed")
            & block_rows["avg_latency_ms"].notna()
        ].copy()
    else:
        block_rows = block_rows[
            (block_rows["run_status"] == "passed")
            & block_rows["speedup_vs_depth1"].notna()
        ].copy()

    if block_rows.empty:
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_axis_off()
        fig.text(
            0.5,
            0.57,
            title,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.42,
            "No data available",
            ha="center",
            va="center",
            fontsize=18,
        )
        fig.patch.set_facecolor("#f7f5f2")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    all_depths = sorted(block_rows["staging_depth"].unique())
    for ax, family_id in zip(axes, FAMILY_IDS, strict=True):
        family_rows = block_rows[block_rows["family_id"] == family_id].copy()
        for seq_len in PLOT_SEQUENCE_LENGTHS:
            seq_rows = family_rows[family_rows["seq_len"] == seq_len].copy()
            if seq_rows.empty:
                continue
            seq_rows = seq_rows.sort_values("staging_depth")
            ax.plot(
                seq_rows["staging_depth"],
                seq_rows[metric_column],
                label=str(seq_len),
                color=SEQ_COLORS[seq_len],
                marker="o",
                linewidth=2.4,
                markersize=7,
            )

        if metric == "latency":
            ax.set_yscale("log")
        ax.set_xticks(all_depths)
        ax.set_xlabel("Staging Depth", fontsize=15)
        ax.set_ylabel(metric_label, fontsize=15)
        ax.set_title(
            FAMILY_LABELS_BY_BLOCK[block_kind][family_id],
            loc="left",
            fontsize=18,
            pad=12,
        )
        ax.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SEQ_COLORS[seq_len],
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=str(seq_len),
        )
        for seq_len in PLOT_SEQUENCE_LENGTHS
        if seq_len in block_rows["seq_len"].unique()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(5, len(legend_handles) or 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        title="Context Length (tokens)",
        fontsize=12,
        title_fontsize=13,
    )
    fig.suptitle(
        title,
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def write_plot(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    block_kind: str,
    metric: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_plot(rows, block_kind=block_kind, metric=metric)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_canonical_plots(results_csv: Path, output_dir: Path) -> None:
    rows = load_plot_rows(results_csv)
    for block_kind in STAGING_BLOCK_KINDS:
        for metric in ("latency", "speedup"):
            stem = variant_stem(block_kind, metric)
            write_plot(
                rows,
                output_path=output_dir / f"{stem}.svg",
                block_kind=block_kind,
                metric=metric,
            )
            write_plot(
                rows,
                output_path=output_dir / f"{stem}.png",
                block_kind=block_kind,
                metric=metric,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render memory-tile staging depth study plots."
    )
    parser.add_argument("--results", type=Path, default=default_results_csv())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--block",
        choices=STAGING_BLOCK_KINDS,
        default=None,
    )
    parser.add_argument(
        "--metric",
        choices=("latency", "speedup"),
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_plot_rows(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.block is None or args.metric is None:
        write_canonical_plots(args.results, args.output_dir)
        return

    stem = variant_stem(args.block, args.metric)
    write_plot(
        rows,
        output_path=args.output_dir / f"{stem}.png",
        block_kind=args.block,
        metric=args.metric,
    )
    write_plot(
        rows,
        output_path=args.output_dir / f"{stem}.svg",
        block_kind=args.block,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
