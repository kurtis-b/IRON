#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def default_results_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "reconfiguration_overhead"
        / "results.csv"
    )


def default_output_svg_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "reconfiguration_overhead"
        / "runlist_vs_offload_reconfiguration_overhead.svg"
    )


def default_output_png_path() -> Path:
    return default_output_svg_path().with_suffix(".png")


FAMILY_LABELS = {
    "baseline_768": "Head Dim = 64 / Num Heads = 12 / FFN Dim = 3072",
    "baseline_1024": "Head Dim = 64 / Num Heads = 16 / FFN Dim = 4096",
}

MODE_LABELS = {
    "offload_gemm_sequence": "Offload",
    "runlist_gemm_sequence": "Runlist",
}

MODE_COLORS = {
    "offload_gemm_sequence": "#81b29a",
    "runlist_gemm_sequence": "#e07a5f",
}


def load_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_plot_rows(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, dict[int, dict[str, dict[str, object]]]]]:
    plot_rows = [
        row
        for row in rows
        if row.get("run_status") == "passed"
        and row.get("execution_mode") in MODE_LABELS
        and row.get("avg_latency_ms")
    ]
    family_order = [
        family
        for family in FAMILY_LABELS
        if any(row["study_case_id"] == family for row in plot_rows)
    ]
    data: dict[str, dict[int, dict[str, dict[str, object]]]] = {}
    for row in plot_rows:
        family = row["study_case_id"]
        seq_len = int(row["seq_len"])
        mode = row["execution_mode"]
        data.setdefault(family, {}).setdefault(seq_len, {})[mode] = {
            "latency_ms": float(row["avg_latency_ms"]),
            "xclbins": int(row["npu_unique_xclbin_count"]),
        }
    return family_order, data


def _mode_legend_label(
    mode: str, data: dict[str, dict[int, dict[str, dict[str, object]]]]
) -> str:
    for family_rows in data.values():
        for seq_rows in family_rows.values():
            if mode in seq_rows:
                xclbins = seq_rows[mode]["xclbins"]
                suffix = "xclbin" if xclbins == 1 else "xclbins"
                return f"{MODE_LABELS[mode]} ({xclbins} {suffix})"
    return MODE_LABELS[mode]


def render_plot(
    rows: list[dict[str, str]],
    *,
    output_svg_path: Path,
    output_png_path: Path,
) -> None:
    family_order, data = _build_plot_rows(rows)
    if not family_order:
        raise ValueError("No passing reconfiguration-overhead rows were found.")

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

    fig, axes = plt.subplots(
        1,
        len(family_order),
        figsize=(20, 8),
        sharey=True,
    )
    if len(family_order) == 1:
        axes = [axes]

    bar_width = 0.32

    for ax, family in zip(axes, family_order, strict=True):
        seq_lens = sorted(data[family])
        x_positions = list(range(len(seq_lens)))
        offload_values = [
            data[family][seq_len]["offload_gemm_sequence"]["latency_ms"]
            for seq_len in seq_lens
        ]
        runlist_values = [
            data[family][seq_len]["runlist_gemm_sequence"]["latency_ms"]
            for seq_len in seq_lens
        ]

        offload_x = [x - bar_width / 2 for x in x_positions]
        runlist_x = [x + bar_width / 2 for x in x_positions]

        ax.bar(
            offload_x,
            offload_values,
            width=bar_width,
            color=MODE_COLORS["offload_gemm_sequence"],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar(
            runlist_x,
            runlist_values,
            width=bar_width,
            color=MODE_COLORS["runlist_gemm_sequence"],
            edgecolor="white",
            linewidth=0.7,
        )

        for x_mid, offload_latency, runlist_latency in zip(
            x_positions, offload_values, runlist_values, strict=True
        ):
            ratio = runlist_latency / offload_latency
            label_y = max(offload_latency, runlist_latency) * 1.14
            ax.text(
                x_mid,
                label_y,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#4a4a4a",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seq_len) for seq_len in seq_lens], fontsize=12)
        ax.set_xlabel("Context Length (tokens)", fontsize=15)
        ax.set_title(
            FAMILY_LABELS.get(family, family),
            loc="left",
            fontsize=18,
            pad=12,
        )
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="major", linewidth=0.9, alpha=0.85)
        ax.grid(True, axis="y", which="minor", linewidth=0.4, alpha=0.35)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("Latency (ms, log scale)", fontsize=15)
    fig.suptitle(
        "Reconfiguration Overhead: Runlist vs Offload",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    legend_handles = [
        Patch(
            facecolor=MODE_COLORS["offload_gemm_sequence"],
            edgecolor="none",
            label=_mode_legend_label("offload_gemm_sequence", data),
        ),
        Patch(
            facecolor=MODE_COLORS["runlist_gemm_sequence"],
            edgecolor="none",
            label=_mode_legend_label("runlist_gemm_sequence", data),
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg_path, bbox_inches="tight")
    fig.savefig(output_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reconfiguration-overhead latency comparison"
    )
    parser.add_argument("--results", type=Path, default=default_results_path())
    parser.add_argument("--output-svg", type=Path, default=default_output_svg_path())
    parser.add_argument("--output-png", type=Path, default=default_output_png_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.results.expanduser())
    render_plot(
        rows,
        output_svg_path=args.output_svg.expanduser(),
        output_png_path=args.output_png.expanduser(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
