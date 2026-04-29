#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
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
from .run_selected_component_aggregates import (
    FULL_PATTERN_GROUP_LABEL,
    HYBRID_GROUP_ORDER,
    OFFLOAD_GROUP_ORDER,
    RUNLIST_GROUP_ORDER,
    default_aggregate_output_path,
    expected_group_order,
    expected_group_component_count,
)

SEQ_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
GROUP_COLORS = {
    "QKV Proj": "#1f6f8b",
    "MHA + Output": "#e07a5f",
    "LN1": "#4c78a8",
    "QKVO Proj": "#1f6f8b",
    "K Transpose": "#59a14f",
    "Attention Scores": "#e15759",
    "Attention Scale": "#f28e2b",
    "Causal Mask": "#b07aa1",
    "Attention Softmax": "#edc948",
    "Attention Output": "#76b7b2",
    "Residual Add": "#9c755f",
    "LN2": "#bab0ac",
    "Up Proj": "#8cd17d",
    "GELU": "#ff9da7",
    "Down Proj": "#a0cbe8",
    "Residual + Norm": "#3d405b",
    "FFN": "#81b29a",
    "GEMMs (NPU)": "#1f6f8b",
    "Non-linear operations (CPU)": "#e07a5f",
}
PATTERN_COLOR = "#d8d4cf"
PATTERN_EDGE = "#2b2b2b"
LONG_SEQUENCE_SUFFIX_TEMPLATE = "after_{threshold}_tokens"
MODE_TITLES = {
    "hybrid": "Aggregate Latency of Selected Dataflow Blocks Compared to End-to-End Hybrid Latency",
    "runlist": "Aggregate Latency of Selected Runlist Operations Compared to End-to-End Runlist Latency",
    "offload": "Aggregate Latency of Selected Offload Groups Compared to End-to-End Offload Latency",
}
MODE_PATTERN_LABELS = {
    "hybrid": "Hybrid End-to-End",
    "runlist": "Runlist End-to-End",
    "offload": "Offload End-to-End",
}


def default_results_csv() -> Path:
    return default_aggregate_output_path()


def default_output_dir() -> Path:
    return default_results_csv().parent


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


def _group_order(execution_mode: str) -> tuple[str, ...]:
    if execution_mode == "hybrid":
        return HYBRID_GROUP_ORDER
    if execution_mode == "runlist":
        return RUNLIST_GROUP_ORDER
    if execution_mode == "offload":
        return OFFLOAD_GROUP_ORDER
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value) or str(value) == ""


def _truthy(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_count(value: object) -> int | None:
    if _is_missing(value):
        return None
    return int(float(str(value)))


def _missing_components_are_empty(value: object) -> bool:
    if _is_missing(value):
        return True
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return False
    return loaded == []


def _complete_full_pattern_row(row) -> bool:
    metadata_complete = _truthy(getattr(row, "is_complete", None))
    if metadata_complete is False:
        return False
    return str(row.run_status) == "passed" and pd.notna(row.avg_latency_ms)


def _complete_isolated_group_row(row, *, execution_mode: str) -> bool:
    if str(row.run_status) != "passed" or pd.isna(row.avg_latency_ms):
        return False
    metadata_complete = _truthy(getattr(row, "is_complete", None))
    if metadata_complete is False:
        return False
    if not _missing_components_are_empty(getattr(row, "missing_components_json", None)):
        return False
    expected_count = _optional_count(getattr(row, "expected_component_count", None))
    if expected_count is None:
        try:
            expected_count = expected_group_component_count(
                execution_mode,
                str(row.workload_variant),
                str(row.group_label),
            )
        except (KeyError, ValueError):
            expected_count = None
    source_count = _optional_count(getattr(row, "source_component_count", None))
    if expected_count is not None and source_count is not None:
        return source_count == expected_count
    return True


def _set_plot_attrs(
    pattern_df: pd.DataFrame,
    group_df: pd.DataFrame,
    *,
    incomplete_pair_keys: set[tuple[str, str, str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attrs = {
        "incomplete_pair_keys": sorted(incomplete_pair_keys),
        "incomplete_comparison_count": len(incomplete_pair_keys),
        "incomplete_family_ids": sorted({str(key[0]) for key in incomplete_pair_keys}),
    }
    pattern_df.attrs.update(attrs)
    group_df.attrs.update(attrs)
    return pattern_df, group_df


def variant_stem(
    execution_mode: str,
    variant: str,
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> str:
    stem = {
        "hybrid": {
            "standard": "hybrid_selected_groups_vs_pattern_latency",
            "slides": "hybrid_selected_groups_vs_pattern_latency_slides",
        },
        "runlist": {
            "standard": "runlist_selected_groups_vs_pattern_latency",
            "slides": "runlist_selected_groups_vs_pattern_latency_slides",
        },
        "offload": {
            "standard": "offload_selected_groups_vs_pattern_latency",
            "slides": "offload_selected_groups_vs_pattern_latency_slides",
        },
    }[execution_mode][variant]
    stem = f"{stem}{_sequence_suffix(min_seq_len_exclusive)}"
    if y_scale == "linear":
        return f"{stem}_linear"
    return stem


def load_plot_rows(
    results_csv: Path,
    *,
    execution_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_csv)
    df = df[df["execution_mode"] == execution_mode].copy()
    for column in (
        "expected_component_count",
        "missing_components_json",
        "is_complete",
        "measurement_sources_json",
    ):
        if column not in df.columns:
            df[column] = pd.NA
    df["seq_len"] = df["seq_len"].astype(int)
    df["avg_latency_ms"] = pd.to_numeric(df["avg_latency_ms"], errors="coerce")

    df["_pair_key"] = [
        (
            str(row.study_case_id),
            str(row.workload_variant),
            str(row.execution_mode),
            int(row.seq_len),
        )
        for row in df.itertuples(index=False)
    ]
    universe_keys = set(df["_pair_key"].tolist())
    full_pattern_mask = [
        row.row_kind == "full_pattern" and _complete_full_pattern_row(row)
        for row in df.itertuples(index=False)
    ]
    pattern_candidates = df[full_pattern_mask].copy()
    group_row_mask = [
        row.row_kind == "isolated_group"
        and _complete_isolated_group_row(row, execution_mode=execution_mode)
        for row in df.itertuples(index=False)
    ]
    group_candidates = df[group_row_mask].copy()

    complete_keys: set[tuple[str, str, str, int]] = set()
    pattern_key_set = set(pattern_candidates["_pair_key"].tolist())
    for pair_key in universe_keys:
        if pair_key not in pattern_key_set:
            continue
        pair_group_rows = group_candidates[group_candidates["_pair_key"] == pair_key]
        present_groups = {str(row.group_label) for row in pair_group_rows.itertuples()}
        required_groups = expected_group_order(execution_mode, str(pair_key[1]))
        if all(group_label in present_groups for group_label in required_groups):
            complete_keys.add(pair_key)

    incomplete_pair_keys = universe_keys - complete_keys
    pattern_df = pattern_candidates[
        pattern_candidates["_pair_key"].isin(complete_keys)
    ].copy()
    group_df = group_candidates[
        group_candidates["_pair_key"].isin(complete_keys)
    ].copy()
    pattern_df = pattern_df.drop(columns=["_pair_key"])
    group_df = group_df.drop(columns=["_pair_key"])
    return _set_plot_attrs(
        pattern_df,
        group_df,
        incomplete_pair_keys=incomplete_pair_keys,
    )


def _render_empty_figure(
    title: str,
    *,
    message: str = "No data available",
    omitted_count: int = 0,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_axis_off()
    fig.patch.set_facecolor("#f7f5f2")
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
        message,
        ha="center",
        va="center",
        fontsize=18,
    )
    if omitted_count:
        fig.text(
            0.5,
            0.33,
            f"Incomplete comparisons omitted: {omitted_count}",
            ha="center",
            va="center",
            fontsize=14,
            color="#5a5249",
        )
    return fig


def render_plot(
    pattern_df: pd.DataFrame,
    group_df: pd.DataFrame,
    *,
    execution_mode: str,
    variant: str = "standard",
    y_scale: str = "log",
    min_seq_len_exclusive: int | None = None,
) -> plt.Figure:
    incomplete_pair_keys = list(
        pattern_df.attrs.get(
            "incomplete_pair_keys",
            group_df.attrs.get("incomplete_pair_keys", []),
        )
    )
    if min_seq_len_exclusive is not None:
        threshold = int(min_seq_len_exclusive)
        incomplete_pair_keys = [
            key for key in incomplete_pair_keys if int(key[3]) > threshold
        ]
    omitted_count = len(incomplete_pair_keys)
    incomplete_family_ids = {str(key[0]) for key in incomplete_pair_keys}
    if min_seq_len_exclusive is not None:
        threshold = int(min_seq_len_exclusive)
        pattern_df = pattern_df[pattern_df["seq_len"] > threshold].copy()
        group_df = group_df[group_df["seq_len"] > threshold].copy()
    if pattern_df.empty and group_df.empty:
        return _render_empty_figure(
            MODE_TITLES[execution_mode],
            message=(
                "Incomplete selected-component data"
                if omitted_count
                else "No data available"
            ),
            omitted_count=omitted_count,
        )

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

    present_family_ids = (
        set(
            pattern_df["study_case_id"].astype(str).tolist()
            + group_df["study_case_id"].astype(str).tolist()
        )
        | incomplete_family_ids
    )
    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=figsize,
        sharey=True,
    )
    axes_grid = axes_obj
    bar_width = 0.34
    x_positions = list(range(len(seq_order)))
    group_order = _group_order(execution_mode)

    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        if family.family_id not in present_family_ids:
            ax.set_axis_off()
            continue

        family_patterns = pattern_df[
            pattern_df["study_case_id"] == family.family_id
        ].copy()
        family_groups = group_df[group_df["study_case_id"] == family.family_id].copy()
        if family_patterns.empty and family.family_id in incomplete_family_ids:
            ax.text(
                0.5,
                0.5,
                "Incomplete selected-component data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=axis_label_size,
                color="#5a5249",
                fontweight="bold",
            )
            ax.set_title(
                plot_family_label(family.family_id),
                loc="left",
                fontsize=family_title_size,
                pad=6,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue
        pattern_map = {
            int(row.seq_len): float(row.avg_latency_ms)
            for row in family_patterns.itertuples(index=False)
        }
        group_map = {
            (int(row.seq_len), str(row.group_label)): float(row.avg_latency_ms)
            for row in family_groups.itertuples(index=False)
        }

        bottoms = [0.0] * len(seq_order)
        for group_label in group_order:
            heights = [
                group_map.get((seq_len, group_label), 0.0) for seq_len in seq_order
            ]
            ax.bar(
                [x - (bar_width / 2) for x in x_positions],
                heights,
                width=bar_width,
                bottom=bottoms,
                color=GROUP_COLORS[group_label],
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
        for x_pos, stack_total, pattern_total in zip(
            x_positions,
            bottoms,
            pattern_heights,
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
            ax.set_ylim(ymin, max(ymax, max(pair_maxima, default=1.0) * 1.45))
        else:
            ax.set_ylim(0, max(pair_maxima, default=1.0) * 1.18)
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
        Patch(facecolor=GROUP_COLORS[label], edgecolor="none", label=label)
        for label in group_order
    ]
    legend_handles.append(
        Patch(
            facecolor=PATTERN_COLOR,
            edgecolor=PATTERN_EDGE,
            label=MODE_PATTERN_LABELS[execution_mode],
        )
    )
    legend_ncol = 2 if execution_mode == "runlist" else 1
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.89, 0.5),
        fontsize=legend_font_size,
        columnspacing=0.9,
        handlelength=1.2,
    )
    title = MODE_TITLES[execution_mode]
    if min_seq_len_exclusive is not None:
        title = f"{title} (>{int(min_seq_len_exclusive)} Tokens)"
    fig.suptitle(
        title,
        fontsize=suptitle_size,
        fontweight="bold",
        y=title_y,
    )
    if omitted_count:
        fig.text(
            0.5,
            title_y - 0.035,
            f"Incomplete comparisons omitted: {omitted_count}",
            ha="center",
            va="center",
            fontsize=max(10, legend_font_size - 2),
            color="#5a5249",
        )
    right_margin = 0.82 if execution_mode == "runlist" else 0.89
    fig.tight_layout(rect=[0, 0.03, right_margin, 0.95])
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stacked comparison of selected hybrid/runlist/offload component-group latencies versus full-pattern latency."
    )
    parser.add_argument("--results", type=Path, default=default_results_csv())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--mode", choices=("hybrid", "runlist", "offload"), required=True
    )
    parser.add_argument("--variant", choices=("standard", "slides"), default="standard")
    parser.add_argument("--y-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--min-seq-len-exclusive", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pattern_df, group_df = load_plot_rows(args.results, execution_mode=args.mode)
    fig = render_plot(
        pattern_df,
        group_df,
        execution_mode=args.mode,
        variant=args.variant,
        y_scale=args.y_scale,
        min_seq_len_exclusive=args.min_seq_len_exclusive,
    )
    stem = args.stem or variant_stem(
        args.mode,
        args.variant,
        args.y_scale,
        args.min_seq_len_exclusive,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
