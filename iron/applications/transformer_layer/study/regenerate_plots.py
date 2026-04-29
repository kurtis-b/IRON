#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from matplotlib import pyplot as plt

from .block.plot_best_latency import (
    load_best_rows as load_block_rows,
    plot_best_latency,
    variant_stem as block_variant_stem,
)
from .end_to_end.plot_dataflow_blocks_vs_pattern import (
    load_plot_rows as load_dataflow_rows,
    render_plot as render_dataflow_plot,
    variant_stem as dataflow_variant_stem,
)
from .end_to_end.plot_tps_by_pattern import (
    load_plot_rows as load_end_to_end_rows,
    render_plot as render_end_to_end_plot,
    variant_stem as end_to_end_variant_stem,
)
from .end_to_end.plot_selected_component_groups_vs_pattern import (
    load_plot_rows as load_selected_component_rows,
    render_plot as render_selected_component_plot,
    variant_stem as selected_component_variant_stem,
)
from .host_comparison.run import write_plots as write_host_comparison_plots
from .memory_tile_staging.plot_staging_depth import write_canonical_plots

LOGGER = logging.getLogger(__name__)
END_TO_END_LONG_SEQUENCE_MIN_SEQ_LEN_EXCLUSIVE = 1024


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected results file is missing: {path}")


def _write_figure(fig, *, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s and %s", png_path, svg_path)


def regenerate_block_plots(results_root: Path) -> None:
    results_csv = results_root / "block" / "results.csv"
    _require_file(results_csv)
    best_rows = load_block_rows(results_csv)
    output_dir = results_csv.parent

    for variant in ("standard", "slides"):
        for y_scale in ("log", "linear"):
            fig = plot_best_latency(
                best_rows,
                "Latency Comparison of Encoder and Decoder Blocks",
                variant=variant,
                y_scale=y_scale,
            )
            _write_figure(
                fig,
                output_dir=output_dir,
                stem=block_variant_stem(variant, y_scale),
            )


def regenerate_end_to_end_summary_plots(
    results_root: Path,
    *,
    require_selected_components: bool = False,
) -> None:
    results_csv = results_root / "end_to_end" / "results_all_power.csv"
    tuning_csv = results_root / "end_to_end" / "tuning_all_power.csv"
    selected_components_csv = (
        results_root / "end_to_end" / "selected_component_aggregates.csv"
    )
    _require_file(results_csv)
    _require_file(tuning_csv)
    output_dir = results_csv.parent

    pattern_rows = load_end_to_end_rows(results_csv)
    for variant in ("standard", "slides"):
        for min_seq_len_exclusive in (
            None,
            END_TO_END_LONG_SEQUENCE_MIN_SEQ_LEN_EXCLUSIVE,
        ):
            throughput_fig = render_end_to_end_plot(
                pattern_rows,
                metric="throughput",
                variant=variant,
                min_seq_len_exclusive=min_seq_len_exclusive,
            )
            _write_figure(
                throughput_fig,
                output_dir=output_dir,
                stem=end_to_end_variant_stem(
                    "throughput",
                    variant,
                    min_seq_len_exclusive=min_seq_len_exclusive,
                ),
            )
        for y_scale in ("log", "linear"):
            for min_seq_len_exclusive in (
                None,
                END_TO_END_LONG_SEQUENCE_MIN_SEQ_LEN_EXCLUSIVE,
            ):
                latency_fig = render_end_to_end_plot(
                    pattern_rows,
                    metric="latency",
                    variant=variant,
                    y_scale=y_scale,
                    min_seq_len_exclusive=min_seq_len_exclusive,
                )
                _write_figure(
                    latency_fig,
                    output_dir=output_dir,
                    stem=end_to_end_variant_stem(
                        "latency",
                        variant,
                        y_scale,
                        min_seq_len_exclusive=min_seq_len_exclusive,
                    ),
                )

    full_pattern_rows, selected_block_rows = load_dataflow_rows(results_csv, tuning_csv)
    for variant in ("standard", "slides"):
        for y_scale in ("log", "linear"):
            for min_seq_len_exclusive in (
                None,
                END_TO_END_LONG_SEQUENCE_MIN_SEQ_LEN_EXCLUSIVE,
            ):
                fig = render_dataflow_plot(
                    full_pattern_rows,
                    selected_block_rows,
                    variant=variant,
                    y_scale=y_scale,
                    min_seq_len_exclusive=min_seq_len_exclusive,
                )
                _write_figure(
                    fig,
                    output_dir=output_dir,
                    stem=dataflow_variant_stem(
                        variant,
                        y_scale,
                        min_seq_len_exclusive=min_seq_len_exclusive,
                    ),
                )

    if selected_components_csv.exists():
        for execution_mode in ("hybrid", "runlist", "offload"):
            pattern_rows, group_rows = load_selected_component_rows(
                selected_components_csv,
                execution_mode=execution_mode,
            )
            for variant in ("standard", "slides"):
                for y_scale in ("log", "linear"):
                    for min_seq_len_exclusive in (
                        None,
                        END_TO_END_LONG_SEQUENCE_MIN_SEQ_LEN_EXCLUSIVE,
                    ):
                        fig = render_selected_component_plot(
                            pattern_rows,
                            group_rows,
                            execution_mode=execution_mode,
                            variant=variant,
                            y_scale=y_scale,
                            min_seq_len_exclusive=min_seq_len_exclusive,
                        )
                        _write_figure(
                            fig,
                            output_dir=output_dir,
                            stem=selected_component_variant_stem(
                                execution_mode,
                                variant,
                                y_scale,
                                min_seq_len_exclusive=min_seq_len_exclusive,
                            ),
                        )
    else:
        if require_selected_components:
            _require_file(selected_components_csv)
        LOGGER.info(
            "Skipping selected-component aggregate plots because %s is missing",
            selected_components_csv,
        )


def regenerate_memory_tile_staging_plots(results_root: Path) -> None:
    results_csv = results_root / "memory_tile_staging" / "results.csv"
    _require_file(results_csv)
    write_canonical_plots(results_csv, results_csv.parent)
    LOGGER.info("Wrote memory-tile staging plots under %s", results_csv.parent)


def regenerate_host_comparison_plots(results_root: Path) -> None:
    results_csv = results_root / "host_comparison" / "results.csv"
    _require_file(results_csv)
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    write_host_comparison_plots(
        rows,
        effective_gflops_plot_path=results_csv.parent
        / "effective_gflops_comparison.svg",
        effective_gflops_per_watt_plot_path=results_csv.parent
        / "effective_gflops_per_watt_comparison.svg",
    )
    LOGGER.info("Wrote host-comparison plots under %s", results_csv.parent)


def regenerate_all_plots(
    results_root: Path,
    *,
    require_selected_components: bool = False,
) -> None:
    regenerate_block_plots(results_root)
    regenerate_end_to_end_summary_plots(
        results_root,
        require_selected_components=require_selected_components,
    )
    regenerate_memory_tile_staging_plots(results_root)
    regenerate_host_comparison_plots(results_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate summary plot suites from an existing transformer-layer results root."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
        help="Root directory containing block/, end_to_end/, memory_tile_staging/, and host_comparison/ results",
    )
    parser.add_argument(
        "--require-selected-components",
        action="store_true",
        help="Fail if selected-component aggregate CSVs are missing.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    regenerate_all_plots(
        args.results_root.expanduser(),
        require_selected_components=args.require_selected_components,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
