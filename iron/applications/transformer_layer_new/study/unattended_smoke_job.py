#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REQUIRED_RESULTS_FILES = (
    ("block", "results.csv"),
    ("end_to_end", "results_all_power.csv"),
    ("end_to_end", "tuning_all_power.csv"),
    ("memory_tile_staging", "results.csv"),
    ("host_comparison", "results.csv"),
)

REQUIRED_PLOT_FILES = (
    ("block", "best_latency_by_block.svg"),
    ("end_to_end", "effective_gflops_per_second_by_pattern.svg"),
    ("end_to_end", "latency_by_pattern.svg"),
    ("end_to_end", "dataflow_selected_blocks_vs_pattern_latency.svg"),
    ("memory_tile_staging", "mha_out_proj_latency_by_staging_depth.svg"),
    ("host_comparison", "effective_gflops_comparison.svg"),
    ("host_comparison", "effective_gflops_per_watt_comparison.svg"),
)


def _copy_required_results(source_root: Path, target_root: Path) -> None:
    for parts in REQUIRED_RESULTS_FILES:
        source_path = source_root.joinpath(*parts)
        target_path = target_root.joinpath(*parts)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing smoke-test fixture input: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _verify_required_outputs(results_root: Path) -> None:
    missing = []
    for parts in REQUIRED_PLOT_FILES:
        plot_path = results_root.joinpath(*parts)
        if not plot_path.exists() or plot_path.stat().st_size <= 0:
            missing.append(str(plot_path))
    if missing:
        raise FileNotFoundError(
            "Smoke-test plot regeneration did not produce expected outputs:\n"
            + "\n".join(missing)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Helper commands for unattended reboot smoke tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-results")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--target-root", type=Path, required=True)

    verify = subparsers.add_parser("verify-results")
    verify.add_argument("--results-root", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-results":
        _copy_required_results(
            args.source_root.expanduser(),
            args.target_root.expanduser(),
        )
        return 0
    if args.command == "verify-results":
        _verify_required_outputs(args.results_root.expanduser())
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
