#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from ..npu_runtime_checks import (
    require_npu_power_mode_turbo,
    warn_if_npu_power_mode_not_turbo,
)
from ..run_lock import default_lock_path, hold_study_lock
import seaborn as sns
from matplotlib import pyplot as plt

from .cases import (
    NUM_CHANNELS,
    NUM_CORES,
    SIZE_LADDER,
    STUDY_ID,
    MemcpyBandwidthCase,
    iter_cases,
)

LOGGER = logging.getLogger(__name__)

REL_TOL = 0.01
ABS_TOL = 1.0e-6
PLOT_SUFFIX = ".svg"
CSV_FIELDNAMES = (
    "study_id",
    "case_id",
    "size_elements",
    "size_bytes",
    "total_moved_bytes",
    "num_cores",
    "num_channels",
    "bypass",
    "tile_size",
    "warmup_iters",
    "timed_iters",
    "latency_us",
    "bandwidth_gbps",
    "validation_error_count",
    "run_status",
    "failure_message",
    "is_size_peak",
    "is_overall_peak",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "memcpy_bandwidth"
        / "results.csv"
    )


def default_resume_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2]
        / "results_final"
        / "memcpy_bandwidth"
        / output_path.name
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def default_bandwidth_plot_path(output_path: Path) -> Path:
    return output_path.with_name("bandwidth_by_shim_tiles.svg")


def _resolve_plot_path(path: Path) -> Path:
    if path.suffix.lower() != PLOT_SUFFIX:
        raise ValueError(f"Plot output must use the {PLOT_SUFFIX} suffix: {path}")
    return path


def _format_metric_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000.0:
        return f"{value:,.0f}"
    if magnitude >= 100.0:
        return f"{value:,.1f}"
    if magnitude >= 10.0:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _new_benchmark_context(case: MemcpyBandwidthCase):
    from iron.common import AIEContext

    context = AIEContext()
    context.build_dir = (
        Path.cwd() / "build" / "transformer_layer_new_memcpy_bandwidth" / case.case_id
    )
    return context


def _validation_result(errors: dict[str, list[int]]) -> dict[str, object]:
    if not errors:
        return {
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    counts = {
        buffer_name: len(indices) for buffer_name, indices in sorted(errors.items())
    }
    failure_message = ", ".join(
        f"{buffer_name}={count}" for buffer_name, count in counts.items()
    )
    return {
        "validation_error_count": sum(counts.values()),
        "run_status": "failed_validation",
        "failure_message": f"validation failed: {failure_message}",
    }


def benchmark_case(
    case: MemcpyBandwidthCase,
    *,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, object]:
    require_npu_power_mode_turbo(study_name="memcpy-bandwidth study")
    from iron.common.test_utils import run_test
    from iron.operators.mem_copy.op import AIEMemCopy
    from iron.operators.mem_copy.reference import generate_golden_reference

    context = _new_benchmark_context(case)
    reference = generate_golden_reference(input_length=case.size_elements)
    operator = AIEMemCopy(
        size=case.size_elements,
        num_cores=case.num_cores,
        num_channels=case.num_channels,
        bypass=case.bypass,
        tile_size=case.tile_size,
        context=context,
    )
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"input": reference["inout"]},
        {"output": reference["inout"]},
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )
    return {
        "latency_us": latency_us,
        "bandwidth_gbps": bandwidth_gbps,
        **_validation_result(errors),
    }


def _failed_result(message: str) -> dict[str, object]:
    return {
        "latency_us": None,
        "bandwidth_gbps": None,
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": message,
    }


def _best_row_key(row: dict[str, object]) -> tuple[float, float, int, int, int]:
    bandwidth = float(row["bandwidth_gbps"])
    latency = float(row["latency_us"])
    return (
        -bandwidth,
        latency,
        -int(row["num_channels"]),
        -int(row["num_cores"]),
        int(bool(row["bypass"])),
    )


def _existing_row_key(row: dict[str, object]) -> str:
    return str(row.get("case_id") or "")


def load_existing_rows(
    paths: tuple[Path, ...],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = _existing_row_key(row)
                if key:
                    rows[key] = dict(row)
    return rows


def reusable_existing_row(
    existing_rows: dict[str, dict[str, object]],
    *,
    case: MemcpyBandwidthCase,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, object] | None:
    row = existing_rows.get(case.case_id)
    if row is None:
        return None
    if str(row.get("warmup_iters") or "") != str(warmup_iters):
        return None
    if str(row.get("timed_iters") or "") != str(timed_iters):
        return None
    if str(row.get("run_status") or "") not in {
        "passed",
        "failed_validation",
        "failed_exception",
    }:
        return None
    return dict(row)


def mark_peak_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["is_size_peak"] = False
        row["is_overall_peak"] = False

    successful_rows = [
        row
        for row in rows
        if row.get("run_status") == "passed"
        and row.get("bandwidth_gbps") is not None
        and row.get("latency_us") is not None
    ]
    if not successful_rows:
        return

    size_values = sorted({int(row["size_elements"]) for row in successful_rows})
    for size_value in size_values:
        size_rows = [
            row for row in successful_rows if int(row["size_elements"]) == size_value
        ]
        best_row = min(size_rows, key=_best_row_key)
        best_row["is_size_peak"] = True

    best_row = min(successful_rows, key=_best_row_key)
    best_row["is_overall_peak"] = True


def build_rows(
    *,
    size_filter: str,
    num_cores_filter: str,
    num_channels_filter: str,
    bypass_filter: str,
    warmup_iters: int,
    timed_iters: int,
    existing_rows: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in iter_cases(
        size_filter=size_filter,
        num_cores_filter=num_cores_filter,
        num_channels_filter=num_channels_filter,
        bypass_filter=bypass_filter,
    ):
        reusable_row = reusable_existing_row(
            {} if existing_rows is None else existing_rows,
            case=case,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        if reusable_row is not None:
            rows.append(reusable_row)
            LOGGER.info("Reusing memcpy bandwidth row for %s", case.case_id)
            continue
        try:
            result = benchmark_case(
                case,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
            )
        except Exception as exc:
            LOGGER.warning("Failed %s: %s", case.case_id, exc)
            result = _failed_result(str(exc))

        row = {
            "study_id": STUDY_ID,
            "case_id": case.case_id,
            "size_elements": case.size_elements,
            "size_bytes": case.size_bytes,
            "total_moved_bytes": case.total_moved_bytes,
            "num_cores": case.num_cores,
            "num_channels": case.num_channels,
            "bypass": case.bypass,
            "tile_size": case.tile_size,
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "is_size_peak": False,
            "is_overall_peak": False,
            **result,
        }
        rows.append(row)
        LOGGER.info(
            "Completed %s -> %s",
            case.case_id,
            row["run_status"],
        )

    mark_peak_rows(rows)
    return rows


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def write_peak_metric_plot(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    metric_key: str,
    title: str,
    y_axis_label: str,
    bar_color: str,
) -> None:
    output_path = _resolve_plot_path(output_path)
    plot_rows = sorted(
        [
            row
            for row in rows
            if row.get("run_status") == "passed"
            and row.get(metric_key) is not None
            and bool(row.get("bypass"))
        ],
        key=lambda row: int(row["num_cores"]) // int(row["num_channels"]),
    )
    plt.rcParams["svg.fonttype"] = "none"
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

    fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)

    if not plot_rows:
        ax.text(
            0.5,
            0.5,
            "No successful results",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.97)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    shim_tile_counts = sorted(
        {int(row["num_cores"]) // int(row["num_channels"]) for row in plot_rows}
    )
    x_positions = list(range(len(shim_tile_counts)))
    grouped_rows = {
        int(row["num_cores"]) // int(row["num_channels"]): row for row in plot_rows
    }
    values = [
        float(grouped_rows[shim_tile_count][metric_key])
        for shim_tile_count in shim_tile_counts
    ]
    ax.bar(
        x_positions,
        values,
        width=0.62,
        color=bar_color,
        edgecolor="#2b2b2b",
        linewidth=1.0,
        zorder=2,
    )
    value_offset = max(values) * 0.04 if max(values) > 0 else 0.1
    for bar_x, value in zip(x_positions, values, strict=True):
        ax.text(
            bar_x,
            value + value_offset,
            _format_metric_value(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2b2b2b",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [str(shim_tile_count) for shim_tile_count in shim_tile_counts], fontsize=11
    )
    ax.set_xlabel("Shim Tile Count", fontsize=14)
    ax.set_ylabel(y_axis_label, fontsize=14)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="x")
    ax.margins(x=0.04)
    max_value = max(float(row[metric_key]) for row in plot_rows)
    ax.set_ylim(0.0, max_value * 1.24 if max_value > 0 else 1.0)
    ax.tick_params(axis="y", labelsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_plots(
    rows: list[dict[str, object]],
    *,
    bandwidth_plot_path: Path,
) -> None:
    write_peak_metric_plot(
        bandwidth_plot_path,
        rows,
        metric_key="bandwidth_gbps",
        title="Bandwidth by Shim Tile Count",
        y_axis_label="Effective Bandwidth (GB/s)",
        bar_color="#0072b2",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark peak NPU memory bandwidth with AIEMemCopy"
    )
    parser.add_argument(
        "--size",
        choices=[*(str(value) for value in SIZE_LADDER), "all"],
        default="all",
    )
    parser.add_argument(
        "--num-cores",
        choices=[*(str(value) for value in NUM_CORES), "all"],
        default="all",
    )
    parser.add_argument(
        "--num-channels",
        choices=[*(str(value) for value in NUM_CHANNELS), "all"],
        default="all",
    )
    parser.add_argument(
        "--bypass",
        choices=("all", "false", "true"),
        default="true",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--timed-iters", type=int, default=500)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--resume-input", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--bandwidth-plot", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be >= 0")
    if args.timed_iters <= 0:
        parser.error("--timed-iters must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    warn_if_npu_power_mode_not_turbo(LOGGER, study_name="memcpy-bandwidth study")

    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="memcpy-bandwidth study",
    ):
        if args.no_resume:
            resume_paths: tuple[Path, ...] = tuple()
        elif args.resume_input is not None:
            resume_paths = (args.resume_input.expanduser(),)
        else:
            resume_paths = default_resume_paths(output_path)
        bandwidth_plot_path = (
            default_bandwidth_plot_path(output_path)
            if args.bandwidth_plot is None
            else _resolve_plot_path(args.bandwidth_plot.expanduser())
        )
        existing_rows = load_existing_rows(resume_paths)
        if resume_paths and existing_rows:
            LOGGER.info(
                "Loaded %d reusable memcpy bandwidth rows from %s",
                len(existing_rows),
                ", ".join(str(path) for path in resume_paths),
            )

        rows = build_rows(
            size_filter=str(args.size),
            num_cores_filter=str(args.num_cores),
            num_channels_filter=str(args.num_channels),
            bypass_filter=str(args.bypass),
            warmup_iters=int(args.warmup_iters),
            timed_iters=int(args.timed_iters),
            existing_rows=existing_rows,
        )
        if not rows:
            LOGGER.warning(
                "No memcpy bandwidth cases matched size=%s num_cores=%s num_channels=%s "
                "bypass=%s; writing empty CSV and placeholder plots",
                args.size,
                args.num_cores,
                args.num_channels,
                args.bypass,
            )

        write_rows(output_path, rows)
        write_plots(
            rows,
            bandwidth_plot_path=bandwidth_plot_path,
        )
        LOGGER.info("Wrote %d memcpy bandwidth rows to %s", len(rows), output_path)
        LOGGER.info("Wrote memcpy bandwidth plot to %s", bandwidth_plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
