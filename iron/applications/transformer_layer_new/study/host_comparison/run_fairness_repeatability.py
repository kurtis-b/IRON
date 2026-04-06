#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .run import (
    SUPPORTED_IGPU_POWER_BACKENDS,
    iteration_schedule,
)
from .select import REFERENCE_EXECUTION_MODES

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "backend",
    "reference_execution_modes_json",
    "iteration_schedule_json",
    "device",
    "power_backend",
    "supported_power_backends_json",
    "validation_policy",
    "latency_variation_policy",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "host_comparison"
        / "fairness_repeatability.csv"
    )


def _iteration_schedule_summary() -> dict[str, dict[str, int]]:
    return {
        "64-256": {
            "warmup_runs": iteration_schedule(64)[0],
            "runs_per_sample": iteration_schedule(64)[1],
        },
        "512-2048": {
            "warmup_runs": iteration_schedule(512)[0],
            "runs_per_sample": iteration_schedule(512)[1],
        },
        "4096": {
            "warmup_runs": iteration_schedule(4096)[0],
            "runs_per_sample": iteration_schedule(4096)[1],
        },
        "8192-16384": {
            "warmup_runs": iteration_schedule(8192)[0],
            "runs_per_sample": iteration_schedule(8192)[1],
        },
    }


def build_rows(
    *,
    igpu_device: str,
    igpu_power_backend: str,
) -> list[dict[str, object]]:
    reference_modes_json = json.dumps(list(REFERENCE_EXECUTION_MODES), sort_keys=True)
    iteration_schedule_json = json.dumps(_iteration_schedule_summary(), sort_keys=True)
    return [
        {
            "study_id": "host_comparison_fairness_repeatability",
            "backend": "igpu",
            "reference_execution_modes_json": reference_modes_json,
            "iteration_schedule_json": iteration_schedule_json,
            "device": igpu_device,
            "power_backend": igpu_power_backend,
            "supported_power_backends_json": json.dumps(
                list(SUPPORTED_IGPU_POWER_BACKENDS)
            ),
            "validation_policy": (
                "The host comparison validates exactly through seq_len=512 and uses "
                "finite-output validation above that threshold."
            ),
            "latency_variation_policy": (
                "The host comparison reuses its default iteration schedule unless "
                "warmup or timed iterations are overridden at the command line."
            ),
        },
    ]


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a fairness/repeatability summary for the host comparison study."
    )
    parser.add_argument("--igpu-device", default="cuda:0")
    parser.add_argument(
        "--igpu-power-backend",
        choices=list(SUPPORTED_IGPU_POWER_BACKENDS),
        default="rocm-smi",
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    rows = build_rows(
        igpu_device=str(args.igpu_device),
        igpu_power_backend=str(args.igpu_power_backend),
    )
    write_rows(args.output.expanduser(), rows)
    LOGGER.info("Wrote %d host comparison fairness rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
