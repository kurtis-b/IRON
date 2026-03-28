#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.pipeline.run_study_pipeline import (
    load_pipeline_config,
    resolve_path,
    run_study_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full transformer_layer study pipeline unattended."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-thermal-recovery", action="store_true")
    parser.add_argument(
        "--skip-power-cycle",
        dest="skip_thermal_recovery",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--power-backend",
        choices=("none", "turbostat_pkgwatt"),
        default=None,
    )
    parser.add_argument("--power-sample-interval-sec", type=float, default=None)
    parser.add_argument("--quiescent-baseline-duration-sec", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_pipeline_config(args.config)
    npu_power = dict(config["npu_power"])
    if args.power_backend is None:
        args.power_backend = str(npu_power["power_backend"])
    if args.power_sample_interval_sec is None:
        args.power_sample_interval_sec = float(npu_power["power_sample_interval_sec"])
    if args.quiescent_baseline_duration_sec is None:
        args.quiescent_baseline_duration_sec = float(
            npu_power["quiescent_baseline_duration_sec"]
        )
    run_study_pipeline(config, args)


__all__ = [
    "resolve_path",
    "load_pipeline_config",
    "run_study_pipeline",
]


if __name__ == "__main__":
    main()
