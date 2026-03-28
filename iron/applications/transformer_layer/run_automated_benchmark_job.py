#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.pipeline.run_automated_benchmark_job import (
    build_command,
    load_job_config,
    resolve_path,
    run_job,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run transformer_layer/automated_benchmark.py from a JSON job."
    )
    parser.add_argument("job_config", help="Path to the job JSON file.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved command and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(run_job(args.job_config, print_command=args.print_command))


__all__ = [
    "load_job_config",
    "resolve_path",
    "build_command",
    "run_job",
]


if __name__ == "__main__":
    main()
