#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AUTOMATED_BENCHMARK = SCRIPT_DIR / "automated_benchmark.py"

OPTION_MAP = {
    "study_manifest": "--study-manifest",
    "execution_modes": "--execution-modes",
    "seq_lens": "--seq-lens",
    "output_csv": "--output-csv",
    "warmup_runs": "--warmup-runs",
    "runs_per_sample": "--runs-per-sample",
    "hidden_size": "--hidden-size",
    "intermediate_size": "--intermediate-size",
    "num_attention_heads": "--num-attention-heads",
    "seed": "--seed",
    "parity_output_csv": "--parity-output-csv",
}
PATH_KEYS = {"study_manifest", "output_csv", "parity_output_csv"}
BOOLEAN_FLAGS = {"run_parity_check": "--run-parity-check"}


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


def load_job_config(job_config_path: str) -> dict[str, object]:
    return json.loads(Path(job_config_path).read_text(encoding="utf-8"))


def resolve_path(job_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((job_dir / path).resolve())


def build_command(job_config_path: str, job: dict[str, object]) -> list[str]:
    job_dir = Path(job_config_path).resolve().parent
    if "study_manifest" not in job:
        raise ValueError("Job config must include study_manifest")

    command = [sys.executable, str(AUTOMATED_BENCHMARK)]
    for key, flag in OPTION_MAP.items():
        value = job.get(key)
        if value is None:
            continue
        if key in PATH_KEYS:
            value = resolve_path(job_dir, str(value))
        command.extend([flag, str(value)])
    for key, flag in BOOLEAN_FLAGS.items():
        if job.get(key):
            command.append(flag)
    return command


def main():
    args = parse_args()
    job = load_job_config(args.job_config)
    command = build_command(args.job_config, job)
    if args.print_command:
        print(" ".join(command))
        return

    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
