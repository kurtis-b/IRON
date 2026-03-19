#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AUTOMATED_BENCHMARK = SCRIPT_DIR / "automated_benchmark.py"

OPTION_MAP = {
    "study_id": "--study-id",
    "study_ids": "--study-ids",
    "study_manifest": "--study-manifest",
    "models_root": "--models-root",
    "modes": "--modes",
    "seq_lens": "--seq-lens",
    "num_samples": "--num-samples",
    "warmup_runs": "--warmup-runs",
    "runs_per_sample": "--runs-per-sample",
    "cpu_thread_counts": "--cpu-thread-counts",
    "npu_num_threads": "--npu-num-threads",
    "igpu_num_threads": "--igpu-num-threads",
    "igpu_dtype": "--igpu-dtype",
    "igpu_device_index": "--igpu-device-index",
    "npu_topology_policy": "--npu-topology-policy",
    "npu_topology_cache": "--npu-topology-cache",
    "npu_candidate_topologies": "--npu-candidate-topologies",
    "npu_autotune_warmup_runs": "--npu-autotune-warmup-runs",
    "npu_autotune_runs": "--npu-autotune-runs",
    "power_backend": "--power-backend",
    "power_interval_sec": "--power-interval-sec",
    "npu_idle_baseline_sec": "--npu-idle-baseline-sec",
    "cooldown_sec": "--cooldown-sec",
    "cooldown_until_temp_c": "--cooldown-until-temp-c",
    "cooldown_temp_source": "--cooldown-temp-source",
    "cooldown_temp_tolerance_frac": "--cooldown-temp-tolerance-frac",
    "cooldown_poll_sec": "--cooldown-poll-sec",
    "cooldown_timeout_sec": "--cooldown-timeout-sec",
    "power_cycle_cmd": "--power-cycle-cmd",
    "state_json": "--state-json",
    "output_csv": "--output-csv",
    "logs_dir": "--logs-dir",
}
PATH_KEYS = {
    "weights_file_path",
    "config_file_path",
    "study_manifest",
    "models_root",
    "npu_topology_cache",
    "state_json",
    "output_csv",
    "logs_dir",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run automated_benchmark.py from a JSON job specification."
    )
    parser.add_argument("job_config", type=str, help="Path to the job JSON file.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved automated_benchmark command and exit.",
    )
    return parser.parse_args()


def load_job_config(job_config_path):
    with open(job_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(job_dir, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((job_dir / path).resolve())


def build_command(job_config_path, job):
    job_dir = Path(job_config_path).resolve().parent
    has_explicit_paths = "weights_file_path" in job and "config_file_path" in job
    has_study_id = "study_id" in job
    has_study_ids = "study_ids" in job
    if sum((has_explicit_paths, has_study_id, has_study_ids)) != 1:
        raise ValueError(
            "Job config must include exactly one of weights_file_path/config_file_path, study_id, or study_ids"
        )

    command = [
        sys.executable,
        str(AUTOMATED_BENCHMARK),
    ]
    if has_explicit_paths:
        command.extend(
            [
                resolve_path(job_dir, job["weights_file_path"]),
                resolve_path(job_dir, job["config_file_path"]),
            ]
        )

    for key, flag in OPTION_MAP.items():
        value = job.get(key)
        if value is None:
            continue
        if key in PATH_KEYS:
            value = resolve_path(job_dir, value)
        command.extend([flag, str(value)])
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
