#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from benchmark_common import (
    detect_logical_thread_count,
    detect_physical_core_count,
    parse_seq_lens,
)

SCRIPT_DIR = Path(__file__).parent
SUITE_FIELDNAMES = [
    "case_id",
    "mode",
    "seq_len",
    "num_threads",
    "dtype",
    "num_samples",
    "runs_per_sample",
    "warmup_runs",
    "model_type",
    "shape",
    "topology_id",
    "parallel_seq",
    "parallel_heads",
    "parallel_ffn",
    "min_latency_ms",
    "avg_latency_ms",
    "max_latency_ms",
    "power_backend",
    "power_sample_count",
    "idle_power_sample_count",
    "idle_pkg_watt",
    "idle_cor_watt",
    "idle_gfx_watt",
    "idle_ram_watt",
    "avg_pkg_watt",
    "max_pkg_watt",
    "avg_cor_watt",
    "max_cor_watt",
    "avg_gfx_watt",
    "max_gfx_watt",
    "avg_ram_watt",
    "max_ram_watt",
    "pseudo_npu_avg_pkg_watt",
    "pseudo_npu_max_pkg_watt",
    "idle_power_log",
    "power_log",
    "benchmark_csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automate encoder CPU/NPU benchmark cases with resumable state, optional "
            "power logging, and optional power-cycle hooks."
        )
    )
    parser.add_argument("weights_file_path", type=str, nargs="?")
    parser.add_argument("config_file_path", type=str, nargs="?")
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Resolve weights/config from the study manifest instead of passing paths.",
    )
    parser.add_argument(
        "--study-manifest",
        type=str,
        default=None,
        help="Optional path to the study manifest JSON.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Optional root directory that holds downloaded study model artifacts.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="cpu,npu",
        help="Comma-separated benchmark modes to run. Supported: cpu,npu",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="64,128,256,512,1024,2048,4096,8192",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--runs-per-sample", type=int, default=100)
    parser.add_argument(
        "--cpu-thread-counts",
        type=str,
        default=None,
        help="Comma-separated CPU thread counts. Defaults to physical and logical.",
    )
    parser.add_argument(
        "--npu-num-threads",
        type=int,
        default=None,
        help="Host CPU thread count for the NPU benchmark. Defaults to physical cores.",
    )
    parser.add_argument(
        "--npu-topology-policy",
        choices=("fixed", "cache", "autotune"),
        default="cache",
    )
    parser.add_argument(
        "--npu-topology-cache",
        type=str,
        default="npu_topology_cache_latest.json",
    )
    parser.add_argument(
        "--npu-candidate-topologies",
        type=str,
        default=None,
        help="Optional comma-separated NPU topology ids for autotune/cache misses.",
    )
    parser.add_argument("--npu-autotune-warmup-runs", type=int, default=2)
    parser.add_argument("--npu-autotune-runs", type=int, default=5)
    parser.add_argument(
        "--power-backend",
        choices=("none", "turbostat"),
        default="turbostat",
        help="Per-case power measurement backend. turbostat requires privileged access.",
    )
    parser.add_argument("--power-interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--npu-idle-baseline-sec",
        type=float,
        default=5.0,
        help="Idle measurement duration before each NPU case when power logging is enabled.",
    )
    parser.add_argument(
        "--power-cycle-cmd",
        type=str,
        default=None,
        help=(
            "Optional shell command invoked after each completed case except the last. "
            "Use this with the resume state file and a boot-time relaunch mechanism."
        ),
    )
    parser.add_argument(
        "--state-json",
        type=str,
        default="automated_benchmark_state.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="automated_benchmark_latest.csv",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/automated_benchmark",
    )
    return parser.parse_args()


def parse_modes(raw_modes):
    modes = []
    for token in raw_modes.split(","):
        token = token.strip()
        if token:
            modes.append(token)
    invalid = sorted(set(modes) - {"cpu", "npu"})
    if invalid:
        raise ValueError(f"Unsupported benchmark modes: {invalid}")
    if not modes:
        raise ValueError("At least one benchmark mode must be selected")
    return modes


def parse_thread_counts(raw_thread_counts):
    if raw_thread_counts is None:
        counts = [detect_physical_core_count()]
        logical = detect_logical_thread_count()
        if logical not in counts:
            counts.append(logical)
        return counts

    parsed = []
    for token in raw_thread_counts.split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count <= 0:
            raise ValueError(f"Thread counts must be positive (got {count})")
        if count not in parsed:
            parsed.append(count)
    if not parsed:
        raise ValueError("At least one CPU thread count must be provided")
    return parsed


def topology_suffix(case):
    topology_id = case.get("topology_id")
    return f"_{topology_id}" if topology_id else ""


def enumerate_cases(args):
    seq_lens = parse_seq_lens(args.seq_lens)
    modes = parse_modes(args.modes)
    cases = []
    if "cpu" in modes:
        for num_threads in parse_thread_counts(args.cpu_thread_counts):
            for seq_len in seq_lens:
                cases.append(
                    {
                        "case_id": f"cpu_seq{seq_len}_{num_threads}t",
                        "mode": "cpu",
                        "seq_len": seq_len,
                        "num_threads": num_threads,
                    }
                )
    if "npu" in modes:
        npu_threads = args.npu_num_threads or detect_physical_core_count()
        for seq_len in seq_lens:
            cases.append(
                {
                    "case_id": f"npu_seq{seq_len}",
                    "mode": "npu",
                    "seq_len": seq_len,
                    "num_threads": npu_threads,
                }
            )
    return cases


def load_state(path):
    state_path = Path(path)
    if not state_path.exists():
        return {"completed_case_ids": []}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path, state):
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def write_suite_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUITE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_suite_rows(path):
    output_path = Path(path)
    if not output_path.exists():
        return []
    with open(output_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def parse_child_row(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one benchmark row in {csv_path}, found {len(rows)}"
        )
    return rows[0]


def parse_turbostat_log(log_path):
    if not Path(log_path).exists():
        raise RuntimeError(f"Missing turbostat log: {log_path}")

    header = None
    samples = []
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "PkgWatt" in line.split():
                header = line.split()
                continue
            if header is None:
                continue
            parts = line.split()
            if len(parts) != len(header):
                continue
            sample = {}
            for name, value in zip(header, parts):
                try:
                    sample[name] = float(value)
                except ValueError:
                    sample[name] = None
            samples.append(sample)

    stats = {
        "power_sample_count": len(samples),
        "avg_pkg_watt": "",
        "max_pkg_watt": "",
        "avg_cor_watt": "",
        "max_cor_watt": "",
        "avg_gfx_watt": "",
        "max_gfx_watt": "",
        "avg_ram_watt": "",
        "max_ram_watt": "",
    }
    if not samples:
        return stats

    def summarize(column, avg_key, max_key):
        values = [
            sample[column] for sample in samples if sample.get(column) is not None
        ]
        if not values:
            return
        stats[avg_key] = f"{sum(values) / len(values):.6f}"
        stats[max_key] = f"{max(values):.6f}"

    summarize("PkgWatt", "avg_pkg_watt", "max_pkg_watt")
    summarize("CorWatt", "avg_cor_watt", "max_cor_watt")
    summarize("GFXWatt", "avg_gfx_watt", "max_gfx_watt")
    summarize("RAMWatt", "avg_ram_watt", "max_ram_watt")
    return stats


def empty_power_stats(sample_count_key="power_sample_count"):
    return {
        sample_count_key: 0,
        "avg_pkg_watt": "",
        "max_pkg_watt": "",
        "avg_cor_watt": "",
        "max_cor_watt": "",
        "avg_gfx_watt": "",
        "max_gfx_watt": "",
        "avg_ram_watt": "",
        "max_ram_watt": "",
    }


def wrap_command_with_xrt_setup(command):
    xrt_root = Path(os.environ.get("XILINX_XRT", "/opt/xilinx/xrt"))
    xrt_setup = xrt_root / "setup.sh"
    quoted_command = " ".join(shlex.quote(part) for part in command)
    shell_parts = []
    if xrt_setup.exists():
        shell_parts.append(f". {shlex.quote(str(xrt_setup))} >/dev/null 2>&1")
    shell_parts.append(f"exec {quoted_command}")
    return ["/bin/bash", "-lc", " && ".join(shell_parts)]


def measure_idle_power(args, case, logs_dir):
    idle_log_path = logs_dir / f"{case['case_id']}_idle_power.log"
    wrapped = [
        "sudo",
        "-n",
        "turbostat",
        "--Summary",
        "--quiet",
        "--show",
        "PkgWatt,CorWatt,GFXWatt,RAMWatt",
        "--interval",
        str(args.power_interval_sec),
        "--out",
        str(idle_log_path),
        "sleep",
        str(args.npu_idle_baseline_sec),
    ]
    result = subprocess.run(
        wrapped,
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Idle power measurement failed for {case['case_id']}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    idle_stats = parse_turbostat_log(idle_log_path)
    idle_stats["idle_power_sample_count"] = idle_stats.pop("power_sample_count")
    return idle_stats, idle_log_path


def subtract_idle(avg_or_max_value, idle_value):
    if avg_or_max_value == "" or idle_value == "":
        return ""
    value = float(avg_or_max_value) - float(idle_value)
    return f"{max(0.0, value):.6f}"


def case_command(args, case, benchmark_csv_path):
    base = [
        sys.executable,
        str(
            SCRIPT_DIR
            / ("cpu_inference.py" if case["mode"] == "cpu" else "npu_inference.py")
        ),
        "--seq-lens",
        str(case["seq_len"]),
        "--num-samples",
        str(args.num_samples),
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs-per-sample",
        str(args.runs_per_sample),
        "--output-csv",
        str(benchmark_csv_path),
        "--num-threads",
        str(case["num_threads"]),
    ]
    if args.study_id is not None:
        base.extend(["--study-id", args.study_id])
        if args.study_manifest is not None:
            base.extend(["--study-manifest", str(Path(args.study_manifest).resolve())])
        if args.models_root is not None:
            base.extend(["--models-root", str(Path(args.models_root).resolve())])
    else:
        if args.weights_file_path is None or args.config_file_path is None:
            raise ValueError(
                "Either pass weights_file_path and config_file_path, or use --study-id"
            )
        base[2:2] = [
            str(Path(args.weights_file_path).resolve()),
            str(Path(args.config_file_path).resolve()),
        ]
    if case["mode"] == "npu":
        base.extend(
            [
                "--topology-policy",
                args.npu_topology_policy,
                "--topology-cache",
                str(Path(args.npu_topology_cache).resolve()),
                "--autotune-warmup-runs",
                str(args.npu_autotune_warmup_runs),
                "--autotune-runs",
                str(args.npu_autotune_runs),
            ]
        )
        if args.npu_candidate_topologies:
            base.extend(["--candidate-topologies", args.npu_candidate_topologies])
    return base


def run_case(args, case, logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    benchmark_csv_path = logs_dir / f"{case['case_id']}.csv"
    power_log_path = logs_dir / f"{case['case_id']}_power.log"
    idle_log_path = None
    command = case_command(args, case, benchmark_csv_path)
    print(
        f"Running case {case['case_id']}: {' '.join(shlex.quote(part) for part in command)}",
        flush=True,
    )

    idle_stats = empty_power_stats(sample_count_key="idle_power_sample_count")
    if args.power_backend == "turbostat" and case["mode"] == "npu":
        idle_stats, idle_log_path = measure_idle_power(args, case, logs_dir)

    if args.power_backend == "turbostat":
        wrapped_command = wrap_command_with_xrt_setup(command)
        wrapped = [
            "sudo",
            "-n",
            "turbostat",
            "--Summary",
            "--quiet",
            "--show",
            "PkgWatt,CorWatt,GFXWatt,RAMWatt",
            "--interval",
            str(args.power_interval_sec),
            "--out",
            str(power_log_path),
            *wrapped_command,
        ]
        result = subprocess.run(
            wrapped,
            cwd=SCRIPT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Power-logged benchmark failed for {case['case_id']}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        power_stats = parse_turbostat_log(power_log_path)
    else:
        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed for {case['case_id']}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        power_stats = (
            parse_turbostat_log(power_log_path)
            if power_log_path.exists()
            else empty_power_stats()
        )

    child_row = parse_child_row(benchmark_csv_path)
    pseudo_npu_avg_pkg_watt = ""
    pseudo_npu_max_pkg_watt = ""
    if case["mode"] == "npu":
        pseudo_npu_avg_pkg_watt = subtract_idle(
            power_stats["avg_pkg_watt"], idle_stats["avg_pkg_watt"]
        )
        pseudo_npu_max_pkg_watt = subtract_idle(
            power_stats["max_pkg_watt"], idle_stats["avg_pkg_watt"]
        )
    suite_row = {
        "case_id": case["case_id"],
        "mode": case["mode"],
        "seq_len": child_row["seq_len"],
        "num_threads": child_row["num_threads"],
        "dtype": child_row["dtype"],
        "num_samples": child_row["num_samples"],
        "runs_per_sample": child_row["runs_per_sample"],
        "warmup_runs": child_row["warmup_runs"],
        "model_type": child_row.get("model_type", ""),
        "shape": child_row["shape"],
        "topology_id": child_row.get("topology_id", ""),
        "parallel_seq": child_row.get("parallel_seq", ""),
        "parallel_heads": child_row.get("parallel_heads", ""),
        "parallel_ffn": child_row.get("parallel_ffn", ""),
        "min_latency_ms": child_row["min_latency_ms"],
        "avg_latency_ms": child_row["avg_latency_ms"],
        "max_latency_ms": child_row["max_latency_ms"],
        "power_backend": args.power_backend,
        "power_sample_count": power_stats["power_sample_count"],
        "idle_power_sample_count": idle_stats["idle_power_sample_count"],
        "idle_pkg_watt": idle_stats["avg_pkg_watt"],
        "idle_cor_watt": idle_stats["avg_cor_watt"],
        "idle_gfx_watt": idle_stats["avg_gfx_watt"],
        "idle_ram_watt": idle_stats["avg_ram_watt"],
        "avg_pkg_watt": power_stats["avg_pkg_watt"],
        "max_pkg_watt": power_stats["max_pkg_watt"],
        "avg_cor_watt": power_stats["avg_cor_watt"],
        "max_cor_watt": power_stats["max_cor_watt"],
        "avg_gfx_watt": power_stats["avg_gfx_watt"],
        "max_gfx_watt": power_stats["max_gfx_watt"],
        "avg_ram_watt": power_stats["avg_ram_watt"],
        "max_ram_watt": power_stats["max_ram_watt"],
        "pseudo_npu_avg_pkg_watt": pseudo_npu_avg_pkg_watt,
        "pseudo_npu_max_pkg_watt": pseudo_npu_max_pkg_watt,
        "idle_power_log": str(idle_log_path) if idle_log_path is not None else "",
        "power_log": str(power_log_path) if power_log_path.exists() else "",
        "benchmark_csv": str(benchmark_csv_path),
    }
    return suite_row


def main():
    args = parse_args()
    cases = enumerate_cases(args)
    state = load_state(args.state_json)
    completed = set(state.get("completed_case_ids", []))
    suite_rows = load_suite_rows(args.output_csv)
    logs_dir = (SCRIPT_DIR / args.logs_dir).resolve()

    for case in cases:
        if case["case_id"] in completed:
            continue

        suite_row = run_case(args, case, logs_dir)
        suite_rows.append(suite_row)
        write_suite_csv(args.output_csv, suite_rows)
        completed.add(case["case_id"])
        state["completed_case_ids"] = sorted(completed)
        save_state(args.state_json, state)
        print(
            f"Completed case {case['case_id']}: avg_latency_ms={suite_row['avg_latency_ms']} "
            f"topology={suite_row['topology_id'] or 'cpu'}",
            flush=True,
        )

        remaining = [
            pending for pending in cases if pending["case_id"] not in completed
        ]
        if remaining and args.power_cycle_cmd:
            print(
                f"Invoking power cycle hook before next case: {args.power_cycle_cmd}",
                flush=True,
            )
            power_cycle = subprocess.run(
                args.power_cycle_cmd,
                cwd=SCRIPT_DIR,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            if power_cycle.returncode != 0:
                raise RuntimeError(
                    f"Power cycle hook failed\nSTDOUT:\n{power_cycle.stdout}\nSTDERR:\n{power_cycle.stderr}"
                )
            print(
                "Power cycle hook completed. Re-run the same automated_benchmark command after boot to resume.",
                flush=True,
            )
            return

    print(
        f"All benchmark cases completed. Results written to {args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
