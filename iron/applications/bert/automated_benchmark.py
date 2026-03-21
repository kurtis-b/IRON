#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from benchmark_common import (
    DEFAULT_BENCHMARK_SEQ_LENS,
    add_cooldown_args,
    cooldown_before_benchmark,
    detect_logical_thread_count,
    detect_max_physical_core_count,
    detect_physical_core_count,
    parse_seq_lens,
)
from benchmark_power import (
    empty_power_stats,
    format_power_stats_for_csv,
    measure_power_for_duration,
    power_stats_from_row,
    resolve_power_backend,
    effective_device_watt,
)
from model_support import load_study_manifest, resolve_study_paths

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RUNS_PER_SAMPLE_OVERRIDES = "2048=50,4096=15,8192=5"
DEFAULT_NPU_AUTOTUNE_RUNS_OVERRIDES = "2048=3,4096=2,8192=1"
TRANSIENT_NPU_STARTUP_RETRY_DELAYS_SEC = (15.0, 30.0, 60.0, 120.0)
SUITE_FIELDNAMES = [
    "case_id",
    "study_id",
    "mode",
    "seq_len",
    "num_threads",
    "dtype",
    "num_samples",
    "runs_per_sample",
    "warmup_runs",
    "measured_inference_count",
    "timed_total_sec",
    "throughput_inferences_per_sec",
    "model_type",
    "shape",
    "estimated_flops_per_inference",
    "throughput_flops_per_sec",
    "topology_id",
    "parallel_seq",
    "parallel_heads",
    "parallel_ffn",
    "min_latency_ms",
    "avg_latency_ms",
    "max_latency_ms",
    "cooldown_wait_sec",
    "cooldown_start_temp_c",
    "cooldown_end_temp_c",
    "cooldown_temp_source",
    "power_backend",
    "power_sample_count",
    "power_window_sec",
    "idle_power_sample_count",
    "idle_power_window_sec",
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
    "pseudo_device_avg_pkg_watt",
    "pseudo_device_max_pkg_watt",
    "pseudo_npu_avg_pkg_watt",
    "pseudo_npu_max_pkg_watt",
    "estimated_total_timed_flops",
    "estimated_gflops_per_watt_sec",
    "pseudo_device_estimated_gflops_per_watt_sec",
    "pseudo_npu_estimated_gflops_per_watt_sec",
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
        "--study-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated study ids to benchmark, or 'all' for all manifest entries. "
            "Cannot be combined with --study-id or explicit weights/config paths."
        ),
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
        help="Comma-separated benchmark modes to run. Supported: cpu,npu,igpu",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=DEFAULT_BENCHMARK_SEQ_LENS,
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--runs-per-sample", type=int, default=100)
    parser.add_argument(
        "--runs-per-sample-overrides",
        type=str,
        default=DEFAULT_RUNS_PER_SAMPLE_OVERRIDES,
        help=(
            "Optional comma-separated seq_len=runs overrides applied on top of "
            "--runs-per-sample. Use an empty string to disable overrides."
        ),
    )
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
        help=(
            "Host CPU thread count for the NPU benchmark. Defaults to the machine's "
            "maximum physical core count."
        ),
    )
    parser.add_argument(
        "--igpu-num-threads",
        type=int,
        default=None,
        help="Host CPU thread count for the iGPU benchmark. Defaults to physical cores.",
    )
    parser.add_argument(
        "--igpu-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
        help="Execution dtype for the iGPU benchmark mode.",
    )
    parser.add_argument(
        "--igpu-device-index",
        type=int,
        default=0,
        help="Torch CUDA/HIP device index for the iGPU benchmark mode.",
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
        "--npu-autotune-runs-overrides",
        type=str,
        default=DEFAULT_NPU_AUTOTUNE_RUNS_OVERRIDES,
        help=(
            "Optional comma-separated seq_len=runs overrides applied on top of "
            "--npu-autotune-runs. Use an empty string to disable overrides."
        ),
    )
    parser.add_argument(
        "--power-backend",
        choices=("none", "auto"),
        default="auto",
        help=(
            "Per-case power measurement backend. auto uses powercap-rapl for cpu, "
            "turbostat for npu, and rocm-smi for igpu. powercap-rapl and turbostat "
            "may require elevated access."
        ),
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
    add_cooldown_args(parser)
    args = parser.parse_args()
    args.runs_per_sample_overrides = parse_seq_len_int_overrides(
        args.runs_per_sample_overrides
    )
    args.npu_autotune_runs_overrides = parse_seq_len_int_overrides(
        args.npu_autotune_runs_overrides
    )
    return args


def parse_modes(raw_modes):
    modes = []
    for token in raw_modes.split(","):
        token = token.strip()
        if token:
            modes.append(token)
    invalid = sorted(set(modes) - {"cpu", "npu", "igpu"})
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


def parse_seq_len_int_overrides(raw_overrides):
    if raw_overrides is None:
        return {}

    overrides = {}
    for token in raw_overrides.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "runs_per_sample overrides must use seq_len=runs entries "
                f"(got {token!r})"
            )
        seq_len_token, runs_token = token.split("=", 1)
        try:
            seq_len = int(seq_len_token)
            runs_per_sample = int(runs_token)
        except ValueError as exc:
            raise ValueError(
                "runs_per_sample overrides must use integer seq_len=runs entries "
                f"(got {token!r})"
            ) from exc
        if seq_len <= 0 or runs_per_sample <= 0:
            raise ValueError(
                "runs_per_sample overrides must use positive integers "
                f"(got {token!r})"
            )
        overrides[seq_len] = runs_per_sample
    return overrides


def resolve_seq_len_override(default_value, overrides, seq_len):
    return overrides.get(seq_len, default_value)


def resolve_runs_per_sample(args, seq_len):
    overrides = getattr(args, "runs_per_sample_overrides", {})
    return resolve_seq_len_override(args.runs_per_sample, overrides, seq_len)


def resolve_npu_autotune_runs(args, seq_len):
    overrides = getattr(args, "npu_autotune_runs_overrides", {})
    return resolve_seq_len_override(args.npu_autotune_runs, overrides, seq_len)


def topology_suffix(case):
    topology_id = case.get("topology_id")
    return f"_{topology_id}" if topology_id else ""


def parse_requested_study_ids(args):
    if args.study_id is not None and args.study_ids is not None:
        raise ValueError("Use either --study-id or --study-ids, not both")

    if args.study_ids is None:
        return None

    _, entries = load_study_manifest(args.study_manifest)
    if args.study_ids.strip().lower() == "all":
        return [entry["study_id"] for entry in entries]

    available = {entry["study_id"] for entry in entries}
    parsed = []
    for token in args.study_ids.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in available:
            raise ValueError(
                f"Unknown study_id {token!r}. Available study ids: {sorted(available)}"
            )
        if token not in parsed:
            parsed.append(token)
    if not parsed:
        raise ValueError("At least one study id must be provided to --study-ids")
    return parsed


def resolve_model_targets(args):
    requested_study_ids = parse_requested_study_ids(args)
    if requested_study_ids is not None:
        _, entries = load_study_manifest(args.study_manifest)
        entry_by_id = {entry["study_id"]: entry for entry in entries}
        return [
            {
                "study_id": study_id,
                "npu_ready": bool(entry_by_id[study_id].get("npu_ready", False)),
            }
            for study_id in requested_study_ids
        ]

    if args.study_id is not None:
        _, entries = load_study_manifest(args.study_manifest)
        for entry in entries:
            if entry.get("study_id") == args.study_id:
                return [
                    {
                        "study_id": args.study_id,
                        "npu_ready": bool(entry.get("npu_ready", False)),
                    }
                ]
        available = sorted(entry.get("study_id", "") for entry in entries)
        raise ValueError(
            f"Unknown study_id {args.study_id!r}. Available study ids: {available}"
        )

    if args.weights_file_path is None or args.config_file_path is None:
        raise ValueError(
            "Either pass weights_file_path and config_file_path, or use --study-id/--study-ids"
        )
    return [
        {
            "study_id": "",
            "weights_file_path": str(Path(args.weights_file_path).resolve()),
            "config_file_path": str(Path(args.config_file_path).resolve()),
            "npu_ready": True,
        }
    ]


def resolve_target_config_path(args, target):
    config_file_path = target.get("config_file_path")
    if config_file_path:
        return str(Path(config_file_path).resolve())
    if target.get("study_id"):
        resolved = resolve_study_paths(
            target["study_id"],
            manifest_path=args.study_manifest,
            models_root=args.models_root,
        )
        return resolved["config_file_path"]
    raise ValueError("Target does not provide a resolvable config_file_path")


def npu_case_skip_reason(args, target, seq_len):
    from npu_inference import (
        current_topology_from_config,
        load_encoder_pipeline_config,
        parse_candidate_topology_ids,
        supported_topologies_for_seq_len,
        topology_id,
        topology_signature,
    )

    config_file_path = resolve_target_config_path(args, target)
    config = load_encoder_pipeline_config(config_file_path, seq_len)
    candidate_ids = parse_candidate_topology_ids(
        getattr(args, "npu_candidate_topologies", None)
    )
    try:
        supported = supported_topologies_for_seq_len(
            config,
            seq_len,
            candidate_ids=candidate_ids,
        )
    except RuntimeError as exc:
        return str(exc)

    if getattr(args, "npu_topology_policy", "cache") == "fixed":
        current_topology = current_topology_from_config(config)
        supported_signatures = {topology_signature(topology) for topology in supported}
        if topology_signature(current_topology) not in supported_signatures:
            return (
                f"Fixed topology {topology_id(current_topology)} is not supported "
                f"for seq_len={seq_len}"
            )
    return None


def enumerate_cases(args):
    seq_lens = parse_seq_lens(args.seq_lens)
    modes = parse_modes(args.modes)
    targets = resolve_model_targets(args)
    cases = []
    skipped = []
    cpu_thread_counts = parse_thread_counts(args.cpu_thread_counts)
    npu_threads = args.npu_num_threads or detect_max_physical_core_count()
    igpu_threads = args.igpu_num_threads or detect_physical_core_count()

    for target in targets:
        study_prefix = f"{target['study_id']}_" if target["study_id"] else ""
        if "cpu" in modes:
            for num_threads in cpu_thread_counts:
                for seq_len in seq_lens:
                    cases.append(
                        {
                            "case_id": f"{study_prefix}cpu_seq{seq_len}_{num_threads}t",
                            "study_id": target["study_id"],
                            "mode": "cpu",
                            "seq_len": seq_len,
                            "num_threads": num_threads,
                            **target,
                        }
                    )
        if "igpu" in modes:
            for seq_len in seq_lens:
                cases.append(
                    {
                        "case_id": f"{study_prefix}igpu_seq{seq_len}",
                        "study_id": target["study_id"],
                        "mode": "igpu",
                        "seq_len": seq_len,
                        "num_threads": igpu_threads,
                        **target,
                    }
                )
        if "npu" in modes:
            if not target["npu_ready"]:
                skipped.append(
                    (
                        target["study_id"] or "<explicit-path-model>",
                        "npu",
                        "manifest marks the study as npu_ready=false",
                    )
                )
            else:
                for seq_len in seq_lens:
                    skip_reason = npu_case_skip_reason(args, target, seq_len)
                    if skip_reason is not None:
                        skipped.append(
                            (
                                target["study_id"] or "<explicit-path-model>",
                                "npu",
                                skip_reason,
                            )
                        )
                        continue
                    cases.append(
                        {
                            "case_id": f"{study_prefix}npu_seq{seq_len}",
                            "study_id": target["study_id"],
                            "mode": "npu",
                            "seq_len": seq_len,
                            "num_threads": npu_threads,
                            **target,
                        }
                    )

    if not cases:
        raise ValueError("No benchmark cases were generated for the requested inputs")
    return cases, skipped


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


def needs_idle_baseline(case):
    return case["mode"] in ("npu", "igpu")


def measure_idle_power(args, case, logs_dir, power_backend):
    idle_log_path = logs_dir / f"{case['case_id']}_idle_power.log"
    stats = measure_power_for_duration(
        power_backend,
        args.npu_idle_baseline_sec,
        args.power_interval_sec,
        device_index=args.igpu_device_index,
        log_path=idle_log_path,
    )
    idle_stats = empty_power_stats(
        sample_count_key="idle_power_sample_count",
        window_key="idle_power_window_sec",
    )
    idle_stats["idle_power_sample_count"] = stats["power_sample_count"]
    idle_stats["idle_power_window_sec"] = stats["power_window_sec"]
    idle_stats["avg_pkg_watt"] = stats["avg_pkg_watt"]
    idle_stats["max_pkg_watt"] = stats["max_pkg_watt"]
    idle_stats["avg_cor_watt"] = stats["avg_cor_watt"]
    idle_stats["max_cor_watt"] = stats["max_cor_watt"]
    idle_stats["avg_gfx_watt"] = stats["avg_gfx_watt"]
    idle_stats["max_gfx_watt"] = stats["max_gfx_watt"]
    idle_stats["avg_ram_watt"] = stats["avg_ram_watt"]
    idle_stats["max_ram_watt"] = stats["max_ram_watt"]
    return idle_stats, idle_log_path


def subtract_idle(avg_or_max_value, idle_value):
    if avg_or_max_value == "" or idle_value == "":
        return ""
    value = float(avg_or_max_value) - float(idle_value)
    return f"{max(0.0, value):.6f}"


def multiply_optional(left_value, right_value, fmt=".6e"):
    if left_value == "" or right_value == "":
        return ""
    return format(float(left_value) * float(right_value), fmt)


def divide_optional(numerator, denominator, fmt=".6f", scale=1.0):
    if numerator == "" or denominator == "":
        return ""
    denominator_value = float(denominator)
    if denominator_value <= 0:
        return ""
    return format((float(numerator) / denominator_value) * scale, fmt)


def child_power_log_path(child_row, fallback_path):
    child_path = child_row.get("power_log", "")
    if child_path:
        return child_path
    return str(fallback_path) if fallback_path is not None else ""


def is_transient_npu_startup_failure(case, result):
    if case["mode"] != "npu":
        return False
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if "Resource temporarily unavailable" not in combined_output:
        return False
    return "pyxrt.device(0)" in combined_output or "mmap(" in combined_output


def clear_case_artifacts(benchmark_csv_path, power_log_path, idle_log_path):
    for path in (benchmark_csv_path, power_log_path, idle_log_path):
        if path is None:
            continue
        Path(path).unlink(missing_ok=True)


def build_suite_row(
    child_row, case, cooldown_stats, power_backend, idle_stats, idle_log_path
):
    power_stats = power_stats_from_row(child_row)
    active_avg_watt = effective_device_watt(power_stats, case["mode"], "avg")
    active_max_watt = effective_device_watt(power_stats, case["mode"], "max")
    idle_avg_watt = effective_device_watt(idle_stats, case["mode"], "avg")

    pseudo_device_avg_pkg_watt = ""
    pseudo_device_max_pkg_watt = ""
    pseudo_npu_avg_pkg_watt = ""
    pseudo_npu_max_pkg_watt = ""
    if needs_idle_baseline(case):
        pseudo_device_avg_pkg_watt = subtract_idle(active_avg_watt, idle_avg_watt)
        pseudo_device_max_pkg_watt = subtract_idle(active_max_watt, idle_avg_watt)
    if case["mode"] == "npu":
        pseudo_npu_avg_pkg_watt = pseudo_device_avg_pkg_watt
        pseudo_npu_max_pkg_watt = pseudo_device_max_pkg_watt

    estimated_total_timed_flops = multiply_optional(
        child_row.get("estimated_flops_per_inference", ""),
        child_row.get("measured_inference_count", ""),
    )
    active_energy_joules = multiply_optional(
        active_avg_watt,
        power_stats["power_window_sec"],
    )
    pseudo_device_energy_joules = multiply_optional(
        pseudo_device_avg_pkg_watt,
        power_stats["power_window_sec"],
    )
    estimated_gflops_per_watt_sec = divide_optional(
        estimated_total_timed_flops,
        active_energy_joules,
        scale=1e-9,
    )
    pseudo_device_estimated_gflops_per_watt_sec = divide_optional(
        estimated_total_timed_flops,
        pseudo_device_energy_joules,
        scale=1e-9,
    )
    pseudo_npu_estimated_gflops_per_watt_sec = (
        pseudo_device_estimated_gflops_per_watt_sec if case["mode"] == "npu" else ""
    )

    return {
        "case_id": case["case_id"],
        "study_id": child_row.get("study_id", case.get("study_id", "")),
        "mode": case["mode"],
        "seq_len": child_row["seq_len"],
        "num_threads": child_row["num_threads"],
        "dtype": child_row["dtype"],
        "num_samples": child_row["num_samples"],
        "runs_per_sample": child_row["runs_per_sample"],
        "warmup_runs": child_row["warmup_runs"],
        "measured_inference_count": child_row.get("measured_inference_count", ""),
        "timed_total_sec": child_row.get("timed_total_sec", ""),
        "throughput_inferences_per_sec": child_row.get(
            "throughput_inferences_per_sec", ""
        ),
        "model_type": child_row.get("model_type", ""),
        "shape": child_row["shape"],
        "estimated_flops_per_inference": child_row.get(
            "estimated_flops_per_inference", ""
        ),
        "throughput_flops_per_sec": child_row.get("throughput_flops_per_sec", ""),
        "topology_id": child_row.get("topology_id", ""),
        "parallel_seq": child_row.get("parallel_seq", ""),
        "parallel_heads": child_row.get("parallel_heads", ""),
        "parallel_ffn": child_row.get("parallel_ffn", ""),
        "min_latency_ms": child_row["min_latency_ms"],
        "avg_latency_ms": child_row["avg_latency_ms"],
        "max_latency_ms": child_row["max_latency_ms"],
        "cooldown_wait_sec": f"{cooldown_stats['cooldown_wait_sec']:.6f}",
        "cooldown_start_temp_c": (
            f"{cooldown_stats['cooldown_start_temp_c']:.3f}"
            if cooldown_stats["cooldown_start_temp_c"] is not None
            else ""
        ),
        "cooldown_end_temp_c": (
            f"{cooldown_stats['cooldown_end_temp_c']:.3f}"
            if cooldown_stats["cooldown_end_temp_c"] is not None
            else ""
        ),
        "cooldown_temp_source": cooldown_stats["cooldown_temp_source"],
        **format_power_stats_for_csv(
            child_row.get("power_backend", power_backend),
            power_stats,
            child_power_log_path(child_row, None),
        ),
        "idle_power_sample_count": idle_stats["idle_power_sample_count"],
        "idle_power_window_sec": idle_stats["idle_power_window_sec"],
        "idle_pkg_watt": idle_stats["avg_pkg_watt"],
        "idle_cor_watt": idle_stats["avg_cor_watt"],
        "idle_gfx_watt": idle_stats["avg_gfx_watt"],
        "idle_ram_watt": idle_stats["avg_ram_watt"],
        "pseudo_device_avg_pkg_watt": pseudo_device_avg_pkg_watt,
        "pseudo_device_max_pkg_watt": pseudo_device_max_pkg_watt,
        "pseudo_npu_avg_pkg_watt": pseudo_npu_avg_pkg_watt,
        "pseudo_npu_max_pkg_watt": pseudo_npu_max_pkg_watt,
        "estimated_total_timed_flops": estimated_total_timed_flops,
        "estimated_gflops_per_watt_sec": estimated_gflops_per_watt_sec,
        "pseudo_device_estimated_gflops_per_watt_sec": (
            pseudo_device_estimated_gflops_per_watt_sec
        ),
        "pseudo_npu_estimated_gflops_per_watt_sec": (
            pseudo_npu_estimated_gflops_per_watt_sec
        ),
        "idle_power_log": str(idle_log_path) if idle_log_path is not None else "",
        "benchmark_csv": child_row.get("benchmark_csv", ""),
    }


def case_command(
    args,
    case,
    benchmark_csv_path,
    requested_power_backend,
    power_log_path,
):
    runs_per_sample = resolve_runs_per_sample(args, case["seq_len"])
    npu_autotune_runs = resolve_npu_autotune_runs(args, case["seq_len"])
    script_name = {
        "cpu": "cpu_inference.py",
        "npu": "npu_inference.py",
        "igpu": "igpu_inference.py",
    }[case["mode"]]
    base = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--seq-lens",
        str(case["seq_len"]),
        "--num-samples",
        str(args.num_samples),
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs-per-sample",
        str(runs_per_sample),
        "--output-csv",
        str(benchmark_csv_path),
        "--num-threads",
        str(case["num_threads"]),
    ]
    if case.get("study_id"):
        base.extend(["--study-id", case["study_id"]])
        if args.study_manifest is not None:
            base.extend(["--study-manifest", str(Path(args.study_manifest).resolve())])
        if args.models_root is not None:
            base.extend(["--models-root", str(Path(args.models_root).resolve())])
    else:
        if (
            case.get("weights_file_path") is None
            or case.get("config_file_path") is None
        ):
            raise ValueError(
                "Either pass weights_file_path and config_file_path, or use --study-id"
            )
        base[2:2] = [
            str(Path(case["weights_file_path"]).resolve()),
            str(Path(case["config_file_path"]).resolve()),
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
                str(npu_autotune_runs),
            ]
        )
        if args.npu_candidate_topologies:
            base.extend(["--candidate-topologies", args.npu_candidate_topologies])
    if case["mode"] == "igpu":
        base.extend(
            [
                "--dtype",
                args.igpu_dtype,
                "--device-index",
                str(args.igpu_device_index),
            ]
        )
    base.extend(["--power-backend", requested_power_backend])
    if requested_power_backend != "none":
        base.extend(["--power-interval-sec", str(args.power_interval_sec)])
        if power_log_path is not None:
            base.extend(["--power-log-path", str(power_log_path)])
    return base


def run_case(args, case, logs_dir, cooldown_stats):
    logs_dir.mkdir(parents=True, exist_ok=True)
    benchmark_csv_path = logs_dir / f"{case['case_id']}.csv"
    requested_power_backend = args.power_backend
    power_backend = resolve_power_backend(requested_power_backend, case["mode"])
    power_log_path = (
        logs_dir / f"{case['case_id']}_power.log"
        if requested_power_backend != "none"
        else None
    )
    idle_log_path = (
        logs_dir / f"{case['case_id']}_idle_power.log"
        if needs_idle_baseline(case)
        else None
    )
    command = case_command(
        args,
        case,
        benchmark_csv_path,
        requested_power_backend,
        power_log_path,
    )
    print(
        f"Running case {case['case_id']}: {' '.join(shlex.quote(part) for part in command)}",
        flush=True,
    )

    max_attempts = (
        len(TRANSIENT_NPU_STARTUP_RETRY_DELAYS_SEC) + 1 if case["mode"] == "npu" else 1
    )
    for attempt_index in range(max_attempts):
        clear_case_artifacts(benchmark_csv_path, power_log_path, idle_log_path)
        idle_stats = empty_power_stats(
            sample_count_key="idle_power_sample_count",
            window_key="idle_power_window_sec",
        )
        if power_backend != "none" and needs_idle_baseline(case):
            idle_stats, idle_log_path = measure_idle_power(
                args, case, logs_dir, power_backend
            )

        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            child_row = parse_child_row(benchmark_csv_path)
            child_row["benchmark_csv"] = str(benchmark_csv_path)
            if not child_row.get("power_log"):
                child_row["power_log"] = child_power_log_path(child_row, power_log_path)
            return build_suite_row(
                child_row,
                case,
                cooldown_stats,
                power_backend,
                idle_stats,
                idle_log_path,
            )

        if attempt_index + 1 < max_attempts and is_transient_npu_startup_failure(
            case, result
        ):
            retry_delay_sec = TRANSIENT_NPU_STARTUP_RETRY_DELAYS_SEC[attempt_index]
            print(
                (
                    f"Transient NPU startup failure for {case['case_id']} "
                    f"on attempt {attempt_index + 1}/{max_attempts}; "
                    f"retrying in {retry_delay_sec:.1f} sec"
                ),
                flush=True,
            )
            time.sleep(retry_delay_sec)
            continue

        failure_label = (
            "Power-logged benchmark failed"
            if power_backend != "none"
            else "Benchmark failed"
        )
        raise RuntimeError(
            f"{failure_label} for {case['case_id']}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    raise RuntimeError(
        f"Benchmark failed for {case['case_id']} with no subprocess result"
    )


def main():
    args = parse_args()
    cases, skipped = enumerate_cases(args)
    state = load_state(args.state_json)
    completed = set(state.get("completed_case_ids", []))
    suite_rows = load_suite_rows(args.output_csv)
    logs_dir = (SCRIPT_DIR / args.logs_dir).resolve()

    for study_id, mode, reason in skipped:
        print(
            f"Skipping study_id={study_id} mode={mode}: {reason}",
            flush=True,
        )

    for case in cases:
        if case["case_id"] in completed:
            continue

        cooldown_stats = cooldown_before_benchmark(
            args,
            mode=case["mode"],
            label=case["case_id"],
        )
        suite_row = run_case(args, case, logs_dir, cooldown_stats)
        suite_rows.append(suite_row)
        write_suite_csv(args.output_csv, suite_rows)
        completed.add(case["case_id"])
        state["completed_case_ids"] = sorted(completed)
        save_state(args.state_json, state)
        print(
            f"Completed case {case['case_id']}: study_id={suite_row['study_id'] or '<explicit-path-model>'} "
            f"avg_latency_ms={suite_row['avg_latency_ms']} "
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
