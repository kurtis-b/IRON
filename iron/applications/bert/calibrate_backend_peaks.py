#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from peak_reference import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PEAK_REFERENCE_OUTPUT_PATH,
    build_peak_reference_artifact,
)

DEFAULT_COLLECT_BACKENDS = ("cpu", "igpu", "npu", "memory")
DEFAULT_CPU_DTYPES = ("bfloat16", "float32")
DEFAULT_IGPU_DTYPES = ("bfloat16", "float16", "float32")
DEFAULT_GEMM_SHAPES = ((2048, 2048, 2048), (4096, 4096, 4096))
DEFAULT_STREAM_NUM_ELEMENTS = 32 * 1024 * 1024
APP_DIR = Path(__file__).resolve().parent
DEFAULT_NPU_BF16_STUDY_ID = "bert-base-uncased"
DEFAULT_NPU_BF16_SEQ_LENS = "256,512,1024"
DEFAULT_NPU_TOPOLOGY_CACHE = APP_DIR / "npu_topology_cache_latest.json"
TOPS_VALUE_RE = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*TOPS", re.I)
TOPS_LABEL_VALUE_RE = re.compile(r"TOPS[^0-9]*(?P<value>[0-9]+(?:\.[0-9]+)?)", re.I)
SHAPE_ITEM_RE = re.compile(r"^(?P<m>\d+)x(?P<k>\d+)x(?P<n>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a structured peak-reference artifact from a host profile plus optional "
            "measured calibration inputs."
        )
    )
    parser.add_argument(
        "--host-profile",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the host-profile JSON emitted by peak_reference.py.",
    )
    parser.add_argument(
        "--measured-input",
        type=str,
        default=None,
        help=(
            "Optional JSON file that supplies measured backend peak and memory-bandwidth "
            "calibration results."
        ),
    )
    parser.add_argument(
        "--collect-measured",
        action="store_true",
        help="Collect measured calibration data directly on this machine before building the artifact.",
    )
    parser.add_argument(
        "--collect-backends",
        type=str,
        default=",".join(DEFAULT_COLLECT_BACKENDS),
        help=(
            "Comma-separated subset of cpu,igpu,npu,memory to collect when "
            "--collect-measured is enabled."
        ),
    )
    parser.add_argument(
        "--gemm-shapes",
        type=str,
        default=",".join(f"{m}x{k}x{n}" for m, k, n in DEFAULT_GEMM_SHAPES),
        help=(
            "Comma-separated GEMM shapes MxKxN used for CPU and iGPU calibration. "
            "The best measured throughput across shapes is retained."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations per measured shape.",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=5,
        help="Measured iterations per repetition.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per measured shape; the best result is retained.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="CPU thread count for CPU GEMM and memory-bandwidth calibration.",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="Optional CPU affinity list such as 0-11 or 0-7,16-23.",
    )
    parser.add_argument(
        "--igpu-device-index",
        type=int,
        default=0,
        help="torch.cuda device index for iGPU calibration.",
    )
    parser.add_argument(
        "--stream-num-elements",
        type=int,
        default=DEFAULT_STREAM_NUM_ELEMENTS,
        help=(
            "Element count used for the built-in STREAM-style copy/triad bandwidth calibration."
        ),
    )
    parser.add_argument(
        "--npu-bf16-study-id",
        type=str,
        default=DEFAULT_NPU_BF16_STUDY_ID,
        help=(
            "Study id used for the first-pass IRON-side NPU bf16 calibration benchmark."
        ),
    )
    parser.add_argument(
        "--npu-bf16-study-manifest",
        type=str,
        default=None,
        help="Optional study manifest path for the NPU bf16 calibration benchmark.",
    )
    parser.add_argument(
        "--npu-bf16-models-root",
        type=str,
        default=None,
        help="Optional models root directory for the NPU bf16 calibration benchmark.",
    )
    parser.add_argument(
        "--npu-bf16-seq-lens",
        type=str,
        default=DEFAULT_NPU_BF16_SEQ_LENS,
        help=(
            "Comma-separated sequence lengths used for the first-pass NPU bf16 calibration."
        ),
    )
    parser.add_argument(
        "--npu-bf16-num-samples",
        type=int,
        default=1,
        help="Number of synthetic samples per sequence length for the NPU bf16 calibration run.",
    )
    parser.add_argument(
        "--npu-bf16-warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per sequence length for the NPU bf16 calibration run.",
    )
    parser.add_argument(
        "--npu-bf16-runs",
        type=int,
        default=5,
        help="Timed runs per sequence length for the NPU bf16 calibration run.",
    )
    parser.add_argument(
        "--npu-bf16-num-threads",
        type=int,
        default=None,
        help=(
            "Host CPU thread count used by the NPU bf16 calibration run. Defaults to --cpu-threads "
            "when set."
        ),
    )
    parser.add_argument(
        "--npu-bf16-topology-policy",
        choices=("fixed", "cache", "autotune"),
        default="cache",
        help="Topology policy used for the NPU bf16 calibration benchmark.",
    )
    parser.add_argument(
        "--npu-bf16-topology-cache",
        type=str,
        default=str(DEFAULT_NPU_TOPOLOGY_CACHE),
        help="Topology cache JSON used by the NPU bf16 calibration benchmark.",
    )
    parser.add_argument(
        "--measured-output",
        type=str,
        default=None,
        help="Optional path to also write the merged measured-calibration JSON input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PEAK_REFERENCE_OUTPUT_PATH),
        help="Path to write the combined peak-reference artifact JSON.",
    )
    return parser.parse_args()


def parse_csv_list(raw_value: str):
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_shape_list(raw_value: str):
    shapes = []
    for item in parse_csv_list(raw_value):
        match = SHAPE_ITEM_RE.fullmatch(item)
        if match is None:
            raise ValueError(f"Invalid GEMM shape '{item}'; expected MxKxN")
        shapes.append(
            (int(match.group("m")), int(match.group("k")), int(match.group("n")))
        )
    if not shapes:
        raise ValueError("Expected at least one GEMM shape")
    return tuple(shapes)


def parse_cpu_affinity(raw_value: str | None):
    if raw_value in ("", None):
        return None
    cpus = set()
    for item in parse_csv_list(raw_value):
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid CPU affinity range '{item}'")
            cpus.update(range(start, end + 1))
            continue
        cpus.add(int(item))
    return tuple(sorted(cpus))


def merge_calibration_data(base: dict | None, update: dict | None):
    if base is None:
        base = {}
    if update is None:
        return dict(base)
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_calibration_data(merged[key], value)
        else:
            merged[key] = value
    return merged


def _maybe_set_cpu_affinity(cpu_affinity):
    if cpu_affinity is None or not hasattr(os, "sched_setaffinity"):
        return None
    os.sched_setaffinity(0, cpu_affinity)
    return list(cpu_affinity)


def _torch_dtype_from_name(torch_module, dtype_name: str):
    dtype_map = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'")
    return dtype_map[dtype_name]


def _configure_cpu_torch(torch_module, cpu_threads: int | None, cpu_affinity):
    affinity_applied = _maybe_set_cpu_affinity(cpu_affinity)
    if cpu_threads is not None:
        torch_module.set_num_threads(cpu_threads)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return affinity_applied


def _measure_torch_gemm_peak(
    *,
    torch_module,
    device,
    dtype_name: str,
    shapes,
    warmup_iters: int,
    measure_iters: int,
    repetitions: int,
):
    dtype = _torch_dtype_from_name(torch_module, dtype_name)
    synchronize = (
        torch_module.cuda.synchronize if device.type == "cuda" else lambda: None
    )
    best_result = None

    with torch_module.no_grad():
        for shape in shapes:
            m, k, n = shape
            lhs = torch_module.randn((m, k), dtype=dtype, device=device)
            rhs = torch_module.randn((k, n), dtype=dtype, device=device)
            out = torch_module.empty((m, n), dtype=dtype, device=device)

            for _ in range(warmup_iters):
                torch_module.mm(lhs, rhs, out=out)
            synchronize()

            for _ in range(repetitions):
                synchronize()
                start = time.perf_counter()
                for _ in range(measure_iters):
                    torch_module.mm(lhs, rhs, out=out)
                synchronize()
                elapsed_sec = time.perf_counter() - start
                if elapsed_sec <= 0.0:
                    continue
                ops_per_sec = (2.0 * m * k * n * measure_iters) / elapsed_sec
                candidate = {
                    "measured_peak_ops_per_sec": ops_per_sec,
                    "shape": {"m": m, "k": k, "n": n},
                    "elapsed_sec": elapsed_sec,
                }
                if (
                    best_result is None
                    or candidate["measured_peak_ops_per_sec"]
                    > best_result["measured_peak_ops_per_sec"]
                ):
                    best_result = candidate
    if best_result is None:
        raise RuntimeError("No valid GEMM measurements were collected")
    return best_result


def collect_torch_gemm_peaks(
    *,
    host_profile: dict,
    backends,
    shapes,
    warmup_iters: int,
    measure_iters: int,
    repetitions: int,
    cpu_threads: int | None,
    cpu_affinity,
    igpu_device_index: int,
):
    import torch

    measured = {"tool_versions": {"torch": torch.__version__}, "peaks": {}}
    cpu_affinity_applied = _configure_cpu_torch(torch, cpu_threads, cpu_affinity)

    if "cpu" in backends:
        cpu_threads_effective = cpu_threads or host_profile.get("cpu", {}).get(
            "threads"
        )
        for dtype_name in DEFAULT_CPU_DTYPES:
            try:
                result = _measure_torch_gemm_peak(
                    torch_module=torch,
                    device=torch.device("cpu"),
                    dtype_name=dtype_name,
                    shapes=shapes,
                    warmup_iters=warmup_iters,
                    measure_iters=measure_iters,
                    repetitions=repetitions,
                )
                shape = result["shape"]
                note = (
                    f"torch.mm cpu gemm peak; shape={shape['m']}x{shape['k']}x{shape['n']} "
                    f"threads={cpu_threads_effective}"
                )
                if cpu_affinity_applied is not None:
                    note += f" affinity={cpu_affinity_applied}"
                measured.setdefault("peaks", {}).setdefault("cpu", {})[dtype_name] = {
                    "measured_peak_ops_per_sec": result["measured_peak_ops_per_sec"],
                    "note": note,
                }
            except Exception as exc:
                measured.setdefault("peaks", {}).setdefault("cpu", {})[dtype_name] = {
                    "note": f"CPU torch GEMM calibration failed: {exc}",
                }

    if "igpu" in backends:
        if not torch.cuda.is_available():
            for dtype_name in DEFAULT_IGPU_DTYPES:
                measured.setdefault("peaks", {}).setdefault("igpu", {})[dtype_name] = {
                    "note": "iGPU calibration skipped: torch.cuda is not available",
                }
        else:
            device = torch.device(f"cuda:{igpu_device_index}")
            device_name = torch.cuda.get_device_name(igpu_device_index)
            for dtype_name in DEFAULT_IGPU_DTYPES:
                try:
                    result = _measure_torch_gemm_peak(
                        torch_module=torch,
                        device=device,
                        dtype_name=dtype_name,
                        shapes=shapes,
                        warmup_iters=warmup_iters,
                        measure_iters=measure_iters,
                        repetitions=repetitions,
                    )
                    shape = result["shape"]
                    measured.setdefault("peaks", {}).setdefault("igpu", {})[
                        dtype_name
                    ] = {
                        "measured_peak_ops_per_sec": result[
                            "measured_peak_ops_per_sec"
                        ],
                        "note": (
                            f"torch.mm rocM gemm peak; device={device_name} "
                            f"shape={shape['m']}x{shape['k']}x{shape['n']}"
                        ),
                    }
                except Exception as exc:
                    measured.setdefault("peaks", {}).setdefault("igpu", {})[
                        dtype_name
                    ] = {
                        "note": f"iGPU torch GEMM calibration failed: {exc}",
                    }

    return measured


def collect_stream_like_memory_bandwidth(
    *,
    cpu_threads: int | None,
    cpu_affinity,
    num_elements: int,
    warmup_iters: int,
    measure_iters: int,
    repetitions: int,
):
    import torch

    _configure_cpu_torch(torch, cpu_threads, cpu_affinity)
    measured = {"tool_versions": {"torch": torch.__version__}, "memory_bandwidth": {}}
    dtype = torch.float32
    scalar = 3.0
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    dst = torch.empty((num_elements,), dtype=dtype)
    src = torch.randn((num_elements,), dtype=dtype)
    src_b = torch.randn((num_elements,), dtype=dtype)
    src_c = torch.randn((num_elements,), dtype=dtype)

    best_copy = None
    best_triad = None

    with torch.no_grad():
        for _ in range(warmup_iters):
            dst.copy_(src)
            torch.add(src_b, src_c, alpha=scalar, out=dst)

        for _ in range(repetitions):
            start = time.perf_counter()
            for _ in range(measure_iters):
                dst.copy_(src)
            copy_elapsed_sec = time.perf_counter() - start
            if copy_elapsed_sec > 0.0:
                copy_bytes = 2.0 * num_elements * bytes_per_element * measure_iters
                copy_bytes_per_sec = copy_bytes / copy_elapsed_sec
                best_copy = max(best_copy or 0.0, copy_bytes_per_sec)

            start = time.perf_counter()
            for _ in range(measure_iters):
                torch.add(src_b, src_c, alpha=scalar, out=dst)
            triad_elapsed_sec = time.perf_counter() - start
            if triad_elapsed_sec > 0.0:
                triad_bytes = 3.0 * num_elements * bytes_per_element * measure_iters
                triad_bytes_per_sec = triad_bytes / triad_elapsed_sec
                best_triad = max(best_triad or 0.0, triad_bytes_per_sec)

    if best_copy is not None:
        measured["memory_bandwidth"]["copy_bytes_per_sec"] = best_copy
    if best_triad is not None:
        measured["memory_bandwidth"]["triad_bytes_per_sec"] = best_triad
        measured["memory_bandwidth"]["measured_peak_bytes_per_sec"] = best_triad
    elif best_copy is not None:
        measured["memory_bandwidth"]["measured_peak_bytes_per_sec"] = best_copy
    measured["memory_bandwidth"][
        "note"
    ] = "Built-in torch STREAM-style copy/triad bandwidth benchmark on CPU memory."
    return measured


def _extract_tops_from_node(node):
    best_tops = None
    if isinstance(node, dict):
        for key, value in node.items():
            candidate = None
            if "tops" in str(key).lower():
                candidate = _extract_tops_from_node(value)
            else:
                candidate = _extract_tops_from_node(value)
            if candidate is not None:
                best_tops = max(best_tops or candidate, candidate)
        return best_tops
    if isinstance(node, list):
        for item in node:
            candidate = _extract_tops_from_node(item)
            if candidate is not None:
                best_tops = max(best_tops or candidate, candidate)
        return best_tops
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        match = TOPS_VALUE_RE.search(node)
        if match is not None:
            return float(match.group("value"))
        match = TOPS_LABEL_VALUE_RE.search(node)
        if match is not None:
            return float(match.group("value"))
        try:
            return float(node)
        except ValueError:
            return None
    return None


def _summarize_log_messages(node):
    messages = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                messages.extend(_summarize_log_messages(value))
            elif value not in ("", None):
                messages.append(f"{key}: {value}")
    elif isinstance(node, list):
        for item in node:
            messages.extend(_summarize_log_messages(item))
    return messages


def parse_xrt_smi_validate_gemm_json(text: str):
    payload = json.loads(text)
    devices = payload.get("logical_devices", [])
    gemm_test = None
    for device in devices:
        for test in device.get("tests", []):
            if test.get("name") == "gemm":
                gemm_test = test
                break
        if gemm_test is not None:
            break

    if gemm_test is None:
        return {
            "note": "xrt-smi validate did not include a gemm test entry",
        }

    tops_value = _extract_tops_from_node(gemm_test)
    messages = _summarize_log_messages(gemm_test.get("log", []))
    status = gemm_test.get("status")
    note_parts = ["xrt-smi validate --run gemm INT8 reference"]
    if status:
        note_parts.append(f"status={status}")
    if messages:
        note_parts.append("; ".join(messages[:4]))

    result = {"note": " ".join(note_parts)}
    if tops_value is not None and status == "PASSED":
        result["measured_peak_ops_per_sec"] = tops_value * 1_000_000_000_000.0
    return result


def parse_npu_benchmark_csv_text(text: str):
    return list(csv.DictReader(text.splitlines()))


def select_best_npu_bf16_peak_row(rows):
    best_row = None
    for row in rows:
        throughput_raw = row.get("throughput_flops_per_sec")
        if throughput_raw in ("", None):
            continue
        throughput = float(throughput_raw)
        candidate = dict(row)
        candidate["throughput_flops_per_sec"] = throughput
        if best_row is None or throughput > best_row["throughput_flops_per_sec"]:
            best_row = candidate
    return best_row


def collect_npu_int8_reference(host_profile: dict):
    command = [
        "xrt-smi",
        "validate",
        "--batch",
        "--force",
        "--run",
        "gemm",
        "-f",
        "JSON",
    ]
    device_bdf = host_profile.get("npu", {}).get("device_bdf")
    if device_bdf:
        command.extend(["--device", device_bdf])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "xrt_validate_gemm.json"
        command.extend(["-o", str(output_path)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if output_path.exists():
            parsed = parse_xrt_smi_validate_gemm_json(
                output_path.read_text(encoding="utf-8")
            )
            if result.returncode != 0 and "measured_peak_ops_per_sec" not in parsed:
                stderr = result.stderr.strip() or result.stdout.strip()
                if stderr:
                    parsed["note"] = f"{parsed['note']} command={stderr}".strip()
            return {
                "tool_versions": {
                    "xrt-smi-validate": host_profile.get("npu", {}).get("xrt_version")
                },
                "peaks": {"npu": {"int8": parsed}},
            }
        message = (
            result.stderr.strip() or result.stdout.strip() or "xrt-smi validate failed"
        )
        return {
            "tool_versions": {
                "xrt-smi-validate": host_profile.get("npu", {}).get("xrt_version")
            },
            "peaks": {
                "npu": {
                    "int8": {
                        "note": f"xrt-smi validate --run gemm failed before emitting JSON: {message}",
                    }
                }
            },
        }


def collect_npu_bf16_reference(args):
    script_path = APP_DIR / "npu_inference.py"
    npu_num_threads = args.npu_bf16_num_threads or args.cpu_threads
    command = [
        sys.executable,
        str(script_path),
        "--study-id",
        args.npu_bf16_study_id,
        "--seq-lens",
        args.npu_bf16_seq_lens,
        "--num-samples",
        str(args.npu_bf16_num_samples),
        "--warmup-runs",
        str(args.npu_bf16_warmup_runs),
        "--runs-per-sample",
        str(args.npu_bf16_runs),
        "--topology-policy",
        args.npu_bf16_topology_policy,
        "--topology-cache",
        str(Path(args.npu_bf16_topology_cache).resolve()),
        "--power-backend",
        "none",
    ]
    if npu_num_threads is not None:
        command.extend(["--num-threads", str(npu_num_threads)])
    if args.npu_bf16_study_manifest is not None:
        command.extend(
            ["--study-manifest", str(Path(args.npu_bf16_study_manifest).resolve())]
        )
    if args.npu_bf16_models_root is not None:
        command.extend(
            ["--models-root", str(Path(args.npu_bf16_models_root).resolve())]
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_csv = Path(tmp_dir) / "npu_bf16_calibration.csv"
        command.extend(["--output-csv", str(output_csv)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "npu_inference.py failed"
            )
            return {
                "peaks": {
                    "npu": {
                        "bfloat16": {
                            "note": f"NPU bf16 calibration benchmark failed: {message}",
                        }
                    }
                }
            }
        if not output_csv.exists():
            return {
                "peaks": {
                    "npu": {
                        "bfloat16": {
                            "note": "NPU bf16 calibration benchmark produced no output CSV",
                        }
                    }
                }
            }

        rows = parse_npu_benchmark_csv_text(output_csv.read_text(encoding="utf-8"))
        best_row = select_best_npu_bf16_peak_row(rows)
        if best_row is None:
            return {
                "peaks": {
                    "npu": {
                        "bfloat16": {
                            "note": "NPU bf16 calibration benchmark produced no throughput rows",
                        }
                    }
                }
            }

        return {
            "peaks": {
                "npu": {
                    "bfloat16": {
                        "measured_peak_ops_per_sec": best_row[
                            "throughput_flops_per_sec"
                        ],
                        "note": (
                            "IRON encoder_pipeline synthetic_dense bf16 calibration via "
                            f"npu_inference.py; study={best_row.get('study_id') or args.npu_bf16_study_id} "
                            f"seq_len={best_row.get('seq_len')} "
                            f"topology={best_row.get('topology_id')} "
                            f"avg_latency_ms={best_row.get('avg_latency_ms')}"
                        ),
                    }
                }
            }
        }


def collect_measured_calibration(host_profile: dict, args):
    backends = set(parse_csv_list(args.collect_backends))
    invalid_backends = sorted(backends - set(DEFAULT_COLLECT_BACKENDS))
    if invalid_backends:
        raise ValueError(
            f"Unsupported backends in --collect-backends: {', '.join(invalid_backends)}"
        )

    shapes = parse_shape_list(args.gemm_shapes)
    cpu_affinity = parse_cpu_affinity(args.cpu_affinity)
    measured = {}

    if backends & {"cpu", "igpu"}:
        measured = merge_calibration_data(
            measured,
            collect_torch_gemm_peaks(
                host_profile=host_profile,
                backends=backends,
                shapes=shapes,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                repetitions=args.repetitions,
                cpu_threads=args.cpu_threads,
                cpu_affinity=cpu_affinity,
                igpu_device_index=args.igpu_device_index,
            ),
        )
    if "memory" in backends:
        measured = merge_calibration_data(
            measured,
            collect_stream_like_memory_bandwidth(
                cpu_threads=args.cpu_threads,
                cpu_affinity=cpu_affinity,
                num_elements=args.stream_num_elements,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                repetitions=args.repetitions,
            ),
        )
    if "npu" in backends:
        measured = merge_calibration_data(
            measured, collect_npu_int8_reference(host_profile)
        )
        measured = merge_calibration_data(measured, collect_npu_bf16_reference(args))
    return measured


def main():
    args = parse_args()
    host_profile = json.loads(Path(args.host_profile).read_text(encoding="utf-8"))
    measured_input = (
        json.loads(Path(args.measured_input).read_text(encoding="utf-8"))
        if args.measured_input is not None
        else {}
    )
    if args.collect_measured:
        measured_input = merge_calibration_data(
            measured_input,
            collect_measured_calibration(host_profile, args),
        )

    if args.measured_output is not None:
        measured_output_path = Path(args.measured_output)
        measured_output_path.write_text(
            json.dumps(measured_input, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote measured calibration data to {measured_output_path}")

    artifact = build_peak_reference_artifact(host_profile, measured_input)
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote peak reference artifact to {output_path}")


if __name__ == "__main__":
    main()
