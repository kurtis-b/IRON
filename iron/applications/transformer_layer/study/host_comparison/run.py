#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import threading
import time

import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F

from ..run_lock import default_lock_path, hold_study_lock
from iron.applications.transformer_layer.study.end_to_end.cases import (
    effective_gflops_per_sec,
    effective_gflops_per_sec_per_watt,
)
from iron.applications.transformer_layer.pattern.reference import (
    generate_golden_reference,
)
from iron.applications.transformer_layer.study.end_to_end.cases import (
    FAMILY_SPECS,
)
from iron.applications.transformer_layer.study.end_to_end.power import (
    PERSISTED_POWER_RESULT_FIELDS,
    create_power_monitor as create_cpu_power_monitor,
    power_probe_is_complete,
    summarize_power_samples,
)
from iron.applications.transformer_layer.study.plot_families import (
    PLOT_FAMILY_GRID_SHAPE,
    PLOT_FAMILY_ORDER,
    ordered_plot_families,
    plot_family_label,
)

from .select import (
    REFERENCE_EXECUTION_MODES,
    ReferenceGroup,
    default_reference_results_path,
    group_reference_rows,
    load_reference_rows,
)

LOGGER = logging.getLogger(__name__)

REFERENCE_VALIDATION_MAX_SEQ_LEN = 512
FINAL_REL_TOL = 0.1
FINAL_ABS_TOL = 0.5
FINAL_ERROR_THRESHOLD = 0.05
SUPPORTED_HOST_BACKENDS: tuple[str, ...] = ("igpu",)
SUPPORTED_CPU_POWER_BACKENDS: tuple[str, ...] = ("none", "turbostat_pkgwatt")
SUPPORTED_IGPU_POWER_BACKENDS: tuple[str, ...] = ("none", "rocm-smi")
COMPARISON_COLUMNS: tuple[str, ...] = (
    "igpu",
    "igpu_rocm_smi",
    *REFERENCE_EXECUTION_MODES,
)
RESULTS_CSV_FIELDNAMES = (
    "workload_variant",
    "study_case_id",
    "seq_len",
    "metric",
    *COMPARISON_COLUMNS,
)
DIRECT_COMPARISON_METRICS: tuple[str, ...] = (
    "effective_gflops_per_sec",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "latency_sample_count",
)
POWER_AUDIT_METRICS: tuple[str, ...] = tuple(
    field
    for field in PERSISTED_POWER_RESULT_FIELDS
    if field
    not in {
        "avg_power_w",
        "min_power_w",
        "max_power_w",
        "power_sample_count",
        "power_outlier_filter_applied",
    }
)
POWER_COMPARISON_METRICS: tuple[str, ...] = (
    "effective_gflops_per_sec_per_watt",
    "avg_power_w",
    "min_power_w",
    "max_power_w",
    "power_sample_count",
    *POWER_AUDIT_METRICS,
)
PLOT_SERIES_THROUGHPUT = (
    ("igpu", "iGPU", "#e07a5f"),
    ("hybrid", "NPU Hybrid", "#1f6f8b"),
)
PLOT_SERIES_PER_WATT = (
    ("igpu_rocm_smi", "iGPU", "#e07a5f"),
    ("hybrid", "NPU Hybrid", "#1f6f8b"),
)
PLOT_SEQ_ORDER = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SUPPORTED_PLOT_SUFFIX = ".svg"
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
ROCM_SMI_MIN_POWER_SAMPLE_INTERVAL_SEC = 0.05
ROCM_SMI_MIN_POWER_SAMPLE_COUNT = 16
# rocm-smi collects one sample per subprocess invocation, so target more than
# the minimum desired count when picking the poll interval.
ROCM_SMI_POWER_SAMPLE_TARGET_COUNT = 24
ROCM_SMI_MIN_POWER_MEASUREMENT_DURATION_SEC = max(
    2.0,
    ROCM_SMI_MIN_POWER_SAMPLE_INTERVAL_SEC * ROCM_SMI_POWER_SAMPLE_TARGET_COUNT,
)
ROCM_SMI_LONG_MIN_POWER_SAMPLE_COUNT = 20
ROCM_SMI_LONG_POWER_SAMPLE_TARGET_COUNT = 30
ROCM_SMI_LONG_MIN_POWER_MEASUREMENT_DURATION_SEC = max(
    3.0,
    ROCM_SMI_MIN_POWER_SAMPLE_INTERVAL_SEC * ROCM_SMI_LONG_POWER_SAMPLE_TARGET_COUNT,
)
TURBOSTAT_MIN_POWER_SAMPLE_INTERVAL_SEC = 0.1
TURBOSTAT_MIN_POWER_SAMPLE_COUNT = 16
TURBOSTAT_POWER_SAMPLE_TARGET_COUNT = 20
TURBOSTAT_MIN_POWER_MEASUREMENT_DURATION_SEC = max(
    2.0,
    TURBOSTAT_MIN_POWER_SAMPLE_INTERVAL_SEC * TURBOSTAT_POWER_SAMPLE_TARGET_COUNT,
)
TURBOSTAT_LONG_MIN_POWER_SAMPLE_COUNT = 20
TURBOSTAT_LONG_POWER_SAMPLE_TARGET_COUNT = 24
TURBOSTAT_LONG_MIN_POWER_MEASUREMENT_DURATION_SEC = max(
    3.0,
    TURBOSTAT_MIN_POWER_SAMPLE_INTERVAL_SEC * TURBOSTAT_LONG_POWER_SAMPLE_TARGET_COUNT,
)
LONG_SEQ_POWER_POLICY_MIN_SEQ_LEN = 4096


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "host_comparison"
        / "results.csv"
    )


def default_resume_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2]
        / "results_final"
        / "host_comparison"
        / output_path.name
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def default_effective_gflops_plot_path(output_path: Path) -> Path:
    return output_path.with_name("effective_gflops_comparison.svg")


def default_effective_gflops_per_watt_plot_path(output_path: Path) -> Path:
    return output_path.with_name("effective_gflops_per_watt_comparison.svg")


def generate_synthetic_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    *,
    workload_variant: str,
    dtype: str,
    seed: int,
    include_output: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
    return generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_attention_heads,
        workload_variant=workload_variant,
        dtype=dtype,
        seed=seed,
        include_output=include_output,
    )


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def _parse_power_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Watts", "").replace("W", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_rocm_smi_average_power_w(
    stdout: str,
    *,
    card_label: str | None,
) -> float | None:
    payload = json.loads(stdout)
    candidates = [card_label] if card_label is not None else list(payload.keys())
    for candidate in candidates:
        card_payload = payload.get(candidate)
        if not isinstance(card_payload, dict):
            continue
        for key, value in card_payload.items():
            lowered = key.lower()
            if "power" in lowered and ("socket" in lowered or "package" in lowered):
                parsed = _parse_power_value(value)
                if parsed is not None:
                    return parsed
    return None


def empty_power_stats() -> dict[str, float | str | None]:
    return {
        "power_backend": None,
        "avg_power_w": None,
        "raw_avg_power_w": None,
        "raw_min_power_w": None,
        "raw_max_power_w": None,
        "raw_power_sample_count": None,
        "power_std_w": None,
        "raw_power_std_w": None,
        "power_outlier_sample_count": None,
        "power_outlier_filter_applied": None,
        "min_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "power_sample_count": None,
    }


def resolve_power_sampling_policy(
    *, power_backend: str, seq_len: int | None = None
) -> tuple[float, int, float, int]:
    long_seq = seq_len is not None and int(seq_len) >= LONG_SEQ_POWER_POLICY_MIN_SEQ_LEN
    if power_backend == "rocm-smi":
        if long_seq:
            return (
                ROCM_SMI_LONG_MIN_POWER_MEASUREMENT_DURATION_SEC,
                ROCM_SMI_LONG_MIN_POWER_SAMPLE_COUNT,
                ROCM_SMI_MIN_POWER_SAMPLE_INTERVAL_SEC,
                ROCM_SMI_LONG_POWER_SAMPLE_TARGET_COUNT,
            )
        return (
            ROCM_SMI_MIN_POWER_MEASUREMENT_DURATION_SEC,
            ROCM_SMI_MIN_POWER_SAMPLE_COUNT,
            ROCM_SMI_MIN_POWER_SAMPLE_INTERVAL_SEC,
            ROCM_SMI_POWER_SAMPLE_TARGET_COUNT,
        )
    if long_seq:
        return (
            TURBOSTAT_LONG_MIN_POWER_MEASUREMENT_DURATION_SEC,
            TURBOSTAT_LONG_MIN_POWER_SAMPLE_COUNT,
            TURBOSTAT_MIN_POWER_SAMPLE_INTERVAL_SEC,
            TURBOSTAT_LONG_POWER_SAMPLE_TARGET_COUNT,
        )
    return (
        TURBOSTAT_MIN_POWER_MEASUREMENT_DURATION_SEC,
        TURBOSTAT_MIN_POWER_SAMPLE_COUNT,
        TURBOSTAT_MIN_POWER_SAMPLE_INTERVAL_SEC,
        TURBOSTAT_POWER_SAMPLE_TARGET_COUNT,
    )


def resolve_power_sample_interval_sec(
    *,
    requested_interval_sec: float,
    estimated_timed_window_sec: float | None,
    min_sample_count: int = 6,
    min_interval_sec: float = 0.05,
) -> float:
    interval = float(requested_interval_sec)
    if estimated_timed_window_sec is None or estimated_timed_window_sec <= 0:
        return interval
    target_interval = float(estimated_timed_window_sec) / float(min_sample_count)
    return max(float(min_interval_sec), min(interval, target_interval))


def resolve_power_probe_runs(
    *,
    avg_iteration_sec: float | None,
    baseline_runs: int,
    min_measurement_duration_sec: float = 0.25,
) -> int:
    runs = max(1, int(baseline_runs))
    if avg_iteration_sec is None or avg_iteration_sec <= 0:
        return runs
    return max(runs, int((min_measurement_duration_sec / avg_iteration_sec) + 0.999999))


class RocmSMIPowerMonitor:
    def __init__(
        self,
        *,
        device_index: int = 0,
        sample_interval_sec: float = 0.2,
    ):
        self.device_index = int(device_index)
        self.sample_interval_sec = float(sample_interval_sec)
        self.samples_w: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._card_label = f"card{self.device_index}"
        self._sample_error_count = 0
        self._first_error_message = ""

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4.0))
        return False

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showpower", "--json"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                sample = parse_rocm_smi_average_power_w(
                    result.stdout,
                    card_label=self._card_label,
                )
                if sample is not None:
                    self.samples_w.append(sample)
                else:
                    self._record_sample_failure(
                        "rocm-smi output did not contain a socket/package power "
                        f"reading for {self._card_label}"
                    )
            except Exception as exc:
                self._record_sample_failure(f"{type(exc).__name__}: {exc}")
            self._stop_event.wait(self.sample_interval_sec)

    def current_sample_count(self) -> int:
        return len(self.samples_w)

    def sample_error_count(self) -> int:
        return self._sample_error_count

    def failure_summary(self) -> str | None:
        if self._sample_error_count <= 0:
            return None
        if self._sample_error_count == 1:
            return self._first_error_message
        return (
            f"{self._first_error_message} "
            f"({self._sample_error_count} sampling failures total)"
        )

    def _record_sample_failure(self, message: str) -> None:
        self._sample_error_count += 1
        if not self._first_error_message:
            self._first_error_message = message

    def stats(self, elapsed_sec: float) -> dict[str, float | str | None]:
        stats = empty_power_stats()
        stats["power_backend"] = "rocm-smi"
        if not self.samples_w:
            return stats
        stats.update(
            summarize_power_samples(
                self.samples_w,
                elapsed_sec=elapsed_sec,
            )
        )
        return stats


def create_rocm_power_monitor(
    *,
    power_backend: str,
    device_index: int = 0,
    sample_interval_sec: float = 0.2,
):
    if power_backend == "none":
        return nullcontext(None)
    if power_backend != "rocm-smi":
        raise ValueError(f"Unsupported AMD iGPU power backend: {power_backend}")
    if shutil.which("rocm-smi") is None:
        raise RuntimeError("rocm-smi is required for power_backend='rocm-smi'")
    return RocmSMIPowerMonitor(
        device_index=device_index,
        sample_interval_sec=sample_interval_sec,
    )


def resolve_host_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(
            f"Host comparison expects a CPU or ROCm/CUDA device: {device_name}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Host comparison iGPU benchmark requires a ROCm-enabled torch build with a visible GPU"
        )
    return device


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _allowed_cpu_ids() -> list[int]:
    try:
        return sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
        return list(range(int(cpu_count)))


def _physical_core_count_for_current_affinity() -> int:
    core_pairs: set[tuple[str, str]] = set()
    for cpu_id in _allowed_cpu_ids():
        topology_dir = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        try:
            core_id = (
                topology_dir.joinpath("core_id").read_text(encoding="utf-8").strip()
            )
            package_id = (
                topology_dir.joinpath("physical_package_id")
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:
            continue
        core_pairs.add((package_id, core_id))
    if core_pairs:
        return len(core_pairs)
    return max(1, len(_allowed_cpu_ids()))


def configure_cpu_runtime_for_max_physical_cores() -> int:
    physical_core_count = max(1, _physical_core_count_for_current_affinity())
    torch.set_num_threads(physical_core_count)
    configured_threads = int(torch.get_num_threads())
    LOGGER.info(
        "Configured CPU benchmark to use %s Torch threads across the visible physical cores",
        configured_threads,
    )
    return configured_threads


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 256:
        return (1, 100)
    if seq_len <= 2048:
        return (1, 10)
    if seq_len <= 4096:
        return (1, 5)
    return (1, 10)


def resolve_sampling(
    group: ReferenceGroup,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
) -> tuple[int, int]:
    scheduled_warmup_runs, scheduled_runs_per_sample = iteration_schedule(group.seq_len)
    resolved_warmup_runs = scheduled_warmup_runs
    resolved_runs_per_sample = scheduled_runs_per_sample
    if warmup_runs is not None:
        resolved_warmup_runs = warmup_runs
    if runs_per_sample is not None:
        resolved_runs_per_sample = runs_per_sample
    return int(resolved_warmup_runs), int(resolved_runs_per_sample)


def _summarize_latencies(latencies_sec: list[float]) -> dict[str, float | int | None]:
    measured_inference_count = len(latencies_sec)
    timed_total_sec = sum(latencies_sec)
    avg_latency_ms = None
    min_latency_ms = None
    max_latency_ms = None
    if measured_inference_count:
        avg_latency_ms = (timed_total_sec / measured_inference_count) * 1000.0
        min_latency_ms = min(latencies_sec) * 1000.0
        max_latency_ms = max(latencies_sec) * 1000.0
    return {
        "measured_inference_count": measured_inference_count,
        "latency_sample_count": measured_inference_count,
        "timed_total_sec": timed_total_sec,
        "avg_latency_ms": avg_latency_ms,
        "min_latency_ms": min_latency_ms,
        "max_latency_ms": max_latency_ms,
    }


def _max_acceptable_errors(total_size: int, *, error_threshold: float) -> int:
    return int(total_size * error_threshold)


def _validate_output(
    group: ReferenceGroup,
    output: torch.Tensor | None,
    reference_output: torch.Tensor | None,
) -> dict[str, object]:
    if output is None:
        return {
            "validation_error_count": 0,
            "run_status": "failed_exception",
            "failure_message": "no output tensor was produced",
        }

    if reference_output is None:
        non_finite = int(torch.count_nonzero(~torch.isfinite(output)).item())
        return {
            "validation_error_count": non_finite,
            "run_status": "passed" if non_finite == 0 else "failed_validation",
            "failure_message": (
                "" if non_finite == 0 else f"non_finite_output_count={non_finite}"
            ),
        }

    if tuple(output.shape) != tuple(reference_output.shape):
        return {
            "validation_error_count": 0,
            "run_status": "failed_validation",
            "failure_message": (
                f"output shape {tuple(output.shape)} does not match "
                f"expected {tuple(reference_output.shape)}"
            ),
        }

    mismatches = ~torch.isclose(
        output.float(),
        reference_output.float(),
        rtol=FINAL_REL_TOL,
        atol=FINAL_ABS_TOL,
    )
    validation_error_count = int(torch.count_nonzero(mismatches).item())
    max_acceptable_errors = _max_acceptable_errors(
        group.seq_len * group.hidden_size,
        error_threshold=FINAL_ERROR_THRESHOLD,
    )
    passed = validation_error_count <= max_acceptable_errors
    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if passed else "failed_validation",
        "failure_message": (
            ""
            if passed
            else (
                f"validation_error_count={validation_error_count} exceeds "
                f"max_acceptable_errors={max_acceptable_errors}"
            )
        ),
    }


def _forward_reference(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_attention_heads: int,
    workload_variant: str,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_attention_heads

    if workload_variant == "decoder_gpt2":
        attn_input = F.layer_norm(
            hidden_states,
            (hidden_size,),
            weights["ln1_weight"],
            None,
        )
        residual_after_attention = hidden_states
    else:
        attn_input = hidden_states
        residual_after_attention = hidden_states

    q = torch.matmul(attn_input, weights["q_weight"])
    k = torch.matmul(attn_input, weights["k_weight"])
    v = torch.matmul(attn_input, weights["v_weight"])

    q = q.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    if workload_variant == "decoder_gpt2":
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(
                    (seq_len, seq_len),
                    device=hidden_states.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        attn_scores = attn_scores.masked_fill(causal_mask, -10000.0)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, weights["attn_output_weight"])

    if workload_variant == "decoder_gpt2":
        residual_hidden_states = attn_output + residual_after_attention
        ffn_input = F.layer_norm(
            residual_hidden_states,
            (hidden_size,),
            weights["ln2_weight"],
            None,
        )
        intermediate = torch.matmul(ffn_input, weights["ffn_up_weight"])
        intermediate = F.gelu(intermediate)
        ffn_output = torch.matmul(intermediate, weights["ffn_down_weight"])
        return ffn_output + residual_hidden_states

    hidden_states = F.layer_norm(
        attn_output + residual_after_attention,
        (hidden_size,),
        weights["ln1_weight"],
        None,
    )
    intermediate = torch.matmul(hidden_states, weights["ffn_up_weight"])
    intermediate = F.gelu(intermediate)
    ffn_output = torch.matmul(intermediate, weights["ffn_down_weight"])
    return F.layer_norm(
        ffn_output + hidden_states,
        (hidden_size,),
        weights["ln2_weight"],
        None,
    )


def _host_power_monitor(
    runtime_device: torch.device,
    *,
    power_backend: str,
    sample_interval_sec: float,
    estimated_timed_window_sec: float | None,
):
    if power_backend == "turbostat_pkgwatt" or runtime_device.type == "cpu":
        return create_cpu_power_monitor(
            power_backend=power_backend,
            sample_interval_sec=sample_interval_sec,
            quiescent_baseline_duration_sec=0.5,
            estimated_timed_window_sec=estimated_timed_window_sec,
        )
    return create_rocm_power_monitor(
        power_backend=power_backend,
        device_index=_device_index(runtime_device),
        sample_interval_sec=sample_interval_sec,
    )


def _measure_host_power_stats(
    *,
    seq_len: int,
    runtime_device: torch.device,
    forward_once,
    power_backend: str,
    power_sample_interval_sec: float,
    runs_per_sample: int,
    avg_iteration_sec: float | None,
    timed_total_sec: float,
) -> dict[str, float | str | None]:
    if power_backend == "none":
        power_stats = empty_power_stats()
        power_stats["power_backend"] = "none"
        return power_stats

    (
        min_measurement_duration_sec,
        min_sample_count,
        min_interval_sec,
        target_sample_count,
    ) = resolve_power_sampling_policy(power_backend=power_backend, seq_len=seq_len)
    power_probe_runs = resolve_power_probe_runs(
        avg_iteration_sec=avg_iteration_sec,
        baseline_runs=runs_per_sample,
        min_measurement_duration_sec=min_measurement_duration_sec,
    )
    estimated_window_sec = (
        None
        if avg_iteration_sec is None
        else avg_iteration_sec * float(power_probe_runs)
    )
    sample_interval_sec = resolve_power_sample_interval_sec(
        requested_interval_sec=power_sample_interval_sec,
        estimated_timed_window_sec=estimated_window_sec,
        min_sample_count=target_sample_count,
        min_interval_sec=min_interval_sec,
    )
    with _host_power_monitor(
        runtime_device,
        power_backend=power_backend,
        sample_interval_sec=sample_interval_sec,
        estimated_timed_window_sec=estimated_window_sec,
    ) as power_monitor:
        started = time.perf_counter()
        completed_runs = 0
        while True:
            forward_once()
            completed_runs += 1
            time.sleep(0)
            elapsed_sec = time.perf_counter() - started
            if power_probe_is_complete(
                completed_runs=completed_runs,
                min_runs=power_probe_runs,
                elapsed_sec=elapsed_sec,
                min_measurement_duration_sec=min_measurement_duration_sec,
                observed_sample_count=power_monitor.current_sample_count(),
                min_sample_count=min_sample_count,
            ):
                break
    power_stats = power_monitor.stats(elapsed_sec)
    power_error_count = (
        power_monitor.sample_error_count()
        if hasattr(power_monitor, "sample_error_count")
        else 0
    )
    power_failure_message = (
        power_monitor.failure_summary()
        if hasattr(power_monitor, "failure_summary")
        else None
    )
    power_stats["power_probe_failed"] = False
    power_stats["power_failure_message"] = ""
    power_stats["power_error_count"] = power_error_count
    if power_stats.get("avg_power_w") is None:
        if power_backend != "none":
            power_stats["power_probe_failed"] = True
            power_stats["power_failure_message"] = (
                power_failure_message
                or f"{power_backend} collected no valid power samples"
            )
            LOGGER.warning(
                "Host power probe failed for %s on %s: %s",
                power_backend,
                runtime_device,
                power_stats["power_failure_message"],
            )
        return power_stats
    if power_error_count > 0:
        power_stats["power_failure_message"] = power_failure_message or ""
        LOGGER.warning(
            "Host power probe for %s on %s ignored %d invalid samples: %s",
            power_backend,
            runtime_device,
            power_error_count,
            power_stats["power_failure_message"],
        )
    if power_stats.get("avg_power_w") is not None:
        power_stats["energy_j"] = float(power_stats["avg_power_w"]) * float(
            timed_total_sec
        )
    return power_stats


def _prepare_host_forward_once(
    group: ReferenceGroup,
    *,
    seed: int,
    device_name: str,
    include_output: bool,
) -> tuple[torch.device, torch.Tensor | None, object]:
    runtime_device = resolve_host_device(device_name)
    if runtime_device.type == "cpu":
        configure_cpu_runtime_for_max_physical_cores()
    reference = generate_synthetic_reference(
        group.seq_len,
        group.hidden_size,
        group.intermediate_size,
        group.num_attention_heads,
        workload_variant=group.workload_variant,
        dtype=group.dtype,
        seed=seed,
        include_output=include_output,
    )
    if not isinstance(reference["input"], torch.Tensor):
        raise RuntimeError("reference generator did not return an input tensor")
    if not isinstance(reference["weights"], dict):
        raise RuntimeError("reference generator did not return a weight mapping")

    runtime_input = reference["input"].to(runtime_device)
    runtime_weights = {
        name: tensor.to(runtime_device)
        for name, tensor in reference["weights"].items()
        if isinstance(tensor, torch.Tensor)
    }
    runtime_reference_output = (
        reference["output"].to(runtime_device)
        if include_output and isinstance(reference["output"], torch.Tensor)
        else None
    )
    runtime_causal_mask = None
    if group.workload_variant == "decoder_gpt2":
        runtime_causal_mask = torch.triu(
            torch.ones(
                (group.seq_len, group.seq_len),
                device=runtime_device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def forward_once() -> torch.Tensor:
        with torch.no_grad():
            output = _forward_reference(
                runtime_input,
                runtime_weights,
                num_attention_heads=group.num_attention_heads,
                workload_variant=group.workload_variant,
                causal_mask=runtime_causal_mask,
            )
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)
        return output

    return runtime_device, runtime_reference_output, forward_once


def benchmark_host_group(
    group: ReferenceGroup,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    device_name: str,
    power_backend: str,
    power_sample_interval_sec: float,
    extra_power_backends: tuple[str, ...] = (),
) -> dict[str, object]:
    include_output = group.seq_len <= REFERENCE_VALIDATION_MAX_SEQ_LEN
    runtime_device, runtime_reference_output, forward_once = _prepare_host_forward_once(
        group,
        seed=seed,
        device_name=device_name,
        include_output=include_output,
    )

    for _ in range(warmup_runs):
        forward_once()

    latencies_sec: list[float] = []
    last_output = None
    for _ in range(runs_per_sample):
        started = time.perf_counter()
        last_output = forward_once()
        latencies_sec.append(time.perf_counter() - started)

    summary = _summarize_latencies(latencies_sec)
    avg_iteration_sec = None
    if summary["measured_inference_count"]:
        avg_iteration_sec = float(summary["timed_total_sec"]) / float(
            summary["measured_inference_count"]
        )

    power_stats = _measure_host_power_stats(
        seq_len=group.seq_len,
        runtime_device=runtime_device,
        forward_once=forward_once,
        power_backend=power_backend,
        power_sample_interval_sec=power_sample_interval_sec,
        runs_per_sample=runs_per_sample,
        avg_iteration_sec=avg_iteration_sec,
        timed_total_sec=float(summary["timed_total_sec"]),
    )
    extra_power_stats = {
        backend: _measure_host_power_stats(
            seq_len=group.seq_len,
            runtime_device=runtime_device,
            forward_once=forward_once,
            power_backend=backend,
            power_sample_interval_sec=power_sample_interval_sec,
            runs_per_sample=runs_per_sample,
            avg_iteration_sec=avg_iteration_sec,
            timed_total_sec=float(summary["timed_total_sec"]),
        )
        for backend in extra_power_backends
    }

    validation = _validate_output(group, last_output, runtime_reference_output)
    effective_gflops = effective_gflops_per_sec(
        seq_len=group.seq_len,
        hidden_size=group.hidden_size,
        intermediate_size=group.intermediate_size,
        num_attention_heads=group.num_attention_heads,
        avg_latency_ms=_optional_float(summary["avg_latency_ms"]),
    )
    return {
        **summary,
        "effective_gflops_per_sec": effective_gflops,
        "power_backend": power_stats.get("power_backend"),
        **{field: power_stats.get(field) for field in PERSISTED_POWER_RESULT_FIELDS},
        "energy_j": power_stats.get("energy_j"),
        "power_probe_failed": power_stats.get("power_probe_failed"),
        "power_failure_message": power_stats.get("power_failure_message"),
        "power_error_count": power_stats.get("power_error_count"),
        "effective_gflops_per_sec_per_watt": effective_gflops_per_sec_per_watt(
            effective_gflops,
            _optional_float(power_stats.get("avg_power_w")),
        ),
        "process_model": "in_process",
        "host_device": str(runtime_device),
        "extra_power_stats": extra_power_stats,
        **validation,
    }


def benchmark_host_group_power_only(
    group: ReferenceGroup,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    device_name: str,
    power_backend: str,
    power_sample_interval_sec: float,
    existing_avg_latency_ms: float | None,
    existing_effective_gflops_per_sec: float | None,
) -> dict[str, object]:
    try:
        runtime_device, _, forward_once = _prepare_host_forward_once(
            group,
            seed=seed,
            device_name=device_name,
            include_output=False,
        )
        for _ in range(warmup_runs):
            forward_once()

        avg_iteration_sec = (
            None
            if existing_avg_latency_ms is None or existing_avg_latency_ms <= 0
            else existing_avg_latency_ms / 1000.0
        )
        timed_total_sec = (
            0.0
            if existing_avg_latency_ms is None or existing_avg_latency_ms <= 0
            else avg_iteration_sec * float(runs_per_sample)
        )
        power_stats = _measure_host_power_stats(
            seq_len=group.seq_len,
            runtime_device=runtime_device,
            forward_once=forward_once,
            power_backend=power_backend,
            power_sample_interval_sec=power_sample_interval_sec,
            runs_per_sample=runs_per_sample,
            avg_iteration_sec=avg_iteration_sec,
            timed_total_sec=timed_total_sec,
        )
        run_status = "passed"
        failure_message = ""
        if bool(power_stats.get("power_probe_failed")):
            run_status = "failed_power_probe"
            failure_message = str(power_stats.get("power_failure_message") or "")
        return {
            "power_backend": power_stats.get("power_backend"),
            **{
                field: power_stats.get(field) for field in PERSISTED_POWER_RESULT_FIELDS
            },
            "energy_j": power_stats.get("energy_j"),
            "effective_gflops_per_sec_per_watt": effective_gflops_per_sec_per_watt(
                existing_effective_gflops_per_sec,
                _optional_float(power_stats.get("avg_power_w")),
            ),
            "run_status": run_status,
            "failure_message": failure_message,
        }
    except Exception as exc:
        return {
            "power_backend": power_backend,
            **{field: None for field in PERSISTED_POWER_RESULT_FIELDS},
            "energy_j": None,
            "effective_gflops_per_sec_per_watt": None,
            "run_status": "failed_exception",
            "failure_message": f"{type(exc).__name__}: {exc}",
        }


def _float_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _reference_metric_means(
    group: ReferenceGroup,
    metric_field: str,
) -> dict[str, float | None]:
    values_by_mode: dict[str, list[float]] = {
        execution_mode: [] for execution_mode in REFERENCE_EXECUTION_MODES
    }
    for reference_row in group.rows:
        execution_mode = str(reference_row.get("execution_mode") or "")
        if execution_mode not in values_by_mode:
            continue
        metric_value = _optional_float(reference_row.get(metric_field))
        if metric_value is not None:
            values_by_mode[execution_mode].append(metric_value)
    return {
        execution_mode: _float_mean(values_by_mode[execution_mode])
        for execution_mode in REFERENCE_EXECUTION_MODES
    }


def _host_metric_value(
    benchmark_result: dict[str, object] | None,
    metric_field: str,
) -> float | None:
    if benchmark_result is None:
        return None
    if str(benchmark_result.get("run_status") or "passed") != "passed":
        return None
    return _optional_float(benchmark_result.get(metric_field))


def _igpu_effective_gflops_per_watt_for_backend(
    benchmark_result: dict[str, object] | None,
    *,
    power_backend: str,
) -> float | None:
    if benchmark_result is None:
        return None
    if str(benchmark_result.get("power_backend") or "") == power_backend:
        return _host_metric_value(
            benchmark_result,
            "effective_gflops_per_sec_per_watt",
        )
    effective_gflops = _host_metric_value(benchmark_result, "effective_gflops_per_sec")
    extra_stats = benchmark_result.get("extra_power_stats")
    if not isinstance(extra_stats, dict):
        return None
    backend_stats = extra_stats.get(power_backend)
    if not isinstance(backend_stats, dict):
        if power_backend != "rocm-smi":
            return None
        return effective_gflops_per_sec_per_watt(
            effective_gflops,
            _host_metric_value(benchmark_result, "avg_power_w"),
        )
    avg_power_w = _optional_float(backend_stats.get("avg_power_w"))
    return effective_gflops_per_sec_per_watt(effective_gflops, avg_power_w)


def _repair_missing_igpu_rocm_smi_per_watt_rows(
    rows: dict[tuple[str, str, int, str], dict[str, object]],
) -> dict[tuple[str, str, int, str], dict[str, object]]:
    repaired_rows = dict(rows)
    for row_key, row in rows.items():
        workload_variant, study_case_id, seq_len, metric = row_key
        if metric != "effective_gflops_per_sec_per_watt":
            continue
        if _optional_float(row.get("igpu_rocm_smi")) is not None:
            continue
        throughput_row = rows.get(
            (workload_variant, study_case_id, seq_len, "effective_gflops_per_sec")
        )
        avg_power_row = rows.get(
            (workload_variant, study_case_id, seq_len, "avg_power_w")
        )
        repaired_value = effective_gflops_per_sec_per_watt(
            _optional_float(
                None if throughput_row is None else throughput_row.get("igpu")
            ),
            _optional_float(
                None if avg_power_row is None else avg_power_row.get("igpu_rocm_smi")
            ),
        )
        if repaired_value is None:
            continue
        repaired_row = dict(row)
        repaired_row["igpu_rocm_smi"] = repaired_value
        repaired_rows[row_key] = repaired_row
    return repaired_rows


def _comparison_row(
    *,
    group: ReferenceGroup,
    metric: str,
    igpu_value: float | None,
    igpu_rocm_smi_value: float | None = None,
    reference_values: dict[str, float | None],
) -> dict[str, object]:
    row: dict[str, object] = {
        "workload_variant": group.workload_variant,
        "study_case_id": group.study_case_id,
        "seq_len": group.seq_len,
        "metric": metric,
        "igpu": igpu_value,
        "igpu_rocm_smi": igpu_rocm_smi_value,
    }
    for execution_mode in REFERENCE_EXECUTION_MODES:
        row[execution_mode] = reference_values.get(execution_mode)
    return row


def _row_key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row.get("workload_variant") or ""),
        str(row.get("study_case_id") or ""),
        int(float(str(row.get("seq_len") or 0))),
        str(row.get("metric") or ""),
    )


def _normalized_existing_row(row: dict[str, str]) -> dict[str, object] | None:
    metric = str(row.get("metric") or "").strip()
    if not metric:
        return None
    study_case_id = str(row.get("study_case_id") or "").strip()
    if not study_case_id:
        return None
    seq_len_text = str(row.get("seq_len") or "").strip()
    if not seq_len_text:
        return None
    try:
        seq_len = int(float(seq_len_text))
    except ValueError:
        return None
    workload_variant = str(row.get("workload_variant") or "").strip()
    if not workload_variant and study_case_id in FAMILY_SPECS:
        workload_variant = FAMILY_SPECS[study_case_id].workload_variant
    if not workload_variant:
        return None
    normalized: dict[str, object] = {
        "workload_variant": workload_variant,
        "study_case_id": study_case_id,
        "seq_len": seq_len,
        "metric": metric,
        "igpu": row.get("igpu", ""),
        "igpu_rocm_smi": row.get("igpu_rocm_smi", ""),
    }
    for execution_mode in REFERENCE_EXECUTION_MODES:
        normalized[execution_mode] = row.get(execution_mode, "")
    return normalized


def load_existing_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, str, int, str], dict[str, object]]:
    rows: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                normalized = _normalized_existing_row(row)
                if normalized is None:
                    continue
                rows[_row_key(normalized)] = normalized
    return _repair_missing_igpu_rocm_smi_per_watt_rows(rows)


def reusable_rows_for_group(
    existing_rows: dict[tuple[str, str, int, str], dict[str, object]],
    *,
    group: ReferenceGroup,
    reference_metric_values: dict[str, dict[str, float | None]],
) -> list[dict[str, object]] | None:
    reused_rows: list[dict[str, object]] = []
    for metric in DIRECT_COMPARISON_METRICS:
        row = existing_rows.get(
            (
                group.workload_variant,
                group.study_case_id,
                group.seq_len,
                metric,
            )
        )
        if row is None:
            return None
        igpu_value = _optional_float(row.get("igpu"))
        if igpu_value is None:
            return None
        reused_rows.append(
            _comparison_row(
                group=group,
                metric=metric,
                igpu_value=igpu_value,
                reference_values=reference_metric_values[metric],
            )
        )
    for metric in POWER_COMPARISON_METRICS:
        row = existing_rows.get(
            (
                group.workload_variant,
                group.study_case_id,
                group.seq_len,
                metric,
            )
        )
        if row is None:
            return None
        igpu_rocm_smi_value = _optional_float(row.get("igpu_rocm_smi"))
        if igpu_rocm_smi_value is None:
            return None
        reused_rows.append(
            _comparison_row(
                group=group,
                metric=metric,
                igpu_value=None,
                igpu_rocm_smi_value=igpu_rocm_smi_value,
                reference_values=reference_metric_values[metric],
            )
        )
    return reused_rows


def _resolve_plot_path(path: Path) -> Path:
    if path.suffix.lower() != SUPPORTED_PLOT_SUFFIX:
        raise ValueError(
            f"Plot output must use the {SUPPORTED_PLOT_SUFFIX} suffix: {path}"
        )
    return path


def _ordered_study_case_ids(metric_rows: list[dict[str, object]]) -> list[str]:
    present = {
        str(row.get("study_case_id") or "")
        for row in metric_rows
        if str(row.get("study_case_id") or "")
    }
    ordered = [family_id for family_id in PLOT_FAMILY_ORDER if family_id in present]
    extras = sorted(present - set(PLOT_FAMILY_ORDER))
    return ordered + extras


def _ordered_seq_lens(metric_rows: list[dict[str, object]]) -> list[int]:
    present = {
        int(row.get("seq_len") or 0)
        for row in metric_rows
        if row.get("seq_len") not in (None, "", "None")
    }
    ordered = [seq_len for seq_len in PLOT_SEQ_ORDER if seq_len in present]
    extras = sorted(present - set(PLOT_SEQ_ORDER))
    return ordered + extras


def render_metric_plot(
    rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    y_axis_label: str,
    variant: str = "standard",
) -> plt.Figure:
    metric_rows = [row for row in rows if str(row.get("metric") or "") == metric]
    plot_series = (
        PLOT_SERIES_THROUGHPUT
        if metric == "effective_gflops_per_sec"
        else PLOT_SERIES_PER_WATT
    )
    plt.rcParams["svg.fonttype"] = "none"
    sns.set_theme(
        style="whitegrid",
        context="talk" if variant == "standard" else "poster",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )

    if not metric_rows:
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_axis_off()
        fig.text(
            0.5,
            0.57,
            title,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.42,
            "No data available",
            ha="center",
            va="center",
            fontsize=18,
        )
        fig.patch.set_facecolor("#f7f5f2")
        return fig

    study_case_ids = _ordered_study_case_ids(metric_rows)
    seq_lens = _ordered_seq_lens(metric_rows)
    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=(18, 10),
        sharey=True,
    )
    axes_grid = axes_obj
    present_study_case_ids = set(study_case_ids)

    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        if family.family_id not in present_study_case_ids:
            ax.set_axis_off()
            continue

        study_case_id = family.family_id
        family_rows = [
            row
            for row in metric_rows
            if str(row.get("study_case_id") or "") == study_case_id
        ]
        series_values = {
            series_name: {
                int(row.get("seq_len") or 0): _optional_float(row.get(series_name))
                for row in family_rows
            }
            for series_name, _, _ in plot_series
        }

        x_positions = list(range(len(seq_lens)))
        group_width = 0.82
        bar_width = group_width / float(len(plot_series))

        for series_index, (series_name, _, color) in enumerate(plot_series):
            x_values: list[float] = []
            y_values: list[float] = []
            for seq_index, seq_len in enumerate(seq_lens):
                value = series_values[series_name].get(seq_len)
                if value is None:
                    continue
                x_values.append(
                    seq_index - (group_width / 2.0) + (series_index * bar_width)
                )
                y_values.append(value)
            if not x_values:
                continue
            ax.bar(
                x_values,
                y_values,
                width=bar_width * 0.94,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                align="edge",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(seq_len) for seq_len in seq_lens], rotation=0)
        ax.set_xlabel("Sequence Length", fontsize=15)
        ax.set_ylabel(y_axis_label, fontsize=15)
        ax.set_title(
            plot_family_label(study_case_id),
            loc="left",
            fontsize=18 if variant == "standard" else 20,
            pad=6,
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.tick_params(axis="both", labelsize=12 if variant == "standard" else 14)
        if family.col_index != 0:
            ax.set_ylabel("")

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in plot_series
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.89, 0.5),
        fontsize=13 if variant == "standard" else 16,
    )
    fig.suptitle(
        title,
        fontsize=24 if variant == "standard" else 30,
        fontweight="bold",
        y=0.98 if variant == "standard" else 0.99,
    )
    fig.tight_layout(rect=[0, 0.03, 0.87, 0.94])
    return fig


def write_metric_plot(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    y_axis_label: str,
    variant: str = "standard",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_metric_plot(
        rows,
        metric=metric,
        title=title,
        y_axis_label=y_axis_label,
        variant=variant,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_plots(
    rows: list[dict[str, object]],
    *,
    effective_gflops_plot_path: Path,
    effective_gflops_per_watt_plot_path: Path,
) -> None:
    write_metric_plot(
        _resolve_plot_path(effective_gflops_plot_path),
        rows,
        metric="effective_gflops_per_sec",
        title="Effective Throughput Comparison",
        y_axis_label="GFLOPS",
    )
    write_metric_plot(
        _resolve_plot_path(effective_gflops_per_watt_plot_path),
        rows,
        metric="effective_gflops_per_sec_per_watt",
        title="Effective Throughput/W Comparison",
        y_axis_label="GFLOPS / W",
    )


def _log_backend_failure(
    backend: str,
    group: ReferenceGroup,
    exc: Exception | None = None,
    benchmark_result: dict[str, object] | None = None,
) -> None:
    if exc is not None:
        LOGGER.warning(
            "%s benchmark unavailable for %s seq_len=%s: %s",
            backend.upper(),
            group.study_case_id,
            group.seq_len,
            exc,
        )
        return
    if (
        benchmark_result is not None
        and str(benchmark_result.get("run_status") or "passed") != "passed"
    ):
        LOGGER.warning(
            "%s benchmark did not pass for %s seq_len=%s: %s",
            backend.upper(),
            group.study_case_id,
            group.seq_len,
            benchmark_result.get("failure_message") or benchmark_result["run_status"],
        )
        return
    if benchmark_result is not None and bool(
        benchmark_result.get("power_probe_failed")
    ):
        LOGGER.warning(
            "%s power probe degraded for %s seq_len=%s: %s",
            backend.upper(),
            group.study_case_id,
            group.seq_len,
            benchmark_result.get("power_failure_message") or "probe failed",
        )


def build_rows_for_group(
    group: ReferenceGroup,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    host_backends: tuple[str, ...],
    igpu_device_name: str,
    igpu_power_backend: str,
    igpu_power_sample_interval_sec: float,
    existing_rows: dict[tuple[str, str, int, str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    resolved_warmup_runs, resolved_runs_per_sample = resolve_sampling(
        group,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
    )
    reference_metric_values = {
        metric: _reference_metric_means(group, metric)
        for metric in (*DIRECT_COMPARISON_METRICS, *POWER_COMPARISON_METRICS)
    }
    reused_rows = reusable_rows_for_group(
        {} if existing_rows is None else existing_rows,
        group=group,
        reference_metric_values=reference_metric_values,
    )
    if reused_rows is not None:
        LOGGER.info(
            "Reusing host comparison rows for %s seq_len=%s",
            group.study_case_id,
            group.seq_len,
        )
        return reused_rows

    benchmark_results: dict[str, dict[str, object] | None] = {"igpu": None}
    backend_configs = {
        "igpu": {
            "device_name": igpu_device_name,
            "power_backend": igpu_power_backend,
            "power_sample_interval_sec": igpu_power_sample_interval_sec,
        },
    }
    for backend in host_backends:
        try:
            benchmark_results[backend] = benchmark_host_group(
                group,
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
                seed=seed,
                device_name=str(backend_configs[backend]["device_name"]),
                power_backend=str(backend_configs[backend]["power_backend"]),
                power_sample_interval_sec=float(
                    backend_configs[backend]["power_sample_interval_sec"]
                ),
            )
        except Exception as exc:
            _log_backend_failure(backend, group, exc=exc)
            benchmark_results[backend] = {
                "run_status": "unsupported",
                "failure_message": str(exc),
            }
        else:
            _log_backend_failure(
                backend,
                group,
                benchmark_result=benchmark_results[backend],
            )
    rows = [
        _comparison_row(
            group=group,
            metric=metric,
            igpu_value=_host_metric_value(benchmark_results["igpu"], metric),
            reference_values=reference_metric_values[metric],
        )
        for metric in DIRECT_COMPARISON_METRICS
    ]
    rows.extend(
        _comparison_row(
            group=group,
            metric=metric,
            igpu_value=None,
            igpu_rocm_smi_value=(
                _igpu_effective_gflops_per_watt_for_backend(
                    benchmark_results["igpu"],
                    power_backend="rocm-smi",
                )
                if metric == "effective_gflops_per_sec_per_watt"
                else _host_metric_value(benchmark_results["igpu"], metric)
            ),
            reference_values=reference_metric_values[metric],
        )
        for metric in POWER_COMPARISON_METRICS
    )
    return rows


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
        description="Benchmark AMD iGPU and write host-vs-NPU comparison CSV and plots"
    )
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--host-backends",
        choices=("all", *SUPPORTED_HOST_BACKENDS),
        default="all",
    )
    parser.add_argument("--igpu-device", default="cuda:0")
    parser.add_argument(
        "--igpu-power-backend",
        choices=list(SUPPORTED_IGPU_POWER_BACKENDS),
        default="rocm-smi",
    )
    parser.add_argument("--igpu-power-sample-interval-sec", type=float, default=0.2)
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=default_reference_results_path(),
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--resume-input", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--effective-gflops-plot",
        "--tps-plot",
        dest="effective_gflops_plot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--effective-gflops-per-watt-plot",
        "--tps-per-watt-plot",
        dest="effective_gflops_per_watt_plot",
        type=Path,
        default=None,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _selected_host_backends(argument: str) -> tuple[str, ...]:
    if argument == "all":
        return SUPPORTED_HOST_BACKENDS
    return (argument,)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    reference_input = args.reference_input.expanduser()
    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="host comparison",
    ):
        resume_paths: tuple[Path, ...] = tuple()
        if args.no_resume:
            resume_paths = tuple()
        elif args.resume_input is not None:
            resume_paths = (args.resume_input.expanduser(),)
        else:
            resume_paths = default_resume_paths(output_path)
        effective_gflops_plot_path = (
            default_effective_gflops_plot_path(output_path)
            if args.effective_gflops_plot is None
            else _resolve_plot_path(args.effective_gflops_plot.expanduser())
        )
        effective_gflops_per_watt_plot_path = (
            default_effective_gflops_per_watt_plot_path(output_path)
            if args.effective_gflops_per_watt_plot is None
            else _resolve_plot_path(args.effective_gflops_per_watt_plot.expanduser())
        )

        if not reference_input.exists():
            LOGGER.warning(
                "Reference end_to_end results not found at %s; writing empty host comparison outputs",
                reference_input,
            )
            write_rows(output_path, [])
            write_plots(
                [],
                effective_gflops_plot_path=effective_gflops_plot_path,
                effective_gflops_per_watt_plot_path=effective_gflops_per_watt_plot_path,
            )
            return 0

        groups = group_reference_rows(
            load_reference_rows(reference_input),
            family_filter=str(args.family),
            seq_len_filter=str(args.seq_len),
        )

        if not groups:
            LOGGER.warning(
                "No matching end_to_end rows found in %s for family=%s seq_len=%s; "
                "writing empty host comparison outputs",
                reference_input,
                args.family,
                args.seq_len,
            )
            write_rows(output_path, [])
            write_plots(
                [],
                effective_gflops_plot_path=effective_gflops_plot_path,
                effective_gflops_per_watt_plot_path=effective_gflops_per_watt_plot_path,
            )
            return 0

        existing_rows = load_existing_rows(resume_paths)
        if resume_paths and existing_rows:
            LOGGER.info(
                "Loaded %d reusable host comparison rows from %s",
                len(existing_rows),
                ", ".join(str(path) for path in resume_paths),
            )

        row_map: dict[tuple[str, str, int, str], dict[str, object]] = dict(
            existing_rows
        )
        rows: list[dict[str, object]] = list(row_map.values())
        for group in groups:
            for row in build_rows_for_group(
                group,
                warmup_runs=args.warmup_iters,
                runs_per_sample=args.timed_iters,
                seed=args.seed,
                host_backends=_selected_host_backends(str(args.host_backends)),
                igpu_device_name=str(args.igpu_device),
                igpu_power_backend=str(args.igpu_power_backend),
                igpu_power_sample_interval_sec=float(
                    args.igpu_power_sample_interval_sec
                ),
                existing_rows=existing_rows,
            ):
                row_map[_row_key(row)] = row
            row_map = _repair_missing_igpu_rocm_smi_per_watt_rows(row_map)
            rows = list(row_map.values())

        write_rows(output_path, rows)
        write_plots(
            rows,
            effective_gflops_plot_path=effective_gflops_plot_path,
            effective_gflops_per_watt_plot_path=effective_gflops_per_watt_plot_path,
        )
        LOGGER.info("Wrote %d host comparison rows to %s", len(rows), output_path)
        LOGGER.info(
            "Wrote comparison plots to %s and %s",
            effective_gflops_plot_path,
            effective_gflops_per_watt_plot_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
