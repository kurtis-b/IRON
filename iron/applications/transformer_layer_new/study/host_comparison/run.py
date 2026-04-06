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

from iron.applications.transformer_layer_new.study.end_to_end.cases import (
    FAMILY_SPECS,
    effective_gflops_per_sec,
    effective_gflops_per_sec_per_watt,
)
from iron.applications.transformer_layer_new.study.end_to_end.power import (
    create_power_monitor as create_cpu_power_monitor,
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
COMPARISON_COLUMNS: tuple[str, ...] = ("igpu", *REFERENCE_EXECUTION_MODES)
RESULTS_CSV_FIELDNAMES = (
    "study_case_id",
    "seq_len",
    "metric",
    *COMPARISON_COLUMNS,
)
PLOT_SERIES = (
    ("igpu", "iGPU", "#e07a5f"),
    ("dataflow", "NPU Dataflow", "#1f6f8b"),
)
PLOT_FAMILY_ORDER = ("tinybert_512", "baseline_768", "baseline_1024")
PLOT_SEQ_ORDER = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PLOT_FAMILY_LABELS = {
    family_id: (
        f"Head Dim = {hidden_size // num_heads} / "
        f"Num Heads = {num_heads} / "
        f"FFN Dim = {intermediate_size}"
    )
    for family_id, (hidden_size, intermediate_size, num_heads) in FAMILY_SPECS.items()
}
SUPPORTED_PLOT_SUFFIX = ".svg"
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "host_comparison"
        / "results.csv"
    )


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
    dtype: str,
    seed: int,
    include_output: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
    torch.manual_seed(seed)
    value_range = 0.05
    torch_dtype = TORCH_DTYPES.get(dtype, torch.bfloat16)

    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch_dtype) * value_range

    q_weight = torch.randn(hidden_size, hidden_size, dtype=torch_dtype) * value_range
    k_weight = torch.randn(hidden_size, hidden_size, dtype=torch_dtype) * value_range
    v_weight = torch.randn(hidden_size, hidden_size, dtype=torch_dtype) * value_range
    attn_output_weight = (
        torch.randn(hidden_size, hidden_size, dtype=torch_dtype) * value_range
    )
    ln1_weight = torch.rand(hidden_size, dtype=torch_dtype)
    ffn_up_weight = (
        torch.randn(hidden_size, intermediate_size, dtype=torch_dtype) * value_range
    )
    ffn_down_weight = (
        torch.randn(intermediate_size, hidden_size, dtype=torch_dtype) * value_range
    )
    ln2_weight = torch.rand(hidden_size, dtype=torch_dtype)

    output = None
    weights = {
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "attn_output_weight": attn_output_weight,
        "ln1_weight": ln1_weight,
        "ffn_up_weight": ffn_up_weight,
        "ffn_down_weight": ffn_down_weight,
        "ln2_weight": ln2_weight,
    }
    if include_output:
        output = _forward_reference(
            input_tensor,
            weights,
            num_attention_heads=num_attention_heads,
        )

    return {
        "input": input_tensor,
        "weights": weights,
        "output": output,
        "attention_mask": None,
    }


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
        "max_power_w": None,
        "energy_j": None,
        "power_sample_count": None,
    }


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
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_sec)

    def stats(self, elapsed_sec: float) -> dict[str, float | str | None]:
        stats = empty_power_stats()
        stats["power_backend"] = "rocm-smi"
        if not self.samples_w:
            return stats
        avg_power_w = sum(self.samples_w) / len(self.samples_w)
        stats.update(
            {
                "avg_power_w": avg_power_w,
                "max_power_w": max(self.samples_w),
                "energy_j": avg_power_w * elapsed_sec,
                "power_sample_count": len(self.samples_w),
            }
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
    return (1, 2)


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
    if measured_inference_count:
        avg_latency_ms = (timed_total_sec / measured_inference_count) * 1000.0
    return {
        "measured_inference_count": measured_inference_count,
        "timed_total_sec": timed_total_sec,
        "avg_latency_ms": avg_latency_ms,
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
) -> torch.Tensor:
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_attention_heads

    q = torch.matmul(hidden_states, weights["q_weight"])
    k = torch.matmul(hidden_states, weights["k_weight"])
    v = torch.matmul(hidden_states, weights["v_weight"])

    q = q.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, weights["attn_output_weight"])

    hidden_states = F.layer_norm(
        attn_output + hidden_states,
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
    if runtime_device.type == "cpu":
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


def benchmark_host_group(
    group: ReferenceGroup,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    device_name: str,
    power_backend: str,
    power_sample_interval_sec: float,
) -> dict[str, object]:
    runtime_device = resolve_host_device(device_name)
    if runtime_device.type == "cpu":
        configure_cpu_runtime_for_max_physical_cores()
    include_output = group.seq_len <= REFERENCE_VALIDATION_MAX_SEQ_LEN
    reference = generate_synthetic_reference(
        group.seq_len,
        group.hidden_size,
        group.intermediate_size,
        group.num_attention_heads,
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
        if isinstance(reference["output"], torch.Tensor)
        else None
    )

    def forward_once() -> torch.Tensor:
        with torch.no_grad():
            output = _forward_reference(
                runtime_input,
                runtime_weights,
                num_attention_heads=group.num_attention_heads,
            )
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)
        return output

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

    if power_backend == "none":
        power_stats = empty_power_stats()
        power_stats["power_backend"] = "none"
    else:
        power_probe_runs = resolve_power_probe_runs(
            avg_iteration_sec=avg_iteration_sec,
            baseline_runs=runs_per_sample,
            min_measurement_duration_sec=0.25,
        )
        estimated_window_sec = (
            None
            if avg_iteration_sec is None
            else avg_iteration_sec * float(power_probe_runs)
        )
        sample_interval_sec = resolve_power_sample_interval_sec(
            requested_interval_sec=power_sample_interval_sec,
            estimated_timed_window_sec=estimated_window_sec,
            min_interval_sec=0.05,
        )
        with _host_power_monitor(
            runtime_device,
            power_backend=power_backend,
            sample_interval_sec=sample_interval_sec,
            estimated_timed_window_sec=estimated_window_sec,
        ) as power_monitor:
            started = time.perf_counter()
            for _ in range(power_probe_runs):
                forward_once()
                time.sleep(0)
            elapsed_sec = time.perf_counter() - started
        power_stats = power_monitor.stats(elapsed_sec)
        if power_stats.get("avg_power_w") is not None:
            power_stats["energy_j"] = float(power_stats["avg_power_w"]) * float(
                summary["timed_total_sec"]
            )

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
        "avg_power_w": power_stats.get("avg_power_w"),
        "max_power_w": power_stats.get("max_power_w"),
        "energy_j": power_stats.get("energy_j"),
        "power_sample_count": power_stats.get("power_sample_count"),
        "effective_gflops_per_sec_per_watt": effective_gflops_per_sec_per_watt(
            effective_gflops,
            _optional_float(power_stats.get("avg_power_w")),
        ),
        "process_model": "in_process",
        "host_device": str(runtime_device),
        **validation,
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


def _comparison_row(
    *,
    group: ReferenceGroup,
    metric: str,
    igpu_value: float | None,
    reference_values: dict[str, float | None],
) -> dict[str, object]:
    row: dict[str, object] = {
        "study_case_id": group.study_case_id,
        "seq_len": group.seq_len,
        "metric": metric,
        "igpu": igpu_value,
    }
    for execution_mode in REFERENCE_EXECUTION_MODES:
        row[execution_mode] = reference_values.get(execution_mode)
    return row


def _resolve_plot_path(path: Path) -> Path:
    if path.suffix.lower() != SUPPORTED_PLOT_SUFFIX:
        raise ValueError(
            f"Plot output must use the {SUPPORTED_PLOT_SUFFIX} suffix: {path}"
        )
    return path


def _format_study_case_label(study_case_id: str) -> str:
    return PLOT_FAMILY_LABELS.get(
        study_case_id,
        " ".join(part.capitalize() for part in study_case_id.split("_")),
    )


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
    if variant == "slides" and len(study_case_ids) == 3:
        fig, axes_grid = plt.subplots(2, 2, figsize=(22, 12), sharey=True)
        axes = [
            axes_grid[0][0],
            axes_grid[0][1],
            axes_grid[1][0],
        ]
        legend_ax = axes_grid[1][1]
        legend_ax.set_axis_off()
    else:
        fig, axes_obj = plt.subplots(
            1,
            max(1, len(study_case_ids)),
            figsize=(8 * max(1, len(study_case_ids)), 8),
            sharey=True,
        )
        axes = [axes_obj] if len(study_case_ids) == 1 else list(axes_obj)
        legend_ax = None

    for ax, study_case_id in zip(axes, study_case_ids, strict=True):
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
            for series_name, _, _ in PLOT_SERIES
        }

        x_positions = list(range(len(seq_lens)))
        group_width = 0.82
        bar_width = group_width / float(len(PLOT_SERIES))

        for series_index, (series_name, _, color) in enumerate(PLOT_SERIES):
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
        ax.set_xlabel("Context Length (tokens)", fontsize=15)
        ax.set_ylabel(y_axis_label, fontsize=15)
        ax.set_title(
            _format_study_case_label(study_case_id),
            loc="left",
            fontsize=18 if variant == "standard" else 20,
            pad=12,
        )
        ax.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.8)
        ax.tick_params(axis="both", labelsize=12 if variant == "standard" else 14)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in PLOT_SERIES
    ]
    if legend_ax is not None:
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=1,
            frameon=False,
            fontsize=16,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(PLOT_SERIES),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=13 if variant == "standard" else 16,
        )
    fig.suptitle(
        title,
        fontsize=24 if variant == "standard" else 30,
        fontweight="bold",
        y=0.98 if variant == "standard" else 0.99,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
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
        y_axis_label="GFLOP / sec",
    )
    write_metric_plot(
        _resolve_plot_path(effective_gflops_per_watt_plot_path),
        rows,
        metric="effective_gflops_per_sec_per_watt",
        title="Effective Throughput/W Comparison",
        y_axis_label="GFLOP / sec / W",
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
) -> list[dict[str, object]]:
    resolved_warmup_runs, resolved_runs_per_sample = resolve_sampling(
        group,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
    )

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

    reference_effective_gflops = _reference_metric_means(
        group,
        "effective_gflops_per_sec",
    )
    reference_effective_gflops_per_watt = _reference_metric_means(
        group,
        "effective_gflops_per_sec_per_watt",
    )
    return [
        _comparison_row(
            group=group,
            metric="effective_gflops_per_sec",
            igpu_value=_host_metric_value(
                benchmark_results["igpu"],
                "effective_gflops_per_sec",
            ),
            reference_values=reference_effective_gflops,
        ),
        _comparison_row(
            group=group,
            metric="effective_gflops_per_sec_per_watt",
            igpu_value=_host_metric_value(
                benchmark_results["igpu"],
                "effective_gflops_per_sec_per_watt",
            ),
            reference_values=reference_effective_gflops_per_watt,
        ),
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

    rows: list[dict[str, object]] = []
    for group in groups:
        rows.extend(
            build_rows_for_group(
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
            )
        )

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
