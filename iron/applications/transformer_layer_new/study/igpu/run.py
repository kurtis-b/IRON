#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import threading
import time

import torch
import torch.nn.functional as F

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
SUPPORTED_POWER_BACKENDS: tuple[str, ...] = ("none", "rocm-smi")
COMPARISON_COLUMNS: tuple[str, ...] = ("igpu",) + tuple(REFERENCE_EXECUTION_MODES)
RESULTS_CSV_FIELDNAMES = (
    "study_case_id",
    "seq_len",
    "metric",
    *COMPARISON_COLUMNS,
)
PLOT_SERIES = (
    ("igpu", "iGPU", "#d55e00"),
    ("dataflow", "dataflow", "#0072b2"),
    ("runlist", "runlist", "#009e73"),
    ("offload", "offload", "#c44e52"),
)
SUPPORTED_PLOT_SUFFIX = ".svg"
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "igpu" / "results.csv"


def default_tps_plot_path(output_path: Path) -> Path:
    return output_path.with_name("tps_comparison.svg")


def default_tps_per_watt_plot_path(output_path: Path) -> Path:
    return output_path.with_name("tps_per_watt_comparison.svg")


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
    head_dim = hidden_size // num_attention_heads

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


def resolve_amd_igpu_device(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "iGPU study requires a ROCm-enabled torch build with a visible GPU"
        )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(
            f"AMD iGPU study expects a ROCm/CUDA device string (got {device_name})"
        )
    return device


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 128:
        return (10, 48)
    if seq_len <= 512:
        return (8, 36)
    if seq_len <= 2048:
        return (5, 24)
    if seq_len <= 4096:
        return (4, 14)
    return (3, 8)


def _positive_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return int(value)


def resolve_sampling(
    group: ReferenceGroup,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
) -> tuple[int, int]:
    scheduled_warmup_runs, scheduled_runs_per_sample = iteration_schedule(group.seq_len)
    resolved_warmup_runs = (
        _positive_or_none(group.warmup_runs)
        if _positive_or_none(group.warmup_runs) is not None
        else scheduled_warmup_runs
    )
    resolved_runs_per_sample = (
        _positive_or_none(group.runs_per_sample)
        if _positive_or_none(group.runs_per_sample) is not None
        else scheduled_runs_per_sample
    )
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


def _tokens_per_sec(seq_len: int, avg_latency_ms: float | None) -> float | None:
    if avg_latency_ms is None or avg_latency_ms <= 0:
        return None
    return float(seq_len) / (avg_latency_ms / 1000.0)


def _tokens_per_sec_per_watt(
    tokens_per_sec: float | None,
    avg_power_w: float | None,
) -> float | None:
    if tokens_per_sec is None or avg_power_w is None or avg_power_w <= 0:
        return None
    return tokens_per_sec / avg_power_w


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


def benchmark_igpu_group(
    group: ReferenceGroup,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    device_name: str,
    power_backend: str,
    power_sample_interval_sec: float,
) -> dict[str, object]:
    runtime_device = resolve_amd_igpu_device(device_name)
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
        with create_rocm_power_monitor(
            power_backend=power_backend,
            device_index=_device_index(runtime_device),
            sample_interval_sec=sample_interval_sec,
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
    tokens_per_sec = _tokens_per_sec(
        group.seq_len, _optional_float(summary["avg_latency_ms"])
    )
    return {
        **summary,
        "tokens_per_sec": tokens_per_sec,
        "power_backend": power_stats.get("power_backend"),
        "avg_power_w": power_stats.get("avg_power_w"),
        "max_power_w": power_stats.get("max_power_w"),
        "energy_j": power_stats.get("energy_j"),
        "power_sample_count": power_stats.get("power_sample_count"),
        "tokens_per_sec_per_watt": _tokens_per_sec_per_watt(
            tokens_per_sec,
            _optional_float(power_stats.get("avg_power_w")),
        ),
        "process_model": "in_process",
        "gpu_device": str(runtime_device),
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


def _igpu_metric_value(
    benchmark_result: dict[str, object],
    metric_field: str,
) -> float | None:
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


def _svg_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _resolve_plot_path(path: Path) -> Path:
    if path.suffix.lower() != SUPPORTED_PLOT_SUFFIX:
        raise ValueError(
            f"Plot output must use the {SUPPORTED_PLOT_SUFFIX} suffix: {path}"
        )
    return path


def _format_metric_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000.0:
        return f"{value:,.0f}"
    if magnitude >= 100.0:
        return f"{value:,.1f}"
    if magnitude >= 10.0:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _value_to_y_position(
    value: float,
    *,
    plot_top: float,
    plot_height: float,
    y_min: float,
    y_max: float,
) -> float:
    denominator = y_max - y_min
    normalized = 0.5 if denominator <= 0 else (value - y_min) / denominator
    return plot_top + plot_height - (normalized * plot_height)


def write_metric_plot(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    y_axis_label: str,
) -> None:
    metric_rows = [row for row in rows if str(row.get("metric") or "") == metric]
    study_ids = sorted(
        {
            str(row.get("study_case_id") or "")
            for row in metric_rows
            if str(row.get("study_case_id") or "")
        }
    )
    width = 1080
    title_height = 36
    legend_height = 34
    subplot_height = 250
    plot_left = 90
    plot_right = 1040
    plot_width = plot_right - plot_left
    plot_height = 170
    plot_title_offset = 28
    top_padding = 18
    subplot_count = max(1, len(study_ids))
    height = (
        top_padding
        + title_height
        + legend_height
        + (subplot_count * subplot_height)
        + 18
    )

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 14px; font-weight: 600; }",
        ".axis { stroke: #334155; stroke-width: 1.5; }",
        ".grid { stroke: #cbd5e1; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #475569; }",
        ".legend { font-size: 13px; font-weight: 600; }",
        ".bar { opacity: 0.95; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="white" />',
        (
            f'<text class="title" x="{width / 2.0}" y="{top_padding + 22}" '
            f'text-anchor="middle">{_svg_escape(title)}</text>'
        ),
    ]

    legend_y = top_padding + title_height + 4
    legend_x = 120
    for _, label, color in PLOT_SERIES:
        svg_lines.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="22" height="14" '
            f'fill="{color}" class="bar" />'
        )
        svg_lines.append(
            f'<text class="legend" x="{legend_x + 34}" y="{legend_y + 5}">'
            f"{_svg_escape(label)}</text>"
        )
        legend_x += 150

    if not metric_rows:
        svg_lines.append(
            (
                f'<text class="subtitle" x="{width / 2.0}" '
                f'y="{top_padding + title_height + legend_height + 80}" '
                'text-anchor="middle">No data available</text>'
            )
        )
        svg_lines.append("</svg>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(svg_lines), encoding="utf-8")
        return

    all_values = [
        _optional_float(row.get(series_name))
        for row in metric_rows
        for series_name, _, _ in PLOT_SERIES
    ]
    plotted_values = [value for value in all_values if value is not None]
    y_min = 0.0
    y_max = 1.0 if not plotted_values else max(plotted_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_max *= 1.08

    for study_index, study_case_id in enumerate(study_ids):
        study_rows = sorted(
            [
                row
                for row in metric_rows
                if str(row.get("study_case_id") or "") == study_case_id
            ],
            key=lambda row: int(row.get("seq_len") or 0),
        )
        section_top = (
            top_padding + title_height + legend_height + (study_index * subplot_height)
        )
        plot_top = section_top + plot_title_offset
        plot_bottom = plot_top + plot_height
        svg_lines.append(
            f'<text class="subtitle" x="{plot_left}" y="{section_top + 18}">'
            f"{_svg_escape(study_case_id)}</text>"
        )
        svg_lines.append(
            f'<text class="tick" x="24" y="{plot_top + (plot_height / 2.0)}" '
            f'transform="rotate(-90 24 {plot_top + (plot_height / 2.0)})">'
            f"{_svg_escape(y_axis_label)}</text>"
        )
        svg_lines.append(
            f'<line class="axis" x1="{plot_left}" y1="{plot_top}" '
            f'x2="{plot_left}" y2="{plot_bottom}" />'
        )
        svg_lines.append(
            f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" '
            f'x2="{plot_right}" y2="{plot_bottom}" />'
        )

        for tick_index in range(5):
            ratio = tick_index / 4.0
            y_position = plot_bottom - (ratio * plot_height)
            tick_value = y_min + ((y_max - y_min) * ratio)
            svg_lines.append(
                f'<line class="grid" x1="{plot_left}" y1="{y_position}" '
                f'x2="{plot_right}" y2="{y_position}" />'
            )
            svg_lines.append(
                f'<text class="tick" x="{plot_left - 10}" y="{y_position + 4}" '
                f'text-anchor="end">{_svg_escape(_format_metric_value(tick_value))}</text>'
            )

        seq_lens = [int(row.get("seq_len") or 0) for row in study_rows]
        group_slot_width = plot_width / float(max(1, len(seq_lens)))
        cluster_width = min(group_slot_width * 0.82, 104.0)
        bar_gap = 4.0
        cluster_gap_total = bar_gap * float(len(PLOT_SERIES) - 1)
        bar_width = max(
            8.0,
            (cluster_width - cluster_gap_total) / float(len(PLOT_SERIES)),
        )
        cluster_total_width = (bar_width * float(len(PLOT_SERIES))) + cluster_gap_total

        for row_index, (study_row, seq_len) in enumerate(zip(study_rows, seq_lens)):
            slot_left = plot_left + (row_index * group_slot_width)
            slot_center = slot_left + (group_slot_width / 2.0)
            cluster_left = slot_center - (cluster_total_width / 2.0)
            svg_lines.append(
                f'<line class="grid" x1="{slot_center}" y1="{plot_top}" '
                f'x2="{slot_center}" y2="{plot_bottom}" />'
            )
            svg_lines.append(
                f'<text class="tick" x="{slot_center}" y="{plot_bottom + 18}" '
                f'text-anchor="middle">{_svg_escape(seq_len)}</text>'
            )
            for series_index, (series_name, _, color) in enumerate(PLOT_SERIES):
                value = _optional_float(study_row.get(series_name))
                if value is None:
                    continue
                bar_left = cluster_left + (series_index * (bar_width + bar_gap))
                bar_top = _value_to_y_position(
                    value,
                    plot_top=plot_top,
                    plot_height=plot_height,
                    y_min=y_min,
                    y_max=y_max,
                )
                bar_height = max(0.0, plot_bottom - bar_top)
                svg_lines.append(
                    f'<rect class="bar" x="{bar_left:.2f}" y="{bar_top:.2f}" '
                    f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                    f'fill="{color}" />'
                )

        svg_lines.append(
            f'<text class="tick" x="{plot_left + (plot_width / 2.0)}" '
            f'y="{plot_bottom + 40}" text-anchor="middle">Sequence Length</text>'
        )

    svg_lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_plots(
    rows: list[dict[str, object]],
    *,
    tps_plot_path: Path,
    tps_per_watt_plot_path: Path,
) -> None:
    write_metric_plot(
        _resolve_plot_path(tps_plot_path),
        rows,
        metric="tps",
        title="TPS Comparison",
        y_axis_label="Tokens / sec",
    )
    write_metric_plot(
        _resolve_plot_path(tps_per_watt_plot_path),
        rows,
        metric="tps_per_watt",
        title="TPS/W Comparison",
        y_axis_label="Tokens / sec / W",
    )


def build_rows_for_group(
    group: ReferenceGroup,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    device_name: str,
    power_backend: str,
    power_sample_interval_sec: float,
) -> list[dict[str, object]]:
    resolved_warmup_runs, resolved_runs_per_sample = resolve_sampling(
        group,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
    )
    try:
        benchmark_result = benchmark_igpu_group(
            group,
            warmup_runs=resolved_warmup_runs,
            runs_per_sample=resolved_runs_per_sample,
            seed=seed,
            device_name=device_name,
            power_backend=power_backend,
            power_sample_interval_sec=power_sample_interval_sec,
        )
    except Exception as exc:
        LOGGER.warning(
            "iGPU benchmark unavailable for %s seq_len=%s: %s",
            group.study_case_id,
            group.seq_len,
            exc,
        )
        benchmark_result = {
            "run_status": "unsupported",
            "failure_message": str(exc),
        }
    else:
        if str(benchmark_result.get("run_status") or "passed") != "passed":
            LOGGER.warning(
                "iGPU benchmark did not pass for %s seq_len=%s: %s",
                group.study_case_id,
                group.seq_len,
                benchmark_result.get("failure_message")
                or benchmark_result["run_status"],
            )

    reference_tps = _reference_metric_means(group, "tokens_per_sec")
    reference_tps_per_watt = _reference_metric_means(
        group,
        "tokens_per_sec_per_watt",
    )
    return [
        _comparison_row(
            group=group,
            metric="tps",
            igpu_value=_igpu_metric_value(benchmark_result, "tokens_per_sec"),
            reference_values=reference_tps,
        ),
        _comparison_row(
            group=group,
            metric="tps_per_watt",
            igpu_value=_igpu_metric_value(
                benchmark_result,
                "tokens_per_sec_per_watt",
            ),
            reference_values=reference_tps_per_watt,
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
        description="Benchmark AMD iGPU and write comparison CSV and plots"
    )
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--power-backend",
        choices=list(SUPPORTED_POWER_BACKENDS),
        default="rocm-smi",
    )
    parser.add_argument("--power-sample-interval-sec", type=float, default=0.2)
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=default_reference_results_path(),
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--tps-plot", type=Path, default=None)
    parser.add_argument("--tps-per-watt-plot", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    reference_input = args.reference_input.expanduser()
    output_path = args.output.expanduser()
    tps_plot_path = (
        default_tps_plot_path(output_path)
        if args.tps_plot is None
        else _resolve_plot_path(args.tps_plot.expanduser())
    )
    tps_per_watt_plot_path = (
        default_tps_per_watt_plot_path(output_path)
        if args.tps_per_watt_plot is None
        else _resolve_plot_path(args.tps_per_watt_plot.expanduser())
    )

    if not reference_input.exists():
        LOGGER.warning(
            "Reference end_to_end results not found at %s; writing empty iGPU outputs",
            reference_input,
        )
        write_rows(output_path, [])
        write_plots(
            [],
            tps_plot_path=tps_plot_path,
            tps_per_watt_plot_path=tps_per_watt_plot_path,
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
            "writing empty iGPU outputs",
            reference_input,
            args.family,
            args.seq_len,
        )
        write_rows(output_path, [])
        write_plots(
            [],
            tps_plot_path=tps_plot_path,
            tps_per_watt_plot_path=tps_per_watt_plot_path,
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
                device_name=str(args.device),
                power_backend=str(args.power_backend),
                power_sample_interval_sec=float(args.power_sample_interval_sec),
            )
        )

    write_rows(output_path, rows)
    write_plots(
        rows,
        tps_plot_path=tps_plot_path,
        tps_per_watt_plot_path=tps_per_watt_plot_path,
    )
    LOGGER.info("Wrote %d iGPU comparison rows to %s", len(rows), output_path)
    LOGGER.info(
        "Wrote comparison plots to %s and %s",
        tps_plot_path,
        tps_per_watt_plot_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
