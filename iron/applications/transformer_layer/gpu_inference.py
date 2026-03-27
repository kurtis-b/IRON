#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time

import torch

from iron.applications.transformer_layer.benchmark_common import (
    summarize_latency_measurements,
    write_results_csv,
)
from iron.applications.transformer_layer.benchmark_power import (
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
)
from iron.applications.transformer_layer.gpu_power import create_rocm_power_monitor
from iron.applications.transformer_layer.measurement_log import (
    MeasurementAuditLogger,
    capture_timed_call,
    default_measurement_log_path,
)
from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
    flops_per_joule,
    gflops_per_joule,
    operational_intensity,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)

SUPPORTED_POWER_BACKENDS = ("none", "rocm-smi")

GPU_INFERENCE_START_POINT = (
    "Immediately before launching the GPU forward pass under torch.no_grad()."
)
GPU_INFERENCE_END_POINT = (
    "Immediately after the forward pass returned and the benchmark harness completed "
    "device synchronization when required."
)
GPU_POWER_PROBE_START_POINT = (
    "Immediately before entering the separate same-workload GPU power probe loop."
)
GPU_POWER_PROBE_END_POINT = (
    "Immediately after the separate same-workload GPU power probe loop completed."
)


def resolve_amd_gpu_device(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AMD GPU comparison requires a ROCm-enabled torch build with a visible GPU"
        )
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(
            f"AMD GPU comparison expects a ROCm/CUDA device string (got {device})"
        )
    return resolved


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    current = torch.cuda.current_device()
    return int(current)


def benchmark_gpu_layer(
    *,
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    output_csv: str,
    seed: int,
    device: torch.device | None = None,
    device_name: str = "cuda:0",
    power_backend: str = "none",
    power_sample_interval_sec: float = 0.2,
    study_id: str = "gpu_compare",
    study_case_id: str | None = None,
    study_case_label: str | None = None,
    execution_mode: str = "amd_gpu_reference",
    pattern_label: str = "amd_gpu_reference",
    enable_measurement_log: bool = False,
    measurement_log_path: str | None = None,
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    runtime_device = (
        device if device is not None else resolve_amd_gpu_device(device_name)
    )
    resolved_measurement_log_path = measurement_log_path
    if resolved_measurement_log_path is None and enable_measurement_log:
        resolved_measurement_log_path = default_measurement_log_path(output_csv)
    audit_logger = MeasurementAuditLogger(
        resolved_measurement_log_path,
        context={
            "study_id": study_id,
            "study_case_id": study_case_id,
            "study_case_label": study_case_label,
            "backend": "gpu",
            "execution_mode": execution_mode,
            "pattern_label": pattern_label,
            "input_boundary": spec.input_boundary,
            "seq_len": spec.seq_len,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_attention_heads": spec.num_attention_heads,
            "attention_head_size": spec.attention_head_size,
            "measurement_log_path": resolved_measurement_log_path,
        },
    )
    audit_logger.append_event(
        event_kind="benchmark_session_started",
        phase="session",
        details={
            "warmup_runs": warmup_runs,
            "runs_per_sample": runs_per_sample,
            "seed": seed,
            "device": str(runtime_device),
            "power_backend": power_backend,
            "power_sample_interval_sec": power_sample_interval_sec,
            "output_csv": output_csv,
        },
    )

    model = ReferenceTransformerLayer(spec).to(runtime_device)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    weights = {name: tensor.to(runtime_device) for name, tensor in weights.items()}
    model.assign_weights(weights)
    model.eval()

    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1).to(runtime_device)

    def _run_once():
        with torch.no_grad():
            model(layer_inputs)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)

    try:
        for warmup_index in range(1, warmup_runs + 1):
            _, timing = capture_timed_call(_run_once)
            audit_logger.append_event(
                event_kind="timing_measurement",
                phase="warmup",
                measurement_name="end_to_end_inference_latency",
                run_index=warmup_index,
                start_time_utc=timing["start_time_utc"],
                end_time_utc=timing["end_time_utc"],
                elapsed_sec=timing["elapsed_sec"],
                start_perf_counter_sec=timing["start_perf_counter_sec"],
                end_perf_counter_sec=timing["end_perf_counter_sec"],
                start_point=GPU_INFERENCE_START_POINT,
                end_point=GPU_INFERENCE_END_POINT,
                measurement_value=timing["elapsed_sec"],
                units="sec",
            )

        latencies = []
        latency_measurements_sec: list[float] = []
        for run_index in range(1, runs_per_sample + 1):
            _, timing = capture_timed_call(_run_once)
            latencies.append(timing["elapsed_sec"])
            latency_measurements_sec.append(timing["elapsed_sec"])
            audit_logger.append_event(
                event_kind="timing_measurement",
                phase="timed",
                measurement_name="end_to_end_inference_latency",
                run_index=run_index,
                start_time_utc=timing["start_time_utc"],
                end_time_utc=timing["end_time_utc"],
                elapsed_sec=timing["elapsed_sec"],
                start_perf_counter_sec=timing["start_perf_counter_sec"],
                end_perf_counter_sec=timing["end_perf_counter_sec"],
                start_point=GPU_INFERENCE_START_POINT,
                end_point=GPU_INFERENCE_END_POINT,
                measurement_value=timing["elapsed_sec"],
                units="sec",
            )

        summary = summarize_latency_measurements(latencies)
        estimated_flops = estimate_layer_flops(spec)
        estimated_bytes = estimate_layer_bytes(spec)
        throughput_flops_per_sec = (
            estimated_flops
            * summary["measured_inference_count"]
            / summary["timed_total_sec"]
        )
        avg_iteration_sec = (
            summary["timed_total_sec"] / summary["measured_inference_count"]
        )
        power_probe_runs = resolve_power_probe_runs(
            avg_iteration_sec=avg_iteration_sec,
            baseline_runs=runs_per_sample,
            min_measurement_duration_sec=0.25,
        )
        power_probe_window_sec = avg_iteration_sec * power_probe_runs
        effective_power_sample_interval_sec = resolve_power_sample_interval_sec(
            requested_interval_sec=power_sample_interval_sec,
            estimated_timed_window_sec=power_probe_window_sec,
            min_interval_sec=0.05,
        )
        power_phase_timing = None
        if power_backend == "none":
            power_stats = empty_power_stats()
            audit_logger.append_event(
                event_kind="power_measurement_skipped",
                phase="power",
                measurement_name="power_probe",
                details={"reason": "power_backend=none"},
            )
        else:
            with create_rocm_power_monitor(
                power_backend=power_backend,
                device_index=(
                    0
                    if runtime_device.type != "cuda"
                    else _device_index(runtime_device)
                ),
                sample_interval_sec=effective_power_sample_interval_sec,
            ) as power_monitor:
                for run_index in range(1, power_probe_runs + 1):
                    _, timing = capture_timed_call(_run_once)
                    if power_phase_timing is None:
                        power_phase_timing = dict(timing)
                    else:
                        power_phase_timing["end_time_utc"] = timing["end_time_utc"]
                        power_phase_timing["end_perf_counter_sec"] = timing[
                            "end_perf_counter_sec"
                        ]
                        power_phase_timing["elapsed_sec"] = float(
                            power_phase_timing["end_perf_counter_sec"]
                        ) - float(power_phase_timing["start_perf_counter_sec"])
                    audit_logger.append_event(
                        event_kind="timing_measurement",
                        phase="power_probe",
                        measurement_name="end_to_end_inference_latency",
                        run_index=run_index,
                        start_time_utc=timing["start_time_utc"],
                        end_time_utc=timing["end_time_utc"],
                        elapsed_sec=timing["elapsed_sec"],
                        start_perf_counter_sec=timing["start_perf_counter_sec"],
                        end_perf_counter_sec=timing["end_perf_counter_sec"],
                        start_point=GPU_INFERENCE_START_POINT,
                        end_point=GPU_INFERENCE_END_POINT,
                        measurement_value=timing["elapsed_sec"],
                        units="sec",
                    )
                    time.sleep(0)
            if power_phase_timing is None:
                power_phase_timing = {
                    "start_time_utc": None,
                    "end_time_utc": None,
                    "elapsed_sec": 0.0,
                    "start_perf_counter_sec": None,
                    "end_perf_counter_sec": None,
                }
            power_stats = power_monitor.stats(float(power_phase_timing["elapsed_sec"]))
            if power_stats.get("avg_power_w") is not None:
                power_stats["energy_j"] = (
                    float(power_stats["avg_power_w"]) * summary["timed_total_sec"]
                )
            measurement_details = (
                power_monitor.measurement_details()
                if hasattr(power_monitor, "measurement_details")
                else {}
            )
            probe = measurement_details.get("probe") if measurement_details else None
            audit_logger.append_event(
                event_kind="power_measurement",
                phase="power_probe",
                measurement_name="gpu_device_power",
                start_time_utc=power_phase_timing["start_time_utc"],
                end_time_utc=power_phase_timing["end_time_utc"],
                elapsed_sec=power_phase_timing["elapsed_sec"],
                start_perf_counter_sec=power_phase_timing["start_perf_counter_sec"],
                end_perf_counter_sec=power_phase_timing["end_perf_counter_sec"],
                start_point=GPU_POWER_PROBE_START_POINT,
                end_point=GPU_POWER_PROBE_END_POINT,
                measurement_value=power_stats.get("avg_power_w"),
                units="W",
                details={
                    "power_backend": power_backend,
                    "effective_power_sample_interval_sec": effective_power_sample_interval_sec,
                    "power_probe_runs": power_probe_runs,
                    "max_power_w": power_stats.get("max_power_w"),
                    "energy_j": power_stats.get("energy_j"),
                    "power_sample_count": power_stats.get("power_sample_count"),
                    "sample_count": (
                        probe.get("sample_count") if isinstance(probe, dict) else None
                    ),
                    "sample_interval_sec": (
                        probe.get("sample_interval_sec")
                        if isinstance(probe, dict)
                        else None
                    ),
                    "samples": (
                        probe.get("samples", []) if isinstance(probe, dict) else []
                    ),
                },
            )
        if power_stats.get("power_backend") is None:
            power_stats["power_backend"] = power_backend
        energy_efficiency_flops_per_joule = flops_per_joule(
            throughput_flops_per_sec=throughput_flops_per_sec,
            avg_power_w=power_stats.get("avg_power_w"),
        )

        row = {
            "study_id": study_id,
            "study_case_id": study_case_id,
            "study_case_label": study_case_label,
            "backend": "gpu",
            "execution_mode": execution_mode,
            "pattern_label": pattern_label,
            "input_boundary": spec.input_boundary,
            "seq_len": spec.seq_len,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_attention_heads": spec.num_attention_heads,
            "attention_head_size": spec.attention_head_size,
            "batch_size": spec.batch_size,
            "dtype": spec.dtype,
            "use_bias": spec.use_bias,
            "weights_source": spec.weights_source,
            "source_model_name": spec.source_model_name,
            "source_layer_index": spec.source_layer_index,
            "warmup_runs": warmup_runs,
            "runs_per_sample": runs_per_sample,
            "throughput_flops_per_sec": throughput_flops_per_sec,
            "estimated_flops_per_inference": estimated_flops,
            "estimated_bytes_per_inference": estimated_bytes,
            "operational_intensity_flops_per_byte": operational_intensity(
                estimated_flops,
                estimated_bytes,
            ),
            "backend_peak_ops_per_sec": None,
            "roofline_bound_ops_per_sec": None,
            "backend_pct_of_peak": None,
            "roofline_pct": None,
            "flops_per_joule": energy_efficiency_flops_per_joule,
            "gflops_per_joule": gflops_per_joule(
                throughput_flops_per_sec=throughput_flops_per_sec,
                avg_power_w=power_stats.get("avg_power_w"),
            ),
            "run_status": "completed",
            "failure_component": None,
            "failure_category": None,
            "failure_message": None,
            "gpu_device": str(runtime_device),
            "gpu_power_backend": power_backend,
            "measurement_log_path": resolved_measurement_log_path,
            "measurement_session_id": (
                audit_logger.session_id
                if resolved_measurement_log_path is not None
                else None
            ),
            **summary,
            **power_stats,
        }
        audit_logger.append_event(
            event_kind="benchmark_summary",
            phase="summary",
            measurement_name="benchmark_row_summary",
            details={
                "latency_measurements_sec": latency_measurements_sec,
                "summary_row": row,
            },
        )
        audit_logger.append_event(
            event_kind="benchmark_session_completed",
            phase="session",
            details={
                "measured_inference_count": summary["measured_inference_count"],
                "timed_total_sec": summary["timed_total_sec"],
                "avg_latency_ms": summary["avg_latency_ms"],
            },
        )
        if write_immediately:
            write_results_csv(output_csv, [row])
        return [row]
    except Exception as exc:
        audit_logger.append_event(
            event_kind="benchmark_session_failed",
            phase="session",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise
    finally:
        audit_logger.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a single transformer layer on an isolated AMD GPU path."
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument(
        "--input-boundary",
        choices=("post_projection", "hidden_states"),
        default="post_projection",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs-per-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--power-backend",
        choices=SUPPORTED_POWER_BACKENDS,
        default="none",
    )
    parser.add_argument(
        "--power-sample-interval-sec",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--enable-measurement-log",
        action="store_true",
        help="Write a JSONL sidecar log with per-run timing and power audit events.",
    )
    parser.add_argument(
        "--measurement-log-path",
        default=None,
        help="Optional JSONL sidecar path for detailed timing and power audit events.",
    )
    parser.add_argument("--output-csv", default="transformer_layer_amd_gpu_latest.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = TransformerLayerSpec(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        seq_len=args.seq_len,
        input_boundary=args.input_boundary,
        use_bias=False,
        weights_source="synthetic",
    )
    benchmark_gpu_layer(
        spec=spec,
        warmup_runs=args.warmup_runs,
        runs_per_sample=args.runs_per_sample,
        output_csv=args.output_csv,
        seed=args.seed,
        device_name=args.device,
        power_backend=args.power_backend,
        power_sample_interval_sec=args.power_sample_interval_sec,
        enable_measurement_log=args.enable_measurement_log,
        measurement_log_path=args.measurement_log_path,
    )


if __name__ == "__main__":
    main()
