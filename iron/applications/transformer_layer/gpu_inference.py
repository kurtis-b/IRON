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
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    runtime_device = (
        device if device is not None else resolve_amd_gpu_device(device_name)
    )

    model = ReferenceTransformerLayer(spec).to(runtime_device)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    weights = {name: tensor.to(runtime_device) for name, tensor in weights.items()}
    model.assign_weights(weights)
    model.eval()

    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1).to(runtime_device)

    for _ in range(warmup_runs):
        with torch.no_grad():
            model(layer_inputs)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)

    latencies = []
    for _ in range(runs_per_sample):
        start = time.perf_counter()
        with torch.no_grad():
            model(layer_inputs)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)
        latencies.append(time.perf_counter() - start)

    summary = summarize_latency_measurements(latencies)
    estimated_flops = estimate_layer_flops(spec)
    estimated_bytes = estimate_layer_bytes(spec)
    throughput_flops_per_sec = (
        estimated_flops
        * summary["measured_inference_count"]
        / summary["timed_total_sec"]
    )
    avg_iteration_sec = summary["timed_total_sec"] / summary["measured_inference_count"]
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
    if power_backend == "none":
        power_stats = empty_power_stats()
    else:
        with create_rocm_power_monitor(
            power_backend=power_backend,
            device_index=(
                0 if runtime_device.type != "cuda" else _device_index(runtime_device)
            ),
            sample_interval_sec=effective_power_sample_interval_sec,
        ) as power_monitor:
            power_phase_start = time.perf_counter()
            for _ in range(power_probe_runs):
                with torch.no_grad():
                    model(layer_inputs)
                if runtime_device.type == "cuda":
                    torch.cuda.synchronize(runtime_device)
                time.sleep(0)
            power_phase_elapsed_sec = time.perf_counter() - power_phase_start
        power_stats = power_monitor.stats(power_phase_elapsed_sec)
        if power_stats.get("avg_power_w") is not None:
            power_stats["energy_j"] = (
                float(power_stats["avg_power_w"]) * summary["timed_total_sec"]
            )
    if power_stats.get("power_backend") is None:
        power_stats["power_backend"] = power_backend
    energy_efficiency_flops_per_joule = flops_per_joule(
        throughput_flops_per_sec=throughput_flops_per_sec,
        avg_power_w=power_stats.get("avg_power_w"),
    )

    row = {
        "study_id": "gpu_compare",
        "backend": "gpu",
        "execution_mode": "amd_gpu_reference",
        "pattern_label": "amd_gpu_reference",
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
        **summary,
        **power_stats,
    }
    if write_immediately:
        write_results_csv(output_csv, [row])
    return [row]


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
    )


if __name__ == "__main__":
    main()
