#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from iron.applications.transformer_layer.benchmark_common import (
    summarize_latency_measurements,
    write_results_csv,
)
from iron.applications.transformer_layer.benchmark_power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
)
from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
    flops_per_joule,
    gflops_per_joule,
    operational_intensity,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pattern_encoder_pipeline import (
    EncoderPipelinePattern,
)
from iron.applications.transformer_layer.src.pattern_gemm_only import (
    GemmOnlyPattern,
)
from iron.applications.transformer_layer.src.pattern_operator_runlist import (
    OperatorRunlistPattern,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)

SUPPORTED_EXECUTION_MODES = (
    "encoder_pipeline",
    "gemm_only",
    "operator_runlist",
)


def _run_pattern_with_stage_timings(pattern, layer_inputs):
    if hasattr(pattern, "forward_with_stage_timings"):
        return pattern.forward_with_stage_timings(layer_inputs)
    start = time.perf_counter()
    output = pattern(layer_inputs)
    end = time.perf_counter()
    return output, {"pattern_sec": end - start}


def _average_stage_timings_ms(
    stage_sums_sec: dict[str, float], measured_inference_count: int
) -> dict[str, float]:
    if measured_inference_count <= 0:
        return {}
    row = {}
    for stage_name, total_sec in stage_sums_sec.items():
        if not stage_name.endswith("_sec"):
            continue
        metric_name = f"avg_{stage_name.removesuffix('_sec')}_latency_ms"
        row[metric_name] = (total_sec / measured_inference_count) * 1000.0
    return row


def _pattern_metadata(pattern) -> dict[str, object]:
    if hasattr(pattern, "get_benchmark_metadata"):
        return pattern.get_benchmark_metadata()
    return {}


def build_pattern(execution_mode: str, spec: TransformerLayerSpec):
    if execution_mode == "encoder_pipeline":
        return EncoderPipelinePattern(spec)
    if execution_mode == "gemm_only":
        return GemmOnlyPattern(spec)
    if execution_mode == "operator_runlist":
        return OperatorRunlistPattern(spec)
    raise ValueError(f"Unsupported execution_mode: {execution_mode}")


def _cleanup_pattern_runtime(pattern) -> None:
    context = getattr(pattern, "context", None)
    if context is None or not hasattr(context, "reset_runtime"):
        return
    try:
        context.reset_runtime()
    except Exception:
        logging.exception("Failed to reset AIE runtime after benchmark run")


def _benchmark_operator_runlist_isolated(
    *,
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    output_csv: str,
    seed: int,
    power_backend: str,
    power_sample_interval_sec: float,
    quiescent_baseline_duration_sec: float,
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    request = {
        "spec": spec.to_dict(),
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "seed": seed,
        "power_backend": power_backend,
        "power_sample_interval_sec": power_sample_interval_sec,
        "quiescent_baseline_duration_sec": quiescent_baseline_duration_sec,
    }
    with tempfile.TemporaryDirectory(
        prefix="transformer_layer_operator_runlist_"
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        request_path = temp_dir_path / "request.json"
        response_path = temp_dir_path / "response.json"
        request_path.write_text(json.dumps(request))
        command = [
            sys.executable,
            "-m",
            "iron.applications.transformer_layer.operator_runlist_worker",
            "--request-json",
            str(request_path),
            "--response-json",
            str(response_path),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout
            raise RuntimeError(
                "operator_runlist child process failed with exit code "
                f"{result.returncode}" + (f": {detail}" if detail else "")
            )
        if not response_path.exists():
            raise RuntimeError(
                "operator_runlist child process did not produce a response payload"
            )
        row = json.loads(response_path.read_text())
    if write_immediately:
        write_results_csv(output_csv, [row])
    return [row]


def benchmark_pattern(
    *,
    execution_mode: str,
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    output_csv: str,
    seed: int,
    power_backend: str = "none",
    power_sample_interval_sec: float = 0.05,
    quiescent_baseline_duration_sec: float = 0.5,
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    if execution_mode == "operator_runlist":
        return _benchmark_operator_runlist_isolated(
            spec=spec,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            output_csv=output_csv,
            seed=seed,
            power_backend=power_backend,
            power_sample_interval_sec=power_sample_interval_sec,
            quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
            write_immediately=write_immediately,
        )

    pattern = build_pattern(execution_mode, spec)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)
    pattern.assign_weights(weights)

    for _ in range(warmup_runs):
        pattern(layer_inputs)

    latencies = []
    stage_sums_sec: dict[str, float] = {}
    for _ in range(runs_per_sample):
        start = time.perf_counter()
        _, stage_timings = _run_pattern_with_stage_timings(pattern, layer_inputs)
        latencies.append(time.perf_counter() - start)
        for stage_name, stage_sec in stage_timings.items():
            stage_sums_sec[stage_name] = stage_sums_sec.get(stage_name, 0.0) + float(
                stage_sec
            )

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
    )
    if power_backend == "none":
        power_stats = empty_power_stats()
    else:
        with create_power_monitor(
            power_backend=power_backend,
            sample_interval_sec=effective_power_sample_interval_sec,
            quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
            estimated_timed_window_sec=power_probe_window_sec,
        ) as power_monitor:
            power_phase_start = time.perf_counter()
            for _ in range(power_probe_runs):
                pattern(layer_inputs)
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
        "study_id": "synthetic_transformer_layer",
        "backend": "npu",
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
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
        **_average_stage_timings_ms(
            stage_sums_sec,
            summary["measured_inference_count"],
        ),
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
        **summary,
        **_pattern_metadata(pattern),
        **power_stats,
    }
    if write_immediately:
        write_results_csv(output_csv, [row])
    _cleanup_pattern_runtime(pattern)
    return [row]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a single transformer layer on NPU."
    )
    parser.add_argument(
        "--execution-mode",
        choices=SUPPORTED_EXECUTION_MODES,
        default="encoder_pipeline",
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
    parser.add_argument(
        "--power-backend",
        choices=("none", "turbostat_pkgwatt"),
        default="none",
    )
    parser.add_argument(
        "--power-sample-interval-sec",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--quiescent-baseline-duration-sec",
        type=float,
        default=0.5,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="transformer_layer_npu_latest.csv")
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
    benchmark_pattern(
        execution_mode=args.execution_mode,
        spec=spec,
        warmup_runs=args.warmup_runs,
        runs_per_sample=args.runs_per_sample,
        output_csv=args.output_csv,
        seed=args.seed,
        power_backend=args.power_backend,
        power_sample_interval_sec=args.power_sample_interval_sec,
        quiescent_baseline_duration_sec=args.quiescent_baseline_duration_sec,
    )


if __name__ == "__main__":
    main()
