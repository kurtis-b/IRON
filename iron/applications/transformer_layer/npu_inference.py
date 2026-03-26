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
)
from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
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
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    request = {
        "spec": spec.to_dict(),
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "seed": seed,
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
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"operator_runlist child process failed with exit code {result.returncode}"
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
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    if execution_mode == "operator_runlist":
        return _benchmark_operator_runlist_isolated(
            spec=spec,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            output_csv=output_csv,
            seed=seed,
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
    with create_power_monitor() as power_stats:
        for _ in range(runs_per_sample):
            start = time.perf_counter()
            _, stage_timings = _run_pattern_with_stage_timings(pattern, layer_inputs)
            latencies.append(time.perf_counter() - start)
            for stage_name, stage_sec in stage_timings.items():
                stage_sums_sec[stage_name] = stage_sums_sec.get(
                    stage_name, 0.0
                ) + float(stage_sec)

    summary = summarize_latency_measurements(latencies)
    estimated_flops = estimate_layer_flops(spec)
    estimated_bytes = estimate_layer_bytes(spec)
    throughput_flops_per_sec = (
        estimated_flops
        * summary["measured_inference_count"]
        / summary["timed_total_sec"]
    )
    row = {
        "study_id": "synthetic_transformer_layer",
        "backend": "npu",
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": spec.seq_len,
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
        **summary,
        **_pattern_metadata(pattern),
        **(power_stats if power_stats is not None else empty_power_stats()),
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
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs-per-sample", type=int, default=20)
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
    )


if __name__ == "__main__":
    main()
