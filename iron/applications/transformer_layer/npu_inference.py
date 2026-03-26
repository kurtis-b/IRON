#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
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
    make_synthetic_hidden_states,
    make_synthetic_layer_weights,
)

SUPPORTED_EXECUTION_MODES = (
    "encoder_pipeline",
    "gemm_only",
    "operator_runlist",
)


def build_pattern(execution_mode: str, spec: TransformerLayerSpec):
    if execution_mode == "encoder_pipeline":
        return EncoderPipelinePattern(spec)
    if execution_mode == "gemm_only":
        return GemmOnlyPattern(spec)
    if execution_mode == "operator_runlist":
        return OperatorRunlistPattern(spec)
    raise ValueError(f"Unsupported execution_mode: {execution_mode}")


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
    pattern = build_pattern(execution_mode, spec)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    hidden_states = make_synthetic_hidden_states(spec, seed=seed + 1)
    pattern.assign_weights(weights)

    for _ in range(warmup_runs):
        pattern(hidden_states)

    latencies = []
    with create_power_monitor() as power_stats:
        for _ in range(runs_per_sample):
            start = time.perf_counter()
            pattern(hidden_states)
            latencies.append(time.perf_counter() - start)

    summary = summarize_latency_measurements(latencies)
    estimated_flops = estimate_layer_flops(spec)
    estimated_bytes = estimate_layer_bytes(spec)
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
        **(power_stats if power_stats is not None else empty_power_stats()),
    }
    if write_immediately:
        write_results_csv(output_csv, [row])
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
