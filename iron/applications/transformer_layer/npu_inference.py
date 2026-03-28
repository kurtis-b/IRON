#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.bench.npu_inference import (
    SUPPORTED_EXECUTION_MODES,
    benchmark_pattern,
    build_pattern,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a single transformer layer on NPU."
    )
    parser.add_argument(
        "--execution-mode",
        choices=SUPPORTED_EXECUTION_MODES,
        default="dataflow",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
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
        power_backend=args.power_backend,
        power_sample_interval_sec=args.power_sample_interval_sec,
        quiescent_baseline_duration_sec=args.quiescent_baseline_duration_sec,
        enable_measurement_log=args.enable_measurement_log,
        measurement_log_path=args.measurement_log_path,
    )


__all__ = [
    "SUPPORTED_EXECUTION_MODES",
    "build_pattern",
    "benchmark_pattern",
]


if __name__ == "__main__":
    main()
