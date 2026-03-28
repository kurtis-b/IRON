#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.bench.gpu_inference import (
    SUPPORTED_POWER_BACKENDS,
    benchmark_gpu_layer,
    resolve_amd_gpu_device,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a single transformer layer on an isolated AMD GPU path."
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
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


__all__ = [
    "SUPPORTED_POWER_BACKENDS",
    "resolve_amd_gpu_device",
    "benchmark_gpu_layer",
]


if __name__ == "__main__":
    main()
