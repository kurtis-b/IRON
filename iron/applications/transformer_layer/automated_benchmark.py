#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.benchmark_common import parse_seq_lens
from iron.applications.transformer_layer.npu_inference import benchmark_pattern
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a synthetic transformer-layer NPU sweep."
    )
    parser.add_argument(
        "--execution-modes", default="encoder_pipeline,gemm_only,operator_runlist"
    )
    parser.add_argument("--seq-lens", default="64,128,256,512")
    parser.add_argument("--output-csv", default="transformer_layer_npu_suite.csv")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs-per-sample", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    execution_modes = [
        mode.strip() for mode in args.execution_modes.split(",") if mode.strip()
    ]
    all_rows = []
    for seq_len in parse_seq_lens(args.seq_lens):
        spec = TransformerLayerSpec(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            seq_len=seq_len,
            use_bias=False,
            weights_source="synthetic",
        )
        for execution_mode in execution_modes:
            all_rows.extend(
                benchmark_pattern(
                    execution_mode=execution_mode,
                    spec=spec,
                    warmup_runs=args.warmup_runs,
                    runs_per_sample=args.runs_per_sample,
                    output_csv=args.output_csv,
                    seed=args.seed,
                    write_immediately=False,
                )
            )
    if all_rows:
        from benchmark_common import write_results_csv

        write_results_csv(args.output_csv, all_rows)


if __name__ == "__main__":
    main()
