#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    _record_parity_results,
    run_manifest_benchmark,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven synthetic transformer-layer NPU sweep."
    )
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--execution-modes", default="dataflow,runlist,gemm_offload")
    parser.add_argument("--seq-lens", default="64,128,256,512")
    parser.add_argument("--output-csv", default="transformer_layer_npu_suite.csv")
    parser.add_argument(
        "--debug-log-csv",
        default=None,
        help="Optional structured programmability/debug event log path.",
    )
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
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
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--block1-topology-id", default=None)
    parser.add_argument("--block2-topology-id", default=None)
    parser.add_argument("--block3-topology-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--peak-reference",
        default=None,
        help="Optional backend peak-reference artifact used to write an annotated suite CSV.",
    )
    parser.add_argument(
        "--annotated-output-csv",
        default=None,
        help="Optional annotated suite CSV output path.",
    )
    parser.add_argument(
        "--run-parity-check",
        action="store_true",
        help="Run layer-level parity checks after the NPU sweep.",
    )
    parser.add_argument(
        "--parity-output-csv",
        default=None,
        help="Optional parity-summary CSV path. Defaults to the manifest parity output.",
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip manifest-configured parity validation for this run.",
    )
    return parser.parse_args()


def main():
    run_manifest_benchmark(parse_args())


__all__ = [
    "_record_parity_results",
]


if __name__ == "__main__":
    main()
