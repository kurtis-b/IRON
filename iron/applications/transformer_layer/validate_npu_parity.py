#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.pipeline.validate_npu_parity import (
    error_stats,
    run_parity_cli,
    run_parity_suite,
    validate_pattern_parity,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate single-layer NPU parity against the CPU reference."
    )
    parser.add_argument(
        "--execution-mode",
        choices=("dataflow", "gemm_offload", "runlist"),
        required=True,
    )
    parser.add_argument("--seq-lens", default="128")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--block1-topology-id", default=None)
    parser.add_argument("--block2-topology-id", default=None)
    parser.add_argument("--block3-topology-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    run_parity_cli(parse_args())


__all__ = [
    "error_stats",
    "validate_pattern_parity",
    "run_parity_suite",
]


if __name__ == "__main__":
    main()
