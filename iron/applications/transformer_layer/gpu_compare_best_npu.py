#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.analysis.gpu_compare_best_npu import (
    benchmark_best_npu_vs_gpu,
    load_compare_config,
    select_best_npu_rows,
)

__all__ = [
    "load_compare_config",
    "select_best_npu_rows",
    "benchmark_best_npu_vs_gpu",
    "parse_args",
    "main",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AMD iGPU against the best completed NPU row per case."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_best_npu_vs_gpu(load_compare_config(args.config))


if __name__ == "__main__":
    main()
