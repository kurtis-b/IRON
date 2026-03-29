#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.analysis.plot_design_pattern_results import (
    SERIES_COLORS,
    generate_plots,
)

__all__ = [
    "SERIES_COLORS",
    "generate_plots",
    "parse_args",
    "main",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis-style SVG/HTML plots for transformer_layer study results."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bottleneck-csv", default=None)
    parser.add_argument("--gpu-compare-csv", default=None)
    parser.add_argument(
        "--x-axis", choices=("seq_len", "hidden_size"), default="seq_len"
    )
    parser.add_argument("--facet-key", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        bottleneck_csv=args.bottleneck_csv,
        gpu_compare_csv=args.gpu_compare_csv,
        x_axis=args.x_axis,
        facet_key=args.facet_key,
    )


if __name__ == "__main__":
    main()
