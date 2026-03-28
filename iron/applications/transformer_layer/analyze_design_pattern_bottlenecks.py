#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iron.applications.transformer_layer.benchmark_common import write_dict_rows_csv
from iron.applications.transformer_layer.src.analysis.bottlenecks import (
    COMPONENT_FIELDS,
    SUMMARY_FIELD_ORDER,
    analyze_results,
    build_execution_mode_summary,
    build_row_summary,
    render_execution_mode_summaries,
)

__all__ = [
    "COMPONENT_FIELDS",
    "SUMMARY_FIELD_ORDER",
    "build_row_summary",
    "build_execution_mode_summary",
    "render_execution_mode_summaries",
    "analyze_results",
    "parse_args",
    "main",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize transformer-layer bottlenecks from a suite CSV."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-text", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_rows, aggregates, text_summary = analyze_results(args.input_csv)

    if args.summary_csv is not None:
        write_dict_rows_csv(
            args.summary_csv,
            summary_rows,
            fieldnames=SUMMARY_FIELD_ORDER,
        )
    if args.summary_json is not None:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    if args.summary_text is not None:
        output_path = Path(args.summary_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text_summary, encoding="utf-8")
    if (
        args.summary_csv is None
        and args.summary_json is None
        and args.summary_text is None
    ):
        print(text_summary, end="")


if __name__ == "__main__":
    main()
