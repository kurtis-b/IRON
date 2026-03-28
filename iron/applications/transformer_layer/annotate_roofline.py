#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.analysis.annotate_roofline import (
    annotate_results_file,
    run_annotate_roofline_cli,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate a transformer_layer suite CSV with roofline metrics."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--peak-reference", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main():
    run_annotate_roofline_cli(parse_args())


__all__ = [
    "annotate_results_file",
]


if __name__ == "__main__":
    main()
