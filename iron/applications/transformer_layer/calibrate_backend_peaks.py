#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from iron.applications.transformer_layer.src.analysis.calibrate_backend_peaks import (
    run_calibrate_backend_peaks_cli,
    write_backend_peak_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a backend peak-reference artifact."
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--peak-ops-per-sec", type=float, required=True)
    parser.add_argument("--peak-bytes-per-sec", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-note",
        default=None,
        help="Optional auditable note about how the peak numbers were obtained.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append or replace this backend entry in a multi-backend peak artifact.",
    )
    return parser.parse_args()


def main():
    run_calibrate_backend_peaks_cli(parse_args())


__all__ = [
    "write_backend_peak_reference",
]


if __name__ == "__main__":
    main()
