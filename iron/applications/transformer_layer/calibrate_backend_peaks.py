#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from iron.applications.transformer_layer.peak_reference import (
    BackendPeakReference,
    save_peak_reference,
    upsert_peak_reference,
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
    args = parse_args()
    peak = BackendPeakReference(
        backend=args.backend,
        peak_ops_per_sec=args.peak_ops_per_sec,
        peak_bytes_per_sec=args.peak_bytes_per_sec,
        source_note=args.source_note,
    )
    if args.append:
        upsert_peak_reference(args.output, peak)
        return
    save_peak_reference(args.output, peak)


if __name__ == "__main__":
    main()
