# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .peak_reference import (
    BackendPeakReference,
    save_peak_reference,
    upsert_peak_reference,
)


def write_backend_peak_reference(
    *,
    backend: str,
    peak_ops_per_sec: float,
    peak_bytes_per_sec: float,
    output: str,
    source_note: str | None = None,
    append: bool = False,
) -> None:
    peak = BackendPeakReference(
        backend=backend,
        peak_ops_per_sec=peak_ops_per_sec,
        peak_bytes_per_sec=peak_bytes_per_sec,
        source_note=source_note,
    )
    if append:
        upsert_peak_reference(output, peak)
        return
    save_peak_reference(output, peak)


def run_calibrate_backend_peaks_cli(args) -> None:
    write_backend_peak_reference(
        backend=args.backend,
        peak_ops_per_sec=args.peak_ops_per_sec,
        peak_bytes_per_sec=args.peak_bytes_per_sec,
        output=args.output,
        source_note=args.source_note,
        append=args.append,
    )


__all__ = [
    "write_backend_peak_reference",
    "run_calibrate_backend_peaks_cli",
]
