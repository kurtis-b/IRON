# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .roofline import annotate_results_csv


def annotate_results_file(
    *, input_csv: str, peak_reference_path: str, output_csv: str
) -> None:
    annotate_results_csv(
        input_csv=input_csv,
        peak_reference_path=peak_reference_path,
        output_csv=output_csv,
    )


def run_annotate_roofline_cli(args) -> None:
    annotate_results_file(
        input_csv=args.input_csv,
        peak_reference_path=args.peak_reference,
        output_csv=args.output_csv,
    )


__all__ = [
    "annotate_results_file",
    "run_annotate_roofline_cli",
]
