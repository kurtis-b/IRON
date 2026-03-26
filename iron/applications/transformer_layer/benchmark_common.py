# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

from iron.applications.transformer_layer.src.result_schema import (
    RESULT_FIELD_ORDER,
    normalize_result_row,
)


def parse_seq_lens(seq_lens: str) -> list[int]:
    return [int(part.strip()) for part in seq_lens.split(",") if part.strip()]


def summarize_latency_measurements(latencies_sec: list[float]) -> dict[str, float]:
    if not latencies_sec:
        raise ValueError("No latency measurements were provided")
    total_sec = sum(latencies_sec)
    return {
        "timed_total_sec": total_sec,
        "avg_latency_ms": (total_sec / len(latencies_sec)) * 1000.0,
        "measured_inference_count": len(latencies_sec),
    }


def write_results_csv(output_csv: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_result_row(row) for row in rows]
    extra_fields = sorted(
        {key for row in normalized for key in row.keys()}.difference(RESULT_FIELD_ORDER)
    )
    fieldnames = RESULT_FIELD_ORDER + extra_fields
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
