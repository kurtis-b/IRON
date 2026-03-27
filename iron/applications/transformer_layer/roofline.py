# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

from iron.applications.transformer_layer.peak_reference import BackendPeakReference
from iron.applications.transformer_layer.peak_reference import load_peak_references
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def estimate_layer_flops(spec: TransformerLayerSpec) -> float:
    b = spec.batch_size
    s = spec.seq_len
    h = spec.hidden_size
    i = spec.intermediate_size
    n_heads = spec.num_attention_heads
    d = spec.attention_head_size
    return float(
        2 * b * n_heads * s * s * d
        + 2 * b * n_heads * s * s * d
        + 2 * b * s * h * h
        + 2 * b * s * h * i
        + 2 * b * s * i * h
    )


def estimate_layer_bytes(spec: TransformerLayerSpec) -> float:
    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[spec.dtype]
    activation_bytes = spec.batch_size * spec.seq_len * spec.hidden_size * dtype_bytes
    weight_bytes = (
        spec.hidden_size * spec.hidden_size
        + 2 * spec.hidden_size * spec.intermediate_size
        + 2 * spec.hidden_size
    ) * dtype_bytes
    return float((4 * activation_bytes) + weight_bytes)


def operational_intensity(
    flops_per_inference: float, bytes_per_inference: float
) -> float:
    if bytes_per_inference <= 0:
        raise ValueError("bytes_per_inference must be positive")
    return flops_per_inference / bytes_per_inference


def roofline_bound_ops_per_sec(
    spec: TransformerLayerSpec,
    peak: BackendPeakReference,
) -> float:
    oi = operational_intensity(
        estimate_layer_flops(spec),
        estimate_layer_bytes(spec),
    )
    bandwidth_bound = oi * peak.peak_bytes_per_sec
    return min(peak.peak_ops_per_sec, bandwidth_bound)


def roofline_bound_from_metrics(
    *,
    operational_intensity_flops_per_byte: float,
    peak: BackendPeakReference,
) -> float:
    return min(
        peak.peak_ops_per_sec,
        operational_intensity_flops_per_byte * peak.peak_bytes_per_sec,
    )


def annotate_result_row_with_peak(
    row: dict[str, object],
    peak: BackendPeakReference | None,
) -> dict[str, object]:
    annotated = dict(row)
    if peak is None:
        return annotated
    if str(row.get("run_status", "completed")) != "completed":
        return annotated
    if row.get("throughput_flops_per_sec") in ("", None, "None"):
        return annotated

    throughput = float(row["throughput_flops_per_sec"])
    estimated_flops = float(row["estimated_flops_per_inference"])
    estimated_bytes = float(row["estimated_bytes_per_inference"])
    operational_intensity_value = row.get("operational_intensity_flops_per_byte")
    if operational_intensity_value in ("", None):
        oi = operational_intensity(estimated_flops, estimated_bytes)
    else:
        oi = float(operational_intensity_value)
    roofline_bound = roofline_bound_from_metrics(
        operational_intensity_flops_per_byte=oi,
        peak=peak,
    )
    annotated.update(
        {
            "backend_peak_ops_per_sec": peak.peak_ops_per_sec,
            "roofline_bound_ops_per_sec": roofline_bound,
            "backend_pct_of_peak": throughput / peak.peak_ops_per_sec,
            "roofline_pct": throughput / roofline_bound if roofline_bound > 0 else None,
        }
    )
    return annotated


def annotate_results_csv(
    input_csv: str | Path,
    peak_reference_path: str | Path,
    output_csv: str | Path,
) -> list[dict[str, object]]:
    peaks = load_peak_references(peak_reference_path)
    with Path(input_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(handle.readline()) if False else None
    annotated_rows = []
    for row in rows:
        peak = peaks.get(str(row["backend"]))
        annotated_rows.append(annotate_result_row_with_peak(row, peak))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        for row in annotated_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated_rows)
    return annotated_rows
