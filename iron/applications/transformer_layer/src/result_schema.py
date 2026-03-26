# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

RESULT_FIELD_ORDER = [
    "study_id",
    "backend",
    "execution_mode",
    "pattern_label",
    "seq_len",
    "batch_size",
    "dtype",
    "use_bias",
    "weights_source",
    "source_model_name",
    "source_layer_index",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
    "estimated_flops_per_inference",
    "estimated_bytes_per_inference",
    "operational_intensity_flops_per_byte",
    "backend_peak_ops_per_sec",
    "roofline_bound_ops_per_sec",
    "backend_pct_of_peak",
    "roofline_pct",
]

REQUIRED_RESULT_FIELDS = {
    "study_id",
    "backend",
    "execution_mode",
    "seq_len",
    "batch_size",
    "dtype",
    "weights_source",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
}


def normalize_result_row(row: dict[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_RESULT_FIELDS.difference(row))
    if missing:
        raise KeyError(f"Missing required result fields: {', '.join(missing)}")
    normalized = {field: row.get(field) for field in RESULT_FIELD_ORDER}
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = value
    return normalized
