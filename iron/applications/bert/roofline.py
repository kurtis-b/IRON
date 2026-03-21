#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from model_support import canonicalize_model_config

RESULT_METRIC_FIELDNAMES = [
    "chip_sku",
    "chip_family",
    "chip_codename",
    "bytes_model_version",
    "bytes_model_weights_policy",
    "dtype_bytes",
    "estimated_bytes_per_inference",
    "backend_peak_ops_per_sec",
    "primary_peak_source_kind",
    "backend_peak_source_note",
    "backend_pct_of_peak",
    "ddr_peak_bytes_per_sec",
    "ddr_peak_source_kind",
    "operational_intensity_flops_per_byte",
    "roofline_bound_ops_per_sec",
    "roofline_pct",
    "official_peak_ops_per_sec",
    "derived_theoretical_peak_ops_per_sec",
    "measured_peak_ops_per_sec",
]

SUPPORTED_BYTES_MODEL_VERSIONS = ("v1",)
SUPPORTED_WEIGHTS_POLICIES = ("resident", "streamed")
DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int8": 1,
}


def load_peak_reference_artifact(path: str | None):
    if path in ("", None):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_dtype_name(dtype):
    raw_dtype = str(dtype or "").replace("torch.", "").strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
        "int8": "int8",
    }
    if raw_dtype not in aliases:
        raise ValueError(f"Unsupported dtype {dtype!r} for roofline analysis")
    return aliases[raw_dtype]


def dtype_size_bytes(dtype):
    normalized = normalize_dtype_name(dtype)
    return DTYPE_BYTES[normalized]


def build_peak_reference_index(artifact):
    if artifact is None:
        return {}
    return {
        (row.get("backend"), row.get("dtype")): row
        for row in artifact.get("references", [])
    }


def select_peak_reference(artifact, backend: str, dtype: str):
    index = build_peak_reference_index(artifact)
    return index.get((backend, normalize_dtype_name(dtype)))


def estimate_encoder_bytes_per_inference(
    model_config,
    seq_len: int,
    dtype_bytes: int,
    bytes_model_version: str = "v1",
    weights_policy: str = "resident",
):
    if bytes_model_version not in SUPPORTED_BYTES_MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported bytes_model_version {bytes_model_version!r}. "
            f"Supported: {SUPPORTED_BYTES_MODEL_VERSIONS}"
        )
    if weights_policy not in SUPPORTED_WEIGHTS_POLICIES:
        raise ValueError(
            f"Unsupported weights_policy {weights_policy!r}. "
            f"Supported: {SUPPORTED_WEIGHTS_POLICIES}"
        )

    cfg = canonicalize_model_config(model_config)
    seq_len = int(seq_len)
    dtype_bytes = int(dtype_bytes)
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])

    embedding_output_bytes = seq_len * hidden_size * dtype_bytes
    per_layer_tensor_bytes = dtype_bytes * (
        (7 * seq_len * hidden_size)
        + (2 * seq_len * intermediate_size)
        + (2 * num_attention_heads * seq_len * seq_len)
    )
    final_hidden_state_write_bytes = seq_len * hidden_size * dtype_bytes
    total_bytes = (
        embedding_output_bytes
        + (num_hidden_layers * per_layer_tensor_bytes)
        + final_hidden_state_write_bytes
    )

    if weights_policy == "streamed":
        per_layer_weight_elements = (4 * hidden_size * hidden_size) + (
            2 * hidden_size * intermediate_size
        )
        total_bytes += num_hidden_layers * per_layer_weight_elements * dtype_bytes

    return int(total_bytes)


def divide_optional(numerator, denominator):
    if numerator in ("", None) or denominator in ("", None):
        return None
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(numerator) / denominator


def annotate_benchmark_row(
    row: dict,
    *,
    model_config,
    peak_reference_artifact=None,
    bytes_model_version: str = "v1",
    weights_policy: str = "resident",
):
    dtype = normalize_dtype_name(row.get("dtype"))
    dtype_bytes = dtype_size_bytes(dtype)
    estimated_flops_per_inference = float(row.get("estimated_flops_per_inference", 0.0))
    estimated_bytes_per_inference = estimate_encoder_bytes_per_inference(
        model_config,
        row["seq_len"],
        dtype_bytes,
        bytes_model_version=bytes_model_version,
        weights_policy=weights_policy,
    )
    throughput_flops_per_sec = divide_optional(
        row.get("throughput_flops_per_sec"),
        1.0,
    )
    operational_intensity = divide_optional(
        estimated_flops_per_inference,
        estimated_bytes_per_inference,
    )

    peak_row = select_peak_reference(peak_reference_artifact, row["mode"], dtype)
    memory_bandwidth = (
        peak_reference_artifact.get("memory_bandwidth", {})
        if peak_reference_artifact is not None
        else {}
    )
    backend_peak_ops_per_sec = (
        peak_row.get("primary_peak_ops_per_sec") if peak_row is not None else None
    )
    backend_pct_of_peak = (
        divide_optional(throughput_flops_per_sec, backend_peak_ops_per_sec)
        if backend_peak_ops_per_sec is not None
        else None
    )
    ddr_peak_bytes_per_sec = memory_bandwidth.get("primary_peak_bytes_per_sec")
    roofline_bound_ops_per_sec = None
    if (
        backend_peak_ops_per_sec is not None
        and ddr_peak_bytes_per_sec is not None
        and operational_intensity is not None
    ):
        roofline_bound_ops_per_sec = min(
            float(backend_peak_ops_per_sec),
            float(ddr_peak_bytes_per_sec) * operational_intensity,
        )
    roofline_pct = (
        divide_optional(throughput_flops_per_sec, roofline_bound_ops_per_sec)
        if roofline_bound_ops_per_sec is not None
        else None
    )

    return {
        "chip_sku": (
            peak_reference_artifact.get("chip_sku", "")
            if peak_reference_artifact is not None
            else ""
        ),
        "chip_family": (
            peak_reference_artifact.get("chip_family", "")
            if peak_reference_artifact is not None
            else ""
        ),
        "chip_codename": (
            peak_reference_artifact.get("chip_codename", "")
            if peak_reference_artifact is not None
            else ""
        ),
        "bytes_model_version": bytes_model_version,
        "bytes_model_weights_policy": weights_policy,
        "dtype_bytes": str(dtype_bytes),
        "estimated_bytes_per_inference": str(estimated_bytes_per_inference),
        "backend_peak_ops_per_sec": (
            f"{backend_peak_ops_per_sec:.6e}"
            if backend_peak_ops_per_sec is not None
            else ""
        ),
        "primary_peak_source_kind": (
            peak_row.get("primary_peak_source_kind", "") if peak_row is not None else ""
        ),
        "backend_peak_source_note": (
            peak_row.get("peak_source_note", "") if peak_row is not None else ""
        ),
        "backend_pct_of_peak": (
            f"{backend_pct_of_peak:.6f}" if backend_pct_of_peak is not None else ""
        ),
        "ddr_peak_bytes_per_sec": (
            f"{float(ddr_peak_bytes_per_sec):.6e}"
            if ddr_peak_bytes_per_sec is not None
            else ""
        ),
        "ddr_peak_source_kind": memory_bandwidth.get("primary_peak_source_kind", ""),
        "operational_intensity_flops_per_byte": (
            f"{operational_intensity:.6f}" if operational_intensity is not None else ""
        ),
        "roofline_bound_ops_per_sec": (
            f"{roofline_bound_ops_per_sec:.6e}"
            if roofline_bound_ops_per_sec is not None
            else ""
        ),
        "roofline_pct": f"{roofline_pct:.6f}" if roofline_pct is not None else "",
        "official_peak_ops_per_sec": (
            f"{float(peak_row['official_peak_ops_per_sec']):.6e}"
            if peak_row is not None
            and peak_row.get("official_peak_ops_per_sec") is not None
            else ""
        ),
        "derived_theoretical_peak_ops_per_sec": (
            f"{float(peak_row['derived_theoretical_peak_ops_per_sec']):.6e}"
            if peak_row is not None
            and peak_row.get("derived_theoretical_peak_ops_per_sec") is not None
            else ""
        ),
        "measured_peak_ops_per_sec": (
            f"{float(peak_row['measured_peak_ops_per_sec']):.6e}"
            if peak_row is not None
            and peak_row.get("measured_peak_ops_per_sec") is not None
            else ""
        ),
    }
