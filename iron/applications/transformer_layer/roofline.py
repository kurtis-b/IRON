# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from iron.applications.transformer_layer.peak_reference import BackendPeakReference
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def estimate_layer_flops(spec: TransformerLayerSpec) -> float:
    b = spec.batch_size
    s = spec.seq_len
    h = spec.hidden_size
    i = spec.intermediate_size
    n_heads = spec.num_attention_heads
    d = spec.attention_head_size
    return float(
        3 * 2 * b * s * h * h
        + 2 * b * n_heads * s * s * d
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
        4 * spec.hidden_size * spec.hidden_size
        + 2 * spec.hidden_size * spec.intermediate_size
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
