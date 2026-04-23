#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import torch


REQUIRED_WEIGHT_NAMES = (
    "q_weight",
    "k_weight",
    "v_weight",
    "attn_output_weight",
    "ln1_weight",
    "ffn_up_weight",
    "ffn_down_weight",
    "ln2_weight",
)


def _percentile(values: torch.Tensor, fraction: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, torch.tensor(float(fraction))).item())


def error_summary(
    output_tensor: torch.Tensor,
    reference_output_tensor: torch.Tensor,
    *,
    abs_tol: float,
    rel_tol: float,
    histogram_bins: int = 32,
) -> dict[str, object]:
    output = output_tensor.float().flatten()
    reference = reference_output_tensor.float().flatten()
    abs_error = torch.abs(output - reference)
    denom = torch.abs(reference)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    rel_error = abs_error / safe_denom
    abs_threshold_fraction = float((abs_error > float(abs_tol)).float().mean().item())
    rel_threshold_fraction = float((rel_error > float(rel_tol)).float().mean().item())
    histogram_max = max(float(abs_error.max().item()) if abs_error.numel() else 0.0, abs_tol)
    counts = torch.histc(
        abs_error,
        bins=int(histogram_bins),
        min=0.0,
        max=float(histogram_max) if histogram_max > 0 else 1.0,
    )
    edges = torch.linspace(
        0.0,
        float(histogram_max) if histogram_max > 0 else 1.0,
        int(histogram_bins) + 1,
    )
    return {
        "abs_error_mean": float(abs_error.mean().item()) if abs_error.numel() else 0.0,
        "abs_error_p50": _percentile(abs_error, 0.50),
        "abs_error_p90": _percentile(abs_error, 0.90),
        "abs_error_p95": _percentile(abs_error, 0.95),
        "abs_error_p99": _percentile(abs_error, 0.99),
        "abs_error_max": float(abs_error.max().item()) if abs_error.numel() else 0.0,
        "rel_error_mean": float(rel_error.mean().item()) if rel_error.numel() else 0.0,
        "rel_error_p95": _percentile(rel_error, 0.95),
        "rel_error_max": float(rel_error.max().item()) if rel_error.numel() else 0.0,
        "abs_threshold_fraction": abs_threshold_fraction,
        "rel_threshold_fraction": rel_threshold_fraction,
        "abs_histogram_edges_json": json.dumps([float(value) for value in edges.tolist()]),
        "abs_histogram_counts_json": json.dumps([int(value) for value in counts.tolist()]),
    }


def _load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor at {path}, found {type(tensor).__name__}")
    return tensor


def load_reference_payload(path: Path) -> dict[str, object]:
    resolved_path = path.expanduser()
    if resolved_path.is_file():
        payload = torch.load(resolved_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(
                f"Reference payload must be a dict, found {type(payload).__name__}"
            )
        payload = dict(payload)
    elif resolved_path.is_dir():
        payload: dict[str, object] = {}
        metadata_path = resolved_path / "metadata.json"
        if metadata_path.exists():
            payload["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        input_path = resolved_path / "input.pt"
        if input_path.exists():
            payload["input"] = _load_tensor(input_path)
        output_path = resolved_path / "output.pt"
        if output_path.exists():
            payload["output"] = _load_tensor(output_path)
        weights_path = resolved_path / "weights.pt"
        if weights_path.exists():
            weights = torch.load(weights_path, map_location="cpu")
            if not isinstance(weights, dict):
                raise TypeError("weights.pt must contain a dict")
            payload["weights"] = dict(weights)
        else:
            weights = {}
            for weight_name in REQUIRED_WEIGHT_NAMES:
                candidate = resolved_path / f"{weight_name}.pt"
                if candidate.exists():
                    weights[weight_name] = _load_tensor(candidate)
            if weights:
                payload["weights"] = weights
    else:
        raise FileNotFoundError(f"Missing reference payload: {resolved_path}")

    if "weights" not in payload:
        flat_weights = {
            weight_name: payload[weight_name]
            for weight_name in REQUIRED_WEIGHT_NAMES
            if weight_name in payload
        }
        if flat_weights:
            payload["weights"] = flat_weights
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Reference payload must include a 'weights' mapping")
    if not isinstance(payload.get("input"), torch.Tensor):
        raise ValueError("Reference payload must include an input tensor")
    return payload
