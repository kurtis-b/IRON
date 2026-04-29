#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from iron.applications.transformer_layer.study.end_to_end.cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    WORKLOAD_VARIANTS,
    canonical_execution_mode,
    effective_dense_layer_flop_count,
    mode_operators,
)
from iron.applications.transformer_layer.study.end_to_end.select import (
    load_result_rows,
    select_result_rows,
)
from iron.applications.transformer_layer.study.plot_families import (
    PLOT_FAMILY_GRID_SHAPE,
    ordered_plot_families,
    plot_family_label,
)

LOGGER = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parents[2]
PLOT_SUFFIX = ".svg"
PNG_SUFFIX = ".png"
DTYPE_BYTES = 2
BF16_MACS_PER_CYCLE_PER_CORE = 64
FLOPS_PER_MAC = 2
NPU_AIE_CORE_CLOCK_HZ = 1.8e9
NPU_AIE_CORES = 32
NPU_SHIM_TILES = 8
MODE_LABELS = {
    "hybrid": "Hybrid",
    "runlist": "Runlist",
    "offload": "Offload",
}
MODE_COLORS = {
    "hybrid": "#1f6f8b",
    "runlist": "#e07a5f",
    "offload": "#6c9a3b",
}
MODE_MARKERS = {
    "hybrid": "o",
    "runlist": "s",
    "offload": "^",
}
HYBRID_OPERATOR_SHORT_LABELS = {
    "qkv_proj": "QKV",
    "mha_out_proj": "MHAO",
    "add_norm1": "AN1",
    "ffn": "FFN",
    "add_norm2": "AN2",
}
RUNLIST_OPERATOR_SHORT_LABELS = {
    "qkvo_proj": "QKVO",
    "k_transpose": "KT",
    "attn_scores": "AS",
    "attn_scale": "SC",
    "attn_softmax": "SM",
    "attn_output": "AO",
    "add": "ADD",
    "ln1": "LN1",
    "up_proj": "UP",
    "gelu": "GELU",
    "down_proj": "DOWN",
    "ln2": "LN2",
}
OFFLOAD_OPERATOR_SHORT_LABELS = {
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "attn_scores": "AS",
    "attn_output": "AO",
    "output_proj": "O",
    "up_proj": "UP",
    "down_proj": "DOWN",
}
KERNEL_POINTS_FIELDNAMES = (
    "study_case_id",
    "family_label",
    "workload_variant",
    "execution_mode",
    "logical_operator",
    "operator_short_label",
    "seq_len",
    "compute_tiles_used",
    "shim_tiles_used",
    "candidate_id",
    "run_status",
    "omission_note",
    "avg_latency_ms",
    "measured_bandwidth_gbps",
    "kernel_flop_count",
    "kernel_byte_count",
    "effective_gflops_per_sec",
    "operational_intensity_flops_per_byte",
    "plot_label",
)
IMPLEMENTATION_POINTS_FIELDNAMES = (
    "study_case_id",
    "family_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "compute_tiles_used",
    "shim_tiles_used",
    "run_status",
    "omission_note",
    "avg_latency_ms",
    "effective_gflops_per_sec",
    "implementation_flop_count",
    "implementation_byte_count",
    "operational_intensity_flops_per_byte",
    "selected_candidate_ids_json",
    "plot_label",
)

ResourceUsageKey = tuple[str, str, str, str, int, str]
ResourceUsageTileCounts = tuple[int | None, int | None]


def _results_snapshot_roots(app_root: Path = APP_ROOT) -> tuple[Path, ...]:
    return tuple(sorted(app_root.glob("results_commit*"), reverse=True))


def default_output_dir() -> Path:
    canonical = APP_ROOT / "results" / "roofline"
    if canonical.parent.exists():
        return canonical
    final = APP_ROOT / "results_final" / "roofline"
    if final.parent.exists():
        return final
    return canonical


def _default_end_to_end_file(filename: str) -> Path:
    canonical = APP_ROOT / "results" / "end_to_end" / filename
    if canonical.exists():
        return canonical
    final = APP_ROOT / "results_final" / "end_to_end" / filename
    if final.exists():
        return final
    for snapshot_root in _results_snapshot_roots():
        snapshot = snapshot_root / "end_to_end" / filename
        if snapshot.exists():
            return snapshot
    return canonical


def default_end_to_end_results_path() -> Path:
    return _default_end_to_end_file("results_all_power.csv")


def default_end_to_end_tuning_path() -> Path:
    return _default_end_to_end_file("tuning_all_power.csv")


def default_memcpy_results_path() -> Path:
    canonical = APP_ROOT / "results" / "memcpy_bandwidth" / "results.csv"
    if canonical.exists():
        return canonical
    final = APP_ROOT / "results_final" / "memcpy_bandwidth" / "results.csv"
    if final.exists():
        return final
    return canonical


def default_kernel_points_path(output_dir: Path) -> Path:
    return output_dir / "kernel_points.csv"


def default_implementation_points_path(output_dir: Path) -> Path:
    return output_dir / "implementation_points.csv"


def default_kernel_plot_path(output_dir: Path) -> Path:
    return output_dir / "kernel_roofline.svg"


def default_implementation_plot_path(output_dir: Path) -> Path:
    return output_dir / "implementation_roofline.svg"


def _resolve_plot_path(path: Path) -> Path:
    if path.suffix.lower() != PLOT_SUFFIX:
        raise ValueError(f"Plot output must use the {PLOT_SUFFIX} suffix: {path}")
    return path


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _load_json_dict(value: object) -> dict[str, object]:
    text = str(value or "").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object, received {type(loaded).__name__}")
    return {str(key): value for key, value in loaded.items()}


def _normalized_execution_mode(value: object) -> str | None:
    try:
        return canonical_execution_mode(str(value or ""))
    except ValueError:
        return None


def _csv_family_label(study_case_id: str) -> str:
    if study_case_id in FAMILY_SPECS:
        return FAMILY_SPECS[study_case_id].display_label
    return study_case_id


def _operator_short_label(execution_mode: str, logical_operator: str) -> str:
    if execution_mode == "hybrid":
        return HYBRID_OPERATOR_SHORT_LABELS.get(logical_operator, logical_operator)
    if execution_mode == "offload":
        return OFFLOAD_OPERATOR_SHORT_LABELS.get(logical_operator, logical_operator)
    return RUNLIST_OPERATOR_SHORT_LABELS.get(logical_operator, logical_operator)


def _positive_config_int(
    operator_config: dict[str, object],
    key: str,
    *,
    default: int = 1,
) -> int:
    value = _optional_int(operator_config.get(key))
    if value is None or value <= 0:
        return default
    return value


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def default_resource_usage_selected_ops_paths(
    end_to_end_results_input: Path,
) -> tuple[Path, Path, Path]:
    results_root = end_to_end_results_input.expanduser().resolve().parent.parent
    resource_usage_dir = results_root / "resource_usage"
    return (
        resource_usage_dir / "hybrid_selected_ops.csv",
        resource_usage_dir / "runlist_selected_ops.csv",
        resource_usage_dir / "offload_selected_ops.csv",
    )


def build_resource_usage_index(
    resource_usage_rows: Iterable[dict[str, str]],
) -> dict[ResourceUsageKey, ResourceUsageTileCounts]:
    index: dict[ResourceUsageKey, ResourceUsageTileCounts] = {}
    for row in resource_usage_rows:
        execution_mode = _normalized_execution_mode(row.get("execution_mode"))
        study_case_id = str(row.get("study_case_id") or row.get("family_id") or "")
        workload_variant = str(row.get("workload_variant") or "")
        logical_operator = str(row.get("logical_operator") or "")
        seq_len = _optional_int(row.get("seq_len"))
        candidate_id = str(
            row.get("selected_candidate_id") or row.get("candidate_id") or ""
        )
        if (
            execution_mode is None
            or not study_case_id
            or not workload_variant
            or not logical_operator
            or seq_len is None
            or not candidate_id
        ):
            continue
        key = (
            execution_mode,
            study_case_id,
            workload_variant,
            logical_operator,
            int(seq_len),
            candidate_id,
        )
        tile_counts = (
            _optional_int(row.get("compute_tiles_used")),
            _optional_int(row.get("shim_tiles_with_dma")),
        )
        current = index.get(key)
        if current is None:
            index[key] = tile_counts
            continue
        current_compute, current_shim = current
        compute_tiles_used, shim_tiles_used = tile_counts
        if current_compute is None and compute_tiles_used is not None:
            current_compute = compute_tiles_used
        if current_shim is None and shim_tiles_used is not None:
            current_shim = shim_tiles_used
        index[key] = (current_compute, current_shim)
    return index


def load_resource_usage_index(
    paths: Iterable[Path],
) -> dict[ResourceUsageKey, ResourceUsageTileCounts]:
    rows: list[dict[str, str]] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved.exists():
            rows.extend(load_csv_rows(resolved))
    return build_resource_usage_index(rows)


def resolve_resource_usage_tile_counts(
    resource_usage_index: dict[ResourceUsageKey, ResourceUsageTileCounts] | None,
    *,
    execution_mode: str,
    study_case_id: str,
    workload_variant: str,
    logical_operator: str,
    seq_len: int,
    candidate_id: str,
) -> ResourceUsageTileCounts:
    if not resource_usage_index:
        return (None, None)
    exact_key = (
        execution_mode,
        study_case_id,
        workload_variant,
        logical_operator,
        seq_len,
        candidate_id,
    )
    exact_match = resource_usage_index.get(exact_key)
    if exact_match is not None:
        return exact_match
    wildcard_matches = [
        tile_counts
        for key, tile_counts in resource_usage_index.items()
        if key[:-1]
        == (
            execution_mode,
            study_case_id,
            workload_variant,
            logical_operator,
            seq_len,
        )
    ]
    if len(wildcard_matches) == 1:
        return wildcard_matches[0]
    return (None, None)


def resolved_operator_tile_counts(
    *,
    execution_mode: str,
    study_case_id: str,
    workload_variant: str,
    logical_operator: str,
    seq_len: int,
    candidate_id: str,
    operator_config: dict[str, object],
    resource_usage_index: dict[ResourceUsageKey, ResourceUsageTileCounts] | None = None,
) -> ResourceUsageTileCounts:
    compute_tiles_used, shim_tiles_used = resolve_resource_usage_tile_counts(
        resource_usage_index,
        execution_mode=execution_mode,
        study_case_id=study_case_id,
        workload_variant=workload_variant,
        logical_operator=logical_operator,
        seq_len=seq_len,
        candidate_id=candidate_id,
    )
    if compute_tiles_used is None:
        compute_tiles_used = operator_compute_tiles_used(
            execution_mode=execution_mode,
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if shim_tiles_used is None:
        shim_tiles_used = operator_shim_tiles_used(
            execution_mode=execution_mode,
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    return (compute_tiles_used, shim_tiles_used)


def _passed_memcpy_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("run_status") == "passed"
        and _optional_float(row.get("bandwidth_gbps")) is not None
    ]


def peak_memcpy_bandwidth_gbps(memcpy_rows: Iterable[dict[str, str]]) -> float:
    passed_rows = _passed_memcpy_rows(memcpy_rows)
    if not passed_rows:
        raise ValueError("No successful memcpy bandwidth rows were found")

    overall_peak_rows = [
        row
        for row in passed_rows
        if str(row.get("is_overall_peak") or "").lower() == "true"
    ]
    source_rows = overall_peak_rows if overall_peak_rows else passed_rows
    return max(float(row["bandwidth_gbps"]) for row in source_rows)


def peak_memcpy_bandwidth_by_shim_tiles_gbps(
    memcpy_rows: Iterable[dict[str, str]],
) -> dict[int, float]:
    passed_rows = _passed_memcpy_rows(memcpy_rows)
    peak_by_shim_tiles: dict[int, float] = {}
    for row in passed_rows:
        num_cores = _optional_int(row.get("num_cores"))
        num_channels = _optional_int(row.get("num_channels"))
        if num_cores is None or num_channels is None or num_channels <= 0:
            continue
        shim_tiles_used = num_cores // num_channels
        if shim_tiles_used <= 0:
            continue
        current = peak_by_shim_tiles.get(shim_tiles_used)
        bandwidth = float(row["bandwidth_gbps"])
        if current is None or bandwidth > current:
            peak_by_shim_tiles[shim_tiles_used] = bandwidth
    return peak_by_shim_tiles


def _batch_factor(operator_config: dict[str, object]) -> int:
    factors: list[int] = []
    for key in ("batch_A", "batch_B", "batch_C"):
        value = operator_config.get(key)
        if isinstance(value, (list, tuple)) and value:
            first = _optional_int(value[0])
            if first is not None and first > 0:
                factors.append(first)
    return max(factors, default=1)


def _gemm_bytes_and_flops(operator_config: dict[str, object]) -> tuple[float, float]:
    m = float(_optional_int(operator_config.get("M")) or 0)
    k = float(_optional_int(operator_config.get("K")) or 0)
    n = float(_optional_int(operator_config.get("N")) or 0)
    batch_factor = float(_batch_factor(operator_config))
    flop_count = 2.0 * batch_factor * m * k * n
    byte_count = float(DTYPE_BYTES) * batch_factor * ((m * k) + (k * n) + (m * n))
    return flop_count, byte_count


def _hybrid_operator_bytes_and_flops(
    *,
    logical_operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    operator_config: dict[str, object],
) -> tuple[float, float]:
    seq = float(seq_len)
    hidden = float(hidden_size)
    intermediate = float(intermediate_size)

    if logical_operator == "qkv_proj":
        flop_count = 6.0 * seq * hidden * hidden
        byte_count = float(DTYPE_BYTES) * (
            (seq * hidden) + (3.0 * hidden * hidden) + (3.0 * seq * hidden)
        )
        return flop_count, byte_count

    if logical_operator == "mha_out_proj":
        flop_count = (4.0 * seq * seq * hidden) + (2.0 * seq * hidden * hidden)
        byte_count = float(DTYPE_BYTES) * ((4.0 * seq * hidden) + (hidden * hidden))
        return flop_count, byte_count

    if logical_operator in {"add_norm1", "add_norm2", "add_norm"}:
        size = float(
            _optional_int(operator_config.get("size")) or (seq_len * hidden_size)
        )
        tile_size = float(
            _optional_int(operator_config.get("tile_size")) or hidden_size
        )
        flop_count = 8.0 * size
        byte_count = float(DTYPE_BYTES) * ((3.0 * size) + tile_size)
        return flop_count, byte_count

    if logical_operator == "ln1":
        size = float(
            _optional_int(operator_config.get("size")) or (seq_len * hidden_size)
        )
        tile_size = float(
            _optional_int(operator_config.get("tile_size")) or hidden_size
        )
        return 7.0 * size, float(DTYPE_BYTES) * ((2.0 * size) + tile_size)

    if logical_operator == "add":
        size = float(_optional_int(operator_config.get("size")) or 0)
        return size, float(DTYPE_BYTES) * (3.0 * size)

    if logical_operator == "ffn":
        m = float(_optional_int(operator_config.get("M")) or seq_len)
        k = float(_optional_int(operator_config.get("K")) or hidden_size)
        n = float(_optional_int(operator_config.get("N")) or intermediate_size)
        flop_count = (4.0 * m * k * n) + (8.0 * m * n)
        byte_count = float(DTYPE_BYTES) * ((2.0 * m * k) + (m * n) + (2.0 * k * n))
        return flop_count, byte_count

    raise ValueError(f"Unsupported hybrid logical operator: {logical_operator}")


def _runlist_operator_bytes_and_flops(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> tuple[float, float]:
    if logical_operator in {
        "qkvo_proj",
        "attn_scores",
        "attn_output",
        "up_proj",
        "down_proj",
    }:
        return _gemm_bytes_and_flops(operator_config)

    if logical_operator == "k_transpose":
        m = float(_optional_int(operator_config.get("M")) or 0)
        n = float(_optional_int(operator_config.get("N")) or 0)
        size = m * n
        return 0.0, float(DTYPE_BYTES) * (2.0 * size)

    if logical_operator == "attn_scale":
        size = float(_optional_int(operator_config.get("size")) or 0)
        return size, float(DTYPE_BYTES) * (2.0 * size)

    if logical_operator == "causal_mask":
        query_block_size = float(
            _optional_int(operator_config.get("query_block_size")) or 0
        )
        seq_len = float(_optional_int(operator_config.get("seq_len")) or 0)
        num_heads = float(_optional_int(operator_config.get("num_heads")) or 0)
        size = query_block_size * seq_len * num_heads
        return size, float(DTYPE_BYTES) * (3.0 * size)

    if logical_operator == "attn_softmax":
        rows = float(_optional_int(operator_config.get("rows")) or 0)
        cols = float(_optional_int(operator_config.get("cols")) or 0)
        size = rows * cols
        return 5.0 * size, float(DTYPE_BYTES) * (2.0 * size)

    if logical_operator == "add":
        size = float(_optional_int(operator_config.get("size")) or 0)
        return size, float(DTYPE_BYTES) * (3.0 * size)

    if logical_operator in {"ln1", "ln2"}:
        size = float(_optional_int(operator_config.get("size")) or 0)
        tile_size = float(_optional_int(operator_config.get("tile_size")) or 0)
        return 7.0 * size, float(DTYPE_BYTES) * ((2.0 * size) + tile_size)

    if logical_operator == "gelu":
        size = float(_optional_int(operator_config.get("size")) or 0)
        return 8.0 * size, float(DTYPE_BYTES) * (2.0 * size)

    raise ValueError(f"Unsupported runlist logical operator: {logical_operator}")


def _offload_operator_bytes_and_flops(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> tuple[float, float]:
    if logical_operator in OFFLOAD_OPERATOR_SHORT_LABELS:
        return _gemm_bytes_and_flops(operator_config)
    raise ValueError(f"Unsupported offload logical operator: {logical_operator}")


def operator_bytes_and_flops(
    *,
    execution_mode: str,
    logical_operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    operator_config: dict[str, object],
) -> tuple[float, float]:
    if execution_mode == "hybrid":
        return _hybrid_operator_bytes_and_flops(
            logical_operator=logical_operator,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            operator_config=operator_config,
        )
    if execution_mode == "runlist":
        return _runlist_operator_bytes_and_flops(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if execution_mode == "offload":
        return _offload_operator_bytes_and_flops(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _hybrid_operator_compute_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator == "qkv_proj":
        return _positive_config_int(
            operator_config,
            "parallel_seq",
        ) * _positive_config_int(operator_config, "parallel_emb")

    if logical_operator == "mha_out_proj":
        return _positive_config_int(
            operator_config,
            "parallel_seq",
        ) * _positive_config_int(operator_config, "parallel_heads")

    if logical_operator in {"add_norm1", "add_norm2", "add_norm"}:
        return _positive_config_int(operator_config, "num_aie_columns")

    if logical_operator in {"ln1", "add"}:
        return _positive_config_int(
            operator_config,
            "num_aie_columns",
        ) * _positive_config_int(operator_config, "num_channels")

    if logical_operator == "ffn":
        return _positive_config_int(operator_config, "num_aie_columns") * 2

    raise ValueError(f"Unsupported hybrid logical operator: {logical_operator}")


def _runlist_operator_compute_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator in {
        "qkvo_proj",
        "attn_scores",
        "attn_output",
        "up_proj",
        "down_proj",
    }:
        return _positive_config_int(operator_config, "num_aie_columns") * 4

    if logical_operator in {
        "k_transpose",
        "attn_scale",
        "causal_mask",
        "attn_softmax",
        "add",
        "ln1",
        "gelu",
        "ln2",
    }:
        return _positive_config_int(
            operator_config,
            "num_aie_columns",
        ) * _positive_config_int(operator_config, "num_channels")

    raise ValueError(f"Unsupported runlist logical operator: {logical_operator}")


def _offload_operator_compute_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator in OFFLOAD_OPERATOR_SHORT_LABELS:
        return _positive_config_int(operator_config, "num_aie_columns") * 4
    raise ValueError(f"Unsupported offload logical operator: {logical_operator}")


def operator_compute_tiles_used(
    *,
    execution_mode: str,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if execution_mode == "hybrid":
        return _hybrid_operator_compute_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if execution_mode == "runlist":
        return _runlist_operator_compute_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if execution_mode == "offload":
        return _offload_operator_compute_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _hybrid_operator_shim_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator == "qkv_proj":
        return _positive_config_int(operator_config, "parallel_emb")

    if logical_operator == "mha_out_proj":
        parallel_seq = _positive_config_int(operator_config, "parallel_seq")
        parallel_heads = _positive_config_int(operator_config, "parallel_heads")
        return max(1, parallel_seq * parallel_heads)

    if logical_operator in {"add_norm1", "add_norm2", "add_norm", "ffn"}:
        return _positive_config_int(operator_config, "num_aie_columns")

    if logical_operator in {"ln1", "add"}:
        return _positive_config_int(operator_config, "num_aie_columns")

    raise ValueError(f"Unsupported hybrid logical operator: {logical_operator}")


def _runlist_operator_shim_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator in {
        "qkvo_proj",
        "k_transpose",
        "attn_scores",
        "attn_scale",
        "causal_mask",
        "attn_softmax",
        "attn_output",
        "add",
        "ln1",
        "up_proj",
        "gelu",
        "down_proj",
        "ln2",
    }:
        return _positive_config_int(operator_config, "num_aie_columns")

    raise ValueError(f"Unsupported runlist logical operator: {logical_operator}")


def _offload_operator_shim_tiles(
    *,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if logical_operator in OFFLOAD_OPERATOR_SHORT_LABELS:
        return _positive_config_int(operator_config, "num_aie_columns")
    raise ValueError(f"Unsupported offload logical operator: {logical_operator}")


def operator_shim_tiles_used(
    *,
    execution_mode: str,
    logical_operator: str,
    operator_config: dict[str, object],
) -> int:
    if execution_mode == "hybrid":
        return _hybrid_operator_shim_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if execution_mode == "runlist":
        return _runlist_operator_shim_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    if execution_mode == "offload":
        return _offload_operator_shim_tiles(
            logical_operator=logical_operator,
            operator_config=operator_config,
        )
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _runlist_query_block_count(
    *,
    seq_len: int,
    selected_config: dict[str, dict[str, object]],
) -> int:
    attn_scores_config = selected_config.get("attn_scores", {})
    query_block_size = _optional_int(attn_scores_config.get("M")) or seq_len
    if query_block_size <= 0 or seq_len % query_block_size != 0:
        return 1
    return max(1, seq_len // query_block_size)


def _offload_query_block_count(
    *,
    seq_len: int,
    selected_config: dict[str, dict[str, object]],
) -> int:
    attn_scores_config = selected_config.get("attn_scores", {})
    query_block_size = (
        _optional_int(attn_scores_config.get("query_block_size"))
        or _optional_int(attn_scores_config.get("M"))
        or seq_len
    )
    if query_block_size <= 0 or seq_len % query_block_size != 0:
        return 1
    return max(1, seq_len // query_block_size)


def operator_multiplicity(
    *,
    execution_mode: str,
    logical_operator: str,
    seq_len: int,
    selected_config: dict[str, dict[str, object]],
) -> int:
    if execution_mode == "hybrid":
        return 1
    if execution_mode == "offload":
        if logical_operator in {"attn_scores", "attn_output"}:
            num_heads = (
                _optional_int(
                    selected_config.get(logical_operator, {}).get("num_heads")
                )
                or 1
            )
            return (
                _offload_query_block_count(
                    seq_len=seq_len,
                    selected_config=selected_config,
                )
                * num_heads
            )
        return 1

    if logical_operator == "qkvo_proj":
        return 4
    if logical_operator == "add":
        return 2
    if logical_operator in {
        "attn_scores",
        "attn_scale",
        "attn_softmax",
        "attn_output",
    }:
        return _runlist_query_block_count(
            seq_len=seq_len,
            selected_config=selected_config,
        )
    return 1


def _kernel_point_row(
    *,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    logical_operator: str,
    seq_len: int | None,
    compute_tiles_used: int | None,
    shim_tiles_used: int | None,
    candidate_id: str,
    run_status: str,
    omission_note: str,
    avg_latency_ms: float | None,
    measured_bandwidth_gbps: float | None,
    kernel_flop_count: float | None,
    kernel_byte_count: float | None,
) -> dict[str, object]:
    effective_gflops = None
    if (
        kernel_flop_count is not None
        and avg_latency_ms is not None
        and avg_latency_ms > 0.0
    ):
        effective_gflops = kernel_flop_count / (avg_latency_ms / 1000.0) / 1.0e9

    operational_intensity = None
    if (
        kernel_flop_count is not None
        and kernel_byte_count is not None
        and kernel_flop_count > 0.0
        and kernel_byte_count > 0.0
    ):
        operational_intensity = kernel_flop_count / kernel_byte_count

    plot_label = ""
    if seq_len is not None:
        plot_label = (
            f"{_operator_short_label(execution_mode, logical_operator)}@{seq_len}"
        )

    return {
        "study_case_id": study_case_id,
        "family_label": _csv_family_label(study_case_id),
        "workload_variant": workload_variant,
        "execution_mode": execution_mode,
        "logical_operator": logical_operator,
        "operator_short_label": _operator_short_label(execution_mode, logical_operator),
        "seq_len": "" if seq_len is None else seq_len,
        "compute_tiles_used": "" if compute_tiles_used is None else compute_tiles_used,
        "shim_tiles_used": "" if shim_tiles_used is None else shim_tiles_used,
        "candidate_id": candidate_id,
        "run_status": run_status,
        "omission_note": omission_note,
        "avg_latency_ms": avg_latency_ms,
        "measured_bandwidth_gbps": measured_bandwidth_gbps,
        "kernel_flop_count": kernel_flop_count,
        "kernel_byte_count": kernel_byte_count,
        "effective_gflops_per_sec": effective_gflops,
        "operational_intensity_flops_per_byte": operational_intensity,
        "plot_label": plot_label,
    }


def build_kernel_points(
    *,
    tuning_rows: list[dict[str, str]],
    workload_variant_filter: str = "all",
    family_filter: str = "all",
    mode_filter: str = "all",
    resource_usage_index: dict[ResourceUsageKey, ResourceUsageTileCounts] | None = None,
) -> list[dict[str, object]]:
    families = [
        family_id
        for family_id in (FAMILY_IDS if family_filter == "all" else (family_filter,))
        if workload_variant_filter == "all"
        or FAMILY_SPECS[family_id].workload_variant == workload_variant_filter
    ]
    modes = (
        EXECUTION_MODES
        if mode_filter == "all"
        else (canonical_execution_mode(mode_filter),)
    )
    rows: list[dict[str, object]] = []

    for study_case_id in families:
        workload_variant = FAMILY_SPECS[study_case_id].workload_variant
        for execution_mode in modes:
            for logical_operator in mode_operators(execution_mode, workload_variant):
                operator_rows = [
                    row
                    for row in tuning_rows
                    if row.get("study_case_id") == study_case_id
                    and _normalized_execution_mode(row.get("execution_mode"))
                    == execution_mode
                    and row.get("internal_operator") == logical_operator
                ]
                measured_rows: list[dict[str, object]] = []
                for row in operator_rows:
                    if row.get("run_status") != "passed":
                        continue
                    avg_latency_ms = _optional_float(row.get("avg_latency_ms"))
                    if avg_latency_ms is None or avg_latency_ms <= 0.0:
                        continue
                    seq_len = _optional_int(row.get("seq_len"))
                    hidden_size = _optional_int(row.get("hidden_size"))
                    intermediate_size = _optional_int(row.get("intermediate_size"))
                    operator_config = _load_json_dict(row.get("operator_config_json"))
                    if None in (seq_len, hidden_size, intermediate_size):
                        continue
                    candidate_id = str(row.get("candidate_id") or "")
                    flop_count, byte_count = operator_bytes_and_flops(
                        execution_mode=execution_mode,
                        logical_operator=logical_operator,
                        seq_len=int(seq_len),
                        hidden_size=int(hidden_size),
                        intermediate_size=int(intermediate_size),
                        operator_config=operator_config,
                    )
                    effective_gflops = flop_count / (avg_latency_ms / 1000.0) / 1.0e9
                    compute_tiles_used, shim_tiles_used = resolved_operator_tile_counts(
                        execution_mode=execution_mode,
                        study_case_id=study_case_id,
                        workload_variant=workload_variant,
                        logical_operator=logical_operator,
                        seq_len=int(seq_len),
                        candidate_id=candidate_id,
                        operator_config=operator_config,
                        resource_usage_index=resource_usage_index,
                    )
                    measured_rows.append(
                        {
                            "row": row,
                            "seq_len": int(seq_len),
                            "compute_tiles_used": compute_tiles_used,
                            "shim_tiles_used": shim_tiles_used,
                            "avg_latency_ms": avg_latency_ms,
                            "measured_bandwidth_gbps": _optional_float(
                                row.get("bandwidth_gbps")
                            ),
                            "flop_count": flop_count,
                            "byte_count": byte_count,
                            "effective_gflops_per_sec": effective_gflops,
                        }
                    )

                if measured_rows:
                    best = max(
                        measured_rows,
                        key=lambda entry: (
                            float(entry["effective_gflops_per_sec"]),
                            -float(entry["avg_latency_ms"]),
                        ),
                    )
                    omission_note = ""
                    if float(best["flop_count"]) <= 0.0:
                        omission_note = (
                            "zero-flop kernel omitted from log-scale roofline"
                        )
                    rows.append(
                        _kernel_point_row(
                            study_case_id=study_case_id,
                            workload_variant=workload_variant,
                            execution_mode=execution_mode,
                            logical_operator=logical_operator,
                            seq_len=int(best["seq_len"]),
                            compute_tiles_used=int(best["compute_tiles_used"]),
                            shim_tiles_used=int(best["shim_tiles_used"]),
                            candidate_id=str(best["row"].get("candidate_id") or ""),
                            run_status="passed",
                            omission_note=omission_note,
                            avg_latency_ms=float(best["avg_latency_ms"]),
                            measured_bandwidth_gbps=best["measured_bandwidth_gbps"],
                            kernel_flop_count=float(best["flop_count"]),
                            kernel_byte_count=float(best["byte_count"]),
                        )
                    )
                    continue

                omission_note = "missing tuning rows"
                candidate_id = ""
                seq_len = None
                compute_tiles_used = None
                shim_tiles_used = None
                if operator_rows:
                    statuses = sorted(
                        {str(row.get("run_status") or "") for row in operator_rows}
                    )
                    omission_note = (
                        f"no measured isolated throughput row ({', '.join(statuses)})"
                    )
                    candidate_id = str(operator_rows[0].get("candidate_id") or "")
                    seq_len = _optional_int(operator_rows[0].get("seq_len"))
                    if seq_len is not None:
                        compute_tiles_used, shim_tiles_used = (
                            resolved_operator_tile_counts(
                                execution_mode=execution_mode,
                                study_case_id=study_case_id,
                                workload_variant=workload_variant,
                                logical_operator=logical_operator,
                                seq_len=int(seq_len),
                                candidate_id=candidate_id,
                                operator_config=_load_json_dict(
                                    operator_rows[0].get("operator_config_json")
                                ),
                                resource_usage_index=resource_usage_index,
                            )
                        )
                rows.append(
                    _kernel_point_row(
                        study_case_id=study_case_id,
                        workload_variant=workload_variant,
                        execution_mode=execution_mode,
                        logical_operator=logical_operator,
                        seq_len=seq_len,
                        compute_tiles_used=compute_tiles_used,
                        shim_tiles_used=shim_tiles_used,
                        candidate_id=candidate_id,
                        run_status="missing_measurement",
                        omission_note=omission_note,
                        avg_latency_ms=None,
                        measured_bandwidth_gbps=None,
                        kernel_flop_count=None,
                        kernel_byte_count=None,
                    )
                )

    return rows


def _implementation_point_row(
    *,
    selected_row,
    compute_tiles_used: int | None,
    shim_tiles_used: int | None,
    avg_latency_ms: float | None,
    effective_gflops_per_sec: float | None,
    implementation_flop_count: float,
    implementation_byte_count: float | None,
    run_status: str,
    omission_note: str,
) -> dict[str, object]:
    operational_intensity = None
    if implementation_byte_count is not None and implementation_byte_count > 0.0:
        operational_intensity = implementation_flop_count / implementation_byte_count

    return {
        "study_case_id": selected_row.study_case_id,
        "family_label": _csv_family_label(selected_row.study_case_id),
        "workload_variant": selected_row.workload_variant,
        "execution_mode": selected_row.execution_mode,
        "seq_len": selected_row.seq_len,
        "compute_tiles_used": "" if compute_tiles_used is None else compute_tiles_used,
        "shim_tiles_used": "" if shim_tiles_used is None else shim_tiles_used,
        "run_status": run_status,
        "omission_note": omission_note,
        "avg_latency_ms": avg_latency_ms,
        "effective_gflops_per_sec": effective_gflops_per_sec,
        "implementation_flop_count": implementation_flop_count,
        "implementation_byte_count": implementation_byte_count,
        "operational_intensity_flops_per_byte": operational_intensity,
        "selected_candidate_ids_json": json.dumps(
            selected_row.selected_candidate_ids,
            sort_keys=True,
        ),
        "plot_label": f"{MODE_LABELS[selected_row.execution_mode]}@{selected_row.seq_len}",
    }


def build_implementation_points(
    *,
    result_rows: list[dict[str, str]],
    workload_variant_filter: str = "all",
    family_filter: str = "all",
    mode_filter: str = "all",
    resource_usage_index: dict[ResourceUsageKey, ResourceUsageTileCounts] | None = None,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        result_rows,
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter="all",
        mode_filter=mode_filter,
    )

    best_rows: dict[tuple[str, str], object] = {}
    for selected_row in selected_rows:
        effective_gflops = _optional_float(
            selected_row.row.get("effective_gflops_per_sec")
        )
        if effective_gflops is None:
            continue
        key = (selected_row.study_case_id, selected_row.execution_mode)
        current_best = best_rows.get(key)
        if current_best is None or effective_gflops > _optional_float(
            current_best.row.get("effective_gflops_per_sec")
        ):
            best_rows[key] = selected_row

    rows: list[dict[str, object]] = []
    for selected_row in best_rows.values():
        implementation_flops = float(
            effective_dense_layer_flop_count(
                seq_len=selected_row.seq_len,
                hidden_size=selected_row.hidden_size,
                intermediate_size=selected_row.intermediate_size,
                num_attention_heads=selected_row.num_attention_heads,
            )
        )
        total_bytes = 0.0
        compute_tiles: list[int] = []
        shim_tiles: list[int] = []
        omission_note = ""
        for logical_operator in mode_operators(
            selected_row.execution_mode,
            selected_row.workload_variant,
        ):
            operator_config = selected_row.selected_config.get(logical_operator)
            if not isinstance(operator_config, dict):
                omission_note = f"missing selected config for {logical_operator}"
                total_bytes = 0.0
                break
            candidate_id = str(
                selected_row.selected_candidate_ids.get(logical_operator) or ""
            )
            compute_tiles_used, shim_tiles_used = resolved_operator_tile_counts(
                execution_mode=selected_row.execution_mode,
                study_case_id=selected_row.study_case_id,
                workload_variant=selected_row.workload_variant,
                logical_operator=logical_operator,
                seq_len=selected_row.seq_len,
                candidate_id=candidate_id,
                operator_config=operator_config,
                resource_usage_index=resource_usage_index,
            )
            if compute_tiles_used is not None:
                compute_tiles.append(compute_tiles_used)
            if shim_tiles_used is not None:
                shim_tiles.append(shim_tiles_used)
            _, byte_count = operator_bytes_and_flops(
                execution_mode=selected_row.execution_mode,
                logical_operator=logical_operator,
                seq_len=selected_row.seq_len,
                hidden_size=selected_row.hidden_size,
                intermediate_size=selected_row.intermediate_size,
                operator_config=operator_config,
            )
            total_bytes += byte_count * float(
                operator_multiplicity(
                    execution_mode=selected_row.execution_mode,
                    logical_operator=logical_operator,
                    seq_len=selected_row.seq_len,
                    selected_config=selected_row.selected_config,
                )
            )

        rows.append(
            _implementation_point_row(
                selected_row=selected_row,
                compute_tiles_used=max(compute_tiles, default=None),
                shim_tiles_used=max(shim_tiles, default=None),
                avg_latency_ms=_optional_float(selected_row.row.get("avg_latency_ms")),
                effective_gflops_per_sec=_optional_float(
                    selected_row.row.get("effective_gflops_per_sec")
                ),
                implementation_flop_count=implementation_flops,
                implementation_byte_count=total_bytes if total_bytes > 0.0 else None,
                run_status="passed" if total_bytes > 0.0 else "missing_measurement",
                omission_note=omission_note,
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            FAMILY_IDS.index(str(row["study_case_id"])),
            EXECUTION_MODES.index(str(row["execution_mode"])),
        ),
    )


def _ordered_families(rows: list[dict[str, object]]) -> list[str]:
    present = {
        str(row.get("study_case_id") or "")
        for row in rows
        if str(row.get("study_case_id") or "")
    }
    return [family_id for family_id in FAMILY_IDS if family_id in present]


def _positive_points(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in rows:
        throughput = _optional_float(row.get("effective_gflops_per_sec"))
        intensity = _optional_float(row.get("operational_intensity_flops_per_byte"))
        if throughput is None or throughput <= 0.0:
            continue
        if intensity is None or intensity <= 0.0:
            continue
        points.append(row)
    return points


def _compute_tile_group_value(row: dict[str, object]) -> int | None:
    value = _optional_int(row.get("compute_tiles_used"))
    if value is None or value <= 0:
        return None
    return value


def _shim_tile_group_value(row: dict[str, object]) -> int | None:
    value = _optional_int(row.get("shim_tiles_used"))
    if value is None or value <= 0:
        return None
    return value


def _group_rows_by_tile_usage(
    rows: list[dict[str, object]],
) -> list[tuple[tuple[int | None, int | None], list[dict[str, object]]]]:
    grouped: dict[tuple[int | None, int | None], list[dict[str, object]]] = {}
    for row in rows:
        key = (_compute_tile_group_value(row), _shim_tile_group_value(row))
        grouped.setdefault(key, []).append(row)
    return sorted(
        grouped.items(),
        key=lambda entry: (
            entry[0][0] is None,
            -1 if entry[0][0] is None else int(entry[0][0]),
            entry[0][1] is None,
            -1 if entry[0][1] is None else int(entry[0][1]),
        ),
    )


def _tile_group_title_suffix(
    compute_tiles_used: int | None,
    shim_tiles_used: int | None,
) -> str:
    parts: list[str] = []
    if compute_tiles_used is None:
        parts.append("Unknown Compute-Tile Count")
    else:
        plural = "" if compute_tiles_used == 1 else "s"
        parts.append(f"{compute_tiles_used} Compute Tile{plural}")
    if shim_tiles_used is None:
        parts.append("Unknown Shim-Tile Count")
    else:
        plural = "" if shim_tiles_used == 1 else "s"
        parts.append(f"{shim_tiles_used} Shim Tile{plural}")
    return ", ".join(parts)


def _tile_group_plot_path(
    base_plot_path: Path,
    compute_tiles_used: int | None,
    shim_tiles_used: int | None,
) -> Path:
    compute_suffix = (
        "unknown_compute_tiles"
        if compute_tiles_used is None
        else f"{int(compute_tiles_used):02d}_compute_tiles"
    )
    shim_suffix = (
        "unknown_shim_tiles"
        if shim_tiles_used is None
        else f"{int(shim_tiles_used):02d}_shim_tiles"
    )
    return base_plot_path.with_name(
        f"{base_plot_path.stem}_{compute_suffix}_{shim_suffix}{base_plot_path.suffix}"
    )


def theoretical_compute_peak_gflops_per_sec() -> float:
    return (
        float(NPU_AIE_CORES)
        * float(BF16_MACS_PER_CYCLE_PER_CORE)
        * float(FLOPS_PER_MAC)
        * float(NPU_AIE_CORE_CLOCK_HZ)
        / 1.0e9
    )


def scaled_compute_peak_gflops_per_sec(compute_tiles_used: int | None) -> float:
    full_peak = theoretical_compute_peak_gflops_per_sec()
    if compute_tiles_used is None or compute_tiles_used <= 0:
        return full_peak
    return full_peak * float(compute_tiles_used) / float(NPU_AIE_CORES)


def _render_empty_figure(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_axis_off()
    fig.patch.set_facecolor("#f7f5f2")
    fig.text(
        0.5,
        0.57,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.42,
        "No data available",
        ha="center",
        va="center",
        fontsize=18,
    )
    return fig


def render_roofline_plot(
    rows: list[dict[str, object]],
    *,
    title: str,
    subtitle: str,
    peak_bandwidth_gbps: float,
    compute_ceiling_gflops: float,
) -> plt.Figure:
    points = _positive_points(rows)
    if not points:
        return _render_empty_figure(title)

    families = _ordered_families(points)
    compute_ceiling = compute_ceiling_gflops

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fcfbf8",
            "grid.color": "#ded8cf",
        },
    )
    plt.rcParams["svg.fonttype"] = "none"

    fig, axes_obj = plt.subplots(
        *PLOT_FAMILY_GRID_SHAPE,
        figsize=(19, 10),
        sharey=True,
    )
    axes_grid = axes_obj
    present_families = set(families)

    for family in ordered_plot_families():
        ax = axes_grid[family.row_index][family.col_index]
        family_id = family.family_id
        if family_id not in present_families:
            ax.set_axis_off()
            continue

        family_points = [
            row for row in points if str(row.get("study_case_id")) == family_id
        ]
        if not family_points:
            ax.set_axis_off()
            continue

        intensities = [
            float(row["operational_intensity_flops_per_byte"]) for row in family_points
        ]
        throughputs = [float(row["effective_gflops_per_sec"]) for row in family_points]
        min_intensity = min(intensities)
        max_intensity = max(intensities)
        x_lower = max(1.0e-4, min_intensity / 2.0)
        x_upper = max(
            max_intensity * 2.0, (compute_ceiling / peak_bandwidth_gbps) * 2.0
        )
        x_values = np.logspace(math.log10(x_lower), math.log10(x_upper), 256)
        roof_y = np.minimum(peak_bandwidth_gbps * x_values, compute_ceiling)

        ax.plot(
            x_values,
            roof_y,
            color="#3d405b",
            linewidth=2.4,
            label="Roofline",
        )

        for row in family_points:
            execution_mode = str(row["execution_mode"])
            intensity = float(row["operational_intensity_flops_per_byte"])
            throughput = float(row["effective_gflops_per_sec"])
            ax.scatter(
                intensity,
                throughput,
                s=80,
                color=MODE_COLORS[execution_mode],
                marker=MODE_MARKERS[execution_mode],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                str(row.get("plot_label") or ""),
                (intensity, throughput),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                color="#2f2a24",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(plot_family_label(family_id), fontsize=15, pad=14)
        ax.set_xlabel("Operational Intensity (FLOP/byte)", fontsize=11)
        ax.set_xlim(left=x_lower, right=x_upper)
        ax.grid(True, which="major", alpha=0.55)
        ax.grid(True, which="minor", alpha=0.18)
        ax.text(
            0.03,
            0.03,
            f"Peak BW = {peak_bandwidth_gbps:.2f} GB/s\nPeak Compute = {compute_ceiling:.2f} GFLOPS",
            transform=ax.transAxes,
            fontsize=9,
            color="#4a433b",
            va="bottom",
        )
        if family.col_index != 0:
            ax.set_ylabel("")

    axes_grid[0][0].set_ylabel("Effective GFLOPS", fontsize=11)

    legend_handles = [
        Line2D([0], [0], color="#3d405b", linewidth=2.4, label="Roofline"),
        *[
            Line2D(
                [0],
                [0],
                marker=MODE_MARKERS[mode],
                linestyle="",
                markersize=8,
                markerfacecolor=MODE_COLORS[mode],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=MODE_LABELS[mode],
            )
            for mode in EXECUTION_MODES
        ],
    ]

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=11)
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.1, wspace=0.22)
    return fig


def write_rows(
    output_path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_figure(fig: plt.Figure, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(PNG_SUFFIX), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _remove_stale_base_plot(base_plot_path: Path) -> None:
    for candidate in (base_plot_path, base_plot_path.with_suffix(PNG_SUFFIX)):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def write_grouped_roofline_plots(
    rows: list[dict[str, object]],
    *,
    title_prefix: str,
    subtitle: str,
    peak_bandwidth_gbps: float,
    peak_bandwidth_by_shim_tiles_gbps: dict[int, float],
    base_plot_path: Path,
) -> list[Path]:
    grouped_rows = _group_rows_by_tile_usage(_positive_points(rows))
    if not grouped_rows:
        fig = _render_empty_figure(title_prefix)
        _write_figure(fig, base_plot_path)
        return [base_plot_path]

    output_paths: list[Path] = []
    for (compute_tiles_used, shim_tiles_used), group_rows in grouped_rows:
        output_path = _tile_group_plot_path(
            base_plot_path,
            compute_tiles_used,
            shim_tiles_used,
        )
        group_peak_bandwidth = peak_bandwidth_gbps
        if shim_tiles_used is not None and shim_tiles_used > 0:
            group_peak_bandwidth = peak_bandwidth_by_shim_tiles_gbps.get(
                shim_tiles_used,
                peak_bandwidth_gbps * float(shim_tiles_used) / float(NPU_SHIM_TILES),
            )
        fig = render_roofline_plot(
            group_rows,
            title=f"{title_prefix}: {_tile_group_title_suffix(compute_tiles_used, shim_tiles_used)}",
            subtitle=subtitle,
            peak_bandwidth_gbps=group_peak_bandwidth,
            compute_ceiling_gflops=scaled_compute_peak_gflops_per_sec(
                compute_tiles_used
            ),
        )
        _write_figure(fig, output_path)
        output_paths.append(output_path)

    _remove_stale_base_plot(base_plot_path)
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roofline summaries from end-to-end and memcpy study outputs."
    )
    parser.add_argument(
        "--workload-variant",
        choices=[*WORKLOAD_VARIANTS, "all"],
        default="all",
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
    )
    parser.add_argument(
        "--end-to-end-results-input",
        type=Path,
        default=default_end_to_end_results_path(),
    )
    parser.add_argument(
        "--end-to-end-tuning-input",
        type=Path,
        default=default_end_to_end_tuning_path(),
    )
    parser.add_argument(
        "--memcpy-results-input",
        type=Path,
        default=default_memcpy_results_path(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
    )
    parser.add_argument("--kernel-points-output", type=Path, default=None)
    parser.add_argument("--implementation-points-output", type=Path, default=None)
    parser.add_argument("--kernel-plot", type=Path, default=None)
    parser.add_argument("--implementation-plot", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    if not args.end_to_end_results_input.exists():
        raise FileNotFoundError(
            f"Missing end-to-end results CSV: {args.end_to_end_results_input}"
        )
    if not args.end_to_end_tuning_input.exists():
        raise FileNotFoundError(
            f"Missing end-to-end tuning CSV: {args.end_to_end_tuning_input}"
        )
    if not args.memcpy_results_input.exists():
        raise FileNotFoundError(
            "Missing memcpy-bandwidth CSV: "
            f"{args.memcpy_results_input}. Run study.memcpy_bandwidth.run first."
        )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_points_output = (
        default_kernel_points_path(output_dir)
        if args.kernel_points_output is None
        else args.kernel_points_output.expanduser()
    )
    implementation_points_output = (
        default_implementation_points_path(output_dir)
        if args.implementation_points_output is None
        else args.implementation_points_output.expanduser()
    )
    kernel_plot = _resolve_plot_path(
        default_kernel_plot_path(output_dir)
        if args.kernel_plot is None
        else args.kernel_plot.expanduser()
    )
    implementation_plot = _resolve_plot_path(
        default_implementation_plot_path(output_dir)
        if args.implementation_plot is None
        else args.implementation_plot.expanduser()
    )

    result_rows = load_result_rows(args.end_to_end_results_input)
    tuning_rows = load_csv_rows(args.end_to_end_tuning_input)
    memcpy_rows = load_csv_rows(args.memcpy_results_input)
    resource_usage_index = load_resource_usage_index(
        default_resource_usage_selected_ops_paths(args.end_to_end_results_input)
    )
    peak_bandwidth = peak_memcpy_bandwidth_gbps(memcpy_rows)
    peak_bandwidth_by_shim_tiles = peak_memcpy_bandwidth_by_shim_tiles_gbps(memcpy_rows)

    kernel_points = build_kernel_points(
        tuning_rows=tuning_rows,
        workload_variant_filter=args.workload_variant,
        family_filter=args.family,
        resource_usage_index=resource_usage_index,
    )
    implementation_points = build_implementation_points(
        result_rows=result_rows,
        workload_variant_filter=args.workload_variant,
        family_filter=args.family,
        resource_usage_index=resource_usage_index,
    )

    write_rows(
        kernel_points_output,
        fieldnames=KERNEL_POINTS_FIELDNAMES,
        rows=kernel_points,
    )
    write_rows(
        implementation_points_output,
        fieldnames=IMPLEMENTATION_POINTS_FIELDNAMES,
        rows=implementation_points,
    )

    kernel_plot_paths = write_grouped_roofline_plots(
        kernel_points,
        title_prefix="Kernel Roofline",
        subtitle="Point labels show operator@sequence-length for the best measured isolated configuration",
        peak_bandwidth_gbps=peak_bandwidth,
        peak_bandwidth_by_shim_tiles_gbps=peak_bandwidth_by_shim_tiles,
        base_plot_path=kernel_plot,
    )

    implementation_plot_paths = write_grouped_roofline_plots(
        implementation_points,
        title_prefix="Implementation Roofline",
        subtitle="Point labels show execution-mode@sequence-length for the best measured full-pattern configuration",
        peak_bandwidth_gbps=peak_bandwidth,
        peak_bandwidth_by_shim_tiles_gbps=peak_bandwidth_by_shim_tiles,
        base_plot_path=implementation_plot,
    )

    LOGGER.info(
        "Wrote %d kernel roofline rows to %s", len(kernel_points), kernel_points_output
    )
    LOGGER.info(
        "Wrote %d implementation roofline rows to %s",
        len(implementation_points),
        implementation_points_output,
    )
    LOGGER.info(
        "Wrote roofline plots to %s and %s",
        ", ".join(str(path) for path in kernel_plot_paths),
        ", ".join(str(path) for path in implementation_plot_paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
