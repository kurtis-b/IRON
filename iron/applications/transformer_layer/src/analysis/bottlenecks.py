#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

COMPONENT_FIELDS = {
    "block1_qkv_proj": "avg_block1_qkv_proj_latency_ms",
    "block2_mha_out_proj": "avg_block2_mha_out_proj_latency_ms",
    "block3_addnorm_ffn_addnorm": "avg_block3_addnorm_ffn_addnorm_latency_ms",
    "npu_projection": "avg_npu_projection_latency_ms",
    "npu_gemm": "avg_npu_gemm_latency_ms",
    "runlist": "avg_runlist_latency_ms",
    "host_preprocess": "avg_host_preprocess_latency_ms",
    "host_postprocess": "avg_host_postprocess_latency_ms",
    "device_sync": "avg_device_sync_latency_ms",
    "gemm_sequence": "avg_gemm_sequence_latency_ms",
}

SUMMARY_FIELD_ORDER = [
    "study_id",
    "study_case_id",
    "study_case_label",
    "execution_mode",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "block1_topology_id",
    "block1_topology_family",
    "block2_topology_id",
    "block2_topology_family",
    "block3_topology_id",
    "block3_topology_family",
    "avg_latency_ms",
    "dominant_component",
    "dominant_component_latency_ms",
    "dominant_component_fraction",
    "secondary_component",
    "secondary_component_latency_ms",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "process_model",
    "bottleneck_summary",
]


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _nonempty_str(value: object) -> str | None:
    if value in (None, "", "None"):
        return None
    return str(value)


def _load_result_rows(input_csv: str | Path) -> list[dict[str, object]]:
    with Path(input_csv).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _component_latencies_ms(row: dict[str, object]) -> dict[str, float]:
    latencies = {}
    for component, field_name in COMPONENT_FIELDS.items():
        value = _optional_float(row.get(field_name))
        if value is not None:
            latencies[component] = value
    return latencies


def _component_order(latencies: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(latencies.items(), key=lambda item: item[1], reverse=True)


def _row_context(row: dict[str, object]) -> str:
    dispatch_count = _optional_int(row.get("npu_dispatch_count"))
    unique_binaries = _optional_int(row.get("npu_unique_instruction_binary_count"))
    unique_xclbins = _optional_int(row.get("npu_unique_xclbin_count"))
    process_model = row.get("process_model")
    parts = []
    if dispatch_count is not None:
        parts.append(f"{dispatch_count} dispatches")
    if unique_binaries is not None:
        parts.append(f"{unique_binaries} instruction binaries")
    if unique_xclbins is not None:
        parts.append(f"{unique_xclbins} xclbins")
    if process_model not in (None, "", "None"):
        parts.append(f"process={process_model}")
    return ", ".join(parts)


def build_row_summary(row: dict[str, object]) -> dict[str, object]:
    total_latency_ms = _optional_float(row.get("avg_latency_ms"))
    components = _component_order(_component_latencies_ms(row))
    dominant_component = None
    dominant_latency_ms = None
    dominant_fraction = None
    secondary_component = None
    secondary_latency_ms = None
    if components:
        dominant_component, dominant_latency_ms = components[0]
        if total_latency_ms not in (None, 0.0):
            dominant_fraction = dominant_latency_ms / total_latency_ms
        if len(components) > 1:
            secondary_component, secondary_latency_ms = components[1]
    context = _row_context(row)
    summary = (
        f"{dominant_component or 'unknown'} dominates"
        if dominant_component is not None
        else "No staged latency breakdown available"
    )
    if dominant_latency_ms is not None and dominant_fraction is not None:
        summary = (
            f"{summary} at {dominant_latency_ms:.3f} ms "
            f"({dominant_fraction:.1%} of total)"
        )
    elif dominant_latency_ms is not None:
        summary = f"{summary} at {dominant_latency_ms:.3f} ms"
    if context:
        summary = f"{summary}; {context}"
    return {
        "study_id": row.get("study_id"),
        "study_case_id": row.get("study_case_id"),
        "study_case_label": row.get("study_case_label"),
        "execution_mode": row.get("execution_mode"),
        "seq_len": _optional_int(row.get("seq_len")),
        "hidden_size": _optional_int(row.get("hidden_size")),
        "intermediate_size": _optional_int(row.get("intermediate_size")),
        "num_attention_heads": _optional_int(row.get("num_attention_heads")),
        "block1_topology_id": row.get("block1_topology_id"),
        "block1_topology_family": row.get("block1_topology_family"),
        "block2_topology_id": row.get("block2_topology_id"),
        "block2_topology_family": row.get("block2_topology_family"),
        "block3_topology_id": row.get("block3_topology_id"),
        "block3_topology_family": row.get("block3_topology_family"),
        "avg_latency_ms": total_latency_ms,
        "dominant_component": dominant_component,
        "dominant_component_latency_ms": dominant_latency_ms,
        "dominant_component_fraction": dominant_fraction,
        "secondary_component": secondary_component,
        "secondary_component_latency_ms": secondary_latency_ms,
        "npu_dispatch_count": _optional_int(row.get("npu_dispatch_count")),
        "npu_unique_instruction_binary_count": _optional_int(
            row.get("npu_unique_instruction_binary_count")
        ),
        "npu_unique_xclbin_count": _optional_int(row.get("npu_unique_xclbin_count")),
        "process_model": row.get("process_model"),
        "bottleneck_summary": summary,
    }


def build_execution_mode_summary(
    summary_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        case_id = row.get("study_case_id")
        execution_mode = row.get("execution_mode")
        key = (
            f"{case_id}:{execution_mode}"
            if case_id not in (None, "", "None")
            else str(execution_mode)
        )
        by_mode[key].append(row)

    output: dict[str, dict[str, object]] = {}
    for mode_key, rows in sorted(by_mode.items()):
        component_counts = Counter(
            row["dominant_component"]
            for row in rows
            if row["dominant_component"] not in (None, "")
        )
        avg_latency_values = [
            _optional_float(row.get("avg_latency_ms"))
            for row in rows
            if _optional_float(row.get("avg_latency_ms")) is not None
        ]
        avg_total_latency_ms = (
            sum(avg_latency_values) / len(avg_latency_values)
            if avg_latency_values
            else None
        )
        dominant_fraction_values = [
            _optional_float(row.get("dominant_component_fraction"))
            for row in rows
            if _optional_float(row.get("dominant_component_fraction")) is not None
        ]
        average_dominant_fraction = (
            sum(dominant_fraction_values) / len(dominant_fraction_values)
            if dominant_fraction_values
            else None
        )
        output[mode_key] = {
            "study_case_id": rows[0].get("study_case_id"),
            "study_case_label": rows[0].get("study_case_label"),
            "execution_mode": rows[0].get("execution_mode"),
            "row_count": len(rows),
            "seq_lens": sorted(
                {
                    int(seq_len)
                    for seq_len in (_optional_int(row.get("seq_len")) for row in rows)
                    if seq_len is not None
                }
            ),
            "dominant_component_counts": dict(component_counts),
            "average_total_latency_ms": avg_total_latency_ms,
            "average_dominant_component_fraction": average_dominant_fraction,
            "max_dispatch_count": max(
                (_optional_int(row.get("npu_dispatch_count")) or 0 for row in rows),
                default=0,
            ),
            "process_models": sorted(
                {
                    str(row["process_model"])
                    for row in rows
                    if row.get("process_model") not in (None, "", "None")
                }
            ),
            "block1_topology_ids": sorted(
                {
                    topology_id
                    for topology_id in (
                        _nonempty_str(row.get("block1_topology_id")) for row in rows
                    )
                    if topology_id is not None
                }
            ),
            "block2_topology_ids": sorted(
                {
                    topology_id
                    for topology_id in (
                        _nonempty_str(row.get("block2_topology_id")) for row in rows
                    )
                    if topology_id is not None
                }
            ),
            "block3_topology_ids": sorted(
                {
                    topology_id
                    for topology_id in (
                        _nonempty_str(row.get("block3_topology_id")) for row in rows
                    )
                    if topology_id is not None
                }
            ),
        }
    return output


def render_execution_mode_summaries(
    summary_rows: list[dict[str, object]],
    aggregates: dict[str, dict[str, object]],
) -> str:
    rows_by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        case_id = row.get("study_case_id")
        execution_mode = row.get("execution_mode")
        key = (
            f"{case_id}:{execution_mode}"
            if case_id not in (None, "", "None")
            else str(execution_mode)
        )
        rows_by_mode[key].append(row)

    lines = []
    for mode_key in sorted(rows_by_mode):
        aggregate = aggregates[mode_key]
        counts = aggregate["dominant_component_counts"]
        if counts:
            dominant_component, dominant_count = max(
                counts.items(), key=lambda item: item[1]
            )
            dominant_text = f"{dominant_component} dominates {dominant_count}/{aggregate['row_count']} rows"
        else:
            dominant_text = "no dominant component could be derived"
        avg_fraction = aggregate["average_dominant_component_fraction"]
        fraction_text = (
            f", average dominant share {avg_fraction:.1%}"
            if avg_fraction is not None
            else ""
        )
        topology_parts = []
        for block_key in ("block1", "block2", "block3"):
            topology_ids = aggregate.get(f"{block_key}_topology_ids") or []
            if topology_ids:
                topology_parts.append(f"{block_key}={','.join(topology_ids)}")
        topology_text = (
            f"; topologies={' '.join(topology_parts)}" if topology_parts else ""
        )
        seq_lens = ",".join(str(seq_len) for seq_len in aggregate["seq_lens"])
        example_row = rows_by_mode[mode_key][0]
        prefix = str(aggregate.get("execution_mode"))
        case_label = aggregate.get("study_case_label")
        if case_label not in (None, "", "None"):
            prefix = f"{case_label}:{prefix}"
        lines.append(
            f"{prefix}: {dominant_text}{fraction_text}; seq_lens={seq_lens}; "
            f"example={example_row['bottleneck_summary']}{topology_text}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def analyze_results(
    input_csv: str | Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], str]:
    input_rows = [
        row
        for row in _load_result_rows(input_csv)
        if row.get("run_status") in (None, "", "None", "completed")
    ]
    if not input_rows:
        raise ValueError(f"No suite rows found in {input_csv}")
    summary_rows = [build_row_summary(row) for row in input_rows]
    aggregates = build_execution_mode_summary(summary_rows)
    text_summary = render_execution_mode_summaries(summary_rows, aggregates)
    return summary_rows, aggregates, text_summary
