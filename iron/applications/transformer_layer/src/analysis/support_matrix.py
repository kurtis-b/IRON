#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

SUPPORT_MATRIX_FIELD_ORDER = [
    "study_id",
    "study_case_id",
    "study_case_label",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "seq_len",
    "execution_mode",
    "run_status",
    "failure_category",
    "failure_message",
]


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def summarize_support_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summarized = []
    for row in rows:
        summarized.append(
            {
                "study_id": row.get("study_id"),
                "study_case_id": row.get("study_case_id"),
                "study_case_label": row.get("study_case_label"),
                "hidden_size": _optional_int(row.get("hidden_size")),
                "intermediate_size": _optional_int(row.get("intermediate_size")),
                "num_attention_heads": _optional_int(row.get("num_attention_heads")),
                "attention_head_size": _optional_int(row.get("attention_head_size")),
                "seq_len": _optional_int(row.get("seq_len")),
                "execution_mode": row.get("execution_mode"),
                "run_status": row.get("run_status", "completed"),
                "failure_category": row.get("failure_category"),
                "failure_message": row.get("failure_message"),
            }
        )
    summarized.sort(
        key=lambda row: (
            str(row.get("study_case_id") or ""),
            int(row.get("seq_len") or 0),
            str(row.get("execution_mode") or ""),
        )
    )
    return summarized


def render_support_summary_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    lines = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("study_case_id") or "default"), []).append(row)
    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        case_label = case_rows[0].get("study_case_label") or case_id
        hidden_size = case_rows[0].get("hidden_size")
        intermediate_size = case_rows[0].get("intermediate_size")
        num_heads = case_rows[0].get("num_attention_heads")
        lines.append(f"{case_label} ({hidden_size}/{intermediate_size}/{num_heads})")
        for row in sorted(
            case_rows,
            key=lambda item: (
                int(item.get("seq_len") or 0),
                str(item.get("execution_mode") or ""),
            ),
        ):
            run_status = row.get("run_status", "completed")
            detail = run_status
            if run_status != "completed" and row.get("failure_category"):
                detail = f"{detail} ({row['failure_category']})"
            lines.append(
                f"  seq_len={row.get('seq_len')}: {row.get('execution_mode')} -> {detail}"
            )
    return "\n".join(lines) + "\n"
