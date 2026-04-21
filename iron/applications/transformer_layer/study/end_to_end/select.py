#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..campaign import (
    DEFAULT_CAMPAIGN_ID,
    normalize_campaign_id,
    normalize_matched_run_id,
    normalize_repeat_index,
    optional_int,
)
from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    SEQUENCE_LADDER,
    canonical_execution_mode,
    canonical_workload_variant,
)
from .validation import REFERENCE_TOLERANCE_VALIDATION_MODE

_MODE_ORDER = {
    execution_mode: index for index, execution_mode in enumerate(EXECUTION_MODES)
}


def _normalized_execution_mode(value: object) -> str | None:
    try:
        return canonical_execution_mode(str(value or ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class SelectedEndToEndRow:
    campaign_id: str
    repeat_index: int
    matched_run_id: str
    study_case_id: str
    study_case_label: str
    workload_variant: str
    execution_mode: str
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    attention_head_size: int
    warmup_runs: int | None
    runs_per_sample: int | None
    selected_candidate_ids: dict[str, str]
    selected_config: dict[str, dict[str, object]]
    row: dict[str, str]


def default_results_path() -> Path:
    results_dir = Path(__file__).resolve().parents[2] / "results" / "end_to_end"
    preferred_paths = (
        results_dir / "results_all_power.csv",
        results_dir / "results.csv",
    )
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


def load_result_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _eligible_row(row: dict[str, str]) -> bool:
    if row.get("backend") != "npu":
        return False
    if row.get("run_status") != "passed":
        return False
    if _normalized_execution_mode(row.get("execution_mode")) is None:
        return False
    if optional_int(row.get("seq_len")) is None:
        return False
    selected_config_json = str(row.get("selected_config_json") or "").strip()
    if not selected_config_json:
        return False
    required_selection_fields = (
        "joint_search_policy",
        "joint_candidate_count_total",
        "joint_candidates_evaluated",
        "selection_provenance",
        "candidate_inventory_json",
        "candidate_count_by_operator_json",
    )
    if any(
        str(row.get(field) or "").strip() == "" for field in required_selection_fields
    ):
        return False
    validation_mode = str(row.get("validation_mode") or "").strip()
    if validation_mode != REFERENCE_TOLERANCE_VALIDATION_MODE:
        return False
    return True


def _normalized_workload_variant(row: dict[str, str]) -> str | None:
    value = str(row.get("workload_variant") or "").strip()
    if value:
        try:
            return canonical_workload_variant(value)
        except ValueError:
            return None
    family_id = str(row.get("study_case_id") or "")
    if family_id in FAMILY_SPECS:
        return FAMILY_SPECS[family_id].workload_variant
    return None


def _matches_filters(
    row: dict[str, str],
    *,
    campaign_id_filter: str,
    repeat_index_filter: str,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
    mode_filter: str,
) -> bool:
    if campaign_id_filter != "all":
        if normalize_campaign_id(row.get("campaign_id")) != normalize_campaign_id(
            campaign_id_filter
        ):
            return False
    if repeat_index_filter != "all":
        if normalize_repeat_index(row.get("repeat_index")) != int(repeat_index_filter):
            return False
    if workload_variant_filter != "all":
        if _normalized_workload_variant(row) != canonical_workload_variant(
            workload_variant_filter
        ):
            return False
    if family_filter != "all" and row.get("study_case_id") != family_filter:
        return False
    if seq_len_filter != "all" and optional_int(row.get("seq_len")) != int(
        seq_len_filter
    ):
        return False
    if mode_filter != "all":
        if _normalized_execution_mode(
            row.get("execution_mode")
        ) != canonical_execution_mode(mode_filter):
            return False
    return True


def _sorted_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    family_order = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    seq_order = {seq_len: index for index, seq_len in enumerate(SEQUENCE_LADDER)}
    return sorted(
        rows,
        key=lambda row: (
            family_order.get(str(row.get("study_case_id") or ""), len(FAMILY_IDS)),
            _MODE_ORDER.get(
                _normalized_execution_mode(row.get("execution_mode")),
                len(EXECUTION_MODES),
            ),
            seq_order.get(optional_int(row.get("seq_len")), len(SEQUENCE_LADDER)),
        ),
    )


def _load_json_dict(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object, received {type(loaded).__name__}")
    return loaded


def select_result_rows(
    rows: list[dict[str, str]],
    *,
    campaign_id_filter: str = "all",
    repeat_index_filter: str = "all",
    workload_variant_filter: str = "all",
    family_filter: str = "all",
    seq_len_filter: str = "all",
    mode_filter: str = "all",
) -> tuple[SelectedEndToEndRow, ...]:
    selected: list[SelectedEndToEndRow] = []
    for row in _sorted_rows(rows):
        if not _eligible_row(row):
            continue
        if not _matches_filters(
            row,
            campaign_id_filter=campaign_id_filter,
            repeat_index_filter=repeat_index_filter,
            workload_variant_filter=workload_variant_filter,
            family_filter=family_filter,
            seq_len_filter=seq_len_filter,
            mode_filter=mode_filter,
        ):
            continue

        campaign_id = normalize_campaign_id(row.get("campaign_id"))
        repeat_index = normalize_repeat_index(row.get("repeat_index"))
        study_case_id = str(row.get("study_case_id") or "")
        execution_mode = canonical_execution_mode(str(row.get("execution_mode") or ""))
        seq_len = int(optional_int(row.get("seq_len")) or 0)
        selected.append(
            SelectedEndToEndRow(
                campaign_id=campaign_id,
                repeat_index=repeat_index,
                matched_run_id=normalize_matched_run_id(
                    row.get("matched_run_id"),
                    campaign_id=campaign_id,
                    study_case_id=study_case_id,
                    execution_mode=execution_mode,
                    seq_len=seq_len,
                    repeat_index=repeat_index,
                ),
                study_case_id=study_case_id,
                study_case_label=str(row.get("study_case_label") or ""),
                workload_variant=str(_normalized_workload_variant(row) or ""),
                execution_mode=execution_mode,
                seq_len=seq_len,
                hidden_size=int(optional_int(row.get("hidden_size")) or 0),
                intermediate_size=int(optional_int(row.get("intermediate_size")) or 0),
                num_attention_heads=int(
                    optional_int(row.get("num_attention_heads")) or 0
                ),
                attention_head_size=int(
                    optional_int(row.get("attention_head_size")) or 0
                ),
                warmup_runs=optional_int(row.get("warmup_runs")),
                runs_per_sample=optional_int(row.get("runs_per_sample")),
                selected_candidate_ids={
                    str(key): str(value)
                    for key, value in _load_json_dict(
                        str(row.get("selected_candidate_ids_json") or "{}")
                    ).items()
                },
                selected_config={
                    str(key): dict(value)
                    for key, value in _load_json_dict(
                        str(row.get("selected_config_json") or "{}")
                    ).items()
                    if isinstance(value, dict)
                },
                row=dict(row),
            )
        )
    return tuple(selected)


__all__ = [
    "DEFAULT_CAMPAIGN_ID",
    "SelectedEndToEndRow",
    "default_results_path",
    "load_result_rows",
    "select_result_rows",
]
