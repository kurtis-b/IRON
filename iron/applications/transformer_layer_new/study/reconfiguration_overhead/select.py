#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .cases import EXECUTION_MODES, ExecutionMode, SOURCE_EXECUTION_MODE_BY_STUDY_MODE


@dataclass(frozen=True)
class ReferenceSelection:
    study_case_id: str
    study_case_label: str
    execution_mode: ExecutionMode
    source_execution_mode: str
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    attention_head_size: int
    batch_size: int
    dtype: str
    use_bias: bool
    weights_source: str
    selected_candidate_ids_json: str
    selected_config_json: str
    selected_candidate_ids: dict[str, object]
    selected_config: dict[str, dict[str, object]]


def default_reference_results_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "results" / "end_to_end" / "results.csv"
    )


def load_reference_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _optional_bool(value: object) -> bool | None:
    if value in (None, "", "None"):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _load_json_dict(value: object) -> dict[str, object]:
    if value in (None, "", "None"):
        return {}
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _required_gemm_config(
    source_execution_mode: str,
    selected_config: dict[str, object],
) -> dict[str, dict[str, object]] | None:
    if source_execution_mode == "offload":
        shared_gemm = selected_config.get("shared_gemm")
        if not isinstance(shared_gemm, dict) or not shared_gemm:
            return None
        return {"shared_gemm": dict(shared_gemm)}

    if source_execution_mode == "runlist":
        required_keys = (
            "qkvo_proj",
            "attn_scores",
            "attn_output",
            "up_proj",
            "down_proj",
        )
        filtered: dict[str, dict[str, object]] = {}
        for key in required_keys:
            value = selected_config.get(key)
            if not isinstance(value, dict) or not value:
                return None
            filtered[key] = dict(value)
        return filtered

    return None


def _filtered_candidate_ids(
    source_execution_mode: str,
    candidate_ids: dict[str, object],
) -> dict[str, object]:
    if source_execution_mode == "offload":
        value = candidate_ids.get("shared_gemm")
        return {} if value is None else {"shared_gemm": value}

    if source_execution_mode == "runlist":
        required_keys = (
            "qkvo_proj",
            "attn_scores",
            "attn_output",
            "up_proj",
            "down_proj",
        )
        return {
            key: candidate_ids[key]
            for key in required_keys
            if key in candidate_ids and candidate_ids[key] not in (None, "")
        }

    return {}


def _eligible_reference_row(row: dict[str, str]) -> bool:
    if row.get("backend") != "npu":
        return False
    if row.get("run_status") != "passed":
        return False
    if row.get("execution_mode") not in SOURCE_EXECUTION_MODE_BY_STUDY_MODE.values():
        return False
    if _optional_int(row.get("seq_len")) is None:
        return False
    if not _load_json_dict(row.get("selected_config_json")):
        return False
    return True


def _matches_filters(
    row: dict[str, str],
    *,
    family_filter: str,
    seq_len_filter: str,
    mode_filter: str,
) -> bool:
    if family_filter != "all" and row.get("study_case_id") != family_filter:
        return False
    if seq_len_filter != "all" and _optional_int(row.get("seq_len")) != int(
        seq_len_filter
    ):
        return False
    if mode_filter != "all":
        expected_source_mode = SOURCE_EXECUTION_MODE_BY_STUDY_MODE[
            mode_filter  # type: ignore[index]
        ]
        if row.get("execution_mode") != expected_source_mode:
            return False
    return True


def select_reference_rows(
    rows: list[dict[str, str]],
    *,
    family_filter: str = "all",
    seq_len_filter: str = "all",
    mode_filter: str = "all",
) -> tuple[ReferenceSelection, ...]:
    selections_by_key: dict[
        tuple[str, int, str],
        tuple[float, ReferenceSelection],
    ] = {}

    for row in rows:
        if not _eligible_reference_row(row):
            continue
        if not _matches_filters(
            row,
            family_filter=family_filter,
            seq_len_filter=seq_len_filter,
            mode_filter=mode_filter,
        ):
            continue

        source_execution_mode = str(row.get("execution_mode") or "")
        selected_config = _required_gemm_config(
            source_execution_mode,
            _load_json_dict(row.get("selected_config_json")),
        )
        if selected_config is None:
            continue

        candidate_ids = _filtered_candidate_ids(
            source_execution_mode,
            _load_json_dict(row.get("selected_candidate_ids_json")),
        )
        execution_mode = next(
            study_mode
            for study_mode, source_mode in SOURCE_EXECUTION_MODE_BY_STUDY_MODE.items()
            if source_mode == source_execution_mode
        )
        selection = ReferenceSelection(
            study_case_id=str(row.get("study_case_id") or ""),
            study_case_label=str(row.get("study_case_label") or ""),
            execution_mode=execution_mode,
            source_execution_mode=source_execution_mode,
            seq_len=int(_optional_int(row.get("seq_len")) or 0),
            hidden_size=int(_optional_int(row.get("hidden_size")) or 0),
            intermediate_size=int(_optional_int(row.get("intermediate_size")) or 0),
            num_attention_heads=int(_optional_int(row.get("num_attention_heads")) or 0),
            attention_head_size=int(_optional_int(row.get("attention_head_size")) or 0),
            batch_size=int(_optional_int(row.get("batch_size")) or 1),
            dtype=str(row.get("dtype") or "bf16"),
            use_bias=bool(_optional_bool(row.get("use_bias")) or False),
            weights_source=str(row.get("weights_source") or "synthetic"),
            selected_candidate_ids_json=json.dumps(candidate_ids, sort_keys=True),
            selected_config_json=json.dumps(selected_config, sort_keys=True),
            selected_candidate_ids=candidate_ids,
            selected_config=selected_config,
        )
        key = (
            selection.study_case_id,
            selection.seq_len,
            selection.source_execution_mode,
        )
        latency = float(str(row.get("avg_latency_ms") or "inf"))
        current = selections_by_key.get(key)
        if current is None or latency < current[0]:
            selections_by_key[key] = (latency, selection)

    return tuple(
        value[1]
        for _, value in sorted(
            selections_by_key.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                EXECUTION_MODES.index(item[1][1].execution_mode),
            ),
        )
    )


__all__ = [
    "ReferenceSelection",
    "default_reference_results_path",
    "load_reference_rows",
    "select_reference_rows",
]
