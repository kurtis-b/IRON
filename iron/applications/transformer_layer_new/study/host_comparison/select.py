#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from iron.applications.transformer_layer_new.study.end_to_end.cases import (
    FAMILY_SPECS,
    canonical_execution_mode,
    canonical_workload_variant,
)

REFERENCE_EXECUTION_MODES: tuple[str, ...] = ("hybrid",)
_REFERENCE_MODE_ORDER = {
    execution_mode: index
    for index, execution_mode in enumerate(REFERENCE_EXECUTION_MODES)
}


@dataclass(frozen=True)
class ReferenceGroup:
    study_case_id: str
    study_case_label: str
    workload_variant: str
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    attention_head_size: int
    batch_size: int
    dtype: str
    use_bias: bool
    weights_source: str
    warmup_runs: int | None
    runs_per_sample: int | None
    rows: tuple[dict[str, str], ...]


def default_reference_results_path() -> Path:
    results_dir = Path(__file__).resolve().parents[2] / "results" / "end_to_end"
    preferred_paths = (
        results_dir / "results_all_power.csv",
        results_dir / "results.csv",
    )
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


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


def _eligible_reference_row(row: dict[str, str]) -> bool:
    if row.get("backend") != "npu":
        return False
    if row.get("run_status") != "passed":
        return False
    try:
        execution_mode = canonical_execution_mode(str(row.get("execution_mode") or ""))
    except ValueError:
        return False
    if execution_mode not in REFERENCE_EXECUTION_MODES:
        return False
    return _optional_int(row.get("seq_len")) is not None


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
    family_filter: str,
    seq_len_filter: str,
) -> bool:
    if family_filter != "all" and row.get("study_case_id") != family_filter:
        return False
    if seq_len_filter != "all" and _optional_int(row.get("seq_len")) != int(
        seq_len_filter
    ):
        return False
    return True


def _sorted_reference_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _REFERENCE_MODE_ORDER.get(
                canonical_execution_mode(str(row.get("execution_mode") or "")),
                len(REFERENCE_EXECUTION_MODES),
            ),
            str(row.get("selected_candidate_ids_json") or ""),
            str(row.get("selected_config_json") or ""),
        ),
    )


def _required_shared_text(rows: list[dict[str, str]], field: str) -> str:
    values = {str(row.get(field) or "") for row in rows}
    if len(values) != 1:
        raise ValueError(
            f"Reference rows disagree on field {field!r}: {sorted(values)}"
        )
    return next(iter(values))


def _required_shared_int(rows: list[dict[str, str]], field: str) -> int:
    values = {_optional_int(row.get(field)) for row in rows}
    if None in values or len(values) != 1:
        raise ValueError(
            f"Reference rows disagree on field {field!r}: {sorted(values)}"
        )
    return next(iter(values))


def _optional_uniform_int(rows: list[dict[str, str]], field: str) -> int | None:
    values = {
        _optional_int(row.get(field))
        for row in rows
        if row.get(field) not in ("", None)
    }
    if not values:
        return None
    if len(values) != 1:
        return None
    return next(iter(values))


def _optional_uniform_bool(rows: list[dict[str, str]], field: str) -> bool | None:
    values = {
        _optional_bool(row.get(field))
        for row in rows
        if row.get(field) not in ("", None)
    }
    if not values:
        return None
    if len(values) != 1:
        return None
    return next(iter(values))


def group_reference_rows(
    rows: list[dict[str, str]],
    *,
    family_filter: str = "all",
    seq_len_filter: str = "all",
) -> tuple[ReferenceGroup, ...]:
    grouped_rows: dict[tuple[str, str, int], list[dict[str, str]]] = {}

    for row in rows:
        if not _eligible_reference_row(row):
            continue
        if not _matches_filters(
            row,
            family_filter=family_filter,
            seq_len_filter=seq_len_filter,
        ):
            continue
        study_case_id = str(row.get("study_case_id") or "")
        seq_len = _optional_int(row.get("seq_len"))
        workload_variant = _normalized_workload_variant(row)
        if not study_case_id or seq_len is None or workload_variant is None:
            continue
        grouped_rows.setdefault((study_case_id, workload_variant, seq_len), []).append(
            dict(row)
        )

    groups: list[ReferenceGroup] = []
    for group_key in sorted(grouped_rows):
        group_rows = _sorted_reference_rows(grouped_rows[group_key])
        attention_head_size = _required_shared_int(group_rows, "attention_head_size")
        use_bias = _optional_uniform_bool(group_rows, "use_bias")
        groups.append(
            ReferenceGroup(
                study_case_id=group_key[0],
                workload_variant=group_key[1],
                study_case_label=_required_shared_text(group_rows, "study_case_label"),
                seq_len=group_key[2],
                hidden_size=_required_shared_int(group_rows, "hidden_size"),
                intermediate_size=_required_shared_int(group_rows, "intermediate_size"),
                num_attention_heads=_required_shared_int(
                    group_rows, "num_attention_heads"
                ),
                attention_head_size=attention_head_size,
                batch_size=_required_shared_int(group_rows, "batch_size"),
                dtype=_required_shared_text(group_rows, "dtype"),
                use_bias=False if use_bias is None else use_bias,
                weights_source=_required_shared_text(group_rows, "weights_source"),
                warmup_runs=_optional_uniform_int(group_rows, "warmup_runs"),
                runs_per_sample=_optional_uniform_int(group_rows, "runs_per_sample"),
                rows=tuple(group_rows),
            )
        )

    return tuple(groups)


__all__ = [
    "REFERENCE_EXECUTION_MODES",
    "ReferenceGroup",
    "default_reference_results_path",
    "group_reference_rows",
    "load_reference_rows",
]
