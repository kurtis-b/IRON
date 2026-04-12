#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iron.applications.transformer_layer_new.study.block.cases import (
    FAMILY_IDS,
    SEQUENCE_LADDER,
    BlockWorkload,
)

StagingBlockKind = Literal["mha_out_proj", "ffn"]

STAGING_BLOCK_KINDS: tuple[StagingBlockKind, ...] = ("mha_out_proj", "ffn")
CONFIG_COLUMNS_BY_BLOCK_KIND: dict[StagingBlockKind, tuple[str, ...]] = {
    "mha_out_proj": (
        "mha_out_proj_parallel_seq",
        "mha_out_proj_q_seq_tile",
        "mha_out_proj_kv_seq_tile",
        "mha_out_proj_emb_tile",
        "mha_out_proj_parallel_heads",
        "mha_out_proj_o_proj_acc_depth",
    ),
    "ffn": (
        "ffn_num_aie_columns",
        "ffn_b_col_maj",
        "ffn_c_col_maj",
        "ffn_tile_m",
        "ffn_tile_k",
        "ffn_tile_n",
        "ffn_down_proj_depth",
        "ffn_n_a_tiles_distributed",
        "ffn_n_b_tiles_distributed",
        "ffn_stage_only",
        "ffn_gelu_stage",
    ),
}


@dataclass(frozen=True)
class ReferenceSelection:
    family_id: str
    family_label: str
    seq_len: int
    block_kind: StagingBlockKind
    source_candidate_index: int
    source_avg_latency_ms: float
    head_dim: int
    num_heads: int
    hidden_size: int
    ffn_dim: int
    source_candidate: tuple[object, ...]

    @property
    def workload(self) -> BlockWorkload:
        return BlockWorkload(
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
        )


def default_reference_results_path() -> Path:
    current = Path(__file__).resolve().parents[2] / "results" / "block" / "results.csv"
    if current.exists():
        return current
    fallback = (
        Path(__file__).resolve().parents[2] / "results_final" / "block" / "results.csv"
    )
    return fallback


def load_reference_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(str(value))


def _optional_bool(value: object) -> bool | None:
    if value in (None, "", "None"):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError("Missing required integer value")
    return parsed


def _required_bool(value: object) -> bool:
    parsed = _optional_bool(value)
    if parsed is None:
        raise ValueError("Missing required boolean value")
    return parsed


def _parse_candidate(
    block_kind: StagingBlockKind,
    row: dict[str, str],
) -> tuple[object, ...]:
    if block_kind == "mha_out_proj":
        return tuple(
            _required_int(row[column])
            for column in CONFIG_COLUMNS_BY_BLOCK_KIND[block_kind]
        )

    return (
        _required_int(row["ffn_num_aie_columns"]),
        _required_bool(row["ffn_b_col_maj"]),
        _required_bool(row["ffn_c_col_maj"]),
        _required_int(row["ffn_tile_m"]),
        _required_int(row["ffn_tile_k"]),
        _required_int(row["ffn_tile_n"]),
        _required_int(row["ffn_down_proj_depth"]),
        _required_int(row["ffn_n_a_tiles_distributed"]),
        _required_int(row["ffn_n_b_tiles_distributed"]),
        _optional_int(row["ffn_stage_only"]),
        _required_int(row["ffn_gelu_stage"]),
    )


def _eligible_reference_row(row: dict[str, str]) -> bool:
    block_kind = str(row.get("block_kind") or "")
    if block_kind not in STAGING_BLOCK_KINDS:
        return False
    if row.get("run_status") != "passed":
        return False
    if _optional_bool(row.get("is_best")) is not True:
        return False
    if _optional_float(row.get("avg_latency_ms")) is None:
        return False
    if _optional_int(row.get("candidate_index")) is None:
        return False
    try:
        _parse_candidate(block_kind, row)  # type: ignore[arg-type]
    except Exception:
        return False
    return True


def _matches_filters(
    row: dict[str, str],
    *,
    family_filter: str,
    seq_len_filter: str,
    block_filter: str,
) -> bool:
    if family_filter != "all" and row.get("family_id") != family_filter:
        return False
    if seq_len_filter != "all" and _optional_int(row.get("seq_len")) != int(
        seq_len_filter
    ):
        return False
    if block_filter != "all" and row.get("block_kind") != block_filter:
        return False
    return True


def select_reference_rows(
    rows: list[dict[str, str]],
    *,
    family_filter: str = "all",
    seq_len_filter: str = "all",
    block_filter: str = "all",
) -> tuple[ReferenceSelection, ...]:
    selections_by_key: dict[
        tuple[str, int, StagingBlockKind],
        tuple[float, int, ReferenceSelection],
    ] = {}

    for row in rows:
        if not _eligible_reference_row(row):
            continue
        if not _matches_filters(
            row,
            family_filter=family_filter,
            seq_len_filter=seq_len_filter,
            block_filter=block_filter,
        ):
            continue

        block_kind = str(row["block_kind"])
        selection = ReferenceSelection(
            family_id=str(row.get("family_id") or ""),
            family_label=str(row.get("family_label") or ""),
            seq_len=int(_optional_int(row.get("seq_len")) or 0),
            block_kind=block_kind,  # type: ignore[arg-type]
            source_candidate_index=int(_optional_int(row.get("candidate_index")) or 0),
            source_avg_latency_ms=float(
                _optional_float(row.get("avg_latency_ms")) or 0
            ),
            head_dim=int(_optional_int(row.get("head_dim")) or 0),
            num_heads=int(_optional_int(row.get("num_heads")) or 0),
            hidden_size=int(_optional_int(row.get("hidden_size")) or 0),
            ffn_dim=int(_optional_int(row.get("ffn_dim")) or 0),
            source_candidate=_parse_candidate(block_kind, row),  # type: ignore[arg-type]
        )
        key = (
            selection.family_id,
            selection.seq_len,
            selection.block_kind,
        )
        current = selections_by_key.get(key)
        if current is None or (
            selection.source_avg_latency_ms,
            selection.source_candidate_index,
        ) < (current[0], current[1]):
            selections_by_key[key] = (
                selection.source_avg_latency_ms,
                selection.source_candidate_index,
                selection,
            )

    block_order = {
        block_kind: index for index, block_kind in enumerate(STAGING_BLOCK_KINDS)
    }
    family_order = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    seq_order = {seq_len: index for index, seq_len in enumerate(SEQUENCE_LADDER)}
    return tuple(
        entry[2]
        for _, entry in sorted(
            selections_by_key.items(),
            key=lambda item: (
                family_order.get(item[0][0], len(FAMILY_IDS)),
                seq_order.get(item[0][1], len(SEQUENCE_LADDER)),
                block_order[item[0][2]],
            ),
        )
    )


__all__ = [
    "CONFIG_COLUMNS_BY_BLOCK_KIND",
    "ReferenceSelection",
    "STAGING_BLOCK_KINDS",
    "default_reference_results_path",
    "load_reference_rows",
    "select_reference_rows",
]
