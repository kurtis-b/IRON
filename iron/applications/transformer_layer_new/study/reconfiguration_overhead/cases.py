#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExecutionMode = Literal["offload_gemm_sequence", "runlist_gemm_sequence"]

EXECUTION_MODES: tuple[ExecutionMode, ...] = (
    "offload_gemm_sequence",
    "runlist_gemm_sequence",
)
SOURCE_EXECUTION_MODE_BY_STUDY_MODE: dict[ExecutionMode, str] = {
    "offload_gemm_sequence": "offload",
    "runlist_gemm_sequence": "runlist",
}
FAMILY_IDS: tuple[str, ...] = ("baseline_768", "baseline_1024")
SEQUENCE_LADDER: tuple[int, ...] = (256, 2048, 16384)

FAMILY_SPECS: dict[str, tuple[int, int, int]] = {
    "baseline_768": (768, 3072, 12),
    "baseline_1024": (1024, 4096, 16),
}


@dataclass(frozen=True)
class ReconfigurationWorkload:
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class ReconfigurationCase:
    study_case_id: str
    workload: ReconfigurationWorkload

    @property
    def study_case_label(self) -> str:
        return (
            f"{self.workload.hidden_size} / {self.workload.intermediate_size} / "
            f"{self.workload.num_attention_heads}"
        )

    @property
    def seq_len(self) -> int:
        return self.workload.seq_len

    @property
    def hidden_size(self) -> int:
        return self.workload.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.workload.intermediate_size

    @property
    def num_attention_heads(self) -> int:
        return self.workload.num_attention_heads

    @property
    def attention_head_size(self) -> int:
        return self.workload.attention_head_size


def get_case(study_case_id: str, seq_len: int) -> ReconfigurationCase:
    hidden_size, intermediate_size, num_attention_heads = FAMILY_SPECS[study_case_id]
    return ReconfigurationCase(
        study_case_id=study_case_id,
        workload=ReconfigurationWorkload(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
        ),
    )


def iter_cases(
    family_filter: str = "all",
    seq_len_filter: int | str = "all",
) -> tuple[ReconfigurationCase, ...]:
    family_ids = FAMILY_IDS if family_filter == "all" else (family_filter,)
    seq_lengths = SEQUENCE_LADDER if seq_len_filter == "all" else (int(seq_len_filter),)
    return tuple(
        get_case(family_id, seq_len)
        for family_id in family_ids
        for seq_len in seq_lengths
    )


__all__ = [
    "EXECUTION_MODES",
    "FAMILY_IDS",
    "SEQUENCE_LADDER",
    "SOURCE_EXECUTION_MODE_BY_STUDY_MODE",
    "ReconfigurationCase",
    "ReconfigurationWorkload",
    "get_case",
    "iter_cases",
]
