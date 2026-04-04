#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

ExecutionMode = Literal["dataflow", "runlist", "offload"]

EXECUTION_MODES: tuple[ExecutionMode, ...] = (
    "dataflow",
    "runlist",
    "offload",
)
FAMILY_IDS: tuple[str, ...] = ("baseline_768", "baseline_1024")
SEQUENCE_LADDER: tuple[int, ...] = (
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)

FAMILY_SPECS: dict[str, tuple[int, int, int]] = {
    "baseline_768": (768, 3072, 12),
    "baseline_1024": (1024, 4096, 16),
}

MODE_OPERATORS: dict[ExecutionMode, tuple[str, ...]] = {
    "dataflow": (
        "qkv_proj",
        "mha_out_proj",
        "add_norm1",
        "ffn",
        "add_norm2",
    ),
    "runlist": (
        "qkvo_proj",
        "k_transpose",
        "attn_scores",
        "attn_scale",
        "attn_softmax",
        "attn_output",
        "add",
        "ln1",
        "up_proj",
        "gelu",
        "down_proj",
        "ln2",
    ),
    "offload": ("shared_gemm",),
}

CandidateRecord = dict[str, Any]
ModeCandidateTable = dict[ExecutionMode, dict[str, tuple[CandidateRecord, ...]]]
CandidatePayloads = dict[ExecutionMode, dict[str, Any]]


@dataclass(frozen=True)
class EndToEndWorkload:
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def effective_dense_layer_flop_count(self) -> int:
        return effective_dense_layer_flop_count(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
        )


@dataclass(frozen=True)
class EndToEndCase:
    study_case_id: str
    workload: EndToEndWorkload

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


def effective_dense_layer_flop_count(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
) -> int:
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads to derive "
            "the effective dense-layer FLOP count"
        )

    qkv_projection_flops = 6 * seq_len * hidden_size * hidden_size
    attention_gemm_flops = 4 * seq_len * seq_len * hidden_size
    output_projection_flops = 2 * seq_len * hidden_size * hidden_size
    ffn_gemm_flops = 4 * seq_len * hidden_size * intermediate_size
    return (
        qkv_projection_flops
        + attention_gemm_flops
        + output_projection_flops
        + ffn_gemm_flops
    )


def effective_gflops_per_sec(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    avg_latency_ms: float | None,
) -> float | None:
    if avg_latency_ms is None or avg_latency_ms <= 0:
        return None
    flop_count = effective_dense_layer_flop_count(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )
    latency_sec = avg_latency_ms / 1000.0
    return flop_count / latency_sec / 1.0e9


def effective_gflops_per_sec_per_watt(
    effective_gflops_per_sec_value: float | None,
    avg_power_w: float | None,
) -> float | None:
    if (
        effective_gflops_per_sec_value is None
        or avg_power_w is None
        or avg_power_w <= 0
    ):
        return None
    return effective_gflops_per_sec_value / avg_power_w


def get_case(study_case_id: str, seq_len: int) -> EndToEndCase:
    hidden_size, intermediate_size, num_attention_heads = FAMILY_SPECS[study_case_id]
    return EndToEndCase(
        study_case_id=study_case_id,
        workload=EndToEndWorkload(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
        ),
    )


def iter_cases(
    family_filter: str = "all",
    seq_len_filter: int | str = "all",
) -> tuple[EndToEndCase, ...]:
    family_ids = FAMILY_IDS if family_filter == "all" else (family_filter,)
    seq_lengths = SEQUENCE_LADDER if seq_len_filter == "all" else (int(seq_len_filter),)
    return tuple(
        get_case(family_id, seq_len)
        for family_id in family_ids
        for seq_len in seq_lengths
    )


def default_candidates_path(execution_mode: ExecutionMode) -> Path:
    return Path(__file__).with_name(f"{execution_mode}_candidates.json")


def _copy_candidate(candidate: CandidateRecord) -> CandidateRecord:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "config": dict(candidate["config"]),
    }


def _validate_candidates_payload(
    payload: Any,
    *,
    source: Path,
    execution_mode: ExecutionMode,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object at the top level")

    family_items = {
        family_id: family_payload
        for family_id, family_payload in payload.items()
        if not str(family_id).startswith("_")
    }

    for family_id in FAMILY_IDS:
        if family_id not in family_items:
            raise ValueError(f"{source} is missing family '{family_id}'")

    valid_seq_keys = {"all", *(str(value) for value in SEQUENCE_LADDER)}
    valid_operators = MODE_OPERATORS[execution_mode]
    for family_id, family_payload in family_items.items():
        if family_id not in FAMILY_IDS:
            raise ValueError(f"{source} has unsupported family '{family_id}'")
        if not isinstance(family_payload, dict):
            raise ValueError(f"{source}:{family_id} must be a JSON object")

        for seq_key, seq_payload in family_payload.items():
            if seq_key not in valid_seq_keys:
                raise ValueError(
                    f"{source}:{family_id} has unsupported seq key '{seq_key}'"
                )
            if not isinstance(seq_payload, dict):
                raise ValueError(f"{source}:{family_id}:{seq_key} must be an object")

            for operator_name, candidates in seq_payload.items():
                if operator_name not in valid_operators:
                    raise ValueError(
                        f"{source}:{family_id}:{seq_key} has unsupported operator "
                        f"'{operator_name}'"
                    )
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError(
                        f"{source}:{family_id}:{seq_key}:{operator_name} "
                        "must be a non-empty list"
                    )
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        raise ValueError(
                            f"{source}:{family_id}:{seq_key}:{operator_name} "
                            "candidates must be JSON objects"
                        )
                    if "candidate_id" not in candidate or not isinstance(
                        candidate["candidate_id"], str
                    ):
                        raise ValueError(
                            f"{source}:{family_id}:{seq_key}:{operator_name} "
                            "candidate is missing string field 'candidate_id'"
                        )
                    if "config" not in candidate or not isinstance(
                        candidate["config"], dict
                    ):
                        raise ValueError(
                            f"{source}:{family_id}:{seq_key}:{operator_name} "
                            "candidate is missing object field 'config'"
                        )


def load_candidate_payload(
    execution_mode: ExecutionMode,
    path: Path | None = None,
) -> dict[str, Any]:
    candidate_path = default_candidates_path(execution_mode) if path is None else path
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_candidates_payload(
        payload,
        source=candidate_path,
        execution_mode=execution_mode,
    )
    return payload


@lru_cache(maxsize=len(EXECUTION_MODES))
def load_default_candidate_payload(execution_mode: ExecutionMode) -> dict[str, Any]:
    return load_candidate_payload(execution_mode)


def load_default_candidate_payloads() -> CandidatePayloads:
    return {
        execution_mode: load_default_candidate_payload(execution_mode)
        for execution_mode in EXECUTION_MODES
    }


def candidate_table_for_case(
    study_case_id: str,
    seq_len: int,
    *,
    payloads: CandidatePayloads | None = None,
) -> ModeCandidateTable:
    resolved_payloads = (
        load_default_candidate_payloads() if payloads is None else payloads
    )
    merged: ModeCandidateTable = {
        mode: {operator_name: tuple() for operator_name in MODE_OPERATORS[mode]}
        for mode in EXECUTION_MODES
    }

    for execution_mode in EXECUTION_MODES:
        family_payload = resolved_payloads[execution_mode][study_case_id]
        for seq_key in ("all", str(seq_len)):
            seq_payload = family_payload.get(seq_key, {})
            for operator_name, candidates in seq_payload.items():
                merged[execution_mode][operator_name] = tuple(
                    _copy_candidate(candidate) for candidate in candidates
                )

    for execution_mode in EXECUTION_MODES:
        for operator_name in MODE_OPERATORS[execution_mode]:
            if not merged[execution_mode][operator_name]:
                raise ValueError(
                    f"Missing candidates for family={study_case_id} seq_len={seq_len} "
                    f"mode={execution_mode} operator={operator_name}"
                )
    return merged
