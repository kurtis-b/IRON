#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

BlockKind = Literal[
    "qkv_proj",
    "mha_out_proj",
    "mha_out_proj_causal",
    "addnorm",
    "layer_norm",
    "elementwise_add",
    "ffn",
]

QKVProjCandidate = tuple[int, int, int, int, int]
MHAOutProjCandidate = tuple[int, int, int, int, int, int]
AddNormCandidate = tuple[int, int]
LayerNormCandidate = tuple[int, int]
ElementwiseAddCandidate = tuple[int, int]
CausalMaskCandidate = tuple[int, int]
FFNCandidate = tuple[int, bool, bool, int, int, int, int, int, int, int | None, int]

BLOCK_KINDS: tuple[BlockKind, ...] = (
    "qkv_proj",
    "mha_out_proj",
    "mha_out_proj_causal",
    "addnorm",
    "layer_norm",
    "elementwise_add",
    "ffn",
)
ENCODER_FAMILY_IDS: tuple[str, ...] = (
    "tinybert_512",
    "baseline_768",
    "baseline_1024",
)
DECODER_FAMILY_IDS: tuple[str, ...] = (
    "gpt2_512",
    "gpt2_small_768",
    "gpt2_medium_1024",
)
FAMILY_IDS: tuple[str, ...] = (*ENCODER_FAMILY_IDS, *DECODER_FAMILY_IDS)
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


@dataclass(frozen=True)
class BlockWorkload:
    seq_len: int
    head_dim: int
    num_heads: int
    ffn_dim: int

    @property
    def hidden_size(self) -> int:
        return self.head_dim * self.num_heads

    @property
    def family_label(self) -> str:
        return f"{self.hidden_size} / {self.ffn_dim} / {self.num_heads}"


@dataclass(frozen=True)
class BlockCase:
    family_id: str
    workload: BlockWorkload
    qkv_proj: tuple[QKVProjCandidate, ...]
    mha_out_proj: tuple[MHAOutProjCandidate, ...]
    mha_out_proj_causal: tuple[MHAOutProjCandidate, ...]
    addnorm: tuple[AddNormCandidate, ...]
    layer_norm: tuple[LayerNormCandidate, ...]
    elementwise_add: tuple[ElementwiseAddCandidate, ...]
    causal_mask: tuple[CausalMaskCandidate, ...]
    ffn: tuple[FFNCandidate, ...]

    @property
    def seq_len(self) -> int:
        return self.workload.seq_len

    @property
    def family_label(self) -> str:
        return self.workload.family_label

    def candidates(self, block_kind: BlockKind) -> tuple[tuple[object, ...], ...]:
        removed_indices = _removed_candidate_indices().get(
            (self.family_id, self.seq_len, block_kind),
            frozenset(),
        )
        return tuple(
            candidate
            for index, candidate in enumerate(getattr(self, block_kind))
            if index not in removed_indices
        )


def _normalize_candidates(
    candidates: Sequence[Sequence[object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(candidate) for candidate in candidates)


def make_case(
    family_id: str,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    ffn_dim: int,
    *,
    qkv_proj: Sequence[QKVProjCandidate] | None = None,
    mha_out_proj: Sequence[MHAOutProjCandidate] | None = None,
    mha_out_proj_causal: Sequence[MHAOutProjCandidate] | None = None,
    addnorm: Sequence[AddNormCandidate] | None = None,
    layer_norm: Sequence[LayerNormCandidate] | None = None,
    elementwise_add: Sequence[ElementwiseAddCandidate] | None = None,
    causal_mask: Sequence[CausalMaskCandidate] | None = None,
    ffn: Sequence[FFNCandidate] | None = None,
) -> BlockCase:
    workload = BlockWorkload(
        seq_len=seq_len,
        head_dim=head_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
    )
    return BlockCase(
        family_id=family_id,
        workload=workload,
        qkv_proj=_normalize_candidates(qkv_proj or ((32, 64, 16, 1, 8),)),
        mha_out_proj=_normalize_candidates(
            mha_out_proj or ((1, 32, 64, workload.head_dim, 1, 1),)
        ),
        mha_out_proj_causal=_normalize_candidates(mha_out_proj_causal or ()),
        addnorm=_normalize_candidates(addnorm or ((8, workload.hidden_size),)),
        layer_norm=_normalize_candidates(layer_norm or ()),
        elementwise_add=_normalize_candidates(elementwise_add or ()),
        causal_mask=_normalize_candidates(causal_mask or ()),
        ffn=_normalize_candidates(
            ffn or ((8, False, False, 64, 48, 96, 8, 4, 4, None, 1),)
        ),
    )


def removed_cases_path() -> Path:
    return Path(__file__).with_name("removed_cases.csv")


@lru_cache(maxsize=1)
def _removed_candidate_indices() -> dict[tuple[str, int, BlockKind], frozenset[int]]:
    path = removed_cases_path()
    if not path.exists():
        return {}

    removed: dict[tuple[str, int, BlockKind], set[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            family_id = str(row.get("family_id") or "").strip()
            seq_len = int(row["seq_len"])
            block_kind = row["block_kind"]
            candidate_index = int(row["candidate_index"])
            key = (family_id, seq_len, block_kind)  # type: ignore[arg-type]
            removed.setdefault(key, set()).add(candidate_index)

    return {key: frozenset(indices) for key, indices in removed.items()}


def _tinybert_512_case(seq_len: int) -> BlockCase:
    qkv_proj = (
        [(16, 64, 64, 4, 8), (16, 128, 64, 4, 4)]
        if seq_len == 64
        else (
            [(32, 64, 64, 4, 8), (32, 128, 64, 4, 4)]
            if seq_len == 128
            else [(64, 64, 64, 4, 8), (64, 128, 64, 4, 4)]
        )
    )
    mha_out_proj = (
        [(1, 32, 64, 64, 4, 8), (2, 32, 64, 64, 2, 8)]
        if seq_len == 64
        else (
            [(4, 32, 64, 64, 1, 8), (2, 32, 64, 64, 2, 8)]
            if seq_len == 128
            else (
                [(8, 32, 64, 64, 1, 8)]
                if seq_len >= 8192
                else [(8, 32, 64, 64, 1, 8), (4, 32, 64, 64, 2, 8)]
            )
        )
    )
    if seq_len == 64:
        ffn = [
            (8, False, False, 64, 64, 64, 8, 1, 4, None, 1),
            (8, False, False, 64, 64, 64, 4, 1, 4, None, 1),
        ]
    elif seq_len == 128:
        ffn = [
            (8, False, False, 64, 64, 64, 8, 2, 4, None, 1),
            (8, False, False, 64, 64, 64, 4, 2, 4, None, 1),
        ]
    else:
        ffn = [
            (8, False, False, 64, 64, 64, 8, 4, 4, None, 1),
            (8, False, False, 64, 64, 64, 4, 4, 4, None, 1),
        ]
    return make_case(
        "tinybert_512",
        seq_len,
        64,
        8,
        2048,
        qkv_proj=qkv_proj,
        mha_out_proj=mha_out_proj,
        addnorm=[(8, 512)],
        ffn=ffn,
    )


# Shared workload entry point per case:
#   seq_len, head_dim, num_heads, ffn_dim
# Workload -> operator mapping:
#   qkv_proj: seq_len, hidden_size=head_dim*num_heads
#   mha_out_proj: seq_len, head_dim, num_heads
#   addnorm: size=seq_len*hidden_size, tile_size=hidden_size
#   ffn: M=seq_len, K=hidden_size, N=ffn_dim
#
# Candidate tuple order by operator:
#   qkv_proj: tile_m, tile_k, tile_n, parallel_seq, parallel_emb
#   mha_out_proj: parallel_seq, q_seq_tile, kv_seq_tile, emb_tile, parallel_heads, o_proj_acc_depth
#   addnorm: num_aie_columns, tile_size
#   ffn: num_aie_columns, b_col_maj, c_col_maj, tile_m, tile_k, tile_n, down_proj_depth, n_a_tiles_distributed, n_b_tiles_distributed, stage_only, gelu_stage
#
# Edit per-case candidate tuples here. Replace a `make_case(...)` call with one
# that passes `qkv_proj=...`, `mha_out_proj=...`, `addnorm=...`, or `ffn=...`
# whenever a specific family/sequence point needs different candidates.
BLOCK_CASES: dict[str, dict[int, BlockCase]] = {
    "tinybert_512": {
        seq_len: _tinybert_512_case(seq_len) for seq_len in SEQUENCE_LADDER
    },
    "baseline_768": {
        64: make_case(
            "baseline_768",
            64,
            64,
            12,
            3072,
            qkv_proj=[
                (16, 96, 96, 4, 8),
                (16, 64, 96, 4, 8),
                (16, 128, 64, 4, 6),
                (32, 96, 96, 2, 8),
            ],
            mha_out_proj=[
                (1, 32, 64, 96, 6, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 96, 8, 2, 4, None, 1),
                (8, False, False, 16, 96, 48, 8, 2, 4, None, 1),
                (8, False, False, 16, 96, 64, 8, 1, 4, None, 1),
            ],
        ),
        128: make_case(
            "baseline_768",
            128,
            64,
            12,
            3072,
            qkv_proj=[
                (32, 64, 96, 4, 8),
                (32, 96, 96, 4, 8),
            ],
            mha_out_proj=[
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 32, 96, 48, 8, 2, 4, None, 1),
                (8, False, False, 16, 96, 96, 8, 8, 2, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        256: make_case(
            "baseline_768",
            256,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 32, 96, 96, 8, 8, 2, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
            ],
        ),
        512: make_case(
            "baseline_768",
            512,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        1024: make_case(
            "baseline_768",
            1024,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        2048: make_case(
            "baseline_768",
            2048,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        4096: make_case(
            "baseline_768",
            4096,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        8192: make_case(
            "baseline_768",
            8192,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
        16384: make_case(
            "baseline_768",
            16384,
            64,
            12,
            3072,
            qkv_proj=[
                (64, 64, 48, 4, 8),
                (64, 64, 64, 4, 6),
            ],
            mha_out_proj=[
                (8, 32, 64, 96, 1, 8),
            ],
            ffn=[
                (8, False, False, 16, 96, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 96, 64, 8, 2, 4, None, 1),
                (8, False, False, 32, 96, 48, 8, 4, 4, None, 1),
            ],
        ),
    },
    "baseline_1024": {
        64: make_case(
            "baseline_1024",
            64,
            64,
            16,
            4096,
            qkv_proj=[
                (16, 64, 128, 4, 8),
                (16, 128, 64, 4, 8),
            ],
            mha_out_proj=[
                (1, 32, 64, 128, 4, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 16, 64, 128, 8, 4, 4, None, 1),
                (8, False, False, 16, 128, 64, 8, 2, 4, None, 1),
                (8, False, False, 16, 64, 128, 8, 2, 4, None, 1),
            ],
        ),
        128: make_case(
            "baseline_1024",
            128,
            64,
            16,
            4096,
            qkv_proj=[
                (32, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 16, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
            ],
        ),
        256: make_case(
            "baseline_1024",
            256,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        512: make_case(
            "baseline_1024",
            512,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        1024: make_case(
            "baseline_1024",
            1024,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        2048: make_case(
            "baseline_1024",
            2048,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        4096: make_case(
            "baseline_1024",
            4096,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        8192: make_case(
            "baseline_1024",
            8192,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
        16384: make_case(
            "baseline_1024",
            16384,
            64,
            16,
            4096,
            qkv_proj=[
                (64, 64, 64, 4, 8),
            ],
            mha_out_proj=[
                (8, 32, 64, 128, 1, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 32, 128, 64, 8, 8, 2, None, 1),
                (8, False, False, 16, 128, 64, 8, 4, 4, None, 1),
                (8, False, False, 32, 128, 64, 8, 2, 4, None, 1),
            ],
        ),
    },
}


def _decoder_only_block_candidates(
    family_id: str,
    seq_len: int,
) -> dict[str, tuple[tuple[object, ...], ...]]:
    if family_id == "gpt2_512":
        causal_mha = (
            ((1, 32, 32, 64, 4, 8),)
            if seq_len == 64
            else (
                ((4, 32, 32, 64, 1, 8),) if seq_len == 128 else ((8, 32, 32, 64, 1, 8),)
            )
        )
        return {
            "mha_out_proj_causal": causal_mha,
            "layer_norm": ((8, 2),),
            "elementwise_add": ((8, 2),),
            "causal_mask": ((8, 2),) if seq_len <= 4096 else tuple(),
        }

    if family_id == "gpt2_small_768":
        causal_mha = (
            ((1, 32, 32, 96, 6, 8),)
            if seq_len == 64
            else (
                ((4, 32, 32, 96, 2, 8),) if seq_len == 128 else ((8, 32, 32, 96, 1, 8),)
            )
        )
        return {
            "mha_out_proj_causal": causal_mha,
            "layer_norm": ((8, 2),),
            "elementwise_add": ((8, 2),),
            "causal_mask": ((8, 2),) if seq_len <= 4096 else tuple(),
        }

    if family_id == "gpt2_medium_1024":
        causal_mha = (
            ((1, 32, 32, 128, 4, 8),)
            if seq_len == 64
            else (
                ((4, 32, 32, 128, 2, 8),)
                if seq_len == 128
                else ((8, 32, 32, 128, 1, 8),)
            )
        )
        return {
            "mha_out_proj_causal": causal_mha,
            "layer_norm": ((8, 2),),
            "elementwise_add": ((8, 2),),
            "causal_mask": ((8, 2),) if seq_len <= 4096 else tuple(),
        }

    return {
        "mha_out_proj_causal": tuple(),
        "layer_norm": tuple(),
        "elementwise_add": tuple(),
        "causal_mask": tuple(),
    }


def _decoder_family_case_table(
    encoder_family_id: str,
    decoder_family_id: str,
) -> dict[int, BlockCase]:
    return {
        seq_len: replace(
            BLOCK_CASES[encoder_family_id][seq_len],
            family_id=decoder_family_id,
            **_decoder_only_block_candidates(decoder_family_id, seq_len),
        )
        for seq_len in SEQUENCE_LADDER
    }


BLOCK_CASES.update(
    {
        "gpt2_512": _decoder_family_case_table("tinybert_512", "gpt2_512"),
        "gpt2_small_768": _decoder_family_case_table(
            "baseline_768",
            "gpt2_small_768",
        ),
        "gpt2_medium_1024": _decoder_family_case_table(
            "baseline_1024",
            "gpt2_medium_1024",
        ),
    }
)


def get_case(family_id: str, seq_len: int) -> BlockCase:
    return BLOCK_CASES[family_id][seq_len]


def iter_cases(
    family_id: str | None = None,
    seq_len: int | None = None,
) -> tuple[BlockCase, ...]:
    family_ids = FAMILY_IDS if family_id is None else (family_id,)
    seq_lens = SEQUENCE_LADDER if seq_len is None else (seq_len,)

    cases = []
    for current_family_id in family_ids:
        for current_seq_len in seq_lens:
            cases.append(get_case(current_family_id, current_seq_len))
    return tuple(cases)
