#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

BlockKind = Literal["qkv_proj", "mha_out_proj", "addnorm", "ffn"]

QKVProjCandidate = tuple[int, int, int, int, int]
MHAOutProjCandidate = tuple[int, int, int, int, int, int]
AddNormCandidate = tuple[int, int]
FFNCandidate = tuple[int, bool, bool, int, int, int, int, int, int, int | None, int]

BLOCK_KINDS: tuple[BlockKind, ...] = ("qkv_proj", "mha_out_proj", "addnorm", "ffn")
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
    addnorm: tuple[AddNormCandidate, ...]
    ffn: tuple[FFNCandidate, ...]

    @property
    def seq_len(self) -> int:
        return self.workload.seq_len

    @property
    def family_label(self) -> str:
        return self.workload.family_label

    def candidates(self, block_kind: BlockKind) -> tuple[tuple[object, ...], ...]:
        return getattr(self, block_kind)


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
    addnorm: Sequence[AddNormCandidate] | None = None,
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
        addnorm=_normalize_candidates(addnorm or ((8, workload.hidden_size),)),
        ffn=_normalize_candidates(
            ffn or ((8, False, False, 64, 48, 96, 8, 4, 4, None, 1),)
        ),
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
            ],
            ffn=[
                (8, False, False, 16, 96, 96, 8, 1, 8, None, 1),
                (8, False, False, 16, 96, 96, 8, 1, 4, None, 1),
                (8, False, False, 16, 96, 48, 8, 1, 6, None, 1),
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
                (8, False, False, 64, 96, 32, 8, 2, 6, None, 1),
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
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (4, 32, 64, 96, 2, 8),
                (2, 32, 64, 96, 4, 8),
            ],
            ffn=[
                (8, False, False, 64, 96, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 96, 48, 8, 4, 4, None, 1),
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
                (1, 32, 64, 128, 6, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 16, 128, 96, 8, 1, 8, None, 1),
                (8, False, False, 16, 128, 96, 8, 1, 4, None, 1),
                (8, False, False, 16, 128, 48, 8, 1, 6, None, 1),
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
                (8, False, False, 64, 128, 32, 8, 2, 6, None, 1),
                (8, False, False, 16, 128, 96, 8, 8, 2, None, 1),
                (8, False, False, 32, 128, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 32, 128, 96, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
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
                (4, 32, 64, 128, 2, 8),
                (2, 32, 64, 128, 4, 8),
            ],
            addnorm=[
                (8, 1024),
            ],
            ffn=[
                (8, False, False, 64, 128, 48, 8, 8, 2, None, 1),
                (8, False, False, 64, 128, 48, 8, 4, 4, None, 1),
            ],
        ),
    },
}


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
