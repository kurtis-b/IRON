# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from itertools import product

_BLOCK3_TOPOLOGIES = {
    (768, 3072): [
        {
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 6,
            "gelu_stage": 0,
        },
        {
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 0,
        },
    ],
    (1024, 4096): [
        {
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
        {
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 128,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
    ],
    (2048, 8192): [
        {
            "tile_m": 32,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_int_dim": 4,
            "gelu_stage": 0,
        },
        {
            "tile_m": 32,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
        {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
        {
            "tile_m": 128,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
        {
            "tile_m": 128,
            "tile_k": 64,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 0,
        },
    ],
}

_BLOCK3_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK3_AIE_ROWS_PER_COL = 4
_BLOCK3_AIE_COLUMNS = 8
_BLOCK3_COMPUTE_TILE_BYTES = 64 * 1024
_BLOCK3_STACK_BYTES = 0xD00
_BLOCK3_BF16_BYTES = 2
_BLOCK3_F32_BYTES = 4
_BLOCK3_PRACTICAL_TILE_M_CHOICES = (16, 32, 64)
_BLOCK3_PRACTICAL_MIN_TILE_K = 16
_BLOCK3_PRACTICAL_MIN_TILE_N = 16
_BLOCK3_PRACTICAL_MIN_AIE_COLUMNS = 8
_BLOCK3_PRACTICAL_MIN_LANE_PARALLELISM = 8
_BLOCK3_PRACTICAL_PARALLEL_SEQ_CHOICES = (1, 2, 4, 8)
_BLOCK3_PRACTICAL_MAX_CANDIDATES_PER_PS = 4
_BLOCK3_PRACTICAL_MAX_CANDIDATES = (
    len(_BLOCK3_PRACTICAL_PARALLEL_SEQ_CHOICES)
    * _BLOCK3_PRACTICAL_MAX_CANDIDATES_PER_PS
)
_BLOCK3_RUNTIME_NUM_AIE_COLUMNS = 8
_BLOCK3_RUNTIME_TILE_M_CHOICES = (16, 32, 64)
_BLOCK3_RUNTIME_MIN_TILE_K = 16
_BLOCK3_RUNTIME_MIN_TILE_N = 16
_BLOCK3_RUNTIME_MAX_CANDIDATES = 8
_BLOCK3_RUNTIME_MAX_DOWN_PROJ_DEPTH = 8
_BLOCK3_BANK_SLOT_CAPACITY_BY_TILE_K = {
    16: 2,
    24: 2,
    32: 4,
    40: 4,
    48: 6,
    64: 8,
    80: 8,
    96: 8,
    120: 8,
    128: 8,
    160: 6,
    192: 8,
    256: 6,
    320: 4,
    384: 4,
    512: 2,
}


def addnorm_ffn_addnorm_topologies(
    *,
    seq_len: int | None = None,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    if seq_len is None:
        raise ValueError("Block 3 runtime topologies require seq_len")
    return [
        dict(candidate)
        for candidate in _addnorm_ffn_addnorm_runtime_topologies_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    ]


@lru_cache(maxsize=None)
def _addnorm_ffn_addnorm_runtime_topologies_cached(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[dict[str, int | str], ...]:
    runtime_pool = _block3_runtime_supported_candidates(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    preferred_ids = _block3_preferred_runtime_ids(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    selected = _block3_select_runtime_candidates(
        hidden_size=hidden_size,
        candidates=runtime_pool,
        preferred_ids=preferred_ids,
        limit=_BLOCK3_RUNTIME_MAX_CANDIDATES,
    )
    return tuple(
        {
            **candidate,
            "topology_id": _topology_id(candidate),
            "topology_family": "pipelined_addnorm_ffn_addnorm",
        }
        for candidate in sorted(
            selected,
            key=lambda topology: _block3_runtime_sort_key(
                hidden_size=hidden_size,
                candidate=topology,
            ),
            reverse=True,
        )
    )


def _block3_runtime_supported_candidates(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    preferred_ids = _block3_preferred_runtime_ids(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    runtime_pool = [
        {
            "tile_m": int(candidate["tile_m"]),
            "tile_k": int(candidate["tile_k"]),
            "tile_n": int(candidate["tile_n"]),
            "down_proj_depth": int(candidate["down_proj_depth"]),
            "num_aie_columns": int(candidate["num_aie_columns"]),
            "parallel_seq": int(candidate["parallel_seq"]),
            "parallel_int_dim": int(candidate["parallel_int_dim"]),
            "gelu_stage": int(candidate["gelu_stage"]),
        }
        for candidate in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        if _block3_runtime_candidate_allowed(
            seq_len=seq_len,
            hidden_size=hidden_size,
            candidate=candidate,
        )
    ]
    preferred_candidates = [
        candidate
        for candidate in runtime_pool
        if _topology_id(candidate) in preferred_ids
    ]
    runtime_pool = _block3_prune_runtime_tile_dominated(runtime_pool)
    runtime_pool = _block3_prune_runtime_chunk_dominated(runtime_pool)

    deduped_pool: list[dict[str, int | str]] = []
    seen_signatures: set[tuple[int, ...]] = set()
    for candidate in runtime_pool:
        signature = _block3_signature(candidate)
        if signature in seen_signatures:
            continue
        deduped_pool.append(candidate)
        seen_signatures.add(signature)
    for candidate in preferred_candidates:
        signature = _block3_signature(candidate)
        if signature in seen_signatures:
            continue
        deduped_pool.append(candidate)
        seen_signatures.add(signature)
    return deduped_pool


def _block3_preferred_runtime_ids(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> set[str]:
    preferred_ids: set[str] = set()
    for candidate in _BLOCK3_TOPOLOGIES.get((hidden_size, intermediate_size), ()):
        if not _block3_runtime_candidate_allowed(
            seq_len=seq_len,
            hidden_size=hidden_size,
            candidate=candidate,
        ):
            continue
        preferred_ids.add(_topology_id(candidate))
    return preferred_ids


def addnorm_ffn_addnorm_theoretical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    return [
        dict(candidate)
        for candidate in _addnorm_ffn_addnorm_theoretical_topologies_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    ]


@lru_cache(maxsize=None)
def _addnorm_ffn_addnorm_theoretical_topologies_cached(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[dict[str, int | str], ...]:
    if seq_len <= 0:
        raise ValueError("Block 3 requires seq_len > 0")
    if hidden_size <= 0:
        raise ValueError("Block 3 requires hidden_size > 0")
    if intermediate_size <= 0:
        raise ValueError("Block 3 requires intermediate_size > 0")

    topologies: list[dict[str, int | str]] = []
    for (
        parallel_seq,
        parallel_int_dim,
        tile_m,
        tile_k,
        tile_n,
        num_aie_columns,
        gelu_stage,
    ) in product(
        _BLOCK3_PARALLEL_SEQ_CHOICES,
        _divisors(intermediate_size),
        _divisors(seq_len),
        _divisors(hidden_size),
        _divisors(intermediate_size),
        range(1, _BLOCK3_AIE_COLUMNS + 1),
        (0, 1),
    ):
        if tile_m % 8 != 0:
            continue
        if tile_k % 8 != 0:
            continue
        if tile_n % 8 != 0:
            continue
        if not _block3_supports_seq_len(
            seq_len=seq_len,
            parallel_seq=parallel_seq,
            tile_m=tile_m,
        ):
            continue
        if intermediate_size % parallel_int_dim != 0:
            continue
        if intermediate_size % (tile_n * parallel_int_dim) != 0:
            continue

        if hidden_size % tile_k != 0:
            continue
        k_tiles = hidden_size // tile_k
        if k_tiles <= 0:
            continue

        if (
            _block3_required_core_count(
                parallel_seq=parallel_seq,
                parallel_int_dim=parallel_int_dim,
            )
            > num_aie_columns * _BLOCK3_AIE_ROWS_PER_COL
        ):
            continue

        for down_proj_depth in _divisors(k_tiles):
            if down_proj_depth <= 0:
                continue
            if k_tiles % down_proj_depth != 0:
                continue

            candidate = {
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "down_proj_depth": down_proj_depth,
                "num_aie_columns": num_aie_columns,
                "parallel_seq": parallel_seq,
                "parallel_int_dim": parallel_int_dim,
                "gelu_stage": gelu_stage,
            }
            topologies.append(
                {
                    **candidate,
                    "topology_id": _theoretical_topology_id(candidate),
                    "topology_family": "pipelined_addnorm_ffn_addnorm_theoretical",
                }
            )

    return tuple(topologies)


def addnorm_ffn_addnorm_practical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    max_candidates: int = _BLOCK3_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    return [
        dict(candidate)
        for candidate in _addnorm_ffn_addnorm_practical_topologies_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_candidates=max_candidates,
        )
    ]


@lru_cache(maxsize=None)
def _addnorm_ffn_addnorm_practical_topologies_cached(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    max_candidates: int = _BLOCK3_PRACTICAL_MAX_CANDIDATES,
) -> tuple[dict[str, int | str], ...]:
    if max_candidates <= 0:
        raise ValueError("Block 3 practical exploration requires max_candidates > 0")

    theoretical = addnorm_ffn_addnorm_theoretical_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    practical = [
        candidate
        for candidate in theoretical
        if _is_block3_practical_candidate(
            candidate,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )
    ]
    practical = _block3_prune_practical_depth_variants(practical)

    selected: list[dict[str, int | str]] = []
    selected_signatures: set[tuple[int, ...]] = set()

    runtime_signatures = {
        _block3_signature(candidate)
        for candidate in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    for candidate in sorted(theoretical, key=_block3_practical_sort_key, reverse=True):
        signature = _block3_signature(candidate)
        if signature not in runtime_signatures or signature in selected_signatures:
            continue
        selected.append(candidate)
        selected_signatures.add(signature)

    for parallel_seq in _BLOCK3_PRACTICAL_PARALLEL_SEQ_CHOICES:
        bucket = sorted(
            (
                candidate
                for candidate in practical
                if int(candidate["parallel_seq"]) == parallel_seq
            ),
            key=_block3_practical_sort_key,
            reverse=True,
        )
        for candidate in bucket:
            if (
                len(
                    [
                        selected_candidate
                        for selected_candidate in selected
                        if int(selected_candidate["parallel_seq"]) == parallel_seq
                    ]
                )
                >= _BLOCK3_PRACTICAL_MAX_CANDIDATES_PER_PS
            ):
                break
            signature = _block3_signature(candidate)
            if signature in selected_signatures:
                continue
            selected.append(candidate)
            selected_signatures.add(signature)

    return tuple(
        {
            **candidate,
            "topology_family": "pipelined_addnorm_ffn_addnorm_practical",
        }
        for candidate in sorted(selected, key=_block3_practical_sort_key, reverse=True)[
            :max_candidates
        ]
    )


def addnorm_ffn_addnorm_design(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    topologies = addnorm_ffn_addnorm_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if not topologies:
        raise ValueError(
            f"No Block 3 runtime topology supports seq_len={seq_len} for {hidden_size}/{intermediate_size}"
        )

    if topology_id is None:
        config = max(
            topologies,
            key=lambda topology: (
                _block3_required_core_count(
                    parallel_seq=int(topology["parallel_seq"]),
                    parallel_int_dim=int(topology["parallel_int_dim"]),
                ),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
            ),
        )
    else:
        try:
            config = next(
                candidate
                for candidate in topologies
                if str(candidate["topology_id"]) == topology_id
            )
        except StopIteration as exc:
            raise ValueError(
                f"Unknown Block 3 topology_id={topology_id!r} for {hidden_size}/{intermediate_size}"
            ) from exc

    tile_m = int(config["tile_m"])
    tile_k = int(config["tile_k"])
    tile_n = int(config["tile_n"])
    parallel_seq = int(config["parallel_seq"])
    parallel_int_dim = int(config["parallel_int_dim"])
    down_proj_depth = int(config["down_proj_depth"])
    gelu_stage = int(config["gelu_stage"])
    num_aie_columns = int(config["num_aie_columns"])

    if seq_len <= 0:
        raise ValueError("Block 3 requires seq_len > 0")
    if parallel_seq not in (1, 2, 4, 6, 8):
        raise ValueError("Block 3 requires parallel_seq in {1, 2, 4, 6, 8}")
    if not _block3_supports_seq_len(
        seq_len=seq_len,
        parallel_seq=parallel_seq,
        tile_m=tile_m,
    ):
        raise ValueError("Block 3 requires seq_len divisible by parallel_seq * tile_m")
    if hidden_size <= 0 or hidden_size % tile_k != 0:
        raise ValueError("Block 3 requires hidden_size divisible by tile_k")
    if intermediate_size <= 0 or intermediate_size % tile_n != 0:
        raise ValueError("Block 3 requires intermediate_size divisible by tile_n")
    if intermediate_size % parallel_int_dim != 0:
        raise ValueError(
            "Block 3 requires intermediate_size divisible by parallel_int_dim"
        )
    if down_proj_depth <= 0:
        raise ValueError("Block 3 requires down_proj_depth > 0")
    if (hidden_size // tile_k) % down_proj_depth != 0:
        raise ValueError(
            "Block 3 requires hidden_size / tile_k to be divisible by down_proj_depth"
        )
    if gelu_stage not in (0, 1):
        raise ValueError("Block 3 requires gelu_stage to be 0 or 1")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 3 requires 1 <= num_aie_columns <= 8")
    if (
        _block3_required_core_count(
            parallel_seq=parallel_seq,
            parallel_int_dim=parallel_int_dim,
        )
        > num_aie_columns * 4
    ):
        raise ValueError(
            "Block 3 topology requires more cores than the selected array columns provide"
        )

    return {
        **config,
        "topology_id": str(config["topology_id"]),
        "topology_family": str(config["topology_family"]),
    }


def _topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


def _block3_required_core_count(*, parallel_seq: int, parallel_int_dim: int) -> int:
    return parallel_seq * (2 + 2 * parallel_int_dim)


def _block3_supports_seq_len(
    *,
    seq_len: int,
    parallel_seq: int,
    tile_m: int,
) -> bool:
    return seq_len > 0 and seq_len % (parallel_seq * tile_m) == 0


def _block3_runtime_feasible(
    *,
    hidden_size: int,
    candidate: dict[str, int | str],
    seq_len: int | None,
) -> bool:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])

    if seq_len is not None and not _block3_supports_seq_len(
        seq_len=seq_len,
        parallel_seq=parallel_seq,
        tile_m=tile_m,
    ):
        return False

    return _block3_compute_tiles_fit_current_pipeline(
        hidden_size=hidden_size,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        parallel_int_dim=parallel_int_dim,
    )


def _block3_compute_tiles_fit_current_pipeline(
    *,
    hidden_size: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    parallel_int_dim: int,
) -> bool:
    return (
        _block3_max_compute_tile_bytes_current_pipeline(
            hidden_size=hidden_size,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            parallel_int_dim=parallel_int_dim,
        )
        <= _BLOCK3_COMPUTE_TILE_BYTES
    )


def _block3_max_compute_tile_bytes_current_pipeline(
    *,
    hidden_size: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    parallel_int_dim: int,
) -> int:
    # Model the mandatory resident storage for the current fused Block 3
    # pipeline assuming every eligible compute-side FIFO can drop to depth 1.
    # The memtile-staged partial-accumulation path remains fixed and therefore
    # still contributes one input and one output A-tile buffer on down-proj.
    a_bytes = tile_m * tile_k * _BLOCK3_BF16_BYTES
    ar_bytes = 2 * a_bytes
    b_up_bytes = tile_k * tile_n * _BLOCK3_BF16_BYTES
    b_down_bytes = tile_n * tile_k * _BLOCK3_BF16_BYTES
    c_up_bytes = tile_m * tile_n * _BLOCK3_BF16_BYTES
    ln_weight_bytes = hidden_size * _BLOCK3_BF16_BYTES
    stats_bytes = tile_m * _BLOCK3_F32_BYTES

    ln1_tile_bytes = (
        ar_bytes + a_bytes + ln_weight_bytes + 2 * stats_bytes + _BLOCK3_STACK_BYTES
    )
    up_proj_tile_bytes = a_bytes + b_up_bytes + c_up_bytes + _BLOCK3_STACK_BYTES
    down_proj_tile_bytes = (
        c_up_bytes
        + b_down_bytes
        + (2 * a_bytes)  # mandatory staged partial tiles via the memtile path
        + a_bytes  # current reduction/output stream
        + (a_bytes if parallel_int_dim > 1 else 0)  # previous reduction input
        + _BLOCK3_STACK_BYTES
    )
    ln2_tile_bytes = (
        ar_bytes
        + a_bytes  # stage1 scratch
        + a_bytes  # final output FIFO object
        + 2 * ln_weight_bytes
        + 4 * stats_bytes
        + _BLOCK3_STACK_BYTES
    )

    return max(
        ln1_tile_bytes,
        up_proj_tile_bytes,
        down_proj_tile_bytes,
        ln2_tile_bytes,
    )


def _block3_runtime_candidate_allowed(
    *,
    seq_len: int,
    hidden_size: int,
    candidate: dict[str, int | str],
) -> bool:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    down_proj_depth = int(candidate["down_proj_depth"])
    if int(candidate["num_aie_columns"]) != _BLOCK3_RUNTIME_NUM_AIE_COLUMNS:
        return False
    if tile_m not in _BLOCK3_RUNTIME_TILE_M_CHOICES:
        return False
    if tile_k < _BLOCK3_RUNTIME_MIN_TILE_K:
        return False
    if tile_n < _BLOCK3_RUNTIME_MIN_TILE_N:
        return False
    if down_proj_depth > _BLOCK3_RUNTIME_MAX_DOWN_PROJ_DEPTH:
        return False
    if _block3_is_known_bad_runtime_candidate(
        seq_len=seq_len,
        hidden_size=hidden_size,
        candidate=candidate,
    ):
        return False
    if not _block3_runtime_stage1_bank_feasible(
        hidden_size=hidden_size,
        candidate=candidate,
    ):
        return False
    if not _block3_runtime_placement_feasible(candidate):
        return False
    return _block3_runtime_feasible(
        hidden_size=hidden_size,
        candidate=candidate,
        seq_len=seq_len,
    )


def _block3_is_known_bad_runtime_candidate(
    *,
    seq_len: int,
    hidden_size: int,
    candidate: dict[str, int | str],
) -> bool:
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])

    # The broad skinny-wide fence is no longer needed. The remaining failures
    # are isolated runtime/layout issues that have been reproduced directly:
    # - `tile_k=24` with very wide `tile_n` still produces runtime NaNs
    # - `512 x 1536` with `tile_k=16, tile_n=512` still fails shim BD lowering
    if tile_k == 24 and tile_n > 128:
        return True
    if seq_len == 512 and hidden_size == 1536 and tile_k == 16 and tile_n == 512:
        return True
    return False


def _block3_runtime_placement_feasible(candidate: dict[str, int | str]) -> bool:
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    num_aie_columns = int(candidate["num_aie_columns"])
    if parallel_seq > num_aie_columns:
        return False
    if parallel_seq > (num_aie_columns // 2):
        return parallel_int_dim == 1
    if parallel_seq < 3:
        return parallel_int_dim <= num_aie_columns - 2
    return (
        parallel_int_dim <= _BLOCK3_AIE_ROWS_PER_COL
        and parallel_seq <= num_aie_columns // 2
        and (2 * parallel_seq) <= num_aie_columns
    )


def _block3_stage1_bank_slot_capacity(*, tile_k: int) -> int:
    capacity = _BLOCK3_BANK_SLOT_CAPACITY_BY_TILE_K.get(tile_k)
    if capacity is not None:
        return capacity
    if tile_k >= 96:
        return 8
    if tile_k >= 64:
        return 8
    if tile_k >= 48:
        return 6
    if tile_k >= 32:
        return 4
    return 2


def _block3_available_stage1_bank_memtiles(
    candidate: dict[str, int | str],
) -> int:
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    num_aie_columns = int(candidate["num_aie_columns"])

    if parallel_seq > (num_aie_columns // 2):
        return 1
    if parallel_seq < 3:
        return 1 if parallel_int_dim > 4 else 2
    return 2


def _block3_required_stage1_bank_memtiles(
    *,
    hidden_size: int,
    candidate: dict[str, int | str],
) -> int:
    tile_k = int(candidate["tile_k"])
    k_tiles = hidden_size // tile_k
    slot_capacity = _block3_stage1_bank_slot_capacity(tile_k=tile_k)
    return math.ceil(k_tiles / slot_capacity)


def _block3_runtime_stage1_bank_feasible(
    *,
    hidden_size: int,
    candidate: dict[str, int | str],
) -> bool:
    required_bank_memtiles = _block3_required_stage1_bank_memtiles(
        hidden_size=hidden_size,
        candidate=candidate,
    )
    available_bank_memtiles = _block3_available_stage1_bank_memtiles(candidate)
    return required_bank_memtiles <= available_bank_memtiles


def _block3_practical_placement_feasible(candidate: dict[str, int | str]) -> bool:
    num_aie_columns = int(candidate["num_aie_columns"])
    return (
        _block3_required_core_count(
            parallel_seq=int(candidate["parallel_seq"]),
            parallel_int_dim=int(candidate["parallel_int_dim"]),
        )
        <= num_aie_columns * _BLOCK3_AIE_ROWS_PER_COL
    )


def _block3_prune_runtime_tile_dominated(
    candidates: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    grouped: dict[tuple[int, int, int, int, int], list[dict[str, int | str]]] = {}
    for candidate in candidates:
        key = (
            int(candidate["tile_m"]),
            int(candidate["parallel_seq"]),
            int(candidate["parallel_int_dim"]),
            int(candidate["down_proj_depth"]),
            int(candidate["gelu_stage"]),
        )
        grouped.setdefault(key, []).append(candidate)

    pruned: list[dict[str, int | str]] = []
    for group in grouped.values():
        for candidate in group:
            tile_m = int(candidate["tile_m"])
            tile_k = int(candidate["tile_k"])
            tile_n = int(candidate["tile_n"])
            dominated = any(
                int(other["tile_m"]) >= tile_m
                and int(other["tile_k"]) >= tile_k
                and int(other["tile_n"]) >= tile_n
                and (
                    int(other["tile_m"]) > tile_m
                    or int(other["tile_k"]) > tile_k
                    or int(other["tile_n"]) > tile_n
                )
                for other in group
                if other is not candidate
            )
            if not dominated:
                pruned.append(candidate)
    return pruned


def _block3_prune_runtime_chunk_dominated(
    candidates: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, int | str]]] = {}
    for candidate in candidates:
        key = (
            int(candidate["tile_m"]),
            int(candidate["parallel_seq"]),
            int(candidate["parallel_int_dim"]),
            int(candidate["gelu_stage"]),
        )
        grouped.setdefault(key, []).append(candidate)

    pruned: list[dict[str, int | str]] = []
    for group in grouped.values():
        for candidate in group:
            sequence_chunk = int(candidate["parallel_seq"]) * int(candidate["tile_m"])
            output_chunk = int(candidate["parallel_int_dim"]) * int(candidate["tile_n"])
            grouped_hidden_chunk = int(candidate["down_proj_depth"]) * int(
                candidate["tile_k"]
            )
            dominated = any(
                int(other["parallel_seq"]) * int(other["tile_m"]) >= sequence_chunk
                and int(other["parallel_int_dim"]) * int(other["tile_n"])
                >= output_chunk
                and int(other["down_proj_depth"]) * int(other["tile_k"])
                >= grouped_hidden_chunk
                and (
                    int(other["parallel_seq"]) * int(other["tile_m"]) > sequence_chunk
                    or int(other["parallel_int_dim"]) * int(other["tile_n"])
                    > output_chunk
                    or int(other["down_proj_depth"]) * int(other["tile_k"])
                    > grouped_hidden_chunk
                )
                for other in group
                if other is not candidate
            )
            if not dominated:
                pruned.append(candidate)
    return pruned


def _block3_runtime_sort_key(
    *,
    hidden_size: int,
    candidate: dict[str, int | str],
) -> tuple[int, ...]:
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    down_proj_depth = int(candidate["down_proj_depth"])
    gelu_stage = int(candidate["gelu_stage"])
    core_count = _block3_required_core_count(
        parallel_seq=parallel_seq,
        parallel_int_dim=parallel_int_dim,
    )
    lane_parallelism = parallel_seq * parallel_int_dim
    sequence_chunk = parallel_seq * tile_m
    output_chunk = parallel_int_dim * tile_n
    required_bank_memtiles = _block3_required_stage1_bank_memtiles(
        hidden_size=hidden_size,
        candidate=candidate,
    )
    max_compute_tile_bytes = _block3_max_compute_tile_bytes_current_pipeline(
        hidden_size=hidden_size,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        parallel_int_dim=parallel_int_dim,
    )
    return (
        core_count,
        lane_parallelism,
        sequence_chunk,
        tile_m,
        output_chunk,
        tile_n,
        -required_bank_memtiles,
        max_compute_tile_bytes,
        -tile_k,
        -down_proj_depth,
        gelu_stage,
    )


def _block3_select_runtime_candidates(
    *,
    hidden_size: int,
    candidates: list[dict[str, int | str]],
    preferred_ids: set[str],
    limit: int,
) -> list[dict[str, int | str]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: _block3_runtime_sort_key(
            hidden_size=hidden_size,
            candidate=candidate,
        ),
        reverse=True,
    )
    chosen: list[dict[str, int | str]] = []
    chosen_signatures: set[tuple[int, ...]] = set()

    def add_candidate(candidate: dict[str, int | str]) -> None:
        signature = _block3_signature(candidate)
        if signature in chosen_signatures:
            return
        chosen.append(candidate)
        chosen_signatures.add(signature)

    for candidate in ranked:
        if len(chosen) >= limit:
            return chosen
        if _topology_id(candidate) not in preferred_ids:
            continue
        add_candidate(candidate)

    by_tile_pair: dict[tuple[int, int], list[dict[str, int | str]]] = {}
    for candidate in ranked:
        tile_pair = (
            int(candidate["tile_m"]),
            int(candidate["tile_k"]),
        )
        by_tile_pair.setdefault(tile_pair, []).append(candidate)

    for tile_pair in sorted(
        by_tile_pair,
        key=lambda key: _block3_runtime_sort_key(
            hidden_size=hidden_size,
            candidate=by_tile_pair[key][0],
        ),
        reverse=True,
    ):
        candidate = by_tile_pair[tile_pair][0]
        add_candidate(candidate)
        if len(chosen) >= limit:
            return chosen

    by_shape: dict[tuple[int, int, int], list[dict[str, int | str]]] = {}
    for candidate in ranked:
        shape_key = (
            int(candidate["parallel_seq"]),
            int(candidate["parallel_int_dim"]),
            int(candidate["gelu_stage"]),
        )
        by_shape.setdefault(shape_key, []).append(candidate)

    for shape_key in sorted(
        by_shape,
        key=lambda key: (
            _block3_required_core_count(
                parallel_seq=key[0],
                parallel_int_dim=key[1],
            ),
            key[0] * key[1],
            key[0],
            key[1],
            key[2],
        ),
        reverse=True,
    ):
        candidate = by_shape[shape_key][0]
        add_candidate(candidate)
        if len(chosen) >= limit:
            return chosen

    for candidate in ranked:
        add_candidate(candidate)
        if len(chosen) >= limit:
            break
    return chosen


def _theoretical_topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}"
        f"_n{config['tile_n']}_c{config['num_aie_columns']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


def _divisors(value: int) -> tuple[int, ...]:
    divisors = set()
    for i in range(1, int(math.isqrt(value)) + 1):
        if value % i != 0:
            continue
        divisors.add(i)
        divisors.add(value // i)
    return tuple(sorted(divisors))


def _block3_signature(candidate: dict[str, int | str]) -> tuple[int, ...]:
    return (
        int(candidate["tile_m"]),
        int(candidate["tile_k"]),
        int(candidate["tile_n"]),
        int(candidate["down_proj_depth"]),
        int(candidate["num_aie_columns"]),
        int(candidate["parallel_seq"]),
        int(candidate["parallel_int_dim"]),
        int(candidate["gelu_stage"]),
    )


def _block3_prune_practical_depth_variants(
    candidates: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    best_by_shape: dict[tuple[int, ...], dict[str, int | str]] = {}
    for candidate in candidates:
        shape_key = (
            int(candidate["tile_m"]),
            int(candidate["tile_k"]),
            int(candidate["tile_n"]),
            int(candidate["num_aie_columns"]),
            int(candidate["parallel_seq"]),
            int(candidate["parallel_int_dim"]),
            int(candidate["gelu_stage"]),
        )
        current = best_by_shape.get(shape_key)
        if current is None or int(candidate["down_proj_depth"]) > int(
            current["down_proj_depth"]
        ):
            best_by_shape[shape_key] = candidate
    return list(best_by_shape.values())


def _is_block3_practical_candidate(
    candidate: dict[str, int | str],
    *,
    seq_len: int,
    hidden_size: int,
) -> bool:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])

    lane_parallelism = parallel_seq * parallel_int_dim
    return (
        num_aie_columns >= _BLOCK3_PRACTICAL_MIN_AIE_COLUMNS
        and _block3_supports_seq_len(
            seq_len=seq_len,
            parallel_seq=parallel_seq,
            tile_m=tile_m,
        )
        and _block3_practical_placement_feasible(candidate)
        and _block3_runtime_feasible(
            hidden_size=hidden_size,
            candidate=candidate,
            seq_len=seq_len,
        )
        and tile_m in _BLOCK3_PRACTICAL_TILE_M_CHOICES
        and tile_k >= _BLOCK3_PRACTICAL_MIN_TILE_K
        and tile_n >= _BLOCK3_PRACTICAL_MIN_TILE_N
        and lane_parallelism >= _BLOCK3_PRACTICAL_MIN_LANE_PARALLELISM
    )


def _block3_practical_sort_key(candidate: dict[str, int | str]) -> tuple[int, ...]:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    down_proj_depth = int(candidate["down_proj_depth"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    gelu_stage = int(candidate["gelu_stage"])

    core_count = _block3_required_core_count(
        parallel_seq=parallel_seq,
        parallel_int_dim=parallel_int_dim,
    )
    lane_parallelism = parallel_seq * parallel_int_dim
    sequence_chunk = parallel_seq * tile_m
    return (
        core_count,
        lane_parallelism,
        num_aie_columns,
        sequence_chunk,
        tile_m,
        parallel_int_dim * tile_n,
        tile_n,
        -tile_k,
        -down_proj_depth,
        gelu_stage,
    )
