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
            "compile_rows": 128,
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
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 96,
            "tile_n": 64,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 3,
            "gelu_stage": 1,
        },
    ],
    (1024, 4096): [
        {
            "compile_rows": 128,
            "tile_m": 32,
            "tile_k": 128,
            "tile_n": 32,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        }
    ],
}

_BLOCK3_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK3_AIE_ROWS_PER_COL = 4
_BLOCK3_AIE_COLUMNS = 8
_BLOCK3_MIN_COMPILE_ROWS = 128
_BLOCK3_PRACTICAL_MIN_COMPILE_ROWS = 128
_BLOCK3_PRACTICAL_MIN_TILE_M = 32
_BLOCK3_PRACTICAL_MIN_TILE_K = 64
_BLOCK3_PRACTICAL_MIN_TILE_N = 32
_BLOCK3_PRACTICAL_MIN_AIE_COLUMNS = 4
_BLOCK3_PRACTICAL_MIN_LANE_PARALLELISM = 8
_BLOCK3_PRACTICAL_MAX_CANDIDATES = 64


def addnorm_ffn_addnorm_topologies(
    *,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    try:
        topologies = _BLOCK3_TOPOLOGIES[(hidden_size, intermediate_size)]
    except KeyError as exc:
        raise ValueError(
            "Block 3 currently supports only the retained thesis families 768/3072 and 1024/4096"
        ) from exc
    return [
        {
            **candidate,
            "topology_id": _topology_id(candidate),
            "topology_family": "pipelined_addnorm_ffn_addnorm",
        }
        for candidate in topologies
    ]


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
        _divisors(_block3_compile_row_limit(seq_len)),
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
        if intermediate_size % parallel_int_dim != 0:
            continue
        if intermediate_size % (tile_n * parallel_int_dim) != 0:
            continue

        down_proj_depth = hidden_size // tile_k
        if hidden_size % tile_k != 0:
            continue
        if down_proj_depth <= 0:
            continue

        if (
            parallel_seq * (1 + 2 * parallel_int_dim)
            > num_aie_columns * _BLOCK3_AIE_ROWS_PER_COL
        ):
            continue

        for compile_rows in _block3_compile_row_candidates(
            seq_len=seq_len,
            parallel_seq=parallel_seq,
            tile_m=tile_m,
        ):
            candidate = {
                "compile_rows": compile_rows,
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
    retained_signatures = {
        _block3_signature(candidate)
        for candidate in addnorm_ffn_addnorm_topologies(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }

    ranked = sorted(theoretical, key=_block3_practical_sort_key, reverse=True)

    selected: list[dict[str, int | str]] = []
    selected_ids: set[str] = set()

    def add_candidate(candidate: dict[str, int | str]) -> None:
        selected.append(
            {
                **candidate,
                "topology_family": "pipelined_addnorm_ffn_addnorm_practical",
            }
        )
        selected_ids.add(str(candidate["topology_id"]))

    for candidate in ranked:
        if _block3_signature(candidate) not in retained_signatures:
            continue
        add_candidate(candidate)

    for candidate in ranked:
        topology_id = str(candidate["topology_id"])
        if topology_id in selected_ids:
            continue
        if not _is_block3_practical_candidate(candidate, seq_len=seq_len):
            continue
        add_candidate(candidate)
        if len(selected) >= max_candidates:
            break

    return tuple(sorted(selected, key=_block3_practical_sort_key, reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Block 3 AddNormFFNAddNorm Design",
        description="Resolve retained thesis topology parameters for Block 3",
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--topology-id", type=str, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            addnorm_ffn_addnorm_design(
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                topology_id=args.topology_id,
            ),
            sort_keys=True,
        )
    )


def addnorm_ffn_addnorm_design(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    topologies = addnorm_ffn_addnorm_topologies(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    if topology_id is None:
        config = max(
            topologies,
            key=lambda topology: (
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

    compile_rows = int(config["compile_rows"])
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
    if compile_rows <= 0 or compile_rows % (parallel_seq * tile_m) != 0:
        raise ValueError(
            "Block 3 requires compile_rows divisible by parallel_seq * tile_m"
        )
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
    if gelu_stage not in (0, 1):
        raise ValueError("Block 3 requires gelu_stage to be 0 or 1")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 3 requires 1 <= num_aie_columns <= 8")
    if parallel_seq * (1 + 2 * parallel_int_dim) > num_aie_columns * 4:
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


def _theoretical_topology_id(config: dict[str, int]) -> str:
    return (
        f"cr{config['compile_rows']}_m{config['tile_m']}_k{config['tile_k']}"
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


def _next_multiple(value: int, factor: int) -> int:
    return ((value + factor - 1) // factor) * factor


def _block3_compile_row_limit(seq_len: int) -> int:
    return max(_BLOCK3_MIN_COMPILE_ROWS, seq_len)


def _block3_compile_row_candidates(
    *,
    seq_len: int,
    parallel_seq: int,
    tile_m: int,
) -> tuple[int, ...]:
    base_rows = parallel_seq * tile_m
    if base_rows <= 0:
        return tuple()
    limit = _next_multiple(_block3_compile_row_limit(seq_len), base_rows)
    return tuple(range(base_rows, limit + 1, base_rows))


def _block3_signature(candidate: dict[str, int | str]) -> tuple[int, ...]:
    return (
        int(candidate["compile_rows"]),
        int(candidate["tile_m"]),
        int(candidate["tile_k"]),
        int(candidate["tile_n"]),
        int(candidate["down_proj_depth"]),
        int(candidate["num_aie_columns"]),
        int(candidate["parallel_seq"]),
        int(candidate["parallel_int_dim"]),
        int(candidate["gelu_stage"]),
    )


def _is_block3_practical_candidate(
    candidate: dict[str, int | str],
    *,
    seq_len: int,
) -> bool:
    compile_rows = int(candidate["compile_rows"])
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    gelu_stage = int(candidate["gelu_stage"])

    lane_parallelism = parallel_seq * parallel_int_dim
    return (
        compile_rows >= _BLOCK3_PRACTICAL_MIN_COMPILE_ROWS
        and compile_rows <= _block3_compile_row_limit(seq_len)
        and tile_m >= _BLOCK3_PRACTICAL_MIN_TILE_M
        and tile_k >= _BLOCK3_PRACTICAL_MIN_TILE_K
        and tile_n >= _BLOCK3_PRACTICAL_MIN_TILE_N
        and num_aie_columns >= _BLOCK3_PRACTICAL_MIN_AIE_COLUMNS
        and lane_parallelism >= _BLOCK3_PRACTICAL_MIN_LANE_PARALLELISM
        and gelu_stage == 1
    )


def _block3_practical_sort_key(candidate: dict[str, int | str]) -> tuple[int, ...]:
    compile_rows = int(candidate["compile_rows"])
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    down_proj_depth = int(candidate["down_proj_depth"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_int_dim = int(candidate["parallel_int_dim"])
    gelu_stage = int(candidate["gelu_stage"])

    lane_parallelism = parallel_seq * parallel_int_dim
    sequence_chunk = parallel_seq * tile_m
    output_chunk = tile_n * parallel_int_dim
    return (
        lane_parallelism,
        num_aie_columns,
        output_chunk,
        tile_k,
        compile_rows,
        sequence_chunk,
        tile_n,
        -down_proj_depth,
        gelu_stage,
    )


if __name__ == "__main__":
    main()
