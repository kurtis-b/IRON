# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from itertools import product

_RUNTIME_FAMILY = "pipelined_addnorm_ffn_addnorm"
_PRACTICAL_FAMILY = "pipelined_addnorm_ffn_addnorm_practical"
_THEORETICAL_FAMILY = "pipelined_addnorm_ffn_addnorm_theoretical"

_BLOCK3_RETAINED_FAMILIES = {
    (96, 96),
    (96, 192),
    (96, 384),
    (96, 768),
    (192, 96),
    (384, 96),
    (768, 96),
    (768, 3072),
}

_BLOCK3_PROMOTED_RUNTIME_TOPOLOGIES = {
    (96, 96): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 0,
        },
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 4,
            "parallel_seq": 2,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
    ),
    (192, 96): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 2,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
    ),
    (384, 96): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 4,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
    ),
    (768, 96): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 8,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
    ),
    (768, 3072): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 8,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
    ),
    (96, 192): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 4,
            "parallel_seq": 1,
            "parallel_int_dim": 2,
            "gelu_stage": 1,
        },
    ),
    (96, 384): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_int_dim": 4,
            "gelu_stage": 1,
        },
    ),
    (96, 768): (
        {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 1,
            "num_aie_columns": 2,
            "parallel_seq": 1,
            "parallel_int_dim": 1,
            "gelu_stage": 1,
        },
    ),
}

_BLOCK3_TILE_M_CHOICES = (16,)
_BLOCK3_TILE_K_CHOICES = (96,)
_BLOCK3_TILE_N_CHOICES = (96,)
_BLOCK3_PARALLEL_SEQ_CHOICES = (1, 2, 4)
_BLOCK3_PARALLEL_INT_DIM_CHOICES = (1, 2, 4)
_BLOCK3_SUPPORTED_DOWN_PROJ_DEPTHS = (1, 2, 4, 8)
_BLOCK3_PRACTICAL_MAX_CANDIDATES = 24
_BLOCK3_RUNTIME_MAX_CANDIDATES = 12


def _topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


def _with_family(config: dict[str, int], topology_family: str) -> dict[str, int | str]:
    result: dict[str, int | str] = dict(config)
    result["topology_id"] = _topology_id(config)
    result["topology_family"] = topology_family
    return result


def _gelu_stage_choices(*, down_proj_depth: int) -> tuple[int, ...]:
    if down_proj_depth == 1:
        return (0, 1)
    return (1,)


def _required_num_aie_columns(*, parallel_seq: int, parallel_int_dim: int) -> int:
    return 2 * max(parallel_seq, parallel_int_dim)


def _n_aie_cores_needed(*, parallel_seq: int, parallel_int_dim: int) -> int:
    # Copied addnorm_ffn structure: 2 LN1 + 2 LN2 + 2 FFN stages per B tile.
    return parallel_seq * (4 + 2 * parallel_int_dim)


def _candidate_sort_key(config: dict[str, int]) -> tuple[int, int, int, int, int, str]:
    parallel_seq = int(config["parallel_seq"])
    parallel_int_dim = int(config["parallel_int_dim"])
    tile_m = int(config["tile_m"])
    tile_n = int(config["tile_n"])
    lane_parallelism = parallel_seq * parallel_int_dim
    sequence_chunk = tile_m * parallel_seq
    intermediate_chunk = tile_n * parallel_int_dim
    return (
        lane_parallelism,
        sequence_chunk,
        intermediate_chunk,
        parallel_seq,
        int(config["gelu_stage"]),
        _topology_id(config),
    )


def _dedupe_configs(
    candidates: list[dict[str, int]],
) -> list[dict[str, int]]:
    unique: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        unique[_topology_id(candidate)] = candidate
    return list(unique.values())


def _is_valid_candidate(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    config: dict[str, int],
) -> bool:
    tile_m = int(config["tile_m"])
    tile_k = int(config["tile_k"])
    tile_n = int(config["tile_n"])
    parallel_seq = int(config["parallel_seq"])
    parallel_int_dim = int(config["parallel_int_dim"])
    down_proj_depth = int(config["down_proj_depth"])
    num_aie_columns = int(config["num_aie_columns"])

    if tile_m not in _BLOCK3_TILE_M_CHOICES:
        return False
    if tile_k not in _BLOCK3_TILE_K_CHOICES:
        return False
    if tile_n not in _BLOCK3_TILE_N_CHOICES:
        return False
    if parallel_seq not in _BLOCK3_PARALLEL_SEQ_CHOICES:
        return False
    if parallel_int_dim not in _BLOCK3_PARALLEL_INT_DIM_CHOICES:
        return False
    if down_proj_depth not in _BLOCK3_SUPPORTED_DOWN_PROJ_DEPTHS:
        return False
    if num_aie_columns != _required_num_aie_columns(
        parallel_seq=parallel_seq, parallel_int_dim=parallel_int_dim
    ):
        return False
    if num_aie_columns > 8:
        return False
    if hidden_size != tile_k * down_proj_depth:
        return False
    if intermediate_size % (tile_n * parallel_int_dim) != 0:
        return False
    if seq_len % (tile_m * parallel_seq) != 0:
        return False
    if _n_aie_cores_needed(
        parallel_seq=parallel_seq, parallel_int_dim=parallel_int_dim
    ) > min(32, num_aie_columns * 4):
        return False
    if int(config["gelu_stage"]) not in _gelu_stage_choices(
        down_proj_depth=down_proj_depth
    ):
        return False
    return True


@lru_cache(maxsize=None)
def _block3_theoretical_configs_cached(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[dict[str, int], ...]:
    candidates: list[dict[str, int]] = []
    for tile_m, tile_k, tile_n in product(
        _BLOCK3_TILE_M_CHOICES,
        _BLOCK3_TILE_K_CHOICES,
        _BLOCK3_TILE_N_CHOICES,
    ):
        if hidden_size % tile_k != 0:
            continue
        down_proj_depth = hidden_size // tile_k
        if down_proj_depth not in _BLOCK3_SUPPORTED_DOWN_PROJ_DEPTHS:
            continue
        for parallel_seq, parallel_int_dim in product(
            _BLOCK3_PARALLEL_SEQ_CHOICES,
            _BLOCK3_PARALLEL_INT_DIM_CHOICES,
        ):
            num_aie_columns = _required_num_aie_columns(
                parallel_seq=parallel_seq,
                parallel_int_dim=parallel_int_dim,
            )
            for gelu_stage in _gelu_stage_choices(down_proj_depth=down_proj_depth):
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
                if _is_valid_candidate(
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    config=candidate,
                ):
                    candidates.append(candidate)
    candidates = _dedupe_configs(candidates)
    return tuple(sorted(candidates, key=_candidate_sort_key, reverse=True))


@lru_cache(maxsize=None)
def _block3_practical_configs_cached(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    max_candidates: int,
) -> tuple[dict[str, int], ...]:
    if max_candidates <= 0:
        raise ValueError("Block 3 practical exploration requires max_candidates > 0")
    candidates = list(
        _block3_theoretical_configs_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    )
    return tuple(candidates[:max_candidates])


@lru_cache(maxsize=None)
def _runtime_topologies_cached(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[dict[str, int | str], ...]:
    retained = _BLOCK3_PROMOTED_RUNTIME_TOPOLOGIES.get((hidden_size, intermediate_size))
    if retained is not None:
        runtime_configs = [
            config
            for config in retained
            if _is_valid_candidate(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                config=config,
            )
        ]
    else:
        runtime_configs = list(
            _block3_practical_configs_cached(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                max_candidates=_BLOCK3_RUNTIME_MAX_CANDIDATES,
            )
        )

    return tuple(_with_family(config, _RUNTIME_FAMILY) for config in runtime_configs)


def addnorm_ffn_addnorm_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    return list(
        _runtime_topologies_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    )


def addnorm_ffn_addnorm_practical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    max_candidates: int = _BLOCK3_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    return [
        _with_family(config, _PRACTICAL_FAMILY)
        for config in _block3_practical_configs_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_candidates=max_candidates,
        )
    ]


def addnorm_ffn_addnorm_theoretical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    return [
        _with_family(config, _THEORETICAL_FAMILY)
        for config in _block3_theoretical_configs_cached(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    ]


def addnorm_ffn_addnorm_design(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    runtime_topologies = addnorm_ffn_addnorm_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if topology_id is None:
        if not runtime_topologies:
            raise ValueError(
                f"No staged Block 3 topology for {seq_len}/{hidden_size}/{intermediate_size}"
            )
        return dict(runtime_topologies[0])

    for candidate in runtime_topologies:
        if str(candidate["topology_id"]) == topology_id:
            return dict(candidate)

    raise ValueError(
        f"Unknown Block 3 topology_id={topology_id!r} for {hidden_size}/{intermediate_size}"
    )
