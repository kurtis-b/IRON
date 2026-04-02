# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

_RUNTIME_FAMILY = "pipelined_addnorm_ffn_addnorm"
_PRACTICAL_FAMILY = "pipelined_addnorm_ffn_addnorm_practical"
_THEORETICAL_FAMILY = "pipelined_addnorm_ffn_addnorm_theoretical"


def _topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_pi{config['parallel_int_dim']}"
        f"_d{config['down_proj_depth']}_g{config['gelu_stage']}"
    )


_BLOCK3_BASE_CONFIGS = {
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
    ),
}


def _with_family(config: dict[str, int], topology_family: str) -> dict[str, int | str]:
    result: dict[str, int | str] = dict(config)
    result["topology_id"] = _topology_id(config)
    result["topology_family"] = topology_family
    return result


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

    if parallel_seq < 1 or parallel_seq > 1:
        return False
    if parallel_int_dim < 1:
        return False
    if num_aie_columns < max(parallel_int_dim + 1, 2):
        return False
    if hidden_size != tile_k * down_proj_depth:
        return False
    if intermediate_size % (tile_n * parallel_int_dim) != 0:
        return False
    if seq_len % (tile_m * parallel_seq) != 0:
        return False
    if tile_m % 16 != 0:
        return False
    return True


@lru_cache(maxsize=None)
def _runtime_topologies_cached(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[dict[str, int | str], ...]:
    configs = _BLOCK3_BASE_CONFIGS.get((hidden_size, intermediate_size), ())
    return tuple(
        _with_family(config, _RUNTIME_FAMILY)
        for config in configs
        if _is_valid_candidate(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            config=config,
        )
    )


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
) -> list[dict[str, int | str]]:
    return [
        _with_family(
            {
                key: int(value)
                for key, value in candidate.items()
                if key not in ("topology_id", "topology_family")
            },
            _PRACTICAL_FAMILY,
        )
        for candidate in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    ]


def addnorm_ffn_addnorm_theoretical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, int | str]]:
    return [
        _with_family(
            {
                key: int(value)
                for key, value in candidate.items()
                if key not in ("topology_id", "topology_family")
            },
            _THEORETICAL_FAMILY,
        )
        for candidate in addnorm_ffn_addnorm_topologies(
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
