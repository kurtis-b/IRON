# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from itertools import product

_BLOCK1_RUNTIME_NUM_AIE_COLUMNS_CHOICES = (6, 8)
_BLOCK1_RETAINED_FAMILIES = {
    (12, 64),
    (16, 64),
}

_BLOCK1_RETAINED_AXES = {
    "tile_m": (64, 32),
    "tile_k": (64,),
    "tile_n": (16, 32),
    "num_aie_columns": (8,),
    "parallel_seq": (1,),
    "parallel_emb": (1,),
}

_BLOCK1_PROMOTED_RUNTIME_TOPOLOGIES = {
    (12, 64): (
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 2,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 4,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 8,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_emb": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_emb": 1,
        },
    ),
    (16, 64): (
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 2,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 4,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_emb": 8,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 2,
            "parallel_emb": 1,
        },
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 4,
            "parallel_emb": 1,
        },
    ),
}

_BLOCK1_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK1_AIE_DATA_MEM_SIZE_BYTES = 65536
_BLOCK1_GEMM_L1_BUFFER_COPIES = 2
_BLOCK1_GEMM_TILE_FIXED_OVERHEAD_BYTES = 3392
_BLOCK1_MICROKERNEL_TILE_M = 8
_BLOCK1_MICROKERNEL_TILE_K = 8
_BLOCK1_MICROKERNEL_TILE_N = 8
_BLOCK1_PRACTICAL_MIN_TILE_M = _BLOCK1_MICROKERNEL_TILE_M
_BLOCK1_PRACTICAL_MIN_TILE_K = _BLOCK1_MICROKERNEL_TILE_K
_BLOCK1_PRACTICAL_MIN_TILE_N = _BLOCK1_MICROKERNEL_TILE_N
_BLOCK1_PRACTICAL_MAX_CANDIDATES = 20
_BLOCK1_RUNTIME_MAX_CANDIDATES = 12
_BLOCK1_SEQ_INDEPENDENT_MAX_CANDIDATES = 12
_BLOCK1_SEED_TILE_M_CHOICES = (8, 16, 32, 64)
_BLOCK1_SEED_TILE_K_MAX = 256
_BLOCK1_SEED_TILE_N_MAX = 64


def qkv_proj_topologies(
    *,
    hidden_size: int,
    num_heads: int,
    seq_len: int | None = None,
) -> list[dict[str, int | str]]:
    if hidden_size % num_heads != 0:
        raise ValueError("Block 1 requires hidden_size divisible by num_heads")
    head_dim = hidden_size // num_heads
    if seq_len is None:
        unique_topologies = _block1_seq_independent_supported_candidates(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        topology_id_fn = _theoretical_topology_id
    else:
        if seq_len <= 0:
            raise ValueError("Block 1 requires seq_len > 0")
        unique_topologies = _block1_runtime_supported_candidates(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        topology_id_fn = _theoretical_topology_id
    return [
        {
            **candidate,
            "topology_id": topology_id_fn(candidate),
            "topology_family": "shared_runtime_qkv_proj",
        }
        for candidate in unique_topologies
    ]


def qkv_proj_practical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    max_candidates: int = _BLOCK1_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    return _block1_practical_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_candidates=max_candidates,
    )


def qkv_proj_design(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    head_dim = hidden_size // num_heads

    if topology_id is None:
        config = _block1_runtime_supported_candidates(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )[0]
        resolved_topology_id = _theoretical_topology_id(config)
    else:
        topologies = qkv_proj_topologies(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=seq_len,
        )
        try:
            config = next(
                candidate
                for candidate in topologies
                if str(candidate["topology_id"]) == topology_id
            )
            resolved_topology_id = str(config["topology_id"])
        except StopIteration as exc:
            raise ValueError(
                f"Unknown Block 1 topology_id={topology_id!r} for {num_heads}x{head_dim}"
            ) from exc

    tile_m = int(config["tile_m"])
    tile_k = int(config["tile_k"])
    tile_n = int(config["tile_n"])
    parallel_seq = int(config["parallel_seq"])
    parallel_emb = int(config["parallel_emb"])
    num_aie_columns = int(config["num_aie_columns"])

    if parallel_seq not in (1, 2, 4, 6, 8):
        raise ValueError("Block 1 requires parallel_seq in {1, 2, 4, 6, 8}")
    if seq_len % (parallel_seq * tile_m) != 0:
        raise ValueError("Block 1 requires seq_len divisible by parallel_seq * tile_m")
    if hidden_size % tile_k != 0:
        raise ValueError("Block 1 requires hidden_size divisible by tile_k")
    combined_hidden_size = hidden_size * 3
    if combined_hidden_size % tile_n != 0:
        raise ValueError("Block 1 requires combined_hidden_size divisible by tile_n")
    if parallel_emb < 1:
        raise ValueError("Block 1 requires parallel_emb >= 1")
    if hidden_size % parallel_emb != 0:
        raise ValueError("Block 1 requires hidden_size divisible by parallel_emb")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 1 requires 1 <= num_aie_columns <= 8")
    if num_aie_columns % parallel_emb != 0:
        raise ValueError("Block 1 requires num_aie_columns divisible by parallel_emb")

    return {
        **config,
        "topology_id": resolved_topology_id,
        "topology_family": str(
            config.get("topology_family", "shared_runtime_qkv_proj")
        ),
    }


def _enumerate_block1_topologies(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    topologies: list[dict[str, int]] = []
    combined_hidden_size = hidden_size * 3
    for (
        tile_m,
        tile_k,
        tile_n,
        num_aie_columns,
        parallel_seq,
        parallel_emb,
    ) in product(
        _BLOCK1_RETAINED_AXES["tile_m"],
        _BLOCK1_RETAINED_AXES["tile_k"],
        _BLOCK1_RETAINED_AXES["tile_n"],
        _BLOCK1_RETAINED_AXES["num_aie_columns"],
        _BLOCK1_RETAINED_AXES["parallel_seq"],
        _BLOCK1_RETAINED_AXES["parallel_emb"],
    ):
        if hidden_size % tile_k != 0:
            continue
        if combined_hidden_size % tile_n != 0:
            continue
        if hidden_size % parallel_emb != 0:
            continue
        topologies.append(
            {
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "num_aie_columns": num_aie_columns,
                "parallel_seq": parallel_seq,
                "parallel_emb": parallel_emb,
            }
        )
    return topologies


def _block1_seq_independent_supported_candidates(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    topologies = _block1_runtime_seed_candidates(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    topologies = _block1_select_candidates(
        topologies,
        sort_key=_block1_runtime_sort_key,
        preferred_ids={
            _theoretical_topology_id(candidate)
            for candidate in _BLOCK1_PROMOTED_RUNTIME_TOPOLOGIES.get(
                (num_heads, head_dim), ()
            )
        }
        | _block1_best_ids_by_num_aie_columns(
            topologies,
            sort_key=_block1_runtime_sort_key,
        ),
        limit=_BLOCK1_SEQ_INDEPENDENT_MAX_CANDIDATES,
    )
    unique_topologies = []
    seen_topology_ids: set[str] = set()
    for candidate in topologies:
        topology_id = _theoretical_topology_id(candidate)
        if topology_id in seen_topology_ids:
            continue
        seen_topology_ids.add(topology_id)
        unique_topologies.append(candidate)
    return unique_topologies


def _block1_runtime_seed_candidates(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    combined_hidden_size = hidden_size * 3
    topologies = list(
        _BLOCK1_PROMOTED_RUNTIME_TOPOLOGIES.get((num_heads, head_dim), ())
    )
    seed_tile_ks = _preferred_divisors(
        hidden_size,
        minimum=_BLOCK1_PRACTICAL_MIN_TILE_K,
        maximum=min(hidden_size, _BLOCK1_SEED_TILE_K_MAX),
    )
    seed_tile_ns = _preferred_divisors(
        combined_hidden_size,
        minimum=_BLOCK1_PRACTICAL_MIN_TILE_N,
        maximum=min(combined_hidden_size, _BLOCK1_SEED_TILE_N_MAX),
    )
    for (
        tile_m,
        tile_k,
        tile_n,
        num_aie_columns,
        parallel_seq,
        parallel_emb,
    ) in product(
        _BLOCK1_SEED_TILE_M_CHOICES,
        seed_tile_ks,
        seed_tile_ns,
        _BLOCK1_RUNTIME_NUM_AIE_COLUMNS_CHOICES,
        (1, 2, 4),
        tuple(
            divisor
            for divisor in _divisors(hidden_size)
            if divisor <= max(_BLOCK1_RUNTIME_NUM_AIE_COLUMNS_CHOICES)
        ),
    ):
        candidate = {
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "num_aie_columns": num_aie_columns,
            "parallel_seq": parallel_seq,
            "parallel_emb": parallel_emb,
        }
        if not _block1_compute_tile_working_set_fits(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
        ):
            continue
        if not _block1_runtime_candidate_allowed(candidate, hidden_size=hidden_size):
            continue
        topologies.append(candidate)
    return topologies


def _block1_runtime_supported_candidates(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    runtime_pool = [
        {
            "tile_m": int(candidate["tile_m"]),
            "tile_k": int(candidate["tile_k"]),
            "tile_n": int(candidate["tile_n"]),
            "num_aie_columns": int(candidate["num_aie_columns"]),
            "parallel_seq": int(candidate["parallel_seq"]),
            "parallel_emb": int(candidate["parallel_emb"]),
        }
        for candidate in _block1_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        if _is_block1_practical_candidate(candidate)
        and _block1_runtime_candidate_allowed(candidate, hidden_size=hidden_size)
    ]
    preferred_ids = {
        _theoretical_topology_id(candidate)
        for candidate in _block1_seq_independent_supported_candidates(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if seq_len % (int(candidate["parallel_seq"]) * int(candidate["tile_m"])) == 0
    }
    preferred_ids.update(
        _block1_best_ids_by_num_aie_columns(
            runtime_pool,
            sort_key=_block1_runtime_sort_key,
        )
    )
    return _block1_select_candidates(
        runtime_pool,
        sort_key=_block1_runtime_sort_key,
        preferred_ids=preferred_ids,
        limit=_BLOCK1_RUNTIME_MAX_CANDIDATES,
    )


def _block1_theoretical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
) -> list[dict[str, int | str]]:
    if seq_len <= 0:
        raise ValueError("Block 1 requires seq_len > 0")
    if hidden_size <= 0:
        raise ValueError("Block 1 requires hidden_size > 0")
    if num_heads <= 0:
        raise ValueError("Block 1 requires num_heads > 0")
    if hidden_size % num_heads != 0:
        raise ValueError("Block 1 requires hidden_size divisible by num_heads")

    head_dim = hidden_size // num_heads
    combined_hidden_size = hidden_size * 3
    r = s = t = 8

    topologies: list[dict[str, int | str]] = []
    for (
        parallel_seq,
        parallel_emb,
        num_aie_columns,
        tile_m,
        tile_k,
        tile_n,
    ) in product(
        _BLOCK1_PARALLEL_SEQ_CHOICES,
        _divisors(hidden_size),
        range(1, 9),
        _divisors(seq_len),
        _divisors(hidden_size),
        _divisors(combined_hidden_size),
    ):
        if tile_m % r != 0:
            continue
        if tile_k % s != 0:
            continue
        if tile_n % t != 0:
            continue
        if seq_len % (parallel_seq * tile_m) != 0:
            continue
        if hidden_size % parallel_emb != 0:
            continue
        if not _block1_compute_tile_working_set_fits(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
        ):
            continue
        if num_aie_columns % parallel_emb != 0:
            continue

        candidate = {
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "num_aie_columns": num_aie_columns,
            "parallel_seq": parallel_seq,
            "parallel_emb": parallel_emb,
        }
        topologies.append(
            {
                **candidate,
                "topology_id": _theoretical_topology_id(candidate),
                "topology_family": "shared_runtime_qkv_proj_theoretical",
            }
        )

    return topologies


def _block1_practical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    max_candidates: int = _BLOCK1_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    if max_candidates <= 0:
        raise ValueError("Block 1 practical exploration requires max_candidates > 0")

    theoretical = _block1_theoretical_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    practical = [
        candidate
        for candidate in theoretical
        if _is_block1_practical_candidate(candidate)
    ]
    preferred_ids = {
        _theoretical_topology_id(candidate)
        for candidate in _block1_seq_independent_supported_candidates(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
        )
    }
    preferred_ids.update(
        _block1_best_ids_by_num_aie_columns(
            practical,
            sort_key=_block1_practical_sort_key,
        )
    )
    preferred_ids.update(
        _theoretical_topology_id(candidate)
        for candidate in _block1_runtime_supported_candidates(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
        )
    )
    selected = _block1_select_candidates(
        practical,
        sort_key=_block1_practical_sort_key,
        preferred_ids=preferred_ids,
        limit=max_candidates,
    )
    return [
        {
            **candidate,
            "topology_family": "shared_runtime_qkv_proj_practical",
        }
        for candidate in selected
    ]


def _divisors(value: int) -> tuple[int, ...]:
    divisors = set()
    for i in range(1, int(value**0.5) + 1):
        if value % i != 0:
            continue
        divisors.add(i)
        divisors.add(value // i)
    return tuple(sorted(divisors))


def _block1_compute_tile_working_set_fits(
    *,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> bool:
    return (
        _block1_compute_tile_working_set_bytes(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
        )
        <= _BLOCK1_AIE_DATA_MEM_SIZE_BYTES
    )


def _block1_compute_tile_working_set_bytes(
    *,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> int:
    bf16_bytes = 2
    payload_bytes = (
        tile_m * tile_k * bf16_bytes
        + tile_k * tile_n * bf16_bytes
        + tile_m * tile_n * bf16_bytes
    )
    return (
        _BLOCK1_GEMM_L1_BUFFER_COPIES * payload_bytes
        + _BLOCK1_GEMM_TILE_FIXED_OVERHEAD_BYTES
    )


def _is_block1_practical_candidate(candidate: dict[str, int | str]) -> bool:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    return (
        tile_m >= _BLOCK1_PRACTICAL_MIN_TILE_M
        and tile_k >= _BLOCK1_PRACTICAL_MIN_TILE_K
        and tile_n >= _BLOCK1_PRACTICAL_MIN_TILE_N
        and num_aie_columns in _BLOCK1_RUNTIME_NUM_AIE_COLUMNS_CHOICES
    )


def _block1_runtime_candidate_allowed(
    candidate: dict[str, int | str], *, hidden_size: int
) -> bool:
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_emb = int(candidate["parallel_emb"])
    return (
        parallel_seq in (1, 2, 4)
        and num_aie_columns in _BLOCK1_RUNTIME_NUM_AIE_COLUMNS_CHOICES
        and num_aie_columns % parallel_emb == 0
        and hidden_size % (tile_n * num_aie_columns) == 0
    )


def _block1_runtime_sort_key(candidate: dict[str, int | str]) -> tuple[int, ...]:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_emb = int(candidate["parallel_emb"])

    lane_parallelism = parallel_seq * parallel_emb
    core_count = parallel_seq * num_aie_columns
    shape_score = _block1_compute_tile_shape_score(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )

    return (
        core_count,
        lane_parallelism,
        shape_score,
    )


def _block1_practical_sort_key(candidate: dict[str, int | str]) -> tuple[int, ...]:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_emb = int(candidate["parallel_emb"])

    lane_parallelism = parallel_seq * parallel_emb
    core_count = parallel_seq * num_aie_columns
    shape_score = _block1_compute_tile_shape_score(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )

    return (
        core_count,
        lane_parallelism,
        shape_score,
    )


def _block1_compute_tile_shape_score(
    *,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> tuple[int, int, int, int]:
    # Block 1 is not internally pipelined, so rank tile shapes by how much
    # compute-tile memory they use after discounting elongated GEMM tiles. Favor
    # square-ish MxK and KxN matrices first, then use the MxN output face and
    # overall working-set fullness as tie-breakers.
    compute_working_set = _block1_compute_tile_working_set_bytes(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )
    left_ratio = 1024 * min(tile_m, tile_k) // max(tile_m, tile_k)
    right_ratio = 1024 * min(tile_k, tile_n) // max(tile_k, tile_n)
    output_ratio = 1024 * min(tile_m, tile_n) // max(tile_m, tile_n)
    balanced_utilization = compute_working_set * left_ratio * right_ratio
    output_face_delta = abs(tile_m - tile_n)
    full_shape_delta = max(tile_m, tile_k, tile_n) - min(tile_m, tile_k, tile_n)
    return (
        balanced_utilization,
        output_ratio,
        compute_working_set,
        -output_face_delta - full_shape_delta,
    )


def _preferred_divisors(
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    return tuple(
        divisor
        for divisor in _divisors(value)
        if minimum <= divisor <= maximum and divisor % 8 == 0
    )


def _block1_select_candidates(
    candidates: list[dict[str, int | str]] | tuple[dict[str, int | str], ...],
    *,
    sort_key,
    preferred_ids: set[str],
    limit: int,
) -> list[dict[str, int | str]]:
    ranked = sorted(candidates, key=sort_key, reverse=True)
    selected: list[dict[str, int | str]] = []
    selected_ids: set[str] = set()

    def add_candidate(candidate: dict[str, int | str]) -> None:
        topology_id = _theoretical_topology_id(
            {
                "tile_m": int(candidate["tile_m"]),
                "tile_k": int(candidate["tile_k"]),
                "tile_n": int(candidate["tile_n"]),
                "num_aie_columns": int(candidate["num_aie_columns"]),
                "parallel_seq": int(candidate["parallel_seq"]),
                "parallel_emb": int(candidate["parallel_emb"]),
            }
        )
        if topology_id in selected_ids:
            return
        selected.append(candidate)
        selected_ids.add(topology_id)

    for candidate in ranked:
        if len(selected) >= limit:
            break
        topology_id = _theoretical_topology_id(
            {
                "tile_m": int(candidate["tile_m"]),
                "tile_k": int(candidate["tile_k"]),
                "tile_n": int(candidate["tile_n"]),
                "num_aie_columns": int(candidate["num_aie_columns"]),
                "parallel_seq": int(candidate["parallel_seq"]),
                "parallel_emb": int(candidate["parallel_emb"]),
            }
        )
        if topology_id not in preferred_ids:
            continue
        add_candidate(candidate)

    for candidate in ranked:
        if len(selected) >= limit:
            break
        add_candidate(candidate)

    return sorted(selected, key=sort_key, reverse=True)


def _block1_best_ids_by_num_aie_columns(
    candidates: list[dict[str, int | str]] | tuple[dict[str, int | str], ...],
    *,
    sort_key,
) -> set[str]:
    best_ids: set[str] = set()
    seen_columns: set[int] = set()
    for candidate in sorted(candidates, key=sort_key, reverse=True):
        num_aie_columns = int(candidate["num_aie_columns"])
        if num_aie_columns in seen_columns:
            continue
        seen_columns.add(num_aie_columns)
        best_ids.add(_theoretical_topology_id(candidate))
    return best_ids


def _theoretical_topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_c{config['num_aie_columns']}"
        f"_ps{config['parallel_seq']}_pe{config['parallel_emb']}"
    )
