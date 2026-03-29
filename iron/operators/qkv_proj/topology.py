# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import product

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
    "parallel_heads": (1,),
    "parallel_head_dim": (1,),
}

_BLOCK1_PROMOTED_RUNTIME_TOPOLOGIES = {
    (12, 64): (
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 24,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_heads": 1,
            "parallel_head_dim": 1,
        },
    ),
    (16, 64): (
        {
            "tile_m": 32,
            "tile_k": 256,
            "tile_n": 16,
            "num_aie_columns": 8,
            "parallel_seq": 1,
            "parallel_heads": 1,
            "parallel_head_dim": 1,
        },
    ),
}

_BLOCK1_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK1_AIE_DATA_MEM_SIZE_BYTES = 65536
_BLOCK1_GEMM_L1_BUFFER_COPIES = 2
_BLOCK1_GEMM_TILE_FIXED_OVERHEAD_BYTES = 3392
_BLOCK1_PRACTICAL_MIN_TILE_M = 32
_BLOCK1_PRACTICAL_MIN_TILE_K = 64
_BLOCK1_PRACTICAL_MIN_TILE_N = 16
_BLOCK1_PRACTICAL_MIN_AIE_COLUMNS = 4
_BLOCK1_PRACTICAL_MAX_CANDIDATES = 64


def qkv_proj_topologies(
    *,
    hidden_size: int,
    num_heads: int,
    seq_len: int | None = None,
) -> list[dict[str, int | str]]:
    if hidden_size % num_heads != 0:
        raise ValueError("Block 1 requires hidden_size divisible by num_heads")
    head_dim = hidden_size // num_heads
    if (num_heads, head_dim) not in _BLOCK1_RETAINED_FAMILIES:
        raise ValueError(
            "Block 1 currently supports only the retained thesis families 12x64 and 16x64"
        )
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
    parallel_heads = int(config["parallel_heads"])
    parallel_head_dim = int(config["parallel_head_dim"])
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
    if num_heads % parallel_heads != 0:
        raise ValueError("Block 1 requires num_heads divisible by parallel_heads")
    if head_dim % parallel_head_dim != 0:
        raise ValueError("Block 1 requires head_dim divisible by parallel_head_dim")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 1 requires 1 <= num_aie_columns <= 8")

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
        parallel_heads,
        parallel_head_dim,
    ) in product(
        _BLOCK1_RETAINED_AXES["tile_m"],
        _BLOCK1_RETAINED_AXES["tile_k"],
        _BLOCK1_RETAINED_AXES["tile_n"],
        _BLOCK1_RETAINED_AXES["num_aie_columns"],
        _BLOCK1_RETAINED_AXES["parallel_seq"],
        _BLOCK1_RETAINED_AXES["parallel_heads"],
        _BLOCK1_RETAINED_AXES["parallel_head_dim"],
    ):
        if hidden_size % tile_k != 0:
            continue
        if combined_hidden_size % tile_n != 0:
            continue
        if num_heads % parallel_heads != 0:
            continue
        if head_dim % parallel_head_dim != 0:
            continue
        topologies.append(
            {
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "num_aie_columns": num_aie_columns,
                "parallel_seq": parallel_seq,
                "parallel_heads": parallel_heads,
                "parallel_head_dim": parallel_head_dim,
            }
        )
    return topologies


def _block1_seq_independent_supported_candidates(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    topologies = _enumerate_block1_topologies(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    topologies.extend(
        _BLOCK1_PROMOTED_RUNTIME_TOPOLOGIES.get((num_heads, head_dim), ())
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


def _block1_runtime_supported_candidates(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    practical = [
        {
            "tile_m": int(candidate["tile_m"]),
            "tile_k": int(candidate["tile_k"]),
            "tile_n": int(candidate["tile_n"]),
            "num_aie_columns": int(candidate["num_aie_columns"]),
            "parallel_seq": int(candidate["parallel_seq"]),
            "parallel_heads": int(candidate["parallel_heads"]),
            "parallel_head_dim": int(candidate["parallel_head_dim"]),
        }
        for candidate in _block1_practical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
    ]
    preferred_ids = {
        _theoretical_topology_id(
            {
                **candidate,
                "num_aie_columns": 8,
            }
        )
        for candidate in _block1_seq_independent_supported_candidates(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    }
    ordered: list[dict[str, int]] = []
    seen_ids: set[str] = set()
    for candidate in practical:
        topology_id = _theoretical_topology_id(candidate)
        if topology_id not in preferred_ids:
            continue
        ordered.append(candidate)
        seen_ids.add(topology_id)
    for candidate in practical:
        topology_id = _theoretical_topology_id(candidate)
        if topology_id in seen_ids:
            continue
        ordered.append(candidate)
        seen_ids.add(topology_id)
    return ordered


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
        parallel_heads,
        parallel_head_dim,
        num_aie_columns,
        tile_m,
        tile_k,
        tile_n,
    ) in product(
        _BLOCK1_PARALLEL_SEQ_CHOICES,
        _divisors(num_heads),
        _divisors(head_dim),
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
        if num_heads % parallel_heads != 0:
            continue
        if head_dim % parallel_head_dim != 0:
            continue
        if parallel_seq * parallel_heads * parallel_head_dim > 8:
            continue
        if not _block1_compute_tile_working_set_fits(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
        ):
            continue

        candidate = {
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "num_aie_columns": num_aie_columns,
            "parallel_seq": parallel_seq,
            "parallel_heads": parallel_heads,
            "parallel_head_dim": parallel_head_dim,
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
    retained_runtime_ids = {
        _theoretical_topology_id(
            {
                "tile_m": int(candidate["tile_m"]),
                "tile_k": int(candidate["tile_k"]),
                "tile_n": int(candidate["tile_n"]),
                "num_aie_columns": int(candidate["num_aie_columns"]),
                "parallel_seq": int(candidate["parallel_seq"]),
                "parallel_heads": int(candidate["parallel_heads"]),
                "parallel_head_dim": int(candidate["parallel_head_dim"]),
            }
        )
        for candidate in qkv_proj_topologies(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
    }

    ranked = sorted(theoretical, key=_block1_practical_sort_key, reverse=True)

    selected: list[dict[str, int | str]] = []
    selected_ids: set[str] = set()

    def add_candidate(candidate: dict[str, int | str]) -> None:
        selected.append(
            {
                **candidate,
                "topology_family": "shared_runtime_qkv_proj_practical",
            }
        )
        selected_ids.add(str(candidate["topology_id"]))

    for candidate in ranked:
        topology_id = str(candidate["topology_id"])
        if topology_id not in retained_runtime_ids:
            continue
        add_candidate(candidate)

    for candidate in ranked:
        topology_id = str(candidate["topology_id"])
        if topology_id in selected_ids:
            continue
        if not _is_block1_practical_candidate(candidate):
            continue
        add_candidate(candidate)
        if len(selected) >= max_candidates:
            break

    return sorted(selected, key=_block1_practical_sort_key, reverse=True)


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
        and num_aie_columns >= _BLOCK1_PRACTICAL_MIN_AIE_COLUMNS
    )


def _block1_practical_sort_key(candidate: dict[str, int | str]) -> tuple[int, ...]:
    tile_m = int(candidate["tile_m"])
    tile_k = int(candidate["tile_k"])
    tile_n = int(candidate["tile_n"])
    num_aie_columns = int(candidate["num_aie_columns"])
    parallel_seq = int(candidate["parallel_seq"])
    parallel_heads = int(candidate["parallel_heads"])
    parallel_head_dim = int(candidate["parallel_head_dim"])

    lane_parallelism = parallel_seq * parallel_heads * parallel_head_dim
    output_chunk = tile_n * num_aie_columns
    sequence_chunk = parallel_seq * tile_m
    compute_working_set = _block1_compute_tile_working_set_bytes(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
    )

    return (
        lane_parallelism,
        num_aie_columns,
        tile_k,
        output_chunk,
        sequence_chunk,
        compute_working_set,
        tile_n,
        tile_m,
    )


def _theoretical_topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_c{config['num_aie_columns']}"
        f"_ps{config['parallel_seq']}_ph{config['parallel_heads']}"
        f"_pd{config['parallel_head_dim']}"
    )
