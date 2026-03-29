# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from itertools import product
import json

from iron.operators.gemm.design_batched import my_matmul as _batched_gemm_design

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

_BLOCK1_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK1_AIE_DATA_MEM_SIZE_BYTES = 65536


def qkv_proj_topologies(
    *,
    hidden_size: int,
    num_heads: int,
) -> list[dict[str, int | str]]:
    if hidden_size % num_heads != 0:
        raise ValueError("Block 1 requires hidden_size divisible by num_heads")
    head_dim = hidden_size // num_heads
    if (num_heads, head_dim) not in _BLOCK1_RETAINED_FAMILIES:
        raise ValueError(
            "Block 1 currently supports only the retained thesis families 12x64 and 16x64"
        )
    topologies = _enumerate_block1_topologies(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return [
        {
            **candidate,
            "topology_id": _topology_id(candidate),
            "topology_family": "shared_runtime_qkv_proj",
        }
        for candidate in topologies
    ]


def qkv_proj_theoretical_topologies(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Block 1 QKV Projector Design",
        description="Resolve retained thesis topology parameters for Block 1",
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--topology-id", type=str, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            qkv_proj_design(
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                topology_id=args.topology_id,
            ),
            sort_keys=True,
        )
    )


def qkv_proj_design(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    head_dim = hidden_size // num_heads
    topologies = qkv_proj_topologies(hidden_size=hidden_size, num_heads=num_heads)

    if topology_id is None:
        config = topologies[0]
    else:
        try:
            config = next(
                candidate
                for candidate in topologies
                if str(candidate["topology_id"]) == topology_id
            )
        except StopIteration as exc:
            head_dim = hidden_size // num_heads
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
    if hidden_size % tile_n != 0:
        raise ValueError("Block 1 requires hidden_size divisible by tile_n")
    if num_heads % parallel_heads != 0:
        raise ValueError("Block 1 requires num_heads divisible by parallel_heads")
    if head_dim % parallel_head_dim != 0:
        raise ValueError("Block 1 requires head_dim divisible by parallel_head_dim")
    if not 1 <= num_aie_columns <= 8:
        raise ValueError("Block 1 requires 1 <= num_aie_columns <= 8")

    return {
        **config,
        "topology_id": str(config["topology_id"]),
        "topology_family": str(config["topology_family"]),
    }


def fused_qkv_proj(
    *,
    dev: str,
    seq_len: int,
    hidden_size: int,
    combined_hidden_size: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    num_aie_columns: int,
    dtype_in_str: str,
    dtype_out_str: str,
    use_scalar: bool,
    emulate_bf16_mmul_with_bfp16: bool,
    prio_accuracy: bool,
    trace_size: int,
    archive: str | None = None,
):
    return _batched_gemm_design(
        dev,
        seq_len,
        hidden_size,
        combined_hidden_size,
        tile_m,
        tile_k,
        tile_n,
        num_aie_columns,
        dtype_in_str,
        dtype_out_str,
        0,
        0,
        use_scalar,
        emulate_bf16_mmul_with_bfp16,
        prio_accuracy,
        trace_size,
        archive,
        False,
        (1, 0),
        (1, 0),
        (1, 0),
    )


def _enumerate_block1_topologies(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int]]:
    topologies: list[dict[str, int]] = []
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
        if hidden_size % tile_n != 0:
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
    bf16_bytes = 2
    a_l1_bytes = tile_m * tile_k * bf16_bytes
    b_l1_bytes = tile_k * tile_n * bf16_bytes
    c_l1_bytes = tile_m * tile_n * bf16_bytes
    return a_l1_bytes + b_l1_bytes + c_l1_bytes <= _BLOCK1_AIE_DATA_MEM_SIZE_BYTES


def _topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_ps{config['parallel_seq']}_ph{config['parallel_heads']}"
        f"_pd{config['parallel_head_dim']}"
    )


def _theoretical_topology_id(config: dict[str, int]) -> str:
    return (
        f"m{config['tile_m']}_k{config['tile_k']}_n{config['tile_n']}"
        f"_c{config['num_aie_columns']}"
        f"_ps{config['parallel_seq']}_ph{config['parallel_heads']}"
        f"_pd{config['parallel_head_dim']}"
    )


if __name__ == "__main__":
    main()
