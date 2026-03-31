# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import math
import copy
import argparse
from pathlib import Path
import logging
from itertools import product

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
    Buffer,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence, TensorAccessPattern
from aie.helpers.dialects.scf import if_, else_
import aie.dialects.index as index
from aie.dialects.aiex import *

base_dir = Path(__file__).parent

dtype_map = {
    "bf16": bfloat16,
    "f32": np.float32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
    },
}

_BLOCK2_TOPOLOGIES = {
    (1, 64): [
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 64,
            "parallel_heads": 1,
            "o_proj_acc_depth": 1,
        }
    ],
    (12, 64): [
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 1,
            "o_proj_acc_depth": 1,
        },
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 2,
            "o_proj_acc_depth": 1,
        },
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 4,
            "o_proj_acc_depth": 1,
        },
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 6,
            "o_proj_acc_depth": 1,
        },
    ],
    (16, 64): [
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 128,
            "parallel_heads": 1,
            "o_proj_acc_depth": 1,
        },
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 128,
            "parallel_heads": 2,
            "o_proj_acc_depth": 1,
        },
        {
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 128,
            "parallel_heads": 4,
            "o_proj_acc_depth": 1,
        },
    ],
}

_BLOCK2_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK2_RUNTIME_LOWERED_PARALLEL_SEQ_CHOICES = (1, 2, 4, 6, 8)
_BLOCK2_MAX_LOWERED_PARALLEL_HEADS = 6
_BLOCK2_AIE_DATA_MEM_SIZE_BYTES = 65536
_BLOCK2_FIFO_STAGE_COPIES = 2
_BLOCK2_O_PROJ_LOCAL_OUTPUT_COPIES = 3
_BLOCK2_STAGE_FIXED_OVERHEAD_BYTES = 4096
_BLOCK2_PRACTICAL_MIN_Q_SEQ_TILE = 32
_BLOCK2_PRACTICAL_MIN_KV_SEQ_TILE = 64
_BLOCK2_PRACTICAL_MIN_EMB_TILE = 64
_BLOCK2_PRACTICAL_MIN_LANE_PARALLELISM = 2
_BLOCK2_PRACTICAL_MIN_SEQUENCE_CHUNK = 32
_BLOCK2_PRACTICAL_MAX_CANDIDATES = 64
_BLOCK2_MAX_O_PROJ_ACC_DEPTH = 8


def mha_out_proj_topologies(
    *,
    seq_len: int | None = None,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int | str]]:
    try:
        retained = copy.deepcopy(_BLOCK2_TOPOLOGIES[(num_heads, head_dim)])
    except KeyError as exc:
        raise ValueError(
            "Block 2 currently supports only the retained thesis families 1x64, 12x64, and 16x64"
        ) from exc

    topologies = retained
    if seq_len is not None:
        if seq_len <= 0:
            raise ValueError("Block 2 requires seq_len > 0")
        retained_seq_candidates = [
            candidate
            for candidate in retained
            if _block2_runtime_candidate_allowed(
                candidate,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        ]
        promoted_seq_candidates = _block2_promoted_parallel_seq_candidates(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            retained=retained,
        )
        topologies = retained_seq_candidates + promoted_seq_candidates

    return [
        {
            **candidate,
            "topology_id": _mha_out_proj_topology_id(candidate),
            "topology_family": "fused_mha_out_proj",
        }
        for candidate in topologies
    ]


def mha_out_proj_theoretical_topologies(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int | str]]:
    if seq_len <= 0:
        raise ValueError("Block 2 requires seq_len > 0")
    if num_heads <= 0:
        raise ValueError("Block 2 requires num_heads > 0")
    if head_dim <= 0:
        raise ValueError("Block 2 requires head_dim > 0")

    embed_sz = num_heads * head_dim
    r, s, t = microkernel_mac_dim_map["npu2"]["bf16"][True]
    if head_dim % s != 0:
        raise ValueError(f"Block 2 requires head_dim divisible by {s}")

    topologies: list[dict[str, int | str]] = []
    for parallel_seq, parallel_heads, q_seq_tile, kv_seq_tile, emb_tile in product(
        _BLOCK2_PARALLEL_SEQ_CHOICES,
        _divisors(num_heads),
        _divisors(seq_len),
        _divisors(seq_len),
        _divisors(embed_sz),
    ):
        if q_seq_tile % r != 0:
            continue
        if kv_seq_tile % t != 0:
            continue
        if emb_tile % t != 0:
            continue
        if seq_len % (parallel_seq * q_seq_tile) != 0:
            continue
        if seq_len % kv_seq_tile != 0:
            continue
        if parallel_seq * parallel_heads > 8:
            continue
        if parallel_heads > _BLOCK2_MAX_LOWERED_PARALLEL_HEADS:
            continue
        if not _block2_stage_working_sets_fit(
            q_seq_tile=q_seq_tile,
            kv_seq_tile=kv_seq_tile,
            emb_tile=emb_tile,
            head_dim=head_dim,
        ):
            continue

        max_acc_depth = embed_sz // emb_tile
        for o_proj_acc_depth in _divisors(max_acc_depth):
            if o_proj_acc_depth > _BLOCK2_MAX_O_PROJ_ACC_DEPTH:
                continue
            if embed_sz % (emb_tile * o_proj_acc_depth) != 0:
                continue
            candidate = {
                "parallel_seq": parallel_seq,
                "q_seq_tile": q_seq_tile,
                "kv_seq_tile": kv_seq_tile,
                "emb_tile": emb_tile,
                "parallel_heads": parallel_heads,
                "o_proj_acc_depth": o_proj_acc_depth,
            }
            topologies.append(
                {
                    **candidate,
                    "topology_id": _mha_out_proj_topology_id(candidate),
                    "topology_family": "fused_mha_out_proj_theoretical",
                }
            )

    return topologies


def mha_out_proj_practical_topologies(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    max_candidates: int = _BLOCK2_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    if max_candidates <= 0:
        raise ValueError("Block 2 practical exploration requires max_candidates > 0")

    theoretical = mha_out_proj_theoretical_topologies(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    retained_runtime_ids = {
        _mha_out_proj_topology_id(
            {
                "parallel_seq": int(candidate["parallel_seq"]),
                "q_seq_tile": int(candidate["q_seq_tile"]),
                "kv_seq_tile": int(candidate["kv_seq_tile"]),
                "emb_tile": int(candidate["emb_tile"]),
                "parallel_heads": int(candidate["parallel_heads"]),
                "o_proj_acc_depth": int(candidate["o_proj_acc_depth"]),
            }
        )
        for candidate in mha_out_proj_topologies(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    }

    ranked = sorted(
        theoretical,
        key=lambda candidate: _block2_practical_sort_key(candidate, head_dim=head_dim),
        reverse=True,
    )

    selected: list[dict[str, int | str]] = []
    selected_ids: set[str] = set()

    def add_candidate(candidate: dict[str, int | str]) -> None:
        selected.append(
            {
                **candidate,
                "topology_family": "fused_mha_out_proj_practical",
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
        if not _is_block2_practical_candidate(candidate):
            continue
        add_candidate(candidate)
        if len(selected) >= max_candidates:
            break

    return sorted(
        selected,
        key=lambda candidate: _block2_practical_sort_key(candidate, head_dim=head_dim),
        reverse=True,
    )


def mha_out_proj_design(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    topology_id: str | None = None,
) -> dict[str, int | str]:
    topologies = mha_out_proj_topologies(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )

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
            raise ValueError(
                f"Unknown Block 2 topology_id={topology_id!r} for {num_heads}x{head_dim}"
            ) from exc

    parallel_seq = int(config["parallel_seq"])
    q_seq_tile = int(config["q_seq_tile"])
    kv_seq_tile = int(config["kv_seq_tile"])
    emb_tile = int(config["emb_tile"])
    parallel_heads = int(config["parallel_heads"])
    o_proj_acc_depth = int(config["o_proj_acc_depth"])
    embed_sz = num_heads * head_dim

    if seq_len <= 0:
        raise ValueError("Block 2 requires seq_len > 0")
    if parallel_seq not in (1, 2, 4, 6, 8):
        raise ValueError("Block 2 requires parallel_seq in {1, 2, 4, 6, 8}")
    if seq_len % (parallel_seq * q_seq_tile) != 0:
        raise ValueError(
            "Block 2 requires seq_len divisible by parallel_seq * q_seq_tile"
        )
    if seq_len % kv_seq_tile != 0:
        raise ValueError("Block 2 requires seq_len divisible by kv_seq_tile")
    if num_heads % parallel_heads != 0:
        raise ValueError("Block 2 requires num_heads divisible by parallel_heads")
    if parallel_seq * parallel_heads > 8:
        raise ValueError(
            "Block 2 topology requires more parallel lanes than the retained surface supports"
        )
    if embed_sz % (emb_tile * o_proj_acc_depth) != 0:
        raise ValueError(
            "Block 2 requires embedding_dim divisible by emb_tile * o_proj_acc_depth"
        )

    return {
        **config,
        "topology_id": str(config["topology_id"]),
        "topology_family": str(config["topology_family"]),
    }


def _mha_out_proj_topology_id(config: dict[str, int]) -> str:
    return (
        f"q{config['q_seq_tile']}_kv{config['kv_seq_tile']}_e{config['emb_tile']}"
        f"_ps{config['parallel_seq']}_ph{config['parallel_heads']}"
        f"_acc{config['o_proj_acc_depth']}"
    )


def _divisors(value: int) -> tuple[int, ...]:
    divisors = set()
    for i in range(1, int(math.isqrt(value)) + 1):
        if value % i != 0:
            continue
        divisors.add(i)
        divisors.add(value // i)
    return tuple(sorted(divisors))


def _block2_stage_working_sets_fit(
    *,
    q_seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    head_dim: int,
) -> bool:
    return (
        _block2_stage_working_set_bytes(
            q_seq_tile=q_seq_tile,
            kv_seq_tile=kv_seq_tile,
            emb_tile=emb_tile,
            head_dim=head_dim,
        )
        <= _BLOCK2_AIE_DATA_MEM_SIZE_BYTES
    )


def _block2_stage_working_set_bytes(
    *,
    q_seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    head_dim: int,
) -> int:
    bf16_bytes = 2

    q_bytes = q_seq_tile * head_dim * bf16_bytes
    k_bytes = head_dim * kv_seq_tile * bf16_bytes
    qk_bytes = q_seq_tile * kv_seq_tile * bf16_bytes
    v_bytes = kv_seq_tile * head_dim * bf16_bytes
    scale_bytes = 4 * q_seq_tile * bf16_bytes
    wo_bytes = head_dim * emb_tile * bf16_bytes
    o_bytes = q_seq_tile * emb_tile * bf16_bytes

    qk_stage_bytes = (
        (_BLOCK2_FIFO_STAGE_COPIES * q_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * k_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * qk_bytes)
        + _BLOCK2_STAGE_FIXED_OVERHEAD_BYTES
    )
    softmax_stage_bytes = (
        (2 * _BLOCK2_FIFO_STAGE_COPIES * qk_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * scale_bytes)
        + _BLOCK2_STAGE_FIXED_OVERHEAD_BYTES
    )
    pv_stage_bytes = (
        (_BLOCK2_FIFO_STAGE_COPIES * qk_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * v_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * q_bytes)
        + (_BLOCK2_FIFO_STAGE_COPIES * scale_bytes)
        + _BLOCK2_STAGE_FIXED_OVERHEAD_BYTES
    )
    o_proj_stage_bytes = (
        q_bytes
        + wo_bytes
        + (_BLOCK2_O_PROJ_LOCAL_OUTPUT_COPIES * o_bytes)
        + _BLOCK2_STAGE_FIXED_OVERHEAD_BYTES
    )

    stage_working_sets = (
        qk_stage_bytes,
        softmax_stage_bytes,
        pv_stage_bytes,
        o_proj_stage_bytes,
    )
    return max(stage_working_sets)


def _is_block2_practical_candidate(candidate: dict[str, int | str]) -> bool:
    q_seq_tile = int(candidate["q_seq_tile"])
    kv_seq_tile = int(candidate["kv_seq_tile"])
    emb_tile = int(candidate["emb_tile"])

    lane_parallelism = _block2_effective_lowered_parallelism(candidate)
    sequence_chunk = _block2_effective_sequence_chunk(candidate)
    return (
        q_seq_tile >= _BLOCK2_PRACTICAL_MIN_Q_SEQ_TILE
        and kv_seq_tile >= _BLOCK2_PRACTICAL_MIN_KV_SEQ_TILE
        and emb_tile >= _BLOCK2_PRACTICAL_MIN_EMB_TILE
        and lane_parallelism >= _BLOCK2_PRACTICAL_MIN_LANE_PARALLELISM
        and sequence_chunk >= _BLOCK2_PRACTICAL_MIN_SEQUENCE_CHUNK
    )


def _block2_practical_sort_key(
    candidate: dict[str, int | str],
    *,
    head_dim: int = 64,
) -> tuple[int, ...]:
    parallel_seq = int(candidate["parallel_seq"])
    q_seq_tile = int(candidate["q_seq_tile"])
    kv_seq_tile = int(candidate["kv_seq_tile"])
    emb_tile = int(candidate["emb_tile"])
    o_proj_acc_depth = int(candidate["o_proj_acc_depth"])

    lane_parallelism = _block2_effective_lowered_parallelism(candidate)
    sequence_chunk = _block2_effective_sequence_chunk(candidate)
    output_chunk = emb_tile * o_proj_acc_depth
    working_set = _block2_stage_working_set_bytes(
        q_seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        head_dim=head_dim,
    )

    return (
        lane_parallelism,
        o_proj_acc_depth,
        kv_seq_tile,
        sequence_chunk,
        emb_tile,
        output_chunk,
        -parallel_seq,
        working_set,
        q_seq_tile,
    )


def _block2_effective_lowered_parallelism(candidate: dict[str, int | str]) -> int:
    parallel_seq = int(candidate["parallel_seq"])
    parallel_heads = int(candidate["parallel_heads"])
    if parallel_seq in _BLOCK2_RUNTIME_LOWERED_PARALLEL_SEQ_CHOICES:
        return parallel_seq * parallel_heads
    return parallel_heads


def _block2_effective_sequence_chunk(candidate: dict[str, int | str]) -> int:
    parallel_seq = int(candidate["parallel_seq"])
    q_seq_tile = int(candidate["q_seq_tile"])
    if parallel_seq in _BLOCK2_RUNTIME_LOWERED_PARALLEL_SEQ_CHOICES:
        return parallel_seq * q_seq_tile
    return q_seq_tile


def _block2_runtime_candidate_allowed(
    candidate: dict[str, int | str],
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    parallel_seq = int(candidate["parallel_seq"])
    q_seq_tile = int(candidate["q_seq_tile"])
    kv_seq_tile = int(candidate["kv_seq_tile"])
    emb_tile = int(candidate["emb_tile"])
    parallel_heads = int(candidate["parallel_heads"])
    o_proj_acc_depth = int(candidate["o_proj_acc_depth"])
    embed_sz = num_heads * head_dim

    if head_dim != 64:
        return False
    if parallel_seq not in _BLOCK2_RUNTIME_LOWERED_PARALLEL_SEQ_CHOICES:
        return False
    if seq_len % (parallel_seq * q_seq_tile) != 0:
        return False
    if seq_len % kv_seq_tile != 0:
        return False
    if num_heads % parallel_heads != 0:
        return False
    if parallel_seq * parallel_heads > 8:
        return False
    if embed_sz % (emb_tile * o_proj_acc_depth) != 0:
        return False
    if parallel_seq > 1:
        if (
            num_heads == 1
            and q_seq_tile == 32
            and kv_seq_tile == 64
            and emb_tile == 64
            and parallel_heads == 1
            and o_proj_acc_depth == 1
        ):
            return True
        return (
            num_heads in (12, 16)
            and q_seq_tile == 32
            and kv_seq_tile == 64
            and o_proj_acc_depth == 1
        )
    return True


def _block2_promoted_parallel_seq_candidates(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    retained: list[dict[str, int]],
) -> list[dict[str, int]]:
    base_seq_candidates = [
        copy.deepcopy(candidate)
        for candidate in retained
        if int(candidate["parallel_seq"]) == 1
        and int(candidate["o_proj_acc_depth"]) == 1
        and int(candidate["q_seq_tile"]) == 32
        and int(candidate["kv_seq_tile"]) == 64
    ]
    if not base_seq_candidates:
        return []

    promoted: list[dict[str, int]] = []
    for base_seq_candidate in base_seq_candidates:
        for parallel_seq in (2, 4, 6, 8):
            candidate = copy.deepcopy(base_seq_candidate)
            candidate["parallel_seq"] = parallel_seq
            if _block2_runtime_candidate_allowed(
                candidate,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            ):
                promoted.append(candidate)
    return promoted
