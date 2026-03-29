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


def mha_out_proj_topologies(
    *,
    num_heads: int,
    head_dim: int,
) -> list[dict[str, int | str]]:
    try:
        topologies = _BLOCK2_TOPOLOGIES[(num_heads, head_dim)]
    except KeyError as exc:
        raise ValueError(
            "Block 2 currently supports only the retained thesis families 1x64, 12x64, and 16x64"
        ) from exc
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
    topologies = mha_out_proj_topologies(num_heads=num_heads, head_dim=head_dim)

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

    runtime_lowering_bonus = int(parallel_seq == 1)
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
        runtime_lowering_bonus,
        lane_parallelism,
        kv_seq_tile,
        sequence_chunk,
        emb_tile,
        output_chunk,
        -parallel_seq,
        -o_proj_acc_depth,
        working_set,
        q_seq_tile,
    )


def _block2_effective_lowered_parallelism(candidate: dict[str, int | str]) -> int:
    return int(candidate["parallel_heads"])


def _block2_effective_sequence_chunk(candidate: dict[str, int | str]) -> int:
    return int(candidate["q_seq_tile"])


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("--seq-len", type=int, default=256)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--q-seq-tile", type=int, default=64)
    argparser.add_argument("--kv-seq-tile", type=int, default=64)
    argparser.add_argument("--emb-tile", type=int, default=96)
    argparser.add_argument("--o-proj-acc-depth", type=int, default=1)
    argparser.add_argument("--parallel-heads", type=int, default=1)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=True)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--kernel-archive", type=str, default="mha_kernels.a")
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default=base_dir / "build" / f"my_mha.mlir",
        help="Output file path for the generated MLIR module",
    )

    args = argparser.parse_args()

    maybe_module = fused_mha(
        heads=args.heads,
        seq_len=args.seq_len,
        d=args.d,
        q_seq_tile=args.q_seq_tile,
        kv_seq_tile=args.kv_seq_tile,
        emb_tile=args.emb_tile,
        o_proj_acc_depth=args.o_proj_acc_depth,
        parallel_heads=args.parallel_heads,
        emulate_bf16_mmul_with_bfp16=args.emulate_bf16_mmul_with_bfp16,
        kernel_archive=args.kernel_archive,
        trace_size=args.trace_size,
    )

    output_file_path = Path(args.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(maybe_module))

    logging.info(f"MLIR module written to {output_file_path}")


# TODO: Add back parallel sequence blocks?


def fused_mha(
    heads: int,
    seq_len: int,
    d: int,
    q_seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    o_proj_acc_depth: int,
    parallel_heads: int,
    emulate_bf16_mmul_with_bfp16: bool,
    kernel_archive: str,
    trace_size: int = 0,
):
    embed_sz = heads * d

    of_depth = 2
    o_proj_weight_consumer_depth = 1
    o_proj_partial_depth = 1
    enable_tracing = True if trace_size > 0 else False
    dtype_str = "bf16"
    dev = "npu2"

    # NOTE: We don't split up the parallel_heads into two like how it's done in MHA operator
    # with parallel sequence blocks. This is because this design will be used for the pipelined
    # encoder, which will likely not require more than 6 parallel heads in order to have space
    # for the the two Add & Norm blocks and FFN block.

    num_q_seq_blocks = seq_len // q_seq_tile
    num_kv_seq_blocks = seq_len // kv_seq_tile
    num_qkv_head_block_per_parallel_head = heads // parallel_heads
    assert embed_sz % (emb_tile * o_proj_acc_depth) == 0, (
        "embed_sz must be divisible by emb_tile * o_proj_acc_depth "
        f"({embed_sz} % ({emb_tile} * {o_proj_acc_depth}) != 0)"
    )
    num_o_col_groups = embed_sz // (emb_tile * o_proj_acc_depth)

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_str]
    r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]

    logging.info(f"Device: {dev}")
    logging.info(f"Number of heads: {heads}")
    logging.info(
        f"MHA Dimensions: seq_len={seq_len}, d={d}, q_seq_tile={q_seq_tile}, kv_seq_tile={kv_seq_tile}, emb_tile={emb_tile}, o_proj_acc_depth={o_proj_acc_depth}, parallel_heads={parallel_heads}"
    )
    logging.info(
        f"num_q_seq_blocks: {num_q_seq_blocks}, num_kv_seq_blocks: {num_kv_seq_blocks}, num_qkv_head_block_per_parallel_head: {num_qkv_head_block_per_parallel_head}, num_o_col_groups: {num_o_col_groups}"
    )
    logging.info(f"Data type: {dtype_str}")
    logging.info(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
    logging.info(f"Enable tracing: {enable_tracing}")

    assert heads > 0, "Number of heads must be greater than 0"
    assert (
        heads % parallel_heads == 0
    ), "Number of heads must be divisible by parallel_heads"

    assert (
        q_seq_tile % r == 0
    ), f"q_seq_tile must be divisible by r ({q_seq_tile} % {r} != 0)"
    assert (
        kv_seq_tile % t == 0
    ), f"kv_seq_tile must be divisible by t ({kv_seq_tile} % {t} != 0)"
    assert d % s == 0, f"d must be divisible by s ({d} % {s} != 0)"

    assert seq_len % q_seq_tile == 0, "seq_len must be divisible by q_seq_tile"

    dtype = dtype_map[dtype_str]

    inv_scale = (1 / np.sqrt(d)) * 1.4453125

    # Tensors living in DRAM
    W_O_ty = np.ndarray[
        (embed_sz, embed_sz),
        np.dtype[dtype],
    ]
    Q_ty = np.ndarray[
        (seq_len, embed_sz),
        np.dtype[dtype],
    ]
    K_ty = np.ndarray[
        (seq_len, embed_sz),
        np.dtype[dtype],
    ]
    V_ty = np.ndarray[
        (seq_len, embed_sz),
        np.dtype[dtype],
    ]
    O_ty = np.ndarray[
        (seq_len, embed_sz),
        np.dtype[dtype],
    ]

    # Tensors living on the AIE-array
    q_ty = np.ndarray[(q_seq_tile, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, kv_seq_tile), np.dtype[dtype]]
    qk_ty = np.ndarray[(q_seq_tile, kv_seq_tile), np.dtype[dtype]]
    v_ty = np.ndarray[(kv_seq_tile, d), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * q_seq_tile,), np.dtype[dtype]]
    wo_ty = np.ndarray[(d, emb_tile), np.dtype[dtype]]
    o_ty = np.ndarray[(q_seq_tile, emb_tile), np.dtype[dtype]]

    # AIE kernel declarations
    bin_name = kernel_archive

    zero_kernel = Kernel(f"zero_{dtype_str}", bin_name, [qk_ty])
    zero_kernel_q = Kernel(f"zero_{dtype_str}_rowmaj", bin_name, [q_ty])

    memcopy_kernel_scale = Kernel(f"passThroughLine", bin_name, [s_ty, s_ty, np.int32])

    scale_buffer_init_kernel = Kernel("init_scale_buffer", bin_name, [s_ty, np.int32])

    partial_softmax_kernel = Kernel(
        "partial_softmax",
        bin_name,
        [
            qk_ty,
            qk_ty,
            s_ty,
            np.ndarray[(2,), np.dtype[np.int32]],
            dtype,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    matmul_QK = Kernel(
        f"matmul_bf16_bf16_wrapper",
        bin_name,
        [q_ty, k_ty, qk_ty, np.ndarray[(2,), np.dtype[np.int32]]],
    )

    matmul_PV = Kernel(
        "matmul_PV",
        bin_name,
        [
            qk_ty,
            v_ty,
            q_ty,
            s_ty,
            np.int32,
            np.int32,
            np.ndarray[(2,), np.dtype[np.int32]],
        ],
    )

    rescale_O = Kernel(
        "rescale_O",
        bin_name,
        [q_ty, s_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
    )

    zero_kernel_o_proj = Kernel(
        f"zero_{dtype_str}_o_proj",
        bin_name,
        [o_ty],
    )
    matmul_kernel_o_proj = Kernel(
        f"matmul_with_acc_bf16_bf16_o_proj",
        bin_name,
        [q_ty, wo_ty, o_ty, o_ty],
    )
    mem_copy_o_proj = Kernel(
        "passThroughLine_o_proj",
        bin_name,
        [o_ty, o_ty, np.int32],
    )
    eltwise_add_vector = Kernel(
        "eltwise_add_bf16_vector_o_proj",
        bin_name,
        [o_ty, o_ty, o_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    q_dims = [(q_seq_tile // r, r * d), (d // s, s), (r, d), (s, 1)]

    inQ = ObjectFifo(
        np.ndarray[(q_seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inQ",
        depth=of_depth,
    )
    memQ = inQ.cons().split(
        offsets=[q_seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[q_ty] * parallel_heads,
        names=[f"memQ{i}" for i in range(parallel_heads)],
        dims_to_stream=[q_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=0, row=1),
    )  # Split between N parallel blocks of sequences

    # VJUNG: The SequentialPlacer will place all of these on the same MemTile if Placement is specified. We would need a list of placement in case of one-many or many-one.
    # I think the Sequential Placer will fail if we do a split/join with more than 6 I/Os cuz it tries to place them all on the same tile.

    # K is stored in column-major order
    k_dims = [(kv_seq_tile // t, t * d), (d // s, s), (t, d), (s, 1)]
    inK = ObjectFifo(
        np.ndarray[(kv_seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inK",
        depth=of_depth,
    )
    memK = inK.cons().split(
        offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[k_ty] * parallel_heads,
        names=[f"memK{i}" for i in range(parallel_heads)],
        dims_to_stream=[k_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=1, row=1),
    )  # Split between N parallel blocks of heads

    v_dims = [
        (kv_seq_tile // s, s * d),
        (d // t, t),
        (s, d),
        (t, 1),
    ]

    inV = ObjectFifo(
        np.ndarray[(kv_seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inV",
        depth=of_depth,
    )
    memV = inV.cons().split(
        offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[v_ty] * parallel_heads,
        names=[f"memV{i}" for i in range(parallel_heads)],
        dims_to_stream=[v_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=2, row=1),
    )  # Split between N parallel blocks of heads

    memA = []
    # Data layout transformation to execute softmax without microtiles
    # First send microkernel tiles across the sequence dimension of the output,
    # then place those microkernel tiles in the correct locations with another DMA
    a_dims_out = [
        (kv_seq_tile // s, r * s),
        (q_seq_tile // r, kv_seq_tile * r),
        (r * s, 1),
    ]
    a_dims_in = [(kv_seq_tile // s, s), (q_seq_tile, kv_seq_tile), (s, 1)]
    for i in range(parallel_heads):
        memA.append(
            ObjectFifo(
                qk_ty,
                depth=of_depth,
                name=f"memA{i}",
                dims_to_stream=a_dims_out,
                dims_from_stream_per_cons=a_dims_in,
            )
        )  # Local to 1 parallel block of heads

    memP = []
    # Data layout transformation to turn tile into microtiles again for mmul
    # First send microkernel tiles across the sequence dimension of the output,
    # then place those microkernel tiles in the correct locations with another DMA
    p_dims_out = [(kv_seq_tile // s, s), (q_seq_tile, kv_seq_tile), (s, 1)]
    p_dims_in = [
        (kv_seq_tile // s, r * s),
        (q_seq_tile // r, kv_seq_tile * r),
        (r * s, 1),
    ]
    for i in range(parallel_heads):
        memP.append(
            ObjectFifo(
                qk_ty,
                depth=of_depth,
                name=f"memP{i}",
                dims_to_stream=p_dims_out,
                dims_from_stream_per_cons=p_dims_in,
            )
        )  # Local to 1 parallel block of heads

    # Scale buffer for partial softmax
    scaleOF = []
    for i in range(parallel_heads):
        scaleOF.append(
            ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}")
        )  # Local to 1 parallel block of sequences

    # Output projection weights
    ow_dims = [
        (d // s, s * emb_tile),
        (emb_tile // t, t),
        (s, emb_tile),
        (t, 1),
    ]

    inOW = ObjectFifo(
        np.ndarray[(d * parallel_heads, emb_tile), np.dtype[dtype]],
        name="inOW",
        depth=of_depth,
    )
    memOW = inOW.cons().split(
        offsets=[d * emb_tile * i for i in range(parallel_heads)],
        obj_types=[wo_ty] * parallel_heads,
        names=[f"memOW{i}" for i in range(parallel_heads)],
        dims_to_stream=[ow_dims] * parallel_heads,
        depths=[o_proj_weight_consumer_depth] * parallel_heads,
        placement=Tile(col=3, row=1),
    )  # Split between N parallel blocks of heads

    # Partial out proj tiles to store accumulations in MTs
    outOProj = []
    outOProjAccumIn = []
    outOProjAccumOut = []
    for i in range(parallel_heads):
        outOProj.append(
            ObjectFifo(q_ty, depth=o_proj_partial_depth, name=f"outOProj{i}")
        )  # Local to 1 parallel block of heads
        outOProjAccumOut.append(ObjectFifo(o_ty, depth=1, name=f"outOProjAccumOut{i}"))
        outOProjAccumIn.append(
            outOProjAccumOut[i]
            .cons(depth=o_proj_acc_depth)
            .forward(
                name=f"outOProjAccumIn{i}",
                depth=o_proj_acc_depth,
                placement=Tile(col=6 + (i % 2), row=1),
            )
        )  # Local to 1 parallel block of heads

    outOPart = []
    for i in range(parallel_heads - 1):
        outOPart.append(
            ObjectFifo(o_ty, depth=o_proj_partial_depth, name=f"outOPart{i}")
        )  # Local to 1 parallel block of heads

    o_dims = [(q_seq_tile // r, r * emb_tile), (r, t), (emb_tile // t, r * t), (t, 1)]
    memO = ObjectFifo(
        o_ty,
        name="memO",
        dims_to_stream=o_dims,
    )
    outO = memO.prod().join(  # TODO: Check if this becomes a forward operation--or might give an error
        offsets=[q_seq_tile * emb_tile],
        obj_types=[o_ty],
        names=[f"outO{i}"],
        depths=[of_depth],
        placement=Tile(col=7, row=1),
    )  # Join onto the output OF

    def batched_matmul_qk(
        of_q,
        of_k,
        of_a_out,
        zero,
        matmul_QK,
        q_block_bias,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            # NOTE: Second element in idx_buffer used to be set to q_block_bias, which
            # seems to be used for causal masking and for when
            # attention is parallelized across the sequence dimension. For this
            # design, it shouldn't be getting used since we parallelize across heads.
            # Since it affects the computations, we set the value to 0.
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                elem_in_q = of_q.acquire(1)

                for _ in range_(num_kv_seq_blocks):

                    elem_in_k = of_k.acquire(1)
                    elem_a_out = of_a_out.acquire(1)

                    zero(elem_a_out)
                    matmul_QK(elem_in_q, elem_in_k, elem_a_out, idx_buffer)

                    of_k.release(1)
                    of_a_out.release(1)

                    idx_buffer[0] += 1
                idx_buffer[0] = 0

                of_q.release(1)

    def softmax(
        of_in_a,
        of_out_p,
        of_out_scale,
        partial_softmax,
        init_scale_buffer,
        memcopy_kernel_scale,
        q_block_bias,
        idx_buffer,
        scale_buffer,
    ):

        # VJUNG: The index buffer count how many Q and KV block this worker has processed
        # From this info we can infer the position in A and P

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                init_scale_buffer(scale_buffer, q_seq_tile)

                for _ in range_(num_kv_seq_blocks):

                    elt_of_out_p = of_out_p.acquire(1)
                    elt_of_in_a = of_in_a.acquire(1)
                    elt_of_out_scale = of_out_scale.acquire(1)

                    partial_softmax(
                        elt_of_in_a,
                        elt_of_out_p,
                        scale_buffer,
                        idx_buffer,
                        inv_scale,
                        q_seq_tile,
                        kv_seq_tile,
                        seq_len,
                        seq_len,
                    )
                    memcopy_kernel_scale(scale_buffer, elt_of_out_scale, 4 * q_seq_tile)

                    of_in_a.release(1)
                    of_out_p.release(1)
                    of_out_scale.release(1)

                    idx_buffer[0] += 1
                idx_buffer[0] = 0

    def batched_matmul_pv(
        of_p,
        of_v,
        of_scale,
        of_o_out,
        zero,
        matmul_PV,
        rescale_O,
        q_block_bias,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                elem_o_out = of_o_out.acquire(1)

                zero(elem_o_out)

                ### First iteration, don't rescale O_{i-1}
                elem_in_p = of_p.acquire(1)
                elem_in_v = of_v.acquire(1)
                elt_of_out_scale = of_scale.acquire(1)

                matmul_PV(
                    elem_in_p,
                    elem_in_v,
                    elem_o_out,
                    elt_of_out_scale,
                    q_seq_tile,
                    0,
                    idx_buffer,
                )

                of_p.release(1)
                of_v.release(1)
                of_scale.release(1)

                idx_buffer[0] += 1
                ###

                if num_kv_seq_blocks > 2:
                    for _ in range_(num_kv_seq_blocks - 2):
                        elem_in_p = of_p.acquire(1)
                        elem_in_v = of_v.acquire(1)
                        elt_of_out_scale2 = of_scale.acquire(1)

                        matmul_PV(
                            elem_in_p,
                            elem_in_v,
                            elem_o_out,
                            elt_of_out_scale2,
                            q_seq_tile,
                            1,
                            idx_buffer,
                        )

                        of_p.release(1)
                        of_v.release(1)
                        of_scale.release(1)

                        idx_buffer[0] += 1

                ### Last iteration, final rescaling
                if num_kv_seq_blocks > 1:
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elt_of_out_scale3 = of_scale.acquire(1)

                    matmul_PV(
                        elem_in_p,
                        elem_in_v,
                        elem_o_out,
                        elt_of_out_scale3,
                        q_seq_tile,
                        1,
                        idx_buffer,
                    )
                    rescale_O(elem_o_out, elt_of_out_scale3, q_seq_tile, idx_buffer)

                    of_p.release(1)
                    of_v.release(1)
                    of_scale.release(1)

                    idx_buffer[0] += 1
                # else:
                else:
                    rescale_O(elem_o_out, elt_of_out_scale, q_seq_tile, idx_buffer)
                    idx_buffer[0] += 1
                ###

                idx_buffer[0] = 0
                idx_buffer[1] += 0  # Used to be parameter for seq block parallelism

                of_o_out.release(1)

    def matmul_o_proj(
        of_o_in,
        of_ow_in,
        of_o_acc_in,
        of_o_acc_out,
        of_o_out,
        buffer_to_reduce,
        zero,
        matmul,
        add,
        copy,
    ):
        """
        Amount of work for prev stages to generate its ouptut to next stage for one head:
            QK: q_seq_tile * d * kv_seq_tile
            S: q_seq_tile * kv_seq_tile
            PV: q_seq_tile * kv_seq_tile * d * (seq_len // q_seq_tile)
        Amount of work for prev stages to process generate full row outputs:
            QK: q_seq_tile * d * kv_seq_tile * (seq_len // q_seq_tile)
            S: q_seq_tile * kv_seq_tile * (seq_len // q_seq_tile)
            PV: q_seq_tile * kv_seq_tile * d * (seq_len // q_seq_tile) * (embed_dim // d)
        Output projection needs all heads to generate one output tile:
            O: q_seq_tile * d * emb_tile * (embed_dim // d)
        So full rows are generated after this amount of compute:
            O: q_seq_tile * d * emb_tile * (embed_dim // d) * (embed_dim // emb_tile)
        The output from PV can be reused to partially accumulate output tiles:
            O: q_seq_tile * d * emb_tile * (embed_dim // emb_tile)
        """

        for _ in range_(sys.maxsize):

            # First iteration just passes the partial C tile through
            for _ in range_(o_proj_acc_depth):

                elem_out_o_acc = of_o_acc_out.acquire(1)
                zero(elem_out_o_acc)
                of_o_acc_out.release(1)

            for _ in range_(num_qkv_head_block_per_parallel_head):

                elem_in_o = of_o_in.acquire(1)

                for _ in range_(o_proj_acc_depth):

                    elem_in_o_acc = of_o_acc_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_o_acc = of_o_acc_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_in_o_acc, elem_out_o_acc)
                    of_o_acc_out.release(1)
                    of_ow_in.release(1)
                    of_o_acc_in.release(1)

                of_o_in.release(1)

            for _ in range_(o_proj_acc_depth):

                # Acquire what's in L2, which is the final accumulated result for the tile
                elem_in_o_acc = of_o_acc_in.acquire(1)

                if buffer_to_reduce:

                    # Don't send any new data to MT, i.e. of_o_acc_out, because that will affect
                    # the data in the subsequent tiles. It's sufficient to just use
                    # the internal buffer as input and output
                    partial_o_acc = buffer_to_reduce.acquire(1)
                    add(
                        partial_o_acc,
                        elem_in_o_acc,
                        elem_in_o_acc,
                        q_seq_tile * emb_tile,
                    )
                    buffer_to_reduce.release(1)

                elem_out_o = of_o_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o, q_seq_tile * emb_tile)
                of_o_acc_in.release(1)
                of_o_out.release(1)

    # Create worker from task
    matmul_workers = []
    softmax_workers = []
    matmul_pv_workers = []
    o_proj_workers = []
    for i in range(parallel_heads):
        idx_buffer_qk = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_qk_{i}",
        )
        matmul_workers.append(
            Worker(
                batched_matmul_qk,
                fn_args=[
                    memQ[i].cons(),
                    memK[i].cons(),
                    memA[i].prod(),
                    zero_kernel,
                    matmul_QK,
                    i,
                    idx_buffer_qk,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=2),
                while_true=False,
            )
        )
        idx_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_softmax_{i}",
        )
        scale_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(4 * q_seq_tile,), dtype=dtype),
            name=f"scale_buffer_softmax_{i}",
        )
        softmax_workers.append(
            Worker(
                softmax,
                fn_args=[
                    memA[i].cons(),
                    memP[i].prod(),
                    scaleOF[i].prod(),
                    partial_softmax_kernel,
                    scale_buffer_init_kernel,
                    memcopy_kernel_scale,
                    i,
                    idx_buffer_softmax,
                    scale_buffer_softmax,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=3),
                while_true=False,
            )
        )
        idx_buffer_pv = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_pv_{i}",
        )
        matmul_pv_workers.append(
            Worker(
                batched_matmul_pv,
                fn_args=[
                    memP[i].cons(),
                    memV[i].cons(),
                    scaleOF[i].cons(),
                    outOProj[i].prod(),
                    zero_kernel_q,
                    matmul_PV,
                    rescale_O,
                    i,
                    idx_buffer_pv,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=4),
                while_true=False,
            )
        )
        o_proj_workers.append(
            Worker(
                matmul_o_proj,
                fn_args=[
                    outOProj[i].cons(),
                    memOW[i].cons(),
                    outOProjAccumIn[i].cons(depth=1),
                    outOProjAccumOut[i].prod(),
                    # Last head writes to the output OF directly, others write to partial accumulation tiles
                    outOPart[i].prod() if i < parallel_heads - 1 else outO[0].prod(),
                    outOPart[i - 1].cons() if i > 0 else None,
                    zero_kernel_o_proj,
                    matmul_kernel_o_proj,
                    eltwise_add_vector,
                    mem_copy_o_proj,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=5),
                while_true=False,
            )
        )

    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    # NOTE: It's important that the tiling of Q/K/V are such that the subsequent tiles
    # across the heads, not the sequence length (i.e. how it's done in the MHA operator),
    # because the cores execute on each head. However, we have to keep in mind that
    # K/v need to have the full sequence length passed for each head.
    q_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (q_seq_tile, d),
        (1, heads),
    )

    k_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (kv_seq_tile, d),
        (num_kv_seq_blocks, parallel_heads),
    )

    v_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (kv_seq_tile, d),
        (num_kv_seq_blocks, parallel_heads),
    )

    Q_tiles = q_tiles_base
    K_tiles = k_tiles_base
    V_tiles = v_tiles_base

    # NOTE: Dividing by num_o_col_groups to get the correct number of tiles expected
    # in the runtime seequence. Also not including o_proj_acc_depth in the tile col
    # dim because the WO buffers operate on tiles with size emb_tile. If we
    # use a tile col dim of emb_tile * o_proj_acc_depth, then the (d, emb_tile)
    # used for WO will be wrong as the data is written contiguously based on the
    # access pattern, i.e. written contiguously as rows of size emb_tile * o_proj_acc_depth,
    # when it should be rows of size emb_tile
    WO_tiles = TensorTiler2D.group_tiler(
        (embed_sz, embed_sz),
        (d, emb_tile),
        (parallel_heads, embed_sz // emb_tile // num_o_col_groups),
    )
    # Flip the first two dimensions of WO so that the 3rd dimension iterates over rows for splitting
    # to FIFOs and 4th dimension iterates over columns for partial accumulations
    for tile in WO_tiles:
        tile._sizes = [tile._sizes[1], tile._sizes[0], tile._sizes[2], tile._sizes[3]]
        tile._strides = [
            tile._strides[1],
            tile._strides[0],
            tile._strides[2],
            tile._strides[3],
        ]

    O_tiles = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (q_seq_tile, emb_tile),
        (1, embed_sz // emb_tile // num_o_col_groups),
    )

    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            logging.info(f"{name} tile {idx}:")
            logging.info(f"  Offset: {tap.offset}")
            logging.info(f"  Sizes: {tap.sizes}")
            logging.info(f"  Strides: {tap.strides}")

    def legalize_tap(tap: TensorAccessPattern, max_dim_size: int):

        sizes = list(tap._sizes)
        strides = list(tap._strides)

        # Skip if no need to legalize
        if all(size <= max_dim_size for size in sizes):
            return tap

        # Split oversized dimensions (working backwards to preserve indices)
        i = len(sizes) - 1
        while i >= 0:
            if sizes[i] > max_dim_size:
                # Calculate quotient and remainder for splitting
                multiplier = 1
                while sizes[i] > max_dim_size:
                    sizes[i] //= 2
                    multiplier *= 2
                quotient = multiplier
                remainder = sizes[i]

                # Insert new outer dimension before this one
                sizes.insert(i, quotient)
                strides.insert(i, strides[i] * remainder)

                # Update the inner dimension
                sizes[i + 1] = remainder
                # stride[i + 1] stays the same

                i -= 1  # Skip the newly inserted dimension
            i -= 1

        # Remove leading dimensions with size 1
        while sizes and sizes[0] == 1:
            sizes.pop(0)
            strides.pop(0)

        # Check that the number of dimensions does not exceed hardware limit
        if len(sizes) > 4:
            raise ValueError(
                f"Cannot legalize: resulting dimensions {len(sizes)} exceed maximum of 4 "
                f"supported by hardware (sizes: {sizes}, strides: {strides})"
            )

        tap._sizes = sizes
        tap._strides = strides

        return tap

    def legalize_tas(tas: TensorAccessSequence):

        max_dim_size = 1023  # Max DMA dimension size for memTile DMA on NPU2

        for tap in tas:
            tap = legalize_tap(tap, max_dim_size)

    legalize_tas(Q_tiles)
    legalize_tas(K_tiles)
    legalize_tas(V_tiles)
    legalize_tas(WO_tiles)
    legalize_tas(O_tiles)

    print_tap_seq_info(Q_tiles, "Q")
    print_tap_seq_info(K_tiles, "K")
    print_tap_seq_info(V_tiles, "V")
    print_tap_seq_info(WO_tiles, "W_O")
    print_tap_seq_info(O_tiles, "O")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(W_O_ty, Q_ty, K_ty, V_ty, O_ty) as (W_O, Q, K, V, O):

        for i in range(parallel_heads):
            rt.start(matmul_workers[i])
            rt.start(softmax_workers[i])
            rt.start(matmul_pv_workers[i])
            rt.start(o_proj_workers[i])

        for q_block_idx in range(num_q_seq_blocks):

            for col_group in range(num_o_col_groups):
                # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
                tg = rt.task_group()

                rt.fill(
                    inQ.prod(),
                    Q,
                    tap=Q_tiles[q_block_idx],
                    placement=Tile(col=0, row=0),
                    task_group=tg,
                )
                logging.debug(
                    f"Scheduling fills for q block {q_block_idx}, col group {col_group} for Q, K, V, W_O, and O"
                )
                logging.debug(f"  Q tap: {Q_tiles[q_block_idx]}")
                for head_idx in range(heads // parallel_heads):
                    tg_head = rt.task_group()
                    rt.fill(
                        inK.prod(),
                        K,
                        tap=K_tiles[head_idx],
                        placement=Tile(col=1, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                    rt.fill(
                        inV.prod(),
                        V,
                        tap=V_tiles[head_idx],
                        placement=Tile(col=2, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                    rt.fill(
                        inOW.prod(),
                        W_O,
                        tap=WO_tiles[head_idx * num_o_col_groups + col_group],
                        placement=Tile(col=3, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                    rt.finish_task_group(tg_head)
                    logging.debug(f"    K tap: {K_tiles[head_idx]}")
                    logging.debug(f"    V tap: {V_tiles[head_idx]}")
                    logging.debug(
                        f"    W_O tap: {WO_tiles[head_idx * num_o_col_groups + col_group]}"
                    )

                rt.drain(
                    memO.cons(),
                    O,
                    tap=O_tiles[q_block_idx * (num_o_col_groups) + col_group],
                    wait=True,
                    placement=Tile(col=7, row=0),
                    task_group=tg,
                )
                logging.debug(
                    f"  O tap: {O_tiles[q_block_idx * (num_o_col_groups) + col_group]}"
                )

                rt.finish_task_group(tg)

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1Col1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
