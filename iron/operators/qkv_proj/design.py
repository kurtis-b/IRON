# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from itertools import product
import json
import sys

import numpy as np

from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU1Col2, NPU2, Tile
from aie.iron.placers import SequentialPlacer

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
    },
    "npu2": {
        "bf16": {
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
    },
}

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


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


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
        topology_id_fn = _topology_id
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


def qkv_proj_practical_topologies(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    max_candidates: int = _BLOCK1_PRACTICAL_MAX_CANDIDATES,
) -> list[dict[str, int | str]]:
    if max_candidates <= 0:
        raise ValueError("Block 1 practical exploration requires max_candidates > 0")

    theoretical = qkv_proj_theoretical_topologies(
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

    if topology_id is None:
        config = _block1_seq_independent_supported_candidates(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )[0]
        resolved_topology_id = _topology_id(config)
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
            legacy_aliases = _block1_legacy_runtime_aliases(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            try:
                config = legacy_aliases[str(topology_id)]
                resolved_topology_id = str(topology_id)
            except KeyError:
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
    del trace_size

    n_aie_rows = 4
    n_shim_mem_A = min(num_aie_columns, n_aie_rows)
    n_A_tiles_per_shim = n_aie_rows // num_aie_columns if num_aie_columns < 4 else 1

    mem_tile_m_A = tile_m * n_A_tiles_per_shim
    mem_tile_m_C = tile_m * n_aie_rows
    mem_tile_n = tile_n * num_aie_columns

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    if prio_accuracy:
        if dtype_out_str != "bf16":
            raise AssertionError(
                "prio_accuracy is supported only for Block 1 bf16 outputs"
            )
        use_larger_internal_buffer = True
        dtype_out_internal = str_to_dtype("f32")
        if np.issubdtype(dtype_in, np.integer) != np.issubdtype(
            dtype_out_internal, np.integer
        ):
            raise AssertionError("Input/output internal dtypes must have matching kind")
        if np.dtype(dtype_out_internal).itemsize < np.dtype(dtype_in).itemsize:
            raise AssertionError("Internal accumulation dtype must not shrink input")
    else:
        use_larger_internal_buffer = False

    if np.issubdtype(dtype_in, np.integer) != np.issubdtype(dtype_out, np.integer):
        raise AssertionError("Input/output dtypes must have matching kind")
    if np.dtype(dtype_out).itemsize < np.dtype(dtype_in).itemsize:
        raise AssertionError("Output dtype must not shrink input")

    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    if dev == "npu" and num_aie_columns > 4:
        raise AssertionError("Invalid configuration: NPU has 4 columns")
    if dev == "npu2" and num_aie_columns > 8:
        raise AssertionError("Invalid configuration: NPU2 has 8 columns")

    if seq_len % mem_tile_m_A != 0:
        raise AssertionError("Block 1 input must tile into mem-tile A blocks")
    if hidden_size % tile_k != 0:
        raise AssertionError("Block 1 hidden_size must be divisible by tile_k")
    if combined_hidden_size % mem_tile_n != 0:
        raise AssertionError(
            "Block 1 fused output width must tile into mem-tile output blocks"
        )
    if seq_len % mem_tile_m_C != 0:
        raise AssertionError("Block 1 output must tile into mem-tile C blocks")

    if not use_scalar:
        if tile_m % r != 0 or tile_k % s != 0 or tile_n % t != 0:
            raise AssertionError(
                "Block 1 tile sizes must match the selected MMUL microkernel"
            )

    fifo_depth = 2
    if dev == "npu":
        if num_aie_columns == 1:
            dev_ty = NPU1Col1()
        elif num_aie_columns == 2:
            dev_ty = NPU1Col2()
        elif num_aie_columns == 4:
            dev_ty = NPU1()
        else:
            raise AssertionError("Invalid Block 1 NPU column count")
    else:
        dev_ty = NPU2()

    A_ty = np.ndarray[(seq_len * hidden_size,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(hidden_size * combined_hidden_size,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(seq_len * combined_hidden_size,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(mem_tile_m_A * tile_k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(tile_k * tile_n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(mem_tile_m_C * tile_n,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(tile_m, tile_k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(tile_k, tile_n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(tile_m, tile_n), np.dtype[dtype_out]]

    scalar_suffix = "_scalar" if use_scalar else ""
    archive_name = (
        f"qkv_proj_{tile_m}x{tile_k}x{tile_n}_archive.a" if archive is None else archive
    )
    if use_larger_internal_buffer:
        fifo_depth_out = 1
        C_l1_ty_internal = np.ndarray[(tile_m, tile_n), np.dtype[dtype_out_internal]]
        convert_copy_kernel = Kernel(
            "convert_copy_f32_to_bf16",
            archive_name,
            [C_l1_ty_internal, C_l1_ty, np.int32],
        )
        zero_kernel = Kernel(
            f"zero{scalar_suffix}_f32",
            archive_name,
            [C_l1_ty_internal],
        )
        matmul_kernel = Kernel(
            f"matmul{scalar_suffix}_{dtype_in_str}_f32",
            archive_name,
            [A_l1_ty, B_l1_ty, C_l1_ty_internal],
        )
    else:
        fifo_depth_out = fifo_depth
        convert_copy_kernel = None
        zero_kernel = Kernel(
            f"zero{scalar_suffix}_{dtype_out_str}",
            archive_name,
            [C_l1_ty],
        )
        matmul_kernel = Kernel(
            f"matmul{scalar_suffix}_{dtype_in_str}_{dtype_out_str}",
            archive_name,
            [A_l1_ty, B_l1_ty, C_l1_ty],
        )

    tiles = [[(col, row) for col in range(num_aie_columns)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * num_aie_columns
    B_l2l1_fifos = [None] * num_aie_columns
    C_l1l2_fifos = [[None] * num_aie_columns for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * num_aie_columns

    rtps = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp{row}_{col}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(num_aie_columns)
        ]
        for row in range(n_aie_rows)
    ]
    worker_barriers = [
        [WorkerRuntimeBarrier(initial_value=0) for col in range(num_aie_columns)]
        for row in range(n_aie_rows)
    ]

    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        offsets = [tile_m * tile_k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (tile_m // r, r * tile_k),
                (tile_k // s, s),
                (r, tile_k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(2 * i if num_aie_columns == 8 else i, 1),
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    for col in range(num_aie_columns):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        dims_to_stream = [
            (tile_k // s, s * tile_n),
            (tile_n // t, t),
            (s, tile_n),
            (t, 1),
        ]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        dims_to_stream = [
            (tile_m // r, r * tile_n),
            (r, t),
            (tile_n // t, r * t),
            (t, 1),
        ]
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=dims_to_stream,
        )
        offsets = [tile_m * tile_n * i for i in range(n_aie_rows)]
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth_out] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    def core_fn(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        convert_copy,
        my_rtp,
        barrier,
        elem_out_internal,
    ):
        for _ in range_(sys.maxsize):
            barrier.wait_for_value(1)
            rtp_K_div_k = my_rtp[0]
            rtp_n_tiles_per_core = my_rtp[1]
            loop = range(1)
            if rtp_n_tiles_per_core > 1:
                loop = range_(rtp_n_tiles_per_core)
            for _ in loop:
                if not use_larger_internal_buffer:
                    elem_out_internal = out_c.acquire(1)
                zero(elem_out_internal)

                for _ in range_(rtp_K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out_internal)
                    in_a.release(1)
                    in_b.release(1)

                if use_larger_internal_buffer:
                    elem_out_transfer = out_c.acquire(1)
                    convert_copy(elem_out_internal, elem_out_transfer, tile_m * tile_n)
                    out_c.release(1)
                else:
                    out_c.release(1)

            barrier.wait_for_value(0)

    workers = []
    for row in range(n_aie_rows):
        for col in range(num_aie_columns):
            tile_col, tile_row = core_tiles[row][col]
            acc_buffer = None
            if use_larger_internal_buffer:
                acc_buffer = Buffer(
                    type=C_l1_ty_internal, name=f"acc_buffer_{row}_{col}"
                )
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                        convert_copy_kernel,
                        rtps[row][col],
                        worker_barriers[row][col],
                        acc_buffer,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    K_div_k = hidden_size // tile_k
    n_c_col_tiles_per_core = combined_hidden_size // mem_tile_n
    n_c_row_tiles_per_core = seq_len // mem_tile_m_C
    tb_max_n_rows = 4

    A_tiles = TensorTiler2D.group_tiler(
        (seq_len, hidden_size),
        (mem_tile_m_A, tile_k),
        (1, K_div_k),
        pattern_repeat=n_c_col_tiles_per_core,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (hidden_size, combined_hidden_size),
        (tile_k, tile_n),
        tile_group_repeats=(K_div_k, n_c_col_tiles_per_core),
        tile_group_steps=(1, num_aie_columns),
        tile_group_col_major=True,
        prune_step=False,
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        def set_rtps(*args):
            for rtps_row in args:
                for rtp_row_col in rtps_row:
                    rtp_row_col[0] = K_div_k
                    rtp_row_col[1] = n_c_row_tiles_per_core * n_c_col_tiles_per_core

        rt.inline_ops(set_rtps, rtps)

        for row in range(n_aie_rows):
            for col in range(num_aie_columns):
                rt.set_barrier(worker_barriers[row][col], 1)

        tg = rt.task_group()
        for tb in range(ceildiv(n_c_row_tiles_per_core, tb_max_n_rows)):
            for pingpong in [0, 1]:
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    tb_max_n_rows // 2, n_c_row_tiles_per_core - row_base
                )
                if current_tb_n_rows <= 0:
                    break
                for col in range(num_aie_columns):
                    C_row_offset = row_base * mem_tile_m_C * combined_hidden_size
                    C_col_offset = col * tile_n
                    C_offset = C_col_offset + C_row_offset
                    C_tile = TensorAccessPattern(
                        (seq_len, combined_hidden_size),
                        offset=C_offset,
                        sizes=[
                            current_tb_n_rows,
                            combined_hidden_size // mem_tile_n,
                            mem_tile_m_C,
                            tile_n,
                        ],
                        strides=[
                            mem_tile_m_C * combined_hidden_size,
                            mem_tile_n,
                            combined_hidden_size,
                            1,
                        ],
                    )
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tile,
                        wait=True,
                        task_group=tg,
                        placement=Tile(col, 0),
                    )

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)

                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                                placement=Tile(
                                    2 * col if num_aie_columns == 8 else col,
                                    0,
                                ),
                            )

                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)
        for row in range(n_aie_rows):
            for col in range(num_aie_columns):
                rt.set_barrier(worker_barriers[row][col], 0)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


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
        topology_id = _topology_id(candidate)
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
        for candidate in qkv_proj_practical_topologies(
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


def _block1_legacy_runtime_aliases(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, dict[str, int | str]]:
    aliases: dict[str, dict[str, int | str]] = {}
    for candidate in _block1_runtime_supported_candidates(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    ):
        if int(candidate["num_aie_columns"]) != 8:
            continue
        alias_id = _topology_id(candidate)
        aliases.setdefault(
            alias_id,
            {
                **candidate,
                "topology_id": _theoretical_topology_id(candidate),
                "topology_family": "shared_runtime_qkv_proj",
            },
        )
    return aliases


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
