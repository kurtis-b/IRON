# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aie.dialects.index as index
import numpy as np
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU1Col2, NPU2, Tile
from aie.iron.placers import SequentialPlacer
from ml_dtypes import bfloat16

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


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def choose_phase1_rows(m: int, K: int, dtype_itemsize: int) -> int:
    # Phase 1 keeps five live row-major buffers on the core:
    # A, R, preadd, tmp_norm, ln1_out. Pick the largest row chunk that
    # divides m evenly while staying within a conservative local-memory budget.
    local_buffer_budget_bytes = 48 * 1024
    live_buffers = 5
    max_rows = max(1, local_buffer_budget_bytes // (live_buffers * K * dtype_itemsize))
    for rows in range(min(m, max_rows), 0, -1):
        if m % rows == 0:
            return rows
    return 1


def fused_addnorm_ffn_addnorm(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    down_proj_depth,
    nA_tiles_distributed,
    nB_tiles_distributed,
    dtype_in_str,
    dtype_out_str,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    gelu_stage,
    stage_only,
    ln1_weight_file,
    ln2_weight_file,
    archive,
    n_aie_cols,
    debug_mode=-1,
):
    del trace_size, debug_mode

    if ln1_weight_file is None:
        static_ln1_weights = np.ones(K, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
    if ln2_weight_file is None:
        static_ln2_weights = np.ones(K, dtype=bfloat16)
    else:
        static_ln2_weights = np.load(ln2_weight_file)

    if static_ln1_weights.shape[0] != K:
        raise ValueError("Static ln1 weights length does not match K")
    if static_ln2_weights.shape[0] != K:
        raise ValueError("Static ln2 weights length does not match K")

    if nA_tiles_distributed < 1 or nA_tiles_distributed > 2:
        raise AssertionError("Staged Block 3 baseline supports parallel_seq in {1, 2}")
    if nB_tiles_distributed < 1:
        raise AssertionError("parallel_int_dim must be at least 1")
    if n_aie_cols < nB_tiles_distributed + 2:
        raise AssertionError(
            "Staged Block 3 baseline requires num_aie_columns >= parallel_int_dim + 2"
        )

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mem_tile_n = n * nB_tiles_distributed
    K_div_k = K // k
    nC_up_col_tiles_per_core = N // mem_tile_n
    ln_iters_per_core = M // (nA_tiles_distributed * m)
    nC_tiles_per_core = nC_up_col_tiles_per_core * ln_iters_per_core
    n_aie_cores_needed = nA_tiles_distributed * (2 + 2 * nB_tiles_distributed)
    phase1_rows = choose_phase1_rows(m, K, np.dtype(dtype_in).itemsize)
    phase1_chunks_per_tile = m // phase1_rows
    phase1_iters_per_core = ln_iters_per_core * phase1_chunks_per_tile

    assert M % (nA_tiles_distributed * m) == 0
    assert K % k == 0
    assert N % n == 0
    assert N % mem_tile_n == 0
    assert K == k * down_proj_depth
    assert m % phase1_rows == 0

    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    assert m % (2 * r) == 0
    assert k % s == 0
    assert n % (2 * t) == 0

    if dev == "npu":
        if n_aie_cols > 4 or n_aie_cores_needed > 16:
            raise AssertionError("Invalid configuration for NPU")
        if n_aie_cores_needed <= 4:
            dev_ty = NPU1Col1()
        elif n_aie_cores_needed <= 8:
            dev_ty = NPU1Col2()
        else:
            dev_ty = NPU1()
    else:
        if n_aie_cols > 8 or n_aie_cores_needed > 32:
            raise AssertionError("Invalid configuration for NPU2")
        dev_ty = NPU2()

    fifo_depth = 2
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    B_stacked_ty = np.ndarray[(2 * K * N,), np.dtype[dtype_in]]
    stage_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    stage_stacked_ty = np.ndarray[(2 * M * K,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * K,), np.dtype[dtype_out]]

    ln_weights_ty = np.ndarray[(K,), np.dtype[dtype_in]]
    phase1_l1_ty = np.ndarray[(phase1_rows, K), np.dtype[dtype_in]]
    A_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    B_up_proj_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    B_down_proj_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
    C_up_proj_l1_ty = np.ndarray[(m, n), np.dtype[dtype_in]]
    sum_l1_ty = np.ndarray[(m,), np.dtype[str_to_dtype("f32")]]

    archive_name = archive or f"staged_block3_{m}x{k}x{n}.a"

    ffn_zero_kernel_up_proj = Kernel(
        f"ffn_zero_{dtype_out_str}_up_proj",
        archive_name,
        [C_up_proj_l1_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        f"ffn_matmul_{dtype_in_str}_{dtype_out_str}_up_proj",
        archive_name,
        [A_l1_ty, B_up_proj_l1_ty, C_up_proj_l1_ty],
    )
    ffn_gelu_kernel = Kernel(
        "ffn_gelu_bf16",
        archive_name,
        [C_up_proj_l1_ty, C_up_proj_l1_ty, np.int32],
    )
    ffn_zero_kernel_down_proj = Kernel(
        f"ffn_zero_{dtype_out_str}_down_proj",
        archive_name,
        [A_l1_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        f"ffn_matmul_with_acc_{dtype_in_str}_{dtype_out_str}_down_proj",
        archive_name,
        [C_up_proj_l1_ty, B_down_proj_l1_ty, A_l1_ty, A_l1_ty],
    )
    ffn_mem_copy_kernel = Kernel(
        "ffn_passThroughLine",
        archive_name,
        [A_l1_ty, A_l1_ty, np.int32],
    )
    phase1_add_kernel = Kernel(
        "block3_ln1_eltwise_add_bf16_vector",
        archive_name,
        [phase1_l1_ty, phase1_l1_ty, phase1_l1_ty, np.int32],
    )
    ffn_eltwise_add_kernel = Kernel(
        "ffn_eltwise_add_bf16_vector",
        archive_name,
        [A_l1_ty, A_l1_ty, A_l1_ty, np.int32],
    )
    ln2_zero_f32_kernel = Kernel(
        "ln_zero_f32",
        archive_name,
        [sum_l1_ty, np.int32],
    )
    ln2_calc_sum_sumsq_kernel = Kernel(
        "ln_calc_sum_sumsq",
        archive_name,
        [A_l1_ty, sum_l1_ty, sum_l1_ty],
    )
    ln2_fused_layer_norm_kernel = Kernel(
        "fused_layer_norm_1outs",
        archive_name,
        [
            A_l1_ty,
            sum_l1_ty,
            sum_l1_ty,
            A_l1_ty,
            np.int32,
        ],
    )
    ln2_mul_weights_kernel = Kernel(
        "ln_mul_weights_1outs",
        archive_name,
        [
            A_l1_ty,
            ln_weights_ty,
            A_l1_ty,
            np.int32,
        ],
    )
    ln1_layer_norm_rows_kernel = Kernel(
        "block3_ln1_layer_norm_rows",
        archive_name,
        [
            phase1_l1_ty,
            phase1_l1_ty,
            np.int32,
            np.int32,
        ],
    )
    ln1_mul_rows_kernel = Kernel(
        "block3_ln1_eltwise_mul_bf16_vector_rows",
        archive_name,
        [
            phase1_l1_ty,
            ln_weights_ty,
            phase1_l1_ty,
            np.int32,
            np.int32,
        ],
    )

    # Phase 1 FIFOs.
    phase1_A_l3l2_fifos = [None] * nA_tiles_distributed
    phase1_A_l2l1_fifos = [None] * nA_tiles_distributed
    phase1_R_l3l2_fifos = [None] * nA_tiles_distributed
    phase1_R_l2l1_fifos = [None] * nA_tiles_distributed
    phase1_preadd_l1l2_fifos = [None] * nA_tiles_distributed
    phase1_preadd_l2l3_fifos = [None] * nA_tiles_distributed
    phase1_ln1_l1l2_fifos = [None] * nA_tiles_distributed
    phase1_ln1_l2l3_fifos = [None] * nA_tiles_distributed

    for a_tile in range(nA_tiles_distributed):
        phase1_A_l3l2_fifos[a_tile] = ObjectFifo(
            phase1_l1_ty, name=f"phase1_A_L3L2_{a_tile}", depth=fifo_depth
        )
        phase1_A_l2l1_fifos[a_tile] = (
            phase1_A_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=phase1_l1_ty,
                name=f"phase1_A_L2L1_{a_tile}",
                placement=Tile(a_tile, 1),
            )
        )

        phase1_R_l3l2_fifos[a_tile] = ObjectFifo(
            phase1_l1_ty, name=f"phase1_R_L3L2_{a_tile}", depth=fifo_depth
        )
        phase1_R_l2l1_fifos[a_tile] = (
            phase1_R_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=phase1_l1_ty,
                name=f"phase1_R_L2L1_{a_tile}",
                placement=Tile(n_aie_cols - 1 - a_tile, 1),
            )
        )

        phase1_preadd_l1l2_fifos[a_tile] = ObjectFifo(
            phase1_l1_ty, name=f"stage_preadd_L1L2_{a_tile}", depth=fifo_depth
        )
        phase1_preadd_l2l3_fifos[a_tile] = (
            phase1_preadd_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=phase1_l1_ty,
                name=f"stage_preadd_L2L3_{a_tile}",
                placement=Tile(n_aie_cols - 1 - a_tile, 1),
            )
        )

        phase1_ln1_l1l2_fifos[a_tile] = ObjectFifo(
            phase1_l1_ty, name=f"stage_ln1_L1L2_{a_tile}", depth=fifo_depth
        )
        phase1_ln1_l2l3_fifos[a_tile] = (
            phase1_ln1_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=phase1_l1_ty,
                name=f"stage_ln1_L2L3_{a_tile}",
                placement=Tile(a_tile, 1),
            )
        )

    # Phase 2 FIFOs.
    phase2_A_l3l2_fifos = [None] * nA_tiles_distributed
    phase2_A_l2l1_fifos = [None] * nA_tiles_distributed
    phase2_R_l3l2_fifos = [None] * nA_tiles_distributed
    phase2_R_l2l1_fifos = [None] * nA_tiles_distributed
    B_up_proj_l3l2_fifos = [None] * nB_tiles_distributed
    B_up_proj_l2l1_fifos = [None] * nB_tiles_distributed
    B_down_proj_l3l2_fifos = [None] * nB_tiles_distributed
    B_down_proj_l2l1_fifos = [None] * nB_tiles_distributed
    C_up_proj_l1l1_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_part_l1l2_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_part_l2l1_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_reduce_l1l1_fifos = [
        [None] * max(0, nB_tiles_distributed - 1) for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_out_l1l1_fifos = [None] * nA_tiles_distributed
    ln2_l1l2_fifos = [None] * nA_tiles_distributed
    ln2_l2l3_fifos = [None] * nA_tiles_distributed

    dims_to_stream_a = [
        (m // r, r * k),
        (k // s, s),
        (r, k),
        (s, 1),
    ]
    for a_tile in range(nA_tiles_distributed):
        phase2_A_l3l2_fifos[a_tile] = ObjectFifo(
            A_l2_ty, name=f"A_L3L2_{a_tile}", depth=fifo_depth
        )
        phase2_A_l2l1_fifos[a_tile] = (
            phase2_A_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"A_L2L1_{a_tile}",
                dims_to_stream=dims_to_stream_a,
                placement=Tile(a_tile, 1),
            )
        )

        phase2_R_l3l2_fifos[a_tile] = ObjectFifo(
            A_l1_ty, name=f"R_L3L2_{a_tile}", depth=fifo_depth
        )
        phase2_R_l2l1_fifos[a_tile] = (
            phase2_R_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"R_L2L1_{a_tile}",
                dims_to_stream=dims_to_stream_a,
                placement=Tile(n_aie_cols - 1 - a_tile, 1),
            )
        )

    for b_tile in range(nB_tiles_distributed):
        dims_to_stream_up = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_up_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_up_L3L2_{b_tile}", depth=fifo_depth
        )
        B_up_proj_l2l1_fifos[b_tile] = (
            B_up_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_up_proj_l1_ty,
                name=f"B_up_L2L1_{b_tile}",
                dims_to_stream=dims_to_stream_up,
                placement=Tile(b_tile + 1, 1),
            )
        )

        dims_to_stream_down = [(n // s, s * k), (k // t, t), (s, k), (t, 1)]
        B_down_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_down_L3L2_{b_tile}", depth=fifo_depth
        )
        B_down_proj_l2l1_fifos[b_tile] = (
            B_down_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_down_proj_l1_ty,
                name=f"B_down_L2L1_{b_tile}",
                dims_to_stream=dims_to_stream_down,
                placement=Tile(b_tile + 1, 1),
            )
        )

    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            C_up_proj_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_up_proj_l1_ty,
                name=f"C_up_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )
            C_down_proj_part_l1l2_fifos[a_tile][b_tile] = ObjectFifo(
                A_l1_ty,
                name=f"C_down_L1L2_{a_tile}_{b_tile}",
                depth=1,
            )
            C_down_proj_part_l2l1_fifos[a_tile][b_tile] = (
                C_down_proj_part_l1l2_fifos[a_tile][b_tile]
                .cons(depth=down_proj_depth)
                .forward(
                    obj_type=A_l1_ty,
                    name=f"C_down_L2L1_{b_tile}_{a_tile}",
                    depth=down_proj_depth,
                    placement=Tile(b_tile + 1, 1),
                )
            )

        for b_tile in range(nB_tiles_distributed - 1):
            C_down_proj_reduce_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                A_l1_ty,
                name=f"C_down_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )

        C_down_proj_out_l1l1_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"C_out_L1L1_{a_tile}",
            depth=fifo_depth,
        )

        ln2_l1l2_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"ln2_L1L2_{a_tile}",
            depth=fifo_depth,
        )
        dims_to_stream_out = [(m // r, r * k), (r, s), (k // s, r * s), (s, 1)]
        ln2_l2l3_fifos[a_tile] = (
            ln2_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"ln2_L2L3_{a_tile}",
                dims_to_stream=dims_to_stream_out,
                placement=Tile(n_aie_cols - 1 - a_tile, 1),
            )
        )

    def core_fn_add_norm1(
        in_a,
        in_r,
        weights,
        tmp_ln1_norm,
        out_preadd,
        out_ln1,
        add,
        layer_norm_rows,
        mul_rows,
        stage_only,
    ):
        for _ in range_(phase1_iters_per_core):
            elem_a = in_a.acquire(1)
            elem_r = in_r.acquire(1)
            elem_preadd = out_preadd.acquire(1)
            elem_ln1 = out_ln1.acquire(1)
            if stage_only in (0, None):
                add(
                    elem_a,
                    elem_r,
                    elem_preadd,
                    phase1_rows * K,
                )
                layer_norm_rows(
                    elem_preadd,
                    tmp_ln1_norm,
                    K,
                    phase1_rows,
                )
                mul_rows(
                    tmp_ln1_norm,
                    weights,
                    elem_ln1,
                    K,
                    phase1_rows,
                )
            in_a.release(1)
            in_r.release(1)
            out_preadd.release(1)
            out_ln1.release(1)

    def core_fn_up_proj(in_a, in_b, out_c, zero, matmul, gelu, stage_only):
        loop = range(1)
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            if stage_only not in (1, None):
                for _ in range_(K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    in_a.release(1)
                    in_b.release(1)
                out_c.release(1)
                continue

            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            if gelu:
                gelu(elem_out, elem_out, m * n)
            out_c.release(1)

    def core_fn_down_proj(
        in_a,
        in_b,
        curr_acc_c,
        new_acc_c,
        out_acc_c,
        zero,
        matmul,
        add,
        copy,
        gelu,
        buffer_to_reduce,
        is_end_of_down_proj,
        stage_only,
    ):
        if stage_only not in (2, None):
            for _ in range_(down_proj_depth):
                elem_acc_c = new_acc_c.acquire(1)
                new_acc_c.release(1)
            for _ in range_(nC_up_col_tiles_per_core):
                elem_in_a = in_a.acquire(1)
                for _ in range_(down_proj_depth):
                    elem_out_internal = curr_acc_c.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    elem_new_acc_c = new_acc_c.acquire(1)
                    new_acc_c.release(1)
                    in_b.release(1)
                    curr_acc_c.release(1)
                in_a.release(1)
            for _ in range_(down_proj_depth):
                elem_out_internal = curr_acc_c.acquire(1)
                if buffer_to_reduce:
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    buffer_to_reduce.release(1)
                elem_out_acc_c = out_acc_c.acquire(1)
                out_acc_c.release(1)
                if is_end_of_down_proj:
                    elem_new_acc_c = new_acc_c.acquire(1)
                    new_acc_c.release(1)
                curr_acc_c.release(1)
            if is_end_of_down_proj:
                for _ in range_(down_proj_depth):
                    elem_out_internal = curr_acc_c.acquire(1)
                    elem_out_acc_c = out_acc_c.acquire(1)
                    out_acc_c.release(1)
                    curr_acc_c.release(1)
            return

        for _ in range_(down_proj_depth):
            elem_acc_c = new_acc_c.acquire(1)
            zero(elem_acc_c)
            new_acc_c.release(1)
        for _ in range_(nC_up_col_tiles_per_core):
            elem_in_a = in_a.acquire(1)
            if gelu:
                gelu(elem_in_a, elem_in_a, m * n)
            for _ in range_(down_proj_depth):
                elem_out_internal = curr_acc_c.acquire(1)
                elem_in_b = in_b.acquire(1)
                elem_new_acc_c = new_acc_c.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out_internal, elem_new_acc_c)
                new_acc_c.release(1)
                in_b.release(1)
                curr_acc_c.release(1)
            in_a.release(1)
        for _ in range_(down_proj_depth):
            elem_out_internal = curr_acc_c.acquire(1)
            elem_out_acc_c = out_acc_c.acquire(1)
            if buffer_to_reduce:
                partial_acc_c = buffer_to_reduce.acquire(1)
                add(partial_acc_c, elem_out_internal, elem_out_internal, m * k)
                buffer_to_reduce.release(1)
            copy(elem_out_internal, elem_out_acc_c, m * k)
            if is_end_of_down_proj:
                elem_new_acc_c = new_acc_c.acquire(1)
                copy(elem_out_acc_c, elem_new_acc_c, m * k)
                new_acc_c.release(1)
            out_acc_c.release(1)
            curr_acc_c.release(1)
        if is_end_of_down_proj:
            for _ in range_(down_proj_depth):
                elem_out_internal = curr_acc_c.acquire(1)
                elem_out_acc_c = out_acc_c.acquire(1)
                copy(elem_out_internal, elem_out_acc_c, m * k)
                out_acc_c.release(1)
                curr_acc_c.release(1)

    def core_fn_add_norm2(
        of_in1,
        of_in2,
        sum_buf,
        sumsq_buf,
        weights,
        combined_buf,
        norm_buf,
        of_out1,
        add,
        fused_layer_norm,
        mul_weights,
        calc_sum_sumsq,
        zero_f32,
        stage_only,
    ):
        if stage_only not in (3, None):
            for _ in range_(down_proj_depth):
                elem_in1 = of_in1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                of_in1.release(1)
                of_in2.release(1)
            for _ in range_(down_proj_depth):
                elem_in1 = of_in1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                elem_out1 = of_out1.acquire(1)
                of_out1.release(1)
                of_in1.release(1)
                of_in2.release(1)
            return

        zero_f32(sum_buf, m)
        zero_f32(sumsq_buf, m)
        for _ in range_(down_proj_depth):
            elem_in1_stats = of_in1.acquire(1)
            elem_in2_stats = of_in2.acquire(1)
            add(elem_in1_stats, elem_in2_stats, combined_buf, m * k)
            calc_sum_sumsq(combined_buf, sum_buf, sumsq_buf)
            of_in1.release(1)
            of_in2.release(1)

        for col_idx in range_(down_proj_depth):
            col_i32 = index.casts(T.i32(), col_idx)
            elem_in1_out = of_in1.acquire(1)
            elem_in2_out = of_in2.acquire(1)
            elem_out1 = of_out1.acquire(1)
            add(elem_in1_out, elem_in2_out, combined_buf, m * k)
            fused_layer_norm(
                combined_buf,
                sum_buf,
                sumsq_buf,
                norm_buf,
                K,
            )
            mul_weights(
                norm_buf,
                weights,
                elem_out1,
                col_i32,
            )
            of_out1.release(1)
            of_in1.release(1)
            of_in2.release(1)

    workers = []
    for a_tile in range(nA_tiles_distributed):
        row_base = 2 + a_tile * 2

        workers.append(
            Worker(
                core_fn_add_norm1,
                [
                    phase1_A_l2l1_fifos[a_tile].cons(),
                    phase1_R_l2l1_fifos[a_tile].cons(),
                    Buffer(
                        type=ln_weights_ty,
                        initial_value=static_ln1_weights,
                        name=f"static_ln1_weights_{a_tile}",
                    ),
                    Buffer(type=phase1_l1_ty, name=f"ln1_tmp_norm_{a_tile}"),
                    phase1_preadd_l1l2_fifos[a_tile].prod(),
                    phase1_ln1_l1l2_fifos[a_tile].prod(),
                    phase1_add_kernel,
                    ln1_layer_norm_rows_kernel,
                    ln1_mul_rows_kernel,
                    stage_only,
                ],
                placement=Tile(0, row_base),
                stack_size=0xF00,
            )
        )

        workers.append(
            Worker(
                core_fn_add_norm2,
                [
                    C_down_proj_out_l1l1_fifos[a_tile].cons(),
                    phase2_R_l2l1_fifos[a_tile].cons(),
                    Buffer(type=sum_l1_ty, name=f"sum_buffer_{a_tile}"),
                    Buffer(type=sum_l1_ty, name=f"sumsq_buffer_{a_tile}"),
                    Buffer(
                        type=ln_weights_ty,
                        initial_value=static_ln2_weights,
                        name=f"static_ln2_weights_{a_tile}",
                    ),
                    Buffer(type=A_l1_ty, name=f"ln2_combined_buffer_{a_tile}"),
                    Buffer(type=A_l1_ty, name=f"ln2_norm_buffer_{a_tile}"),
                    ln2_l1l2_fifos[a_tile].prod(),
                    ffn_eltwise_add_kernel,
                    ln2_fused_layer_norm_kernel,
                    ln2_mul_weights_kernel,
                    ln2_calc_sum_sumsq_kernel,
                    ln2_zero_f32_kernel,
                    stage_only,
                ],
                placement=Tile(n_aie_cols - 1, row_base),
                stack_size=0xF00,
            )
        )

        for b_tile in range(nB_tiles_distributed):
            workers.append(
                Worker(
                    core_fn_up_proj,
                    [
                        phase2_A_l2l1_fifos[a_tile].cons(),
                        B_up_proj_l2l1_fifos[b_tile].cons(),
                        C_up_proj_l1l1_fifos[a_tile][b_tile].prod(),
                        ffn_zero_kernel_up_proj,
                        ffn_matmul_kernel_up_proj,
                        ffn_gelu_kernel if gelu_stage == 0 else None,
                        stage_only,
                    ],
                    placement=Tile(b_tile + 1, row_base + 1),
                    stack_size=0xF00,
                )
            )
            workers.append(
                Worker(
                    core_fn_down_proj,
                    [
                        C_up_proj_l1l1_fifos[a_tile][b_tile].cons(),
                        B_down_proj_l2l1_fifos[b_tile].cons(),
                        C_down_proj_part_l2l1_fifos[a_tile][b_tile].cons(depth=1),
                        C_down_proj_part_l1l2_fifos[a_tile][b_tile].prod(),
                        (
                            C_down_proj_out_l1l1_fifos[a_tile].prod(fifo_depth)
                            if b_tile == nB_tiles_distributed - 1
                            else C_down_proj_reduce_l1l1_fifos[a_tile][b_tile].prod(
                                fifo_depth
                            )
                        ),
                        ffn_zero_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        ffn_eltwise_add_kernel,
                        ffn_mem_copy_kernel,
                        ffn_gelu_kernel if gelu_stage == 1 else None,
                        (
                            None
                            if b_tile == 0
                            else C_down_proj_reduce_l1l1_fifos[a_tile][
                                b_tile - 1
                            ].cons()
                        ),
                        b_tile == nB_tiles_distributed - 1,
                        stage_only,
                    ],
                    placement=Tile(b_tile + 1, row_base),
                    stack_size=0xF00,
                )
            )

    stage_preadd_base = 0
    stage_ln1_base = M * K
    B_up_base = 0
    B_down_base = K * N
    stacked_stage_dims = (2, M * K)
    stacked_B_dims = (2, K * N)

    def stacked_tap(tap, *, tensor_dims, base_offset):
        return TensorAccessPattern(
            tensor_dims,
            base_offset + tap.offset,
            tap.sizes,
            tap.strides,
        )

    rt = Runtime()
    with rt.sequence(A_ty, A_ty, B_stacked_ty, stage_stacked_ty, C_ty) as (
        A,
        R,
        B_stacked,
        stage_stacked,
        C,
    ):
        rt.start(*workers)

        phase1_tg = rt.task_group()
        for row_tile in range(ln_iters_per_core):
            for a_tile in range(nA_tiles_distributed):
                row_base = (row_tile * nA_tiles_distributed + a_tile) * m
                for phase1_chunk in range(phase1_chunks_per_tile):
                    row_offset = (row_base + phase1_chunk * phase1_rows) * K
                    tap = TensorAccessPattern(
                        (M, K),
                        row_offset,
                        [1, 1, phase1_rows, K],
                        [0, 0, K, 1],
                    )
                    logging.debug(
                        "Phase1 lane %s row_tile=%s chunk=%s tap offset=%s rows=%s",
                        a_tile,
                        row_tile,
                        phase1_chunk,
                        tap.offset,
                        phase1_rows,
                    )
                    rt.fill(
                        phase1_A_l3l2_fifos[a_tile].prod(),
                        A,
                        tap=tap,
                        task_group=phase1_tg,
                        placement=Tile(a_tile, 0),
                    )
                    rt.fill(
                        phase1_R_l3l2_fifos[a_tile].prod(),
                        R,
                        tap=tap,
                        task_group=phase1_tg,
                        placement=Tile(n_aie_cols - 1 - a_tile, 0),
                    )
                    rt.drain(
                        phase1_preadd_l2l3_fifos[a_tile].cons(),
                        stage_stacked,
                        tap=stacked_tap(
                            tap,
                            tensor_dims=stacked_stage_dims,
                            base_offset=stage_preadd_base,
                        ),
                        wait=True,
                        task_group=phase1_tg,
                        placement=Tile(n_aie_cols - 1 - a_tile, 0),
                    )
                    rt.drain(
                        phase1_ln1_l2l3_fifos[a_tile].cons(),
                        stage_stacked,
                        tap=stacked_tap(
                            tap,
                            tensor_dims=stacked_stage_dims,
                            base_offset=stage_ln1_base,
                        ),
                        wait=True,
                        task_group=phase1_tg,
                        placement=Tile(a_tile, 0),
                    )
        rt.finish_task_group(phase1_tg)

        phase2_tg = rt.task_group()
        for row_tile in range(ln_iters_per_core):
            for a_tile in range(nA_tiles_distributed):
                a_offset = (row_tile * nA_tiles_distributed + a_tile) * m * K
                a_tap = TensorAccessPattern(
                    (M, K),
                    offset=a_offset,
                    sizes=[nC_up_col_tiles_per_core, K_div_k, m, k],
                    strides=[0, k, K, 1],
                )
                c_tap = TensorAccessPattern(
                    (M, K),
                    offset=a_offset,
                    sizes=[1, down_proj_depth, m, k],
                    strides=[0, k, K, 1],
                )
                logging.debug(
                    "Phase2 lane %s row_tile=%s A_offset=%s",
                    a_tile,
                    row_tile,
                    a_offset,
                )
                rt.fill(
                    phase2_A_l3l2_fifos[a_tile].prod(),
                    stage_stacked,
                    tap=stacked_tap(
                        a_tap,
                        tensor_dims=stacked_stage_dims,
                        base_offset=stage_ln1_base,
                    ),
                    task_group=phase2_tg,
                    placement=Tile(a_tile, 0),
                )
                rt.fill(
                    phase2_R_l3l2_fifos[a_tile].prod(),
                    stage_stacked,
                    tap=stacked_tap(
                        TensorAccessPattern(
                            (M, K),
                            offset=a_offset,
                            sizes=[2, down_proj_depth, m, k],
                            strides=[0, k, K, 1],
                        ),
                        tensor_dims=stacked_stage_dims,
                        base_offset=stage_preadd_base,
                    ),
                    task_group=phase2_tg,
                    placement=Tile(n_aie_cols - 1 - a_tile, 0),
                )
                rt.drain(
                    ln2_l2l3_fifos[a_tile].cons(),
                    C,
                    tap=c_tap,
                    wait=True,
                    task_group=phase2_tg,
                    placement=Tile(n_aie_cols - 1 - a_tile, 0),
                )

            for b_tile in range(nB_tiles_distributed):
                b_up_tap = TensorAccessPattern(
                    (N, K),
                    offset=b_tile * n,
                    sizes=[nC_up_col_tiles_per_core, K_div_k, k, n],
                    strides=[mem_tile_n, k * N, N, 1],
                )
                rt.fill(
                    B_up_proj_l3l2_fifos[b_tile].prod(),
                    B_stacked,
                    tap=stacked_tap(
                        b_up_tap,
                        tensor_dims=stacked_B_dims,
                        base_offset=B_up_base,
                    ),
                    task_group=phase2_tg,
                    placement=Tile(b_tile + 1, 0),
                )
                b_down_tap = TensorAccessPattern(
                    (K, N),
                    offset=b_tile * n * K,
                    sizes=[nC_up_col_tiles_per_core, down_proj_depth, n, k],
                    strides=[mem_tile_n * K, k, K, 1],
                )
                rt.fill(
                    B_down_proj_l3l2_fifos[b_tile].prod(),
                    B_stacked,
                    tap=stacked_tap(
                        b_down_tap,
                        tensor_dims=stacked_B_dims,
                        base_offset=B_down_base,
                    ),
                    task_group=phase2_tg,
                    placement=Tile(b_tile + 1, 0),
                )
        rt.finish_task_group(phase2_tg)

    my_program = Program(dev_ty, rt)
    return my_program.resolve_program(SequentialPlacer())
