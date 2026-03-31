# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    str_to_dtype,
)
import aie.dialects.index as index
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU1Col2, NPU2, Tile
from aie.iron.placers import SequentialPlacer

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


def fused_addnorm_ffn_addnorm(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    down_proj_depth,
    n_aie_cols,
    nA_tiles_distributed,
    nB_tiles_distributed,
    dtype_in_str,
    dtype_out_str,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    gelu_stage,
    ln1_weight_file,
    ln2_weight_file,
    archive=None,
):
    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    if ln1_weight_file is None or ln2_weight_file is None:
        logging.warning(
            "Layer norm weight files not provided; using default weights of all ones."
        )
        static_ln1_weights = np.ones(K, dtype=bfloat16)
        static_ln2_weights = np.ones(K, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
        static_ln2_weights = np.load(ln2_weight_file)
        if static_ln1_weights.shape[0] != K:
            raise ValueError("Static ln1 weights length does not match K")
        if static_ln2_weights.shape[0] != K:
            raise ValueError("Static ln2 weights length does not match K")

    if n_aie_cols < 2:
        raise AssertionError(
            "n_aie_cols must be at least 2 due to packed AR, B_Up, and B_Down streams"
        )
    compact_seq_layout = nA_tiles_distributed > (n_aie_cols // 2)
    if compact_seq_layout and nB_tiles_distributed != 1:
        raise AssertionError(
            "Compact Block 3 sequence-lane layout currently requires parallel_int_dim == 1"
        )
    horizontal_layout = nA_tiles_distributed < 3 and not compact_seq_layout
    # n_aie_cols will be used to determine whether to send the same data to through different shim tiles, while
    # nB_tiles_distributed will be used to determine what data to send through which shim tiles
    # nA_tiles_distributed replicates the pipelined design across the NPU array
    # There's 2 pipeline stages, and both use the same nB_tiles_distributed parameter since the core fcn loops are the same
    n_aie_rows = 4
    shim_dma_ch_per_col = 2
    cores_per_col = 4
    num_ffn_stages = 2  # Up projection and fused down projection-GeLU stages
    n_aie_cores_needed = nA_tiles_distributed * (
        2 + num_ffn_stages * nB_tiles_distributed
    )  # nA_tiles_distributed duplicates the full Block 3 lane: LN1, FFN, and LN2

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mem_tile_n = n * nB_tiles_distributed
    # Calculate loop bounds for the reduction loop and total C tiles
    K_div_k = K // k
    if K_div_k % down_proj_depth != 0:
        raise AssertionError("K / k must be divisible by down_proj_depth")
    n_col_groups = K_div_k // down_proj_depth
    nC_up_col_tiles_per_core = N // mem_tile_n
    ln_iters_per_core = M // (nA_tiles_distributed * m)
    n_grouped_ffn_sweeps = 2
    nC_tiles_per_core = (
        nC_up_col_tiles_per_core
        * ln_iters_per_core
        * n_col_groups
        * n_grouped_ffn_sweeps
    )

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev == "npu" and n_aie_cores_needed > 16:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 16 cores")
    if dev == "npu" and n_aie_cores_needed > n_aie_cols * 4:
        raise AssertionError(
            f"Invalid configuration: NPU (Phoenix/Hawk) has 4 rows per column, configuring for {n_aie_cores_needed} cores with {n_aie_cols} columns"
        )
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )
    if dev == "npu2" and n_aie_cores_needed > 32:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 32 cores"
        )
    if dev == "npu2" and n_aie_cores_needed > n_aie_cols * 4:
        raise AssertionError(
            f"Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 4 rows per column, configuring for {n_aie_cores_needed} cores with {n_aie_cols} columns"
        )

    # Add & Norm checks:
    if static_ln1_weights.shape[0] != K:
        raise ValueError(
            "Static ln1 weights length does not match the specified weight length"
        )
    if static_ln2_weights.shape[0] != K:
        raise ValueError(
            "Static ln2 weights length does not match the specified weight length"
        )
    assert (
        M % (nA_tiles_distributed * m) == 0
    ), "M must be multiple of nA_tiles_distributed * m"

    # FFN checks:
    # Input matrix A and output matrix C_Down:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * nA_tiles_distributed) == 0
    ), """A and C must be tileable into (m * nA_tiles_distributed, k)-sized blocks for up projection loop"""

    # Inner dimensions for GEMM:
    # Both A and B_Up are tiled in the K dimension into size k.
    assert (
        K % k == 0
    ), """K must be tileable into k-sized blocks for up projection loop"""
    # Both C_Up and B_Down are tiled in the N dimension into size n.
    assert (
        N % n == 0
    ), """N must be tileable into n-sized blocks for down projection loop"""

    # Input matrix B_Up:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % mem_tile_n == 0
    ), """B_Up must be tileable into (k, n * nB_tiles_distributed)-sized blocks"""

    # Intermediate C_Down
    # Conceptually, we divide the C_Down matrix into (m, k * down_proj_depth)-sized blocks.
    # The partial accumulations of C_Down are stored in the Memory tiles with the
    # object FIFO depth based on the down_proj_depth parameter. The nB_tiles_distributed
    # parameter changes how much of the portion of down projection accumulation is done
    # at each of the cores doing the compute. It also decides how many reduction steps
    # have to be taken across these cores to get the final C_Down output.
    assert (
        K % (k * down_proj_depth) == 0
    ), """Partial C_Down must tile equally into (m, K) with (m, k * down_proj_depth)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # Base FIFO depth. Individual compute-tile FIFOs may be reduced to depth 1
    # when their object size would otherwise overrun compute-tile memory.
    fifo_depth = 2

    if dev == "npu":
        if n_aie_cores_needed <= 4:
            dev_ty = NPU1Col1()
        elif n_aie_cores_needed <= 8:
            dev_ty = NPU1Col2()
        elif n_aie_cores_needed <= 16:
            dev_ty = NPU1()
    else:
        dev_ty = NPU2()

    # Define tensor types
    packed_hidden_residual_ty = np.ndarray[(2 * M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * K,), np.dtype[dtype_out]]

    def split_runtime_fill_tap(
        tap: TensorAccessPattern,
        tensor_shape: tuple[int, ...],
        *,
        max_dim_size: int = 64,
        repeat_chunk_size: int | None = None,
    ) -> list[TensorAccessPattern]:
        sizes = [int(size) for size in tap.sizes]
        strides = [int(stride) for stride in tap.strides]
        dim0_size = sizes[0]
        chunk_size = (
            repeat_chunk_size
            if strides[0] == 0 and repeat_chunk_size is not None
            else max_dim_size
        )
        if dim0_size <= chunk_size:
            return [tap]
        if chunk_size > max_dim_size:
            raise ValueError(
                "Cannot split TAP leading dimension: "
                f"size={dim0_size}, chunk_size={chunk_size}"
            )
        taps: list[TensorAccessPattern] = []
        consumed = 0
        while consumed < dim0_size:
            current_chunk_size = min(chunk_size, dim0_size - consumed)
            chunk_sizes = list(sizes)
            chunk_sizes[0] = current_chunk_size
            taps.append(
                TensorAccessPattern(
                    tensor_shape,
                    offset=int(tap.offset) + consumed * strides[0],
                    sizes=chunk_sizes,
                    strides=strides,
                )
            )
            consumed += current_chunk_size
        return taps

    # Add & Norm tensor types
    ln_weights_ty = np.ndarray[(K,), np.dtype[dtype_in]]
    AR_l1_ty = np.ndarray[(2 * m * k,), np.dtype[dtype_in]]

    # GEMM tensor types
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    B_up_proj_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    B_down_proj_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
    C_up_proj_l1_ty = np.ndarray[(m, n), np.dtype[dtype_in]]
    sum_l1_ty = np.ndarray[(m,), np.dtype[str_to_dtype("f32")]]

    dtype_in_bytes = np.dtype(dtype_in).itemsize
    dtype_out_bytes = np.dtype(dtype_out).itemsize

    ar_l1_bytes = 2 * m * k * dtype_in_bytes
    a_l1_bytes = m * k * dtype_in_bytes
    b_up_l1_bytes = k * n * dtype_in_bytes
    b_down_l1_bytes = n * k * dtype_in_bytes
    c_up_l1_bytes = m * n * dtype_in_bytes

    def compute_tile_fifo_depth(object_bytes: int) -> int:
        # Depth-2 double buffering is fine for smaller tiles, but larger tiles
        # can overflow the 64 KiB compute-tile memory once local buffers are
        # included. Keep the memtile-staged partial-accumulation path separate.
        return 1 if object_bytes * fifo_depth >= 16 * 1024 else fifo_depth

    fifo_depth_ar_compute = compute_tile_fifo_depth(ar_l1_bytes)
    fifo_depth_a_compute = compute_tile_fifo_depth(a_l1_bytes)
    fifo_depth_b_up_compute = compute_tile_fifo_depth(b_up_l1_bytes)
    fifo_depth_b_down_compute = compute_tile_fifo_depth(b_down_l1_bytes)
    fifo_depth_c_up_compute = compute_tile_fifo_depth(c_up_l1_bytes)

    # AIE Core Function declarations
    archive_name = f"ffn_{m}x{k}x{n}_archive.a" if archive is None else archive
    # No need to use separate buffers for accumulation and transfer to L2, so
    # we only need the zero and matmul kernels
    fifo_depth_out = fifo_depth_a_compute
    # Up projection
    matmul_func_name = f"ffn_matmul_{dtype_in_str}_{dtype_out_str}"
    ffn_zero_kernel_up_proj = Kernel(
        f"ffn_zero_{dtype_out_str}_up_proj",
        archive_name,
        [C_up_proj_l1_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        matmul_func_name + "_up_proj",
        archive_name,
        [A_l1_ty, B_up_proj_l1_ty, C_up_proj_l1_ty],
    )
    ffn_gelu_kernel = Kernel(
        "ffn_gelu_bf16",
        archive_name,
        [C_up_proj_l1_ty, C_up_proj_l1_ty, np.int32],
    )
    # Down projection
    matmul_func_name = f"ffn_matmul_with_acc_{dtype_in_str}_{dtype_out_str}"
    ffn_zero_kernel_down_proj = Kernel(
        f"ffn_zero_{dtype_out_str}_down_proj",
        archive_name,
        [A_l1_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        matmul_func_name + "_down_proj",
        archive_name,
        [C_up_proj_l1_ty, B_down_proj_l1_ty, A_l1_ty, A_l1_ty],
    )
    ffn_mem_copy_fcn = (
        Kernel(  # Copy fcn for reduction across down proj cores and to ln core
            "ffn_passThroughLine",
            archive_name,
            [A_l1_ty, A_l1_ty, np.int32],
        )
    )
    ffn_eltwise_add_vector = Kernel(
        "ffn_eltwise_add_bf16_vector",
        archive_name,
        [A_l1_ty, A_l1_ty, A_l1_ty, np.int32],
    )
    # Add & Norm
    ln_zero_f32_kernel = Kernel(
        "ln_zero_f32",
        archive_name,
        [sum_l1_ty, np.int32],
    )
    ln_copy_f32_kernel = Kernel(
        "ln_passThroughLine_f32",
        archive_name,
        [sum_l1_ty, sum_l1_ty, np.int32],
    )
    ln_calc_sum_sumsq_kernel = Kernel(
        "ln_calc_sum_sumsq",
        archive_name,
        [A_l1_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_add_calc_sum_sumsq_kernel = Kernel(
        "ln_add_calc_sum_sumsq",
        archive_name,
        [A_l1_ty, A_l1_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_packed_add_calc_sum_sumsq_kernel = Kernel(
        "packed_ln_add_calc_sum_sumsq",
        archive_name,
        [AR_l1_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_fused_add_layer_norm_from_inputs_kernel = Kernel(
        "fused_add_layer_norm_1outs_from_inputs",
        archive_name,
        [
            A_l1_ty,
            A_l1_ty,
            ln_weights_ty,
            sum_l1_ty,
            sum_l1_ty,
            A_l1_ty,
            np.int32,
            np.int32,
        ],
    )
    ln_packed_fused_add_layer_norm_from_inputs_kernel = Kernel(
        "packed_fused_add_layer_norm_1outs_from_inputs",
        archive_name,
        [
            AR_l1_ty,
            ln_weights_ty,
            sum_l1_ty,
            sum_l1_ty,
            A_l1_ty,
            np.int32,
            np.int32,
        ],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    def ln1_stream_tile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(0, 1)
        if compact_seq_layout:
            return Tile(a_tile, 1)
        return Tile(a_tile * 2, 1)

    def ln2_stream_tile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(nB_tiles_distributed + 1, 1)
        if compact_seq_layout:
            return Tile(a_tile, 1)
        return Tile(a_tile * 2 + 1, 1)

    def ln1_shim_tile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(0, 0)
        if compact_seq_layout:
            return Tile(a_tile, 0)
        return Tile(a_tile * 2, 0)

    def ln2_shim_tile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(nB_tiles_distributed + 1, 0)
        if compact_seq_layout:
            return Tile(a_tile, 0)
        return Tile(a_tile * 2 + 1, 0)

    def c_shim_tile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(nB_tiles_distributed + 1, 0)
        if compact_seq_layout:
            return Tile(a_tile, 0)
        return Tile(a_tile * 2 + 1, 0)

    def partial_c_memtile(a_tile: int, b_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(
                (a_tile * nB_tiles_distributed + b_tile + 1) % n_aie_cols,
                1,
            )
        if compact_seq_layout:
            return Tile(a_tile, 1)
        return Tile(
            ((b_tile % 2) + (a_tile * 2)) % n_aie_cols,
            1,
        )

    def ln2_output_memtile(a_tile: int) -> Tile:
        if horizontal_layout:
            return Tile(n_aie_cols - 1 - a_tile, 1)
        if compact_seq_layout:
            return Tile(a_tile, 1)
        return Tile((a_tile * nB_tiles_distributed + 1) % n_aie_cols, 1)

    def worker_tiles(a_tile: int, b_tile: int) -> tuple[Tile, Tile, Tile, Tile]:
        if compact_seq_layout:
            if b_tile != 0:
                raise AssertionError(
                    "Compact Block 3 sequence-lane layout supports only one FFN lane"
                )
            tile_col = a_tile
            return (
                Tile(tile_col, 3),
                Tile(tile_col, 4),
                Tile(tile_col, 2),
                Tile(tile_col, 5),
            )
        if horizontal_layout:
            tile_col, tile_row = core_tiles[a_tile * num_ffn_stages][b_tile]
            return (
                Tile(tile_col, tile_row + 1),
                Tile(tile_col, tile_row),
                Tile(nB_tiles_distributed + 1, tile_row),
                Tile(nB_tiles_distributed, tile_row),
            )
        tile_col, tile_row = core_tiles[-1 * (b_tile + 1)][a_tile * num_ffn_stages]
        return (
            Tile(tile_col, tile_row),
            Tile(tile_col + 1, tile_row),
            Tile(tile_col, tile_row - 1),
            Tile(tile_col + 1, tile_row - 1),
        )

    # AIE-array data movement with object fifos
    # Dedicated packed [A | R] streams for LN1 and LN2. Replay stays encoded
    # in the runtime TAPs; the ingress path is L3 -> L2 -> L1.
    AR_ln1_l3l2_fifos = [None] * nA_tiles_distributed
    AR_ln1_l2l1_fifos = [None] * nA_tiles_distributed
    AR_ln2_l3l2_fifos = [None] * nA_tiles_distributed
    AR_ln2_l2l1_fifos = [None] * nA_tiles_distributed
    dims_to_stream_ar = [
        ((2 * m) // r, r * k),
        (k // s, s),
        (r, k),
        (s, 1),
    ]

    # FFN streams
    # The same data may be sent through different shim tiles depending on the num aie cols available and num b tiles to distribute to reduce routing distance
    B_up_proj_l3l2_fifos = [None] * (nB_tiles_distributed)
    B_up_proj_l2l1_fifos = [None] * (nB_tiles_distributed)
    B_down_proj_l3l2_fifos = [None] * (nB_tiles_distributed)
    B_down_proj_l2l1_fifos = [None] * (nB_tiles_distributed)

    # First AddNorm output pipelined into the up projection workers.
    ln1_stage_l1l1_fifos = [None] * nA_tiles_distributed
    ln1_stage_to_ln2_l1l2_fifos = [None] * nA_tiles_distributed
    ln1_stage_to_ln2_l2l1_fifos = [None] * nA_tiles_distributed

    # C tiles pipelined from up_proj core to down_proj core
    C_up_proj_l1l1_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]

    # Partial C tiles for accumulation between down_proj core and mem tile
    C_down_proj_part_l1l2_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_part_l2l1_fifos = [
        [None] * nB_tiles_distributed for _ in range(nA_tiles_distributed)
    ]

    # Output C tiles from down_proj core
    C_down_proj_reduce_l1l1_fifos = [
        [None] * (nB_tiles_distributed - 1) for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_out_l1l1_fifos = [None] * nA_tiles_distributed
    # Output tiles for second Add & Norm
    ln2_l1l2_fifos = [None] * nA_tiles_distributed
    ln2_l2l3_fifos = [None] * nA_tiles_distributed

    # Dedicated packed [A | R] streams for LN1 and LN2 workers.
    for a_tile in range(nA_tiles_distributed):
        AR_ln1_l3l2_fifos[a_tile] = ObjectFifo(
            AR_l1_ty,
            name=f"AR_ln1_L3L2_{a_tile}",
            depth=fifo_depth,
        )
        AR_ln1_l2l1_fifos[a_tile] = (
            AR_ln1_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=AR_l1_ty,
                name=f"AR_ln1_L2L1_{a_tile}",
                depth=fifo_depth_ar_compute,
                dims_to_stream=dims_to_stream_ar,
                placement=ln1_stream_tile(a_tile),
            )
        )
        if not compact_seq_layout:
            AR_ln2_l3l2_fifos[a_tile] = ObjectFifo(
                AR_l1_ty,
                name=f"AR_ln2_L3L2_{a_tile}",
                depth=fifo_depth,
            )
            AR_ln2_l2l1_fifos[a_tile] = (
                AR_ln2_l3l2_fifos[a_tile]
                .cons()
                .forward(
                    obj_type=AR_l1_ty,
                    name=f"AR_ln2_L2L1_{a_tile}",
                    depth=fifo_depth_ar_compute,
                    dims_to_stream=dims_to_stream_ar,
                    placement=ln2_stream_tile(a_tile),
                )
            )
    # Input B_Up
    for b_tile in range(nB_tiles_distributed):
        B_up_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_up_L3L2_{b_tile}", depth=fifo_depth
        )
        dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_up_proj_l2l1_fifos[b_tile] = (
            B_up_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_up_proj_l1_ty,
                name=f"B_up_L2L1_{b_tile}",
                depth=fifo_depth_b_up_compute,
                dims_to_stream=dims_to_stream,
                placement=(
                    Tile((b_tile + 2) % n_aie_cols, 1)
                    if nA_tiles_distributed < 3
                    else Tile(
                        (
                            b_tile * nB_tiles_distributed
                            + n_aie_cols // nB_tiles_distributed
                        )
                        % n_aie_cols,
                        1,
                    )
                ),  # Switch between up and down proj B tile streams across shim tiles
            )
        )

    # Input B_Down: n and k are swapped compared to B_Up
    for b_tile in range(nB_tiles_distributed):
        B_down_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_down_L3L2_{b_tile}", depth=fifo_depth
        )
        dims_to_stream = [(n // s, s * k), (k // t, t), (s, k), (t, 1)]
        B_down_proj_l2l1_fifos[b_tile] = (
            B_down_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_down_proj_l1_ty,
                name=f"B_down_L2L1_{b_tile}",
                depth=fifo_depth_b_down_compute,
                dims_to_stream=dims_to_stream,
                placement=(
                    Tile((b_tile + 2) % n_aie_cols, 1)
                    if nA_tiles_distributed < 3
                    else Tile(
                        (
                            b_tile * nB_tiles_distributed
                            + n_aie_cols // nB_tiles_distributed
                            + 1
                        )
                        % n_aie_cols,
                        1,
                    )
                ),  # Switch between up and down proj B tile streams across shim tiles
            )
        )

    # Up proj C
    for a_tile in range(nA_tiles_distributed):
        ln1_stage_l1l1_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"ln1_stage_L1L1_{a_tile}",
            depth=fifo_depth_a_compute,
        )
        if compact_seq_layout:
            ln1_stage_to_ln2_l1l2_fifos[a_tile] = ObjectFifo(
                A_l1_ty,
                name=f"ln1_stage_to_ln2_L1L2_{a_tile}",
                depth=1,
            )
            ln1_stage_to_ln2_l2l1_fifos[a_tile] = (
                ln1_stage_to_ln2_l1l2_fifos[a_tile]
                .cons(depth=down_proj_depth)
                .forward(
                    obj_type=A_l1_ty,
                    name=f"ln1_stage_to_ln2_L2L1_{a_tile}",
                    depth=down_proj_depth,
                    placement=ln2_output_memtile(a_tile),
                )
            )

    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            C_up_proj_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_up_proj_l1_ty,
                name=f"C_up_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth_c_up_compute,
            )

    # Down proj partial C
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            # Per MT, at most 2 of these objfifos can be connected considering other streams and that the max is 6 S2MM/MM2S per MT
            max_obfifos_per_mt = 2
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
                    placement=partial_c_memtile(a_tile, b_tile),
                )
            )
    # Down proj partial C for reduction
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed - 1):
            C_down_proj_reduce_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                A_l1_ty,
                name=f"C_down_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth_a_compute,
            )

    # Down proj output C, m-by-k tiles
    # NOTE: Can't undo the microtiles like with softmax in MHA output projection
    # pipeline because it would use a DMA channel, and one is needed for residual
    # connection, and another for the stream from MT back to core
    for a_tile in range(nA_tiles_distributed):
        C_down_proj_out_l1l1_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"C_out_L1L1_{a_tile}",
            depth=fifo_depth_out,
        )

    # Second Add & Norm output streams
    for a_tile in range(nA_tiles_distributed):
        dims_to_stream = [(m // r, r * k), (r, s), (k // s, r * s), (s, 1)]
        ln2_l1l2_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"ln2_L1L2_{a_tile}",
            depth=fifo_depth_a_compute,
        )
        ln2_l2l3_fifos[a_tile] = (
            ln2_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"ln2_L2L3_{a_tile}",
                dims_to_stream=dims_to_stream,
                placement=ln2_output_memtile(a_tile),
            )
        )

    # Tasks for each worker to perform
    def core_fn_add_norm1(
        in_ar,
        out_stage1,
        sum_buf,
        sumsq_buf,
        weights,
        zero_f32,
        packed_add_calc_sum_sumsq,
        packed_fused_add_layer_norm_from_inputs,
    ):
        loop = range_(1)
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for _ in loop:
            zero_f32(sum_buf, m)
            zero_f32(sumsq_buf, m)
            for _ in range_(K_div_k):
                elem_in_ar = in_ar.acquire(1)
                packed_add_calc_sum_sumsq(elem_in_ar, sum_buf, sumsq_buf)
                in_ar.release(1)

            for col_idx in range_(K_div_k):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in_ar = in_ar.acquire(1)
                elem_out_stage1 = out_stage1.acquire(1)
                packed_fused_add_layer_norm_from_inputs(
                    elem_in_ar,
                    weights,
                    sum_buf,
                    sumsq_buf,
                    elem_out_stage1,
                    K,
                    col_i32,
                )
                out_stage1.release(1)
                in_ar.release(1)

    def core_fn_add_norm1_compact(
        in_ar,
        out_stage1,
        out_stage1_to_ln2,
        sum_buf,
        sumsq_buf,
        weights,
        zero_f32,
        copy,
        packed_add_calc_sum_sumsq,
        packed_fused_add_layer_norm_from_inputs,
    ):
        down_proj_depth_idx = index.constant(down_proj_depth)
        k_div_k_idx = index.constant(K_div_k)

        def compute_stats():
            zero_f32(sum_buf, m)
            zero_f32(sumsq_buf, m)
            for _ in range_(K_div_k):
                elem_in_ar = in_ar.acquire(1)
                packed_add_calc_sum_sumsq(elem_in_ar, sum_buf, sumsq_buf)
                in_ar.release(1)

        def emit_stage1_tile(col_idx, *, copy_to_ln2: bool):
            col_i32 = index.casts(T.i32(), col_idx)
            elem_in_ar = in_ar.acquire(1)
            elem_out_stage1 = out_stage1.acquire(1)
            packed_fused_add_layer_norm_from_inputs(
                elem_in_ar,
                weights,
                sum_buf,
                sumsq_buf,
                elem_out_stage1,
                K,
                col_i32,
            )
            if copy_to_ln2:
                elem_out_stage1_to_ln2 = out_stage1_to_ln2.acquire(1)
                copy(elem_out_stage1, elem_out_stage1_to_ln2, m * k)
                out_stage1_to_ln2.release(1)
            out_stage1.release(1)
            in_ar.release(1)

        def compute_output_full():
            compute_stats()
            for col_idx in range_(K_div_k):
                emit_stage1_tile(col_idx, copy_to_ln2=False)

        def compute_output_group(copy_group_base):
            compute_stats()
            for col_idx in range_(copy_group_base):
                emit_stage1_tile(col_idx, copy_to_ln2=False)
            for local_col_idx in range_(down_proj_depth):
                col_idx = index.add(copy_group_base, local_col_idx)
                emit_stage1_tile(col_idx, copy_to_ln2=True)
            suffix = index.sub(
                k_div_k_idx,
                index.add(copy_group_base, down_proj_depth_idx),
            )
            for local_col_idx in range_(suffix):
                col_idx = index.add(
                    index.add(copy_group_base, down_proj_depth_idx),
                    local_col_idx,
                )
                emit_stage1_tile(col_idx, copy_to_ln2=False)

        for _ in range_(ln_iters_per_core):
            for _ in range_(n_grouped_ffn_sweeps):
                for col_group_idx in range_(n_col_groups):
                    group_base = index.mul(col_group_idx, down_proj_depth_idx)
                    compute_output_group(group_base)
                    if nC_up_col_tiles_per_core > 1:
                        for _ in range_(nC_up_col_tiles_per_core - 1):
                            compute_output_full()

    def core_fn_up_proj(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        gelu,
    ):
        loop = range_(1)
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for _ in loop:
            elem_out_matmul = out_c.acquire(1)
            zero(elem_out_matmul)
            for _ in range_(K_div_k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out_matmul)
                in_a.release(1)
                in_b.release(1)
            if gelu:
                gelu(elem_out_matmul, elem_out_matmul, m * n)
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
    ):
        for _ in range_(ln_iters_per_core * n_col_groups * n_grouped_ffn_sweeps):
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
                if buffer_to_reduce:
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    add(partial_acc_c, elem_out_internal, elem_out_internal, m * k)
                    buffer_to_reduce.release(1)
                elem_out_acc_c = out_acc_c.acquire(1)
                copy(elem_out_internal, elem_out_acc_c, m * k)
                out_acc_c.release(1)
                curr_acc_c.release(1)

    def core_fn_add_norm2(
        in_ar,
        in_down,
        stage1_buf,
        ln1_sum_buf,
        ln1_sumsq_buf,
        ln1_weights,
        ln2_sum_buf,
        ln2_sumsq_buf,
        ln2_weights,
        out_ln2,
        add_calc_sum_sumsq,
        fused_add_layer_norm_from_inputs,
        packed_fused_add_layer_norm_from_inputs,
        packed_add_calc_sum_sumsq,
        zero_f32,
    ):
        def compute_ln1_prepass():
            zero_f32(ln1_sum_buf, m)
            zero_f32(ln1_sumsq_buf, m)
            for _ in range_(K_div_k):
                elem_in_ar = in_ar.acquire(1)
                packed_add_calc_sum_sumsq(elem_in_ar, ln1_sum_buf, ln1_sumsq_buf)
                in_ar.release(1)

        def compute_group_stats(group_base):
            for local_col_idx in range_(down_proj_depth):
                col_idx = group_base + local_col_idx
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in_ar = in_ar.acquire(1)
                elem_in_down = in_down.acquire(1)
                packed_fused_add_layer_norm_from_inputs(
                    elem_in_ar,
                    ln1_weights,
                    ln1_sum_buf,
                    ln1_sumsq_buf,
                    stage1_buf,
                    K,
                    col_i32,
                )
                add_calc_sum_sumsq(
                    elem_in_down,
                    stage1_buf,
                    ln2_sum_buf,
                    ln2_sumsq_buf,
                )
                in_down.release(1)
                in_ar.release(1)

        def compute_group_output(group_base):
            for local_col_idx in range_(down_proj_depth):
                col_idx = group_base + local_col_idx
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in_ar = in_ar.acquire(1)
                elem_in_down = in_down.acquire(1)
                elem_out_ln2 = out_ln2.acquire(1)
                packed_fused_add_layer_norm_from_inputs(
                    elem_in_ar,
                    ln1_weights,
                    ln1_sum_buf,
                    ln1_sumsq_buf,
                    stage1_buf,
                    K,
                    col_i32,
                )
                fused_add_layer_norm_from_inputs(
                    elem_in_down,
                    stage1_buf,
                    ln2_weights,
                    ln2_sum_buf,
                    ln2_sumsq_buf,
                    elem_out_ln2,
                    K,
                    col_i32,
                )
                out_ln2.release(1)
                in_down.release(1)
                in_ar.release(1)

        for _ in range_(ln_iters_per_core):
            compute_ln1_prepass()
            zero_f32(ln2_sum_buf, m)
            zero_f32(ln2_sumsq_buf, m)
            for col_group_idx in range_(n_col_groups):
                compute_group_stats(col_group_idx * down_proj_depth)
            for col_group_idx in range_(n_col_groups):
                compute_group_output(col_group_idx * down_proj_depth)

    def core_fn_add_norm2_from_stage1(
        in_stage1,
        in_down,
        ln2_sum_buf,
        ln2_sumsq_buf,
        ln2_weights,
        out_ln2,
        add_calc_sum_sumsq,
        fused_add_layer_norm_from_inputs,
        zero_f32,
    ):
        def compute_group_stats():
            for _ in range_(down_proj_depth):
                elem_stage1 = in_stage1.acquire(1)
                elem_in_down = in_down.acquire(1)
                add_calc_sum_sumsq(
                    elem_in_down,
                    elem_stage1,
                    ln2_sum_buf,
                    ln2_sumsq_buf,
                )
                in_down.release(1)
                in_stage1.release(1)

        def compute_group_output(group_base):
            for local_col_idx in range_(down_proj_depth):
                col_idx = index.add(group_base, local_col_idx)
                col_i32 = index.casts(T.i32(), col_idx)
                elem_stage1 = in_stage1.acquire(1)
                elem_in_down = in_down.acquire(1)
                elem_out_ln2 = out_ln2.acquire(1)
                fused_add_layer_norm_from_inputs(
                    elem_in_down,
                    elem_stage1,
                    ln2_weights,
                    ln2_sum_buf,
                    ln2_sumsq_buf,
                    elem_out_ln2,
                    K,
                    col_i32,
                )
                out_ln2.release(1)
                in_down.release(1)
                in_stage1.release(1)

        for _ in range_(ln_iters_per_core):
            zero_f32(ln2_sum_buf, m)
            zero_f32(ln2_sumsq_buf, m)
            for col_group_idx in range_(n_col_groups):
                compute_group_stats()
            for col_group_idx in range_(n_col_groups):
                compute_group_output(
                    index.mul(col_group_idx, index.constant(down_proj_depth))
                )

    # Set up compute tiles
    workers = []
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            up_tile, down_tile, ln1_tile, ln2_tile = worker_tiles(a_tile, b_tile)
            stream_to_ln = b_tile == nB_tiles_distributed - 1
            if stream_to_ln:
                ln1_weight_buffer = Buffer(
                    type=ln_weights_ty,
                    initial_value=static_ln1_weights,
                    name=f"static_ln1_weights_{a_tile}",
                )
                ln1_sum_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"ln1_sum_buffer_{a_tile}",
                )
                ln1_sumsq_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"ln1_sumsq_buffer_{a_tile}",
                )
                workers.append(
                    Worker(
                        (
                            core_fn_add_norm1_compact
                            if compact_seq_layout
                            else core_fn_add_norm1
                        ),
                        (
                            [
                                AR_ln1_l2l1_fifos[a_tile].cons(),
                                ln1_stage_l1l1_fifos[a_tile].prod(),
                                ln1_stage_to_ln2_l1l2_fifos[a_tile].prod(),
                                ln1_sum_buffer,
                                ln1_sumsq_buffer,
                                ln1_weight_buffer,
                                ln_zero_f32_kernel,
                                ffn_mem_copy_fcn,
                                ln_packed_add_calc_sum_sumsq_kernel,
                                ln_packed_fused_add_layer_norm_from_inputs_kernel,
                            ]
                            if compact_seq_layout
                            else [
                                AR_ln1_l2l1_fifos[a_tile].cons(),
                                ln1_stage_l1l1_fifos[a_tile].prod(),
                                ln1_sum_buffer,
                                ln1_sumsq_buffer,
                                ln1_weight_buffer,
                                ln_zero_f32_kernel,
                                ln_packed_add_calc_sum_sumsq_kernel,
                                ln_packed_fused_add_layer_norm_from_inputs_kernel,
                            ]
                        ),
                        placement=ln1_tile,
                        stack_size=0xD00,
                    )
                )
                if not compact_seq_layout:
                    ln2_stage1_buffer = Buffer(
                        type=A_l1_ty,
                        name=f"ln2_stage1_buffer_{a_tile}",
                    )
                    ln2_stage1_weight_buffer = Buffer(
                        type=ln_weights_ty,
                        initial_value=static_ln1_weights,
                        name=f"static_ln1_weights_ln2_{a_tile}",
                    )
                    ln2_stage1_sum_buffer = Buffer(
                        type=sum_l1_ty,
                        name=f"ln2_stage1_sum_buffer_{a_tile}",
                    )
                    ln2_stage1_sumsq_buffer = Buffer(
                        type=sum_l1_ty,
                        name=f"ln2_stage1_sumsq_buffer_{a_tile}",
                    )
                ln2_weight_buffer = Buffer(
                    type=ln_weights_ty,
                    initial_value=static_ln2_weights,
                    name=f"static_ln2_weights_{a_tile}",
                )
                sum_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"ln2_sum_buffer_{a_tile}",
                )
                sumsq_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"ln2_sumsq_buffer_{a_tile}",
                )
                workers.append(
                    Worker(
                        (
                            core_fn_add_norm2_from_stage1
                            if compact_seq_layout
                            else core_fn_add_norm2
                        ),
                        (
                            [
                                ln1_stage_to_ln2_l2l1_fifos[a_tile].cons(),
                                C_down_proj_out_l1l1_fifos[a_tile].cons(),
                                sum_buffer,
                                sumsq_buffer,
                                ln2_weight_buffer,
                                ln2_l1l2_fifos[a_tile].prod(),
                                ln_add_calc_sum_sumsq_kernel,
                                ln_fused_add_layer_norm_from_inputs_kernel,
                                ln_zero_f32_kernel,
                            ]
                            if compact_seq_layout
                            else [
                                AR_ln2_l2l1_fifos[a_tile].cons(),
                                C_down_proj_out_l1l1_fifos[a_tile].cons(),
                                ln2_stage1_buffer,
                                ln2_stage1_sum_buffer,
                                ln2_stage1_sumsq_buffer,
                                ln2_stage1_weight_buffer,
                                sum_buffer,
                                sumsq_buffer,
                                ln2_weight_buffer,
                                ln2_l1l2_fifos[a_tile].prod(),
                                ln_add_calc_sum_sumsq_kernel,
                                ln_fused_add_layer_norm_from_inputs_kernel,
                                ln_packed_fused_add_layer_norm_from_inputs_kernel,
                                ln_packed_add_calc_sum_sumsq_kernel,
                                ln_zero_f32_kernel,
                            ]
                        ),
                        placement=ln2_tile,
                        stack_size=0xD00,
                    )
                )

            # Up projection stage.
            workers.append(
                Worker(
                    core_fn_up_proj,
                    [
                        ln1_stage_l1l1_fifos[a_tile].cons(),
                        B_up_proj_l2l1_fifos[b_tile].cons(),
                        C_up_proj_l1l1_fifos[a_tile][b_tile].prod(),
                        ffn_zero_kernel_up_proj,
                        ffn_matmul_kernel_up_proj,
                        ffn_gelu_kernel if gelu_stage == 0 else None,
                    ],
                    placement=up_tile,
                    stack_size=0xD00,
                )
            )
            # Down projection stage
            # Arguments at position 4, 8, 9, 11 below all relate to how the reduction across down projection
            # cores is performed
            workers.append(
                Worker(
                    core_fn_down_proj,
                    [
                        C_up_proj_l1l1_fifos[a_tile][b_tile].cons(),
                        B_down_proj_l2l1_fifos[b_tile].cons(),
                        C_down_proj_part_l2l1_fifos[a_tile][b_tile].cons(depth=1),
                        C_down_proj_part_l1l2_fifos[a_tile][b_tile].prod(),
                        (
                            C_down_proj_out_l1l1_fifos[a_tile].prod()
                            if stream_to_ln
                            else C_down_proj_reduce_l1l1_fifos[a_tile][b_tile].prod()
                        ),
                        ffn_zero_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        ffn_eltwise_add_vector,
                        ffn_mem_copy_fcn,
                        ffn_gelu_kernel if gelu_stage == 1 else None,
                        (
                            None
                            if b_tile == 0
                            else C_down_proj_reduce_l1l1_fifos[a_tile][
                                b_tile - 1
                            ].cons()
                        ),
                        stream_to_ln,
                    ],
                    placement=down_tile,
                    stack_size=0xD00,
                )
            )

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(packed_hidden_residual_ty, B_ty, B_ty, C_ty) as (
        packed_hidden_residual,
        B_Up,
        B_Down,
        C,
    ):
        rt.start(*workers)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        packed_hidden_residual_shape = (2 * M * K,)
        packed_tile_elems = 2 * m * k
        packed_fill_chunk_size = 64
        tg = rt.task_group()

        def emit_fill_with_task_group_reuse(
            fifo,
            tensor,
            fill_taps: list[TensorAccessPattern],
            *,
            placement: Tile,
        ):
            nonlocal tg
            if len(fill_taps) == 1:
                rt.fill(
                    fifo.prod(),
                    tensor,
                    tap=fill_taps[0],
                    task_group=tg,
                    placement=placement,
                )
                return
            for fill_tap in fill_taps:
                split_tg = rt.task_group()
                rt.fill(
                    fifo.prod(),
                    tensor,
                    tap=fill_tap,
                    wait=True,
                    task_group=split_tg,
                    placement=placement,
                )
                rt.finish_task_group(split_tg)

        def emit_ln2_prepass(row_tile_idx: int):
            nonlocal tg
            if compact_seq_layout:
                return
            for a_tile in range(nA_tiles_distributed):
                packed_tile_offset = (
                    (row_tile_idx * nA_tiles_distributed + a_tile)
                    * K_div_k
                    * packed_tile_elems
                )
                ln2_ar_tile = TensorAccessPattern(
                    packed_hidden_residual_shape,
                    offset=packed_tile_offset,
                    sizes=[1, K_div_k, 2 * m, k],
                    strides=[0, packed_tile_elems, k, 1],
                )
                ln2_place = ln2_shim_tile(a_tile)
                ln2_ar_fill_taps = split_runtime_fill_tap(
                    ln2_ar_tile,
                    packed_hidden_residual_shape,
                    repeat_chunk_size=packed_fill_chunk_size,
                )
                for ln2_ar_fill_tap in ln2_ar_fill_taps:
                    rt.fill(
                        AR_ln2_l3l2_fifos[a_tile].prod(),
                        packed_hidden_residual,
                        tap=ln2_ar_fill_tap,
                        wait=True,
                        task_group=tg,
                        placement=ln2_place,
                    )
            rt.finish_task_group(tg)
            tg = rt.task_group()

        def emit_grouped_phase(row_tile_idx: int, col_group: int, *, emit_output: bool):
            nonlocal tg
            row_packed_tile_offsets = [
                (row_tile_idx * nA_tiles_distributed + a_tile)
                * K_div_k
                * packed_tile_elems
                for a_tile in range(nA_tiles_distributed)
            ]
            if not compact_seq_layout:
                for a_tile in range(nA_tiles_distributed):
                    packed_tile_offset = (
                        row_packed_tile_offsets[a_tile]
                        + col_group * down_proj_depth * packed_tile_elems
                    )
                    ln2_ar_tile = TensorAccessPattern(
                        packed_hidden_residual_shape,
                        offset=packed_tile_offset,
                        sizes=[1, down_proj_depth, 2 * m, k],
                        strides=[0, packed_tile_elems, k, 1],
                    )
                    ln2_place = ln2_shim_tile(a_tile)
                    ln2_ar_fill_taps = split_runtime_fill_tap(
                        ln2_ar_tile,
                        packed_hidden_residual_shape,
                        repeat_chunk_size=packed_fill_chunk_size,
                    )
                    for ln2_ar_fill_tap in ln2_ar_fill_taps:
                        rt.fill(
                            AR_ln2_l3l2_fifos[a_tile].prod(),
                            packed_hidden_residual,
                            tap=ln2_ar_fill_tap,
                            wait=True,
                            task_group=tg,
                            placement=ln2_place,
                        )

            for col_tile_base in range(
                0, nC_up_col_tiles_per_core, packed_fill_chunk_size
            ):
                current_col_tile_block = min(
                    packed_fill_chunk_size,
                    nC_up_col_tiles_per_core - col_tile_base,
                )
                block_tg = rt.task_group()
                for a_tile in range(nA_tiles_distributed):
                    ln1_ar_tile = TensorAccessPattern(
                        packed_hidden_residual_shape,
                        offset=row_packed_tile_offsets[a_tile],
                        sizes=[2 * current_col_tile_block, K_div_k, 2 * m, k],
                        strides=[0, packed_tile_elems, k, 1],
                    )
                    place = ln1_shim_tile(a_tile)
                    ln1_ar_fill_taps = split_runtime_fill_tap(
                        ln1_ar_tile,
                        packed_hidden_residual_shape,
                        repeat_chunk_size=packed_fill_chunk_size,
                    )
                    for ln1_ar_fill_tap in ln1_ar_fill_taps:
                        rt.fill(
                            AR_ln1_l3l2_fifos[a_tile].prod(),
                            packed_hidden_residual,
                            tap=ln1_ar_fill_tap,
                            wait=True,
                            task_group=block_tg,
                            placement=place,
                        )

                for b_tile in range(nB_tiles_distributed):
                    B_up_proj_col_offset = b_tile * n + col_tile_base * mem_tile_n
                    B_up_proj_sizes = [current_col_tile_block, K_div_k, k, n]
                    B_up_proj_strides = [mem_tile_n, k * N, N, 1]
                    B_up_proj_tile = TensorAccessPattern(
                        (N, K),
                        offset=B_up_proj_col_offset,
                        sizes=B_up_proj_sizes,
                        strides=B_up_proj_strides,
                    )
                    place = (
                        Tile(b_tile + 1, 0)
                        if nA_tiles_distributed < 3
                        else Tile(b_tile * 2, 0)
                    )
                    rt.fill(
                        B_up_proj_l3l2_fifos[b_tile].prod(),
                        B_Up,
                        tap=B_up_proj_tile,
                        task_group=block_tg,
                        placement=place,
                    )

                    B_down_proj_col_offset = (
                        b_tile * n * K
                        + col_group * k * down_proj_depth
                        + col_tile_base * mem_tile_n * K
                    )
                    B_down_proj_sizes = [
                        current_col_tile_block,
                        down_proj_depth,
                        n,
                        k,
                    ]
                    B_down_proj_strides = [mem_tile_n * K, k, K, 1]
                    B_down_proj_tile = TensorAccessPattern(
                        (K, N),
                        offset=B_down_proj_col_offset,
                        sizes=B_down_proj_sizes,
                        strides=B_down_proj_strides,
                    )
                    place = (
                        Tile(b_tile + 1, 0)
                        if nA_tiles_distributed < 3
                        else Tile(b_tile * 2 + 1, 0)
                    )
                    rt.fill(
                        B_down_proj_l3l2_fifos[b_tile].prod(),
                        B_Down,
                        tap=B_down_proj_tile,
                        task_group=block_tg,
                        placement=place,
                    )
                rt.finish_task_group(block_tg)

            for a_tile in range(nA_tiles_distributed):
                if emit_output:
                    c_place = c_shim_tile(a_tile)
                    c_offset = (
                        row_tile_idx * nA_tiles_distributed + a_tile
                    ) * m * K + col_group * k * down_proj_depth
                    C_tile = TensorAccessPattern(
                        (M, K),
                        offset=c_offset,
                        sizes=[1, down_proj_depth, m, k],
                        strides=[0, k, K, 1],
                    )
                    rt.drain(
                        ln2_l2l3_fifos[a_tile].cons(),
                        C,
                        tap=C_tile,
                        wait=True,
                        task_group=tg,
                        placement=c_place,
                    )
            rt.finish_task_group(tg)
            tg = rt.task_group()

        for row_tile_idx in range(ln_iters_per_core):
            emit_ln2_prepass(row_tile_idx)
            for col_group in range(n_col_groups):
                emit_grouped_phase(row_tile_idx, col_group, emit_output=False)
            for col_group in range(n_col_groups):
                emit_grouped_phase(row_tile_idx, col_group, emit_output=True)
        rt.finish_task_group(tg)

    # Create the program from the device type and runtime
    my_program = Program(dev_ty, rt)
    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module
