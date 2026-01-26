# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path

import numpy as np
import argparse
import sys
import logging

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Buffer,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    str_to_dtype,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2, Tile
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_


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


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=768)
    argparser.add_argument("-N", type=int, default=3072)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=48)
    argparser.add_argument("-n", type=int, default=96)
    argparser.add_argument("--down-proj-depth", type=int, default=1)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    argparser.add_argument("--n-a-tiles-distributed", type=int, default=8)
    argparser.add_argument("--n-b-tiles-distributed", type=int, default=2)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    # Whether to use the scalar kernel; this is low, but can be useful for debugging smaller sizes
    argparser.add_argument("--scalar", type=bool, choices=[0, 1], default=0)
    argparser.add_argument(
        "--emulate-bf16-mmul-with-bfp16", action="store_true", default=True
    )
    argparser.add_argument("--prio-accuracy", action="store_true", default=False)
    argparser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Name of the archive file for the AIE kernels",
    )
    argparser.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "f32"],
        default="bf16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--stage-only", type=int, choices=[0, 1], default=None)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )

    args = argparser.parse_args()
    maybe_module = my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.down_proj_depth,
        args.n_aie_cols,
        args.n_a_tiles_distributed,
        args.n_b_tiles_distributed,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.c_col_maj,
        args.scalar,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.stage_only,
        args.archive,
        args.generate_taps,
    )

    if args.generate_taps:
        return maybe_module
    else:
        output_file_path = Path(args.output_file_path)

        with open(output_file_path, "w") as f:
            f.write(str(maybe_module))


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    down_proj_depth,
    n_aie_cols,
    n_a_tiles_distributed,
    n_b_tiles_distributed,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    use_scalar,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    stage_only=None,
    archive=None,
    generate_taps=False,
):
    if n_aie_cols < 2:
        raise AssertionError(
            "n_aie_cols must be at least 2 due to 3 inputs (A, B_Up, B_Down)"
        )
    # n_aie_cols will be used to determine whether to send the same data to through different shim tiles, while
    # n_b_tiles_distributed will be used to determine what data to send through which shim tiles
    # n_a_tiles_distributed replicates the pipelined design across the NPU array
    # There's 2 pipeline stages, and both use the same n_b_tiles_distributed parameter since the core fcn loops are the same
    shim_dma_ch_per_col = 2
    cores_per_col = 4
    num_pipeline_stages = 2  # Fused Up projection-GeLU and down projection stages
    n_aie_cores_needed = n_a_tiles_distributed * (
        num_pipeline_stages * n_b_tiles_distributed
    )  # 2 stages with a separate B input for each stage
    n_dup_shim_b_streams = (
        n_aie_cols * shim_dma_ch_per_col - n_a_tiles_distributed
    ) // (
        n_b_tiles_distributed * num_pipeline_stages
    )  # 1 means no duplication, 2 means each B stream is duplicated once through another shim DMA channel, etc.
    if n_dup_shim_b_streams < 1:
        raise AssertionError(
            f"Not enough AIE columns to distribute the A and B tiles needed (1 shim DMA channel * {n_a_tiles_distributed} for A and 1 shim DMA channel * {n_b_tiles_distributed} * {num_pipeline_stages} for B per pipeline stage)"
        )
    elif n_dup_shim_b_streams > n_a_tiles_distributed:
        n_dup_shim_b_streams = n_a_tiles_distributed

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mem_tile_n = n * n_b_tiles_distributed
    logging.debug(
        f"n_aie_cores_needed:{n_aie_cores_needed}, n_dup_shim_b_streams:{n_dup_shim_b_streams}, mem_tile_n:{mem_tile_n}"
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
            "Invalid configuration: NPU (Phoenix/Hawk) has 4 rows per column"
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
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 4 rows per column"
        )

    # Input matrix A and output matrix C_Down:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_a_tiles_distributed) == 0
    ), """A and C must be tileable into (m * n_a_tiles_distributed, k)-sized blocks for up projection loop"""

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
    ), """B_Up must be tileable into (k, n * n_b_tiles_distributed)-sized blocks"""

    # Intermediate C_Down
    # Conceptually, we divide the C_Down matrix into (m, k * down_proj_depth)-sized blocks.
    # The partial accumulations of C_Down are stored in the Memory tiles with the
    # object FIFO depth based on the down_proj_depth parameter.
    assert (
        K % (k * down_proj_depth) == 0
    ), """Partial C_Down must be tileable into (m, k * down_proj_depth)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    if not use_scalar:
        assert m % r == 0
        assert k % s == 0
        assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
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

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_up_proj_taps = []
    B_down_proj_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * K,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_up_proj_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    B_down_proj_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
    C_up_proj_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    C_down_proj_l1_ty = np.ndarray[(m, k), np.dtype[dtype_out]]

    # AIE Core Function declarations
    scalar_suffix = "_scalar" if use_scalar else ""
    archive_name = f"ffn_{m}x{k}x{n}_archive.a" if archive is None else archive
    # No need to use separate buffers for accumulation and transfer to L2, so
    # we only need the zero and matmul kernels
    fifo_depth_out = fifo_depth
    # Up projection
    matmul_func_name = f"matmul{scalar_suffix}_{dtype_in_str}_{dtype_out_str}"
    zero_kernel_up_proj = Kernel(
        f"zero{scalar_suffix}_{dtype_out_str}_up_proj",
        archive_name,
        [C_up_proj_l1_ty],
    )
    matmul_kernel_up_proj = Kernel(
        matmul_func_name + "_up_proj",
        archive_name,
        [A_l1_ty, B_up_proj_l1_ty, C_up_proj_l1_ty],
    )
    gelu_kernel = Kernel(
        "gelu_bf16",
        archive_name,
        [C_up_proj_l1_ty, C_up_proj_l1_ty, np.int32],
    )
    # Down projection
    matmul_func_name = f"matmul_with_acc_{dtype_in_str}_{dtype_out_str}"
    zero_kernel_down_proj = Kernel(
        f"zero{scalar_suffix}_{dtype_out_str}_down_proj",
        archive_name,
        [C_down_proj_l1_ty],
    )
    matmul_kernel_down_proj = Kernel(
        matmul_func_name + "_down_proj",
        archive_name,
        [C_up_proj_l1_ty, B_down_proj_l1_ty, C_down_proj_l1_ty, C_down_proj_l1_ty],
    )
    mem_copy_fcn = Kernel(
        "passThroughLine",
        archive_name,
        [C_down_proj_l1_ty, C_down_proj_l1_ty, np.int32],
    )
    eltwise_add_vector = Kernel(
        "eltwise_add_bf16_vector",
        archive_name,
        [C_down_proj_l1_ty, C_down_proj_l1_ty, C_down_proj_l1_ty, np.int32],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_a_tiles_distributed
    A_l2l1_fifos = [None] * n_a_tiles_distributed
    logging.debug(
        f"Len A_l2l1_fifos: {len(A_l2l1_fifos)} len A_l3l2_fifos: {len(A_l3l2_fifos)}"
    )

    # The same data may be sent through different shim tiles depending on the num aie cols available and num b tiles to distribute to reduce routing distance
    B_up_proj_l3l2_fifos = [None] * (n_b_tiles_distributed * n_dup_shim_b_streams)
    B_up_proj_l2l1_fifos = [None] * (n_b_tiles_distributed * n_dup_shim_b_streams)
    B_down_proj_l3l2_fifos = [None] * (n_b_tiles_distributed * n_dup_shim_b_streams)
    B_down_proj_l2l1_fifos = [None] * (n_b_tiles_distributed * n_dup_shim_b_streams)
    logging.debug(
        f"Len B_up_proj_l3l2_fifos: {len(B_up_proj_l3l2_fifos)}, Len B_up_proj_l2l1_fifos: {len(B_up_proj_l2l1_fifos)}, len B_up_proj_l3l2_fifos: {len(B_up_proj_l3l2_fifos)}, len B_down_proj_l2l1_fifos: {len(B_down_proj_l2l1_fifos)},"
    )

    # C tiles pipelined from up_proj core to down_proj core
    C_up_proj_l1l1_fifos = [
        [None] * n_b_tiles_distributed for _ in range(n_a_tiles_distributed)
    ]

    # Partial C tiles for accumulation between down_proj core and mem tile
    C_down_proj_part_l1l2_fifos = [
        [None] * n_b_tiles_distributed for _ in range(n_a_tiles_distributed)
    ]
    C_down_proj_part_l2l1_fifos = [
        [None] * n_b_tiles_distributed for _ in range(n_a_tiles_distributed)
    ]
    logging.debug(
        f"len C_up_proj_l1l1_fifos: {len(C_up_proj_l1l1_fifos)}, len C_down_proj_part_l1l2_fifos: {len(C_down_proj_part_l1l2_fifos)}, len C_down_proj_part_l2l1_fifos: {len(C_down_proj_part_l2l1_fifos)}"
    )

    # Output C tiles from down_proj core
    C_down_proj_out_l1l1_fifos = [
        [None] * (n_b_tiles_distributed - 1) for _ in range(n_a_tiles_distributed)
    ]
    C_down_proj_out_l1l2_fifos = [None] * n_a_tiles_distributed
    C_down_proj_out_l2l3_fifos = [None] * n_a_tiles_distributed
    logging.debug(
        f"len C_down_proj_out_l1l1_fifos: {len(C_down_proj_out_l1l1_fifos)}, len C_down_proj_out_l1l2_fifos: {len(C_down_proj_out_l1l2_fifos)}, len C_down_proj_out_l2l3_fifos: {len(C_down_proj_out_l2l3_fifos)}"
    )

    # Runtime parameters
    rtps_up_proj = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp_up_proj{a_tile}_{b_tile}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for b_tile in range(n_b_tiles_distributed)
        ]
        for a_tile in range(n_a_tiles_distributed)
    ]
    rtps_down_proj = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp_down_proj{a_tile}_{b_tile}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for b_tile in range(n_b_tiles_distributed)
        ]
        for a_tile in range(n_a_tiles_distributed)
    ]

    # Create barriers to synchronize individual workers with the runtime sequence
    workerBarriersUpProj = [
        [WorkerRuntimeBarrier() for b_tile in range(n_b_tiles_distributed)]
        for a_tile in range(n_a_tiles_distributed)
    ]
    workerBarriersDownProj = [
        [WorkerRuntimeBarrier() for b_tile in range(n_b_tiles_distributed)]
        for a_tile in range(n_a_tiles_distributed)
    ]

    # Input
    for a_tile in range(n_a_tiles_distributed):
        A_l3l2_fifos[a_tile] = ObjectFifo(
            A_l2_ty, name=f"A_L3L2_{a_tile}", depth=fifo_depth
        )
        dims_to_stream = [
            (m // r, r * k),
            (k // s, s),
            (r, k),
            (s, 1),
        ]
        A_l2l1_fifos[a_tile] = (
            A_l3l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"A_L2L1_{a_tile}",
                dims_to_stream=dims_to_stream,
                placement=Tile(a_tile, 1),
            )
        )

    # Input B_Up
    for b_tile in range(n_b_tiles_distributed * n_dup_shim_b_streams):
        B_up_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_up_proj_L3L2_{b_tile}", depth=fifo_depth
        )
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_up_proj_l2l1_fifos[b_tile] = (
            B_up_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_up_proj_l1_ty,
                name=f"B_up_proj_L2L1_{b_tile}",
                dims_to_stream=dims_to_stream,
                placement=Tile(
                    b_tile * 2, 1
                ),  # Switch between up and down proj B tile streams across shim tiles
            )
        )

    # Input B_Down: n and k are swapped compared to B_Up
    for b_tile in range(n_b_tiles_distributed * n_dup_shim_b_streams):
        B_down_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_down_proj_L3L2_{b_tile}", depth=fifo_depth
        )
        if b_col_maj:
            dims_to_stream = [(k // t, t * n), (n // s, s), (t, n), (s, 1)]
        else:
            dims_to_stream = [(n // s, s * k), (k // t, t), (s, k), (t, 1)]
        B_down_proj_l2l1_fifos[b_tile] = (
            B_down_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_down_proj_l1_ty,
                name=f"B_down_proj_L2L1_{b_tile}",
                dims_to_stream=dims_to_stream,
                placement=Tile(
                    b_tile * 2 + 1, 1
                ),  # Switch between up and down proj B tile streams across shim tiles
            )
        )

    # Up proj C
    for a_tile in range(n_a_tiles_distributed):
        for b_tile in range(n_b_tiles_distributed):
            C_up_proj_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_up_proj_l1_ty,
                name=f"C_up_proj_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )

    # Down proj partial C
    for a_tile in range(n_a_tiles_distributed):
        for b_tile in range(n_b_tiles_distributed):
            # Per MT, at most 2 of these objfifos can be connected considering other streams and that the max is 6 S2MM/MM2S per MT
            logging.debug(
                f"Placeing C_down_proj_part fifos at {(a_tile * n_b_tiles_distributed + b_tile) // 2, 1}"
            )
            C_down_proj_part_l1l2_fifos[a_tile][b_tile] = ObjectFifo(
                C_down_proj_l1_ty,
                name=f"C_down_proj_part_L1L2_{a_tile}_{b_tile}",
                depth=1,
            )
            C_down_proj_part_l2l1_fifos[a_tile][b_tile] = (
                C_down_proj_part_l1l2_fifos[a_tile][b_tile]
                .cons(depth=down_proj_depth)
                .forward(
                    obj_type=C_down_proj_l1_ty,
                    name=f"C_down_proj_part_L2L1_{b_tile}_{a_tile}",
                    depth=down_proj_depth,
                    placement=Tile(
                        (a_tile * n_b_tiles_distributed + b_tile) // 2,
                        1,
                    ),
                )
            )

    # Down proj partial C for reduction
    for a_tile in range(n_a_tiles_distributed):
        for b_tile in range(n_b_tiles_distributed - 1):
            C_down_proj_out_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_down_proj_l1_ty,
                name=f"C_down_proj_out_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth_out,
            )

    # Down proj output C, m-by-k tiles
    for a_tile in range(n_a_tiles_distributed):
        if c_col_maj:
            dims_to_stream = [(k // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        else:
            dims_to_stream = [(m // r, r * k), (r, t), (k // t, r * t), (t, 1)]
        C_down_proj_out_l1l2_fifos[a_tile] = ObjectFifo(
            C_down_proj_l1_ty,
            name=f"C_down_proj_out_L1L2_{a_tile}",
            depth=fifo_depth,
        )
        C_down_proj_out_l2l3_fifos[a_tile] = (
            C_down_proj_out_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=C_l2_ty,
                name=f"C_down_proj_out_L2L3_{a_tile}_{b_tile}",
                depth=fifo_depth,
                dims_to_stream=dims_to_stream,
                placement=Tile(a_tile, 1),
            )
        )

    # Tasks for each worker to perform
    def core_fn_up_proj(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        gelu,
        my_rtp,
        barrier,
        stage_only,
    ):
        barrier.wait_for_value(1)
        rtp_K_div_k = my_rtp[0]
        rtp_n_tiles_per_core = my_rtp[1]
        loop = range(1)  # Workaround for issue #1547
        if rtp_n_tiles_per_core > 1:
            loop = range_(rtp_n_tiles_per_core)
        for _ in loop:
            # Check if up projection stage is enabled, None means all stages are enabled
            if stage_only not in [0, None]:  # Skip computation for up projection stage
                for _ in range_(rtp_K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    in_a.release(1)
                    in_b.release(1)
                elem_out_matmul = out_c.acquire(1)
                out_c.release(1)
            else:  # Perform up projection stage computation
                elem_out_matmul = out_c.acquire(1)
                zero(elem_out_matmul)
                for _ in range_(rtp_K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out_matmul)
                    in_a.release(1)
                    in_b.release(1)
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
        my_rtp,
        barrier,
        buffer_to_reduce,
        stage_only,
    ):
        # No need to pass in internal buffer here since that will be in MT and not in the core
        barrier.wait_for_value(1)
        rtp_n_c_col_tiles_per_core = my_rtp[0]
        rtp_down_proj_depth = my_rtp[1]
        # Check if down projection stage is enabled, None means all stages are enabled
        if stage_only not in [1, None]:  # Skip computation for down projection stage
            for _ in range_(rtp_down_proj_depth):
                elem_acc_c = new_acc_c.acquire(1)
                new_acc_c.release(1)
            for _ in range_(rtp_n_c_col_tiles_per_core):
                elem_in_a = in_a.acquire(1)
                for _ in range_(rtp_down_proj_depth):
                    elem_out_internal = curr_acc_c.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    elem_new_acc_c = new_acc_c.acquire(1)
                    new_acc_c.release(1)
                    in_b.release(1)
                    curr_acc_c.release(1)
                in_a.release(1)
            for _ in range_(rtp_down_proj_depth):
                elem_out_acc_c = out_acc_c.acquire(1)
                elem_final_acc_c = curr_acc_c.acquire(1)
                if buffer_to_reduce:
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    buffer_to_reduce.release(1)
                curr_acc_c.release(1)
                out_acc_c.release(1)
        else:  # Perform down projection stage computation
            # First iteration just passes the partial C tile through
            for _ in range_(rtp_down_proj_depth):
                elem_acc_c = new_acc_c.acquire(1)
                zero(elem_acc_c)
                new_acc_c.release(1)
            for _ in range_(rtp_n_c_col_tiles_per_core):
                elem_in_a = in_a.acquire(1)
                for _ in range_(rtp_down_proj_depth):
                    elem_in_b = in_b.acquire(1)
                    elem_out_internal = curr_acc_c.acquire(1)
                    elem_new_acc_c = new_acc_c.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out_internal, elem_new_acc_c)
                    new_acc_c.release(1)
                    in_b.release(1)
                    curr_acc_c.release(1)
                in_a.release(1)
            for _ in range_(rtp_down_proj_depth):
                # Acquire what's in L2, which is the final accumulated result for the tile
                elem_out_internal = curr_acc_c.acquire(1)
                elem_out_acc_c = out_acc_c.acquire(1)
                if buffer_to_reduce:
                    # Don't send any new data to MT, i.e. new_acc_c, because that will affect
                    # the data in the subsequent tiles. It's sufficient to just use
                    # the internal buffer as input and output
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    add(partial_acc_c, elem_out_internal, elem_out_internal, m * k)
                    buffer_to_reduce.release(1)
                copy(elem_out_internal, elem_out_acc_c, m * k)
                curr_acc_c.release(1)
                out_acc_c.release(1)

    # Set up compute tiles
    workers = []
    for a_tile in range(n_a_tiles_distributed):
        b_tile_offset = a_tile // n_dup_shim_b_streams
        for b_tile in range(n_b_tiles_distributed):
            # Up projection stage
            # Calculate the tile placement (indexing by [row][col])
            # Dividing by 2 since the design can be duplicated within the same 2 columns (4 rows each column),
            # i.e. each column can have 4 up_proj or 4 down_proj cores
            # Modulo 2 as a row offset for the duplicated design in the same column
            tile_col, tile_row = core_tiles[
                (a_tile * n_b_tiles_distributed + b_tile) % cores_per_col
            ][
                ((a_tile * n_b_tiles_distributed + b_tile) // cores_per_col)
                * num_pipeline_stages
            ]
            logging.debug(
                f"Placing up projection worker based on a_tile {a_tile} b_tile {b_tile} b_tile_offset {b_tile_offset} at tile ({tile_col}, {tile_row})"
            )
            workers.append(
                Worker(
                    core_fn_up_proj,
                    [
                        A_l2l1_fifos[a_tile].cons(),
                        B_up_proj_l2l1_fifos[
                            b_tile_offset
                            // n_b_tiles_distributed
                            * n_b_tiles_distributed
                            + b_tile
                        ].cons(),
                        C_up_proj_l1l1_fifos[a_tile][b_tile].prod(),
                        zero_kernel_up_proj,
                        matmul_kernel_up_proj,
                        gelu_kernel,
                        rtps_up_proj[a_tile][b_tile],
                        workerBarriersUpProj[a_tile][b_tile],
                        stage_only,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )
            # Down projection stage
            logging.debug(
                f"Placing down projection worker based on a_tile {a_tile} b_tile {b_tile} b_tile_offset {b_tile_offset} at tile ({tile_col + 1}, {tile_row})"
            )
            # The direction of reduction is always from left to right to avoid the
            # partial data for each core to be written in the same Data Memory
            workers.append(
                Worker(
                    core_fn_down_proj,
                    [
                        C_up_proj_l1l1_fifos[a_tile][b_tile].cons(),
                        B_down_proj_l2l1_fifos[
                            b_tile_offset
                            // n_b_tiles_distributed
                            * n_b_tiles_distributed
                            + b_tile
                        ].cons(),
                        C_down_proj_part_l2l1_fifos[a_tile][b_tile].cons(depth=1),
                        C_down_proj_part_l1l2_fifos[a_tile][b_tile].prod(),
                        (
                            C_down_proj_out_l1l2_fifos[a_tile].prod()
                            if b_tile == 0
                            else C_down_proj_out_l1l1_fifos[a_tile][b_tile - 1].prod()
                        ),
                        zero_kernel_down_proj,
                        matmul_kernel_down_proj,
                        eltwise_add_vector,
                        mem_copy_fcn,
                        rtps_down_proj[a_tile][b_tile],
                        workerBarriersDownProj[a_tile][b_tile],
                        (
                            None
                            if b_tile == n_b_tiles_distributed - 1
                            else C_down_proj_out_l1l1_fifos[a_tile][b_tile].cons()
                        ),
                        stage_only,
                    ],
                    placement=Tile(tile_col + 1, tile_row),
                    stack_size=0xD00,
                )
            )

    # Calculate RTP values for the reduction loop and total C tiles
    K_div_k = K // k
    n_c_up_col_tiles_per_core = N // mem_tile_n
    n_c_row_tiles_per_core = M // m // n_a_tiles_distributed

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4 if not c_col_maj else 2

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, B_ty, C_ty) as (A, B_Up, B_Down, C):
        rt.start(*workers)

        # Set runtime parameters
        def set_rtps_up_proj(*args):
            for a_tile, rtps_row in enumerate(args):
                for b_tile, rtp_row_col in enumerate(rtps_row):
                    rtp_row_col[0] = K_div_k
                    rtp_row_col[1] = n_c_up_col_tiles_per_core * n_c_row_tiles_per_core

        logging.debug(
            f"Up proj RTPs: K_div_k={K_div_k}, n_c_up_col_tiles_per_core={n_c_up_col_tiles_per_core}, n_c_row_tiles_per_core={n_c_row_tiles_per_core}"
        )

        rt.inline_ops(set_rtps_up_proj, rtps_up_proj)

        def set_rtps_down_proj(*args):
            for a_tile, rtps_row in enumerate(args):
                for b_tile, rtp_row_col in enumerate(rtps_row):
                    rtp_row_col[0] = n_c_up_col_tiles_per_core
                    rtp_row_col[1] = down_proj_depth

        logging.debug(
            f"Down proj RTPs: n_c_up_col_tiles_per_core={n_c_up_col_tiles_per_core}, down_proj_depth={down_proj_depth}"
        )

        rt.inline_ops(set_rtps_down_proj, rtps_down_proj)

        # Set the barriers to 1 to allow the worker to read the
        # runtime parameters and start the computation
        for a_tile in range(n_a_tiles_distributed):
            for b_tile in range(n_b_tiles_distributed):
                rt.set_barrier(workerBarriersUpProj[a_tile][b_tile], 1)
                rt.set_barrier(workerBarriersDownProj[a_tile][b_tile], 1)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        for col_group in range(K_div_k // down_proj_depth):
            tg = rt.task_group()
            for tb in range(ceildiv(n_c_row_tiles_per_core, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    current_tb_n_rows = min(
                        [tb_max_n_rows // 2, n_c_row_tiles_per_core - row_base]
                    )
                    if current_tb_n_rows <= 0:
                        # For small input sizes, we may not even need a "pong" iteration
                        break
                    for a_tile in range(n_a_tiles_distributed):
                        logging.debug(
                            f"Col group: {col_group}, TB: {tb}, PP: {pingpong}, A tile: {a_tile}"
                        )
                        # C Output Transfer:
                        C_col_offset = (
                            (col_group * k * down_proj_depth)
                            if not c_col_maj
                            else (col_group * k * M * down_proj_depth)
                        )
                        if not c_col_maj:
                            C_row_offset = (
                                row_base * m * n_a_tiles_distributed * K
                                + a_tile * m * K
                            )  # base address for this transfer block for all BDs
                            C_offset = C_col_offset + C_row_offset
                            C_sizes = [
                                current_tb_n_rows,
                                down_proj_depth,
                                m,
                                k,
                            ]
                            C_strides = [m * K * n_a_tiles_distributed, k, K, 1]
                        else:
                            C_row_offset = (
                                row_base * m * n_a_tiles_distributed + a_tile * m
                            )  # base address for this transfer block for all BDs
                            C_offset = C_col_offset + C_row_offset
                            C_sizes = [down_proj_depth, n_a_tiles_distributed, k, m]
                            C_strides = [M * k, m, M, 1]
                        C_tile = TensorAccessPattern(
                            (K, M) if c_col_maj else (M, K),
                            offset=C_offset,
                            sizes=C_sizes,
                            strides=C_strides,
                        )
                        # This line does not change MLIR output at all - it's just for recording data movement
                        C_taps.append(C_tile)

                        rt.drain(
                            C_down_proj_out_l2l3_fifos[a_tile].cons(),
                            C,
                            tap=C_tile,
                            wait=True,
                            task_group=tg,
                            placement=Tile(a_tile, 0),
                        )
                        logging.debug(
                            f"    Placed C output {a_tile} transfer at ({a_tile}, 0), offset: {C_offset}, sizes: {C_sizes}, strides: {C_strides}"
                        )
                        for tile_row in range(current_tb_n_rows):
                            logging.debug(f"    Tile row: {tile_row}")
                            # A input transfer:
                            A_block_offset = (
                                (row_base + tile_row) * n_a_tiles_distributed * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                a_tile * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset
                            A_sizes = [
                                n_c_up_col_tiles_per_core,
                                K_div_k,
                                m,
                                k,
                            ]
                            A_strides = [0, k, K, 1]
                            A_tile = TensorAccessPattern(
                                (M, K),
                                offset=A_offset,
                                sizes=A_sizes,
                                strides=A_strides,
                            )
                            rt.fill(
                                A_l3l2_fifos[a_tile].prod(),
                                A,
                                tap=A_tile,
                                task_group=tg,
                                placement=Tile(
                                    a_tile,
                                    0,
                                ),
                            )
                            logging.debug(
                                f"        Placed A input {a_tile} transfer at ({a_tile}, 0), offset: {A_offset}, sizes: {A_sizes}, strides: {A_strides}"
                            )
                            # This line does not change MLIR output at all - it's just for recording data movement
                            A_taps.append(A_tile)

                    for duplicate_b_tile in range(n_dup_shim_b_streams):
                        b_tile_offset = duplicate_b_tile * n_b_tiles_distributed
                        logging.debug(f"B tile offset: {b_tile_offset}")
                        for b_tile in range(n_b_tiles_distributed):
                            for stage in range(num_pipeline_stages):
                                for tile_row in range(current_tb_n_rows):
                                    logging.debug(
                                        f"    B tile: {b_tile}, Stage: {stage}, Tile row: {tile_row}"
                                    )
                                    if stage == 0:
                                        # B_Up input transfer:
                                        B_up_proj_col_offset = (
                                            b_tile * n
                                            if not b_col_maj
                                            else b_tile * n * K
                                        )
                                        if not b_col_maj:
                                            B_up_proj_sizes = [
                                                n_c_up_col_tiles_per_core,
                                                K_div_k,
                                                k,
                                                n,
                                            ]
                                            B_up_proj_strides = [
                                                mem_tile_n,
                                                k * N,
                                                N,
                                                1,
                                            ]
                                        else:
                                            B_up_proj_sizes = [
                                                n_c_up_col_tiles_per_core,
                                                K_div_k,
                                                n,
                                                k,
                                            ]
                                            B_up_proj_strides = [
                                                mem_tile_n * K,
                                                k,
                                                K,
                                                1,
                                            ]
                                        B_up_proj_tile = TensorAccessPattern(
                                            (K, N) if not b_col_maj else (N, K),
                                            offset=B_up_proj_col_offset,
                                            sizes=B_up_proj_sizes,
                                            strides=B_up_proj_strides,
                                        )
                                        rt.fill(
                                            B_up_proj_l3l2_fifos[
                                                b_tile + b_tile_offset
                                            ].prod(),
                                            B_Up,
                                            tap=B_up_proj_tile,
                                            task_group=tg,
                                            placement=Tile(
                                                (b_tile + b_tile_offset)
                                                * num_pipeline_stages,
                                                0,
                                            ),
                                        )
                                        logging.debug(
                                            f"        Placed B_Up input {b_tile + b_tile_offset} transfer at ({(b_tile + b_tile_offset) * num_pipeline_stages}, 0), offset: {B_up_proj_col_offset}, sizes: {B_up_proj_sizes}, strides: {B_up_proj_strides}"
                                        )
                                        # This line does not change MLIR output at all - it's just for recording data movement
                                        B_up_proj_taps.append(B_up_proj_tile)
                                    elif stage == 1:
                                        # B_Down input transfer:
                                        B_down_proj_col_offset = (
                                            (
                                                b_tile * n * K
                                                + col_group * k * down_proj_depth
                                            )
                                            if not b_col_maj
                                            else (
                                                b_tile * n
                                                + col_group * k * N * down_proj_depth
                                            )
                                        )
                                        # Notice how some of the sizes/strides are the same or similar
                                        # to the ones in B_Up, but with accounting for the swapped n and k dimensions
                                        if not b_col_maj:
                                            B_down_proj_sizes = [
                                                n_c_up_col_tiles_per_core,
                                                down_proj_depth,
                                                n,
                                                k,
                                            ]
                                            B_down_proj_strides = [
                                                mem_tile_n * K,
                                                k,
                                                K,
                                                1,
                                            ]
                                        else:
                                            B_down_proj_sizes = [
                                                n_c_up_col_tiles_per_core,
                                                down_proj_depth,
                                                k,
                                                n,
                                            ]
                                            B_down_proj_strides = [
                                                mem_tile_n,
                                                k * N,
                                                N,
                                                1,
                                            ]
                                        B_down_proj_tile = TensorAccessPattern(
                                            (N, K) if not b_col_maj else (K, N),
                                            offset=B_down_proj_col_offset,
                                            sizes=B_down_proj_sizes,
                                            strides=B_down_proj_strides,
                                        )
                                        rt.fill(
                                            B_down_proj_l3l2_fifos[
                                                b_tile + b_tile_offset
                                            ].prod(),
                                            B_Down,
                                            tap=B_down_proj_tile,
                                            task_group=tg,
                                            placement=Tile(
                                                (b_tile + b_tile_offset)
                                                * num_pipeline_stages
                                                + 1,
                                                0,
                                            ),
                                        )
                                        logging.debug(
                                            f"        Placed B_Down input {b_tile + b_tile_offset} transfer at ({(b_tile + b_tile_offset) * num_pipeline_stages + 1}, 0), offset: {B_down_proj_col_offset}, sizes: {B_down_proj_sizes}, strides: {B_down_proj_strides}"
                                        )
                                        # These lines do not change MLIR output at all - they are just for recording data movement
                                        B_down_proj_taps.append(B_down_proj_tile)
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        rt.finish_task_group(tg)
                        tg = rt.task_group()
            rt.finish_task_group(tg)

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_up_proj_taps),
            TensorAccessSequence.from_taps(B_down_proj_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    my_program = Program(dev_ty, rt)

    for worker in workers:
        if worker is None:
            raise ValueError("Worker not properly created")
        logging.debug(f"Worker {worker}")
        for fifo in worker.fifos:
            if fifo is None:
                raise ValueError("FIFO in worker not properly created")
            logging.debug(f"    FIFO {fifo}")
            for ofe in fifo.all_of_endpoints():
                if ofe is None:
                    raise ValueError(
                        f"ObjectFifoEndpoint not properly created for FIFO {fifo} in worker {worker}"
                    )
    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
