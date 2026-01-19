# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path

import numpy as np
import argparse
import sys

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    GlobalBuffer,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    LocalBuffer,
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
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument("-down-proj-depth", type=int, default=1)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    # Whether to use the scalar kernel; this is low, but can be useful for debugging smaller sizes
    argparser.add_argument("--scalar", type=bool, choices=[0, 1], default=0)
    argparser.add_argument(
        "--emulate-bf16-mmul-with-bfp16", action="store_true", default=False
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
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.c_col_maj,
        args.scalar,
        args.emulate_bf16_mmul_with_bfp16,
        args.prio_accuracy,
        args.trace_size,
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
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    use_scalar,
    emulate_bf16_mmul_with_bfp16,
    prio_accuracy,
    trace_size,
    archive=None,
    generate_taps=False,
):
    n_aie_rows = 2

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    # When using more AIE columns than n_aie_rows (2) (applicable to NPU2),
    # restrict the number of shim/mem tiles to n_aie_rows,
    # since we have only n_aie_rows row tiles for matrix A
    # When using n_aie_rows (2) or less AIE columns (both NPU and NPU2),
    # the number of shim/mem tiles are equal to n_aie_cols.
    # We use the distribute pattern in object FIFO (see linking for A below),
    # since we have n_aie_rows (2) row tiles for matrix A
    n_shim_mem_A = min(n_aie_cols, n_aie_rows)

    # Integer division when n_aie_cols < n_aie_rows, otherwise set to 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < n_aie_rows else 1

    mem_tile_m_A = m * n_A_tiles_per_shim
    mem_tile_m_C = m * n_aie_rows
    mem_tile_n = n * n_aie_cols

    if prio_accuracy:
        assert (
            dtype_out_str == "bf16"
        ), f"prio_accuracy flag is a feature only for bfloat16 output data types"
        use_larger_internal_buffer = True
        # If prio_accuracy flag is enabled, gemm for bfloat16 will accumulate in place with a f32 buffer,
        # which will be converted to bf16 after the reduction loop finishes for output transfer to L2
        dtype_out_internal = str_to_dtype("f32")
        assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
            dtype_out_internal, np.integer
        ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out_internal}) must either both be integral or both be float"
        assert (
            np.dtype(dtype_out_internal).itemsize >= np.dtype(dtype_in).itemsize
        ), f"Output dtype ({dtype_out_internal}) must be equal or larger to input dtype ({dtype_in})"
    else:
        use_larger_internal_buffer = False

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
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % mem_tile_m_A == 0
    ), """A must be tileable into (m * n_A_tiles_per_shim, k)-sized blocks"""

    # Both A and B_Up are tiled in the K dimension into size k.
    assert K % k == 0
    # B_Down is tiled in the N dimension into size k.
    assert N % k == 0
    # n has to be the same as k since the tiling dimensions for up/down projection switch
    assert n == k

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (mem_tile_n * down_proj_depth) == 0
    ), """B must be tileable into (k, n * n_aie_cols * down_proj_depth)-sized blocks"""

    # Output matrix C:
    # Conceptually, we divide output C into (m * n_rows, n)-sized blocks. These
    # blocks are _distributed_ across AIE core columns, and _joined_ across
    # rows, s.t. each of the n_rows compute cores in a column send a
    # contiguous (m, n)-sized block of C.
    assert (
        M % mem_tile_m_C == 0
    ), """C must be tileable into (m * n_aie_rows, n)-sized blocks"""

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
        if n_aie_cols == 1:
            dev_ty = NPU1Col1()
        elif n_aie_cols == 2:
            dev_ty = NPU1Col2()
        elif n_aie_cols == 4:
            dev_ty = NPU1()
    else:
        dev_ty = NPU2()

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(mem_tile_m_A * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(mem_tile_m_C * n,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_Up_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    B_Down_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    scalar_suffix = "_scalar" if use_scalar else ""
    archive_name = f"ffn_{m}x{k}x{n}_archive.a" if archive is None else archive
    if use_larger_internal_buffer:
        # Fix fifo depth for C objfifo to 1 since 1 buffer will be used for accumulation
        # and another for transfer to L2
        fifo_depth_out = 1
        # Set the type for accumulation
        C_l1_ty_internal = np.ndarray[(m, n), np.dtype[dtype_out_internal]]
        # A kernel to convert from the internal f32 accumulation to bf16 for transfer to L2 is needed
        convert_copy_kernel = Kernel(
            f"convert_copy_f32_to_bf16",
            archive_name,
            [C_l1_ty_internal, C_l1_ty, np.int32],
        )
        # Fix the kernels to use f32 outputs
        zero_kernel = Kernel(
            f"zero{scalar_suffix}_f32",
            archive_name,
            [C_l1_ty_internal],
        )
        matmul_func_name = f"matmul{scalar_suffix}_{dtype_in_str}_f32"
        matmul_kernel = Kernel(
            matmul_func_name + "_up_proj",
            archive_name,
            [A_l1_ty, B_Up_l1_ty, C_l1_ty_internal],
        )
        matmul_kernel = Kernel(
            matmul_func_name + "_down_proj",
            archive_name,
            [C_l1_ty_internal, B_Down_l1_ty, A_l1_ty],
        )
    else:
        # No need to use separate buffers for accumulation and transfer to L2, so
        # we only need the zero and matmul kernels
        fifo_depth_out = fifo_depth
        zero_kernel_up_proj = Kernel(
            f"zero{scalar_suffix}_{dtype_out_str}_up_proj",
            archive_name,
            [C_l1_ty],
        )
        zero_kernel_down_proj = Kernel(
            f"zero{scalar_suffix}_{dtype_out_str}_down_proj",
            archive_name,
            [C_l1_ty],
        )
        matmul_func_name = f"matmul{scalar_suffix}_{dtype_in_str}_{dtype_out_str}"
        matmul_kernel = Kernel(
            matmul_func_name + "_up_proj",
            archive_name,
            [A_l1_ty, B_Up_l1_ty, C_l1_ty],
        )
        matmul_kernel = Kernel(
            matmul_func_name + "_down_proj",
            archive_name,
            [C_l1_ty, B_Down_l1_ty, A_l1_ty],
        )
    eltwise_add_vector = Kernel(
        "eltwise_add_bf16_vector", archive_name, [C_l1_ty, C_l1_ty, C_l1_ty, np.int32]
    )
    mem_copy_fcn = Kernel(
        "passThroughLine",
        archive_name,
        [C_l1_ty, C_l1_ty, np.int32],
    )
    gelu_kernel = Kernel(
        "gelu_bf16",
        archive_name,
        [C_l1_ty, C_l1_ty, np.int32],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_Up_l3l2_fifos = [None] * n_aie_cols
    B_Up_l2l1_fifos = [None] * n_aie_cols
    B_Down_l3l2_fifos = [None] * n_aie_cols
    B_Down_l2l1_fifos = [None] * n_aie_cols

    # Partial C tiles pipelined from up_proj_gelu core to add core
    C_partial_l1l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]

    # Partial C tiles for accumulation
    C_partial_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_partial_l2l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]

    # Output C tiles
    C_out_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_out_l2l3_fifos = [None] * n_aie_cols

    # Runtime parameters
    rtps_up_proj_gelu = [
        [
            GlobalBuffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp_up_proj_gelu{row}_{col}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]
    rtps_down_proj = [
        [
            GlobalBuffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"rtp_down_proj{row}_{col}",
                initial_value=np.array([0, 0], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]

    # Create barriers to synchronize individual workers with the runtime sequence
    workerBarriersUpProj = [
        [WorkerRuntimeBarrier() for col in range(n_aie_cols)]
        for row in range(n_aie_rows)
    ]
    workerBarriersDownProj = [
        [WorkerRuntimeBarrier() for col in range(n_aie_cols)]
        for row in range(n_aie_rows)
    ]

    # Input A
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        # If n_shim_mem_A == n_rows, n_A_tiles_per_shim is 1 and
        # this simply links a_l3l2_fifos[i] to a_l2l1_fifos[i] directly,
        # If n_shim_mem_A < n_rows, each column receives multiple rows of
        # tiles; distribute it along rows of AIE cores.
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(
                    2 * i if n_aie_cols == 8 else i, 1
                ),  # alternate columns in full 4x8 NPU2 case
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B_Up
    for col in range(n_aie_cols):
        B_Up_l3l2_fifos[col] = ObjectFifo(
            B_l2_ty, name=f"B_Up_L3L2_{col}", depth=fifo_depth
        )
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_Up_l2l1_fifos[col] = (
            B_Up_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_Up_l1_ty,
                name=f"B_Up_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

    # Input B_Down
    for col in range(n_aie_cols):
        B_Down_l3l2_fifos[col] = ObjectFifo(
            B_l2_ty, name=f"B_Down_L3L2_{col}", depth=fifo_depth
        )
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_Down_l2l1_fifos[col] = (
            B_Down_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_Down_l1_ty,
                name=f"B_Down_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

    # Partial C
    for col in range(n_aie_cols):
        for row in range(n_aie_rows):
            C_partial_l1l1_fifos[row][col] = ObjectFifo(
                C_l1_ty_internal if use_larger_internal_buffer else C_l1_ty,
                name=f"C_partial_L1L1_{col}_{row}",
                depth=fifo_depth,
            )
            C_partial_l1l2_fifos[row][col] = ObjectFifo(
                C_l1_ty_internal if use_larger_internal_buffer else C_l1_ty,
                name=f"C_partial_L1L2_{col}_{row}",
                depth=fifo_depth_out,
            )
            C_partial_l2l1_fifos[row][col] = (
                C_partial_l1l2_fifos[row][col]
                .cons(depth=down_proj_depth)
                .forward(
                    obj_type=(
                        C_l1_ty_internal if use_larger_internal_buffer else C_l1_ty
                    ),
                    name=f"C_partial_L2L1_{col}_{row}",
                    depth=down_proj_depth,
                    placement=Tile(col, 1),
                )
            )

    # Output C
    for col in range(n_aie_cols):
        if c_col_maj:
            dims_to_stream = [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        else:
            dims_to_stream = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        C_out_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_out_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=dims_to_stream,
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        # join along one column
        c_tmp_fifos = (
            C_out_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_out_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_out_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Tasks for each worker to perform
    def core_fn_up_proj_gelu(in_a, in_b, out_c, zero, matmul, my_rtp, barrier):
        barrier.wait_for_value(1)
        rtp_K_div_k = my_rtp[0]
        rtp_n_c_col_tiles_per_core = my_rtp[1]
        for _ in range_(rtp_K_div_k):
            elem_in_a = in_a.acquire(1)
            for _ in range_(rtp_n_c_col_tiles_per_core):
                elem_out_internal = out_c.acquire(1)
                zero(elem_out_internal)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out_internal)
                in_b.release(1)
                out_c.release(1)
            in_a.release(1)

    def core_fn_down_proj(
        in_c, curr_acc_c, new_acc_c, out_acc_c, zero, add, copy, my_rtp, barrier
    ):
        barrier.wait_for_value(1)
        rtp_K_div_k = my_rtp[0]
        rtp_n_c_col_tiles_per_core = my_rtp[1]
        # First iteration just passes the partial C tile through
        for _ in range_(rtp_n_c_col_tiles_per_core):
            elem_acc_c = new_acc_c.acquire(1)
            zero(elem_acc_c)
            new_acc_c.release(1)
        for _ in range_(rtp_K_div_k):
            for _ in range_(rtp_n_c_col_tiles_per_core):
                elem_in_c = in_c.acquire(1)
                elem_curr_acc_c = curr_acc_c.acquire(1)
                elem_new_acc_c = new_acc_c.acquire(1)
                add(elem_in_c, elem_curr_acc_c, elem_new_acc_c, m * n)
                new_acc_c.release(1)
                curr_acc_c.release(1)
                in_c.release(1)
        for _ in range_(rtp_n_c_col_tiles_per_core):
            elem_out_acc_c = out_acc_c.acquire(1)
            elem_final_acc_c = curr_acc_c.acquire(1)
            copy(elem_final_acc_c, elem_out_acc_c, m * n)
            curr_acc_c.release(1)
            out_acc_c.release(1)

    # Set up compute tiles
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row * 2][col]
            workers.append(
                Worker(
                    core_fn_down_proj,
                    [
                        C_partial_l1l1_fifos[row][col].cons(),
                        C_partial_l2l1_fifos[row][col].cons(depth=fifo_depth_out),
                        C_partial_l1l2_fifos[row][col].prod(),
                        C_out_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        eltwise_add_vector,
                        (
                            convert_copy_kernel
                            if use_larger_internal_buffer
                            else mem_copy_fcn
                        ),
                        rtps_down_proj[row][col],
                        workerBarriersDownProj[row][col],
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )
            workers.append(
                Worker(
                    core_fn_up_proj_gelu,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_Up_l2l1_fifos[col].cons(),
                        C_partial_l1l1_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                        rtps_up_proj_gelu[row][col],
                        workerBarriersUpProj[row][col],
                    ],
                    placement=Tile(tile_col, tile_row + 1),
                    stack_size=0xD00,
                )
            )

    # Calculate RTP values for the reduction loop and total C tiles
    K_div_k = K // k
    n_c_col_tiles_per_core = N // mem_tile_n
    n_c_row_tiles_per_core = M // mem_tile_m_C

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4 if not c_col_maj else 2

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, B_ty, C_ty) as (A, B_Up, B_Down, C):
        rt.start(*workers)

        # Set runtime parameters
        def set_rtps_up_proj_gelu(*args):
            for row, rtps_row in enumerate(args):
                for col, rtp_row_col in enumerate(rtps_row):
                    rtp_row_col[0] = K_div_k
                    rtp_row_col[1] = down_proj_depth

        rt.inline_ops(set_rtps_up_proj_gelu, rtps_up_proj_gelu)

        def set_rtps_down_proj(*args):
            for row, rtps_row in enumerate(args):
                for col, rtp_row_col in enumerate(rtps_row):
                    rtp_row_col[0] = K_div_k
                    rtp_row_col[1] = down_proj_depth

        rt.inline_ops(set_rtps_down_proj, rtps_down_proj)

        # Set the barriers to 1 to allow the worker to read the
        # runtime parameters and start the computation
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rt.set_barrier(workerBarriersUpProj[row][col], 1)
                rt.set_barrier(workerBarriersDownProj[row][col], 1)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        for col_group in range(n_c_col_tiles_per_core // down_proj_depth):
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
                    for col in range(n_aie_cols):
                        for tile_row in range(current_tb_n_rows):
                            # C Output Transfer:
                            C_col_offset = (
                                (col * n + col_group * mem_tile_n * down_proj_depth)
                                if not c_col_maj
                                else (
                                    col * n * M
                                    + col_group * mem_tile_n * M * down_proj_depth
                                )
                            )
                            if not c_col_maj:
                                C_block_offset = (
                                    (row_base + tile_row) * n_aie_rows * m * N
                                )  # base address for this transfer block for all BDs
                                C_offset = C_col_offset + C_block_offset
                                C_sizes = [
                                    1,
                                    down_proj_depth,
                                    mem_tile_m_C,
                                    n,
                                ]
                                C_strides = [0, mem_tile_n, N, 1]
                            else:
                                C_block_offset = (
                                    (row_base + tile_row) * n_aie_rows * m
                                )  # base address for this transfer block for all BDs
                                C_offset = C_col_offset + C_block_offset
                                C_sizes = [down_proj_depth, 1, n, m]
                                C_strides = [M * mem_tile_n, 0, M, 1]
                            C_tile = TensorAccessPattern(
                                (N, M) if c_col_maj else (M, N),
                                offset=C_offset,
                                sizes=C_sizes,
                                strides=C_strides,
                            )
                            rt.drain(
                                C_out_l2l3_fifos[col].cons(),
                                C,
                                tap=C_tile,
                                wait=True,
                                task_group=tg,
                                placement=Tile(col, 0),
                            )
                            # This line does not change MLIR output at all - it's just for recording data movement
                            C_taps.append(C_tile)

                            # A input transfer:
                            A_block_offset = (
                                (row_base + tile_row) * n_aie_rows * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                col * n_A_tiles_per_shim * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset
                            A_sizes = [
                                1,
                                K_div_k,
                                mem_tile_m_A,
                                k,
                            ]
                            A_strides = [0, k, K, 1]
                            # always equal to n_aie_rows since we have n_aie_rows row tiles for matrix A
                            if col < n_aie_rows:
                                A_tile = TensorAccessPattern(
                                    (M, K),
                                    offset=A_offset,
                                    sizes=A_sizes,
                                    strides=A_strides,
                                )
                                rt.fill(
                                    A_l3l2_fifos[col].prod(),
                                    A,
                                    tap=A_tile,
                                    task_group=tg,
                                    placement=Tile(
                                        2 * col if n_aie_cols == 8 else col, 0
                                    ),  # alternate columns in full 4x8 NPU2 case
                                )
                                # This line does not change MLIR output at all - it's just for recording data movement
                                A_taps.append(A_tile)

                            # B input transfer:
                            B_col_offset = (
                                (col * n + col_group * mem_tile_n * down_proj_depth)
                                if not b_col_maj
                                else (
                                    col * n * K
                                    + col_group * mem_tile_n * K * down_proj_depth
                                )
                            )
                            if not b_col_maj:
                                B_sizes = [K_div_k, down_proj_depth, k, n]
                                B_strides = [k * N, n * n_aie_cols, N, 1]
                            else:
                                B_sizes = [K_div_k, down_proj_depth, n, k]
                                B_strides = [k, n * n_aie_cols * K, K, 1]
                            B_tile = TensorAccessPattern(
                                (K, N),
                                offset=B_col_offset,
                                sizes=B_sizes,
                                strides=B_strides,
                            )
                            rt.fill(
                                B_Up_l3l2_fifos[col].prod(),
                                B,
                                tap=B_tile,
                                task_group=tg,
                                placement=Tile(col, 0),
                            )

                            # These lines do not change MLIR output at all - they are just for recording data movement
                            B_taps.append(B_tile)
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        rt.finish_task_group(tg)
                        tg = rt.task_group()
            rt.finish_task_group(tg)

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
