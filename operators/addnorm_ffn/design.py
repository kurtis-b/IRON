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
import aie.dialects.index as index
from aie.dialects.aiex import *

from operators.common.utils import torch_dtype_map

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
    argparser.add_argument("-M", type=int, default=8, help="Left matrix rows")
    argparser.add_argument("-K", type=int, default=96, help="Inner dimension")
    argparser.add_argument("-N", type=int, default=96, help="Right matrix columns")
    argparser.add_argument(
        "-m", type=int, default=8, help="GEMM tile size across M dimension"
    )
    argparser.add_argument(
        "-k", type=int, default=96, help="GEMM tile size across K dimension"
    )
    argparser.add_argument(
        "-n", type=int, default=96, help="GEMM tile size across N dimension"
    )
    argparser.add_argument("--down-proj-depth", type=int, default=1)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=2)
    argparser.add_argument("--nA-tiles-distributed", type=int, default=1)
    argparser.add_argument("--nB-tiles-distributed", type=int, default=1)
    argparser.add_argument(
        "--emulate-bf16-mmul-with-bfp16", action="store_true", default=True
    )
    argparser.add_argument(
        "--gelu-stage",
        type=int,
        choices=[0, 1],
        default=1,
        help="Stage to insert the GeLU activation function: 0 for after up projection, 1 for after down projection",
    )
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
    argparser.add_argument(
        "--ln1-w-file",
        type=str,
        default=None,
        help="File path for the first layer norm weights",
    )
    argparser.add_argument(
        "--ln2-w-file",
        type=str,
        default=None,
        help="File path for the second layer norm weights",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--stage-only",
        type=int,
        choices=[-1, 0, 1, 2, 3],
        default=None,
        help="Compute enabled for 0: first add & norm only, 1: up_proj only, 2: down_proj only, 3: final add & norm only, None: all, -1: No compute",
    )
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
        args.nA_tiles_distributed,
        args.nB_tiles_distributed,
        args.dtype_in,
        args.dtype_out,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.gelu_stage,
        args.ln1_w_file,
        args.ln2_w_file,
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
    nA_tiles_distributed,
    nB_tiles_distributed,
    dtype_in_str,
    dtype_out_str,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    gelu_stage,
    ln1_weight_file,
    ln2_weight_file,
    stage_only=None,
    archive=None,
    generate_taps=False,
):

    if (
        ln1_weight_file is None or ln2_weight_file is None
    ):  # Generate default weights if not provided
        logging.warning(
            "Layer norm weight files not provided; using default weights of all ones."
        )
        static_ln1_weights = np.ones(K, dtype=bfloat16)
        static_ln2_weights = np.ones(K, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
        if static_ln1_weights.shape[0] != K:
            raise ValueError("Static ln1 weights length does not match K")
        static_ln2_weights = np.load(ln2_weight_file)
        if static_ln2_weights.shape[0] != K:
            raise ValueError("Static ln2 weights length does not match K")

    if n_aie_cols < 2:
        raise AssertionError(
            "n_aie_cols must be at least 2 due to 4 inputs (A, R, B_Up, B_Down)"
        )
    # n_aie_cols will be used to determine whether to send the same data to through different shim tiles, while
    # nB_tiles_distributed will be used to determine what data to send through which shim tiles
    # nA_tiles_distributed replicates the pipelined design across the NPU array
    # There's 2 pipeline stages, and both use the same nB_tiles_distributed parameter since the core fcn loops are the same
    n_aie_rows = 4
    shim_dma_ch_per_col = 2
    cores_per_col = 4
    ln_cores_per_nA = 2  # 2 cores will generate the ln outputs, which for the first stage will be broadcast to the up proj cores without using MT
    ln_rows_to_process = m // ln_cores_per_nA
    num_ffn_stages = 2  # Up projection and fused down projection-GeLU stages
    n_aie_cores_needed = nA_tiles_distributed * (
        ln_cores_per_nA + ln_cores_per_nA + num_ffn_stages * nB_tiles_distributed
    )  # nA_tiles_distributed essentially duplicates the design, which has two add & norm blocks and an FFN block

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mem_tile_n = n * nB_tiles_distributed
    logging.debug(f"n_aie_cores_needed:{n_aie_cores_needed}, mem_tile_n:{mem_tile_n}")

    # Calculate loop bounds for the reduction loop and total C tiles
    K_div_k = K // k
    nC_up_col_tiles_per_core = N // mem_tile_n
    ln_iters_per_core = M // (
        nA_tiles_distributed * ln_cores_per_nA * ln_rows_to_process
    )
    nC_tiles_per_core = nC_up_col_tiles_per_core * ln_iters_per_core

    logging.debug(
        f"Up proj loop bounds: K_div_k={K_div_k}, nC_up_col_tiles_per_core={nC_up_col_tiles_per_core}, nC_tiles_per_core={nC_tiles_per_core}"
    )
    logging.debug(
        f"Down proj loop bounds: nC_up_col_tiles_per_core={nC_up_col_tiles_per_core}, down_proj_depth={down_proj_depth}"
    )
    logging.debug(
        f"Add & Norm loop bounds: ln_iters_per_core={ln_iters_per_core}, nC_tiles_per_core={nC_tiles_per_core}"
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
    assert m % (ln_cores_per_nA) == 0, "m must be multiple of ln_cores_per_nA"

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
        K == k * down_proj_depth
    ), """Partial C_Down must tile equally into (m, K) with (m, k * down_proj_depth)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert (
        m == r
    )  # These have to match since the add & norm cores output m/2, i.e. m/ln_cores_per_nA, rows each and two cores make up the full m rows
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
    R_taps = []
    B_up_proj_taps = []
    B_down_proj_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * K,), np.dtype[dtype_out]]

    # Add & Norm tensor types ln_cores_per_nA
    ln_weights_ty = np.ndarray[(K,), np.dtype[dtype_in]]
    ln_processing_mt_ty = np.ndarray[
        (m * K,), np.dtype[dtype_in]
    ]  # tile type for streams into/out of NPU
    ln_processing_ty = np.ndarray[
        (ln_rows_to_process * K,), np.dtype[dtype_in]
    ]  # tile type for processing within the LN cores
    ln_in_out_ty = np.ndarray[
        (ln_rows_to_process * k,), np.dtype[dtype_out]
    ]  # tile type coming into and out of FFN

    # GEMM tensor types
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    B_up_proj_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    B_down_proj_l1_ty = np.ndarray[(n, k), np.dtype[dtype_in]]
    C_up_proj_l1_ty = np.ndarray[(m, n), np.dtype[dtype_in]]
    C_down_proj_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]

    # AIE Core Function declarations
    archive_name = f"ffn_{m}x{k}x{n}_archive.a" if archive is None else archive
    # No need to use separate buffers for accumulation and transfer to L2, so
    # we only need the zero and matmul kernels
    fifo_depth_out = fifo_depth
    # Up projection
    matmul_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
    ffn_zero_kernel_up_proj = Kernel(
        f"zero_{dtype_out_str}_up_proj",
        archive_name,
        [C_up_proj_l1_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        matmul_func_name + "_up_proj_half_inps",
        archive_name,
        [ln_in_out_ty, ln_in_out_ty, B_up_proj_l1_ty, C_up_proj_l1_ty],
    )
    ffn_gelu_kernel = Kernel(
        "ffn_gelu_bf16",
        archive_name,
        [C_up_proj_l1_ty, C_up_proj_l1_ty, np.int32],
    )
    # Down projection
    matmul_func_name = f"matmul_with_acc_{dtype_in_str}_{dtype_out_str}"
    ffn_zero_kernel_down_proj = Kernel(
        f"zero_{dtype_out_str}_down_proj",
        archive_name,
        [C_down_proj_l1_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        matmul_func_name + "_down_proj",
        archive_name,
        [C_up_proj_l1_ty, B_down_proj_l1_ty, C_down_proj_l1_ty, C_down_proj_l1_ty],
    )
    ffn_mem_copy_fcn = Kernel(  # Copy fcn for reduction across down proj cores
        "ffn_passThroughLine",
        archive_name,
        [C_down_proj_l1_ty, C_down_proj_l1_ty, np.int32],
    )
    ffn_mem_copy_halves_fcn = Kernel(  # Will write to add & norm, sending half of the output tile in a loop (since there's 2 cores per nA tile)
        "ffn_passThroughTile_out",
        archive_name,
        [C_down_proj_l1_ty, ln_in_out_ty, np.int32, np.int32, np.int32],
    )
    ffn_eltwise_add_vector = Kernel(
        "eltwise_add_bf16_vector",
        archive_name,
        [C_down_proj_l1_ty, C_down_proj_l1_ty, C_down_proj_l1_ty, np.int32],
    )
    # Add & Norm
    ln1_fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_2outs",
        archive_name,
        [
            ln_processing_ty,
            ln_processing_ty,
            ln_weights_ty,
            ln_processing_ty,
            ln_processing_ty,
            np.int32,
            np.int32,
        ],
    )
    ln2_fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_1outs",
        archive_name,
        [
            ln_processing_ty,
            ln_processing_ty,
            ln_weights_ty,
            ln_processing_ty,
            np.int32,
            np.int32,
        ],
    )
    ln1_copy_in_kernel = Kernel(
        "ln_passThroughTile_in",
        archive_name,
        [ln_processing_ty, ln_in_out_ty, np.int32, np.int32, np.int32, np.int32],
    )
    ln1_copy_passthrough_kernel = Kernel(
        "ln_passThroughLine",
        archive_name,
        [ln_in_out_ty, ln_in_out_ty, np.int32],
    )
    ln2_copy_out_kernel = Kernel(
        "ln_passThroughTile_out",
        archive_name,
        [ln_in_out_ty, ln_processing_ty, np.int32, np.int32, np.int32, np.int32],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    # First Add & Norm streams
    A_l3l2_fifos = [None] * nA_tiles_distributed
    A_l2l1_fifos = [[None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)]
    R_l3l2_fifos = [None] * nA_tiles_distributed
    R_l2l1_fifos = [[None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)]
    ln1_l1l1_fifos = [
        [None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)
    ]  # One as input to FFN, another neighboring mem access for next input to FFN
    # TODO: If there's not enough MT channels, just make the objfifo from AN core to next AN core a direct stream instead of through a link in memtile
    ln1_l1l2_fifos = [[None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)]
    ln1_l2l1_fifos = [
        [None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)
    ]  # Input to Second Add & Norm as residual input
    logging.debug(
        f"Len A_l2l1_fifos: {len(A_l2l1_fifos)} len A_l3l2_fifos: {len(A_l3l2_fifos)}, Len R_l2l1_fifos: {len(R_l2l1_fifos)}, len R_l3l2_fifos: {len(R_l3l2_fifos)}"
    )

    # FFN streams
    # The same data may be sent through different shim tiles depending on the num aie cols available and num b tiles to distribute to reduce routing distance
    B_up_proj_l3l2_fifos = [None] * (nB_tiles_distributed)
    B_up_proj_l2l1_fifos = [None] * (nB_tiles_distributed)
    B_down_proj_l3l2_fifos = [None] * (nB_tiles_distributed)
    B_down_proj_l2l1_fifos = [None] * (nB_tiles_distributed)
    logging.debug(
        f"Len B_up_proj_l3l2_fifos: {len(B_up_proj_l3l2_fifos)}, Len B_up_proj_l2l1_fifos: {len(B_up_proj_l2l1_fifos)}, len B_up_proj_l3l2_fifos: {len(B_up_proj_l3l2_fifos)}, len B_down_proj_l2l1_fifos: {len(B_down_proj_l2l1_fifos)},"
    )

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
    logging.debug(
        f"len C_up_proj_l1l1_fifos: {len(C_up_proj_l1l1_fifos)}, len C_down_proj_part_l1l2_fifos: {len(C_down_proj_part_l1l2_fifos)}, len C_down_proj_part_l2l1_fifos: {len(C_down_proj_part_l2l1_fifos)}"
    )

    # Output C tiles from down_proj core
    C_down_proj_reduce_l1l1_fifos = [
        [None] * (nB_tiles_distributed - 1) for _ in range(nA_tiles_distributed)
    ]
    C_down_proj_out_l1l1_fifos = [
        [None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)
    ]
    logging.debug(
        f"len C_down_proj_reduce_l1l1_fifos: {len(C_down_proj_reduce_l1l1_fifos)}, len C_down_proj_out_l1l1_fifos: {len(C_down_proj_out_l1l1_fifos)}"
    )

    # Output tiles for second Add & Norm
    ln2_l1l2_fifos = [[None] * ln_cores_per_nA for _ in range(nA_tiles_distributed)]
    ln2_l2l3_fifos = [None] * nA_tiles_distributed

    # Input
    for a_tile in range(nA_tiles_distributed):
        A_l3l2_fifos[a_tile] = ObjectFifo(
            ln_processing_mt_ty, name=f"A_L3L2_{a_tile}", depth=fifo_depth
        )
        R_l3l2_fifos[a_tile] = ObjectFifo(
            ln_processing_mt_ty, name=f"R_L3L2_{a_tile}", depth=fifo_depth
        )
        of_offsets = [ln_rows_to_process * K * i for i in range(ln_cores_per_nA)]
        # Distribute A and R along one column
        a_tmp_fifos = (
            A_l3l2_fifos[a_tile]
            .cons()
            .split(
                of_offsets,
                obj_types=[ln_processing_ty] * ln_cores_per_nA,
                names=[f"A_L2L1_{a_tile}_{i}" for i in range(ln_cores_per_nA)],
                depths=[fifo_depth] * ln_cores_per_nA,
                placement=(
                    Tile(0, 1)
                    if nA_tiles_distributed < 3
                    else Tile((a_tile * ln_cores_per_nA) % n_aie_cols, 1)
                ),
            )
        )
        r_tmp_fifos = (
            R_l3l2_fifos[a_tile]
            .cons()
            .split(
                of_offsets,
                obj_types=[ln_processing_ty] * ln_cores_per_nA,
                names=[f"R_L2L1_{a_tile}_{i}" for i in range(ln_cores_per_nA)],
                depths=[fifo_depth] * ln_cores_per_nA,
                placement=(
                    Tile(1, 1)
                    if nA_tiles_distributed < 3
                    else Tile((a_tile * ln_cores_per_nA + 1) % n_aie_cols, 1)
                ),
            )
        )
        for ln_core in range(ln_cores_per_nA):
            # Input A and R to first Add & Norm cores
            A_l2l1_fifos[a_tile][ln_core] = a_tmp_fifos[ln_core]
            R_l2l1_fifos[a_tile][ln_core] = r_tmp_fifos[ln_core]
            # Output of first Add & Norm cores
            dims_to_stream_out = [(k // s, s), (ln_rows_to_process, k), (s, 1)]
            ln1_l1l1_fifos[a_tile][ln_core] = ObjectFifo(
                ln_in_out_ty,
                name=f"ln1_L1L1_{a_tile}_{ln_core}",
                depth=fifo_depth,
                dims_to_stream=(  # Placement moves upwards, so last objfifo will be stream to FFN cores
                    dims_to_stream_out if ln_core == ln_cores_per_nA - 1 else None
                ),
            )
            # Residual connection to cores for second Add & Norm
            ln1_l1l2_fifos[a_tile][ln_core] = ObjectFifo(
                ln_processing_ty, name=f"ln1_L1L2_{a_tile}_{ln_core}", depth=fifo_depth
            )
            ln1_l2l1_fifos[a_tile][ln_core] = (
                ln1_l1l2_fifos[a_tile][ln_core]
                .cons()
                .forward(
                    obj_type=ln_processing_ty,
                    name=f"ln1_L2L1_{a_tile}_{ln_core}",
                    placement=(
                        Tile((a_tile + 2 + ln_core) % n_aie_cols, 1)
                        if nA_tiles_distributed < 3
                        else Tile(
                            (a_tile * ln_cores_per_nA + ln_core + 1) % n_aie_cols, 1
                        )
                    ),  # Place second Add & Norm cores further to reduce routing congestion
                    depth=fifo_depth,  # TODO: Try increasing fifo depth to check if this fifo causes the first Add & Norm core to stall
                )
            )

    # Input B_Up
    for b_tile in range(nB_tiles_distributed):
        B_up_proj_l3l2_fifos[b_tile] = ObjectFifo(
            B_l2_ty, name=f"B_up_proj_L3L2_{b_tile}", depth=fifo_depth
        )
        dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_up_proj_l2l1_fifos[b_tile] = (
            B_up_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_up_proj_l1_ty,
                name=f"B_up_proj_L2L1_{b_tile}",
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
            B_l2_ty, name=f"B_down_proj_L3L2_{b_tile}", depth=fifo_depth
        )
        dims_to_stream = [(n // s, s * k), (k // t, t), (s, k), (t, 1)]
        B_down_proj_l2l1_fifos[b_tile] = (
            B_down_proj_l3l2_fifos[b_tile]
            .cons()
            .forward(
                obj_type=B_down_proj_l1_ty,
                name=f"B_down_proj_L2L1_{b_tile}",
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
        for b_tile in range(nB_tiles_distributed):
            C_up_proj_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_up_proj_l1_ty,
                name=f"C_up_proj_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )

    # Down proj partial C
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            # Per MT, at most 2 of these objfifos can be connected considering other streams and that the max is 6 S2MM/MM2S per MT
            max_obfifos_per_mt = 2
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
                    placement=(
                        Tile(
                            (a_tile * nB_tiles_distributed + b_tile + 1) % n_aie_cols,
                            1,
                        )
                        if nA_tiles_distributed < 3
                        else Tile(
                            (a_tile * nB_tiles_distributed + b_tile) % n_aie_cols,
                            1,
                        )
                    ),
                )
            )
            logging.debug(
                f"Placing C_down_proj_part fifos at {((b_tile + 1) % n_aie_cols, 1) if nA_tiles_distributed < 3 else ((a_tile * 2 + b_tile) % n_aie_cols, 1)}"
            )

    # Down proj partial C for reduction
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed - 1):
            C_down_proj_reduce_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_down_proj_l1_ty,
                name=f"C_down_proj_reduce_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )

    # Down proj output C, m-by-k tiles
    for a_tile in range(nA_tiles_distributed):
        for ln_core in range(ln_cores_per_nA):
            dims_to_stream = [
                (ln_rows_to_process, t),
                (k // t, ln_rows_to_process * t),
                (t, 1),
            ]
            C_down_proj_out_l1l1_fifos[a_tile][ln_core] = ObjectFifo(
                ln_in_out_ty,
                name=f"C_down_proj_out_L1L1_{a_tile}_{ln_core}",
                depth=fifo_depth,
                dims_to_stream=dims_to_stream if ln_core == 0 else None,
            )

    # Second Add & Norm output streams
    for a_tile in range(nA_tiles_distributed):
        of_offsets = [ln_rows_to_process * K * i for i in range(ln_cores_per_nA)]
        ln2_l2l3_fifos[a_tile] = ObjectFifo(
            ln_processing_mt_ty,
            name=f"ln2_L2L3_{a_tile}",
            depth=fifo_depth,
        )
        ln2_tmp_fifos = (
            ln2_l2l3_fifos[a_tile]
            .prod()
            .join(
                of_offsets,
                obj_types=[ln_processing_ty] * ln_cores_per_nA,
                names=[f"ln2_L1L2_{a_tile}_{i}" for i in range(ln_cores_per_nA)],
                depths=[fifo_depth] * ln_cores_per_nA,
                placement=(
                    Tile(n_aie_cols - 1, 1)
                    if nA_tiles_distributed < 3
                    else Tile((a_tile * nB_tiles_distributed + 1) % n_aie_cols, 1)
                ),
            )
        )
        for ln_core in range(ln_cores_per_nA):
            ln2_l1l2_fifos[a_tile][ln_core] = ln2_tmp_fifos[ln_core]

    # Tasks for each worker to perform
    def core_fn_add_norm1(
        of_in1,
        of_in2,
        weights,
        internal_out,
        of_out1,
        of_out2,
        of_out_from_adj,
        fused_add_layer_norm,
        copy_tiled,
        copy_passthrough,
        stage_only,
    ):
        # Check if first add & norm stage is enabled, None means all stages are enabled
        if stage_only not in [0, None]:  # Skip computation for first add & norm stage
            for row_idx in range_(ln_iters_per_core):
                elem_in1 = of_in1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                elem_out2 = of_out2.acquire(1)
                of_in1.release(1)
                of_in2.release(1)
                of_out2.release(1)
                for _ in range_(nC_up_col_tiles_per_core):
                    for col_idx in range_(K_div_k):
                        if (
                            of_out_from_adj
                        ):  # This is to send the next set of rows of the tile in the same column group for up projection
                            elem_out = of_out1.acquire(1)
                            elem_out_from_adj = of_out_from_adj.acquire(1)
                            of_out_from_adj.release(1)
                            of_out1.release(1)
                        elem_out = of_out1.acquire(1)
                        of_out1.release(1)
        else:
            for row_idx in range_(ln_iters_per_core):
                elem_in1 = of_in1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                elem_out2 = of_out2.acquire(1)
                fused_add_layer_norm(
                    elem_in1,
                    elem_in2,
                    weights,
                    internal_out,
                    elem_out2,
                    K,
                    ln_rows_to_process,
                )
                of_in1.release(1)
                of_in2.release(1)
                of_out2.release(1)
                for _ in range_(nC_up_col_tiles_per_core):
                    for col_idx in range_(K_div_k):
                        # TODO: Maybe pass in a tile index to the fcn to create a for-loop in case there's more
                        # than 2 layer norm cores per nA tile, which will require more iterations of the block
                        # below
                        if (
                            of_out_from_adj
                        ):  # This is to send the next set of rows of the tile in the same column group for up projection
                            elem_out1 = of_out1.acquire(1)
                            elem_out_from_adj = of_out_from_adj.acquire(1)
                            copy_passthrough(
                                elem_out_from_adj,
                                elem_out1,
                                ln_rows_to_process * k,
                            )
                            of_out_from_adj.release(1)
                            of_out1.release(1)
                        col_i32 = index.casts(T.i32(), col_idx)
                        elem_out1 = of_out1.acquire(1)
                        copy_tiled(
                            internal_out, elem_out1, K, k, ln_rows_to_process, col_i32
                        )
                        of_out1.release(1)

    def core_fn_up_proj(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        gelu,
        stage_only,
    ):
        loop = range(1)  # Workaround for issue #1547
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for _ in loop:
            # Check if up projection stage is enabled, None means all stages are enabled
            if stage_only not in [1, None]:  # Skip computation for up projection stage
                elem_out_matmul = out_c.acquire(1)
                for _ in range_(K_div_k):
                    elem_in_a = in_a.acquire(2)
                    elem_in_b = in_b.acquire(1)
                    in_a.release(2)
                    in_b.release(1)
                out_c.release(1)
            else:  # Perform up projection stage computation
                elem_out_matmul = out_c.acquire(1)
                zero(elem_out_matmul)
                for _ in range_(K_div_k):
                    elem_in_a = in_a.acquire(2)
                    elem_in_b = in_b.acquire(1)
                    # NOTE: Which one is the first and second set of rows depends on the MT connections
                    matmul(elem_in_a[0], elem_in_a[1], elem_in_b, elem_out_matmul)
                    in_a.release(2)
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
        is_transfer_to_ln_core,
        gelu,
        buffer_to_reduce,
        stage_only,
    ):
        # Check if down projection stage is enabled, None means all stages are enabled
        if stage_only not in [2, None]:  # Skip computation for down projection stage
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
                elem_final_acc_c = curr_acc_c.acquire(1)
                if buffer_to_reduce:
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    buffer_to_reduce.release(1)
                if is_transfer_to_ln_core:
                    for row_idx in range_(ln_cores_per_nA):
                        elem_out_acc_c = out_acc_c.acquire(1)
                        out_acc_c.release(1)
                else:
                    elem_out_acc_c = out_acc_c.acquire(1)
                    out_acc_c.release(1)
                curr_acc_c.release(1)
        else:  # Perform down projection stage computation
            # First iteration just passes the partial C tile through
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
                # Acquire what's in L2, which is the final accumulated result for the tile
                elem_out_internal = curr_acc_c.acquire(1)
                if buffer_to_reduce:
                    # Don't send any new data to MT, i.e. new_acc_c, because that will affect
                    # the data in the subsequent tiles. It's sufficient to just use
                    # the internal buffer as input and output
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    add(partial_acc_c, elem_out_internal, elem_out_internal, m * k)
                    buffer_to_reduce.release(1)
                if is_transfer_to_ln_core:
                    for row_idx in range_(ln_cores_per_nA):
                        row_i32 = index.casts(T.i32(), row_idx)
                        elem_out_acc_c = out_acc_c.acquire(1)
                        copy(
                            elem_out_internal,
                            elem_out_acc_c,
                            k,
                            ln_rows_to_process,
                            row_i32,
                        )
                        out_acc_c.release(1)
                else:
                    elem_out_acc_c = out_acc_c.acquire(1)
                    copy(elem_out_internal, elem_out_acc_c, m * k)
                    out_acc_c.release(1)
                curr_acc_c.release(1)

    def core_fn_add_norm2(
        of_in1,
        of_in2,
        weights,
        internal_in,
        of_out1,
        of_out_to_adj,
        fused_add_layer_norm,
        copy_tiled,
        copy_passthrough,
        stage_only,
    ):
        # Check if second add & norm stage is enabled, None means all stages are enabled
        if stage_only not in [3, None]:  # Skip computation for second add & norm stage
            for row_idx in range_(ln_iters_per_core):
                for col_idx in range_(K_div_k):
                    elem_out = of_in1.acquire(1)
                    of_in1.release(1)
                    if of_out_to_adj:
                        for _ in range_(ln_cores_per_nA - 1):
                            elem_out = of_in1.acquire(1)
                            elem_out_to_adj = of_out_to_adj.acquire(1)
                            of_out_to_adj.release(1)
                            of_in1.release(1)
                elem_out1 = of_out1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                of_out1.release(1)
                of_in2.release(1)
        else:
            for row_idx in range_(ln_iters_per_core):
                for col_idx in range_(K_div_k):
                    col_i32 = index.casts(T.i32(), col_idx)
                    elem_in = of_in1.acquire(1)
                    copy_tiled(elem_in, internal_in, K, k, ln_rows_to_process, col_i32)
                    of_in1.release(1)
                    if of_out_to_adj:
                        for _ in range_(ln_cores_per_nA - 1):
                            elem_in = of_in1.acquire(1)
                            elem_out_to_adj = of_out_to_adj.acquire(1)
                            copy_passthrough(
                                elem_in,
                                elem_out_to_adj,
                                ln_rows_to_process * k,
                            )
                            of_out_to_adj.release(1)
                            of_in1.release(1)
                elem_out1 = of_out1.acquire(1)
                elem_in2 = of_in2.acquire(1)
                fused_add_layer_norm(
                    internal_in, elem_in2, weights, elem_out1, K, ln_rows_to_process
                )
                of_out1.release(1)
                of_in2.release(1)

    # Set up compute tiles
    workers = []
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            # Calculate the tile placement (indexing by core_tiles[row][col])
            if nA_tiles_distributed < 3 and nB_tiles_distributed < n_aie_cols - 2:
                # FFN cores will be placed horizontally, 2 cols (left and right adjacent cols) will be for Add & Norm cores
                # The direction of reduction will be from left to right
                # Add 1 to col index since the left adjacent col is for an Add & Norm core
                tile_col, tile_row = core_tiles[a_tile * num_ffn_stages][b_tile + 1]
                ln1_tile_col = 0
                ln1_tile_row = tile_row
                ln2_tile_col = nB_tiles_distributed + 1
                ln2_tile_row = tile_row
            else:
                # FFN cores will be placed vertically
                # The direction of reduction will be from up to down
                # Bottom row will be for Add & Norm cores since nB_tiles_distributed can at most be 2 to get even partitioning of power of 2 workloads
                tile_col, tile_row = core_tiles[b_tile + ln_cores_per_nA][
                    a_tile * num_ffn_stages
                ]
                ln1_tile_col = tile_col
                ln1_tile_row = core_tiles[0][0][1]  # Get the bottom-most row index
                ln2_tile_col = tile_col + 1
                ln2_tile_row = core_tiles[0][0][1]  # Get the bottom-most row index
            if b_tile == 0:
                # First Add & Norm stage
                for ln_core in range(ln_cores_per_nA):
                    ln1_weight_buffer = Buffer(
                        type=ln_weights_ty,
                        initial_value=static_ln1_weights,
                        name=f"static_ln1_weights_{a_tile}_{ln_core}",
                    )
                    ln1_out_buffer = Buffer(
                        type=ln_processing_ty,
                        name=f"ln1_internal_buffer_{a_tile}_{ln_core}",
                    )
                    workers.append(
                        Worker(
                            core_fn_add_norm1,
                            [
                                A_l2l1_fifos[a_tile][ln_core].cons(),
                                R_l2l1_fifos[a_tile][ln_core].cons(),
                                ln1_weight_buffer,
                                ln1_out_buffer,
                                ln1_l1l1_fifos[a_tile][ln_core].prod(),
                                ln1_l1l2_fifos[a_tile][ln_core].prod(),
                                (
                                    ln1_l1l1_fifos[a_tile][
                                        ln_core - 1
                                    ].cons()  # TODO: Maybe it's ok to make depth=1 here?
                                    if ln_core != 0
                                    else None
                                ),
                                ln1_fused_add_layer_norm_kernel,
                                ln1_copy_in_kernel,
                                ln1_copy_passthrough_kernel,
                                stage_only,
                            ],
                            placement=Tile(ln1_tile_col, ln1_tile_row + ln_core),
                            stack_size=0xF00,
                        )
                    )
                    logging.debug(
                        f"Placing add & norm stg 1 worker {ln_core} based on a_tile {a_tile} at {workers[-1]._tile}"
                    )
            if b_tile == nB_tiles_distributed - 1:
                # Second Add & Norm stage
                for ln_core in range(ln_cores_per_nA):
                    ln2_weight_buffer = Buffer(
                        type=ln_weights_ty,
                        initial_value=static_ln2_weights,
                        name=f"static_ln2_weights_{a_tile}_{ln_core}",
                    )
                    ln2_in_buffer = Buffer(
                        type=ln_processing_ty,
                        name=f"ln2_internal_buffer_{a_tile}_{ln_core}",
                    )
                    workers.append(
                        Worker(
                            core_fn_add_norm2,
                            [
                                C_down_proj_out_l1l1_fifos[a_tile][ln_core].cons(
                                    depth=fifo_depth
                                ),
                                ln1_l2l1_fifos[a_tile][ln_core].cons(),
                                ln2_weight_buffer,
                                ln2_in_buffer,
                                ln2_l1l2_fifos[a_tile][ln_core].prod(),
                                (
                                    C_down_proj_out_l1l1_fifos[a_tile][
                                        ln_core + 1
                                    ].prod()
                                    if ln_core == 0
                                    else None
                                ),
                                ln2_fused_add_layer_norm_kernel,
                                ln2_copy_out_kernel,
                                ln1_copy_passthrough_kernel,
                                stage_only,
                            ],
                            placement=Tile(
                                ln2_tile_col,
                                ln2_tile_row + ln_cores_per_nA - 1 - ln_core,
                            ),
                            stack_size=0xF00,
                        )
                    )
                    logging.debug(
                        f"Placing add & norm stg 2 worker {ln_core} based on a_tile {a_tile} at {workers[-1]._tile}"
                    )
            # Up projection stage
            workers.append(
                Worker(
                    core_fn_up_proj,
                    [
                        # Need two buffers to create the full left mtx tile for GEMM, since each Add & Norm core only outputs half the tile
                        # TODO: The index here is hardcoded to -1, maybe there's a way to avoid having to hardcode to -1
                        ln1_l1l1_fifos[a_tile][-1].cons(
                            # depth=fifo_depth * ln_cores_per_nA
                            # TODO: this is commented out for now since it allows to compile when nA_tiles_distributed > 1,
                            # but should be possible to use above line?
                            depth=fifo_depth
                        ),
                        B_up_proj_l2l1_fifos[b_tile].cons(),
                        C_up_proj_l1l1_fifos[a_tile][b_tile].prod(),
                        ffn_zero_kernel_up_proj,
                        ffn_matmul_kernel_up_proj,
                        ffn_gelu_kernel if gelu_stage == 0 else None,
                        stage_only,
                    ],
                    placement=(
                        Tile(tile_col, tile_row + 1)
                        if nA_tiles_distributed < 3
                        else Tile(tile_col, tile_row)
                    ),
                    stack_size=0xF00,
                )
            )
            logging.debug(
                f"Placing up projection worker based on a_tile {a_tile} b_tile {b_tile} at {workers[-1]._tile}"
            )
            # Down projection stage
            workers.append(
                Worker(
                    core_fn_down_proj,
                    [
                        C_up_proj_l1l1_fifos[a_tile][b_tile].cons(),
                        B_down_proj_l2l1_fifos[b_tile].cons(),
                        C_down_proj_part_l2l1_fifos[a_tile][b_tile].cons(depth=1),
                        C_down_proj_part_l1l2_fifos[a_tile][b_tile].prod(),
                        (
                            # TODO: The index here is hardcoded to 0, so the rest of the objfifos will be used for the add & norm stage
                            # Maybe there's a way to avoid having to hardcode to 0
                            C_down_proj_out_l1l1_fifos[a_tile][0].prod()
                            if b_tile == 0
                            else C_down_proj_reduce_l1l1_fifos[a_tile][
                                b_tile - 1
                            ].prod()
                        ),
                        ffn_zero_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        ffn_eltwise_add_vector,
                        ffn_mem_copy_halves_fcn if b_tile == 0 else ffn_mem_copy_fcn,
                        b_tile == 0,  # is_transfer_to_ln_core
                        ffn_gelu_kernel if gelu_stage == 1 else None,
                        (
                            None
                            if b_tile == nB_tiles_distributed - 1
                            else C_down_proj_reduce_l1l1_fifos[a_tile][b_tile].cons()
                        ),
                        stage_only,
                    ],
                    placement=(
                        Tile(tile_col, tile_row)
                        if nA_tiles_distributed < 3
                        else Tile(tile_col + 1, tile_row)
                    ),
                    stack_size=0xF00,
                )
            )
            logging.debug(
                f"Placing down projection worker based on a_tile {a_tile} b_tile {b_tile} at {workers[-1]._tile}"
            )

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, A_ty, B_ty, B_ty, C_ty) as (A, R, B_Up, B_Down, C):
        rt.start(*workers)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        tg = rt.task_group()
        for tb in range(ceildiv(ln_iters_per_core, tb_max_n_rows)):
            for pingpong in [0, 1]:
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, ln_iters_per_core - row_base]
                )
                if current_tb_n_rows <= 0:
                    # For small input sizes, we may not even need a "pong" iteration
                    break
                ln_taps = [
                    TensorAccessPattern(
                        (1, M * K),
                        row_base * m * nA_tiles_distributed * K
                        + a_tile * current_tb_n_rows * m * K,
                        [1, 1, 1, current_tb_n_rows * m * K],
                        [0, 0, 0, 1],
                    )
                    for a_tile in range(nA_tiles_distributed)
                ]
                for a_tile in range(nA_tiles_distributed):
                    logging.debug(
                        f"TB: {tb}, PP: {pingpong}, row_base: {row_base}, current_tb_n_rows: {current_tb_n_rows}, A tile: {a_tile}"
                    )
                    C_tile = ln_taps[a_tile]
                    # This line does not change MLIR output at all - it's just for recording data movement
                    C_taps.append(C_tile)

                    rt.drain(
                        ln2_l2l3_fifos[a_tile].cons(),
                        C,
                        tap=C_tile,
                        wait=True,
                        task_group=tg,
                        placement=Tile((a_tile * nB_tiles_distributed) % n_aie_cols, 0),
                    )
                    logging.debug(
                        f"    Placed C output {a_tile} transfer at ({(a_tile * nB_tiles_distributed) % n_aie_cols}, 0) with offset {C_tile.offset}, sizes {C_tile.sizes}, strides {C_tile.strides}"
                    )
                    # A input transfer:
                    A_tile = ln_taps[a_tile]
                    rt.fill(
                        A_l3l2_fifos[a_tile].prod(),
                        A,
                        tap=A_tile,
                        task_group=tg,
                        placement=Tile((a_tile * nB_tiles_distributed) % n_aie_cols, 0),
                    )
                    logging.debug(
                        f"    Placed A input {a_tile} transfer at ({(a_tile * nB_tiles_distributed) % n_aie_cols}, 0) with offset {A_tile.offset}, sizes {A_tile.sizes}, strides {A_tile.strides}"
                    )
                    # This line does not change MLIR output at all - it's just for recording data movement
                    A_taps.append(A_tile)
                    # R input transfer:
                    R_tile = ln_taps[a_tile]
                    rt.fill(
                        R_l3l2_fifos[a_tile].prod(),
                        R,
                        tap=R_tile,
                        task_group=tg,
                        placement=Tile((a_tile * nB_tiles_distributed) % n_aie_cols, 0),
                    )
                    logging.debug(
                        f"    Placed R input {a_tile} transfer at ({(a_tile * nB_tiles_distributed) % n_aie_cols}, 0) with offset {R_tile.offset}, sizes {R_tile.sizes}, strides {R_tile.strides}"
                    )

                for tile_row in range(current_tb_n_rows):
                    for b_tile in range(nB_tiles_distributed):
                        for stage in range(num_ffn_stages):
                            logging.debug(
                                f"    B tile: {b_tile}, Stage: {stage}, tile_row: {tile_row}"
                            )
                            if stage == 0:
                                # B_Up input transfer:
                                B_up_proj_col_offset = b_tile * n
                                B_up_proj_sizes = [
                                    nC_up_col_tiles_per_core,
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
                                B_up_proj_tile = TensorAccessPattern(
                                    (N, K),
                                    offset=B_up_proj_col_offset,
                                    sizes=B_up_proj_sizes,
                                    strides=B_up_proj_strides,
                                )
                                rt.fill(
                                    B_up_proj_l3l2_fifos[b_tile].prod(),
                                    B_Up,
                                    tap=B_up_proj_tile,
                                    task_group=tg,
                                    placement=Tile(
                                        (a_tile * nB_tiles_distributed + b_tile)
                                        % (n_aie_cols - 1)
                                        + 1,
                                        0,
                                    ),
                                )
                                logging.debug(
                                    f"        Placed B_Up input {b_tile} transfer at ({(a_tile * nB_tiles_distributed + b_tile) % (n_aie_cols - 1) + 1}, 0) with offset {B_up_proj_tile.offset}, sizes {B_up_proj_tile.sizes}, strides {B_up_proj_tile.strides}"
                                )
                                # This line does not change MLIR output at all - it's just for recording data movement
                                B_up_proj_taps.append(B_up_proj_tile)
                            elif stage == 1:
                                # B_Down input transfer:
                                B_down_proj_col_offset = b_tile * n * K
                                # Notice how some of the sizes/strides are the same or similar
                                # to the ones in B_Up, but with accounting for the swapped n and k dimensions
                                B_down_proj_sizes = [
                                    nC_up_col_tiles_per_core,
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
                                B_down_proj_tile = TensorAccessPattern(
                                    (K, N),
                                    offset=B_down_proj_col_offset,
                                    sizes=B_down_proj_sizes,
                                    strides=B_down_proj_strides,
                                )
                                rt.fill(
                                    B_down_proj_l3l2_fifos[b_tile].prod(),
                                    B_Down,
                                    tap=B_down_proj_tile,
                                    task_group=tg,
                                    placement=Tile(
                                        (a_tile * nB_tiles_distributed + b_tile)
                                        % (n_aie_cols - 1)
                                        + 1,
                                        0,
                                    ),
                                )
                                logging.debug(
                                    f"        Placed B_Down input {b_tile} transfer at ({(a_tile * nB_tiles_distributed + b_tile) % (n_aie_cols - 1) + 1}, 0) with offset {B_down_proj_tile.offset}, sizes {B_down_proj_tile.sizes}, strides {B_down_proj_tile.strides}"
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
            TensorAccessSequence.from_taps(R_taps),
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
