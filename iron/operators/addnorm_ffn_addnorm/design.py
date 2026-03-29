# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
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
    WorkerRuntimeBarrier,
    str_to_dtype,
)
import aie.dialects.index as index
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU1Col2, NPU2, Tile
from aie.iron.placers import SequentialPlacer
from iron.operators.addnorm_ffn_addnorm.topology import addnorm_ffn_addnorm_design

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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Block 3 AddNormFFNAddNorm Design",
        description="Resolve retained thesis topology parameters for Block 3",
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--topology-id", type=str, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            addnorm_ffn_addnorm_design(
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                topology_id=args.topology_id,
            ),
            sort_keys=True,
        )
    )


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
    stage_only=None,
    archive=None,
    generate_taps=False,
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
    # n_aie_cols will be used to determine whether to send the same data to through different shim tiles, while
    # nB_tiles_distributed will be used to determine what data to send through which shim tiles
    # nA_tiles_distributed replicates the pipelined design across the NPU array
    # There's 2 pipeline stages, and both use the same nB_tiles_distributed parameter since the core fcn loops are the same
    n_aie_rows = 4
    shim_dma_ch_per_col = 2
    cores_per_col = 4
    num_ffn_stages = 2  # Up projection and fused down projection-GeLU stages
    n_aie_cores_needed = nA_tiles_distributed * (
        1 + num_ffn_stages * nB_tiles_distributed
    )  # nA_tiles_distributed essentially duplicates the design, which has one add & norm block (i.e. the 1) and an FFN block

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mem_tile_n = n * nB_tiles_distributed
    logging.debug(
        f"n_aie_cores_needed:{n_aie_cores_needed}, mem_tile_n:{mem_tile_n}, m:{m}"
    )

    # Calculate loop bounds for the reduction loop and total C tiles
    K_div_k = K // k
    nC_up_col_tiles_per_core = N // mem_tile_n
    ln_iters_per_core = M // (nA_tiles_distributed * m)
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
    AR_taps = []
    B_up_proj_taps = []
    B_down_proj_taps = []
    C_taps = []

    # Define tensor types
    packed_hidden_residual_ty = np.ndarray[(2 * M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * K,), np.dtype[dtype_out]]

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

    # AIE Core Function declarations
    archive_name = f"ffn_{m}x{k}x{n}_archive.a" if archive is None else archive
    # No need to use separate buffers for accumulation and transfer to L2, so
    # we only need the zero and matmul kernels
    fifo_depth_out = fifo_depth
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

    # AIE-array data movement with object fifos
    # Dedicated packed [A | R] streams for LN1 and LN2. Replay stays encoded
    # in the runtime TAPs; the ingress path is L3 -> L2 -> L1.
    AR_ln1_l3l2_fifos = [None] * nA_tiles_distributed
    AR_ln1_l2l1_fifos = [None] * nA_tiles_distributed
    AR_ln2_l3l2_fifos = [None] * nA_tiles_distributed
    AR_ln2_l2l1_fifos = [None] * nA_tiles_distributed
    logging.debug(
        "Len AR_ln1_l3l2_fifos: %s, Len AR_ln1_l2l1_fifos: %s, Len AR_ln2_l3l2_fifos: %s, Len AR_ln2_l2l1_fifos: %s",
        len(AR_ln1_l3l2_fifos),
        len(AR_ln1_l2l1_fifos),
        len(AR_ln2_l3l2_fifos),
        len(AR_ln2_l2l1_fifos),
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

    # First AddNorm output pipelined into the up projection workers.
    ln1_stage_l1l1_fifos = [None] * nA_tiles_distributed

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
    C_down_proj_out_l1l1_fifos = [None] * nA_tiles_distributed
    logging.debug(
        f"len C_down_proj_reduce_l1l1_fifos: {len(C_down_proj_reduce_l1l1_fifos)}, len C_down_proj_out_l1l1_fifos: {len(C_down_proj_out_l1l1_fifos)}"
    )

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
                placement=(
                    Tile(0, 1) if nA_tiles_distributed < 3 else Tile(a_tile * 2, 1)
                ),
            )
        )
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
                placement=(
                    Tile(nB_tiles_distributed + 1, 1)
                    if nA_tiles_distributed < 3
                    else Tile(a_tile * 2 + 1, 1)
                ),
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
            depth=fifo_depth,
        )

    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            C_up_proj_l1l1_fifos[a_tile][b_tile] = ObjectFifo(
                C_up_proj_l1_ty,
                name=f"C_up_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
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
                    placement=(
                        Tile(
                            (a_tile * nB_tiles_distributed + b_tile + 1) % n_aie_cols,
                            1,
                        )
                        if nA_tiles_distributed < 3
                        else Tile(
                            ((b_tile % 2) + (a_tile * 2)) % n_aie_cols,
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
                A_l1_ty,
                name=f"C_down_L1L1_{a_tile}_{b_tile}",
                depth=fifo_depth,
            )

    # Down proj output C, m-by-k tiles
    # NOTE: Can't undo the microtiles like with softmax in MHA output projection
    # pipeline because it would use a DMA channel, and one is needed for residual
    # connection, and another for the stream from MT back to core
    for a_tile in range(nA_tiles_distributed):
        C_down_proj_out_l1l1_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"C_out_L1L1_{a_tile}",
            depth=fifo_depth,
        )

    # Second Add & Norm output streams
    for a_tile in range(nA_tiles_distributed):
        dims_to_stream = [(m // r, r * k), (r, s), (k // s, r * s), (s, 1)]
        ln2_l1l2_fifos[a_tile] = ObjectFifo(
            A_l1_ty,
            name=f"ln2_L1L2_{a_tile}",
            depth=fifo_depth,
        )
        ln2_l2l3_fifos[a_tile] = (
            ln2_l1l2_fifos[a_tile]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"ln2_L2L3_{a_tile}",
                dims_to_stream=dims_to_stream,
                placement=(
                    Tile(n_aie_cols - 1 - a_tile, 1)
                    if nA_tiles_distributed < 3
                    else Tile((a_tile * nB_tiles_distributed + 1) % n_aie_cols, 1)
                ),
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
        stage_only,
    ):
        loop = range_(1)
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for output_tile_idx in loop:
            if stage_only not in [0, None]:
                for _ in range_(K_div_k):
                    in_ar.release(1)
                for _ in range_(K_div_k):
                    out_stage1.acquire(1)
                    out_stage1.release(1)
                    in_ar.release(1)
                continue

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

    def core_fn_up_proj(
        in_a,
        in_b,
        out_c,
        zero,
        matmul,
        gelu,
        stage_only,
    ):
        loop = range_(1)
        if nC_tiles_per_core > 1:
            loop = range_(nC_tiles_per_core)
        for _ in loop:
            if stage_only not in [0, None]:
                elem_out_matmul = out_c.acquire(1)
                for _ in range_(K_div_k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    in_a.release(1)
                    in_b.release(1)
                out_c.release(1)
                continue

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
        stage_only,
    ):
        # Check if down projection stage is enabled, None means all stages are enabled
        if stage_only not in [1, None]:  # Skip computation for down projection stage
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
                # Below is executed a second time due to the subsequent layer norm needing to process
                # Only do this with the core that's sending the fully accumulated tile to the LN core
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
                elem_out_acc_c = out_acc_c.acquire(1)
                if buffer_to_reduce:
                    # Don't send any new data to MT, i.e. new_acc_c, because that will affect
                    # the data in the subsequent tiles. It's sufficient to just use
                    # the internal buffer as input and output
                    partial_acc_c = buffer_to_reduce.acquire(1)
                    add(partial_acc_c, elem_out_internal, elem_out_internal, m * k)
                    buffer_to_reduce.release(1)
                copy(elem_out_internal, elem_out_acc_c, m * k)
                # Below is executed a second time due to the subsequent layer norm needing to process
                # Only do this with the core that's sending the fully accumulated tile to the LN core
                if is_end_of_down_proj:
                    elem_new_acc_c = new_acc_c.acquire(1)
                    # Make sure to copy the final accumulated C tile
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
        fused_add_layer_norm_from_inputs,
        add_calc_sum_sumsq,
        packed_fused_add_layer_norm_from_inputs,
        packed_add_calc_sum_sumsq,
        zero_f32,
        stage_only,
    ):
        # Check if second add & norm stage is enabled, None means all stages are enabled
        if stage_only not in [2, None]:  # Skip computation for second add & norm stage
            for _ in range_(K_div_k):
                in_ar.acquire(1)
                in_ar.release(1)
            for _ in range_(K_div_k):
                in_ar.acquire(1)
                in_down.acquire(1)
                in_down.release(1)
                in_ar.release(1)
            for _ in range_(K_div_k):
                in_ar.acquire(1)
                in_down.acquire(1)
                out_ln2.acquire(1)
                out_ln2.release(1)
                in_down.release(1)
                in_ar.release(1)
            return

        zero_f32(ln1_sum_buf, m)
        zero_f32(ln1_sumsq_buf, m)
        for _ in range_(K_div_k):
            elem_in_ar = in_ar.acquire(1)
            packed_add_calc_sum_sumsq(elem_in_ar, ln1_sum_buf, ln1_sumsq_buf)
            in_ar.release(1)

        for col_idx in range_(K_div_k):
            elem_in_ar = in_ar.acquire(1)
            elem_in_down = in_down.acquire(1)
            col_i32 = index.casts(T.i32(), col_idx)
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

        for col_idx in range_(K_div_k):
            elem_in_ar = in_ar.acquire(1)
            elem_in_down = in_down.acquire(1)
            elem_out_ln2 = out_ln2.acquire(1)
            col_i32 = index.casts(T.i32(), col_idx)
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

    # Set up compute tiles
    workers = []
    for a_tile in range(nA_tiles_distributed):
        for b_tile in range(nB_tiles_distributed):
            # Calculate the tile placement (indexing by core_tiles[row][col])
            if nA_tiles_distributed < 3:
                # FFN cores will be placed horizontally, adjacent col will be for Add & Norm core
                # The direction of reduction will be from left to right
                tile_col, tile_row = core_tiles[a_tile * num_ffn_stages][b_tile]
                ln1_tile_col = nB_tiles_distributed
                ln1_tile_row = tile_row
                ln2_tile_col = nB_tiles_distributed + 1
                ln2_tile_row = tile_row
            else:
                # FFN cores will be placed vertically
                # The direction of reduction will be from up to down
                tile_col, tile_row = core_tiles[-1 * (b_tile + 1)][
                    a_tile * num_ffn_stages
                ]
                ln1_tile_col = tile_col
                ln1_tile_row = tile_row - 1
                ln2_tile_col = tile_col + 1
                ln2_tile_row = tile_row - 1
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
                        core_fn_add_norm1,
                        [
                            AR_ln1_l2l1_fifos[a_tile].cons(),
                            ln1_stage_l1l1_fifos[a_tile].prod(),
                            ln1_sum_buffer,
                            ln1_sumsq_buffer,
                            ln1_weight_buffer,
                            ln_zero_f32_kernel,
                            ln_packed_add_calc_sum_sumsq_kernel,
                            ln_packed_fused_add_layer_norm_from_inputs_kernel,
                            stage_only,
                        ],
                        placement=Tile(ln1_tile_col, ln1_tile_row),
                        stack_size=0xF00,
                    )
                )
                logging.debug(
                    f"Placing add & norm stg 1 worker based on a_tile {a_tile} at {workers[-1]._tile}"
                )

                # Second Add & Norm stage
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
                        core_fn_add_norm2,
                        [
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
                            ln_fused_add_layer_norm_from_inputs_kernel,
                            ln_add_calc_sum_sumsq_kernel,
                            ln_packed_fused_add_layer_norm_from_inputs_kernel,
                            ln_packed_add_calc_sum_sumsq_kernel,
                            ln_zero_f32_kernel,
                            stage_only,
                        ],
                        placement=Tile(
                            ln2_tile_col,
                            ln2_tile_row,
                        ),
                        stack_size=0xF00,
                    )
                )
                logging.debug(
                    f"Placing add & norm stg 2 worker based on a_tile {a_tile} at {workers[-1]._tile}"
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
                            C_down_proj_out_l1l1_fifos[a_tile].prod(fifo_depth)
                            if stream_to_ln
                            else C_down_proj_reduce_l1l1_fifos[a_tile][b_tile].prod(
                                fifo_depth
                            )
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
    with rt.sequence(packed_hidden_residual_ty, B_ty, B_ty, C_ty) as (
        packed_hidden_residual,
        B_Up,
        B_Down,
        C,
    ):
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

                packed_hidden_residual_shape = (2 * M * K,)
                packed_tile_elems = 2 * m * k
                for a_tile in range(nA_tiles_distributed):
                    packed_tile_offset = (
                        (a_tile * ln_iters_per_core + row_base)
                        * K_div_k
                        * packed_tile_elems
                    )
                    ln1_ar_tile = TensorAccessPattern(
                        packed_hidden_residual_shape,
                        offset=packed_tile_offset,
                        sizes=[
                            2 * nC_up_col_tiles_per_core,
                            current_tb_n_rows * K_div_k,
                            2 * m,
                            k,
                        ],
                        strides=[0, packed_tile_elems, k, 1],
                    )
                    place = (
                        Tile(0, 0) if nA_tiles_distributed < 3 else Tile(a_tile * 2, 0)
                    )
                    rt.fill(
                        AR_ln1_l3l2_fifos[a_tile].prod(),
                        packed_hidden_residual,
                        tap=ln1_ar_tile,
                        task_group=tg,
                        placement=place,
                    )
                    AR_taps.append(ln1_ar_tile)
                    logging.debug(
                        f"    Placed LN1 packed AR input {a_tile} transfer at {place} with offset {ln1_ar_tile.offset}, sizes {ln1_ar_tile.sizes}, strides {ln1_ar_tile.strides}"
                    )

                    ln2_ar_tile = TensorAccessPattern(
                        packed_hidden_residual_shape,
                        offset=packed_tile_offset,
                        sizes=[3, current_tb_n_rows * K_div_k, 2 * m, k],
                        strides=[0, packed_tile_elems, k, 1],
                    )
                    place = (
                        Tile(nB_tiles_distributed + 1, 0)
                        if nA_tiles_distributed < 3
                        else Tile(a_tile * 2 + 1, 0)
                    )
                    rt.fill(
                        AR_ln2_l3l2_fifos[a_tile].prod(),
                        packed_hidden_residual,
                        tap=ln2_ar_tile,
                        task_group=tg,
                        placement=place,
                    )
                    AR_taps.append(ln2_ar_tile)
                    logging.debug(
                        f"    Placed LN2 packed AR input {a_tile} transfer at {place} with offset {ln2_ar_tile.offset}, sizes {ln2_ar_tile.sizes}, strides {ln2_ar_tile.strides}"
                    )

                    for tile_row in range(current_tb_n_rows):
                        row_tile_idx = row_base + tile_row
                        c_offset = (
                            (row_tile_idx * nA_tiles_distributed + a_tile) * m * K
                        )
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
                            placement=place,
                        )
                        logging.debug(
                            f"    Placed C output {a_tile} transfer at {place} with offset {C_tile.offset}, sizes {C_tile.sizes}, strides {C_tile.strides}"
                        )
                        C_taps.append(C_tile)

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
                                place = (
                                    Tile(b_tile + 1, 0)
                                    if nA_tiles_distributed < 3
                                    else Tile(b_tile * 2, 0)
                                )
                                rt.fill(
                                    B_up_proj_l3l2_fifos[b_tile].prod(),
                                    B_Up,
                                    tap=B_up_proj_tile,
                                    task_group=tg,
                                    placement=place,
                                )
                                logging.debug(
                                    f"        Placed B_Up input {b_tile} transfer at {place} with offset {B_up_proj_tile.offset}, sizes {B_up_proj_tile.sizes}, strides {B_up_proj_tile.strides}"
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
                                place = (
                                    Tile(b_tile + 1, 0)
                                    if nA_tiles_distributed < 3
                                    else Tile(b_tile * 2 + 1, 0)
                                )
                                rt.fill(
                                    B_down_proj_l3l2_fifos[b_tile].prod(),
                                    B_Down,
                                    tap=B_down_proj_tile,
                                    task_group=tg,
                                    placement=place,
                                )
                                logging.debug(
                                    f"        Placed B_Down input {b_tile} transfer at {place} with offset {B_down_proj_tile.offset}, sizes {B_down_proj_tile.sizes}, strides {B_down_proj_tile.strides}"
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
            TensorAccessSequence.from_taps(AR_taps),
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
