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


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("--seq-len", type=int, default=256)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--parallel-seq", type=int, default=1)
    argparser.add_argument("--q-seq-tile", type=int, default=64)
    argparser.add_argument("--kv-seq-tile", type=int, default=64)
    argparser.add_argument("--emb-tile", type=int, default=96)
    argparser.add_argument("--o-proj-acc-depth", type=int, default=1)
    argparser.add_argument("--parallel-heads", type=int, default=1)
    argparser.add_argument("--packed-output-parallel-seq", type=int, default=None)
    argparser.add_argument("--packed-output-rows", type=int, default=None)
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
        parallel_seq=args.parallel_seq,
        q_seq_tile=args.q_seq_tile,
        kv_seq_tile=args.kv_seq_tile,
        emb_tile=args.emb_tile,
        o_proj_acc_depth=args.o_proj_acc_depth,
        parallel_heads=args.parallel_heads,
        packed_output_parallel_seq=args.packed_output_parallel_seq,
        packed_output_rows=args.packed_output_rows,
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
    parallel_seq: int,
    q_seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    o_proj_acc_depth: int,
    parallel_heads: int,
    packed_output_parallel_seq: int | None,
    packed_output_rows: int | None,
    emulate_bf16_mmul_with_bfp16: bool,
    kernel_archive: str,
    trace_size: int = 0,
):
    embed_sz = heads * d
    sequence_parallel_mode = parallel_seq > 1
    parallel_lanes = parallel_seq if sequence_parallel_mode else parallel_heads

    of_depth = 2
    o_proj_weight_consumer_depth = 1
    o_proj_partial_depth = 1
    enable_tracing = True if trace_size > 0 else False
    dtype_str = "bf16"
    dev = "npu2"

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
        f"MHA Dimensions: seq_len={seq_len}, d={d}, parallel_seq={parallel_seq}, q_seq_tile={q_seq_tile}, kv_seq_tile={kv_seq_tile}, emb_tile={emb_tile}, o_proj_acc_depth={o_proj_acc_depth}, parallel_heads={parallel_heads}"
    )
    logging.info(
        f"num_q_seq_blocks: {num_q_seq_blocks}, num_kv_seq_blocks: {num_kv_seq_blocks}, num_qkv_head_block_per_parallel_head: {num_qkv_head_block_per_parallel_head}, num_o_col_groups: {num_o_col_groups}"
    )
    logging.info(f"Data type: {dtype_str}")
    logging.info(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
    logging.info(f"Enable tracing: {enable_tracing}")

    assert heads > 0, "Number of heads must be greater than 0"
    assert parallel_seq > 0, "parallel_seq must be greater than 0"
    assert (
        heads % parallel_heads == 0
    ), "Number of heads must be divisible by parallel_heads"
    assert (
        seq_len % (parallel_seq * q_seq_tile) == 0
    ), "seq_len must be divisible by parallel_seq * q_seq_tile"
    if sequence_parallel_mode:
        assert parallel_seq in (
            2,
            4,
        ), "parallel sequence lowering currently supports ps in {2, 4}"
        assert (
            parallel_heads == 1
        ), "parallel sequence lowering currently requires parallel_heads == 1"
        if packed_output_parallel_seq is not None:
            raise ValueError(
                "packed Block 3 handoff currently requires Block 2 parallel_seq == 1"
            )

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
    if packed_output_parallel_seq is None:
        O_ty = np.ndarray[
            (seq_len, embed_sz),
            np.dtype[dtype],
        ]
    else:
        O_ty = np.ndarray[
            (2 * packed_output_rows * embed_sz,),
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

    if sequence_parallel_mode:
        inQ = ObjectFifo(
            np.ndarray[(parallel_seq * q_seq_tile, d), np.dtype[dtype]],
            name="inQ",
            depth=of_depth,
        )
        memQ = inQ.cons().split(
            offsets=[q_seq_tile * d * i for i in range(parallel_seq)],
            obj_types=[q_ty] * parallel_seq,
            names=[f"memQ{i}" for i in range(parallel_seq)],
            dims_to_stream=[q_dims] * parallel_seq,
            depths=[of_depth] * parallel_seq,
            placement=Tile(col=0, row=1),
        )
    else:
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
        )  # Split between N parallel blocks of heads

    # VJUNG: The SequentialPlacer will place all of these on the same MemTile if Placement is specified. We would need a list of placement in case of one-many or many-one.
    # I think the Sequential Placer will fail if we do a split/join with more than 6 I/Os cuz it tries to place them all on the same tile.

    # K is stored in column-major order
    k_dims = [(kv_seq_tile // t, t * d), (d // s, s), (t, d), (s, 1)]
    if sequence_parallel_mode:
        inK = ObjectFifo(
            np.ndarray[(parallel_seq * kv_seq_tile, d), np.dtype[dtype]],
            name="inK",
            depth=of_depth,
        )
        memK = inK.cons().split(
            offsets=[kv_seq_tile * d * i for i in range(parallel_seq)],
            obj_types=[k_ty] * parallel_seq,
            names=[f"memK{i}" for i in range(parallel_seq)],
            dims_to_stream=[k_dims] * parallel_seq,
            depths=[of_depth] * parallel_seq,
            placement=Tile(col=1, row=1),
        )
    else:
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

    if sequence_parallel_mode:
        inV = ObjectFifo(
            np.ndarray[(parallel_seq * kv_seq_tile, d), np.dtype[dtype]],
            name="inV",
            depth=of_depth,
        )
        memV = inV.cons().split(
            offsets=[kv_seq_tile * d * i for i in range(parallel_seq)],
            obj_types=[v_ty] * parallel_seq,
            names=[f"memV{i}" for i in range(parallel_seq)],
            dims_to_stream=[v_dims] * parallel_seq,
            depths=[of_depth] * parallel_seq,
            placement=Tile(col=2, row=1),
        )
    else:
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
    for i in range(parallel_lanes):
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
    for i in range(parallel_lanes):
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
    for i in range(parallel_lanes):
        scaleOF.append(
            ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}")
        )  # Local to 1 parallel lane

    # Output projection weights
    ow_dims = [
        (d // s, s * emb_tile),
        (emb_tile // t, t),
        (s, emb_tile),
        (t, 1),
    ]

    if sequence_parallel_mode:
        inOW = ObjectFifo(
            np.ndarray[(parallel_seq * d, emb_tile), np.dtype[dtype]],
            name="inOW",
            depth=of_depth,
        )
        memOW = inOW.cons().split(
            offsets=[d * emb_tile * i for i in range(parallel_seq)],
            obj_types=[wo_ty] * parallel_seq,
            names=[f"memOW{i}" for i in range(parallel_seq)],
            dims_to_stream=[ow_dims] * parallel_seq,
            depths=[o_proj_weight_consumer_depth] * parallel_seq,
            placement=Tile(col=3, row=1),
        )
    else:
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
    for i in range(parallel_lanes):
        outOProj.append(
            ObjectFifo(q_ty, depth=o_proj_partial_depth, name=f"outOProj{i}")
        )  # Local to 1 parallel lane
        outOProjAccumOut.append(ObjectFifo(o_ty, depth=1, name=f"outOProjAccumOut{i}"))
        outOProjAccumIn.append(
            outOProjAccumOut[i]
            .cons(depth=o_proj_acc_depth)
            .forward(
                name=f"outOProjAccumIn{i}",
                depth=o_proj_acc_depth,
                placement=Tile(col=6 + (i % 2), row=1),
            )
        )  # Local to 1 parallel lane

    outOPart = []
    if not sequence_parallel_mode:
        for i in range(parallel_heads - 1):
            outOPart.append(
                ObjectFifo(o_ty, depth=o_proj_partial_depth, name=f"outOPart{i}")
            )  # Local to 1 parallel block of heads

    o_dims = [(q_seq_tile // r, r * emb_tile), (r, t), (emb_tile // t, r * t), (t, 1)]
    if sequence_parallel_mode:
        memO = ObjectFifo(
            np.ndarray[(parallel_seq * q_seq_tile, emb_tile), np.dtype[dtype]],
            name="memO",
            dims_to_stream=o_dims,
        )
        outO = memO.prod().join(
            offsets=[q_seq_tile * emb_tile * i for i in range(parallel_seq)],
            obj_types=[o_ty] * parallel_seq,
            names=[f"outO{i}" for i in range(parallel_seq)],
            depths=[of_depth] * parallel_seq,
            placement=Tile(col=7, row=1),
        )
    else:
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
        q_block_stride,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

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
        q_block_stride,
        idx_buffer,
        scale_buffer,
    ):

        # VJUNG: The index buffer count how many Q and KV block this worker has processed
        # From this info we can infer the position in A and P

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

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
        q_block_stride,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

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
    q_block_stride = 0
    for i in range(parallel_lanes):
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
                    0,
                    q_block_stride,
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
                    0,
                    q_block_stride,
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
                    0,
                    q_block_stride,
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
                    (
                        outO[i].prod()
                        if sequence_parallel_mode
                        else (
                            outOPart[i].prod()
                            if i < parallel_heads - 1
                            else outO[0].prod()
                        )
                    ),
                    (
                        None
                        if sequence_parallel_mode
                        else (outOPart[i - 1].cons() if i > 0 else None)
                    ),
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
    q_tile_rows = parallel_seq * q_seq_tile if sequence_parallel_mode else q_seq_tile
    q_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (q_tile_rows, d),
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

    def duplicate_split_taps_for_parallel_seq(
        taps: TensorAccessSequence,
    ) -> list[TensorAccessPattern]:
        duplicated: list[TensorAccessPattern] = []
        for tap in taps:
            sizes = list(tap.sizes)
            strides = list(tap.strides)
            sizes[0] = parallel_seq
            strides[0] = 0
            duplicated.append(
                TensorAccessPattern(
                    tap._tensor_dims,
                    offset=tap.offset,
                    sizes=sizes,
                    strides=strides,
                )
            )
        return duplicated

    if sequence_parallel_mode:
        K_tiles = duplicate_split_taps_for_parallel_seq(K_tiles)
        V_tiles = duplicate_split_taps_for_parallel_seq(V_tiles)

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
    if sequence_parallel_mode:
        WO_tiles = duplicate_split_taps_for_parallel_seq(WO_tiles)

    if packed_output_parallel_seq is None:
        o_tile_rows = (
            parallel_seq * q_seq_tile if sequence_parallel_mode else q_seq_tile
        )
        O_tiles = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (o_tile_rows, emb_tile),
            (1, embed_sz // emb_tile // num_o_col_groups),
        )
    else:
        if sequence_parallel_mode:
            raise ValueError(
                "packed Block 3 handoff currently requires Block 2 parallel_seq == 1"
            )
        if o_proj_acc_depth != 1:
            raise ValueError(
                "packed Block 3 handoff currently requires o_proj_acc_depth == 1"
            )
        if packed_output_rows is None:
            raise ValueError("packed Block 3 handoff requires packed_output_rows")
        if packed_output_rows < seq_len or packed_output_rows % q_seq_tile != 0:
            raise ValueError(
                "packed Block 3 handoff requires packed_output_rows to be "
                "q_seq_tile-aligned and cover seq_len"
            )
        num_packed_q_seq_blocks = packed_output_rows // q_seq_tile
        if num_packed_q_seq_blocks % packed_output_parallel_seq != 0:
            raise ValueError(
                "packed Block 3 handoff requires "
                "(packed_output_rows / q_seq_tile) divisible by "
                "packed_output_parallel_seq"
            )
        row_iters_per_a_tile = num_packed_q_seq_blocks // packed_output_parallel_seq
        packed_tile_elems = 2 * q_seq_tile * emb_tile
        O_tiles = []
        for q_block_idx in range(num_q_seq_blocks):
            a_tile = q_block_idx % packed_output_parallel_seq
            row_iter = q_block_idx // packed_output_parallel_seq
            for col_group in range(num_o_col_groups):
                tile_index = (
                    a_tile * row_iters_per_a_tile + row_iter
                ) * num_o_col_groups + col_group
                O_tiles.append(
                    TensorAccessPattern(
                        (2 * packed_output_rows * embed_sz,),
                        offset=tile_index * packed_tile_elems,
                        sizes=[1, 1, q_seq_tile, emb_tile],
                        strides=[0, 0, emb_tile, 1],
                    )
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

        for i in range(parallel_lanes):
            rt.start(matmul_workers[i])
            rt.start(softmax_workers[i])
            rt.start(matmul_pv_workers[i])
            rt.start(o_proj_workers[i])

        num_runtime_q_groups = (
            num_q_seq_blocks // parallel_seq
            if sequence_parallel_mode
            else num_q_seq_blocks
        )
        for q_block_idx in range(num_runtime_q_groups):

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
                    tap=O_tiles[q_block_idx * num_o_col_groups + col_group],
                    wait=True,
                    placement=Tile(col=7, row=0),
                    task_group=tg,
                )
                logging.debug(
                    f"  O tap: {O_tiles[q_block_idx * num_o_col_groups + col_group]}"
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
