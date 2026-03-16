# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import if_, else_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from operators.encoder_pipeline.placements import TOPOLOGY_PLACEMENTS

BASE_DIR = Path(__file__).parent


def main():
    argparser = argparse.ArgumentParser(
        prog="Encoder Pipeline Design",
        description="Emits MLIR for the minimal fused encoder pipeline design",
    )
    argparser.add_argument("--heads", type=int, default=12)
    argparser.add_argument("--seq-len", type=int, default=64)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--seq-tile", type=int, default=32)
    argparser.add_argument("--kv-seq-tile", type=int, default=64)
    argparser.add_argument("--emb-tile", type=int, default=96)
    argparser.add_argument("--proj-acc-depth", type=int, default=8)
    argparser.add_argument("--parallel-seq", type=int, default=1)
    argparser.add_argument("--parallel-heads", type=int, default=1)
    argparser.add_argument("--o-proj-acc-group-size", type=int, default=1)
    argparser.add_argument("--n-b-tiles-distributed", type=int, default=1)
    argparser.add_argument("--ffn-intermediate-size", type=int, default=3072)
    argparser.add_argument(
        "--kernel-archive",
        type=str,
        default="encoder_pipeline_kernels.a",
    )
    argparser.add_argument("--trace-size", type=int, default=0)
    argparser.add_argument("--ln1-weight-file", type=str, default=None)
    argparser.add_argument("--ln2-weight-file", type=str, default=None)
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default=BASE_DIR / "build" / "encoder_pipeline.mlir",
    )
    args = argparser.parse_args()

    module = encoder_pipeline(
        heads=args.heads,
        seq_len=args.seq_len,
        d=args.d,
        seq_tile=args.seq_tile,
        kv_seq_tile=args.kv_seq_tile,
        emb_tile=args.emb_tile,
        proj_acc_depth=args.proj_acc_depth,
        parallel_heads=args.parallel_heads,
        emulate_bf16_mmul_with_bfp16=True,
        kernel_archive=args.kernel_archive,
        parallel_seq=args.parallel_seq,
        trace_size=args.trace_size,
        ln1_weight_file=args.ln1_weight_file,
        ln2_weight_file=args.ln2_weight_file,
        nB_tiles_distributed=args.n_b_tiles_distributed,
        ffn_intermediate_size=args.ffn_intermediate_size,
        o_proj_acc_group_size=args.o_proj_acc_group_size,
    )

    output_file_path = Path(args.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(module))


def encoder_pipeline(
    heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    proj_acc_depth: int,
    parallel_heads: int,
    emulate_bf16_mmul_with_bfp16: bool,
    kernel_archive: str,
    parallel_seq: int = 1,
    trace_size: int = 0,
    ln1_weight_file=None,
    ln2_weight_file=None,
    nB_tiles_distributed: int = 1,
    ffn_intermediate_size: int | None = None,
    o_proj_acc_group_size: int = 1,
):
    if ffn_intermediate_size is None:
        ffn_intermediate_size = 4 * heads * d

    topology_key = (
        heads,
        seq_len,
        d,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_seq,
        parallel_heads,
        proj_acc_depth,
        o_proj_acc_group_size,
        nB_tiles_distributed,
        ffn_intermediate_size,
    )
    if topology_key not in TOPOLOGY_PLACEMENTS:
        raise ValueError(
            "encoder_pipeline only supports hardcoded placement topologies "
            f"{sorted(TOPOLOGY_PLACEMENTS)} (got {topology_key})"
        )
    placement = TOPOLOGY_PLACEMENTS[topology_key]

    if trace_size != 0:
        raise ValueError("encoder_pipeline does not support tracing")
    if parallel_seq != 1:
        raise ValueError(
            "encoder_pipeline only supports the non-sequence-parallel path "
            f"(parallel_seq=1, got {parallel_seq})"
        )
    if parallel_heads != 1:
        raise ValueError(
            "encoder_pipeline only supports parallel_heads=1 in the hardcoded "
            f"placement path (got {parallel_heads})"
        )
    if nB_tiles_distributed != 1:
        raise ValueError(
            "encoder_pipeline only supports nB_tiles_distributed=1 in the "
            f"hardcoded placement path (got {nB_tiles_distributed})"
        )
    if o_proj_acc_group_size != 1:
        raise ValueError(
            "encoder_pipeline only supports o_proj_acc_group_size=1 in the "
            f"hardcoded placement path (got {o_proj_acc_group_size})"
        )
    if d != 64:
        raise ValueError(f"encoder_pipeline requires d=64 (got {d})")
    if not emulate_bf16_mmul_with_bfp16:
        raise ValueError("encoder_pipeline requires emulate_bf16_mmul_with_bfp16=True")

    embed_sz = heads * d
    num_q_seq_blocks = seq_len // seq_tile
    num_kv_seq_blocks = seq_len // kv_seq_tile
    num_qkv_head_block_per_parallel_head = heads
    num_o_col_groups = embed_sz // (emb_tile * proj_acc_depth)
    if num_o_col_groups != 1:
        raise ValueError(
            "encoder_pipeline hardcoded path requires num_o_col_groups == 1 "
            f"(got {num_o_col_groups})"
        )
    if seq_len % seq_tile != 0:
        raise ValueError("seq_len must be divisible by seq_tile")
    if seq_len % kv_seq_tile != 0:
        raise ValueError("seq_len must be divisible by kv_seq_tile")
    if embed_sz != emb_tile * proj_acc_depth:
        raise ValueError(
            "emb_tile * proj_acc_depth must equal embed_sz "
            f"({emb_tile} * {proj_acc_depth} != {embed_sz})"
        )
    if ffn_intermediate_size % emb_tile != 0:
        raise ValueError(
            "ffn_intermediate_size must be divisible by emb_tile "
            f"({ffn_intermediate_size} % {emb_tile} != 0)"
        )

    dtype = bfloat16
    inv_scale = (1 / np.sqrt(d)) * 1.4453125
    of_depth = 2
    ln1_broadcast_groups = ffn_intermediate_size // emb_tile
    ln_tiles_per_q_block = proj_acc_depth
    ln1_dram_stage_rows = ln1_broadcast_groups * seq_tile
    or_tensor_shape = (2 * seq_len + ln1_dram_stage_rows, embed_sz)

    if ln1_weight_file is None:
        static_ln1_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
    if ln2_weight_file is None:
        static_ln2_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln2_weights = np.load(ln2_weight_file)

    qk_tile = Tile(*placement["compute_tiles"]["qk"])
    softmax_tile = Tile(*placement["compute_tiles"]["softmax"])
    pv_tile = Tile(*placement["compute_tiles"]["pv"])
    o_proj_tile = Tile(*placement["compute_tiles"]["o_proj"])
    ln1_tile = Tile(*placement["compute_tiles"]["ln1"])
    ffn_up_tile = Tile(*placement["compute_tiles"]["ffn_up"])
    ffn_down_tile = Tile(*placement["compute_tiles"]["ffn_down"])
    ln2_tile = Tile(*placement["compute_tiles"]["ln2"])

    q_mem_col = placement["mem_tiles"]["q"]
    k_mem_col = placement["mem_tiles"]["k"]
    v_mem_col = placement["mem_tiles"]["v"]
    ow_mem_col = placement["mem_tiles"]["w_o"]
    o_proj_acc_mem_col = placement["mem_tiles"]["o_proj_acc"]
    bdown_mem_col = placement["mem_tiles"]["b_down"]
    ln2_replay_mem_col = placement["mem_tiles"]["ln2_replay"]
    ln1_replay_mem_col = placement["mem_tiles"]["ln1_replay"]
    bup_mem_col = placement["mem_tiles"]["b_up"]
    ffn_down_acc_mem_col = placement["mem_tiles"]["ffn_down_acc"]
    ln1_stage_col = placement["mem_tiles"]["ln1_stage"]
    residual_mem_col = placement["mem_tiles"]["residual"]
    ffn_residual_mem_col = placement["mem_tiles"]["ffn_residual"]
    output_mem_col = placement["mem_tiles"]["output"]

    q_shim_col = placement["shim_tiles"]["q"]
    k_shim_col = placement["shim_tiles"]["k"]
    v_shim_col = placement["shim_tiles"]["v"]
    ow_shim_col = placement["shim_tiles"]["w_o"]
    bdown_shim_col = placement["shim_tiles"]["b_down"]
    bup_shim_col = placement["shim_tiles"]["b_up"]
    ln1_stage_shim_col = placement["shim_tiles"]["ln1_stage"]
    residual_shim_col = placement["shim_tiles"]["residual"]
    output_shim_col = placement["shim_tiles"]["output"]

    W_O_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    QKV_ty = np.ndarray[(3 * seq_len, embed_sz), np.dtype[dtype]]
    OR_ty = np.ndarray[(2 * seq_len + ln1_dram_stage_rows, embed_sz), np.dtype[dtype]]
    B_Up_ty = np.ndarray[(embed_sz * ffn_intermediate_size,), np.dtype[dtype]]
    B_Down_ty = np.ndarray[(ffn_intermediate_size * embed_sz,), np.dtype[dtype]]

    q_ty = np.ndarray[(seq_tile, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, kv_seq_tile), np.dtype[dtype]]
    qk_ty = np.ndarray[(seq_tile, kv_seq_tile), np.dtype[dtype]]
    v_ty = np.ndarray[(kv_seq_tile, d), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * seq_tile,), np.dtype[dtype]]
    wo_ty = np.ndarray[(d, emb_tile), np.dtype[dtype]]
    o_ty = np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]]
    ffn_b_ty = np.ndarray[(emb_tile, emb_tile), np.dtype[dtype]]
    ln_weights_ty = np.ndarray[(embed_sz,), np.dtype[dtype]]
    sum_l1_ty = np.ndarray[(seq_tile,), np.dtype[np.float32]]

    zero_kernel = Kernel("zero_bf16", kernel_archive, [qk_ty])
    memcopy_kernel_scale = Kernel(
        "passThroughLine", kernel_archive, [s_ty, s_ty, np.int32]
    )
    mem_copy_o_proj = Kernel(
        "passThroughLine_o_proj", kernel_archive, [o_ty, o_ty, np.int32]
    )
    partial_softmax_kernel = Kernel(
        "partial_softmax",
        kernel_archive,
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
    scale_buffer_init_kernel = Kernel(
        "init_scale_buffer", kernel_archive, [s_ty, np.int32]
    )
    matmul_qk_kernel = Kernel(
        "matmul_bf16_bf16_wrapper",
        kernel_archive,
        [q_ty, k_ty, qk_ty, np.ndarray[(2,), np.dtype[np.int32]]],
    )
    matmul_pv_kernel = Kernel(
        "matmul_PV",
        kernel_archive,
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
    rescale_o_kernel = Kernel(
        "rescale_O",
        kernel_archive,
        [qk_ty, s_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
    )
    zero_kernel_o_proj = Kernel("zero_bf16_o_proj", kernel_archive, [o_ty])
    matmul_kernel_o_proj = Kernel(
        "matmul_with_acc_bf16_bf16_o_proj",
        kernel_archive,
        [q_ty, wo_ty, o_ty, o_ty],
    )
    pack_stats_kernel = Kernel(
        "pack_stats_f32_to_bf16_packet",
        kernel_archive,
        [sum_l1_ty, sum_l1_ty, o_ty, np.int32],
    )
    unpack_stats_kernel = Kernel(
        "unpack_stats_bf16_packet_to_f32",
        kernel_archive,
        [o_ty, sum_l1_ty, sum_l1_ty, np.int32],
    )
    ln_zero_f32_kernel = Kernel("ln_zero_f32", kernel_archive, [sum_l1_ty, np.int32])
    ln_calc_sum_sumsq_kernel = Kernel(
        "ln_calc_sum_sumsq",
        kernel_archive,
        [o_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_1outs",
        kernel_archive,
        [o_ty, o_ty, ln_weights_ty, sum_l1_ty, sum_l1_ty, o_ty, np.int32, np.int32],
    )
    ffn_zero_kernel_up_proj = Kernel("ffn_zero_bf16_up_proj", kernel_archive, [o_ty])
    ffn_zero_kernel_down_proj = Kernel(
        "ffn_zero_bf16_down_proj", kernel_archive, [o_ty]
    )
    ffn_matmul_init_kernel_up_proj = Kernel(
        "ffn_matmul_init_bf16_bf16_up_proj",
        kernel_archive,
        [o_ty, ffn_b_ty, o_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        "ffn_matmul_bf16_bf16_up_proj",
        kernel_archive,
        [o_ty, ffn_b_ty, o_ty],
    )
    ffn_matmul_init_kernel_down_proj = Kernel(
        "ffn_matmul_init_bf16_bf16_down_proj",
        kernel_archive,
        [o_ty, ffn_b_ty, o_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        "ffn_matmul_with_acc_bf16_bf16_down_proj",
        kernel_archive,
        [o_ty, ffn_b_ty, o_ty, o_ty],
    )
    ffn_gelu_kernel = Kernel("ffn_gelu_bf16", kernel_archive, [o_ty, o_ty, np.int32])

    q_dims = [(seq_tile // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)]
    k_dims = [(kv_seq_tile // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)]
    v_dims = [
        (kv_seq_tile // 8, 8 * kv_seq_tile),
        (d // 8, 8),
        (8, kv_seq_tile),
        (8, 1),
    ]
    ow_dims = [(d // 8, 8 * emb_tile), (emb_tile // 8, 8), (8, emb_tile), (8, 1)]
    r_dims = [(seq_tile // 8, 8 * emb_tile), (emb_tile // 8, 8), (8, emb_tile), (8, 1)]
    o_dims = [(seq_tile // 8, 8 * emb_tile), (8, 8), (emb_tile // 8, 8 * 8), (8, 1)]
    a_dims_out = [(kv_seq_tile // 8, 8 * 8), (seq_tile // 8, kv_seq_tile * 8), (64, 1)]
    a_dims_in = [(kv_seq_tile // 8, 8), (seq_tile, kv_seq_tile), (8, 1)]
    p_dims_out = [(kv_seq_tile // 8, 8), (seq_tile, kv_seq_tile), (8, 1)]
    p_dims_in = [(kv_seq_tile // 8, 8 * 8), (seq_tile // 8, kv_seq_tile * 8), (64, 1)]
    b_dims = [(emb_tile // 8, 8 * emb_tile), (emb_tile // 8, 8), (8, emb_tile), (8, 1)]

    inQ = ObjectFifo(
        np.ndarray[(seq_tile, d), np.dtype[dtype]], name="inQ", depth=of_depth
    )
    memQ0 = inQ.cons().split(
        offsets=[0],
        obj_types=[q_ty],
        names=["memQ0"],
        dims_to_stream=[q_dims],
        depths=[of_depth],
        placement=Tile(col=q_mem_col, row=1),
    )[0]
    inK = ObjectFifo(k_ty, name="inK", depth=of_depth)
    memK0 = inK.cons().split(
        offsets=[0],
        obj_types=[k_ty],
        names=["memK0"],
        dims_to_stream=[k_dims],
        depths=[of_depth],
        placement=Tile(col=k_mem_col, row=1),
    )[0]
    inV = ObjectFifo(v_ty, name="inV", depth=of_depth)
    memV0 = inV.cons().split(
        offsets=[0],
        obj_types=[v_ty],
        names=["memV0"],
        dims_to_stream=[v_dims],
        depths=[of_depth],
        placement=Tile(col=v_mem_col, row=1),
    )[0]
    inOW = ObjectFifo(wo_ty, name="inOW", depth=of_depth)
    memOW0 = inOW.cons().split(
        offsets=[0],
        obj_types=[wo_ty],
        names=["memOW0"],
        dims_to_stream=[ow_dims],
        depths=[of_depth],
        placement=Tile(col=ow_mem_col, row=1),
    )[0]

    memA0 = ObjectFifo(
        qk_ty,
        depth=of_depth,
        name="memA0",
        dims_to_stream=a_dims_out,
        dims_from_stream_per_cons=a_dims_in,
    )
    memP0 = ObjectFifo(
        qk_ty,
        depth=of_depth,
        name="memP0",
        dims_to_stream=p_dims_out,
        dims_from_stream_per_cons=p_dims_in,
    )
    scaleOF0 = ObjectFifo(s_ty, depth=of_depth, name="scaleOF0")
    outOProj0 = ObjectFifo(q_ty, depth=of_depth, name="outOProj0")
    outOProjAccumOut0 = ObjectFifo(o_ty, depth=1, name="outOProjAccumOut0")
    outOProjAccumIn0 = outOProjAccumOut0.cons(depth=proj_acc_depth).forward(
        obj_type=o_ty,
        name="outOProjAccumIn0",
        depth=proj_acc_depth,
        placement=Tile(col=o_proj_acc_mem_col, row=1),
    )
    outOProjInput = ObjectFifo(o_ty, name="outOProjInput", depth=2)

    ln1ReplayPart = ObjectFifo(o_ty, name="ln1ReplayPart", depth=1)
    ln1Replay = ln1ReplayPart.cons(depth=ln_tiles_per_q_block).forward(
        obj_type=o_ty,
        name="ln1Replay",
        depth=ln_tiles_per_q_block,
        placement=Tile(col=ln1_replay_mem_col, row=1),
    )
    inR = ObjectFifo(o_ty, name="inR", depth=of_depth)
    memR = inR.cons().forward(
        obj_type=o_ty,
        name="memR",
        dims_to_stream=r_dims,
        depth=of_depth,
        placement=Tile(col=residual_mem_col, row=1),
    )

    outLNBroadcast = ObjectFifo(o_ty, name="outLNBroadcast", depth=1)
    ln1StageOut = outLNBroadcast.cons(depth=1)
    inLNFromDDR = ObjectFifo(o_ty, name="inLNFromDDR", depth=1)
    memOutLN = inLNFromDDR.cons(depth=2)

    ffnRFromDDR = ObjectFifo(o_ty, name="ffnRFromDDR", depth=1)
    ffnRIn = ffnRFromDDR.cons(depth=proj_acc_depth).forward(
        obj_type=o_ty,
        name="ffnRIn",
        depth=proj_acc_depth,
        placement=Tile(col=ffn_residual_mem_col, row=1),
    )
    inBUp = ObjectFifo(ffn_b_ty, name="inBUp", depth=1)
    memBUp = inBUp.cons().forward(
        obj_type=ffn_b_ty,
        name="memBUp",
        dims_to_stream=b_dims,
        depth=1,
        placement=Tile(col=bup_mem_col, row=1),
    )
    inBDown = ObjectFifo(ffn_b_ty, name="inBDown", depth=1)
    memBDown = inBDown.cons().forward(
        obj_type=ffn_b_ty,
        name="memBDown",
        dims_to_stream=b_dims,
        depth=1,
        placement=Tile(col=bdown_mem_col, row=1),
    )
    ffnUpOut = ObjectFifo(o_ty, name="ffnUpOut", depth=2)
    ffnDownPart = ObjectFifo(o_ty, name="ffnDownPart", depth=1)
    ffnDownAccum = ffnDownPart.cons(depth=proj_acc_depth).forward(
        obj_type=o_ty,
        name="ffnDownAccum",
        depth=proj_acc_depth,
        placement=Tile(col=ffn_down_acc_mem_col, row=1),
    )
    ffnDownOut = ObjectFifo(o_ty, name="ffnDownOut", depth=2)

    ln2ReplayPart = ObjectFifo(o_ty, name="ln2ReplayPart", depth=1)
    ln2Replay = ln2ReplayPart.cons(depth=proj_acc_depth).forward(
        obj_type=o_ty,
        name="ln2Replay",
        depth=proj_acc_depth,
        placement=Tile(col=ln2_replay_mem_col, row=1),
    )
    outLN2 = ObjectFifo(o_ty, name="outLN2", depth=2)
    memLN2 = outLN2.cons().forward(
        obj_type=o_ty,
        name="memLN2",
        dims_to_stream=o_dims,
        depth=2,
        placement=Tile(col=output_mem_col, row=1),
    )

    def batched_matmul_qk(of_q, of_k, of_a_out, zero, matmul_qk, idx_buffer):
        for _ in range_(sys.maxsize):
            idx_buffer[0] = 0
            idx_buffer[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_in_q = of_q.acquire(1)
                for _ in range_(num_kv_seq_blocks):
                    elem_in_k = of_k.acquire(1)
                    elem_a_out = of_a_out.acquire(1)
                    zero(elem_a_out)
                    matmul_qk(elem_in_q, elem_in_k, elem_a_out, idx_buffer)
                    of_a_out.release(1)
                    of_k.release(1)
                    idx_buffer[0] += 1
                idx_buffer[0] = 0
                of_q.release(1)

    def softmax(
        of_in_a,
        of_out_p,
        of_out_scale,
        partial_softmax,
        init_scale,
        copy_scale,
        idx_buffer,
        scale_buffer,
    ):
        for _ in range_(sys.maxsize):
            idx_buffer[0] = 0
            idx_buffer[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                init_scale(scale_buffer, seq_tile)
                for _ in range_(num_kv_seq_blocks):
                    elem_in_a = of_in_a.acquire(1)
                    elem_out_p = of_out_p.acquire(1)
                    elem_out_scale = of_out_scale.acquire(1)
                    partial_softmax(
                        elem_in_a,
                        elem_out_p,
                        scale_buffer,
                        idx_buffer,
                        inv_scale,
                        seq_tile,
                        kv_seq_tile,
                        seq_len,
                        seq_len,
                    )
                    copy_scale(scale_buffer, elem_out_scale, 4 * seq_tile)
                    of_out_scale.release(1)
                    of_out_p.release(1)
                    of_in_a.release(1)
                    idx_buffer[0] += 1
                idx_buffer[0] = 0

    def batched_matmul_pv(
        of_p, of_v, of_scale, of_o_out, zero, matmul_pv, rescale_o, idx_buffer
    ):
        for _ in range_(sys.maxsize):
            idx_buffer[0] = 0
            idx_buffer[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_out_o = of_o_out.acquire(1)
                zero(elem_out_o)
                elem_in_p = of_p.acquire(1)
                elem_in_v = of_v.acquire(1)
                elem_scale = of_scale.acquire(1)
                matmul_pv(
                    elem_in_p,
                    elem_in_v,
                    elem_out_o,
                    elem_scale,
                    seq_tile,
                    0,
                    idx_buffer,
                )
                of_scale.release(1)
                of_v.release(1)
                of_p.release(1)
                idx_buffer[0] += 1

                if num_kv_seq_blocks > 2:
                    for _ in range_(num_kv_seq_blocks - 2):
                        elem_in_p = of_p.acquire(1)
                        elem_in_v = of_v.acquire(1)
                        elem_scale_mid = of_scale.acquire(1)
                        matmul_pv(
                            elem_in_p,
                            elem_in_v,
                            elem_out_o,
                            elem_scale_mid,
                            seq_tile,
                            1,
                            idx_buffer,
                        )
                        of_scale.release(1)
                        of_v.release(1)
                        of_p.release(1)
                        idx_buffer[0] += 1

                if num_kv_seq_blocks > 1:
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elem_scale_last = of_scale.acquire(1)
                    matmul_pv(
                        elem_in_p,
                        elem_in_v,
                        elem_out_o,
                        elem_scale_last,
                        seq_tile,
                        1,
                        idx_buffer,
                    )
                    rescale_o(elem_out_o, elem_scale_last, seq_tile, idx_buffer)
                    of_scale.release(1)
                    of_v.release(1)
                    of_p.release(1)
                    idx_buffer[0] += 1
                else:
                    rescale_o(elem_out_o, elem_scale, seq_tile, idx_buffer)
                    idx_buffer[0] += 1

                idx_buffer[0] = 0
                of_o_out.release(1)

    def matmul_o_proj(
        of_o_in,
        of_ow_in,
        of_o_acc_in,
        of_o_acc_out,
        of_o_out,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        zero,
        matmul,
        copy,
    ):
        emit_ln1_stats = emb_tile >= 2 * seq_tile
        for _ in range_(sys.maxsize):
            if emit_ln1_stats:
                zero_f32(stats_sum_buf, seq_tile)
                zero_f32(stats_sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_out_o_acc = of_o_acc_out.acquire(1)
                zero(elem_out_o_acc)
                of_o_acc_out.release(1)
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_in_o = of_o_in.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_in_o_acc = of_o_acc_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_o_acc = of_o_acc_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_in_o_acc, elem_out_o_acc)
                    of_o_acc_out.release(1)
                    of_ow_in.release(1)
                    of_o_acc_in.release(1)
                of_o_in.release(1)
            for _ in range_(proj_acc_depth):
                elem_in_o_acc = of_o_acc_in.acquire(1)
                if emit_ln1_stats:
                    calc_sum_sumsq(elem_in_o_acc, stats_sum_buf, stats_sumsq_buf)
                elem_out_o = of_o_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                of_o_out.release(1)
                of_o_acc_in.release(1)
            if emit_ln1_stats:
                elem_stats = of_o_out.acquire(1)
                pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats, seq_tile)
                of_o_out.release(1)

    def core_fn_ln1_fused(
        of_in_o_proj,
        of_replay_curr,
        of_replay_new,
        of_in_residual,
        sum_buf,
        sumsq_buf,
        weights,
        of_out_up,
        fused_add_layer_norm,
        calc_sum_sumsq,
        unpack_stats,
        zero_f32,
        copy,
    ):
        use_packed_ln1_stats = emb_tile >= 2 * seq_tile
        for _ in range_(sys.maxsize):
            if not use_packed_ln1_stats:
                zero_f32(sum_buf, seq_tile)
                zero_f32(sumsq_buf, seq_tile)
            for _ in range_(ln_tiles_per_q_block):
                elem_in = of_in_o_proj.acquire(1)
                if not use_packed_ln1_stats:
                    calc_sum_sumsq(elem_in, sum_buf, sumsq_buf)
                elem_replay = of_replay_new.acquire(1)
                copy(elem_in, elem_replay, seq_tile * emb_tile)
                of_replay_new.release(1)
                of_in_o_proj.release(1)
            if use_packed_ln1_stats:
                elem_stats = of_in_o_proj.acquire(1)
                unpack_stats(elem_stats, sum_buf, sumsq_buf, seq_tile)
                of_in_o_proj.release(1)
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_replay_curr.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                elem_out_up = of_out_up.acquire(1)
                fused_add_layer_norm(
                    elem_in,
                    elem_residual,
                    weights,
                    sum_buf,
                    sumsq_buf,
                    elem_out_up,
                    embed_sz,
                    col_i32,
                )
                elem_replay = of_replay_new.acquire(1)
                copy(elem_out_up, elem_replay, seq_tile * emb_tile)
                of_replay_new.release(1)
                of_out_up.release(1)
                of_replay_curr.release(1)
                of_in_residual.release(1)
            for group_idx in range_(ln1_broadcast_groups - 1):
                group_idx_i32 = index.casts(T.i32(), group_idx)
                replay_next_group = group_idx_i32 < (ln1_broadcast_groups - 2)
                for _ in range_(ln_tiles_per_q_block):
                    elem_in = of_replay_curr.acquire(1)
                    elem_residual = of_in_residual.acquire(1)
                    elem_out_up = of_out_up.acquire(1)
                    copy(elem_in, elem_out_up, seq_tile * emb_tile)
                    of_out_up.release(1)
                    with if_(replay_next_group):
                        elem_replay = of_replay_new.acquire(1)
                        copy(elem_in, elem_replay, seq_tile * emb_tile)
                        of_replay_new.release(1)
                    of_replay_curr.release(1)
                    of_in_residual.release(1)

    def core_fn_ffn_up_proj(
        of_in_a, of_in_b, of_out_c, zero, matmul_init, matmul, gelu
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(ln1_broadcast_groups):
                elem_out = of_out_c.acquire(1)
                for acc_idx in range_(proj_acc_depth):
                    elem_in_a = of_in_a.acquire(1)
                    elem_in_b = of_in_b.acquire(1)
                    acc_idx_i32 = index.casts(T.i32(), acc_idx)
                    with if_(acc_idx_i32 == 0) as if_first:
                        matmul_init(elem_in_a, elem_in_b, elem_out)
                    with else_(if_first):
                        matmul(elem_in_a, elem_in_b, elem_out)
                    of_in_b.release(1)
                    of_in_a.release(1)
                gelu(elem_out, elem_out, seq_tile * emb_tile)
                of_out_c.release(1)

    def core_fn_ffn_down_proj(
        of_in_a, of_in_b, of_curr_acc, of_new_acc, of_out, matmul_init, matmul, copy
    ):
        for _ in range_(sys.maxsize):
            elem_in_a = of_in_a.acquire(1)
            for _ in range_(proj_acc_depth):
                elem_new_acc = of_new_acc.acquire(1)
                elem_in_b = of_in_b.acquire(1)
                matmul_init(elem_in_a, elem_in_b, elem_new_acc)
                of_in_b.release(1)
                of_new_acc.release(1)
            of_in_a.release(1)
            for _ in range_(ln1_broadcast_groups - 1):
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_new_acc = of_new_acc.acquire(1)
                    elem_in_b = of_in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_curr_acc, elem_new_acc)
                    of_in_b.release(1)
                    of_new_acc.release(1)
                    of_curr_acc.release(1)
                of_in_a.release(1)
            for _ in range_(proj_acc_depth):
                elem_out = of_out.acquire(1)
                elem_curr_acc = of_curr_acc.acquire(1)
                copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                of_curr_acc.release(1)
                of_out.release(1)

    def core_fn_add_norm2(
        of_in1,
        of_in2,
        of_replay_curr,
        of_replay_new,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        fused_add_layer_norm,
        calc_sum_sumsq,
        zero_f32,
        copy,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_in1 = of_in1.acquire(1)
                calc_sum_sumsq(elem_in1, sum_buf, sumsq_buf)
                elem_replay = of_replay_new.acquire(1)
                copy(elem_in1, elem_replay, seq_tile * emb_tile)
                of_replay_new.release(1)
                of_in1.release(1)
            for col_idx in range_(proj_acc_depth):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_ffn = of_replay_curr.acquire(1)
                elem_residual = of_in2.acquire(1)
                elem_out = of_out.acquire(1)
                fused_add_layer_norm(
                    elem_ffn,
                    elem_residual,
                    weights,
                    sum_buf,
                    sumsq_buf,
                    elem_out,
                    embed_sz,
                    col_i32,
                )
                of_out.release(1)
                of_in2.release(1)
                of_replay_curr.release(1)

    idx_buffer_qk = Buffer(
        initial_value=np.zeros(shape=(2,), dtype=np.int32), name="idx_buffer_qk_0"
    )
    idx_buffer_softmax = Buffer(
        initial_value=np.zeros(shape=(2,), dtype=np.int32), name="idx_buffer_softmax_0"
    )
    idx_buffer_pv = Buffer(
        initial_value=np.zeros(shape=(2,), dtype=np.int32), name="idx_buffer_pv_0"
    )
    scale_buffer_softmax = Buffer(
        initial_value=np.zeros(shape=(4 * seq_tile,), dtype=dtype),
        name="scale_buffer_softmax_0",
    )
    o_proj_stats_sum_buffer = Buffer(type=sum_l1_ty, name="o_proj_stats_sum_0")
    o_proj_stats_sumsq_buffer = Buffer(type=sum_l1_ty, name="o_proj_stats_sumsq_0")
    ln1_weight_buffer = Buffer(
        type=ln_weights_ty, initial_value=static_ln1_weights, name="static_ln1_weights"
    )
    ln2_weight_buffer = Buffer(
        type=ln_weights_ty, initial_value=static_ln2_weights, name="static_ln2_weights"
    )
    ln1_norm_sum_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sum_buffer")
    ln1_norm_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sumsq_buffer")
    ln2_sum_buffer = Buffer(type=sum_l1_ty, name="ln2_sum_buffer")
    ln2_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln2_sumsq_buffer")

    qk_worker = Worker(
        batched_matmul_qk,
        fn_args=[
            memQ0.cons(),
            memK0.cons(),
            memA0.prod(),
            zero_kernel,
            matmul_qk_kernel,
            idx_buffer_qk,
        ],
        placement=qk_tile,
        stack_size=0xD00,
        while_true=False,
    )
    softmax_worker = Worker(
        softmax,
        fn_args=[
            memA0.cons(),
            memP0.prod(),
            scaleOF0.prod(),
            partial_softmax_kernel,
            scale_buffer_init_kernel,
            memcopy_kernel_scale,
            idx_buffer_softmax,
            scale_buffer_softmax,
        ],
        placement=softmax_tile,
        stack_size=0xD00,
        while_true=False,
    )
    pv_worker = Worker(
        batched_matmul_pv,
        fn_args=[
            memP0.cons(),
            memV0.cons(),
            scaleOF0.cons(),
            outOProj0.prod(),
            zero_kernel,
            matmul_pv_kernel,
            rescale_o_kernel,
            idx_buffer_pv,
        ],
        placement=pv_tile,
        stack_size=0xD00,
        while_true=False,
    )
    o_proj_worker = Worker(
        matmul_o_proj,
        fn_args=[
            outOProj0.cons(),
            memOW0.cons(),
            outOProjAccumIn0.cons(depth=1),
            outOProjAccumOut0.prod(),
            outOProjInput.prod(),
            o_proj_stats_sum_buffer,
            o_proj_stats_sumsq_buffer,
            ln_zero_f32_kernel,
            ln_calc_sum_sumsq_kernel,
            pack_stats_kernel,
            zero_kernel_o_proj,
            matmul_kernel_o_proj,
            mem_copy_o_proj,
        ],
        placement=o_proj_tile,
        stack_size=0xD00,
        while_true=False,
    )
    ln1_worker = Worker(
        core_fn_ln1_fused,
        fn_args=[
            outOProjInput.cons(),
            ln1Replay.cons(),
            ln1ReplayPart.prod(),
            memR.cons(),
            ln1_norm_sum_buffer,
            ln1_norm_sumsq_buffer,
            ln1_weight_buffer,
            outLNBroadcast.prod(),
            ln_fused_add_layer_norm_kernel,
            ln_calc_sum_sumsq_kernel,
            unpack_stats_kernel,
            ln_zero_f32_kernel,
            mem_copy_o_proj,
        ],
        placement=ln1_tile,
        while_true=False,
    )
    ffn_up_worker = Worker(
        core_fn_ffn_up_proj,
        fn_args=[
            memOutLN,
            memBUp.cons(),
            ffnUpOut.prod(),
            ffn_zero_kernel_up_proj,
            ffn_matmul_init_kernel_up_proj,
            ffn_matmul_kernel_up_proj,
            ffn_gelu_kernel,
        ],
        placement=ffn_up_tile,
        stack_size=0x700,
        while_true=False,
    )
    ffn_down_worker = Worker(
        core_fn_ffn_down_proj,
        fn_args=[
            ffnUpOut.cons(),
            memBDown.cons(),
            ffnDownAccum.cons(depth=1),
            ffnDownPart.prod(),
            ffnDownOut.prod(2),
            ffn_matmul_init_kernel_down_proj,
            ffn_matmul_kernel_down_proj,
            mem_copy_o_proj,
        ],
        placement=ffn_down_tile,
        stack_size=0xF00,
        while_true=False,
    )
    ln2_worker = Worker(
        core_fn_add_norm2,
        fn_args=[
            ffnDownOut.cons(),
            ffnRIn.cons(),
            ln2Replay.cons(depth=1),
            ln2ReplayPart.prod(),
            ln2_sum_buffer,
            ln2_sumsq_buffer,
            ln2_weight_buffer,
            outLN2.prod(),
            ln_fused_add_layer_norm_kernel,
            ln_calc_sum_sumsq_kernel,
            ln_zero_f32_kernel,
            mem_copy_o_proj,
        ],
        placement=ln2_tile,
        stack_size=0xF00,
        while_true=False,
    )

    qkv_tensor_shape = (3 * seq_len, embed_sz)
    q_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz), (seq_tile, d), (1, heads)
    )
    k_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz), (kv_seq_tile, d), (num_kv_seq_blocks, 1)
    )
    v_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz), (kv_seq_tile, d), (num_kv_seq_blocks, 1)
    )
    Q_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                qkv_tensor_shape,
                offset=tap.offset,
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in q_tiles_base
        ]
    )
    K_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                qkv_tensor_shape,
                offset=tap.offset + seq_len * embed_sz,
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in k_tiles_base
        ]
    )
    V_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                qkv_tensor_shape,
                offset=tap.offset + 2 * seq_len * embed_sz,
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in v_tiles_base
        ]
    )
    WO_tiles = TensorTiler2D.group_tiler(
        (embed_sz, embed_sz), (d, emb_tile), (1, embed_sz // emb_tile)
    )
    for tile in WO_tiles:
        tile._sizes = [tile._sizes[1], tile._sizes[0], tile._sizes[2], tile._sizes[3]]
        tile._strides = [
            tile._strides[1],
            tile._strides[0],
            tile._strides[2],
            tile._strides[3],
        ]
    o_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (seq_tile, emb_tile),
        (1, embed_sz // emb_tile // num_o_col_groups),
    )
    r_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (seq_tile, emb_tile),
        (1, embed_sz // emb_tile // num_o_col_groups),
    )
    O_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                or_tensor_shape,
                offset=tap.offset,
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in o_tiles_base
        ]
    )
    R_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                or_tensor_shape,
                offset=tap.offset + seq_len * embed_sz,
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in r_tiles_base
        ]
    )
    for tile in R_tiles:
        tile._sizes[0] = ln1_broadcast_groups
        tile._strides[0] = 0

    B_Up_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                (embed_sz, ffn_intermediate_size),
                offset=0,
                sizes=[ln1_broadcast_groups, proj_acc_depth, emb_tile, emb_tile],
                strides=[
                    emb_tile,
                    emb_tile * ffn_intermediate_size,
                    ffn_intermediate_size,
                    1,
                ],
            )
        ]
    )
    B_Down_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                (ffn_intermediate_size, embed_sz),
                offset=0,
                sizes=[ln1_broadcast_groups, proj_acc_depth, emb_tile, emb_tile],
                strides=[emb_tile * embed_sz, emb_tile, embed_sz, 1],
            )
        ]
    )

    def legalize_tap(tap: TensorAccessPattern, max_dim_size: int):
        sizes = list(tap._sizes)
        strides = list(tap._strides)
        if all(size <= max_dim_size for size in sizes):
            return
        i = len(sizes) - 1
        while i >= 0:
            if sizes[i] > max_dim_size:
                multiplier = 1
                while sizes[i] > max_dim_size:
                    sizes[i] //= 2
                    multiplier *= 2
                remainder = sizes[i]
                sizes.insert(i, multiplier)
                strides.insert(i, strides[i] * remainder)
                sizes[i + 1] = remainder
                i -= 1
            i -= 1
        while sizes and sizes[0] == 1:
            sizes.pop(0)
            strides.pop(0)
        if len(sizes) > 4:
            raise ValueError(
                f"Cannot legalize TAP into <=4 dimensions: sizes={sizes}, strides={strides}"
            )
        tap._sizes = sizes
        tap._strides = strides

    def legalize_tas(tas: TensorAccessSequence):
        for tap in tas:
            legalize_tap(tap, 1023)

    for tas in (
        Q_tiles,
        K_tiles,
        V_tiles,
        WO_tiles,
        O_tiles,
        R_tiles,
        B_Up_tiles,
        B_Down_tiles,
    ):
        legalize_tas(tas)

    for tap_seq, obj_shape, expected, message in (
        (Q_tiles, (seq_tile, d), heads, "Q tap count does not match heads"),
        (
            K_tiles,
            (kv_seq_tile, d),
            num_kv_seq_blocks,
            "K tap count does not match KV block count",
        ),
        (
            V_tiles,
            (kv_seq_tile, d),
            num_kv_seq_blocks,
            "V tap count does not match KV block count",
        ),
        (
            WO_tiles,
            (d, emb_tile),
            proj_acc_depth,
            "W_O tap count does not match proj_acc_depth",
        ),
        (
            R_tiles,
            (seq_tile, emb_tile),
            ln1_broadcast_groups * proj_acc_depth,
            "Residual tap count does not match LN1 broadcast groups",
        ),
        (
            O_tiles,
            (seq_tile, emb_tile),
            proj_acc_depth,
            "Output tap count does not match proj_acc_depth",
        ),
        (
            B_Up_tiles,
            (emb_tile, emb_tile),
            ln1_broadcast_groups * proj_acc_depth,
            "B_Up tap count does not match FFN loop count",
        ),
        (
            B_Down_tiles,
            (emb_tile, emb_tile),
            ln1_broadcast_groups * proj_acc_depth,
            "B_Down tap count does not match FFN loop count",
        ),
    ):
        if (
            math.prod(int(s) for s in tap_seq[0].sizes) // math.prod(obj_shape)
            != expected
        ):
            raise ValueError(message)

    rt = Runtime()
    with rt.sequence(W_O_ty, QKV_ty, OR_ty, B_Up_ty, B_Down_ty) as (
        W_O,
        QKV,
        OR,
        B_Up,
        B_Down,
    ):
        rt.start(qk_worker)
        rt.start(softmax_worker)
        rt.start(pv_worker)
        rt.start(o_proj_worker)
        rt.start(ln1_worker)
        rt.start(ffn_up_worker)
        rt.start(ffn_down_worker)
        rt.start(ln2_worker)

        pending_ln1_refill_tg = None
        pending_output_tap_idx = None

        for tap_idx in range(num_q_seq_blocks):
            if pending_ln1_refill_tg is not None:
                if pending_output_tap_idx is not None:
                    tg_out = rt.task_group()
                    rt.drain(
                        memLN2.cons(),
                        OR,
                        tap=O_tiles[pending_output_tap_idx],
                        placement=Tile(col=output_shim_col, row=0),
                        task_group=tg_out,
                        wait=True,
                    )
                    rt.finish_task_group(tg_out)
                    pending_output_tap_idx = None
                rt.finish_task_group(pending_ln1_refill_tg)
                pending_ln1_refill_tg = None

            tg_q = rt.task_group()
            rt.fill(
                inQ.prod(),
                QKV,
                tap=Q_tiles[tap_idx],
                placement=Tile(col=q_shim_col, row=0),
                task_group=tg_q,
                wait=False,
            )
            rt.finish_task_group(tg_q)

            for head_idx in range(heads):
                tg_head = rt.task_group()
                rt.fill(
                    inK.prod(),
                    QKV,
                    tap=K_tiles[head_idx],
                    placement=Tile(col=k_shim_col, row=0),
                    task_group=tg_head,
                    wait=True,
                )
                rt.fill(
                    inV.prod(),
                    QKV,
                    tap=V_tiles[head_idx],
                    placement=Tile(col=v_shim_col, row=0),
                    task_group=tg_head,
                    wait=True,
                )
                rt.fill(
                    inOW.prod(),
                    W_O,
                    tap=WO_tiles[head_idx],
                    placement=Tile(col=ow_shim_col, row=0),
                    task_group=tg_head,
                    wait=True,
                )
                rt.finish_task_group(tg_head)

            tg_tail = rt.task_group()
            rt.fill(
                inR.prod(),
                OR,
                tap=R_tiles[tap_idx],
                placement=Tile(col=residual_shim_col, row=0),
                task_group=tg_tail,
                wait=False,
            )

            ln1_stage_base_offset = 2 * seq_len * embed_sz
            branch_stage_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln1_stage_base_offset,
                sizes=[ln1_broadcast_groups, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            residual_stage_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln1_stage_base_offset,
                sizes=[1, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            if (
                math.prod(int(s) for s in branch_stage_tap.sizes)
                // math.prod((seq_tile, emb_tile))
                != ln1_broadcast_groups * proj_acc_depth
            ):
                raise ValueError("LN1 stage tap count mismatch")
            if (
                math.prod(int(s) for s in residual_stage_tap.sizes)
                // math.prod((seq_tile, emb_tile))
                != proj_acc_depth
            ):
                raise ValueError("LN1 residual stage tap count mismatch")

            tg_ln1_drain = rt.task_group()
            rt.drain(
                ln1StageOut,
                OR,
                tap=branch_stage_tap,
                placement=Tile(col=ln1_stage_shim_col, row=0),
                task_group=tg_ln1_drain,
                wait=True,
            )
            rt.finish_task_group(tg_ln1_drain)

            tg_ln1_refill_and_weights = rt.task_group()
            rt.fill(
                inLNFromDDR.prod(),
                OR,
                tap=branch_stage_tap,
                placement=Tile(col=ln1_stage_shim_col, row=0),
                task_group=tg_ln1_refill_and_weights,
                wait=True,
            )
            rt.fill(
                ffnRFromDDR.prod(),
                OR,
                tap=residual_stage_tap,
                placement=Tile(col=ln1_stage_shim_col, row=0),
                task_group=tg_ln1_refill_and_weights,
                wait=True,
            )
            rt.fill(
                inBUp.prod(),
                B_Up,
                tap=B_Up_tiles[0],
                placement=Tile(col=bup_shim_col, row=0),
                task_group=tg_ln1_refill_and_weights,
                wait=False,
            )
            rt.fill(
                inBDown.prod(),
                B_Down,
                tap=B_Down_tiles[0],
                placement=Tile(col=bdown_shim_col, row=0),
                task_group=tg_ln1_refill_and_weights,
                wait=False,
            )
            rt.finish_task_group(tg_tail)

            pending_ln1_refill_tg = tg_ln1_refill_and_weights
            pending_output_tap_idx = tap_idx

        if pending_output_tap_idx is not None:
            tg_out = rt.task_group()
            rt.drain(
                memLN2.cons(),
                OR,
                tap=O_tiles[pending_output_tap_idx],
                placement=Tile(col=output_shim_col, row=0),
                task_group=tg_out,
                wait=True,
            )
            rt.finish_task_group(tg_out)
        if pending_ln1_refill_tg is not None:
            rt.finish_task_group(pending_ln1_refill_tg)

    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


if __name__ == "__main__":
    main()
