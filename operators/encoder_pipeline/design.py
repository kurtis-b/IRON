# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encoder pipeline: MHA+AN → FFN+AN fused into a single AIE design.
#
# Data flow (no intermediate DRAM round-trip):
#   h1 = LN(MHA(Q,K,V) @ W_O, weight=ln1) + R       [MHA+AN block]
#   h2 = LN(GeLU(h1 @ B_Up) @ B_Down, weight=ln2) + h1  [FFN+AN block]
#
# h1 is passed from the MHA LN worker to the FFN block via a new
# ObjectFifo (mha_to_ffn), replacing the DRAM drain/fill round-trip.
#
# Design constraints (required for the pipeline tile sizes to be compatible):
#   m_ffn  = seq_tile           (FFN M-tile  = MHA sequence tile)
#   k_ffn  = emb_tile           (FFN K-tile  = MHA embedding tile)
#   down_proj_depth = K // k    (= embed_sz // emb_tile)
#   N = n_ffn * nB_tiles_distributed  (nC_up_col_tiles_per_core = 1)
#   nA_tiles_distributed = 1

import sys
import math
import argparse
import logging
from pathlib import Path

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
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern, TensorAccessSequence
from aie.helpers.dialects.ext.scf import if_, else_
import aie.dialects.index as index
from aie.dialects.aiex import *

base_dir = Path(__file__).parent

microkernel_mac_dim_map = {
    "npu2": {
        "bf16": {
            True: (8, 8, 8),   # emulate_bf16_mmul_with_bfp16
            False: (4, 8, 8),
        },
    },
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def encoder_pipeline(
    # MHA+AN parameters
    heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    emb_tile: int,
    o_proj_acc_depth: int,
    parallel_heads: int,
    emulate_bf16_mmul_with_bfp16: bool,
    kernel_archive_mha: str,
    ln1_weight_file=None,
    # FFN+AN parameters
    ffn_N: int = None,
    ffn_n: int = None,
    nB_tiles_distributed: int = 1,
    gelu_stage: int = 1,
    kernel_archive_ffn: str = None,
    ln2_weight_file=None,
    trace_size: int = 0,
):
    """
    Fused encoder pipeline AIE design.

    Combines operators/mha_to_an/design.py and operators/ffn_addnorm/design.py
    into a single Program.  h1 (MHA+AN output) is routed directly to the FFN+AN
    block via the 'mha_to_ffn' ObjectFifo — no intermediate DRAM transfer.

    Hardware layout (NPU2, 8 columns × 4 compute rows):
      Cols 0 .. parallel_heads-1 : MHA QK / softmax / PV / O-proj workers
      Col  parallel_heads        : MHA LN1 worker (placed at row 2)
      Cols parallel_heads+1 ..   : FFN up/down-proj + LN2 workers
    """

    # ------------------------------------------------------------------
    # Derived dimensions (shared between the two blocks)
    # ------------------------------------------------------------------
    embed_sz = heads * d          # = K (FFN hidden size)
    K = embed_sz
    N = ffn_N if ffn_N is not None else 4 * embed_sz
    n = ffn_n if ffn_n is not None else 64

    # Pipeline tile-size constraints
    m = seq_tile    # FFN m-tile must equal MHA seq_tile
    k = emb_tile    # FFN k-tile must equal MHA emb_tile

    down_proj_depth = K // k    # enforced by constraint
    K_div_k = K // k
    mem_tile_n = n * nB_tiles_distributed

    assert K % k == 0, f"embed_sz ({K}) must be divisible by emb_tile ({k})"
    assert K == k * down_proj_depth, (
        f"down_proj_depth must equal embed_sz // emb_tile = {K // k}"
    )
    assert N == mem_tile_n, (
        f"Pipeline constraint: N ({N}) must equal n*nB_tiles_distributed "
        f"({n}*{nB_tiles_distributed}={mem_tile_n}).  "
        f"Increase nB_tiles_distributed or n so they multiply to N."
    )

    nA_tiles_distributed = 1          # fixed; the pipeline streams one A copy
    nC_up_col_tiles_per_core = 1      # enforced by the N == mem_tile_n constraint
    num_qkv_seq_blocks = seq_len // seq_tile
    num_qkv_head_block_per_parallel_head = heads // parallel_heads
    num_o_col_groups = embed_sz // (emb_tile * o_proj_acc_depth)
    an_depth = embed_sz // emb_tile   # = K_div_k
    ln_iters_per_core = seq_len // (nA_tiles_distributed * m)
    nC_tiles_per_core = nC_up_col_tiles_per_core * ln_iters_per_core

    # MHA column layout
    mha_col_ln = parallel_heads        # LN1 worker column
    ffn_col_start = parallel_heads + 1 # first column available to FFN

    assert heads % parallel_heads == 0
    assert seq_tile % 8 == 0
    assert d % 8 == 0
    assert seq_len % seq_tile == 0

    inv_scale = (1 / np.sqrt(d)) * 1.4453125
    dtype = bfloat16
    dtype_str = "bf16"
    dev = "npu2"

    r, s, t = microkernel_mac_dim_map[dev][dtype_str][emulate_bf16_mmul_with_bfp16]

    # ------------------------------------------------------------------
    # Static weight files
    # ------------------------------------------------------------------
    if ln1_weight_file is None:
        static_ln1_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)

    if ln2_weight_file is None:
        static_ln2_weights = np.ones(K, dtype=bfloat16)
    else:
        static_ln2_weights = np.load(ln2_weight_file)

    # ------------------------------------------------------------------
    # Tensor type aliases
    # ------------------------------------------------------------------
    # DRAM types
    W_O_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    Q_ty   = np.ndarray[(seq_len, embed_sz),  np.dtype[dtype]]
    KV_ty  = np.ndarray[(seq_len, embed_sz),  np.dtype[dtype]]
    R_ty   = np.ndarray[(seq_len, embed_sz),  np.dtype[dtype]]  # MHA residual
    B_up_dram_ty   = np.ndarray[(K * N,), np.dtype[dtype]]
    B_down_dram_ty = np.ndarray[(N * K,), np.dtype[dtype]]
    C_ty   = np.ndarray[(seq_len * K,), np.dtype[dtype]]        # FFN+AN output

    # MHA on-chip types
    q_ty    = np.ndarray[(seq_tile, d),           np.dtype[dtype]]
    k_ty    = np.ndarray[(d, seq_tile),            np.dtype[dtype]]
    qk_ty   = np.ndarray[(seq_tile, seq_tile),    np.dtype[dtype]]
    s_ty    = np.ndarray[(4 * seq_tile,),          np.dtype[dtype]]
    wo_ty   = np.ndarray[(d, emb_tile),            np.dtype[dtype]]
    o_ty    = np.ndarray[(seq_tile, emb_tile),     np.dtype[dtype]]

    # FFN on-chip types  (m = seq_tile, k = emb_tile, n = ffn_n)
    A_l1_ty            = np.ndarray[(m, k),  np.dtype[dtype]]  # same shape as o_ty
    A_l2_ty            = np.ndarray[(m * k,), np.dtype[dtype]]
    B_l2_ty            = np.ndarray[(k * n,), np.dtype[dtype]]
    B_up_proj_l1_ty    = np.ndarray[(k, n),  np.dtype[dtype]]
    B_down_proj_l1_ty  = np.ndarray[(n, k),  np.dtype[dtype]]
    C_up_proj_l1_ty    = np.ndarray[(m, n),  np.dtype[dtype]]
    sum_l1_ty          = np.ndarray[(m,),    np.dtype[np.float32]]
    ln_weights_ty      = np.ndarray[(K,),    np.dtype[dtype]]

    # ------------------------------------------------------------------
    # MHA kernel declarations
    # ------------------------------------------------------------------
    mha_bin = kernel_archive_mha
    zero_kernel       = Kernel(f"zero_{dtype_str}",          mha_bin, [qk_ty])
    memcopy_scale     = Kernel("passThroughLine",             mha_bin, [s_ty, s_ty, np.int32])
    scale_buf_init    = Kernel("init_scale_buffer",           mha_bin, [s_ty, np.int32])
    partial_softmax   = Kernel("partial_softmax",             mha_bin,
                               [qk_ty, qk_ty, s_ty,
                                np.ndarray[(2,), np.dtype[np.int32]],
                                dtype, np.int32, np.int32, np.int32, np.int32])
    matmul_QK         = Kernel("matmul_bf16_bf16_wrapper",   mha_bin,
                               [q_ty, k_ty, qk_ty,
                                np.ndarray[(2,), np.dtype[np.int32]]])
    matmul_PV         = Kernel("matmul_PV",                  mha_bin,
                               [qk_ty, k_ty, qk_ty, s_ty, np.int32, np.int32,
                                np.ndarray[(2,), np.dtype[np.int32]]])
    rescale_O         = Kernel("rescale_O",                  mha_bin,
                               [qk_ty, s_ty, np.int32,
                                np.ndarray[(2,), np.dtype[np.int32]]])
    zero_o_proj       = Kernel(f"zero_{dtype_str}_o_proj",   mha_bin, [o_ty])
    matmul_o_proj     = Kernel("matmul_with_acc_bf16_bf16_o_proj", mha_bin,
                               [q_ty, wo_ty, o_ty, o_ty])
    memcopy_o_proj    = Kernel("passThroughLine_o_proj",     mha_bin,
                               [o_ty, o_ty, np.int32])
    eltwise_add_o_proj = Kernel("eltwise_add_bf16_vector_o_proj", mha_bin,
                                [o_ty, o_ty, o_ty, np.int32])

    # LN1 (MHA Add & Norm) kernels
    ln1_zero_f32      = Kernel("ln_zero_f32",                mha_bin, [sum_l1_ty, np.int32])
    ln1_calc_sumsq    = Kernel("ln_calc_sum_sumsq",          mha_bin, [o_ty, sum_l1_ty, sum_l1_ty])
    ln1_fused_add_ln  = Kernel("fused_add_layer_norm_1outs", mha_bin,
                               [o_ty, o_ty, ln_weights_ty, sum_l1_ty, sum_l1_ty,
                                o_ty, np.int32, np.int32])

    # ------------------------------------------------------------------
    # FFN kernel declarations
    # ------------------------------------------------------------------
    ffn_archive = (
        kernel_archive_ffn
        if kernel_archive_ffn is not None
        else f"ffn_{m}x{k}x{n}_archive.a"
    )
    ffn_zero_up    = Kernel(f"ffn_zero_{dtype_str}_up_proj",   ffn_archive, [C_up_proj_l1_ty])
    ffn_matmul_up  = Kernel(f"ffn_matmul_{dtype_str}_{dtype_str}_up_proj", ffn_archive,
                            [A_l1_ty, B_up_proj_l1_ty, C_up_proj_l1_ty])
    ffn_gelu       = Kernel("ffn_gelu_bf16", ffn_archive,
                            [C_up_proj_l1_ty, C_up_proj_l1_ty, np.int32])
    ffn_zero_down  = Kernel(f"ffn_zero_{dtype_str}_down_proj", ffn_archive, [A_l1_ty])
    ffn_matmul_down = Kernel(f"ffn_matmul_with_acc_{dtype_str}_{dtype_str}_down_proj",
                             ffn_archive,
                             [C_up_proj_l1_ty, B_down_proj_l1_ty, A_l1_ty, A_l1_ty])
    ffn_copy       = Kernel("ffn_passThroughLine", ffn_archive,
                            [A_l1_ty, A_l1_ty, np.int32])
    ffn_add        = Kernel("ffn_eltwise_add_bf16_vector", ffn_archive,
                            [A_l1_ty, A_l1_ty, A_l1_ty, np.int32])
    ln2_zero_f32   = Kernel("ln_zero_f32",                ffn_archive, [sum_l1_ty, np.int32])
    ln2_calc_sumsq = Kernel("ln_calc_sum_sumsq",          ffn_archive, [A_l1_ty, sum_l1_ty, sum_l1_ty])
    ln2_fused_add_ln = Kernel("fused_add_layer_norm_1outs", ffn_archive,
                              [A_l1_ty, A_l1_ty, ln_weights_ty,
                               sum_l1_ty, sum_l1_ty, A_l1_ty, np.int32, np.int32])

    # ==================================================================
    # ObjectFifo declarations
    # ==================================================================

    of_depth = 2

    # ------------------------------------------------------------------
    # MHA input FIFOs  (unchanged from mha_to_an/design.py)
    # ------------------------------------------------------------------
    q_dims = [(seq_tile // r, r * d), (d // s, s), (r, d), (s, 1)]
    inQ = ObjectFifo(
        np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inQ", depth=of_depth,
    )
    memQ = inQ.cons().split(
        offsets=[seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[q_ty] * parallel_heads,
        names=[f"memQ{i}" for i in range(parallel_heads)],
        dims_to_stream=[q_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=0, row=1),
    )

    k_dims = [(seq_tile // t, t * d), (d // s, s), (t, d), (s, 1)]
    inK = ObjectFifo(
        np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inK", depth=of_depth,
    )
    memK = inK.cons().split(
        offsets=[seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[k_ty] * parallel_heads,
        names=[f"memK{i}" for i in range(parallel_heads)],
        dims_to_stream=[k_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=1, row=1),
    )

    v_dims = [(seq_tile // s, s * seq_tile), (d // t, t), (s, seq_tile), (t, 1)]
    inV = ObjectFifo(
        np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inV", depth=of_depth,
    )
    memV = inV.cons().split(
        offsets=[seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[q_ty] * parallel_heads,
        names=[f"memV{i}" for i in range(parallel_heads)],
        dims_to_stream=[v_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=2, row=1),
    )

    ow_dims = [(d // s, s * emb_tile), (emb_tile // t, t), (s, emb_tile), (t, 1)]
    inOW = ObjectFifo(
        np.ndarray[(d * parallel_heads, emb_tile), np.dtype[dtype]],
        name="inOW", depth=of_depth,
    )
    memOW = inOW.cons().split(
        offsets=[d * emb_tile * i for i in range(parallel_heads)],
        obj_types=[wo_ty] * parallel_heads,
        names=[f"memOW{i}" for i in range(parallel_heads)],
        dims_to_stream=[ow_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=3, row=1),
    )

    # MHA residual R (original encoder input, for the MHA Add & Norm stage)
    o_dims = [(seq_tile // r, r * emb_tile), (r, t), (emb_tile // t, r * t), (t, 1)]
    inR_mha = ObjectFifo(
        np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]],
        name="inR_mha", depth=of_depth,
    )
    memR_mha = inR_mha.cons().forward(
        obj_type=o_ty, name="memR_mha",
        dims_to_stream=o_dims,
        placement=Tile(col=mha_col_ln, row=1),
    )

    # MHA intermediate FIFOs
    memA, memP, scaleOF = [], [], []
    a_dims_out = [(seq_tile // s, s), (seq_tile, r * s), (s, 1)]
    a_dims_in  = [(seq_tile // s, s), (seq_tile // s, seq_tile), (s, 1)]
    p_dims_out = [(seq_tile // s, s), (seq_tile, seq_tile), (s, 1)]
    p_dims_in  = [(seq_tile // s, r * s), (seq_tile // r, seq_tile * r), (r * s, 1)]
    for i in range(parallel_heads):
        memA.append(ObjectFifo(qk_ty, depth=of_depth, name=f"memA{i}",
                               dims_to_stream=a_dims_out,
                               dims_from_stream_per_cons=a_dims_in))
        memP.append(ObjectFifo(qk_ty, depth=of_depth, name=f"memP{i}",
                               dims_to_stream=p_dims_out,
                               dims_from_stream_per_cons=p_dims_in))
        scaleOF.append(ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}"))

    # O-proj partial accumulation FIFOs
    outOProj, outOProjAccumOut, outOProjAccumIn, outOPart = [], [], [], []
    for i in range(parallel_heads):
        outOProj.append(ObjectFifo(q_ty, depth=of_depth, name=f"outOProj{i}"))
        outOProjAccumOut.append(ObjectFifo(o_ty, depth=1, name=f"outOProjAccumOut{i}"))
        outOProjAccumIn.append(
            outOProjAccumOut[i].cons(depth=o_proj_acc_depth).forward(
                name=f"outOProjAccumIn{i}",
                depth=o_proj_acc_depth,
                placement=Tile(col=mha_col_ln - 1 - (i % 2), row=1),
            )
        )
    for i in range(parallel_heads - 1):
        outOPart.append(ObjectFifo(o_ty, depth=of_depth, name=f"outOPart{i}"))

    # FIFO from last O-proj core into LN1 worker
    outOToLN1 = ObjectFifo(
        np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]],
        name="outOToLN1", dims_to_stream=o_dims,
    )
    outO = outOToLN1.prod().join(
        offsets=[seq_tile * emb_tile],
        obj_types=[o_ty],
        names=["outO0"],
        depths=[of_depth],
        placement=Tile(col=mha_col_ln, row=1),
    )

    # ------------------------------------------------------------------
    # Connecting ObjectFifo: h1 from MHA LN1 → FFN block
    #
    # Two consumers:
    #   mha_to_ffn_a  →  FFN up-proj workers (h1 used as matrix A)
    #   mha_to_ffn_r  →  FFN LN2 worker      (h1 used as residual R)
    #
    # The LN1 core produces one o_ty = (seq_tile, emb_tile) = (m, k) tile
    # per inner loop iteration (an_depth iterations per M-tile row).
    # ------------------------------------------------------------------
    mha_to_ffn = ObjectFifo(
        o_ty,   # (seq_tile, emb_tile) == (m, k)
        name="mha_to_ffn",
        depth=of_depth,
    )
    # Forward to FFN A input (with microkernel dims_to_stream reordering)
    dims_to_stream_a = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    mha_to_ffn_a = mha_to_ffn.cons().forward(
        obj_type=A_l1_ty,
        name="mha_to_ffn_a",
        dims_to_stream=dims_to_stream_a,
        placement=Tile(col=ffn_col_start, row=1),
    )
    # Forward to FFN LN2 R input (same reordering, different MemTile channel)
    mha_to_ffn_r = mha_to_ffn.cons().forward(
        obj_type=A_l1_ty,
        name="mha_to_ffn_r",
        dims_to_stream=dims_to_stream_a,
        placement=Tile(col=ffn_col_start, row=1),
    )

    # ------------------------------------------------------------------
    # FFN FIFOs (B_Up, B_Down, intermediate C, LN2 output)
    # ------------------------------------------------------------------
    B_up_l3l2_fifos   = [None] * nB_tiles_distributed
    B_up_l2l1_fifos   = [None] * nB_tiles_distributed
    B_down_l3l2_fifos = [None] * nB_tiles_distributed
    B_down_l2l1_fifos = [None] * nB_tiles_distributed

    for b in range(nB_tiles_distributed):
        B_up_l3l2_fifos[b] = ObjectFifo(B_l2_ty, name=f"B_up_L3L2_{b}", depth=of_depth)
        dims_up = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_up_l2l1_fifos[b] = B_up_l3l2_fifos[b].cons().forward(
            obj_type=B_up_proj_l1_ty,
            name=f"B_up_L2L1_{b}",
            dims_to_stream=dims_up,
            placement=Tile((ffn_col_start + b + 1) % 8, 1),
        )

        B_down_l3l2_fifos[b] = ObjectFifo(B_l2_ty, name=f"B_down_L3L2_{b}", depth=of_depth)
        dims_down = [(n // s, s * k), (k // t, t), (s, k), (t, 1)]
        B_down_l2l1_fifos[b] = B_down_l3l2_fifos[b].cons().forward(
            obj_type=B_down_proj_l1_ty,
            name=f"B_down_L2L1_{b}",
            dims_to_stream=dims_down,
            placement=Tile((ffn_col_start + b + 1) % 8, 1),
        )

    # Up-proj output (piped to down-proj)
    C_up_l1l1_fifos = [None] * nB_tiles_distributed
    for b in range(nB_tiles_distributed):
        C_up_l1l1_fifos[b] = ObjectFifo(
            C_up_proj_l1_ty, name=f"C_up_L1L1_{b}", depth=of_depth,
        )

    # Down-proj partial accumulation (L1 ↔ MemTile)
    C_down_part_l1l2 = [None] * nB_tiles_distributed
    C_down_part_l2l1 = [None] * nB_tiles_distributed
    for b in range(nB_tiles_distributed):
        C_down_part_l1l2[b] = ObjectFifo(A_l1_ty, name=f"C_down_L1L2_{b}", depth=1)
        C_down_part_l2l1[b] = C_down_part_l1l2[b].cons(depth=down_proj_depth).forward(
            obj_type=A_l1_ty,
            name=f"C_down_L2L1_{b}",
            depth=down_proj_depth,
            placement=Tile((ffn_col_start + b) % 8, 1),
        )

    # Down-proj cross-core reduction FIFOs
    C_down_reduce = [None] * max(nB_tiles_distributed - 1, 0)
    for b in range(nB_tiles_distributed - 1):
        C_down_reduce[b] = ObjectFifo(A_l1_ty, name=f"C_down_L1L1_{b}", depth=of_depth)

    # Down-proj output → LN2
    C_down_out = ObjectFifo(A_l1_ty, name="C_out_L1L1_0", depth=of_depth)

    # LN2 output → DRAM
    dims_ln2_out = [(m // r, r * k), (r, s), (k // s, r * s), (s, 1)]
    ln2_l1l2 = ObjectFifo(A_l1_ty, name="ln2_L1L2_0", depth=of_depth)
    ln2_l2l3 = ln2_l1l2.cons().forward(
        obj_type=A_l1_ty,
        name="ln2_L2L3_0",
        dims_to_stream=dims_ln2_out,
        placement=Tile(7, 1),
    )

    # ==================================================================
    # Worker core functions
    # ==================================================================

    # ---- MHA workers (copied from mha_to_an/design.py) ---------------

    def batched_matmul_qk(of_q, of_k, of_a_out, zero, matmul_QK, q_block_bias, idx_buf):
        for _ in range_(sys.maxsize):
            idx_buf[0] = 0
            idx_buf[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_q = of_q.acquire(1)
                for _ in range_(num_qkv_seq_blocks):
                    elem_k   = of_k.acquire(1)
                    elem_a   = of_a_out.acquire(1)
                    zero(elem_a)
                    matmul_QK(elem_q, elem_k, elem_a, idx_buf)
                    of_k.release(1)
                    of_a_out.release(1)
                    idx_buf[0] += 0
                idx_buf[0] = 0
                idx_buf[1] += 0
                of_q.release(1)

    def softmax(of_a, of_p, of_scale, partial_softmax, init_scale, memcopy, q_bias,
                idx_buf, scale_buf):
        for _ in range_(sys.maxsize):
            idx_buf[0] = 0
            idx_buf[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                init_scale(scale_buf, seq_tile)
                for _ in range_(num_qkv_seq_blocks):
                    p   = of_p.acquire(1)
                    a   = of_a.acquire(1)
                    sc  = of_scale.acquire(1)
                    partial_softmax(a, p, scale_buf, idx_buf,
                                    inv_scale, seq_tile, seq_tile, seq_len, seq_len)
                    memcopy(scale_buf, sc, 4 * seq_tile)
                    of_a.release(1)
                    of_p.release(1)
                    of_scale.release(1)
                    idx_buf[0] += 0
                idx_buf[0] = 0
                idx_buf[1] += 0

    def batched_matmul_pv(of_p, of_v, of_scale, of_o_out, zero, matmul_PV, rescale_O,
                          q_bias, idx_buf):
        for _ in range_(sys.maxsize):
            idx_buf[0] = 0
            idx_buf[1] = 0
            for _ in range_(num_qkv_seq_blocks):
                elem_o = of_o_out.acquire(1)
                zero(elem_o)
                elem_p  = of_p.acquire(1)
                elem_v  = of_v.acquire(1)
                elem_sc = of_scale.acquire(1)
                matmul_PV(elem_p, elem_v, elem_o, elem_sc, seq_tile, 0, idx_buf)
                of_p.release(1);  of_v.release(1);  of_scale.release(1)
                idx_buf[0] += 0
                if num_qkv_seq_blocks > 2:
                    for _ in range_(num_qkv_seq_blocks - 2):
                        ep = of_p.acquire(1); ev = of_v.acquire(1); es = of_scale.acquire(1)
                        matmul_PV(ep, ev, elem_o, es, seq_tile, 1, idx_buf)
                        of_p.release(1); of_v.release(1); of_scale.release(1)
                        idx_buf[0] += 0
                if num_qkv_seq_blocks > 1:
                    ep = of_p.acquire(1); ev = of_v.acquire(1); es = of_scale.acquire(1)
                    matmul_PV(ep, ev, elem_o, es, seq_tile, 1, idx_buf)
                    rescale_O(elem_o, es, seq_tile, idx_buf)
                    of_p.release(1); of_v.release(1); of_scale.release(1)
                    idx_buf[0] += 0
                else:
                    rescale_O(elem_o, elem_sc, seq_tile, idx_buf)
                    idx_buf[0] += 0
                idx_buf[0] = 0
                idx_buf[1] += 0
                of_o_out.release(1)

    def matmul_o_proj_fn(of_o_in, of_ow, of_acc_in, of_acc_out, of_o_out,
                         buf_to_reduce, zero, matmul, add, copy, is_last):
        for _ in range_(sys.maxsize):
            for _ in range_(o_proj_acc_depth):
                e = of_acc_out.acquire(1);  zero(e);  of_acc_out.release(1)
            for _ in range_(num_qkv_head_block_per_parallel_head):
                e_o = of_o_in.acquire(1)
                for _ in range_(o_proj_acc_depth):
                    e_acc_in  = of_acc_in.acquire(1)
                    e_ow      = of_ow.acquire(1)
                    e_acc_out = of_acc_out.acquire(1)
                    matmul(e_o, e_ow, e_acc_in, e_acc_out)
                    of_acc_out.release(1);  of_ow.release(1);  of_acc_in.release(1)
                of_o_in.release(1)
            for _ in range_(o_proj_acc_depth):
                e_acc_in = of_acc_in.acquire(1)
                if buf_to_reduce:
                    partial = buf_to_reduce.acquire(1)
                    add(partial, e_acc_in, e_acc_in, seq_tile * emb_tile)
                    buf_to_reduce.release(1)
                e_out = of_o_out.acquire(1)
                copy(e_acc_in, e_out, seq_tile * emb_tile)
                if is_last:
                    e_new = of_acc_out.acquire(1)
                    copy(e_out, e_new, seq_tile * emb_tile)
                    of_acc_out.release(1)
                of_acc_in.release(1);  of_o_out.release(1)
            if is_last:
                for _ in range_(o_proj_acc_depth):
                    e_acc_in = of_acc_in.acquire(1)
                    e_out    = of_o_out.acquire(1)
                    copy(e_acc_in, e_out, seq_tile * emb_tile)
                    of_acc_in.release(1);  of_o_out.release(1)

    def core_fn_ln1(of_in1, of_in2, sum_buf, sumsq_buf, weights, of_out,
                    fused_ln, calc_sumsq, zero_f32):
        """
        MHA Add & Norm (LN1).  Produces h1 tiles into of_out (= mha_to_ffn).
        """
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for _ in range_(an_depth):
                e1 = of_in1.acquire(1)
                calc_sumsq(e1, sum_buf, sumsq_buf)
                of_in1.release(1)
            for col_idx in range_(an_depth):
                col_i32 = index.casts(T.i32(), col_idx)
                e1   = of_in1.acquire(1)
                e2   = of_in2.acquire(1)
                eout = of_out.acquire(1)
                fused_ln(e1, e2, weights, sum_buf, sumsq_buf, eout, embed_sz, col_i32)
                of_out.release(1)
                of_in1.release(1)
                of_in2.release(1)

    # ---- FFN workers (adapted from ffn_addnorm/design.py) -------------

    def core_fn_up_proj(in_a, in_b, out_c, zero, matmul, gelu):
        loop = range(1) if nC_tiles_per_core <= 1 else range_(nC_tiles_per_core)
        for _ in loop:
            e_out = out_c.acquire(1)
            zero(e_out)
            for _ in range_(K_div_k):
                e_a = in_a.acquire(1)
                e_b = in_b.acquire(1)
                matmul(e_a, e_b, e_out)
                in_a.release(1);  in_b.release(1)
            if gelu:
                gelu(e_out, e_out, m * n)
            out_c.release(1)

    def core_fn_down_proj(in_a, in_b, curr_acc, new_acc, out_acc,
                          zero, matmul, add, copy, gelu,
                          buf_to_reduce, is_end):
        for _ in range_(down_proj_depth):
            e = new_acc.acquire(1);  zero(e);  new_acc.release(1)
        for _ in range_(nC_up_col_tiles_per_core):
            e_a = in_a.acquire(1)
            if gelu:
                gelu(e_a, e_a, m * n)
            for _ in range_(down_proj_depth):
                e_curr   = curr_acc.acquire(1)
                e_b      = in_b.acquire(1)
                e_new    = new_acc.acquire(1)
                matmul(e_a, e_b, e_curr, e_new)
                new_acc.release(1);  in_b.release(1);  curr_acc.release(1)
            in_a.release(1)
        for _ in range_(down_proj_depth):
            e_curr  = curr_acc.acquire(1)
            e_out   = out_acc.acquire(1)
            if buf_to_reduce:
                partial = buf_to_reduce.acquire(1)
                add(partial, e_curr, e_curr, m * k)
                buf_to_reduce.release(1)
            copy(e_curr, e_out, m * k)
            if is_end:
                e_new = new_acc.acquire(1)
                copy(e_out, e_new, m * k)
                new_acc.release(1)
            out_acc.release(1);  curr_acc.release(1)
        if is_end:
            for _ in range_(down_proj_depth):
                e_curr = curr_acc.acquire(1)
                e_out  = out_acc.acquire(1)
                copy(e_curr, e_out, m * k)
                out_acc.release(1);  curr_acc.release(1)

    def core_fn_ln2(of_in1, of_in2, sum_buf, sumsq_buf, weights, of_out,
                    fused_ln, calc_sumsq, zero_f32):
        """
        FFN Add & Norm (LN2).
        of_in1 = FFN down-proj output (read twice: once for sum/sumsq, once for norm+add)
        of_in2 = h1 residual from mha_to_ffn_r (read once for norm+add)
        """
        zero_f32(sum_buf, m)
        zero_f32(sumsq_buf, m)
        for _ in range_(down_proj_depth):
            e1 = of_in1.acquire(1)
            calc_sumsq(e1, sum_buf, sumsq_buf)
            of_in1.release(1)
        for col_idx in range_(down_proj_depth):
            col_i32 = index.casts(T.i32(), col_idx)
            e1   = of_in1.acquire(1)
            e2   = of_in2.acquire(1)
            eout = of_out.acquire(1)
            fused_ln(e1, e2, weights, sum_buf, sumsq_buf, eout, K, col_i32)
            of_out.release(1);  of_in1.release(1);  of_in2.release(1)

    # ==================================================================
    # Worker instantiation
    # ==================================================================
    workers = []

    # ---- MHA compute workers -----------------------------------------
    for i in range(parallel_heads):
        idx_qk = Buffer(initial_value=np.zeros(2, dtype=np.int32), name=f"idx_qk_{i}")
        workers.append(Worker(
            batched_matmul_qk,
            fn_args=[memQ[i].cons(), memK[i].cons(), memA[i].prod(),
                     zero_kernel, matmul_QK, i, idx_qk],
            stack_size=0xD00, placement=Tile(col=i, row=2), while_true=False,
        ))
        idx_sm = Buffer(initial_value=np.zeros(2, dtype=np.int32), name=f"idx_sm_{i}")
        scale_buf = Buffer(initial_value=np.zeros(4 * seq_tile, dtype=dtype),
                           name=f"scale_buf_{i}")
        workers.append(Worker(
            softmax,
            fn_args=[memA[i].cons(), memP[i].prod(), scaleOF[i].prod(),
                     partial_softmax, scale_buf_init, memcopy_scale,
                     i, idx_sm, scale_buf],
            stack_size=0xD00, placement=Tile(col=i, row=3), while_true=False,
        ))
        idx_pv = Buffer(initial_value=np.zeros(2, dtype=np.int32), name=f"idx_pv_{i}")
        workers.append(Worker(
            batched_matmul_pv,
            fn_args=[memP[i].cons(), memV[i].cons(), scaleOF[i].cons(),
                     outOProj[i].prod(), zero_kernel, matmul_PV, rescale_O,
                     i, idx_pv],
            stack_size=0xD00, placement=Tile(col=i, row=4), while_true=False,
        ))
        workers.append(Worker(
            matmul_o_proj_fn,
            fn_args=[
                outOProj[i].cons(),
                memOW[i].cons(),
                outOProjAccumIn[i].cons(depth=1),
                outOProjAccumOut[i].prod(),
                outOPart[i].prod() if i < parallel_heads - 1 else outO[0].prod(),
                outOPart[i - 1].cons() if i > 0 else None,
                zero_o_proj, matmul_o_proj, eltwise_add_o_proj, memcopy_o_proj,
                i == parallel_heads - 1,
            ],
            stack_size=0xD00, placement=Tile(col=i, row=5), while_true=False,
        ))

    # ---- MHA LN1 worker (produces h1 into mha_to_ffn) ---------------
    ln1_weights_buf = Buffer(type=ln_weights_ty, initial_value=static_ln1_weights,
                             name="ln1_weights")
    ln1_sum    = Buffer(type=sum_l1_ty, name="ln1_sum")
    ln1_sumsq  = Buffer(type=sum_l1_ty, name="ln1_sumsq")
    workers.append(Worker(
        core_fn_ln1,
        fn_args=[outOToLN1.cons(), memR_mha.cons(),
                 ln1_sum, ln1_sumsq, ln1_weights_buf,
                 mha_to_ffn.prod(),          # ← output goes into connecting FIFO
                 ln1_fused_add_ln, ln1_calc_sumsq, ln1_zero_f32],
        placement=Tile(col=mha_col_ln, row=2), while_true=False,
    ))

    # ---- FFN workers -------------------------------------------------
    # With nB_tiles_distributed cores in the down-proj reduction chain:
    for b in range(nB_tiles_distributed):
        stream_to_ln2 = (b == nB_tiles_distributed - 1)

        if stream_to_ln2:
            ln2_weights_buf = Buffer(type=ln_weights_ty, initial_value=static_ln2_weights,
                                     name="ln2_weights_0")
            ln2_sum   = Buffer(type=sum_l1_ty, name="ln2_sum_0")
            ln2_sumsq = Buffer(type=sum_l1_ty, name="ln2_sumsq_0")
            workers.append(Worker(
                core_fn_ln2,
                fn_args=[C_down_out.cons(),
                         mha_to_ffn_r.cons(),   # ← h1 residual from connecting FIFO
                         ln2_sum, ln2_sumsq, ln2_weights_buf,
                         ln2_l1l2.prod(),
                         ln2_fused_add_ln, ln2_calc_sumsq, ln2_zero_f32],
                placement=Tile(col=ffn_col_start + nB_tiles_distributed, row=2),
                stack_size=0xF00,
            ))

        # Up-proj worker
        workers.append(Worker(
            core_fn_up_proj,
            fn_args=[mha_to_ffn_a.cons(),       # ← h1 as A from connecting FIFO
                     B_up_l2l1_fifos[b].cons(),
                     C_up_l1l1_fifos[b].prod(),
                     ffn_zero_up, ffn_matmul_up,
                     ffn_gelu if gelu_stage == 0 else None],
            placement=Tile(col=ffn_col_start + b, row=3),
            stack_size=0xF00,
        ))

        # Down-proj worker
        workers.append(Worker(
            core_fn_down_proj,
            fn_args=[
                C_up_l1l1_fifos[b].cons(),
                B_down_l2l1_fifos[b].cons(),
                C_down_part_l2l1[b].cons(depth=1),
                C_down_part_l1l2[b].prod(),
                (C_down_out.prod(of_depth)
                 if stream_to_ln2
                 else C_down_reduce[b].prod(of_depth)),
                ffn_zero_down, ffn_matmul_down, ffn_add, ffn_copy,
                ffn_gelu if gelu_stage == 1 else None,
                (None if b == 0 else C_down_reduce[b - 1].cons()),
                stream_to_ln2,
            ],
            placement=Tile(col=ffn_col_start + b, row=4),
            stack_size=0xF00,
        ))

    # ==================================================================
    # Runtime sequence
    # ==================================================================
    rt = Runtime()

    with rt.sequence(W_O_ty, Q_ty, KV_ty, KV_ty, R_ty,
                     B_up_dram_ty, B_down_dram_ty, C_ty) as (
        W_O, Q, K_dram, V_dram, R_mha, B_Up, B_Down, C
    ):
        rt.start(*workers)

        # Tile access patterns (from mha_to_an/design.py)
        Q_tiles = TensorTiler2D.group_tiler(
            (seq_len, embed_sz), (seq_tile, d), (1, heads),
        )
        K_tiles = TensorTiler2D.group_tiler(
            (seq_len, embed_sz), (seq_tile, d), (num_qkv_seq_blocks, parallel_heads),
        )
        V_tiles = TensorTiler2D.group_tiler(
            (seq_len, embed_sz), (seq_tile, d), (num_qkv_seq_blocks, parallel_heads),
        )
        WO_tiles = TensorTiler2D.group_tiler(
            (embed_sz, embed_sz), (d, emb_tile),
            (parallel_heads, embed_sz // emb_tile // num_o_col_groups),
        )
        for tile in WO_tiles:
            tile._sizes  = [tile._sizes[1],  tile._sizes[0],  tile._sizes[2],  tile._sizes[3]]
            tile._strides = [tile._strides[1], tile._strides[0], tile._strides[2], tile._strides[3]]

        R_mha_tiles = TensorTiler2D.group_tiler(
            (seq_len, embed_sz), (seq_tile, emb_tile),
            (1, embed_sz // emb_tile // num_o_col_groups),
        )

        # FFN B tiles (from ffn_addnorm/design.py)
        # B_Up:  (K, N) stored as (N, K) col-major, tiled (k, n)
        # B_Down: (N, K) stored as (K, N) col-major, tiled (n, k)

        for q_idx in range(num_qkv_seq_blocks):
            for col_group in range(num_o_col_groups):
                tg = rt.task_group()

                rt.fill(inQ.prod(), Q, tap=Q_tiles[q_idx],
                        placement=Tile(0, 0), task_group=tg)

                for h_idx in range(heads // parallel_heads):
                    tg_h = rt.task_group()
                    rt.fill(inK.prod(), K_dram, tap=K_tiles[h_idx],
                            placement=Tile(1, 0), task_group=tg_h, wait=True)
                    rt.fill(inV.prod(), V_dram, tap=V_tiles[h_idx],
                            placement=Tile(2, 0), task_group=tg_h, wait=True)
                    rt.fill(inOW.prod(), W_O,
                            tap=WO_tiles[h_idx * num_o_col_groups + col_group],
                            placement=Tile(3, 0), task_group=tg_h, wait=True)
                    rt.finish_task_group(tg_h)

                rt.fill(inR_mha.prod(), R_mha,
                        tap=R_mha_tiles[q_idx * num_o_col_groups + col_group],
                        placement=Tile(mha_col_ln, 0), task_group=tg)

                # h1 flows from LN1 → mha_to_ffn → FFN workers → LN2 → ln2_l2l3
                # No intermediate DRAM drain/fill for h1.

                # FFN B_Up tiles: reuse stride-0 across K for each (m, k) A row tile.
                # For nC_up_col_tiles_per_core=1 there is exactly one N-column pass.
                for b in range(nB_tiles_distributed):
                    B_up_col_offset = b * n
                    B_up_tile = TensorAccessPattern(
                        (N, K),
                        offset=B_up_col_offset,
                        sizes=[nC_up_col_tiles_per_core, K_div_k, k, n],
                        strides=[mem_tile_n, k * N, N, 1],
                    )
                    rt.fill(B_up_l3l2_fifos[b].prod(), B_Up, tap=B_up_tile,
                            placement=Tile((ffn_col_start + b + 1) % 8, 0),
                            task_group=tg)

                    B_down_col_offset = b * n * K
                    B_down_tile = TensorAccessPattern(
                        (K, N),
                        offset=B_down_col_offset,
                        sizes=[nC_up_col_tiles_per_core, down_proj_depth, n, k],
                        strides=[mem_tile_n * K, k, K, 1],
                    )
                    rt.fill(B_down_l3l2_fifos[b].prod(), B_Down, tap=B_down_tile,
                            placement=Tile((ffn_col_start + b + 1) % 8, 0),
                            task_group=tg)

                # Drain final h2 output from LN2
                C_offset = (q_idx * m + 0) * K   # one M-tile row of embed_sz elements
                C_tile = TensorAccessPattern(
                    (seq_len, K),
                    offset=C_offset,
                    sizes=[1, down_proj_depth, m, k],
                    strides=[0, k, K, 1],
                )
                rt.drain(ln2_l2l3.cons(), C, tap=C_tile, wait=True,
                         placement=Tile(7, 0), task_group=tg)

                rt.finish_task_group(tg)

    # ==================================================================
    # Build and return program
    # ==================================================================
    my_program = Program(NPU2(), rt)
    module = my_program.resolve_program(SequentialPlacer())
    return module


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Encoder pipeline AIE design (MHA+AN → FFN+AN, pipelined via ObjectFifo)"
    )
    ap.add_argument("--heads",              type=int,   default=12)
    ap.add_argument("--seq-len",            type=int,   default=512)
    ap.add_argument("-d",                   type=int,   default=64)
    ap.add_argument("--seq-tile",           type=int,   default=64)
    ap.add_argument("--emb-tile",           type=int,   default=96)
    ap.add_argument("--o-proj-acc-depth",   type=int,   default=8)
    ap.add_argument("--parallel-heads",     type=int,   default=4)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=True)
    ap.add_argument("--kernel-archive-mha", type=str,   default="mha_kernels.a")
    ap.add_argument("--ln1-weight-file",    type=str,   default=None)
    ap.add_argument("--ffn-N",              type=int,   default=None)
    ap.add_argument("--ffn-n",              type=int,   default=96)
    ap.add_argument("--nB-tiles-distributed", type=int, default=1)
    ap.add_argument("--gelu-stage",         type=int,   default=1)
    ap.add_argument("--kernel-archive-ffn", type=str,   default=None)
    ap.add_argument("--ln2-weight-file",    type=str,   default=None)
    ap.add_argument("--trace-size",         type=int,   default=0)
    ap.add_argument("-o", "--output-file-path", type=str,
                    default=str(base_dir / "build" / "encoder_pipeline.mlir"))
    args = ap.parse_args()

    module = encoder_pipeline(
        heads=args.heads,
        seq_len=args.seq_len,
        d=args.d,
        seq_tile=args.seq_tile,
        emb_tile=args.emb_tile,
        o_proj_acc_depth=args.o_proj_acc_depth,
        parallel_heads=args.parallel_heads,
        emulate_bf16_mmul_with_bfp16=args.emulate_bf16_mmul_with_bfp16,
        kernel_archive_mha=args.kernel_archive_mha,
        ln1_weight_file=args.ln1_weight_file,
        ffn_N=args.ffn_N,
        ffn_n=args.ffn_n,
        nB_tiles_distributed=args.nB_tiles_distributed,
        gelu_stage=args.gelu_stage,
        kernel_archive_ffn=args.kernel_archive_ffn,
        ln2_weight_file=args.ln2_weight_file,
        trace_size=args.trace_size,
    )

    out = Path(args.output_file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(str(module))
    logging.info(f"Written to {out}")


if __name__ == "__main__":
    main()
