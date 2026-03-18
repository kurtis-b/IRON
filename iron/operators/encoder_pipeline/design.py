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
from .placements import TOPOLOGY_PLACEMENTS

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
    argparser.add_argument("--ffn-tile", type=int, default=None)
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
        ffn_tile=args.ffn_tile,
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
    ffn_tile: int | None,
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
    if ffn_tile is None:
        ffn_tile = emb_tile

    topology_key = (
        heads,
        seq_len,
        d,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        ffn_tile,
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
    sequence_parallel = placement["sequence_parallel"]

    if trace_size != 0:
        raise ValueError("encoder_pipeline does not support tracing")
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
    if num_q_seq_blocks % parallel_seq != 0:
        raise ValueError(
            "encoder_pipeline requires num_q_seq_blocks divisible by parallel_seq "
            f"({num_q_seq_blocks} % {parallel_seq} != 0)"
        )
    q_blocks_per_lane = num_q_seq_blocks // parallel_seq
    if heads % parallel_heads != 0:
        raise ValueError(
            "encoder_pipeline requires heads divisible by parallel_heads "
            f"({heads} % {parallel_heads} != 0)"
        )
    num_qkv_head_block_per_parallel_head = heads // parallel_heads
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
    if ffn_tile % 16 != 0:
        raise ValueError(f"ffn_tile must be divisible by 16 ({ffn_tile} % 16 != 0)")
    if ffn_intermediate_size % ffn_tile != 0:
        raise ValueError(
            "ffn_intermediate_size must be divisible by ffn_tile "
            f"({ffn_intermediate_size} % {ffn_tile} != 0)"
        )
    if len(placement["mha_cols"]) != parallel_heads:
        raise ValueError(
            "encoder_pipeline hardcoded placement must provide one MHA column per "
            f"parallel head (cols={placement['mha_cols']}, parallel_heads={parallel_heads})"
        )

    dtype = bfloat16
    inv_scale = (1 / np.sqrt(d)) * 1.4453125
    of_depth = 2
    weight_forward_depth = 2 if ffn_tile <= 64 else 1
    ln1_broadcast_groups = ffn_intermediate_size // ffn_tile
    effective_ffn_branches = len(placement["tail_tiles"]["ffn_up_by_branch"])
    if effective_ffn_branches != nB_tiles_distributed:
        raise ValueError(
            "encoder_pipeline hardcoded placement must provide one FFN branch per "
            "requested nB_tiles_distributed "
            f"(branches={effective_ffn_branches}, requested={nB_tiles_distributed})"
        )
    if ln1_broadcast_groups % effective_ffn_branches != 0:
        raise ValueError(
            "encoder_pipeline requires FFN branch count to divide ln1_broadcast_groups "
            f"({ln1_broadcast_groups} % {effective_ffn_branches} != 0)"
        )
    if sequence_parallel is not None:
        if len(sequence_parallel["lane_tiles"]) != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one lane "
                f"tile map per sequence lane ({len(sequence_parallel['lane_tiles'])} "
                f"!= {parallel_seq})"
            )
        if len(sequence_parallel["lane_o_proj_acc_mem_cols"]) != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one "
                "O-proj accumulation memtile per sequence lane "
                f"({len(sequence_parallel['lane_o_proj_acc_mem_cols'])} != {parallel_seq})"
            )
        if len(sequence_parallel["lane_tail_mem_cols"]) != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one tail "
                f"memtile per sequence lane ({len(sequence_parallel['lane_tail_mem_cols'])} "
                f"!= {parallel_seq})"
            )
        lane_ffn_down_acc_mem_cols = sequence_parallel.get("lane_ffn_down_acc_mem_cols")
        if (
            lane_ffn_down_acc_mem_cols is not None
            and len(lane_ffn_down_acc_mem_cols) != parallel_seq
        ):
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one "
                "FFN-down accumulation memtile per sequence lane "
                f"({len(lane_ffn_down_acc_mem_cols)} != {parallel_seq})"
            )
        transport_groups = sequence_parallel.get("transport_groups")
        if transport_groups is not None:
            grouped_lanes = tuple(
                lane_idx for group in transport_groups for lane_idx in group["lanes"]
            )
            if tuple(sorted(grouped_lanes)) != tuple(range(parallel_seq)):
                raise ValueError(
                    "encoder_pipeline sequence-parallel transport groups must cover "
                    f"every lane exactly once (got {grouped_lanes})"
                )
            group_sizes = {len(group["lanes"]) for group in transport_groups}
            if len(group_sizes) != 1:
                raise ValueError(
                    "encoder_pipeline sequence-parallel transport groups must have "
                    f"uniform size (got {sorted(group_sizes)})"
                )
    ffn_col_group_count = ln1_broadcast_groups // effective_ffn_branches
    ln_tiles_per_q_block = proj_acc_depth
    use_split_ln1_inputs = sequence_parallel is None and parallel_heads > 1
    ln1_dram_stage_rows = (
        seq_len if sequence_parallel is not None else proj_acc_depth * seq_tile
    )
    or_tensor_shape = (2 * seq_len + ln1_dram_stage_rows, embed_sz)

    if ln1_weight_file is None:
        static_ln1_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
    if ln2_weight_file is None:
        static_ln2_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln2_weights = np.load(ln2_weight_file)

    qk_tiles = [Tile(col=col, row=2) for col in placement["mha_cols"]]
    softmax_tiles = [Tile(col=col, row=3) for col in placement["mha_cols"]]
    pv_tiles = [Tile(col=col, row=4) for col in placement["mha_cols"]]

    def normalize_slot_coords(value, expected_count, name):
        if expected_count == 1:
            if isinstance(value[0], int):
                return [tuple(value)]
            if len(value) != 1:
                raise ValueError(
                    f"{name} must provide exactly one placement (got {value})"
                )
            return [tuple(value[0])]
        if isinstance(value[0], int):
            raise ValueError(
                f"{name} must provide {expected_count} placements (got {value})"
            )
        if len(value) != expected_count:
            raise ValueError(
                f"{name} must provide {expected_count} placements (got {value})"
            )
        return [tuple(coord) for coord in value]

    def normalize_slot_mem_cols(value, expected_count, name):
        if expected_count == 1:
            if isinstance(value, int):
                return [value]
            if len(value) != 1:
                raise ValueError(
                    f"{name} must provide exactly one memtile column (got {value})"
                )
            return [int(value[0])]
        if isinstance(value, int):
            raise ValueError(
                f"{name} must provide {expected_count} memtile columns (got {value})"
            )
        if len(value) != expected_count:
            raise ValueError(
                f"{name} must provide {expected_count} memtile columns (got {value})"
            )
        return [int(col) for col in value]

    o_proj_tiles = [Tile(col=col, row=5) for col in placement["mha_cols"]]
    ln1_tile = Tile(*placement["tail_tiles"]["ln1"])
    ffn_up_tiles = [Tile(*tile) for tile in placement["tail_tiles"]["ffn_up_by_branch"]]
    ffn_down_tiles = [
        Tile(*tile) for tile in placement["tail_tiles"]["ffn_down_by_branch"]
    ]
    ln2_tile = Tile(*placement["tail_tiles"]["ln2"])
    accumulation_mem_tiles = placement["accumulation_mem_tiles"]
    weight_mem_tiles = placement["weight_mem_tiles"]

    q_mem_col = placement["mem_tiles"]["q"]
    k_mem_col = placement["mem_tiles"]["k"]
    v_mem_col = placement["mem_tiles"]["v"]
    ow_mem_col = placement["mem_tiles"]["w_o"]
    o_proj_stage_mem_col = accumulation_mem_tiles["o_proj_stage"]
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
    q_stream_ty = np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]]
    kv_stream_ty = np.ndarray[(kv_seq_tile, d * parallel_heads), np.dtype[dtype]]
    wo_stream_ty = np.ndarray[(d * parallel_heads, emb_tile), np.dtype[dtype]]

    q_ty = np.ndarray[(seq_tile, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, kv_seq_tile), np.dtype[dtype]]
    qk_ty = np.ndarray[(seq_tile, kv_seq_tile), np.dtype[dtype]]
    v_ty = np.ndarray[(kv_seq_tile, d), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * seq_tile,), np.dtype[dtype]]
    wo_ty = np.ndarray[(d, emb_tile), np.dtype[dtype]]
    o_ty = np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]]
    ffn_up_ty = np.ndarray[(seq_tile, ffn_tile), np.dtype[dtype]]
    ffn_b_up_ty = np.ndarray[(emb_tile, ffn_tile), np.dtype[dtype]]
    ffn_b_down_ty = np.ndarray[(ffn_tile, emb_tile), np.dtype[dtype]]
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
    eltwise_add_vector_kernel = Kernel(
        "eltwise_add_bf16_vector_o_proj",
        kernel_archive,
        [o_ty, o_ty, o_ty, np.int32],
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
    ffn_zero_kernel_up_proj = Kernel(
        "ffn_zero_bf16_up_proj", kernel_archive, [ffn_up_ty]
    )
    ffn_zero_kernel_down_proj = Kernel(
        "ffn_zero_bf16_down_proj", kernel_archive, [o_ty]
    )
    ffn_matmul_init_kernel_up_proj = Kernel(
        "ffn_matmul_init_bf16_bf16_up_proj",
        kernel_archive,
        [o_ty, ffn_b_up_ty, ffn_up_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        "ffn_matmul_bf16_bf16_up_proj",
        kernel_archive,
        [o_ty, ffn_b_up_ty, ffn_up_ty],
    )
    ffn_matmul_init_kernel_down_proj = Kernel(
        "ffn_matmul_init_bf16_bf16_down_proj",
        kernel_archive,
        [ffn_up_ty, ffn_b_down_ty, o_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        "ffn_matmul_with_acc_bf16_bf16_down_proj",
        kernel_archive,
        [ffn_up_ty, ffn_b_down_ty, o_ty, o_ty],
    )
    ffn_gelu_kernel = Kernel(
        "ffn_gelu_bf16", kernel_archive, [ffn_up_ty, ffn_up_ty, np.int32]
    )

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
    b_up_dims = [
        (emb_tile // 8, 8 * ffn_tile),
        (ffn_tile // 8, 8),
        (8, ffn_tile),
        (8, 1),
    ]
    b_down_dims = [
        (ffn_tile // 8, 8 * emb_tile),
        (emb_tile // 8, 8),
        (8, emb_tile),
        (8, 1),
    ]

    inQ = ObjectFifo(q_stream_ty, name="inQ", depth=of_depth)
    memQ = inQ.cons().split(
        offsets=[seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[q_ty] * parallel_heads,
        names=[f"memQ{i}" for i in range(parallel_heads)],
        dims_to_stream=[q_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=q_mem_col, row=1),
    )
    inK = ObjectFifo(kv_stream_ty, name="inK", depth=of_depth)
    memK = inK.cons().split(
        offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[k_ty] * parallel_heads,
        names=[f"memK{i}" for i in range(parallel_heads)],
        dims_to_stream=[k_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=k_mem_col, row=1),
    )
    inV = ObjectFifo(kv_stream_ty, name="inV", depth=of_depth)
    memV = inV.cons().split(
        offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[v_ty] * parallel_heads,
        names=[f"memV{i}" for i in range(parallel_heads)],
        dims_to_stream=[v_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=v_mem_col, row=1),
    )
    inOW = ObjectFifo(wo_stream_ty, name="inOW", depth=of_depth)
    memOW = inOW.cons().split(
        offsets=[d * emb_tile * i for i in range(parallel_heads)],
        obj_types=[wo_ty] * parallel_heads,
        names=[f"memOW{i}" for i in range(parallel_heads)],
        dims_to_stream=[ow_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=ow_mem_col, row=1),
    )

    memA = [
        ObjectFifo(
            qk_ty,
            depth=of_depth,
            name=f"memA{i}",
            dims_to_stream=a_dims_out,
            dims_from_stream_per_cons=a_dims_in,
        )
        for i in range(parallel_heads)
    ]
    memP = [
        ObjectFifo(
            qk_ty,
            depth=of_depth,
            name=f"memP{i}",
            dims_to_stream=p_dims_out,
            dims_from_stream_per_cons=p_dims_in,
        )
        for i in range(parallel_heads)
    ]
    scaleOF = [
        ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}")
        for i in range(parallel_heads)
    ]
    outOProj = [
        ObjectFifo(q_ty, depth=of_depth, name=f"outOProj{i}")
        for i in range(parallel_heads)
    ]
    outOProjAccumOut = [
        ObjectFifo(o_ty, depth=1, name=f"outOProjAccumOut{i}")
        for i in range(parallel_heads)
    ]
    outOProjAccumIn = [
        outOProjAccumOut[i]
        .cons(depth=proj_acc_depth)
        .forward(
            obj_type=o_ty,
            name=f"outOProjAccumIn{i}",
            depth=proj_acc_depth,
            placement=Tile(
                col=accumulation_mem_tiles["o_proj_acc_by_head"][i],
                row=1,
            ),
        )
        for i in range(parallel_heads)
    ]
    o_proj_stage_chain_depth = 1 if emb_tile >= 128 else of_depth
    outOPart = [
        ObjectFifo(o_ty, depth=o_proj_stage_chain_depth, name=f"outOPart{i}")
        for i in range(parallel_heads - 1)
    ]
    outOProjInput = ObjectFifo(o_ty, name="outOProjInput", depth=2)

    if use_split_ln1_inputs:
        oProjStagePart = ObjectFifo(o_ty, name="oProjStagePart", depth=1)
        oProjStage = oProjStagePart.cons(depth=ln_tiles_per_q_block).forward(
            obj_type=o_ty,
            name="oProjStage",
            depth=ln_tiles_per_q_block,
            placement=Tile(col=o_proj_stage_mem_col, row=1),
        )
    else:
        oProjStagePart = None
        oProjStage = None
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
    memOutLN = [inLNFromDDR.cons(depth=2) for _ in range(effective_ffn_branches)]

    ffnRFromDDR = ObjectFifo(o_ty, name="ffnRFromDDR", depth=1)
    ffnRIn = ffnRFromDDR.cons(depth=2).forward(
        obj_type=o_ty,
        name="ffnRIn",
        depth=2,
        placement=Tile(col=ffn_residual_mem_col, row=1),
    )
    inBUp = []
    memBUp = []
    inBDown = []
    memBDown = []
    ffnUpOut = []
    ffnDownPart = []
    ffnDownAccum = []
    for branch_idx in range(effective_ffn_branches):
        inBUp.append(
            ObjectFifo(
                ffn_b_up_ty,
                name="inBUp" if branch_idx == 0 else f"inBUp{branch_idx}",
                depth=1,
            )
        )
        memBUp.append(
            inBUp[branch_idx]
            .cons()
            .forward(
                obj_type=ffn_b_up_ty,
                name="memBUp" if branch_idx == 0 else f"memBUp{branch_idx}",
                dims_to_stream=b_up_dims,
                depth=weight_forward_depth,
                placement=Tile(
                    col=weight_mem_tiles["b_up_by_branch"][branch_idx],
                    row=1,
                ),
            )
        )
        inBDown.append(
            ObjectFifo(
                ffn_b_down_ty,
                name="inBDown" if branch_idx == 0 else f"inBDown{branch_idx}",
                depth=1,
            )
        )
        memBDown.append(
            inBDown[branch_idx]
            .cons()
            .forward(
                obj_type=ffn_b_down_ty,
                name="memBDown" if branch_idx == 0 else f"memBDown{branch_idx}",
                dims_to_stream=b_down_dims,
                depth=weight_forward_depth,
                placement=Tile(
                    col=weight_mem_tiles["b_down_by_branch"][branch_idx],
                    row=1,
                ),
            )
        )
        ffnUpOut.append(
            ObjectFifo(
                ffn_up_ty,
                name="ffnUpOut" if branch_idx == 0 else f"ffnUpOut{branch_idx}",
                depth=2,
            )
        )
        ffnDownPart.append(
            ObjectFifo(
                o_ty,
                name="ffnDownPart" if branch_idx == 0 else f"ffnDownPart{branch_idx}",
                depth=1,
            )
        )
        ffnDownAccum.append(
            ffnDownPart[branch_idx]
            .cons(depth=proj_acc_depth)
            .forward(
                obj_type=o_ty,
                name=(
                    "ffnDownAccum" if branch_idx == 0 else f"ffnDownAccum{branch_idx}"
                ),
                depth=proj_acc_depth,
                placement=Tile(
                    col=accumulation_mem_tiles["ffn_down_acc_by_branch"][branch_idx],
                    row=1,
                ),
            )
        )
    ffn_down_reduce_depth = 1 if emb_tile >= 128 else 2
    ffnDownReduce = [
        ObjectFifo(
            o_ty,
            name=f"ffnDownReduce{branch_idx}",
            depth=ffn_down_reduce_depth,
        )
        for branch_idx in range(effective_ffn_branches - 1)
    ]
    ffnDownOut = ObjectFifo(o_ty, name="ffnDownOut", depth=2)

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
        buffer_to_reduce,
        of_o_out,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        add,
        zero,
        matmul,
        copy,
        emit_ln1_stats,
    ):
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
                if buffer_to_reduce is not None:
                    partial_o_acc = buffer_to_reduce.acquire(1)
                    add(
                        partial_o_acc,
                        elem_in_o_acc,
                        elem_in_o_acc,
                        seq_tile * emb_tile,
                    )
                    buffer_to_reduce.release(1)
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

    def matmul_o_proj_emit_stats_first(
        of_o_in,
        of_ow_in,
        of_o_acc_in,
        of_o_acc_out,
        buffer_to_reduce,
        of_o_out,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        add,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
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
                if buffer_to_reduce is not None:
                    partial_o_acc = buffer_to_reduce.acquire(1)
                    add(
                        partial_o_acc,
                        elem_in_o_acc,
                        elem_in_o_acc,
                        seq_tile * emb_tile,
                    )
                    buffer_to_reduce.release(1)
                calc_sum_sumsq(elem_in_o_acc, stats_sum_buf, stats_sumsq_buf)
                elem_out_o_acc = of_o_acc_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o_acc, seq_tile * emb_tile)
                of_o_acc_out.release(1)
                of_o_acc_in.release(1)
            elem_stats = of_o_out.acquire(1)
            pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats, seq_tile)
            of_o_out.release(1)
            for _ in range_(proj_acc_depth):
                elem_in_o_acc = of_o_acc_in.acquire(1)
                elem_out_o = of_o_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                of_o_out.release(1)
                of_o_acc_in.release(1)

    def matmul_o_proj_stage_then_emit_stats(
        of_o_in,
        of_ow_in,
        of_o_acc_in,
        of_o_acc_out,
        buffer_to_reduce,
        of_stage_out,
        of_stats_out,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        add,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
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
                if buffer_to_reduce is not None:
                    partial_o_acc = buffer_to_reduce.acquire(1)
                    add(
                        partial_o_acc,
                        elem_in_o_acc,
                        elem_in_o_acc,
                        seq_tile * emb_tile,
                    )
                    buffer_to_reduce.release(1)
                calc_sum_sumsq(elem_in_o_acc, stats_sum_buf, stats_sumsq_buf)
                elem_stage = of_stage_out.acquire(1)
                copy(elem_in_o_acc, elem_stage, seq_tile * emb_tile)
                of_stage_out.release(1)
                of_o_acc_in.release(1)
            elem_stats = of_stats_out.acquire(1)
            pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats, seq_tile)
            of_stats_out.release(1)

    def core_fn_ln1_stage_from_stats(
        of_in_o_proj,
        of_in_residual,
        sum_buf,
        sumsq_buf,
        weights,
        of_out_stage,
        fused_add_layer_norm,
        unpack_stats,
    ):
        for _ in range_(sys.maxsize):
            elem_stats = of_in_o_proj.acquire(1)
            unpack_stats(elem_stats, sum_buf, sumsq_buf, seq_tile)
            of_in_o_proj.release(1)
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                elem_out = of_out_stage.acquire(1)
                fused_add_layer_norm(
                    elem_in,
                    elem_residual,
                    weights,
                    sum_buf,
                    sumsq_buf,
                    elem_out,
                    embed_sz,
                    col_i32,
                )
                of_out_stage.release(1)
                of_in_o_proj.release(1)
                of_in_residual.release(1)

    def core_fn_ln1_stage_from_split_inputs(
        of_in_stats,
        of_in_o_proj,
        of_in_residual,
        sum_buf,
        sumsq_buf,
        weights,
        of_out_stage,
        fused_add_layer_norm,
        unpack_stats,
    ):
        for _ in range_(sys.maxsize):
            elem_stats = of_in_stats.acquire(1)
            unpack_stats(elem_stats, sum_buf, sumsq_buf, seq_tile)
            of_in_stats.release(1)
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                elem_out = of_out_stage.acquire(1)
                fused_add_layer_norm(
                    elem_in,
                    elem_residual,
                    weights,
                    sum_buf,
                    sumsq_buf,
                    elem_out,
                    embed_sz,
                    col_i32,
                )
                of_out_stage.release(1)
                of_in_o_proj.release(1)
                of_in_residual.release(1)

    def core_fn_ffn_up_proj(
        of_in_a, of_in_b, of_out_c, zero, matmul_init, matmul, gelu, group_count
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(group_count):
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
                gelu(elem_out, elem_out, seq_tile * ffn_tile)
                of_out_c.release(1)

    def core_fn_ffn_down_proj(
        of_in_a,
        of_in_b,
        of_curr_acc,
        of_new_acc,
        reduce_in,
        of_out,
        matmul_init,
        matmul,
        add,
        copy,
        group_count,
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
            for _ in range_(group_count - 1):
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
                if reduce_in is not None:
                    elem_reduce = reduce_in.acquire(1)
                    add(elem_reduce, elem_curr_acc, elem_curr_acc, seq_tile * emb_tile)
                    reduce_in.release(1)
                copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                of_curr_acc.release(1)
                of_out.release(1)

    def core_fn_ffn_down_proj_emit_stats_first(
        of_in_a,
        of_in_b,
        of_curr_acc,
        of_new_acc,
        reduce_in,
        of_out,
        sum_buf,
        sumsq_buf,
        matmul_init,
        matmul,
        add,
        calc_sum_sumsq,
        pack_stats,
        zero_f32,
        copy,
        group_count,
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
            for _ in range_(group_count - 1):
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

            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_curr_acc = of_curr_acc.acquire(1)
                if reduce_in is not None:
                    elem_reduce = reduce_in.acquire(1)
                    add(elem_reduce, elem_curr_acc, elem_curr_acc, seq_tile * emb_tile)
                    reduce_in.release(1)
                calc_sum_sumsq(elem_curr_acc, sum_buf, sumsq_buf)
                elem_new_acc = of_new_acc.acquire(1)
                copy(elem_curr_acc, elem_new_acc, seq_tile * emb_tile)
                of_new_acc.release(1)
                of_curr_acc.release(1)

            elem_stats = of_out.acquire(1)
            pack_stats(sum_buf, sumsq_buf, elem_stats, seq_tile)
            of_out.release(1)
            for _ in range_(proj_acc_depth):
                elem_curr_acc = of_curr_acc.acquire(1)
                elem_out = of_out.acquire(1)
                copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                of_curr_acc.release(1)
                of_out.release(1)

    def core_fn_add_norm2_from_stats(
        of_in1,
        of_in2,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        fused_add_layer_norm,
        unpack_stats,
    ):
        for _ in range_(sys.maxsize):
            elem_stats = of_in1.acquire(1)
            unpack_stats(elem_stats, sum_buf, sumsq_buf, seq_tile)
            of_in1.release(1)
            for col_idx in range_(proj_acc_depth):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_ffn = of_in1.acquire(1)
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
                of_in1.release(1)

    if sequence_parallel is not None:
        default_group_shim_cols = sequence_parallel.get(
            "shim_cols",
            {
                "q": q_shim_col,
                "k": k_shim_col,
                "v": v_shim_col,
                "w_o": ow_shim_col,
                "b_up": bup_shim_col,
                "b_down": bdown_shim_col,
            },
        )
        default_group_joined_or_shim_cols = sequence_parallel.get(
            "joined_or_shim_cols",
            {
                "residual": residual_shim_col,
                "ln1_stage": ln1_stage_shim_col,
                "ln1_refill": q_shim_col,
                "ffn_residual_refill": k_shim_col,
                "output": output_shim_col,
            },
        )

        transport_groups = sequence_parallel.get("transport_groups")
        if transport_groups is None:
            transport_groups = (
                {
                    "lanes": tuple(range(parallel_seq)),
                    "joined_q_mem_col": sequence_parallel["joined_q_mem_col"],
                    "shared_ingress_cols": sequence_parallel["shared_ingress_cols"],
                    "joined_or_mem_cols": sequence_parallel["joined_or_mem_cols"],
                    "shim_cols": default_group_shim_cols,
                    "joined_or_shim_cols": default_group_joined_or_shim_cols,
                    "weight_mem_cols": {
                        "b_up": tuple(weight_mem_tiles["b_up_by_branch"]),
                        "b_down": tuple(weight_mem_tiles["b_down_by_branch"]),
                    },
                },
            )
        group_count = len(transport_groups)
        group_size = len(transport_groups[0]["lanes"])
        if group_count * group_size != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel transport groups must partition "
                f"the sequence lanes ({group_count} * {group_size} != {parallel_seq})"
            )
        if emb_tile < 2 * seq_tile:
            raise ValueError(
                "encoder_pipeline seq-par no-replay LN contract requires "
                f"emb_tile >= 2 * seq_tile ({emb_tile} < {2 * seq_tile})"
            )
        unified_qr_split = sequence_parallel.get("unified_qr_split")
        unified_shared_streams = sequence_parallel.get("unified_shared_streams", {})
        if unified_qr_split is not None and (
            parallel_heads != 1 or effective_ffn_branches != 1
        ):
            raise ValueError(
                "encoder_pipeline unified seq-par shim sharing currently supports "
                "only parallel_heads=1 and nB_tiles_distributed=1 "
                f"(got parallel_heads={parallel_heads}, "
                f"nB_tiles_distributed={effective_ffn_branches})"
            )

        q_batch_ty = np.ndarray[
            (group_size * seq_tile, d * parallel_heads), np.dtype[dtype]
        ]
        joined_o_ty = np.ndarray[(group_size * seq_tile, emb_tile), np.dtype[dtype]]
        unified_q_batch_ty = (
            np.ndarray[(parallel_seq * seq_tile, d * parallel_heads), np.dtype[dtype]]
            if unified_qr_split is not None
            else None
        )
        unified_joined_o_ty = (
            np.ndarray[(parallel_seq * seq_tile, emb_tile), np.dtype[dtype]]
            if unified_qr_split is not None
            else None
        )
        joined_o_dims = [
            (group_size * seq_tile // 8, 8 * emb_tile),
            (8, 8),
            (emb_tile // 8, 8 * 8),
            (8, 1),
        ]
        q_offsets = [
            (head_idx * group_size + lane_idx) * seq_tile * d
            for head_idx in range(parallel_heads)
            for lane_idx in range(group_size)
        ]
        joined_offsets = [
            lane_idx * seq_tile * emb_tile for lane_idx in range(group_size)
        ]
        unified_q_offsets = [
            lane_idx * seq_tile * d for lane_idx in range(parallel_seq)
        ]
        unified_joined_offsets = [
            lane_idx * seq_tile * emb_tile for lane_idx in range(parallel_seq)
        ]
        lane_tiles = sequence_parallel["lane_tiles"]
        lane_front_tiles = []
        lane_tail_tiles = []
        for lane_idx, lane_cfg in enumerate(lane_tiles):
            lane_front_tiles.append(
                {
                    "qk": normalize_slot_coords(
                        lane_cfg["qk"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].qk",
                    ),
                    "softmax": normalize_slot_coords(
                        lane_cfg["softmax"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].softmax",
                    ),
                    "pv": normalize_slot_coords(
                        lane_cfg["pv"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].pv",
                    ),
                    "o_proj": normalize_slot_coords(
                        lane_cfg["o_proj"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].o_proj",
                    ),
                }
            )
            lane_tail_tiles.append(
                {
                    "ln1": tuple(lane_cfg["ln1"]),
                    "ffn_up": normalize_slot_coords(
                        lane_cfg["ffn_up"],
                        effective_ffn_branches,
                        f"sequence_parallel.lane_tiles[{lane_idx}].ffn_up",
                    ),
                    "ffn_down": normalize_slot_coords(
                        lane_cfg["ffn_down"],
                        effective_ffn_branches,
                        f"sequence_parallel.lane_tiles[{lane_idx}].ffn_down",
                    ),
                    "ln2": tuple(lane_cfg["ln2"]),
                }
            )
        lane_o_proj_acc_mem_cols = [
            normalize_slot_mem_cols(
                cols,
                parallel_heads,
                f"sequence_parallel.lane_o_proj_acc_mem_cols[{lane_idx}]",
            )
            for lane_idx, cols in enumerate(
                sequence_parallel["lane_o_proj_acc_mem_cols"]
            )
        ]
        lane_o_proj_stage_mem_cols = sequence_parallel.get("lane_o_proj_stage_mem_cols")
        if lane_o_proj_stage_mem_cols is not None:
            lane_o_proj_stage_mem_cols = [
                int(col) for col in lane_o_proj_stage_mem_cols
            ]
        if parallel_heads > 1 and lane_o_proj_stage_mem_cols is None:
            raise ValueError(
                "encoder_pipeline mixed seq-par with parallel_heads > 1 requires "
                "sequence_parallel.lane_o_proj_stage_mem_cols"
            )
        if (
            lane_o_proj_stage_mem_cols is not None
            and len(lane_o_proj_stage_mem_cols) != parallel_seq
        ):
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one "
                "O-proj stage memtile per sequence lane "
                f"({len(lane_o_proj_stage_mem_cols)} != {parallel_seq})"
            )
        lane_tail_mem_cols = sequence_parallel["lane_tail_mem_cols"]
        lane_ffn_down_acc_mem_cols = sequence_parallel.get(
            "lane_ffn_down_acc_mem_cols", lane_tail_mem_cols
        )
        lane_ffn_down_acc_mem_cols = [
            normalize_slot_mem_cols(
                cols,
                effective_ffn_branches,
                f"sequence_parallel.lane_ffn_down_acc_mem_cols[{lane_idx}]",
            )
            for lane_idx, cols in enumerate(lane_ffn_down_acc_mem_cols)
        ]
        shared_forward_depth = max(of_depth, group_size)

        lane_to_group_idx = [-1] * parallel_seq
        lane_to_local_idx = [-1] * parallel_seq
        for group_idx, transport_group in enumerate(transport_groups):
            for local_idx, lane_idx in enumerate(transport_group["lanes"]):
                lane_to_group_idx[lane_idx] = group_idx
                lane_to_local_idx[lane_idx] = local_idx

        group_inQSeq = []
        group_inKSeq = []
        group_memKSeq = []
        group_inVSeq = []
        group_memVSeq = []
        group_inOWSeq = []
        group_memOWSeq = []
        group_inBUpSeq = []
        group_memBUpSeq = []
        group_inBDownSeq = []
        group_memBDownSeq = []
        group_weight_b_up_shim_cols = []
        group_weight_b_down_shim_cols = []
        group_inRSeq = []
        group_ln1StageJoined = []
        group_inLNSeq = []
        group_ffnRFromDDRSeq = []
        group_memLN2Joined = []
        memQ_by_lane = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_in_r = [None] * parallel_seq
        lane_ln1_stage_out = [None] * parallel_seq
        lane_memOutLN = [None] * parallel_seq
        lane_ffn_r_in = [None] * parallel_seq

        lane_memA = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_memP = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_scaleOF = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_outOProj = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_o_proj_acc_in = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_o_proj_acc_out = [[None] * parallel_heads for _ in range(parallel_seq)]
        lane_o_proj_stage_part = [None] * parallel_seq
        lane_o_proj_stage = [None] * parallel_seq
        lane_o_proj_partial = [
            [None] * max(parallel_heads - 1, 0) for _ in range(parallel_seq)
        ]
        lane_ln1_input = []
        lane_ffn_up_out = [[None] * effective_ffn_branches for _ in range(parallel_seq)]
        lane_ffn_down_part = [
            [None] * effective_ffn_branches for _ in range(parallel_seq)
        ]
        lane_ffn_down_acc = [
            [None] * effective_ffn_branches for _ in range(parallel_seq)
        ]
        lane_ffn_down_reduce = [
            [None] * max(effective_ffn_branches - 1, 0) for _ in range(parallel_seq)
        ]
        lane_ffn_down_out = [None] * parallel_seq
        lane_qk_workers = []
        lane_softmax_workers = []
        lane_pv_workers = []
        lane_o_proj_workers = []
        lane_ln1_workers = []
        lane_ffn_up_workers = []
        lane_ffn_down_workers = []
        lane_ln2_workers = []

        lane_outLN2 = [None] * parallel_seq
        if unified_qr_split is not None:
            unified_inQSeq = ObjectFifo(
                unified_q_batch_ty, name="inQSeqAll", depth=of_depth
            )
            unified_memQ = unified_inQSeq.cons().split(
                offsets=unified_q_offsets,
                obj_types=[q_ty] * parallel_seq,
                names=[f"memQSeqL{lane_idx}" for lane_idx in range(parallel_seq)],
                dims_to_stream=[q_dims] * parallel_seq,
                depths=[of_depth] * parallel_seq,
                placement=Tile(col=unified_qr_split["q_mem_col"], row=1),
            )
            for lane_idx in range(parallel_seq):
                memQ_by_lane[lane_idx][0] = unified_memQ[lane_idx]

            unified_inRSeq = ObjectFifo(
                unified_joined_o_ty, name="inRSeqAll", depth=of_depth
            )
            unified_lane_in_r = unified_inRSeq.cons().split(
                offsets=unified_joined_offsets,
                obj_types=[o_ty] * parallel_seq,
                names=[f"inRSeqL{lane_idx}" for lane_idx in range(parallel_seq)],
                dims_to_stream=[r_dims] * parallel_seq,
                depths=[of_depth] * parallel_seq,
                placement=Tile(col=unified_qr_split["residual_mem_col"], row=1),
            )
            for lane_idx in range(parallel_seq):
                lane_in_r[lane_idx] = unified_lane_in_r[lane_idx]
        else:
            unified_inQSeq = None
            unified_inRSeq = None

        unified_inOWSeq = None
        unified_memOWSeq = None
        if "w_o" in unified_shared_streams:
            w_o_cfg = unified_shared_streams["w_o"]
            unified_inOWSeq = ObjectFifo(wo_stream_ty, name="inOWAll", depth=of_depth)
            unified_memOWSeq = unified_inOWSeq.cons().forward(
                obj_type=wo_ty,
                name="memOWSeqAll",
                dims_to_stream=ow_dims,
                depth=max(of_depth, parallel_seq),
                placement=Tile(col=w_o_cfg["mem_col"], row=1),
            )

        unified_inBUpSeq = None
        unified_memBUpSeq = None
        if "b_up" in unified_shared_streams:
            b_up_cfg = unified_shared_streams["b_up"]
            unified_inBUpSeq = ObjectFifo(ffn_b_up_ty, name="inBUpAll", depth=1)
            unified_memBUpSeq = unified_inBUpSeq.cons().forward(
                obj_type=ffn_b_up_ty,
                name="memBUpSeqAll",
                dims_to_stream=b_up_dims,
                depth=weight_forward_depth,
                placement=Tile(col=b_up_cfg["mem_col"], row=1),
            )

        for group_idx, transport_group in enumerate(transport_groups):
            lanes = transport_group["lanes"]
            suffix = "" if group_count == 1 else f"G{group_idx}"
            group_shim_cols = transport_group.get("shim_cols", default_group_shim_cols)
            group_joined_or_shim_cols = transport_group.get(
                "joined_or_shim_cols", default_group_joined_or_shim_cols
            )

            if unified_qr_split is None:
                inQSeq = ObjectFifo(q_batch_ty, name=f"inQSeq{suffix}", depth=of_depth)
                group_inQSeq.append(inQSeq)
                memQ_group = inQSeq.cons().split(
                    offsets=q_offsets,
                    obj_types=[q_ty] * (group_size * parallel_heads),
                    names=[
                        f"memQSeqL{lane_idx}H{head_idx}"
                        for head_idx in range(parallel_heads)
                        for lane_idx in lanes
                    ],
                    dims_to_stream=[q_dims] * (group_size * parallel_heads),
                    depths=[of_depth] * (group_size * parallel_heads),
                    placement=Tile(col=transport_group["joined_q_mem_col"], row=1),
                )
                for head_idx in range(parallel_heads):
                    for local_idx, lane_idx in enumerate(lanes):
                        memQ_by_lane[lane_idx][head_idx] = memQ_group[
                            head_idx * group_size + local_idx
                        ]
            else:
                group_inQSeq.append(None)

            inKSeq = ObjectFifo(kv_stream_ty, name=f"inK{suffix}", depth=of_depth)
            group_inKSeq.append(inKSeq)
            if parallel_heads == 1:
                memKSeq = inKSeq.cons().forward(
                    obj_type=k_ty,
                    name=f"memKSeq{suffix}",
                    dims_to_stream=k_dims,
                    depth=shared_forward_depth,
                    placement=Tile(
                        col=transport_group["shared_ingress_cols"]["k"], row=1
                    ),
                )
                group_memKSeq.append([memKSeq])
            else:
                memKSeq = inKSeq.cons().split(
                    offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
                    obj_types=[k_ty] * parallel_heads,
                    names=[f"memKSeq{suffix}H{i}" for i in range(parallel_heads)],
                    dims_to_stream=[k_dims] * parallel_heads,
                    depths=[shared_forward_depth] * parallel_heads,
                    placement=Tile(
                        col=transport_group["shared_ingress_cols"]["k"], row=1
                    ),
                )
                group_memKSeq.append(list(memKSeq))

            inVSeq = ObjectFifo(kv_stream_ty, name=f"inV{suffix}", depth=of_depth)
            group_inVSeq.append(inVSeq)
            if parallel_heads == 1:
                memVSeq = inVSeq.cons().forward(
                    obj_type=v_ty,
                    name=f"memVSeq{suffix}",
                    dims_to_stream=v_dims,
                    depth=shared_forward_depth,
                    placement=Tile(
                        col=transport_group["shared_ingress_cols"]["v"], row=1
                    ),
                )
                group_memVSeq.append([memVSeq])
            else:
                memVSeq = inVSeq.cons().split(
                    offsets=[kv_seq_tile * d * i for i in range(parallel_heads)],
                    obj_types=[v_ty] * parallel_heads,
                    names=[f"memVSeq{suffix}H{i}" for i in range(parallel_heads)],
                    dims_to_stream=[v_dims] * parallel_heads,
                    depths=[shared_forward_depth] * parallel_heads,
                    placement=Tile(
                        col=transport_group["shared_ingress_cols"]["v"], row=1
                    ),
                )
                group_memVSeq.append(list(memVSeq))

            if unified_memOWSeq is None:
                inOWSeq = ObjectFifo(wo_stream_ty, name=f"inOW{suffix}", depth=of_depth)
                group_inOWSeq.append(inOWSeq)
                if parallel_heads == 1:
                    memOWSeq = inOWSeq.cons().forward(
                        obj_type=wo_ty,
                        name=f"memOWSeq{suffix}",
                        dims_to_stream=ow_dims,
                        depth=shared_forward_depth,
                        placement=Tile(
                            col=transport_group["shared_ingress_cols"]["w_o"], row=1
                        ),
                    )
                    group_memOWSeq.append([memOWSeq])
                else:
                    memOWSeq = inOWSeq.cons().split(
                        offsets=[d * emb_tile * i for i in range(parallel_heads)],
                        obj_types=[wo_ty] * parallel_heads,
                        names=[f"memOWSeq{suffix}H{i}" for i in range(parallel_heads)],
                        dims_to_stream=[ow_dims] * parallel_heads,
                        depths=[shared_forward_depth] * parallel_heads,
                        placement=Tile(
                            col=transport_group["shared_ingress_cols"]["w_o"], row=1
                        ),
                    )
                    group_memOWSeq.append(list(memOWSeq))
            else:
                group_inOWSeq.append(None)
                group_memOWSeq.append([unified_memOWSeq])

            weight_b_up_cols = normalize_slot_mem_cols(
                transport_group["weight_mem_cols"]["b_up"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].weight_mem_cols.b_up",
            )
            weight_b_down_cols = normalize_slot_mem_cols(
                transport_group["weight_mem_cols"]["b_down"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].weight_mem_cols.b_down",
            )
            weight_b_up_shim_cols = normalize_slot_mem_cols(
                group_shim_cols["b_up"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].shim_cols.b_up",
            )
            weight_b_down_shim_cols = normalize_slot_mem_cols(
                group_shim_cols["b_down"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].shim_cols.b_down",
            )
            group_weight_b_up_shim_cols.append(weight_b_up_shim_cols)
            group_weight_b_down_shim_cols.append(weight_b_down_shim_cols)

            group_inBUpSeq.append([])
            group_memBUpSeq.append([])
            for branch_idx in range(effective_ffn_branches):
                branch_suffix = (
                    f"{suffix}B{branch_idx}" if group_count > 1 else f"B{branch_idx}"
                )
                if unified_memBUpSeq is None:
                    inBUpSeq = ObjectFifo(
                        ffn_b_up_ty, name=f"inBUpSeq{branch_suffix}", depth=1
                    )
                    memBUpSeq = inBUpSeq.cons().forward(
                        obj_type=ffn_b_up_ty,
                        name=f"memBUpSeq{branch_suffix}",
                        dims_to_stream=b_up_dims,
                        depth=weight_forward_depth,
                        placement=Tile(col=weight_b_up_cols[branch_idx], row=1),
                    )
                    group_inBUpSeq[group_idx].append(inBUpSeq)
                    group_memBUpSeq[group_idx].append(memBUpSeq)
                else:
                    group_inBUpSeq[group_idx].append(None)
                    group_memBUpSeq[group_idx].append(unified_memBUpSeq)

            group_inBDownSeq.append([])
            group_memBDownSeq.append([])
            for branch_idx in range(effective_ffn_branches):
                branch_suffix = (
                    f"{suffix}B{branch_idx}" if group_count > 1 else f"B{branch_idx}"
                )
                inBDownSeq = ObjectFifo(
                    ffn_b_down_ty, name=f"inBDownSeq{branch_suffix}", depth=1
                )
                memBDownSeq = inBDownSeq.cons().forward(
                    obj_type=ffn_b_down_ty,
                    name=f"memBDownSeq{branch_suffix}",
                    dims_to_stream=b_down_dims,
                    depth=weight_forward_depth,
                    placement=Tile(col=weight_b_down_cols[branch_idx], row=1),
                )
                group_inBDownSeq[group_idx].append(inBDownSeq)
                group_memBDownSeq[group_idx].append(memBDownSeq)

            if unified_qr_split is None:
                inRSeq = ObjectFifo(joined_o_ty, name=f"inRSeq{suffix}", depth=of_depth)
                group_inRSeq.append(inRSeq)
                lane_in_r_group = inRSeq.cons().split(
                    offsets=joined_offsets,
                    obj_types=[o_ty] * group_size,
                    names=[f"inRSeqL{lane_idx}" for lane_idx in lanes],
                    dims_to_stream=[r_dims] * group_size,
                    depths=[of_depth] * group_size,
                    placement=Tile(
                        col=transport_group["joined_or_mem_cols"]["residual"], row=1
                    ),
                )
                for local_idx, lane_idx in enumerate(lanes):
                    lane_in_r[lane_idx] = lane_in_r_group[local_idx]
            else:
                group_inRSeq.append(None)

            ln1StageJoined = ObjectFifo(
                joined_o_ty,
                name=f"outLNBroadcastSeq{suffix}",
                depth=1,
                dims_to_stream=o_dims,
            )
            group_ln1StageJoined.append(ln1StageJoined)
            lane_ln1_stage_out_group = ln1StageJoined.prod().join(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"outLNBroadcastSeqL{lane_idx}" for lane_idx in lanes],
                depths=[1] * group_size,
                placement=Tile(
                    col=transport_group["joined_or_mem_cols"]["ln1_stage"], row=1
                ),
            )
            for local_idx, lane_idx in enumerate(lanes):
                lane_ln1_stage_out[lane_idx] = lane_ln1_stage_out_group[local_idx]

            inLNSeq = ObjectFifo(joined_o_ty, name=f"inLNFromDDRSeq{suffix}", depth=1)
            group_inLNSeq.append(inLNSeq)
            lane_memOutLN_group = inLNSeq.cons().split(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"memLNStageSeqL{lane_idx}" for lane_idx in lanes],
                dims_to_stream=[r_dims] * group_size,
                depths=[2] * group_size,
                placement=Tile(
                    col=transport_group["joined_or_mem_cols"]["ln1_refill"], row=1
                ),
            )
            for local_idx, lane_idx in enumerate(lanes):
                lane_memOutLN[lane_idx] = lane_memOutLN_group[local_idx]

            ffnRFromDDRSeq = ObjectFifo(
                joined_o_ty, name=f"ffnRFromDDRSeq{suffix}", depth=1
            )
            group_ffnRFromDDRSeq.append(ffnRFromDDRSeq)
            lane_ffn_r_in_group = ffnRFromDDRSeq.cons().split(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"ffnRInSeqL{lane_idx}" for lane_idx in lanes],
                dims_to_stream=[r_dims] * group_size,
                depths=[2] * group_size,
                placement=Tile(
                    col=transport_group["joined_or_mem_cols"]["ffn_residual_refill"],
                    row=1,
                ),
            )
            for local_idx, lane_idx in enumerate(lanes):
                lane_ffn_r_in[lane_idx] = lane_ffn_r_in_group[local_idx]

            memLN2Joined = ObjectFifo(
                joined_o_ty,
                name=f"memLN2Seq{suffix}",
                depth=2,
                dims_to_stream=o_dims,
            )
            group_memLN2Joined.append(memLN2Joined)
            lane_outLN2_group = memLN2Joined.prod().join(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"outLN2SeqL{lane_idx}" for lane_idx in lanes],
                depths=[2] * group_size,
                placement=Tile(
                    col=transport_group["joined_or_mem_cols"]["output"], row=1
                ),
            )
            for local_idx, lane_idx in enumerate(lanes):
                lane_outLN2[lane_idx] = lane_outLN2_group[local_idx]

        for lane_idx in range(parallel_seq):
            lane_group_idx = lane_to_group_idx[lane_idx]
            lane_ln1_input.append(
                ObjectFifo(
                    o_ty,
                    name=f"outOProjInputSeqL{lane_idx}",
                    depth=2,
                )
            )
            lane_ffn_down_out[lane_idx] = ObjectFifo(
                o_ty, name=f"ffnDownOutSeqL{lane_idx}", depth=2
            )
            if parallel_heads > 1:
                lane_o_proj_stage_part[lane_idx] = ObjectFifo(
                    o_ty,
                    depth=o_proj_stage_chain_depth,
                    name=f"oProjStagePartSeqL{lane_idx}",
                )
                lane_o_proj_stage[lane_idx] = (
                    lane_o_proj_stage_part[lane_idx]
                    .cons(depth=ln_tiles_per_q_block)
                    .forward(
                        obj_type=o_ty,
                        name=f"oProjStageSeqL{lane_idx}",
                        depth=ln_tiles_per_q_block,
                        placement=Tile(
                            col=lane_o_proj_stage_mem_cols[lane_idx],
                            row=1,
                        ),
                    )
                )

            for head_idx in range(parallel_heads):
                lane_memA[lane_idx][head_idx] = ObjectFifo(
                    qk_ty,
                    depth=of_depth,
                    name=f"memASeqL{lane_idx}H{head_idx}",
                    dims_to_stream=a_dims_out,
                    dims_from_stream_per_cons=a_dims_in,
                )
                lane_memP[lane_idx][head_idx] = ObjectFifo(
                    qk_ty,
                    depth=of_depth,
                    name=f"memPSeqL{lane_idx}H{head_idx}",
                    dims_to_stream=p_dims_out,
                    dims_from_stream_per_cons=p_dims_in,
                )
                lane_scaleOF[lane_idx][head_idx] = ObjectFifo(
                    s_ty, depth=of_depth, name=f"scaleOFSeqL{lane_idx}H{head_idx}"
                )
                lane_outOProj[lane_idx][head_idx] = ObjectFifo(
                    q_ty, depth=of_depth, name=f"outOProjSeqL{lane_idx}H{head_idx}"
                )
                lane_acc_fifo = ObjectFifo(
                    o_ty,
                    depth=1,
                    name=f"outOProjAccumOutSeqL{lane_idx}H{head_idx}",
                )
                lane_o_proj_acc_out[lane_idx][head_idx] = lane_acc_fifo
                lane_o_proj_acc_in[lane_idx][head_idx] = lane_acc_fifo.cons(
                    depth=proj_acc_depth
                ).forward(
                    obj_type=o_ty,
                    name=f"outOProjAccumInSeqL{lane_idx}H{head_idx}",
                    depth=proj_acc_depth,
                    placement=Tile(
                        col=lane_o_proj_acc_mem_cols[lane_idx][head_idx],
                        row=1,
                    ),
                )
                if head_idx < (parallel_heads - 1):
                    lane_o_proj_partial[lane_idx][head_idx] = ObjectFifo(
                        o_ty,
                        depth=o_proj_stage_chain_depth,
                        name=f"outOPartSeqL{lane_idx}H{head_idx}",
                    )

                idx_buffer_qk = Buffer(
                    initial_value=np.zeros(shape=(2,), dtype=np.int32),
                    name=f"idx_buffer_qk_seq_l{lane_idx}_h{head_idx}",
                )
                idx_buffer_softmax = Buffer(
                    initial_value=np.zeros(shape=(2,), dtype=np.int32),
                    name=f"idx_buffer_softmax_seq_l{lane_idx}_h{head_idx}",
                )
                idx_buffer_pv = Buffer(
                    initial_value=np.zeros(shape=(2,), dtype=np.int32),
                    name=f"idx_buffer_pv_seq_l{lane_idx}_h{head_idx}",
                )
                scale_buffer_softmax = Buffer(
                    initial_value=np.zeros(shape=(4 * seq_tile,), dtype=dtype),
                    name=f"scale_buffer_softmax_seq_l{lane_idx}_h{head_idx}",
                )
                o_proj_stats_sum_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"o_proj_stats_sum_seq_l{lane_idx}_h{head_idx}",
                )
                o_proj_stats_sumsq_buffer = Buffer(
                    type=sum_l1_ty,
                    name=f"o_proj_stats_sumsq_seq_l{lane_idx}_h{head_idx}",
                )
                reduce_in = (
                    lane_o_proj_partial[lane_idx][head_idx - 1].cons()
                    if head_idx > 0
                    else None
                )

                lane_qk_workers.append(
                    Worker(
                        batched_matmul_qk,
                        fn_args=[
                            memQ_by_lane[lane_idx][head_idx].cons(),
                            group_memKSeq[lane_group_idx][head_idx].cons(),
                            lane_memA[lane_idx][head_idx].prod(),
                            zero_kernel,
                            matmul_qk_kernel,
                            idx_buffer_qk,
                        ],
                        placement=Tile(*lane_front_tiles[lane_idx]["qk"][head_idx]),
                        stack_size=0xD00,
                        while_true=False,
                    )
                )
                lane_softmax_workers.append(
                    Worker(
                        softmax,
                        fn_args=[
                            lane_memA[lane_idx][head_idx].cons(),
                            lane_memP[lane_idx][head_idx].prod(),
                            lane_scaleOF[lane_idx][head_idx].prod(),
                            partial_softmax_kernel,
                            scale_buffer_init_kernel,
                            memcopy_kernel_scale,
                            idx_buffer_softmax,
                            scale_buffer_softmax,
                        ],
                        placement=Tile(
                            *lane_front_tiles[lane_idx]["softmax"][head_idx]
                        ),
                        stack_size=0xD00,
                        while_true=False,
                    )
                )
                lane_pv_workers.append(
                    Worker(
                        batched_matmul_pv,
                        fn_args=[
                            lane_memP[lane_idx][head_idx].cons(),
                            group_memVSeq[lane_group_idx][head_idx].cons(),
                            lane_scaleOF[lane_idx][head_idx].cons(),
                            lane_outOProj[lane_idx][head_idx].prod(),
                            zero_kernel,
                            matmul_pv_kernel,
                            rescale_o_kernel,
                            idx_buffer_pv,
                        ],
                        placement=Tile(*lane_front_tiles[lane_idx]["pv"][head_idx]),
                        stack_size=0xD00,
                        while_true=False,
                    )
                )
                if parallel_heads > 1 and head_idx == (parallel_heads - 1):
                    lane_o_proj_workers.append(
                        Worker(
                            matmul_o_proj_stage_then_emit_stats,
                            fn_args=[
                                lane_outOProj[lane_idx][head_idx].cons(),
                                group_memOWSeq[lane_group_idx][head_idx].cons(),
                                lane_o_proj_acc_in[lane_idx][head_idx].cons(depth=1),
                                lane_o_proj_acc_out[lane_idx][head_idx].prod(),
                                reduce_in,
                                lane_o_proj_stage_part[lane_idx].prod(),
                                lane_ln1_input[lane_idx].prod(),
                                o_proj_stats_sum_buffer,
                                o_proj_stats_sumsq_buffer,
                                ln_zero_f32_kernel,
                                ln_calc_sum_sumsq_kernel,
                                pack_stats_kernel,
                                eltwise_add_vector_kernel,
                                zero_kernel_o_proj,
                                matmul_kernel_o_proj,
                                mem_copy_o_proj,
                            ],
                            placement=Tile(
                                *lane_front_tiles[lane_idx]["o_proj"][head_idx]
                            ),
                            stack_size=0xD00,
                            while_true=False,
                        )
                    )
                elif head_idx == (parallel_heads - 1):
                    lane_o_proj_workers.append(
                        Worker(
                            matmul_o_proj_emit_stats_first,
                            fn_args=[
                                lane_outOProj[lane_idx][head_idx].cons(),
                                group_memOWSeq[lane_group_idx][head_idx].cons(),
                                lane_o_proj_acc_in[lane_idx][head_idx].cons(depth=1),
                                lane_o_proj_acc_out[lane_idx][head_idx].prod(),
                                reduce_in,
                                lane_ln1_input[lane_idx].prod(),
                                o_proj_stats_sum_buffer,
                                o_proj_stats_sumsq_buffer,
                                ln_zero_f32_kernel,
                                ln_calc_sum_sumsq_kernel,
                                pack_stats_kernel,
                                eltwise_add_vector_kernel,
                                zero_kernel_o_proj,
                                matmul_kernel_o_proj,
                                mem_copy_o_proj,
                            ],
                            placement=Tile(
                                *lane_front_tiles[lane_idx]["o_proj"][head_idx]
                            ),
                            stack_size=0xD00,
                            while_true=False,
                        )
                    )
                else:
                    lane_o_proj_workers.append(
                        Worker(
                            matmul_o_proj,
                            fn_args=[
                                lane_outOProj[lane_idx][head_idx].cons(),
                                group_memOWSeq[lane_group_idx][head_idx].cons(),
                                lane_o_proj_acc_in[lane_idx][head_idx].cons(depth=1),
                                lane_o_proj_acc_out[lane_idx][head_idx].prod(),
                                reduce_in,
                                lane_o_proj_partial[lane_idx][head_idx].prod(),
                                o_proj_stats_sum_buffer,
                                o_proj_stats_sumsq_buffer,
                                ln_zero_f32_kernel,
                                ln_calc_sum_sumsq_kernel,
                                pack_stats_kernel,
                                eltwise_add_vector_kernel,
                                zero_kernel_o_proj,
                                matmul_kernel_o_proj,
                                mem_copy_o_proj,
                                False,
                            ],
                            placement=Tile(
                                *lane_front_tiles[lane_idx]["o_proj"][head_idx]
                            ),
                            stack_size=0xD00,
                            while_true=False,
                        )
                    )

            lane_ln1_weight_buffer = Buffer(
                type=ln_weights_ty,
                initial_value=static_ln1_weights,
                name=f"static_ln1_weights_seq_l{lane_idx}",
            )
            lane_ln2_weight_buffer = Buffer(
                type=ln_weights_ty,
                initial_value=static_ln2_weights,
                name=f"static_ln2_weights_seq_l{lane_idx}",
            )
            lane_ln1_norm_sum_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ln1_norm_sum_buffer_seq_l{lane_idx}",
            )
            lane_ln1_norm_sumsq_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ln1_norm_sumsq_buffer_seq_l{lane_idx}",
            )
            lane_ffn_down_sum_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ffn_down_sum_buffer_seq_l{lane_idx}",
            )
            lane_ffn_down_sumsq_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ffn_down_sumsq_buffer_seq_l{lane_idx}",
            )
            lane_ln2_sum_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ln2_sum_buffer_seq_l{lane_idx}",
            )
            lane_ln2_sumsq_buffer = Buffer(
                type=sum_l1_ty,
                name=f"ln2_sumsq_buffer_seq_l{lane_idx}",
            )

            lane_ln1_workers.append(
                Worker(
                    (
                        core_fn_ln1_stage_from_split_inputs
                        if parallel_heads > 1
                        else core_fn_ln1_stage_from_stats
                    ),
                    fn_args=(
                        [
                            lane_ln1_input[lane_idx].cons(),
                            lane_o_proj_stage[lane_idx].cons(depth=1),
                            lane_in_r[lane_idx].cons(),
                            lane_ln1_norm_sum_buffer,
                            lane_ln1_norm_sumsq_buffer,
                            lane_ln1_weight_buffer,
                            lane_ln1_stage_out[lane_idx].prod(),
                            ln_fused_add_layer_norm_kernel,
                            unpack_stats_kernel,
                        ]
                        if parallel_heads > 1
                        else [
                            lane_ln1_input[lane_idx].cons(),
                            lane_in_r[lane_idx].cons(),
                            lane_ln1_norm_sum_buffer,
                            lane_ln1_norm_sumsq_buffer,
                            lane_ln1_weight_buffer,
                            lane_ln1_stage_out[lane_idx].prod(),
                            ln_fused_add_layer_norm_kernel,
                            unpack_stats_kernel,
                        ]
                    ),
                    placement=Tile(*lane_tail_tiles[lane_idx]["ln1"]),
                    while_true=False,
                )
            )

            for branch_idx in range(effective_ffn_branches):
                lane_ffn_up_out[lane_idx][branch_idx] = ObjectFifo(
                    ffn_up_ty,
                    name=f"ffnUpOutSeqL{lane_idx}B{branch_idx}",
                    depth=2,
                )
                lane_ffn_down_part[lane_idx][branch_idx] = ObjectFifo(
                    o_ty,
                    name=f"ffnDownPartSeqL{lane_idx}B{branch_idx}",
                    depth=1,
                )
                lane_ffn_down_acc[lane_idx][branch_idx] = (
                    lane_ffn_down_part[lane_idx][branch_idx]
                    .cons(depth=proj_acc_depth)
                    .forward(
                        obj_type=o_ty,
                        name=f"ffnDownAccumSeqL{lane_idx}B{branch_idx}",
                        depth=proj_acc_depth,
                        placement=Tile(
                            col=lane_ffn_down_acc_mem_cols[lane_idx][branch_idx],
                            row=1,
                        ),
                    )
                )
                if branch_idx < (effective_ffn_branches - 1):
                    lane_ffn_down_reduce[lane_idx][branch_idx] = ObjectFifo(
                        o_ty,
                        name=f"ffnDownReduceSeqL{lane_idx}B{branch_idx}",
                        depth=ffn_down_reduce_depth,
                    )

                lane_ffn_up_workers.append(
                    Worker(
                        core_fn_ffn_up_proj,
                        fn_args=[
                            lane_memOutLN[lane_idx].cons(depth=2),
                            group_memBUpSeq[lane_group_idx][branch_idx].cons(),
                            lane_ffn_up_out[lane_idx][branch_idx].prod(),
                            ffn_zero_kernel_up_proj,
                            ffn_matmul_init_kernel_up_proj,
                            ffn_matmul_kernel_up_proj,
                            ffn_gelu_kernel,
                            ffn_col_group_count,
                        ],
                        placement=Tile(
                            *lane_tail_tiles[lane_idx]["ffn_up"][branch_idx]
                        ),
                        stack_size=0x700,
                        while_true=False,
                    )
                )

                reduce_in = (
                    lane_ffn_down_reduce[lane_idx][branch_idx - 1].cons()
                    if branch_idx > 0
                    else None
                )
                reduce_out = (
                    lane_ffn_down_reduce[lane_idx][branch_idx].prod()
                    if branch_idx < (effective_ffn_branches - 1)
                    else lane_ffn_down_out[lane_idx].prod(2)
                )
                lane_ffn_down_workers.append(
                    Worker(
                        (
                            core_fn_ffn_down_proj_emit_stats_first
                            if branch_idx == (effective_ffn_branches - 1)
                            else core_fn_ffn_down_proj
                        ),
                        fn_args=(
                            [
                                lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                group_memBDownSeq[lane_group_idx][branch_idx].cons(),
                                lane_ffn_down_acc[lane_idx][branch_idx].cons(depth=1),
                                lane_ffn_down_part[lane_idx][branch_idx].prod(),
                                reduce_in,
                                lane_ffn_down_out[lane_idx].prod(2),
                                lane_ffn_down_sum_buffer,
                                lane_ffn_down_sumsq_buffer,
                                ffn_matmul_init_kernel_down_proj,
                                ffn_matmul_kernel_down_proj,
                                eltwise_add_vector_kernel,
                                ln_calc_sum_sumsq_kernel,
                                pack_stats_kernel,
                                ln_zero_f32_kernel,
                                mem_copy_o_proj,
                                ffn_col_group_count,
                            ]
                            if branch_idx == (effective_ffn_branches - 1)
                            else [
                                lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                group_memBDownSeq[lane_group_idx][branch_idx].cons(),
                                lane_ffn_down_acc[lane_idx][branch_idx].cons(depth=1),
                                lane_ffn_down_part[lane_idx][branch_idx].prod(),
                                reduce_in,
                                reduce_out,
                                ffn_matmul_init_kernel_down_proj,
                                ffn_matmul_kernel_down_proj,
                                eltwise_add_vector_kernel,
                                mem_copy_o_proj,
                                ffn_col_group_count,
                            ]
                        ),
                        placement=Tile(
                            *lane_tail_tiles[lane_idx]["ffn_down"][branch_idx]
                        ),
                        stack_size=0xF00,
                        while_true=False,
                    )
                )

            lane_ln2_workers.append(
                Worker(
                    core_fn_add_norm2_from_stats,
                    fn_args=[
                        lane_ffn_down_out[lane_idx].cons(),
                        lane_ffn_r_in[lane_idx].cons(),
                        lane_ln2_sum_buffer,
                        lane_ln2_sumsq_buffer,
                        lane_ln2_weight_buffer,
                        lane_outLN2[lane_idx].prod(),
                        ln_fused_add_layer_norm_kernel,
                        unpack_stats_kernel,
                    ],
                    placement=Tile(*lane_tail_tiles[lane_idx]["ln2"]),
                    stack_size=0xF00,
                    while_true=False,
                )
            )

        qkv_tensor_shape = (3 * seq_len, embed_sz)
        q_batch_tiles_base = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (group_size * seq_tile, d),
            (1, parallel_heads),
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
        q_batch_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    qkv_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in q_batch_tiles_base
            ]
        )
        if unified_qr_split is not None:
            unified_q_batch_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (parallel_seq * seq_tile, d),
                (1, parallel_heads),
            )
            unified_q_batch_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        qkv_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in unified_q_batch_tiles_base
                ]
            )
        else:
            unified_q_batch_tiles = None
        k_tiles = TensorAccessSequence.from_taps(
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
        v_tiles = TensorAccessSequence.from_taps(
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
        wo_tiles = TensorTiler2D.group_tiler(
            (embed_sz, embed_sz),
            (d, emb_tile),
            (parallel_heads, embed_sz // emb_tile),
        )
        for tile in wo_tiles:
            tile._sizes = [
                tile._sizes[1],
                tile._sizes[0],
                tile._sizes[2],
                tile._sizes[3],
            ]
            tile._strides = [
                tile._strides[1],
                tile._strides[0],
                tile._strides[2],
                tile._strides[3],
            ]
        joined_o_tiles_base = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (group_size * seq_tile, emb_tile),
            (1, embed_sz // emb_tile // num_o_col_groups),
        )
        joined_r_tiles_base = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (group_size * seq_tile, emb_tile),
            (1, embed_sz // emb_tile // num_o_col_groups),
        )
        joined_o_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    or_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in joined_o_tiles_base
            ]
        )
        joined_i_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    or_tensor_shape,
                    offset=tap.offset + 2 * seq_len * embed_sz,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in joined_o_tiles_base
            ]
        )
        joined_r_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    or_tensor_shape,
                    offset=tap.offset + seq_len * embed_sz,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in joined_r_tiles_base
            ]
        )
        if unified_qr_split is not None:
            unified_joined_r_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (parallel_seq * seq_tile, emb_tile),
                (1, embed_sz // emb_tile // num_o_col_groups),
            )
            unified_joined_r_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + seq_len * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in unified_joined_r_tiles_base
                ]
            )
        else:
            unified_joined_r_tiles = None

        b_up_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    (embed_sz, ffn_intermediate_size),
                    offset=branch_idx * ffn_col_group_count * ffn_tile,
                    sizes=[ffn_col_group_count, proj_acc_depth, emb_tile, ffn_tile],
                    strides=[
                        ffn_tile,
                        emb_tile * ffn_intermediate_size,
                        ffn_intermediate_size,
                        1,
                    ],
                )
                for branch_idx in range(effective_ffn_branches)
            ]
        )
        b_down_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    (ffn_intermediate_size, embed_sz),
                    offset=branch_idx * ffn_col_group_count * ffn_tile * embed_sz,
                    sizes=[ffn_col_group_count, proj_acc_depth, ffn_tile, emb_tile],
                    strides=[ffn_tile * embed_sz, emb_tile, embed_sz, 1],
                )
                for branch_idx in range(effective_ffn_branches)
            ]
        )

        joined_refill_taps = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    or_tensor_shape,
                    offset=joined_i_tiles[
                        lane_batch_idx * group_count + group_idx
                    ].offset,
                    sizes=[
                        ffn_col_group_count,
                        proj_acc_depth,
                        group_size * seq_tile,
                        emb_tile,
                    ],
                    strides=[0, emb_tile, embed_sz, 1],
                )
                for lane_batch_idx in range(q_blocks_per_lane)
                for group_idx in range(group_count)
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

        def split_fill_tap_on_outer_dim(
            tap: TensorAccessPattern,
            tensor_shape: tuple[int, ...],
            max_outer_dim: int,
        ) -> list[TensorAccessPattern]:
            outer_size = int(tap.sizes[0])
            if outer_size <= max_outer_dim:
                return [tap]
            chunk_taps = []
            chunk_start = 0
            while chunk_start < outer_size:
                chunk_len = min(max_outer_dim, outer_size - chunk_start)
                chunk_taps.append(
                    TensorAccessPattern(
                        tensor_shape,
                        offset=int(tap.offset) + chunk_start * int(tap.strides[0]),
                        sizes=[chunk_len, *[int(s) for s in tap.sizes[1:]]],
                        strides=[int(s) for s in tap.strides],
                    )
                )
                chunk_start += chunk_len
            return chunk_taps

        for tas in (
            q_batch_tiles,
            k_tiles,
            v_tiles,
            wo_tiles,
            joined_o_tiles,
            joined_i_tiles,
            joined_r_tiles,
            b_up_tiles,
            b_down_tiles,
        ):
            legalize_tas(tas)
        if unified_q_batch_tiles is not None:
            legalize_tas(unified_q_batch_tiles)
        if unified_joined_r_tiles is not None:
            legalize_tas(unified_joined_r_tiles)
        for tas in (joined_refill_taps,):
            legalize_tas(tas)

        max_host_fill_outer_dim = 64
        K_fill_taps = [
            split_fill_tap_on_outer_dim(tap, qkv_tensor_shape, max_host_fill_outer_dim)
            for tap in k_tiles
        ]
        V_fill_taps = [
            split_fill_tap_on_outer_dim(tap, qkv_tensor_shape, max_host_fill_outer_dim)
            for tap in v_tiles
        ]

        if (
            math.prod(int(s) for s in q_batch_tiles[0].sizes)
            // math.prod((group_size * seq_tile, d * parallel_heads))
            != 1
        ):
            raise ValueError("Sequence-parallel Q batch tap count mismatch")
        if (
            unified_q_batch_tiles is not None
            and math.prod(int(s) for s in unified_q_batch_tiles[0].sizes)
            // math.prod((parallel_seq * seq_tile, d * parallel_heads))
            != 1
        ):
            raise ValueError("Sequence-parallel unified Q batch tap count mismatch")
        if (
            math.prod(int(s) for s in k_tiles[0].sizes)
            // math.prod((kv_seq_tile, d * parallel_heads))
            != num_kv_seq_blocks
        ):
            raise ValueError("Sequence-parallel K tap count mismatch")
        if (
            math.prod(int(s) for s in v_tiles[0].sizes)
            // math.prod((kv_seq_tile, d * parallel_heads))
            != num_kv_seq_blocks
        ):
            raise ValueError("Sequence-parallel V tap count mismatch")
        if (
            math.prod(int(s) for s in wo_tiles[0].sizes)
            // math.prod((d * parallel_heads, emb_tile))
            != proj_acc_depth
        ):
            raise ValueError("Sequence-parallel W_O tap count mismatch")
        if (
            math.prod(int(s) for s in joined_r_tiles[0].sizes)
            // math.prod((group_size * seq_tile, emb_tile))
            != proj_acc_depth
        ):
            raise ValueError("Sequence-parallel residual tap count mismatch")
        if (
            unified_joined_r_tiles is not None
            and math.prod(int(s) for s in unified_joined_r_tiles[0].sizes)
            // math.prod((parallel_seq * seq_tile, emb_tile))
            != proj_acc_depth
        ):
            raise ValueError("Sequence-parallel unified residual tap count mismatch")
        if (
            math.prod(int(s) for s in joined_o_tiles[0].sizes)
            // math.prod((group_size * seq_tile, emb_tile))
            != proj_acc_depth
        ):
            raise ValueError("Sequence-parallel output tap count mismatch")
        if (
            math.prod(int(s) for s in joined_i_tiles[0].sizes)
            // math.prod((group_size * seq_tile, emb_tile))
            != proj_acc_depth
        ):
            raise ValueError("Sequence-parallel intermediate tap count mismatch")
        if (
            math.prod(int(s) for s in joined_refill_taps[0].sizes)
            // math.prod((group_size * seq_tile, emb_tile))
            != ffn_col_group_count * proj_acc_depth
        ):
            raise ValueError("Sequence-parallel LN1 refill tap count mismatch")

        rt = Runtime()
        with rt.sequence(W_O_ty, QKV_ty, OR_ty, B_Up_ty, B_Down_ty) as (
            W_O,
            QKV,
            OR,
            B_Up,
            B_Down,
        ):
            for worker in lane_qk_workers:
                rt.start(worker)
            for worker in lane_softmax_workers:
                rt.start(worker)
            for worker in lane_pv_workers:
                rt.start(worker)
            for worker in lane_o_proj_workers:
                rt.start(worker)
            for worker in lane_ln1_workers:
                rt.start(worker)
            for worker in lane_ffn_up_workers:
                rt.start(worker)
            for worker in lane_ffn_down_workers:
                rt.start(worker)
            for worker in lane_ln2_workers:
                rt.start(worker)

            pending_tail_tg = None
            for lane_batch_idx in range(q_blocks_per_lane):
                for head_group_idx in range(num_qkv_head_block_per_parallel_head):
                    tg_head = rt.task_group()
                    if unified_qr_split is not None:
                        rt.fill(
                            unified_inQSeq.prod(),
                            QKV,
                            tap=unified_q_batch_tiles[
                                lane_batch_idx * num_qkv_head_block_per_parallel_head
                                + head_group_idx
                            ],
                            placement=Tile(
                                col=unified_qr_split["q_shim_col"],
                                row=0,
                            ),
                            task_group=tg_head,
                            wait=True,
                        )
                    if unified_inOWSeq is not None:
                        rt.fill(
                            unified_inOWSeq.prod(),
                            W_O,
                            tap=wo_tiles[head_group_idx],
                            placement=Tile(
                                col=unified_shared_streams["w_o"]["shim_col"],
                                row=0,
                            ),
                            task_group=tg_head,
                            wait=True,
                        )
                    for group_idx in range(group_count):
                        group_batch_idx = lane_batch_idx * group_count + group_idx
                        if unified_qr_split is None:
                            rt.fill(
                                group_inQSeq[group_idx].prod(),
                                QKV,
                                tap=q_batch_tiles[
                                    group_batch_idx
                                    * num_qkv_head_block_per_parallel_head
                                    + head_group_idx
                                ],
                                placement=Tile(
                                    col=transport_groups[group_idx].get(
                                        "shim_cols", default_group_shim_cols
                                    )["q"],
                                    row=0,
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                        for k_tap in K_fill_taps[head_group_idx]:
                            rt.fill(
                                group_inKSeq[group_idx].prod(),
                                QKV,
                                tap=k_tap,
                                placement=Tile(
                                    col=transport_groups[group_idx].get(
                                        "shim_cols", default_group_shim_cols
                                    )["k"],
                                    row=0,
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                        for v_tap in V_fill_taps[head_group_idx]:
                            rt.fill(
                                group_inVSeq[group_idx].prod(),
                                QKV,
                                tap=v_tap,
                                placement=Tile(
                                    col=transport_groups[group_idx].get(
                                        "shim_cols", default_group_shim_cols
                                    )["v"],
                                    row=0,
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                        if unified_inOWSeq is None:
                            rt.fill(
                                group_inOWSeq[group_idx].prod(),
                                W_O,
                                tap=wo_tiles[head_group_idx],
                                placement=Tile(
                                    col=transport_groups[group_idx].get(
                                        "shim_cols", default_group_shim_cols
                                    )["w_o"],
                                    row=0,
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                    rt.finish_task_group(tg_head)

                tg_residual = rt.task_group()
                if unified_qr_split is not None:
                    rt.fill(
                        unified_inRSeq.prod(),
                        OR,
                        tap=unified_joined_r_tiles[lane_batch_idx],
                        placement=Tile(
                            col=unified_qr_split["residual_shim_col"],
                            row=0,
                        ),
                        task_group=tg_residual,
                        wait=False,
                    )
                else:
                    for group_idx in range(group_count):
                        group_batch_idx = lane_batch_idx * group_count + group_idx
                        rt.fill(
                            group_inRSeq[group_idx].prod(),
                            OR,
                            tap=joined_r_tiles[group_batch_idx],
                            placement=Tile(
                                col=transport_groups[group_idx].get(
                                    "joined_or_shim_cols",
                                    default_group_joined_or_shim_cols,
                                )["residual"],
                                row=0,
                            ),
                            task_group=tg_residual,
                            wait=False,
                        )
                rt.finish_task_group(tg_residual)

                tg_ln1_drain = rt.task_group()
                for group_idx in range(group_count):
                    group_batch_idx = lane_batch_idx * group_count + group_idx
                    rt.drain(
                        group_ln1StageJoined[group_idx].cons(),
                        OR,
                        tap=joined_i_tiles[group_batch_idx],
                        placement=Tile(
                            col=transport_groups[group_idx].get(
                                "joined_or_shim_cols",
                                default_group_joined_or_shim_cols,
                            )["ln1_stage"],
                            row=0,
                        ),
                        task_group=tg_ln1_drain,
                        wait=True,
                    )
                rt.finish_task_group(tg_ln1_drain)

                if pending_tail_tg is not None:
                    rt.finish_task_group(pending_tail_tg)
                    pending_tail_tg = None

                tg_tail = rt.task_group()
                if unified_inBUpSeq is not None:
                    rt.fill(
                        unified_inBUpSeq.prod(),
                        B_Up,
                        tap=b_up_tiles[0],
                        placement=Tile(
                            col=unified_shared_streams["b_up"]["shim_col"],
                            row=0,
                        ),
                        task_group=tg_tail,
                        wait=True,
                    )
                for group_idx in range(group_count):
                    for branch_idx in range(effective_ffn_branches):
                        if unified_inBUpSeq is None:
                            rt.fill(
                                group_inBUpSeq[group_idx][branch_idx].prod(),
                                B_Up,
                                tap=b_up_tiles[branch_idx],
                                placement=Tile(
                                    col=group_weight_b_up_shim_cols[group_idx][
                                        branch_idx
                                    ],
                                    row=0,
                                ),
                                task_group=tg_tail,
                                wait=True,
                            )
                        rt.fill(
                            group_inBDownSeq[group_idx][branch_idx].prod(),
                            B_Down,
                            tap=b_down_tiles[branch_idx],
                            placement=Tile(
                                col=group_weight_b_down_shim_cols[group_idx][
                                    branch_idx
                                ],
                                row=0,
                            ),
                            task_group=tg_tail,
                            wait=True,
                        )
                    group_batch_idx = lane_batch_idx * group_count + group_idx
                    rt.fill(
                        group_inLNSeq[group_idx].prod(),
                        OR,
                        tap=joined_refill_taps[group_batch_idx],
                        placement=Tile(
                            col=transport_groups[group_idx].get(
                                "joined_or_shim_cols",
                                default_group_joined_or_shim_cols,
                            )["ln1_refill"],
                            row=0,
                        ),
                        task_group=tg_tail,
                        wait=True,
                    )
                    rt.fill(
                        group_ffnRFromDDRSeq[group_idx].prod(),
                        OR,
                        tap=joined_i_tiles[group_batch_idx],
                        placement=Tile(
                            col=transport_groups[group_idx].get(
                                "joined_or_shim_cols",
                                default_group_joined_or_shim_cols,
                            )["ffn_residual_refill"],
                            row=0,
                        ),
                        task_group=tg_tail,
                        wait=True,
                    )
                    rt.drain(
                        group_memLN2Joined[group_idx].cons(),
                        OR,
                        tap=joined_o_tiles[group_batch_idx],
                        placement=Tile(
                            col=transport_groups[group_idx].get(
                                "joined_or_shim_cols",
                                default_group_joined_or_shim_cols,
                            )["output"],
                            row=0,
                        ),
                        task_group=tg_tail,
                        wait=True,
                    )
                pending_tail_tg = tg_tail
            if pending_tail_tg is not None:
                rt.finish_task_group(pending_tail_tg)

        program = Program(NPU2(), rt)
        return program.resolve_program(SequentialPlacer())

    idx_buffer_qk = [
        Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_qk_{i}",
        )
        for i in range(parallel_heads)
    ]
    idx_buffer_softmax = [
        Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_softmax_{i}",
        )
        for i in range(parallel_heads)
    ]
    idx_buffer_pv = [
        Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_pv_{i}",
        )
        for i in range(parallel_heads)
    ]
    scale_buffer_softmax = [
        Buffer(
            initial_value=np.zeros(shape=(4 * seq_tile,), dtype=dtype),
            name=f"scale_buffer_softmax_{i}",
        )
        for i in range(parallel_heads)
    ]
    o_proj_stats_sum_buffers = [
        Buffer(type=sum_l1_ty, name=f"o_proj_stats_sum_{i}")
        for i in range(parallel_heads)
    ]
    o_proj_stats_sumsq_buffers = [
        Buffer(type=sum_l1_ty, name=f"o_proj_stats_sumsq_{i}")
        for i in range(parallel_heads)
    ]
    ln1_weight_buffer = Buffer(
        type=ln_weights_ty, initial_value=static_ln1_weights, name="static_ln1_weights"
    )
    ln2_weight_buffer = Buffer(
        type=ln_weights_ty, initial_value=static_ln2_weights, name="static_ln2_weights"
    )
    ln1_norm_sum_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sum_buffer")
    ln1_norm_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sumsq_buffer")
    ffn_down_sum_buffer = Buffer(type=sum_l1_ty, name="ffn_down_sum_buffer")
    ffn_down_sumsq_buffer = Buffer(type=sum_l1_ty, name="ffn_down_sumsq_buffer")
    ln2_sum_buffer = Buffer(type=sum_l1_ty, name="ln2_sum_buffer")
    ln2_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln2_sumsq_buffer")

    qk_workers = []
    softmax_workers = []
    pv_workers = []
    o_proj_workers = []
    for i in range(parallel_heads):
        qk_workers.append(
            Worker(
                batched_matmul_qk,
                fn_args=[
                    memQ[i].cons(),
                    memK[i].cons(),
                    memA[i].prod(),
                    zero_kernel,
                    matmul_qk_kernel,
                    idx_buffer_qk[i],
                ],
                placement=qk_tiles[i],
                stack_size=0xD00,
                while_true=False,
            )
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
                    idx_buffer_softmax[i],
                    scale_buffer_softmax[i],
                ],
                placement=softmax_tiles[i],
                stack_size=0xD00,
                while_true=False,
            )
        )
        pv_workers.append(
            Worker(
                batched_matmul_pv,
                fn_args=[
                    memP[i].cons(),
                    memV[i].cons(),
                    scaleOF[i].cons(),
                    outOProj[i].prod(),
                    zero_kernel,
                    matmul_pv_kernel,
                    rescale_o_kernel,
                    idx_buffer_pv[i],
                ],
                placement=pv_tiles[i],
                stack_size=0xD00,
                while_true=False,
            )
        )
        o_proj_workers.append(
            Worker(
                (
                    matmul_o_proj_stage_then_emit_stats
                    if parallel_heads > 1 and i == (parallel_heads - 1)
                    else (
                        matmul_o_proj_emit_stats_first
                        if i == (parallel_heads - 1)
                        else matmul_o_proj
                    )
                ),
                fn_args=(
                    [
                        outOProj[i].cons(),
                        memOW[i].cons(),
                        outOProjAccumIn[i].cons(depth=1),
                        outOProjAccumOut[i].prod(),
                        outOPart[i - 1].cons() if i > 0 else None,
                        oProjStagePart.prod(),
                        outOProjInput.prod(),
                        o_proj_stats_sum_buffers[i],
                        o_proj_stats_sumsq_buffers[i],
                        ln_zero_f32_kernel,
                        ln_calc_sum_sumsq_kernel,
                        pack_stats_kernel,
                        eltwise_add_vector_kernel,
                        zero_kernel_o_proj,
                        matmul_kernel_o_proj,
                        mem_copy_o_proj,
                    ]
                    if parallel_heads > 1 and i == (parallel_heads - 1)
                    else (
                        [
                            outOProj[i].cons(),
                            memOW[i].cons(),
                            outOProjAccumIn[i].cons(depth=1),
                            outOProjAccumOut[i].prod(),
                            outOPart[i - 1].cons() if i > 0 else None,
                            outOProjInput.prod(),
                            o_proj_stats_sum_buffers[i],
                            o_proj_stats_sumsq_buffers[i],
                            ln_zero_f32_kernel,
                            ln_calc_sum_sumsq_kernel,
                            pack_stats_kernel,
                            eltwise_add_vector_kernel,
                            zero_kernel_o_proj,
                            matmul_kernel_o_proj,
                            mem_copy_o_proj,
                        ]
                        if i == (parallel_heads - 1)
                        else [
                            outOProj[i].cons(),
                            memOW[i].cons(),
                            outOProjAccumIn[i].cons(depth=1),
                            outOProjAccumOut[i].prod(),
                            outOPart[i - 1].cons() if i > 0 else None,
                            (
                                outOPart[i].prod()
                                if i < (parallel_heads - 1)
                                else outOProjInput.prod()
                            ),
                            o_proj_stats_sum_buffers[i],
                            o_proj_stats_sumsq_buffers[i],
                            ln_zero_f32_kernel,
                            ln_calc_sum_sumsq_kernel,
                            pack_stats_kernel,
                            eltwise_add_vector_kernel,
                            zero_kernel_o_proj,
                            matmul_kernel_o_proj,
                            mem_copy_o_proj,
                            i == (parallel_heads - 1) and emb_tile >= 2 * seq_tile,
                        ]
                    )
                ),
                placement=o_proj_tiles[i],
                stack_size=0xD00,
                while_true=False,
            )
        )
    ln1_worker = Worker(
        (
            core_fn_ln1_stage_from_split_inputs
            if parallel_heads > 1
            else core_fn_ln1_stage_from_stats
        ),
        fn_args=(
            [
                outOProjInput.cons(),
                oProjStage.cons(depth=1),
                memR.cons(),
                ln1_norm_sum_buffer,
                ln1_norm_sumsq_buffer,
                ln1_weight_buffer,
                outLNBroadcast.prod(1),
                ln_fused_add_layer_norm_kernel,
                unpack_stats_kernel,
            ]
            if parallel_heads > 1
            else (
                [
                    outOProjInput.cons(),
                    memR.cons(),
                    ln1_norm_sum_buffer,
                    ln1_norm_sumsq_buffer,
                    ln1_weight_buffer,
                    outLNBroadcast.prod(1),
                    ln_fused_add_layer_norm_kernel,
                    unpack_stats_kernel,
                ]
            )
        ),
        placement=ln1_tile,
        while_true=False,
    )
    ffn_up_workers = []
    ffn_down_workers = []
    for branch_idx in range(effective_ffn_branches):
        ffn_up_workers.append(
            Worker(
                core_fn_ffn_up_proj,
                fn_args=[
                    memOutLN[branch_idx],
                    memBUp[branch_idx].cons(),
                    ffnUpOut[branch_idx].prod(),
                    ffn_zero_kernel_up_proj,
                    ffn_matmul_init_kernel_up_proj,
                    ffn_matmul_kernel_up_proj,
                    ffn_gelu_kernel,
                    ffn_col_group_count,
                ],
                placement=ffn_up_tiles[branch_idx],
                stack_size=0x700,
                while_true=False,
            )
        )
        reduce_in = ffnDownReduce[branch_idx - 1].cons() if branch_idx > 0 else None
        reduce_out = (
            ffnDownReduce[branch_idx].prod()
            if branch_idx < (effective_ffn_branches - 1)
            else ffnDownOut.prod(2)
        )
        ffn_down_workers.append(
            Worker(
                (
                    core_fn_ffn_down_proj_emit_stats_first
                    if branch_idx == (effective_ffn_branches - 1)
                    else core_fn_ffn_down_proj
                ),
                fn_args=(
                    [
                        ffnUpOut[branch_idx].cons(),
                        memBDown[branch_idx].cons(),
                        ffnDownAccum[branch_idx].cons(depth=1),
                        ffnDownPart[branch_idx].prod(),
                        reduce_in,
                        ffnDownOut.prod(2),
                        ffn_down_sum_buffer,
                        ffn_down_sumsq_buffer,
                        ffn_matmul_init_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        eltwise_add_vector_kernel,
                        ln_calc_sum_sumsq_kernel,
                        pack_stats_kernel,
                        ln_zero_f32_kernel,
                        mem_copy_o_proj,
                        ffn_col_group_count,
                    ]
                    if branch_idx == (effective_ffn_branches - 1)
                    else [
                        ffnUpOut[branch_idx].cons(),
                        memBDown[branch_idx].cons(),
                        ffnDownAccum[branch_idx].cons(depth=1),
                        ffnDownPart[branch_idx].prod(),
                        reduce_in,
                        reduce_out,
                        ffn_matmul_init_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        eltwise_add_vector_kernel,
                        mem_copy_o_proj,
                        ffn_col_group_count,
                    ]
                ),
                placement=ffn_down_tiles[branch_idx],
                stack_size=0xF00,
                while_true=False,
            )
        )
    ln2_worker = Worker(
        core_fn_add_norm2_from_stats,
        fn_args=[
            ffnDownOut.cons(),
            ffnRIn.cons(),
            ln2_sum_buffer,
            ln2_sumsq_buffer,
            ln2_weight_buffer,
            outLN2.prod(),
            ln_fused_add_layer_norm_kernel,
            unpack_stats_kernel,
        ],
        placement=ln2_tile,
        stack_size=0xF00,
        while_true=False,
    )

    qkv_tensor_shape = (3 * seq_len, embed_sz)
    q_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (seq_tile, d),
        (1, parallel_heads),
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
        (embed_sz, embed_sz),
        (d, emb_tile),
        (parallel_heads, embed_sz // emb_tile),
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
    B_Up_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                (embed_sz, ffn_intermediate_size),
                offset=branch_idx * ffn_col_group_count * ffn_tile,
                sizes=[ffn_col_group_count, proj_acc_depth, emb_tile, ffn_tile],
                strides=[
                    ffn_tile,
                    emb_tile * ffn_intermediate_size,
                    ffn_intermediate_size,
                    1,
                ],
            )
            for branch_idx in range(effective_ffn_branches)
        ]
    )
    B_Down_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                (ffn_intermediate_size, embed_sz),
                offset=branch_idx * ffn_col_group_count * ffn_tile * embed_sz,
                sizes=[ffn_col_group_count, proj_acc_depth, ffn_tile, emb_tile],
                strides=[ffn_tile * embed_sz, emb_tile, embed_sz, 1],
            )
            for branch_idx in range(effective_ffn_branches)
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

    def split_fill_tap_on_outer_dim(
        tap: TensorAccessPattern, tensor_shape: tuple[int, ...], max_outer_dim: int
    ) -> list[TensorAccessPattern]:
        outer_size = int(tap.sizes[0])
        if outer_size <= max_outer_dim:
            return [tap]
        chunk_taps = []
        chunk_start = 0
        while chunk_start < outer_size:
            chunk_len = min(max_outer_dim, outer_size - chunk_start)
            chunk_taps.append(
                TensorAccessPattern(
                    tensor_shape,
                    offset=int(tap.offset) + chunk_start * int(tap.strides[0]),
                    sizes=[chunk_len, *[int(s) for s in tap.sizes[1:]]],
                    strides=[int(s) for s in tap.strides],
                )
            )
            chunk_start += chunk_len
        return chunk_taps

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

    max_host_fill_outer_dim = 64
    K_fill_taps = [
        split_fill_tap_on_outer_dim(tap, qkv_tensor_shape, max_host_fill_outer_dim)
        for tap in K_tiles
    ]
    V_fill_taps = [
        split_fill_tap_on_outer_dim(tap, qkv_tensor_shape, max_host_fill_outer_dim)
        for tap in V_tiles
    ]

    for tap_seq, obj_shape, expected, message in (
        (
            Q_tiles,
            (seq_tile, d * parallel_heads),
            1,
            "Q tap count does not match grouped head transfer",
        ),
        (
            K_tiles,
            (kv_seq_tile, d * parallel_heads),
            num_kv_seq_blocks,
            "K tap count does not match KV block count",
        ),
        (
            V_tiles,
            (kv_seq_tile, d * parallel_heads),
            num_kv_seq_blocks,
            "V tap count does not match KV block count",
        ),
        (
            WO_tiles,
            (d * parallel_heads, emb_tile),
            proj_acc_depth,
            "W_O tap count does not match proj_acc_depth",
        ),
        (
            R_tiles,
            (seq_tile, emb_tile),
            proj_acc_depth,
            "Residual tap count does not match LN1 runtime contract",
        ),
        (
            O_tiles,
            (seq_tile, emb_tile),
            proj_acc_depth,
            "Output tap count does not match proj_acc_depth",
        ),
        (
            B_Up_tiles,
            (emb_tile, ffn_tile),
            ffn_col_group_count * proj_acc_depth,
            "B_Up tap count does not match FFN loop count",
        ),
        (
            B_Down_tiles,
            (ffn_tile, emb_tile),
            ffn_col_group_count * proj_acc_depth,
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
        for i in range(parallel_heads):
            rt.start(qk_workers[i])
            rt.start(softmax_workers[i])
            rt.start(pv_workers[i])
            rt.start(o_proj_workers[i])
        rt.start(ln1_worker)
        for branch_idx in range(effective_ffn_branches):
            rt.start(ffn_up_workers[branch_idx])
            rt.start(ffn_down_workers[branch_idx])
        rt.start(ln2_worker)

        pending_ln1_refill_tg = None
        pending_output_tap_idx = None
        q_block_schedule = [
            lane_idx * q_blocks_per_lane + lane_block_idx
            for lane_block_idx in range(q_blocks_per_lane)
            for lane_idx in range(parallel_seq)
        ]

        for tap_idx in q_block_schedule:
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

            for head_group_idx in range(num_qkv_head_block_per_parallel_head):
                tg_head = rt.task_group()
                rt.fill(
                    inQ.prod(),
                    QKV,
                    tap=Q_tiles[
                        tap_idx * num_qkv_head_block_per_parallel_head + head_group_idx
                    ],
                    placement=Tile(col=q_shim_col, row=0),
                    task_group=tg_head,
                    wait=True,
                )
                for k_tap in K_fill_taps[head_group_idx]:
                    rt.fill(
                        inK.prod(),
                        QKV,
                        tap=k_tap,
                        placement=Tile(col=k_shim_col, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                for v_tap in V_fill_taps[head_group_idx]:
                    rt.fill(
                        inV.prod(),
                        QKV,
                        tap=v_tap,
                        placement=Tile(col=v_shim_col, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                rt.fill(
                    inOW.prod(),
                    W_O,
                    tap=WO_tiles[head_group_idx],
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
                sizes=[1, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            branch_refill_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln1_stage_base_offset,
                sizes=[ffn_col_group_count, proj_acc_depth, seq_tile, emb_tile],
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
                != proj_acc_depth
            ):
                raise ValueError("LN1 stage tap count mismatch")
            if (
                math.prod(int(s) for s in branch_refill_tap.sizes)
                // math.prod((seq_tile, emb_tile))
                != ffn_col_group_count * proj_acc_depth
            ):
                raise ValueError("LN1 branch refill tap count mismatch")
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
                tap=branch_refill_tap,
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
            for branch_idx in range(effective_ffn_branches):
                rt.fill(
                    inBUp[branch_idx].prod(),
                    B_Up,
                    tap=B_Up_tiles[branch_idx],
                    placement=Tile(
                        col=weight_mem_tiles["b_up_by_branch"][branch_idx],
                        row=0,
                    ),
                    task_group=tg_ln1_refill_and_weights,
                    wait=effective_ffn_branches > 1,
                )
                rt.fill(
                    inBDown[branch_idx].prod(),
                    B_Down,
                    tap=B_Down_tiles[branch_idx],
                    placement=Tile(
                        col=weight_mem_tiles["b_down_by_branch"][branch_idx],
                        row=0,
                    ),
                    task_group=tg_ln1_refill_and_weights,
                    wait=effective_ffn_branches > 1,
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
