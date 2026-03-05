# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import math
import argparse
import os
from pathlib import Path
import logging

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence, TensorAccessPattern
import aie.dialects.index as index
from aie.dialects.aiex import *
from operators.encoder_pipeline.mapping_validation import (
    find_ffn_layout,
    manhattan_distance,
)

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

FFN_STAGE_ONLY_CHOICES = (-1, 0, 1, 2, 3, 4)
ADDNORM_DEBUG_CHOICES = (-1, 0, 1)


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Encoder Pipeline MLIR Design",
        description="Emits MLIR code for encoder_pipeline design of the given input size",
    )
    argparser.add_argument("--heads", type=int, default=12)
    argparser.add_argument("--seq-len", type=int, default=64)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--seq-tile", type=int, default=32)
    argparser.add_argument("--kv-seq-tile", type=int, default=64)
    argparser.add_argument("--emb-tile", type=int, default=96)
    argparser.add_argument("--proj-acc-depth", type=int, default=8)
    argparser.add_argument("--nB-tiles-distributed", type=int, default=1)
    argparser.add_argument("--ffn-intermediate-size", type=int, default=None)
    argparser.add_argument("--parallel-heads", type=int, default=1)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=True)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--kernel-archive", type=str, default="encoder_pipeline_kernels.a"
    )
    argparser.add_argument("--ln1-weight-file", type=str, default=None)
    argparser.add_argument("--ln2-weight-file", type=str, default=None)
    argparser.add_argument(
        "--ffn-stage-only",
        type=int,
        choices=list(FFN_STAGE_ONLY_CHOICES),
        default=-1,
        help=(
            "FFN stage isolation: 0=up-proj only, 1=down-proj only, "
            "2=AddNorm2 only, 3=MHA-focused (downstream skipped), "
            "4=AddNorm1-focused (downstream skipped), -1=all stages"
        ),
    )
    argparser.add_argument(
        "--addnorm1-debug-mode",
        type=int,
        choices=list(ADDNORM_DEBUG_CHOICES),
        default=-1,
        help="-1=normal, 0=pass AddNorm1 input, 1=pass AddNorm1 residual",
    )
    argparser.add_argument(
        "--addnorm2-debug-mode",
        type=int,
        choices=list(ADDNORM_DEBUG_CHOICES),
        default=-1,
        help="-1=normal, 0=pass AddNorm2 input, 1=pass AddNorm2 residual",
    )
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default=base_dir / "build" / "encoder_pipeline.mlir",
        help="Output file path for the generated MLIR module",
    )

    args = argparser.parse_args()

    module = fused_mha(
        heads=args.heads,
        seq_len=args.seq_len,
        d=args.d,
        seq_tile=args.seq_tile,
        kv_seq_tile=args.kv_seq_tile,
        emb_tile=args.emb_tile,
        proj_acc_depth=args.proj_acc_depth,
        parallel_heads=args.parallel_heads,
        emulate_bf16_mmul_with_bfp16=args.emulate_bf16_mmul_with_bfp16,
        kernel_archive=args.kernel_archive,
        trace_size=args.trace_size,
        ln1_weight_file=args.ln1_weight_file,
        ln2_weight_file=args.ln2_weight_file,
        nB_tiles_distributed=args.nB_tiles_distributed,
        ffn_intermediate_size=args.ffn_intermediate_size,
        ffn_stage_only=None if args.ffn_stage_only == -1 else args.ffn_stage_only,
        addnorm1_debug_mode=args.addnorm1_debug_mode,
        addnorm2_debug_mode=args.addnorm2_debug_mode,
    )

    output_file_path = Path(args.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

    logging.info(f"MLIR module written to {output_file_path}")


# TODO: Add back parallel sequence blocks?


def fused_mha(
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
    trace_size: int = 0,
    ln1_weight_file=None,
    ln2_weight_file=None,
    nB_tiles_distributed: int = 1,
    ffn_intermediate_size: int | None = None,
    ffn_stage_only: int | None = None,
    addnorm1_debug_mode: int = -1,
    addnorm2_debug_mode: int = -1,
):
    def _override_int_list_env(name: str, default: list[int]) -> list[int]:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        values = [int(v.strip()) for v in raw.split(",") if v.strip() != ""]
        if len(values) != len(default):
            raise ValueError(
                f"{name} must have {len(default)} comma-separated ints "
                f"(got {len(values)} from '{raw}')"
            )
        return values

    def _override_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        return int(raw.strip())

    embed_sz = heads * d
    if ffn_intermediate_size is None:
        ffn_intermediate_size = 4 * embed_sz

    # Load static layer norm weights for AddNorm-1 and AddNorm-2.
    if ln1_weight_file is None:
        static_ln1_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln1_weights = np.load(ln1_weight_file)
    if ln2_weight_file is None:
        static_ln2_weights = np.ones(embed_sz, dtype=bfloat16)
    else:
        static_ln2_weights = np.load(ln2_weight_file)

    of_depth = 2
    enable_tracing = True if trace_size > 0 else False
    dtype_str = "bf16"
    dev = "npu2"

    num_q_seq_blocks = seq_len // seq_tile
    num_kv_seq_blocks = seq_len // kv_seq_tile
    num_qkv_head_block_per_parallel_head = heads // parallel_heads
    if nB_tiles_distributed <= 0:
        raise ValueError(
            f"nB_tiles_distributed must be > 0 (got {nB_tiles_distributed})"
        )
    if embed_sz != emb_tile * proj_acc_depth:
        raise ValueError(
            "proj_acc_depth must satisfy emb_tile * proj_acc_depth == embed_sz "
            f"({emb_tile} * {proj_acc_depth} != {embed_sz})"
        )
    if ffn_intermediate_size % emb_tile != 0:
        raise ValueError(
            "ffn_intermediate_size must be divisible by emb_tile "
            f"({ffn_intermediate_size} % {emb_tile} != 0)"
        )
    valid_ffn_stage_only = (None, *FFN_STAGE_ONLY_CHOICES[1:])
    if ffn_stage_only not in valid_ffn_stage_only:
        raise ValueError(
            f"ffn_stage_only must be one of {{None, 0, 1, 2, 3, 4}} (got {ffn_stage_only})"
        )
    if addnorm1_debug_mode not in ADDNORM_DEBUG_CHOICES:
        raise ValueError(
            "addnorm1_debug_mode must be one of {-1, 0, 1} "
            f"(got {addnorm1_debug_mode})"
        )
    if addnorm2_debug_mode not in ADDNORM_DEBUG_CHOICES:
        raise ValueError(
            "addnorm2_debug_mode must be one of {-1, 0, 1} "
            f"(got {addnorm2_debug_mode})"
        )
    ffn_col_groups = ffn_intermediate_size // emb_tile
    # In MHA/AddNorm1 focused profiling modes, downstream FFN/AddNorm2 are
    # intentionally de-emphasized. Keep only one replay group to reduce
    # non-target traffic while preserving stage-to-stage liveness.
    profile_replay_groups = 1 if ffn_stage_only in (3, 4) else ffn_col_groups
    if nB_tiles_distributed > ffn_col_groups:
        raise ValueError(
            "nB_tiles_distributed must be <= ffn_col_groups "
            f"({nB_tiles_distributed} > {ffn_col_groups})"
        )

    num_o_col_groups = embed_sz // (emb_tile * proj_acc_depth)
    ln_tiles_per_q_block = num_o_col_groups * proj_acc_depth
    ffn_layout = find_ffn_layout(
        parallel_heads=parallel_heads,
        max_non_neighbor_down_to_ln2=2,
    )
    ln1_tile = tuple(ffn_layout["ln1_tile"])
    all_ffn_up_tiles = [tuple(t) for t in ffn_layout["up_tiles"]]
    all_ffn_down_tiles = [tuple(t) for t in ffn_layout["down_tiles"]]
    ln2_tile = tuple(ffn_layout["ln2_tile"])
    down_root_tile = tuple(ffn_layout["down_root_tile"])
    max_ffn_branches = len(all_ffn_up_tiles)
    if max_ffn_branches == 0:
        raise ValueError("FFN layout did not provide any branch tiles")

    requested_ffn_branches = min(nB_tiles_distributed, max_ffn_branches)
    down_root_idx = all_ffn_down_tiles.index(down_root_tile)
    if down_root_idx != max_ffn_branches - 1:
        raise ValueError(
            "FFN layout invariant violated: down_root must be final down tile "
            f"(root_idx={down_root_idx}, max={max_ffn_branches})"
        )
    # Keep the active branch subset as a suffix ending at root to preserve
    # neighbor-chain reduction order when pruning.
    selected_start_idx = max(0, max_ffn_branches - requested_ffn_branches)
    selected_branch_indices = list(range(selected_start_idx, max_ffn_branches))
    if max_ffn_branches > 3:
        raise ValueError(
            "encoder_pipeline supports at most 3 FFN branches "
            f"(layout returned {max_ffn_branches})"
        )

    def drop_non_root_branch(reason: str):
        non_root_selected = [i for i in selected_branch_indices if i != down_root_idx]
        if not non_root_selected:
            raise ValueError(
                f"Unable to prune FFN branches ({reason}): only root branch is selectable"
            )
        dropped_idx = max(
            non_root_selected,
            key=lambda i: (
                manhattan_distance(all_ffn_down_tiles[i], down_root_tile),
                manhattan_distance(all_ffn_up_tiles[i], ln1_tile),
                i,
            ),
        )
        selected_branch_indices.remove(dropped_idx)
        logging.warning(
            "Reduced effective FFN branch count from %d to %d: %s "
            "(dropped branch idx=%d, up=%s, down=%s)",
            len(selected_branch_indices) + 1,
            len(selected_branch_indices),
            reason,
            dropped_idx,
            all_ffn_up_tiles[dropped_idx],
            all_ffn_down_tiles[dropped_idx],
        )

    # Prefer placing LN1 post worker and optional LN1 route helper core on free
    # tiles outside active FFN branch tiles. If resources are insufficient, drop
    # one non-root FFN branch and retry.
    free_ffn_tiles = [tuple(t) for t in ffn_layout["free_tiles"]]
    layout_reduction_edges = {
        (tuple(src), tuple(dst))
        for (src, dst) in ffn_layout.get("down_reduction_edges", [])
    }
    ln1_route_tile = None
    while True:
        if len(selected_branch_indices) <= 0:
            raise ValueError("No FFN branches selected")
        selected_up_tiles = [all_ffn_up_tiles[i] for i in selected_branch_indices]
        selected_down_tiles = [all_ffn_down_tiles[i] for i in selected_branch_indices]
        used_tiles_base = {ln1_tile, ln2_tile, *selected_up_tiles, *selected_down_tiles}
        effective_candidate_branches = len(selected_branch_indices)
        if effective_candidate_branches > 1:
            expected_chain_edges = [
                (selected_down_tiles[i], selected_down_tiles[i + 1])
                for i in range(effective_candidate_branches - 1)
            ]
            missing_chain_edges = [
                edge
                for edge in expected_chain_edges
                if edge not in layout_reduction_edges
            ]
            if missing_chain_edges:
                drop_non_root_branch(
                    "selected FFN branches are not on a valid neighbor reduction chain "
                    f"(missing_edges={missing_chain_edges})"
                )
                continue
        # A single LN1 mul/add core can directly emit at most two FFN streams.
        ln1_router_tiles_needed = 1 if effective_candidate_branches > 2 else 0
        required_compute_tiles = (
            parallel_heads * 4
            + 3  # LN1 norm + LN1 mul/add + LN2
            + 2 * effective_candidate_branches  # FFN up/down compute branches
            + ln1_router_tiles_needed
        )
        if required_compute_tiles > 32:
            if len(selected_branch_indices) <= 1:
                raise ValueError(
                    "Configuration exceeds NPU2 compute tile capacity (32): "
                    f"needs {required_compute_tiles}"
                )
            drop_non_root_branch(
                f"compute tile capacity exceeded (needs {required_compute_tiles})"
            )
            continue
        # For wide-head/high-acc configurations, memtile BD-ID allocation for
        # FFN tail staging can fail even when nominal channel counts fit.
        # Keep pruning non-root branches until the generated topology is
        # within known feasible memtile BD/channel budgets.
        if (
            parallel_heads >= 6
            and proj_acc_depth >= 6
            and len(selected_branch_indices) > 1
        ):
            drop_non_root_branch(
                "memtile BD/channel budget exceeded for wide-head high-acc tail staging"
            )
            continue
        if (
            parallel_heads >= 4
            and proj_acc_depth >= 8
            and len(selected_branch_indices) > 1
        ):
            drop_non_root_branch(
                "memtile BD/channel budget exceeded for high-acc tail staging"
            )
            continue
        if ffn_stage_only in (3, 4) and len(selected_branch_indices) > 1:
            drop_non_root_branch("downstream-profile mode uses a single FFN branch")
            continue
        ln1_post_candidates = [
            tile for tile in free_ffn_tiles if tile not in used_tiles_base
        ]
        placement_found = False
        for candidate_ln1_post_tile in ln1_post_candidates:
            used_with_ln1_post = used_tiles_base | {candidate_ln1_post_tile}
            aux_candidates = [
                tile for tile in free_ffn_tiles if tile not in used_with_ln1_post
            ]
            if len(aux_candidates) < ln1_router_tiles_needed:
                continue
            candidate_ln1_route_tile = None
            if ln1_router_tiles_needed:
                candidate_ln1_route_tile = min(
                    aux_candidates,
                    key=lambda t: (
                        manhattan_distance(t, candidate_ln1_post_tile),
                        manhattan_distance(t, ln1_tile),
                        t[0],
                        t[1],
                    ),
                )
            ln1_post_tile = candidate_ln1_post_tile
            ln1_route_tile = candidate_ln1_route_tile
            placement_found = True
            break
        if placement_found:
            break
        if len(selected_branch_indices) <= 1:
            raise ValueError(
                "Insufficient free FFN tiles for LN1 post/route helper placement "
                f"(branches={len(selected_branch_indices)})"
            )
        drop_non_root_branch("insufficient free tile budget for LN1 post/route helpers")

    effective_ffn_branches = len(selected_branch_indices)
    if effective_ffn_branches <= 0:
        raise ValueError("No FFN branches selected")
    if selected_branch_indices[-1] != down_root_idx:
        raise ValueError(
            "FFN branch ordering invariant violated: root branch must be final "
            f"(selected={selected_branch_indices}, root_idx={down_root_idx})"
        )

    ffn_group_base = profile_replay_groups // effective_ffn_branches
    ffn_group_rem = profile_replay_groups % effective_ffn_branches
    ffn_col_group_counts = [
        ffn_group_base + (1 if i < ffn_group_rem else 0)
        for i in range(effective_ffn_branches)
    ]
    ffn_col_group_offsets = []
    running_group_offset = 0
    for count in ffn_col_group_counts:
        ffn_col_group_offsets.append(running_group_offset)
        running_group_offset += count
    final_ffn_branch_idx = effective_ffn_branches - 1
    # Reduction chain topology over active branches (one incoming reduction
    # stream per down core): b0 -> b1 -> ... -> bN(final/root).
    ffn_reduction_sources = list(range(final_ffn_branch_idx))
    ffn_requires_ln1_router = effective_ffn_branches > 2

    # Worker count:
    # - MHA path: 4 workers per parallel head
    # - Encoder tail:
    #   LN1(norm) + LN1(mul+resadd) + AddNorm2
    #   + (FFN up/down)*branches
    #   + optional LN1 route worker for branch fanout > 2
    required_compute_tiles = (
        parallel_heads * 4
        + 3
        + 2 * effective_ffn_branches
        + (1 if ffn_requires_ln1_router else 0)
    )
    if required_compute_tiles > 32:
        raise ValueError(
            "Configuration exceeds NPU2 compute tile capacity (32): "
            f"needs {required_compute_tiles}"
        )

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_str]
    r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]

    logging.info(f"Device: {dev}")
    logging.info(f"Number of heads: {heads}")
    logging.info(
        f"MHA Dimensions: seq_len={seq_len}, d={d}, seq_tile={seq_tile}, kv_seq_tile={kv_seq_tile}, emb_tile={emb_tile}, proj_acc_depth={proj_acc_depth}, parallel_heads={parallel_heads}"
    )
    logging.info(
        "Encoder pipeline parameters: nB_tiles_distributed(requested)=%d effective=%d",
        nB_tiles_distributed,
        effective_ffn_branches,
    )
    logging.info(
        "Debug controls: ffn_stage_only=%s, addnorm1_debug_mode=%d, addnorm2_debug_mode=%d",
        "all" if ffn_stage_only is None else ffn_stage_only,
        addnorm1_debug_mode,
        addnorm2_debug_mode,
    )
    logging.info(
        f"num_q_seq_blocks: {num_q_seq_blocks}, num_kv_seq_blocks: {num_kv_seq_blocks}, num_qkv_head_block_per_parallel_head: {num_qkv_head_block_per_parallel_head}, num_o_col_groups: {num_o_col_groups}"
    )
    logging.info(
        "FFN distribution: groups=%d counts=%s offsets=%s",
        profile_replay_groups,
        ffn_col_group_counts,
        ffn_col_group_offsets,
    )
    logging.info(
        "FFN mapped placement: ln1_norm=%s ln1_post=%s ln1_route=%s up(all)=%s down(all)=%s up(active)=%s down(active)=%s ln2=%s down_root=%s down_reduction_edges=%s down_to_ln2=%s non_neighbor_down_to_ln2=%s active_branch_indices=%s",
        ln1_tile,
        ln1_post_tile,
        ln1_route_tile,
        all_ffn_up_tiles,
        all_ffn_down_tiles,
        selected_up_tiles,
        selected_down_tiles,
        ln2_tile,
        down_root_tile,
        ffn_layout["down_reduction_edges"],
        ffn_layout["down_to_ln2_tiles"],
        ffn_layout["non_neighbor_down_to_ln2"],
        selected_branch_indices,
    )
    logging.info(
        "FFN reduction topology(active branch order): chain_edges=%s final_branch_idx=%d "
        "(down_tile=%s) ln1_router=%s",
        [(i, i + 1) for i in range(final_ffn_branch_idx)],
        final_ffn_branch_idx,
        selected_down_tiles[final_ffn_branch_idx],
        ffn_requires_ln1_router,
    )
    logging.info(f"Data type: {dtype_str}")
    logging.info(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
    logging.info(f"Enable tracing: {enable_tracing}")

    assert heads > 0, "Number of heads must be greater than 0"
    assert (
        heads % parallel_heads == 0
    ), "Number of heads must be divisible by parallel_heads"

    assert seq_tile % r == 0, f"seq_tile must be divisible by r ({seq_tile} % {r} != 0)"
    assert (
        kv_seq_tile % t == 0
    ), f"kv_seq_tile must be divisible by t ({kv_seq_tile} % {t} != 0)"
    assert d % s == 0, f"d must be divisible by s ({d} % {s} != 0)"

    assert seq_len % seq_tile == 0, "seq_len must be divisible by seq_tile"
    assert seq_len % kv_seq_tile == 0, "seq_len must be divisible by kv_seq_tile"

    dtype = dtype_map[dtype_str]

    inv_scale = (1 / np.sqrt(d)) * 1.4453125

    # Tensors living in DRAM
    W_O_ty = np.ndarray[
        (embed_sz, embed_sz),
        np.dtype[dtype],
    ]
    QKV_ty = np.ndarray[
        (3 * seq_len, embed_sz),
        np.dtype[dtype],
    ]
    OR_ty = np.ndarray[
        (2 * seq_len, embed_sz),
        np.dtype[dtype],
    ]
    # Keep FFN weight tensors as flat L3 buffers, matching ffn_addnorm runtime
    # transfer semantics. TensorAccessPattern provides the 2D logical views.
    B_Up_ty = np.ndarray[
        (embed_sz * ffn_intermediate_size,),
        np.dtype[dtype],
    ]
    B_Down_ty = np.ndarray[
        (ffn_intermediate_size * embed_sz,),
        np.dtype[dtype],
    ]

    # Tensors living on the AIE-array
    q_ty = np.ndarray[(seq_tile, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, kv_seq_tile), np.dtype[dtype]]
    qk_ty = np.ndarray[(seq_tile, kv_seq_tile), np.dtype[dtype]]
    v_ty = np.ndarray[(kv_seq_tile, d), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * seq_tile,), np.dtype[dtype]]
    wo_ty = np.ndarray[(d, emb_tile), np.dtype[dtype]]
    o_ty = np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]]
    ffn_b_ty = np.ndarray[(emb_tile, emb_tile), np.dtype[dtype]]

    # AIE kernel declarations
    bin_name = kernel_archive

    zero_kernel = Kernel(f"zero_{dtype_str}", bin_name, [qk_ty])

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
        [qk_ty, s_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
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

    # Layer norm (Add & Norm stage) kernel declarations
    ln_weights_ty = np.ndarray[(embed_sz,), np.dtype[dtype]]
    sum_l1_ty = np.ndarray[(seq_tile,), np.dtype[np.float32]]

    ln_zero_f32_kernel = Kernel(
        "ln_zero_f32",
        bin_name,
        [sum_l1_ty, np.int32],
    )
    ln_calc_sum_sumsq_kernel = Kernel(
        "ln_calc_sum_sumsq",
        bin_name,
        [o_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_fused_layer_norm_kernel = Kernel(
        "fused_layer_norm_1outs",
        bin_name,
        [o_ty, sum_l1_ty, sum_l1_ty, o_ty, np.int32],
    )
    ln_fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_1outs",
        bin_name,
        [o_ty, o_ty, ln_weights_ty, sum_l1_ty, sum_l1_ty, o_ty, np.int32, np.int32],
    )
    ln_mul_add_kernel = Kernel(
        "ln_mul_add_1outs",
        bin_name,
        [o_ty, o_ty, ln_weights_ty, o_ty, np.int32],
    )

    # FFN + second AddNorm kernels from encoder.cc
    ffn_zero_kernel_up_proj = Kernel(
        f"ffn_zero_{dtype_str}_up_proj",
        bin_name,
        [o_ty],
    )
    ffn_zero_kernel_down_proj = Kernel(
        f"ffn_zero_{dtype_str}_down_proj",
        bin_name,
        [o_ty],
    )
    ffn_matmul_kernel_up_proj = Kernel(
        f"ffn_matmul_{dtype_str}_{dtype_str}_up_proj",
        bin_name,
        [o_ty, ffn_b_ty, o_ty],
    )
    ffn_matmul_kernel_down_proj = Kernel(
        f"ffn_matmul_with_acc_{dtype_str}_{dtype_str}_down_proj",
        bin_name,
        [o_ty, ffn_b_ty, o_ty, o_ty],
    )
    ffn_gelu_kernel = Kernel(
        f"ffn_gelu_{dtype_str}",
        bin_name,
        [o_ty, o_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    q_dims = [(seq_tile // r, r * d), (d // s, s), (r, d), (s, 1)]

    inQ = ObjectFifo(
        np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]],
        name="inQ",
        depth=of_depth,
    )
    memQ = inQ.cons().split(
        offsets=[seq_tile * d * i for i in range(parallel_heads)],
        obj_types=[q_ty] * parallel_heads,
        names=[f"memQ{i}" for i in range(parallel_heads)],
        dims_to_stream=[q_dims] * parallel_heads,
        depths=[of_depth] * parallel_heads,
        placement=Tile(col=0, row=1),
    )  # Split between N parallel blocks of sequences

    # VJUNG: The SequentialPlacer will place all of these on the same MemTile if Placement is specified. We would need a list of placement in case of one-many or many-one.
    # I think the Sequential Placer will fail if we do a split/join with more than 6 I/Os cuz it tries to place them all on the same tile.

    # K is stored in column-major order
    k_dims = [(kv_seq_tile // t, t * d), (d // s, s), (t, d), (s, 1)]
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
        (kv_seq_tile // s, s * kv_seq_tile),
        (d // t, t),
        (s, kv_seq_tile),
        (t, 1),
    ]

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
    a_dims_out = [
        (kv_seq_tile // s, r * s),
        (seq_tile // r, kv_seq_tile * r),
        (r * s, 1),
    ]
    a_dims_in = [(kv_seq_tile // s, s), (seq_tile, kv_seq_tile), (s, 1)]
    for i in range(parallel_heads):
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
    # First send microkernel tiles across the sequence dimension of the output,
    # then place those microkernel tiles in the correct locations with another DMA
    p_dims_out = [(kv_seq_tile // s, s), (seq_tile, kv_seq_tile), (s, 1)]
    p_dims_in = [
        (kv_seq_tile // s, r * s),
        (seq_tile // r, kv_seq_tile * r),
        (r * s, 1),
    ]
    for i in range(parallel_heads):
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
    for i in range(parallel_heads):
        scaleOF.append(
            ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}")
        )  # Local to 1 parallel block of sequences

    # Output projection weights
    ow_dims = [
        (d // s, s * emb_tile),
        (emb_tile // t, t),
        (s, emb_tile),
        (t, 1),
    ]

    # Large emb_tile (e.g. 128) with depth=2 can overflow o-proj core L1 on
    # larger models; keep a single buffered WO tile in that case.
    ow_fifo_depth = 1 if emb_tile >= 128 else of_depth
    inOW = ObjectFifo(
        np.ndarray[(d * parallel_heads, emb_tile), np.dtype[dtype]],
        name="inOW",
        depth=ow_fifo_depth,
    )
    memOW = inOW.cons().split(
        offsets=[d * emb_tile * i for i in range(parallel_heads)],
        obj_types=[wo_ty] * parallel_heads,
        names=[f"memOW{i}" for i in range(parallel_heads)],
        dims_to_stream=[ow_dims] * parallel_heads,
        depths=[ow_fifo_depth] * parallel_heads,
        placement=Tile(col=3, row=1),
    )  # Split between N parallel blocks of heads

    # Partial out proj tiles to store accumulations in MTs.
    outOProj = []
    outOProjAccumIn = []
    outOProjAccumOut = []
    # Memtile DMA/BD pressure model (per tile):
    # - col4: memBUp stream pair (low depth), can host 2 O-proj accum FIFOs.
    # - col5: memBDown stream pair (low depth), can host 2 O-proj accum FIFOs.
    # - col6: ffnDownAccum stream pair (depth=proj_acc_depth), host at most 1.
    # - col7: memR + memLN2 stream pairs, host at most 2 O-proj accum FIFOs.
    # Keep O-proj accum FIFOs off cols 6/7 for <=4 parallel heads so those
    # memtiles can absorb FFN/LN traffic at high seq/head configurations.
    if parallel_heads <= 4:
        # Keep col5 in the <=4-head map and avoid concentrating depth-8
        # O-proj accumulators on col6 where FFN tail staging is anchored.
        acc_mem_tile_order = [4, 5, 7, 3]
    elif proj_acc_depth >= 6:
        # Keep high-head accumulator streams on 4/5/6/7. Col3 already carries
        # high fanout from W_O staging and can hit output-channel limits.
        acc_mem_tile_order = [4, 5, 6, 7, 4, 5, 7]
        acc_mem_tile_order = _override_int_list_env(
            "ENCODER_ACC_MEM_TILE_ORDER_PH_GE6_ACC_GE6",
            acc_mem_tile_order,
        )
    else:
        # For wider MHA parallelism, spread across available memtiles.
        acc_mem_tile_order = [4, 5, 6, 7, 4, 5, 7]
    if parallel_heads > len(acc_mem_tile_order):
        raise ValueError(
            "Unsupported parallel_heads for current memtile DMA budget: "
            f"{parallel_heads} > {len(acc_mem_tile_order)}"
        )
    acc_mem_tile_cols = acc_mem_tile_order[:parallel_heads]
    for i in range(parallel_heads):
        outOProj.append(
            ObjectFifo(q_ty, depth=of_depth, name=f"outOProj{i}")
        )  # Local to 1 parallel block of heads
        outOProjAccumOut.append(ObjectFifo(o_ty, depth=1, name=f"outOProjAccumOut{i}"))
        outOProjAccumIn.append(
            outOProjAccumOut[i]
            .cons(depth=proj_acc_depth)
            .forward(
                name=f"outOProjAccumIn{i}",
                depth=proj_acc_depth,
                placement=Tile(col=acc_mem_tile_cols[i], row=1),
            )
        )
        logging.debug(
            "Placed outOProjAccum[%d] on mem tile (%d,1) with acc_depth=%d",
            i,
            acc_mem_tile_cols[i],
            proj_acc_depth,
        )

    outOPart = []
    for i in range(parallel_heads - 1):
        outOPart.append(
            ObjectFifo(o_ty, depth=of_depth, name=f"outOPart{i}")
        )  # Local to 1 parallel block of heads

    # Keep non-accumulation FIFOs shallow to avoid L1 over-allocation on FFN-down.
    ln_fifo_depth = of_depth
    # FFN-up input FIFO depth (kept shallow to reduce FFN-up core L1 usage).
    ffn_up_input_depth = 1
    # Keep LN1 buffering shallow to stay within L1 budget on LN1 tiles.
    ln_input_depth = 1
    # O-proj stream into LN1 norm worker (no DMA layout transform).
    outOProjInput = ObjectFifo(
        o_ty,
        name="outOProjInput",
        depth=1,
    )
    # LN1 norm replays the per-col O-proj tiles for FFN-group fanout.
    ln1_replay_mem_tile_col = 5
    ln1_replay_mem_tile_col = _override_int_env(
        "ENCODER_LN1_REPLAY_MEM_TILE_COL",
        ln1_replay_mem_tile_col,
    )
    ln1ReplayPart = ObjectFifo(o_ty, name="ln1ReplayPart", depth=1)
    ln1Replay = ln1ReplayPart.cons(depth=ln_tiles_per_q_block).forward(
        obj_type=o_ty,
        name="ln1Replay",
        depth=ln_tiles_per_q_block,
        placement=Tile(col=ln1_replay_mem_tile_col, row=1),
    )
    # LN1 split-stage link (norm output -> mul+resadd input).
    ln1Norm = ObjectFifo(o_ty, name="ln1Norm", depth=ln_input_depth)
    r_dims = [
        (seq_tile // r, r * emb_tile),
        (emb_tile // s, s),
        (r, emb_tile),
        (s, 1),
    ]
    # AddNorm-2 consumes residual only on its second pass. Keep enough depth
    # for one full output tile-group to avoid backpressure deadlock on LN1.
    ffn_residual_depth = proj_acc_depth
    ln_mem_tile_col = 7
    # Residual R
    inR = ObjectFifo(
        o_ty,
        name="inR",
        depth=ln_fifo_depth,
    )
    memR = inR.cons().forward(
        obj_type=o_ty,
        name="memR",
        dims_to_stream=r_dims,
        depth=ln_fifo_depth,
        placement=Tile(col=ln_mem_tile_col, row=1),
    )

    # LN output
    o_dims = [(seq_tile // r, r * emb_tile), (r, s), (emb_tile // s, r * s), (s, 1)]
    # Keep FFN branch staging streams off LN replay (col 5) when MHA uses many
    # heads; col 5 already carries LN1 full-row replay traffic.
    if parallel_heads >= 6 and proj_acc_depth >= 6:
        branch_stage_cols = [6, 4, 5]
        branch_down_b_cols = [4, 6, 5]
        branch_stage_cols = _override_int_list_env(
            "ENCODER_BRANCH_STAGE_COLS_PH_GE6_ACC_GE6",
            branch_stage_cols,
        )
        branch_down_b_cols = _override_int_list_env(
            "ENCODER_BRANCH_DOWN_B_COLS_PH_GE6_ACC_GE6",
            branch_down_b_cols,
        )
    elif parallel_heads >= 6:
        branch_stage_cols = [6, 4, 5]
        branch_down_b_cols = [4, 6, 5]
    else:
        branch_stage_cols = [6, 5, 4]
        branch_down_b_cols = [6, 5, 4]
    if effective_ffn_branches > len(branch_stage_cols):
        raise ValueError(
            "Unsupported effective_ffn_branches for current memtile assignment "
            f"({effective_ffn_branches} > {len(branch_stage_cols)})"
        )
    ffn_a_stage_mem_tile_cols = branch_stage_cols[:effective_ffn_branches]
    outLN = []
    memOutLN = []
    for branch_idx in range(effective_ffn_branches):
        outLN.append(
            ObjectFifo(
                o_ty,
                name="outLN" if branch_idx == 0 else f"outLNFfn{branch_idx}",
                depth=ffn_up_input_depth,
            )
        )
        # Stage FFN-up A-input tiles in mem tile(s) before FFN-up consumption.
        memOutLN.append(
            outLN[branch_idx]
            .cons(depth=ffn_up_input_depth)
            .forward(
                obj_type=o_ty,
                name="memOutLN" if branch_idx == 0 else f"memOutLNFfn{branch_idx}",
                depth=proj_acc_depth,
                placement=Tile(col=ffn_a_stage_mem_tile_cols[branch_idx], row=1),
            )
        )
    # FFN residual path for AddNorm-2, staged through mem tile from LN1 post core.
    ffnROut = ObjectFifo(o_ty, name="ffnROut", depth=ffn_residual_depth)
    ffnRIn = ffnROut.cons(depth=ffn_residual_depth).forward(
        obj_type=o_ty,
        name="ffnRIn",
        depth=ffn_residual_depth,
        placement=Tile(col=ln_mem_tile_col, row=1),
    )
    ffn_residual_prod = ffnROut.prod()

    # FFN weights (Up/Down projections)
    b_dims = [
        (emb_tile // s, s * emb_tile),
        (emb_tile // t, t),
        (s, emb_tile),
        (t, 1),
    ]
    # Keep FFN weight FIFOs at depth 1 to reduce L1 pressure on FFN cores.
    ffn_weight_fifo_depth = 1
    ffn_bup_mem_tile_cols = branch_stage_cols[:effective_ffn_branches]
    ffn_bdown_mem_tile_cols = branch_down_b_cols[:effective_ffn_branches]
    inBUp = []
    memBUp = []
    inBDown = []
    memBDown = []
    for branch_idx in range(effective_ffn_branches):
        inBUp.append(
            ObjectFifo(
                ffn_b_ty,
                name="inBUp" if branch_idx == 0 else f"inBUp{branch_idx}",
                depth=ffn_weight_fifo_depth,
            )
        )
        memBUp.append(
            inBUp[branch_idx]
            .cons()
            .forward(
                obj_type=ffn_b_ty,
                name="memBUp" if branch_idx == 0 else f"memBUp{branch_idx}",
                dims_to_stream=b_dims,
                depth=ffn_weight_fifo_depth,
                placement=Tile(col=ffn_bup_mem_tile_cols[branch_idx], row=1),
            )
        )
        inBDown.append(
            ObjectFifo(
                ffn_b_ty,
                name="inBDown" if branch_idx == 0 else f"inBDown{branch_idx}",
                depth=ffn_weight_fifo_depth,
            )
        )
        memBDown.append(
            inBDown[branch_idx]
            .cons()
            .forward(
                obj_type=ffn_b_ty,
                name="memBDown" if branch_idx == 0 else f"memBDown{branch_idx}",
                dims_to_stream=b_dims,
                depth=ffn_weight_fifo_depth,
                placement=Tile(col=ffn_bdown_mem_tile_cols[branch_idx], row=1),
            )
        )

    # FFN internal pipelines and final encoder output
    ffnUpOut = []
    ffnDownPart = []
    ffnDownAccum = []
    ffn_down_acc_mem_tile_cols = branch_stage_cols[:effective_ffn_branches]
    for branch_idx in range(effective_ffn_branches):
        ffnUpOut.append(
            ObjectFifo(
                o_ty,
                name="ffnUpOut" if branch_idx == 0 else f"ffnUpOut{branch_idx}",
                depth=1,
            )
        )
        # Keep FFN-down accumulation in mem tile FIFO(s) so down-proj core L1 stays
        # within limits while replaying for LN2's two-pass consumption.
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
                placement=Tile(col=ffn_down_acc_mem_tile_cols[branch_idx], row=1),
            )
        )
    ffnDownReduce = []
    for src_branch_idx in ffn_reduction_sources:
        ffnDownReduce.append(
            ObjectFifo(
                o_ty,
                name=f"ffnDownReduce{src_branch_idx}",
                depth=1,
            )
        )
    ffn_down_out_depth = 1 if emb_tile >= 128 else ln_fifo_depth
    ffnDownOut = ObjectFifo(
        o_ty,
        name="ffnDownOut",
        depth=ffn_down_out_depth,
    )
    outLN2 = ObjectFifo(o_ty, name="outLN2", depth=ln_fifo_depth)
    memLN2 = outLN2.cons().forward(
        obj_type=o_ty,
        name="memLN2",
        dims_to_stream=o_dims,
        depth=ln_fifo_depth,
        placement=Tile(col=ln_mem_tile_col, row=1),
    )

    # Additional LN1 fanout stage when more than two FFN branches are active.
    outLNRest = None
    if ffn_requires_ln1_router:
        outLNRest = ObjectFifo(
            o_ty,
            name="outLNFfnRest",
            depth=1,
        )

    def batched_matmul_qk(
        of_q,
        of_k,
        of_a_out,
        zero,
        matmul_QK,
        q_block_bias,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            # NOTE: Second element in idx_buffer used to be set to q_block_bias, which
            # seems to be used for causal masking and for when
            # attention is parallelized across the sequence dimension. For this
            # design, it shouldn't be getting used since we parallelize across heads.
            # Since it affects the computations, we set the value to 0.
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                elem_in_q = of_q.acquire(1)

                for _ in range_(num_kv_seq_blocks):

                    elem_in_k = of_k.acquire(1)
                    elem_a_out = of_a_out.acquire(1)

                    zero(elem_a_out)
                    matmul_QK(elem_in_q, elem_in_k, elem_a_out, idx_buffer)

                    of_k.release(1)
                    of_a_out.release(1)

                    idx_buffer[0] += 0
                idx_buffer[0] = 0
                idx_buffer[1] += 0

                of_q.release(1)

    def softmax(
        of_in_a,
        of_out_p,
        of_out_scale,
        partial_softmax,
        init_scale_buffer,
        memcopy_kernel_scale,
        q_block_bias,
        idx_buffer,
        scale_buffer,
    ):

        # VJUNG: The index buffer count how many Q and KV block this worker has processed
        # From this info we can infer the position in A and P

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                init_scale_buffer(scale_buffer, seq_tile)

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
                        seq_tile,
                        kv_seq_tile,
                        seq_len,
                        seq_len,
                    )
                    memcopy_kernel_scale(scale_buffer, elt_of_out_scale, 4 * seq_tile)

                    of_in_a.release(1)
                    of_out_p.release(1)
                    of_out_scale.release(1)

                    idx_buffer[0] += 0
                idx_buffer[0] = 0
                idx_buffer[1] += 0  # Used to be parameter for seq block parallelism

    def batched_matmul_pv(
        of_p,
        of_v,
        of_scale,
        of_o_out,
        zero,
        matmul_PV,
        rescale_O,
        q_block_bias,
        idx_buffer,
    ):

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = 0

            # One outer iteration per head-block chunk. The inner body already
            # consumes all KV sequence blocks for that chunk.
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
                    seq_tile,
                    0,
                    idx_buffer,
                )

                of_p.release(1)
                of_v.release(1)
                of_scale.release(1)

                idx_buffer[0] += 0
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
                            seq_tile,
                            1,
                            idx_buffer,
                        )

                        of_p.release(1)
                        of_v.release(1)
                        of_scale.release(1)

                        idx_buffer[0] += 0

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
                        seq_tile,
                        1,
                        idx_buffer,
                    )
                    rescale_O(elem_o_out, elt_of_out_scale3, seq_tile, idx_buffer)

                    of_p.release(1)
                    of_v.release(1)
                    of_scale.release(1)

                    idx_buffer[0] += 0
                # else:
                else:
                    rescale_O(elem_o_out, elt_of_out_scale, seq_tile, idx_buffer)
                    idx_buffer[0] += 0
                ###

                idx_buffer[0] = 0
                idx_buffer[1] += 0  # Used to be parameter for seq block parallelism

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
        is_last_o_proj_worker,
    ):
        """
        Amount of work for prev stages to generate its ouptut to next stage for one head:
            QK: seq_tile * d * seq_tile
            S: seq_tile * seq_tile
            PV: seq_tile * seq_tile * d * (seq_len // seq_tile)
        Amount of work for prev stages to process generate full row outputs:
            QK: seq_tile * d * seq_tile * (seq_len // seq_tile)
            S: seq_tile * seq_tile * (seq_len // seq_tile)
            PV: seq_tile * seq_tile * d * (seq_len // seq_tile) * (embed_dim // d)
        Output projection needs all heads to generate one output tile:
            O: seq_tile * d * emb_tile * (embed_dim // d)
        So full rows are generated after this amount of compute:
            O: seq_tile * d * emb_tile * (embed_dim // d) * (embed_dim // emb_tile)
        The output from PV can be reused to partially accumulate output tiles:
            O: seq_tile * d * emb_tile * (embed_dim // emb_tile)
        """

        for _ in range_(sys.maxsize):
            # First iteration just passes the partial C tile through
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

                # Acquire what's in L2, which is the final accumulated result for the tile
                elem_in_o_acc = of_o_acc_in.acquire(1)

                if buffer_to_reduce:

                    # Don't send any new data to MT, i.e. of_o_acc_out, because that will affect
                    # the data in the subsequent tiles. It's sufficient to just use
                    # the internal buffer as input and output
                    partial_o_acc = buffer_to_reduce.acquire(1)
                    add(
                        partial_o_acc, elem_in_o_acc, elem_in_o_acc, seq_tile * emb_tile
                    )
                    buffer_to_reduce.release(1)

                elem_out_o = of_o_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                of_o_acc_in.release(1)
                of_o_out.release(1)

    def core_fn_ln1_norm(
        of_in_o_proj,
        of_replay_curr,
        of_replay_new,
        sum_buf,
        sumsq_buf,
        of_out_norm,
        norm_scratch,
        fused_layer_norm,
        calc_sum_sumsq,
        zero_f32,
        copy,
        addnorm1_mode,
        stage_only,
    ):
        # Mirror down-proj style stage-only semantics: if this stage is not active,
        # keep FIFO traffic/replay shape but bypass heavy LN statistics/math.
        ln1_norm_compute_enabled = (stage_only in [None, 4]) and (addnorm1_mode == -1)
        for _ in range_(sys.maxsize):
            if ln1_norm_compute_enabled:
                zero_f32(sum_buf, seq_tile)
                zero_f32(sumsq_buf, seq_tile)
            # Pass 1 on raw O-proj output: accumulate row-wise statistics and seed replay FIFO.
            for _ in range_(ln_tiles_per_q_block):
                elem_in = of_in_o_proj.acquire(1)
                if ln1_norm_compute_enabled:
                    calc_sum_sumsq(elem_in, sum_buf, sumsq_buf)
                elem_replay = of_replay_new.acquire(1)
                copy(elem_in, elem_replay, seq_tile * emb_tile)
                of_replay_new.release(1)
                of_in_o_proj.release(1)
            # Pass 2: emit normalized tiles in FFN-group-major order.
            # Compute LN once per tile, then replay normalized tiles for later groups.
            for _ in range_(ln_tiles_per_q_block):
                elem_in = of_replay_curr.acquire(1)
                if ln1_norm_compute_enabled:
                    fused_layer_norm(
                        elem_in,
                        sum_buf,
                        sumsq_buf,
                        norm_scratch,
                        embed_sz,
                    )
                else:
                    copy(elem_in, norm_scratch, seq_tile * emb_tile)
                elem_out_norm = of_out_norm.acquire(1)
                copy(norm_scratch, elem_out_norm, seq_tile * emb_tile)
                of_out_norm.release(1)
                if profile_replay_groups > 1:
                    elem_replay = of_replay_new.acquire(1)
                    copy(norm_scratch, elem_replay, seq_tile * emb_tile)
                    of_replay_new.release(1)
                of_replay_curr.release(1)
            if profile_replay_groups > 2:
                for _ in range_(profile_replay_groups - 2):
                    for _ in range_(ln_tiles_per_q_block):
                        elem_in = of_replay_curr.acquire(1)
                        elem_out_norm = of_out_norm.acquire(1)
                        copy(elem_in, elem_out_norm, seq_tile * emb_tile)
                        of_out_norm.release(1)
                        elem_replay = of_replay_new.acquire(1)
                        copy(elem_in, elem_replay, seq_tile * emb_tile)
                        of_replay_new.release(1)
                        of_replay_curr.release(1)
            if profile_replay_groups > 1:
                for _ in range_(ln_tiles_per_q_block):
                    elem_in = of_replay_curr.acquire(1)
                    elem_out_norm = of_out_norm.acquire(1)
                    copy(elem_in, elem_out_norm, seq_tile * emb_tile)
                    of_out_norm.release(1)
                    of_replay_curr.release(1)

    def core_fn_ln1_mul_add_single(
        of_in_norm,
        of_in_residual,
        weights,
        of_out_up,
        of_out_residual,
        ln_mul_add,
        copy,
        addnorm1_mode,
    ):
        for _ in range_(sys.maxsize):
            # Group 0.
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in1 = of_in_norm.acquire(1)
                elem_in2 = of_in_residual.acquire(1)
                elem_out_up = of_out_up.acquire(1)
                if addnorm1_mode == 0:
                    copy(elem_in1, elem_out_up, seq_tile * emb_tile)
                elif addnorm1_mode == 1:
                    copy(elem_in2, elem_out_up, seq_tile * emb_tile)
                else:
                    ln_mul_add(
                        elem_in1,
                        elem_in2,
                        weights,
                        elem_out_up,
                        col_i32,
                    )
                if of_out_residual:
                    elem_out_res = of_out_residual.acquire(1)
                    copy(elem_out_up, elem_out_res, seq_tile * emb_tile)
                    of_out_residual.release(1)
                of_out_up.release(1)
                of_in_norm.release(1)
                of_in_residual.release(1)
            for _ in range_(profile_replay_groups - 1):
                for col_idx in range_(ln_tiles_per_q_block):
                    col_i32 = index.casts(T.i32(), col_idx)
                    elem_in1 = of_in_norm.acquire(1)
                    elem_in2 = of_in_residual.acquire(1)
                    elem_out_up = of_out_up.acquire(1)
                    if addnorm1_mode == 0:
                        copy(elem_in1, elem_out_up, seq_tile * emb_tile)
                    elif addnorm1_mode == 1:
                        copy(elem_in2, elem_out_up, seq_tile * emb_tile)
                    else:
                        ln_mul_add(
                            elem_in1,
                            elem_in2,
                            weights,
                            elem_out_up,
                            col_i32,
                        )
                    of_out_up.release(1)
                    of_in_norm.release(1)
                    of_in_residual.release(1)

    def core_fn_ln1_mul_add_multi(
        of_in_norm,
        of_in_residual,
        weights,
        of_out_up_0,
        of_out_up_1,
        of_out_up_2_or_rest,
        ln_mul_add,
        copy,
        addnorm1_mode,
        route_remaining_branches,
    ):
        of_out_up_list = [of_out_up_0, of_out_up_1, of_out_up_2_or_rest]
        for _ in range_(sys.maxsize):
            # Emit FFN input in branch-major group order so each branch receives
            # only its assigned ffn_col_group_count tiles.
            for branch_idx in range(effective_ffn_branches):
                if route_remaining_branches and branch_idx >= 1:
                    of_out_up = of_out_up_list[2]
                else:
                    of_out_up = of_out_up_list[branch_idx]
                for branch_group_idx in range_(ffn_col_group_counts[branch_idx]):
                    for col_idx in range_(ln_tiles_per_q_block):
                        col_i32 = index.casts(T.i32(), col_idx)
                        elem_in1 = of_in_norm.acquire(1)
                        elem_in2 = of_in_residual.acquire(1)
                        elem_out_up = of_out_up.acquire(1)
                        if addnorm1_mode == 0:
                            copy(elem_in1, elem_out_up, seq_tile * emb_tile)
                        elif addnorm1_mode == 1:
                            copy(elem_in2, elem_out_up, seq_tile * emb_tile)
                        else:
                            ln_mul_add(
                                elem_in1,
                                elem_in2,
                                weights,
                                elem_out_up,
                                col_i32,
                            )
                        of_out_up.release(1)
                        of_in_norm.release(1)
                        of_in_residual.release(1)

    def core_fn_ln1_route_rest(
        of_in_rest,
        of_out_up_1,
        of_out_up_2,
        copy,
    ):
        for _ in range_(sys.maxsize):
            # Branch-1 groups.
            for _ in range_(ffn_col_group_counts[1]):
                for _ in range_(ln_tiles_per_q_block):
                    elem_in_rest = of_in_rest.acquire(1)
                    elem_out_up_1 = of_out_up_1.acquire(1)
                    copy(elem_in_rest, elem_out_up_1, seq_tile * emb_tile)
                    of_out_up_1.release(1)
                    of_in_rest.release(1)
            # Branch-2 groups.
            for _ in range_(ffn_col_group_counts[2]):
                for _ in range_(ln_tiles_per_q_block):
                    elem_in_rest = of_in_rest.acquire(1)
                    elem_out_up_2 = of_out_up_2.acquire(1)
                    copy(elem_in_rest, elem_out_up_2, seq_tile * emb_tile)
                    of_out_up_2.release(1)
                    of_in_rest.release(1)

    def core_fn_ffn_up_proj_single(
        of_in_a_curr,
        of_in_b,
        of_out_c,
        of_out_residual,
        zero,
        matmul,
        gelu,
        copy,
        group_count,
        stage_only,
    ):
        for _ in range_(sys.maxsize):
            for group_idx in range_(group_count):
                elem_out_matmul = of_out_c.acquire(1)
                # Keep stage-only paths deterministic: when FFN-up is disabled,
                # still emit a zero tile so downstream consumers do not read stale data.
                zero(elem_out_matmul)
                for _ in range_(proj_acc_depth):
                    elem_in_a = of_in_a_curr.acquire(1)
                    if of_out_residual and group_idx == 0:
                        elem_out_res = of_out_residual.acquire(1)
                        copy(elem_in_a, elem_out_res, seq_tile * emb_tile)
                        of_out_residual.release(1)
                    elem_in_b = of_in_b.acquire(1)
                    if stage_only in [0, None]:
                        matmul(elem_in_a, elem_in_b, elem_out_matmul)
                    of_in_b.release(1)
                    of_in_a_curr.release(1)
                if stage_only in [0, None] and gelu:
                    gelu(elem_out_matmul, elem_out_matmul, seq_tile * emb_tile)
                of_out_c.release(1)

    def core_fn_ffn_up_proj_multi(
        of_in_a_curr,
        of_in_b,
        of_out_c,
        of_out_residual,
        zero,
        matmul,
        gelu,
        copy,
        group_count,
        stage_only,
    ):
        for _ in range_(sys.maxsize):
            for group_idx in range_(group_count):
                elem_out_matmul = of_out_c.acquire(1)
                # Keep stage-only paths deterministic: when FFN-up is disabled,
                # still emit a zero tile so downstream consumers do not read stale data.
                zero(elem_out_matmul)
                for _ in range_(proj_acc_depth):
                    elem_in_a = of_in_a_curr.acquire(1)
                    if of_out_residual and group_idx == 0:
                        elem_out_res = of_out_residual.acquire(1)
                        copy(elem_in_a, elem_out_res, seq_tile * emb_tile)
                        of_out_residual.release(1)
                    elem_in_b = of_in_b.acquire(1)
                    if stage_only in [0, None]:
                        matmul(elem_in_a, elem_in_b, elem_out_matmul)
                    of_in_b.release(1)
                    of_in_a_curr.release(1)
                if stage_only in [0, None] and gelu:
                    gelu(elem_out_matmul, elem_out_matmul, seq_tile * emb_tile)
                of_out_c.release(1)

    def core_fn_ffn_down_proj_single(
        of_in_a,
        of_in_b,
        of_curr_acc,
        of_new_acc,
        of_out,
        zero,
        matmul,
        copy,
        group_count,
        stage_only,
    ):
        for _ in range_(sys.maxsize):
            # Check if down projection stage is enabled, None means all stages are enabled.
            if stage_only not in [
                1,
                None,
            ]:  # Skip computation for down projection stage
                for _ in range_(proj_acc_depth):
                    elem_acc = of_new_acc.acquire(1)
                    zero(elem_acc)
                    of_new_acc.release(1)
                for _ in range_(group_count):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_in_b = of_in_b.acquire(1)
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        zero(elem_new_acc)
                        of_in_b.release(1)
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_out = of_out.acquire(1)
                    zero(elem_out)
                    of_curr_acc.release(1)
                    elem_new_acc = of_new_acc.acquire(1)
                    zero(elem_new_acc)
                    of_new_acc.release(1)
                    of_out.release(1)
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_out = of_out.acquire(1)
                    zero(elem_out)
                    of_out.release(1)
                    of_curr_acc.release(1)
            else:  # Perform down projection stage computation
                # First iteration just passes the partial C tile through.
                for _ in range_(proj_acc_depth):
                    elem_acc = of_new_acc.acquire(1)
                    zero(elem_acc)
                    of_new_acc.release(1)
                for _ in range_(group_count):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_in_b = of_in_b.acquire(1)
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_curr_acc, elem_new_acc)
                        of_in_b.release(1)
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)
                for _ in range_(proj_acc_depth):
                    # Acquire what's in L2, which is the final accumulated result for the tile.
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_out = of_out.acquire(1)
                    copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                    of_curr_acc.release(1)
                    elem_new_acc = of_new_acc.acquire(1)
                    # Make sure to copy the final accumulated C tile for the LN second pass.
                    copy(elem_out, elem_new_acc, seq_tile * emb_tile)
                    of_new_acc.release(1)
                    of_out.release(1)
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_out = of_out.acquire(1)
                    copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                    of_out.release(1)
                    of_curr_acc.release(1)

    def core_fn_ffn_down_proj_multi(
        of_in_a,
        of_in_b,
        of_curr_acc,
        of_new_acc,
        of_out,
        zero,
        matmul,
        add,
        copy,
        group_count,
        buffer_to_reduce,
        is_final_branch,
        stage_only,
    ):
        for _ in range_(sys.maxsize):
            # Check if down projection stage is enabled, None means all stages are enabled.
            if stage_only not in [
                1,
                None,
            ]:  # Skip computation for down projection stage
                for _ in range_(proj_acc_depth):
                    elem_acc = of_new_acc.acquire(1)
                    zero(elem_acc)
                    of_new_acc.release(1)
                for _ in range_(group_count):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_in_b = of_in_b.acquire(1)
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        zero(elem_new_acc)
                        of_in_b.release(1)
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    if buffer_to_reduce:
                        partial_acc = buffer_to_reduce.acquire(1)
                        buffer_to_reduce.release(1)
                    elem_out = of_out.acquire(1)
                    zero(elem_out)
                    of_curr_acc.release(1)
                    if is_final_branch:
                        elem_new_acc = of_new_acc.acquire(1)
                        zero(elem_new_acc)
                        of_new_acc.release(1)
                    of_out.release(1)
                if is_final_branch:
                    for _ in range_(proj_acc_depth):
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_out = of_out.acquire(1)
                        zero(elem_out)
                        of_out.release(1)
                        of_curr_acc.release(1)
            else:  # Perform down projection stage computation
                # First iteration just passes the partial C tile through.
                for _ in range_(proj_acc_depth):
                    elem_acc = of_new_acc.acquire(1)
                    zero(elem_acc)
                    of_new_acc.release(1)
                for _ in range_(group_count):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_in_b = of_in_b.acquire(1)
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_curr_acc, elem_new_acc)
                        of_in_b.release(1)
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)
                for _ in range_(proj_acc_depth):
                    # Acquire what's in L2, which is the final accumulated result for the tile.
                    elem_curr_acc = of_curr_acc.acquire(1)
                    if buffer_to_reduce:
                        partial_acc = buffer_to_reduce.acquire(1)
                        add(
                            partial_acc,
                            elem_curr_acc,
                            elem_curr_acc,
                            seq_tile * emb_tile,
                        )
                        buffer_to_reduce.release(1)
                    elem_out = of_out.acquire(1)
                    copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                    of_curr_acc.release(1)
                    if is_final_branch:
                        elem_new_acc = of_new_acc.acquire(1)
                        # Make sure to copy the final accumulated C tile for the LN second pass.
                        copy(elem_out, elem_new_acc, seq_tile * emb_tile)
                        of_new_acc.release(1)
                    of_out.release(1)
                if is_final_branch:
                    for _ in range_(proj_acc_depth):
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_out = of_out.acquire(1)
                        copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                        of_out.release(1)
                        of_curr_acc.release(1)

    def core_fn_add_norm2(
        of_in1,
        of_in2,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        fused_add_layer_norm,
        calc_sum_sumsq,
        zero_f32,
        copy,
        stage_only,
        addnorm2_mode,
    ):
        for _ in range_(sys.maxsize):
            # Check if second add & norm stage is enabled, None means all stages are enabled.
            if stage_only not in [
                2,
                None,
            ]:  # Skip computation for second add & norm stage
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    of_in1.release(1)
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    copy(elem_in2, elem_out, seq_tile * emb_tile)
                    of_out.release(1)
                    of_in1.release(1)
                    of_in2.release(1)
            elif addnorm2_mode not in [-1]:
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    of_in1.release(1)
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    if addnorm2_mode == 0:
                        copy(elem_in1, elem_out, seq_tile * emb_tile)
                    else:
                        copy(elem_in2, elem_out, seq_tile * emb_tile)
                    of_out.release(1)
                    of_in1.release(1)
                    of_in2.release(1)
            else:
                zero_f32(sum_buf, seq_tile)
                zero_f32(sumsq_buf, seq_tile)
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    calc_sum_sumsq(elem_in1, sum_buf, sumsq_buf)
                    of_in1.release(1)

                for col_idx in range_(proj_acc_depth):
                    col_i32 = index.casts(T.i32(), col_idx)
                    elem_in1 = of_in1.acquire(1)
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    fused_add_layer_norm(
                        elem_in1,
                        elem_in2,
                        weights,
                        sum_buf,
                        sumsq_buf,
                        elem_out,
                        embed_sz,
                        col_i32,
                    )
                    of_out.release(1)
                    of_in1.release(1)
                    of_in2.release(1)

    # Create worker from task
    matmul_workers = []
    softmax_workers = []
    matmul_pv_workers = []
    o_proj_workers = []
    for i in range(parallel_heads):
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
                    i,
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
            initial_value=np.zeros(shape=(4 * seq_tile,), dtype=dtype),
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
                    i,
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
                    zero_kernel,
                    matmul_PV,
                    rescale_O,
                    i,
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
                    # Last head writes to the LN input stream source, others write to partial accumulation tiles
                    (
                        outOPart[i].prod()
                        if i < parallel_heads - 1
                        else outOProjInput.prod()
                    ),
                    outOPart[i - 1].cons() if i > 0 else None,
                    zero_kernel_o_proj,
                    matmul_kernel_o_proj,
                    eltwise_add_vector,
                    mem_copy_o_proj,
                    i == parallel_heads - 1,  # is_last_o_proj_worker
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=5),
                while_true=False,
            )
        )
        logging.debug(
            "Configured o_proj worker %d with acc_depth=%d (is_last=%s): "
            "per q-block emit order is pass1[acc=0..%d] then pass2[acc=0..%d]",
            i,
            proj_acc_depth,
            i == parallel_heads - 1,
            proj_acc_depth - 1,
            proj_acc_depth - 1,
        )

    # Create split LN1 workers (norm + mul/add).
    ln1_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=static_ln1_weights,
        name="static_ln1_weights",
    )
    ln1_norm_sum_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sum_buffer")
    ln1_norm_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sumsq_buffer")
    ln1_norm_scratch = Buffer(type=o_ty, name="ln1_norm_scratch")

    ln1_norm_worker = Worker(
        core_fn_ln1_norm,
        fn_args=[
            outOProjInput.cons(),
            ln1Replay.cons(depth=1),
            ln1ReplayPart.prod(),
            ln1_norm_sum_buffer,
            ln1_norm_sumsq_buffer,
            ln1Norm.prod(),
            ln1_norm_scratch,
            ln_fused_layer_norm_kernel,
            ln_calc_sum_sumsq_kernel,
            ln_zero_f32_kernel,
            mem_copy_o_proj,
            addnorm1_debug_mode,
            ffn_stage_only,
        ],
        placement=Tile(col=ln1_tile[0], row=ln1_tile[1]),
        while_true=False,
    )
    if effective_ffn_branches == 1:
        ln1_muladd_worker = Worker(
            core_fn_ln1_mul_add_single,
            fn_args=[
                ln1Norm.cons(),
                memR.cons(),
                ln1_weight_buffer,
                outLN[0].prod(),
                ffn_residual_prod,
                ln_mul_add_kernel,
                mem_copy_o_proj,
                addnorm1_debug_mode,
            ],
            placement=Tile(col=ln1_post_tile[0], row=ln1_post_tile[1]),
            while_true=False,
        )
    else:
        ln1_muladd_worker = Worker(
            core_fn_ln1_mul_add_multi,
            fn_args=[
                ln1Norm.cons(),
                memR.cons(),
                ln1_weight_buffer,
                outLN[0].prod() if effective_ffn_branches > 0 else None,
                (
                    None
                    if ffn_requires_ln1_router
                    else (outLN[1].prod() if effective_ffn_branches > 1 else None)
                ),
                (
                    outLNRest.prod()
                    if ffn_requires_ln1_router
                    else (outLN[2].prod() if effective_ffn_branches > 2 else None)
                ),
                ln_mul_add_kernel,
                mem_copy_o_proj,
                addnorm1_debug_mode,
                ffn_requires_ln1_router,
            ],
            placement=Tile(col=ln1_post_tile[0], row=ln1_post_tile[1]),
            while_true=False,
        )
    ln1_route_worker = None
    if ffn_requires_ln1_router:
        if ln1_route_tile is None:
            raise ValueError("Missing LN1 route tile for multi-branch fanout")
        ln1_route_worker = Worker(
            core_fn_ln1_route_rest,
            fn_args=[
                outLNRest.cons(),
                outLN[1].prod(),
                outLN[2].prod(),
                mem_copy_o_proj,
            ],
            placement=Tile(col=ln1_route_tile[0], row=ln1_route_tile[1]),
            while_true=False,
        )

    ffn_up_workers = []
    for branch_idx in range(effective_ffn_branches):
        if effective_ffn_branches == 1:
            ffn_up_worker_fn = core_fn_ffn_up_proj_single
            ffn_up_worker_args = [
                memOutLN[branch_idx].cons(depth=1),
                memBUp[branch_idx].cons(),
                ffnUpOut[branch_idx].prod(),
                None,
                ffn_zero_kernel_up_proj,
                ffn_matmul_kernel_up_proj,
                ffn_gelu_kernel,
                mem_copy_o_proj,
                ffn_col_group_counts[branch_idx],
                ffn_stage_only,
            ]
        else:
            ffn_up_worker_fn = core_fn_ffn_up_proj_multi
            ffn_up_worker_args = [
                memOutLN[branch_idx].cons(depth=1),
                memBUp[branch_idx].cons(),
                ffnUpOut[branch_idx].prod(),
                ffn_residual_prod if branch_idx == 0 else None,
                ffn_zero_kernel_up_proj,
                ffn_matmul_kernel_up_proj,
                ffn_gelu_kernel,
                mem_copy_o_proj,
                ffn_col_group_counts[branch_idx],
                ffn_stage_only,
            ]
        ffn_up_workers.append(
            Worker(
                ffn_up_worker_fn,
                fn_args=ffn_up_worker_args,
                placement=Tile(
                    col=selected_up_tiles[branch_idx][0],
                    row=selected_up_tiles[branch_idx][1],
                ),
                stack_size=0x700,
                while_true=False,
            )
        )

    ffn_down_workers = []
    for branch_idx in range(effective_ffn_branches):
        is_final_branch = branch_idx == final_ffn_branch_idx
        if effective_ffn_branches == 1:
            ffn_down_worker_fn = core_fn_ffn_down_proj_single
            ffn_down_worker_args = [
                ffnUpOut[branch_idx].cons(),
                memBDown[branch_idx].cons(),
                ffnDownAccum[branch_idx].cons(depth=1),
                ffnDownPart[branch_idx].prod(),
                ffnDownOut.prod(ffn_down_out_depth),
                ffn_zero_kernel_down_proj,
                ffn_matmul_kernel_down_proj,
                mem_copy_o_proj,
                ffn_col_group_counts[branch_idx],
                ffn_stage_only,
            ]
        else:
            reduce_in = ffnDownReduce[branch_idx - 1].cons() if branch_idx > 0 else None
            reduce_out = (
                ffnDownOut.prod(ffn_down_out_depth)
                if is_final_branch
                else ffnDownReduce[branch_idx].prod()
            )
            ffn_down_worker_fn = core_fn_ffn_down_proj_multi
            ffn_down_worker_args = [
                ffnUpOut[branch_idx].cons(),
                memBDown[branch_idx].cons(),
                ffnDownAccum[branch_idx].cons(depth=1),
                ffnDownPart[branch_idx].prod(),
                reduce_out,
                ffn_zero_kernel_down_proj,
                ffn_matmul_kernel_down_proj,
                eltwise_add_vector,
                mem_copy_o_proj,
                ffn_col_group_counts[branch_idx],
                reduce_in,
                is_final_branch,
                ffn_stage_only,
            ]
        ffn_down_workers.append(
            Worker(
                ffn_down_worker_fn,
                fn_args=ffn_down_worker_args,
                placement=Tile(
                    col=selected_down_tiles[branch_idx][0],
                    row=selected_down_tiles[branch_idx][1],
                ),
                stack_size=0xF00,
                while_true=False,
            )
        )

    ln2_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=static_ln2_weights,
        name="static_ln2_weights",
    )
    ln2_sum_buffer = Buffer(type=sum_l1_ty, name="ln2_sum_buffer")
    ln2_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln2_sumsq_buffer")
    ln2_worker = Worker(
        core_fn_add_norm2,
        fn_args=[
            ffnDownOut.cons(),
            ffnRIn.cons(),
            ln2_sum_buffer,
            ln2_sumsq_buffer,
            ln2_weight_buffer,
            outLN2.prod(),
            ln_fused_add_layer_norm_kernel,
            ln_calc_sum_sumsq_kernel,
            ln_zero_f32_kernel,
            mem_copy_o_proj,
            ffn_stage_only,
            addnorm2_debug_mode,
        ],
        placement=Tile(col=ln2_tile[0], row=ln2_tile[1]),
        stack_size=0xF00,
        while_true=False,
    )

    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    # NOTE: It's important that the tiling of Q/K/V are such that the subsequent tiles
    # across the heads, not the sequence length (i.e. how it's done in the MHA operator),
    # because the cores execute on each head. However, we have to keep in mind that
    # K/v need to have the full sequence length passed for each head.
    q_tiles_base = TensorTiler2D.group_tiler(
        (seq_len, embed_sz),
        (seq_tile, d),
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

    def retarget_tas(base_tas, tensor_shape, offset_delta=0):
        return TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    tensor_shape,
                    offset=tap.offset + offset_delta,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in base_tas
            ]
        )

    qkv_tensor_shape = (3 * seq_len, embed_sz)
    Q_tiles = retarget_tas(q_tiles_base, qkv_tensor_shape, offset_delta=0)
    K_tiles = retarget_tas(
        k_tiles_base, qkv_tensor_shape, offset_delta=seq_len * embed_sz
    )
    V_tiles = retarget_tas(
        v_tiles_base, qkv_tensor_shape, offset_delta=2 * seq_len * embed_sz
    )

    # NOTE: Dividing by num_o_col_groups to get the correct number of tiles expected
    # in the runtime seequence. Also not including proj_acc_depth in the tile col
    # dim because the WO buffers operate on tiles with size emb_tile. If we
    # use a tile col dim of emb_tile * proj_acc_depth, then the (d, emb_tile)
    # used for WO will be wrong as the data is written contiguously based on the
    # access pattern, i.e. written contiguously as rows of size emb_tile * proj_acc_depth,
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
    or_tensor_shape = (2 * seq_len, embed_sz)
    O_tiles = retarget_tas(o_tiles_base, or_tensor_shape, offset_delta=0)
    R_tiles = retarget_tas(
        r_tiles_base, or_tensor_shape, offset_delta=seq_len * embed_sz
    )
    # LN1 mul+add consumes residual in FFN-group-major order:
    # [group][proj-acc-col]. Expand leading repeat dimension (stride=0) so
    # each residual col tile is replayed once per FFN column group.
    for tile in R_tiles:
        if len(tile._sizes) < 1:
            raise ValueError(f"Unexpected R tile rank for replay: {tile._sizes}")
        tile._sizes[0] = profile_replay_groups
        tile._strides[0] = 0

    # FFN weight taps (per branch):
    # - Up projection: [branch_col_group, k_chunk, m_tile, n_tile]
    # - Down projection: [branch_col_group, out_chunk, m_tile, n_tile]
    # The outer dimensions enumerate objects consumed by acquire(1) in worker loops.
    B_Up_tiles = []
    B_Down_tiles = []
    for branch_idx in range(effective_ffn_branches):
        group_count = ffn_col_group_counts[branch_idx]
        group_offset = ffn_col_group_offsets[branch_idx]
        B_Up_tiles.append(
            TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        (embed_sz, ffn_intermediate_size),
                        offset=(
                            col_group
                            * (proj_acc_depth * emb_tile * ffn_intermediate_size)
                            + group_offset * emb_tile
                        ),
                        sizes=[group_count, proj_acc_depth, emb_tile, emb_tile],
                        strides=[
                            emb_tile,
                            emb_tile * ffn_intermediate_size,
                            ffn_intermediate_size,
                            1,
                        ],
                    )
                    for col_group in range(num_o_col_groups)
                ]
            )
        )
        B_Down_tiles.append(
            TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        (ffn_intermediate_size, embed_sz),
                        offset=(
                            col_group * (proj_acc_depth * emb_tile)
                            + group_offset * emb_tile * embed_sz
                        ),
                        sizes=[group_count, proj_acc_depth, emb_tile, emb_tile],
                        strides=[
                            emb_tile * embed_sz,
                            emb_tile,
                            embed_sz,
                            1,
                        ],
                    )
                    for col_group in range(num_o_col_groups)
                ]
            )
        )

    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            logging.info(f"{name} tile {idx}:")
            logging.info(f"  Offset: {tap.offset}")
            logging.info(f"  Sizes: {tap.sizes}")
            logging.info(f"  Strides: {tap.strides}")

    def enumerate_outer_object_offsets(tap: TensorAccessPattern, inner_rank: int = 2):
        """
        Enumerate object start offsets represented by a TAP.
        For O/R/WO taps, the object rank is 2 (tile rows x tile cols),
        so the leading dimensions enumerate distinct objects.
        """

        sizes = list(tap.sizes)
        strides = list(tap.strides)
        if len(sizes) <= inner_rank:
            return [tap.offset]

        outer_sizes = sizes[:-inner_rank]
        outer_strides = strides[:-inner_rank]
        offsets = []

        def walk(dim: int, running_offset: int):
            if dim == len(outer_sizes):
                offsets.append(tap.offset + running_offset)
                return
            for idx in range(outer_sizes[dim]):
                walk(dim + 1, running_offset + idx * outer_strides[dim])

        walk(0, 0)
        return offsets

    def transfer_count_for_fifo_obj(
        tap: TensorAccessPattern, obj_shape: tuple[int, ...]
    ) -> int:
        tap_elems = math.prod(int(s) for s in tap.sizes)
        obj_elems = math.prod(int(d) for d in obj_shape)
        if obj_elems <= 0:
            raise ValueError(f"Invalid FIFO object shape {obj_shape}")
        if tap_elems % obj_elems != 0:
            raise ValueError(
                "TAP element count is not a whole multiple of FIFO object size: "
                f"tap_sizes={list(tap.sizes)} tap_elems={tap_elems} "
                f"obj_shape={obj_shape} obj_elems={obj_elems}"
            )
        return tap_elems // obj_elems

    def assert_all_taps_match_count(
        taps: TensorAccessSequence,
        obj_shape: tuple[int, ...],
        expected_count: int,
        name: str,
    ):
        for tap_idx, tap in enumerate(taps):
            count = transfer_count_for_fifo_obj(tap, obj_shape)
            if count != expected_count:
                raise ValueError(
                    f"{name}[{tap_idx}] transfers {count} objects but worker loops "
                    f"expect {expected_count}. tap_sizes={list(tap.sizes)} "
                    f"tap_strides={list(tap.strides)} obj_shape={obj_shape}"
                )

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
    legalize_tas(R_tiles)
    for branch_idx in range(effective_ffn_branches):
        legalize_tas(B_Up_tiles[branch_idx])
        legalize_tas(B_Down_tiles[branch_idx])

    # Validate host transfer counts against per-iteration worker consumption.
    assert_all_taps_match_count(
        Q_tiles,
        (seq_tile, d * parallel_heads),
        num_qkv_head_block_per_parallel_head,
        "Q",
    )
    assert_all_taps_match_count(
        K_tiles,
        (kv_seq_tile, d * parallel_heads),
        num_kv_seq_blocks,
        "K",
    )
    assert_all_taps_match_count(
        V_tiles,
        (kv_seq_tile, d * parallel_heads),
        num_kv_seq_blocks,
        "V",
    )
    assert_all_taps_match_count(
        WO_tiles,
        (d * parallel_heads, emb_tile),
        proj_acc_depth,
        "W_O",
    )
    assert_all_taps_match_count(
        R_tiles,
        (seq_tile, emb_tile),
        profile_replay_groups * proj_acc_depth,
        "R",
    )
    assert_all_taps_match_count(
        O_tiles,
        (seq_tile, emb_tile),
        proj_acc_depth,
        "O",
    )
    for branch_idx in range(effective_ffn_branches):
        assert_all_taps_match_count(
            B_Up_tiles[branch_idx],
            (emb_tile, emb_tile),
            ffn_col_group_counts[branch_idx] * proj_acc_depth,
            f"B_Up[{branch_idx}]",
        )
        assert_all_taps_match_count(
            B_Down_tiles[branch_idx],
            (emb_tile, emb_tile),
            ffn_col_group_counts[branch_idx] * proj_acc_depth,
            f"B_Down[{branch_idx}]",
        )

    print_tap_seq_info(Q_tiles, "Q")
    print_tap_seq_info(K_tiles, "K")
    print_tap_seq_info(V_tiles, "V")
    print_tap_seq_info(WO_tiles, "W_O")
    print_tap_seq_info(O_tiles, "O")
    print_tap_seq_info(R_tiles, "R")
    for branch_idx in range(effective_ffn_branches):
        print_tap_seq_info(B_Up_tiles[branch_idx], f"B_Up[{branch_idx}]")
        print_tap_seq_info(B_Down_tiles[branch_idx], f"B_Down[{branch_idx}]")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(W_O_ty, QKV_ty, OR_ty, B_Up_ty, B_Down_ty) as (
        W_O,
        QKV,
        OR,
        B_Up,
        B_Down,
    ):
        # Full pipeline is sensitive to host DMA ordering of tail-stage IO.
        # Keep strict ordering by default, with an opt-in override for profiling.
        serialize_tail_io = True
        serialize_tail_io_env = os.getenv("ENCODER_SERIALIZE_TAIL_IO")
        if serialize_tail_io_env is not None:
            serialize_tail_io = serialize_tail_io_env.strip().lower() not in (
                "0",
                "false",
                "off",
                "no",
            )
        serialize_q_prestage = serialize_tail_io
        serialize_q_prestage_env = os.getenv("ENCODER_SERIALIZE_Q_PRESTAGE")
        if serialize_q_prestage_env is not None:
            serialize_q_prestage = serialize_q_prestage_env.strip().lower() not in (
                "0",
                "false",
                "off",
                "no",
            )
        if not serialize_tail_io:
            serialize_q_prestage = False
        tail_wait_mode = os.getenv("ENCODER_TAIL_WAIT_MODE", "strict").strip().lower()
        wait_residual_fill = serialize_tail_io
        wait_ffn_weight_fill = serialize_tail_io
        wait_output_drain = serialize_tail_io
        if serialize_tail_io:
            if tail_wait_mode in ("", "strict"):
                pass
            elif tail_wait_mode in ("relax_ffn_weights", "relax_weights"):
                wait_ffn_weight_fill = False
            elif tail_wait_mode in (
                "relax_ffn_weights_residual",
                "relax_weights_residual",
                "relax_weights_r",
            ):
                wait_ffn_weight_fill = False
                wait_residual_fill = False
            elif tail_wait_mode in ("relax_all_tail", "relax_tail", "relax_all"):
                wait_ffn_weight_fill = False
                wait_residual_fill = False
                wait_output_drain = False
            else:
                raise ValueError(
                    "ENCODER_TAIL_WAIT_MODE must be one of "
                    "{strict, relax_ffn_weights, relax_ffn_weights_residual, "
                    "relax_all_tail} "
                    f"(got '{tail_wait_mode}')"
                )

        for i in range(parallel_heads):
            rt.start(matmul_workers[i])
            rt.start(softmax_workers[i])
            rt.start(matmul_pv_workers[i])
            rt.start(o_proj_workers[i])
        rt.start(ln1_norm_worker)
        rt.start(ln1_muladd_worker)
        if ln1_route_worker is not None:
            rt.start(ln1_route_worker)
        for branch_idx in range(effective_ffn_branches):
            rt.start(ffn_up_workers[branch_idx])
            rt.start(ffn_down_workers[branch_idx])
        rt.start(ln2_worker)

        for q_block_idx in range(num_q_seq_blocks):
            for col_group in range(num_o_col_groups):
                # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
                tg = rt.task_group()
                if serialize_q_prestage:
                    # Stage Q tiles first so head-0 compute cannot race ahead of Q load.
                    tg_q = rt.task_group()
                    rt.fill(
                        inQ.prod(),
                        QKV,
                        tap=Q_tiles[q_block_idx],
                        placement=Tile(col=0, row=0),
                        task_group=tg_q,
                    )
                    rt.finish_task_group(tg_q)
                else:
                    rt.fill(
                        inQ.prod(),
                        QKV,
                        tap=Q_tiles[q_block_idx],
                        placement=Tile(col=0, row=0),
                        task_group=tg,
                    )
                logging.debug(
                    f"Scheduling fills for q block {q_block_idx}, col group {col_group} for QKV, W_O, and OR"
                )
                logging.debug(f"  Q tap: {Q_tiles[q_block_idx]}")
                tap_idx = q_block_idx * num_o_col_groups + col_group
                o_offsets = enumerate_outer_object_offsets(
                    O_tiles[tap_idx], inner_rank=2
                )
                r_offsets = enumerate_outer_object_offsets(
                    R_tiles[tap_idx], inner_rank=2
                )
                logging.debug(
                    "  O/R tap mapping for q_block=%d col_group=%d tap_idx=%d: "
                    "O_outer_offsets=%s R_outer_offsets=%s",
                    q_block_idx,
                    col_group,
                    tap_idx,
                    o_offsets,
                    r_offsets,
                )
                for pass_idx in range(2):
                    for acc_idx in range(proj_acc_depth):
                        logical_col_idx = col_group * proj_acc_depth + acc_idx
                        o_off = o_offsets[acc_idx] if acc_idx < len(o_offsets) else None
                        r_off = r_offsets[acc_idx] if acc_idx < len(r_offsets) else None
                        logging.debug(
                            "    Expected O-proj emit: pass=%d acc_idx=%d logical_col=%d "
                            "-> O_offset=%s R_offset=%s",
                            pass_idx + 1,
                            acc_idx,
                            logical_col_idx,
                            o_off,
                            r_off,
                        )
                for head_idx in range(heads // parallel_heads):
                    tg_head = rt.task_group()
                    rt.fill(
                        inK.prod(),
                        QKV,
                        tap=K_tiles[head_idx],
                        placement=Tile(col=1, row=0),
                        task_group=tg_head,
                        wait=True,
                    )
                    rt.fill(
                        inV.prod(),
                        QKV,
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
                    wo_offsets = enumerate_outer_object_offsets(
                        WO_tiles[head_idx * num_o_col_groups + col_group], inner_rank=2
                    )
                    logging.debug(
                        "    W_O outer offsets (head_idx=%d, q_block=%d, col_group=%d): %s",
                        head_idx,
                        q_block_idx,
                        col_group,
                        wo_offsets,
                    )

                rt.fill(
                    inR.prod(),
                    OR,
                    tap=R_tiles[q_block_idx * (num_o_col_groups) + col_group],
                    placement=Tile(col=ln_mem_tile_col, row=0),
                    task_group=tg,
                    wait=wait_residual_fill,
                )
                for branch_idx in range(effective_ffn_branches):
                    rt.fill(
                        inBUp[branch_idx].prod(),
                        B_Up,
                        tap=B_Up_tiles[branch_idx][col_group],
                        placement=Tile(col=ffn_bup_mem_tile_cols[branch_idx], row=0),
                        task_group=tg,
                        wait=wait_ffn_weight_fill,
                    )
                    rt.fill(
                        inBDown[branch_idx].prod(),
                        B_Down,
                        tap=B_Down_tiles[branch_idx][col_group],
                        placement=Tile(
                            col=ffn_bdown_mem_tile_cols[branch_idx],
                            row=0,
                        ),
                        task_group=tg,
                        wait=wait_ffn_weight_fill,
                    )

                rt.drain(
                    memLN2.cons(),
                    OR,
                    tap=O_tiles[q_block_idx * (num_o_col_groups) + col_group],
                    placement=Tile(col=ln_mem_tile_col, row=0),
                    task_group=tg,
                    wait=wait_output_drain,
                )
                logging.debug(
                    f"  O tap: {O_tiles[q_block_idx * (num_o_col_groups) + col_group]}"
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
