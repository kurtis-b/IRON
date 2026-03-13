#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def ln1_dram_stage_rows(*, profile_replay_groups, seq_tile):
    del profile_replay_groups, seq_tile
    return 0


def choose_ln1_replay_mem_tile_col(
    *,
    parallel_heads,
    proj_acc_depth,
    effective_ffn_branches,
    o_proj_acc_group_size=1,
):
    if parallel_heads <= 2 and proj_acc_depth >= 8 and effective_ffn_branches > 1:
        return 1
    if parallel_heads >= 6 and proj_acc_depth >= 6:
        if effective_ffn_branches <= 2 and o_proj_acc_group_size > 1:
            return 5
        return 4
    if parallel_heads >= 4 and proj_acc_depth >= 8 and effective_ffn_branches > 1:
        if o_proj_acc_group_size > 1 and effective_ffn_branches <= 2:
            return 4
        return 3
    return 5


def build_ln1_to_ffn_up_path(
    *,
    o_ty,
    ffn_up_broadcast_depth,
    ffn_up_consumer_depth,
    effective_ffn_branches,
    object_fifo_ctor,
    **_unused,
):
    ln1_broadcast = object_fifo_ctor(
        o_ty,
        name="outLNBroadcast",
        depth=ffn_up_broadcast_depth,
    )
    return {
        "ln1Broadcast": ln1_broadcast,
        "memOutLNCons": [
            ln1_broadcast.cons(depth=ffn_up_consumer_depth)
            for _ in range(effective_ffn_branches)
        ],
        "ln1OutStageToDDR": {},
        "ln1InFromDDR": {},
    }


def adjust_ffn_down_acc_mem_tile_cols(
    *,
    ffn_down_acc_mem_tile_cols,
    effective_ffn_branches,
    proj_acc_depth,
    parallel_heads,
    ln1_replay_mem_tile_col,
    **_unused,
):
    cols = list(ffn_down_acc_mem_tile_cols)
    if effective_ffn_branches <= 1:
        return cols
    if parallel_heads <= 2 and proj_acc_depth >= 8:
        # Keep high-depth down-acc streams off the LN1 replay memtile when
        # possible, and avoid col5 where low-head topologies often place
        # O-proj accumulation staging.
        preferred_fallback_cols = (2, 4, 3, 1, 0, 7, 6)
        remapped = []
        for idx, col in enumerate(cols):
            new_col = col
            if idx > 0 and col in (ln1_replay_mem_tile_col, 5):
                for candidate_col in preferred_fallback_cols:
                    if candidate_col == ln1_replay_mem_tile_col:
                        continue
                    if candidate_col in remapped:
                        continue
                    new_col = candidate_col
                    break
            remapped.append(new_col)
        return remapped
    if (
        parallel_heads == 4
        and proj_acc_depth >= 6
        and effective_ffn_branches >= 6
        and len(cols) >= 4
        and cols[3] == 7
    ):
        # In the 4pheads/6pffn memtile topology, col7 already carries the
        # AddNorm tail streams. For the 16-pacc case, col2 also carries an
        # extra K fanout plus branch-5 accumulation, so move branch-3 to col1.
        cols[3] = 1 if proj_acc_depth >= 16 else 2
    return cols
