#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from operators.encoder_pipeline.design import fused_mha as _fused_mha_base


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


def plan_branch_stage_configuration(
    *,
    branch_stage_cols,
    branch_bup_cols,
    branch_down_b_cols,
    effective_ffn_branches,
    parallel_heads,
    proj_acc_depth,
    **_unused,
):
    del effective_ffn_branches, parallel_heads, proj_acc_depth
    return branch_stage_cols, branch_bup_cols, branch_down_b_cols, []


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
        and proj_acc_depth < 8
        and effective_ffn_branches >= 6
        and len(cols) >= 4
        and cols[3] == 7
    ):
        # Col7 already carries residual/LN2 traffic. Move one FFN down-acc
        # stream to col6 to reduce BD pressure when B_Down split streams are
        # also mapped onto col7.
        cols[3] = 6
    return cols


def build_runtime_state(**_unused):
    return {
        "ln1_ddr_stage_enabled_branches": [],
        "ln1_ddr_stage_enabled": False,
        "ln1_ddr_mixed_mode": False,
    }


def adjust_wait_ffn_weight_fill(
    *,
    wait_ffn_weight_fill,
    use_ln1_broadcast,
    effective_ffn_branches,
    **_unused,
):
    if use_ln1_broadcast and effective_ffn_branches > 1:
        return True
    return wait_ffn_weight_fill


def should_prefill_ffn_weights(**_unused):
    return True


def use_bup_broadcast_priming(*, use_ln1_broadcast, effective_ffn_branches, **_unused):
    return False


def schedule_runtime_tap(
    *,
    rt,
    schedule_final_output_for_tap,
    tap_idx,
    tg,
    tg_tail_fill,
    decouple_tail_fill,
    pending_ln1_refill_tg,
    pending_output_tap_idx,
    **_unused,
):
    rt.finish_task_group(tg)
    schedule_final_output_for_tap(tap_idx)
    if decouple_tail_fill:
        rt.finish_task_group(tg_tail_fill)
    return pending_ln1_refill_tg, pending_output_tap_idx


def finalize_runtime(
    *,
    pending_ln1_refill_tg,
    pending_output_tap_idx,
    **_unused,
):
    return pending_ln1_refill_tg, pending_output_tap_idx


def _hook_namespace():
    return SimpleNamespace(
        ln1_dram_stage_rows=ln1_dram_stage_rows,
        choose_ln1_replay_mem_tile_col=choose_ln1_replay_mem_tile_col,
        plan_branch_stage_configuration=plan_branch_stage_configuration,
        build_ln1_to_ffn_up_path=build_ln1_to_ffn_up_path,
        adjust_ffn_down_acc_mem_tile_cols=adjust_ffn_down_acc_mem_tile_cols,
        build_runtime_state=build_runtime_state,
        adjust_wait_ffn_weight_fill=adjust_wait_ffn_weight_fill,
        should_prefill_ffn_weights=should_prefill_ffn_weights,
        use_bup_broadcast_priming=use_bup_broadcast_priming,
        schedule_runtime_tap=schedule_runtime_tap,
        finalize_runtime=finalize_runtime,
    )


def fused_mha(*args, **kwargs):
    kwargs["ln1_stage_mode"] = "memtile"
    kwargs["_ln1_mode_hooks"] = _hook_namespace()
    return _fused_mha_base(*args, **kwargs)
