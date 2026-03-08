#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from operators.encoder_pipeline.design import fused_mha as _fused_mha_base


def ln1_dram_stage_rows(*, profile_replay_groups, seq_tile):
    return profile_replay_groups * seq_tile


def choose_ln1_replay_mem_tile_col(
    *,
    parallel_heads,
    proj_acc_depth,
    effective_ffn_branches,
    o_proj_acc_group_size=1,
):
    del o_proj_acc_group_size
    if parallel_heads >= 6 and proj_acc_depth >= 6:
        return 5
    if parallel_heads >= 4 and proj_acc_depth >= 8 and effective_ffn_branches > 1:
        return 3
    return 5


def plan_branch_stage_configuration(
    *,
    branch_stage_cols,
    branch_bup_cols,
    branch_down_b_cols,
    effective_ffn_branches,
    parallel_heads,
    ffn_col_group_counts,
    bup_split_enabled,
    bdown_split_enabled,
    branch_split_chunks,
    _override_int_env,
    _allocate_ffn_weight_mem_tile_cols_by_streams,
    logging_module,
):
    branch_stage_cols = list(branch_stage_cols)
    if parallel_heads >= 6 and len(branch_stage_cols) >= 2:
        branch_stage_cols = [
            branch_stage_cols[1],
            branch_stage_cols[0],
            *branch_stage_cols[2:],
        ]
    if 5 in branch_stage_cols:
        branch_stage_cols = [c for c in branch_stage_cols if c != 5] + [5]

    ln1_ddr_stage_branch_limit = _override_int_env(
        "ENCODER_LN1_DDR_STAGE_BRANCH_LIMIT",
        effective_ffn_branches,
    )
    invalid_stage_limit_checks = [
        ln1_ddr_stage_branch_limit <= 0,
        ln1_ddr_stage_branch_limit > effective_ffn_branches,
    ]
    if any(invalid_stage_limit_checks):
        raise ValueError(
            "ENCODER_LN1_DDR_STAGE_BRANCH_LIMIT must be in [1, effective_ffn_branches] "
            f"(got {ln1_ddr_stage_branch_limit}, branches={effective_ffn_branches})"
        )
    branch_priority = sorted(
        range(effective_ffn_branches),
        key=lambda idx: (-ffn_col_group_counts[idx], idx),
    )
    ln1_ddr_staged_branch_indices = sorted(branch_priority[:ln1_ddr_stage_branch_limit])
    logging_module.info(
        "LN1 DDR staging branches: %s / %d (group_counts=%s)",
        ln1_ddr_staged_branch_indices,
        effective_ffn_branches,
        ffn_col_group_counts,
    )
    reserved_ln1_refill_cols = {}
    for branch_idx in ln1_ddr_staged_branch_indices:
        stage_col = branch_stage_cols[branch_idx]
        reserved_ln1_refill_cols[stage_col] = (
            reserved_ln1_refill_cols.get(stage_col, 0) + 1
        )
    if bup_split_enabled or bdown_split_enabled:
        split_chunk_count = len(branch_split_chunks)
        up_stream_count = (
            split_chunk_count if bup_split_enabled else effective_ffn_branches
        )
        down_stream_count = (
            split_chunk_count if bdown_split_enabled else effective_ffn_branches
        )
        alloc_bup_cols, alloc_down_cols = _allocate_ffn_weight_mem_tile_cols_by_streams(
            up_stream_count,
            down_stream_count,
            reserved_output_load_by_col=reserved_ln1_refill_cols,
        )
        branch_bup_cols = [0] * effective_ffn_branches
        branch_down_b_cols = [0] * effective_ffn_branches
        if bup_split_enabled:
            for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                for branch_idx in chunk_branches:
                    branch_bup_cols[branch_idx] = alloc_bup_cols[chunk_idx]
        else:
            for branch_idx in range(effective_ffn_branches):
                branch_bup_cols[branch_idx] = alloc_bup_cols[branch_idx]
        if bdown_split_enabled:
            for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                for branch_idx in chunk_branches:
                    branch_down_b_cols[branch_idx] = alloc_down_cols[chunk_idx]
        else:
            for branch_idx in range(effective_ffn_branches):
                branch_down_b_cols[branch_idx] = alloc_down_cols[branch_idx]
    else:
        branch_bup_cols, branch_down_b_cols = (
            _allocate_ffn_weight_mem_tile_cols_by_streams(
                effective_ffn_branches,
                effective_ffn_branches,
                reserved_output_load_by_col=reserved_ln1_refill_cols,
            )
        )

    return (
        branch_stage_cols,
        branch_bup_cols,
        branch_down_b_cols,
        ln1_ddr_staged_branch_indices,
    )


def build_ln1_to_ffn_up_path(
    *,
    o_ty,
    ffn_up_input_depth,
    effective_ffn_branches,
    ffn_a_stage_mem_tile_cols,
    ln1_ddr_staged_branch_indices,
    _override_int_env,
    _override_bool_env,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    object_fifo_ctor,
    tile_ctor,
):
    mem_out_ln_cons = []
    ln1_out_stage_to_ddr = {}
    ln1_in_from_ddr = {}
    ln1_ddr_staged_branch_set = set(ln1_ddr_staged_branch_indices)

    ln1_ddr_stage_fifo_depth = _override_int_env(
        "ENCODER_LN1_DDR_STAGE_FIFO_DEPTH",
        1,
    )
    ln1_ddr_direct_shim_io = _override_bool_env(
        "ENCODER_LN1_DDR_DIRECT_SHIM_IO",
        parallel_heads >= 6 and proj_acc_depth >= 6 and o_proj_acc_group_size > 1,
    )
    ln1_broadcast = object_fifo_ctor(
        o_ty,
        name="outLNBroadcast",
        depth=ffn_up_input_depth,
    )

    for branch_idx in range(effective_ffn_branches):
        stage_col = ffn_a_stage_mem_tile_cols[branch_idx]
        if branch_idx in ln1_ddr_staged_branch_set:
            if ln1_ddr_direct_shim_io:
                ln1_out_stage_to_ddr[branch_idx] = ln1_broadcast
            else:
                ln_stage_source = ln1_broadcast.cons(depth=ffn_up_input_depth)
                ln1_out_stage_to_ddr[branch_idx] = ln_stage_source.forward(
                    obj_type=o_ty,
                    name=(
                        "memOutLNStageToDDR"
                        if branch_idx == 0
                        else f"memOutLNStageToDDR{branch_idx}"
                    ),
                    depth=ln1_ddr_stage_fifo_depth,
                    placement=tile_ctor(col=stage_col, row=1),
                )
            ln1_in_from_ddr[branch_idx] = object_fifo_ctor(
                o_ty,
                name=("inLNFromDDR" if branch_idx == 0 else f"inLNFromDDR{branch_idx}"),
                depth=ffn_up_input_depth,
            )
            if ln1_ddr_direct_shim_io:
                mem_out_ln_cons.append(ln1_in_from_ddr[branch_idx].cons(depth=1))
            else:
                mem_out_ln_cons.append(
                    ln1_in_from_ddr[branch_idx]
                    .cons(depth=ffn_up_input_depth)
                    .forward(
                        obj_type=o_ty,
                        name=(
                            "memOutLN"
                            if branch_idx == 0
                            else f"memOutLNFfn{branch_idx}"
                        ),
                        depth=ln1_ddr_stage_fifo_depth,
                        placement=tile_ctor(col=stage_col, row=1),
                    )
                    .cons(depth=1)
                )
        else:
            # Non-staged branch in DDR mode: keep direct memtile hop.
            mem_out_ln_cons.append(
                ln1_broadcast.cons(depth=ffn_up_input_depth)
                .forward(
                    obj_type=o_ty,
                    name="memOutLN" if branch_idx == 0 else f"memOutLNFfn{branch_idx}",
                    depth=ffn_up_input_depth,
                    placement=tile_ctor(col=stage_col, row=1),
                )
                .cons(depth=1)
            )

    return {
        "ln1Broadcast": ln1_broadcast,
        "memOutLNCons": mem_out_ln_cons,
        "ln1OutStageToDDR": ln1_out_stage_to_ddr,
        "ln1InFromDDR": ln1_in_from_ddr,
    }


def adjust_ffn_down_acc_mem_tile_cols(
    *,
    ffn_down_acc_mem_tile_cols,
    effective_ffn_branches,
):
    if effective_ffn_branches >= 6:
        replacement_col = 1
        if replacement_col in ffn_down_acc_mem_tile_cols[:-1]:
            for candidate_col in (0, 1, 2, 3, 4, 5, 6, 7):
                if candidate_col not in ffn_down_acc_mem_tile_cols[:-1]:
                    replacement_col = candidate_col
                    break
        ffn_down_acc_mem_tile_cols[-1] = replacement_col
    return ffn_down_acc_mem_tile_cols


def build_runtime_state(
    *,
    effective_ffn_branches,
    ln1OutStageToDDR,
    ln1InFromDDR,
    ln1_ddr_staged_branch_indices,
):
    ln1_ddr_stage_enabled_branches = sorted(ln1OutStageToDDR.keys())
    ln1_ddr_stage_enabled = len(ln1_ddr_stage_enabled_branches) > 0
    ln1_ddr_mixed_mode = ln1_ddr_stage_enabled and (
        len(ln1_ddr_stage_enabled_branches) < effective_ffn_branches
    )
    if not ln1_ddr_stage_enabled:
        raise ValueError(
            "LN1 DDR staging is enabled but no staged branch FIFOs were created "
            f"(branches={effective_ffn_branches})"
        )
    missing_refill_branches = [
        branch_idx
        for branch_idx in ln1_ddr_stage_enabled_branches
        if branch_idx not in ln1InFromDDR
    ]
    if missing_refill_branches:
        raise ValueError(
            "LN1 DDR staging branch refill FIFOs missing for branches "
            f"{missing_refill_branches}; staged={ln1_ddr_stage_enabled_branches}, "
            f"available_refills={sorted(ln1InFromDDR.keys())}"
        )
    if sorted(ln1_ddr_staged_branch_indices) != ln1_ddr_stage_enabled_branches:
        raise ValueError(
            "LN1 DDR staged branch mismatch between planning and FIFO creation: "
            f"planned={sorted(ln1_ddr_staged_branch_indices)} "
            f"created={ln1_ddr_stage_enabled_branches}"
        )
    return {
        "ln1_ddr_stage_enabled_branches": ln1_ddr_stage_enabled_branches,
        "ln1_ddr_stage_enabled": ln1_ddr_stage_enabled,
        "ln1_ddr_mixed_mode": ln1_ddr_mixed_mode,
    }


def adjust_wait_ffn_weight_fill(
    *,
    wait_ffn_weight_fill,
    runtime_state,
    **_unused,
):
    if runtime_state["ln1_ddr_mixed_mode"]:
        return False
    return wait_ffn_weight_fill


def should_prefill_ffn_weights(
    *,
    runtime_state,
    **_unused,
):
    return (not runtime_state["ln1_ddr_stage_enabled"]) or runtime_state[
        "ln1_ddr_mixed_mode"
    ]


def use_bup_broadcast_priming(**_unused):
    return False


def schedule_runtime_tap(
    *,
    rt,
    OR,
    or_tensor_shape,
    seq_len,
    seq_tile,
    embed_sz,
    profile_replay_groups,
    proj_acc_depth,
    emb_tile,
    runtime_state,
    ln1OutStageToDDR,
    ln1InFromDDR,
    ffn_a_stage_mem_tile_cols,
    transfer_count_for_fifo_obj,
    tensor_access_pattern_cls,
    schedule_ffn_weight_fills,
    schedule_final_output_for_tap,
    tap_idx,
    col_group,
    tg,
    tg_tail_fill,
    decouple_tail_fill,
    pending_ln1_refill_tg,
    pending_output_tap_idx,
    tile_ctor,
    **_unused,
):
    # Single-buffered LN1 scratch region: complete prior refill first.
    if pending_ln1_refill_tg is not None:
        rt.finish_task_group(pending_ln1_refill_tg)
        pending_ln1_refill_tg = None
        if pending_output_tap_idx is not None:
            schedule_final_output_for_tap(pending_output_tap_idx)
            pending_output_tap_idx = None

    ln1_stage_region_offset = 2 * seq_len * embed_sz
    ln1_stage_base_offset = (
        ln1_stage_region_offset + col_group * proj_acc_depth * emb_tile
    )
    ln1_branch_stage_taps = {}
    tg_ln1_drain = rt.task_group()
    for branch_idx in runtime_state["ln1_ddr_stage_enabled_branches"]:
        stage_col = ffn_a_stage_mem_tile_cols[branch_idx]
        branch_stage_tap = tensor_access_pattern_cls(
            or_tensor_shape,
            offset=ln1_stage_base_offset,
            sizes=[
                profile_replay_groups,
                proj_acc_depth,
                seq_tile,
                emb_tile,
            ],
            strides=[0, emb_tile, embed_sz, 1],
        )
        expected_branch_tokens = profile_replay_groups * proj_acc_depth
        staged_branch_tokens = transfer_count_for_fifo_obj(
            branch_stage_tap,
            (seq_tile, emb_tile),
        )
        if staged_branch_tokens != expected_branch_tokens:
            raise ValueError(
                "LN1 DDR staging transfer count mismatch for branch "
                f"{branch_idx}: staged={staged_branch_tokens} "
                f"expected={expected_branch_tokens}"
            )
        ln1_branch_stage_taps[branch_idx] = branch_stage_tap
        rt.drain(
            ln1OutStageToDDR[branch_idx].cons(),
            OR,
            tap=branch_stage_tap,
            placement=tile_ctor(col=stage_col, row=0),
            task_group=tg_ln1_drain,
            wait=True,
        )
    rt.finish_task_group(tg_ln1_drain)

    tg_ln1_refill_and_weights = rt.task_group()
    for branch_idx in runtime_state["ln1_ddr_stage_enabled_branches"]:
        stage_col = ffn_a_stage_mem_tile_cols[branch_idx]
        rt.fill(
            ln1InFromDDR[branch_idx].prod(),
            OR,
            tap=ln1_branch_stage_taps[branch_idx],
            placement=tile_ctor(col=stage_col, row=0),
            task_group=tg_ln1_refill_and_weights,
            wait=True,
        )
    if not runtime_state["ln1_ddr_mixed_mode"]:
        schedule_ffn_weight_fills(tg_ln1_refill_and_weights, col_group)
    rt.finish_task_group(tg)
    if decouple_tail_fill:
        rt.finish_task_group(tg_tail_fill)
    return tg_ln1_refill_and_weights, tap_idx


def finalize_runtime(
    *,
    rt,
    schedule_final_output_for_tap,
    pending_ln1_refill_tg,
    pending_output_tap_idx,
    **_unused,
):
    if pending_ln1_refill_tg is not None:
        rt.finish_task_group(pending_ln1_refill_tg)
        pending_ln1_refill_tg = None
    if pending_output_tap_idx is not None:
        schedule_final_output_for_tap(pending_output_tap_idx)
        pending_output_tap_idx = None
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
    kwargs["ln1_stage_mode"] = "ddr"
    kwargs["_ln1_mode_hooks"] = _hook_namespace()
    return _fused_mha_base(*args, **kwargs)
