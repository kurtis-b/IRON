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
    if parallel_heads <= 2 and proj_acc_depth >= 8 and effective_ffn_branches > 1:
        return 1
    if parallel_heads >= 6 and proj_acc_depth >= 6:
        return 5
    if parallel_heads >= 4 and proj_acc_depth >= 8 and effective_ffn_branches > 1:
        # For 4-way FFN tails, col3 already carries W_O split fanout and can
        # exceed memtile output-channel limits when LN1 replay is colocated.
        if effective_ffn_branches >= 4:
            return 3
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
    ln1_replay_mem_tile_col,
    ffn_col_group_counts,
    bup_split_enabled,
    bdown_split_enabled,
    branch_split_chunks,
    _override_int_env,
    _allocate_ffn_weight_mem_tile_cols_by_streams,
    logging_module,
):
    del ln1_replay_mem_tile_col, _override_int_env
    branch_stage_cols = list(branch_stage_cols)
    if parallel_heads >= 6 and len(branch_stage_cols) >= 2:
        branch_stage_cols = [
            branch_stage_cols[1],
            branch_stage_cols[0],
            *branch_stage_cols[2:],
        ]
    if 5 in branch_stage_cols:
        branch_stage_cols = [c for c in branch_stage_cols if c != 5] + [5]

    if bup_split_enabled or bdown_split_enabled:
        split_chunk_count = len(branch_split_chunks)
        up_stream_count = (
            split_chunk_count if bup_split_enabled else effective_ffn_branches
        )
        down_stream_count = (
            split_chunk_count if bdown_split_enabled else effective_ffn_branches
        )
    else:
        up_stream_count = effective_ffn_branches
        down_stream_count = effective_ffn_branches

    # DDR mode uses one LN1 stage drain and one LN1 stage refill stream.
    # Refill is then broadcast on-chip to all FFN-up branches.
    ln1_ddr_stage_source_branch_idx = 0
    ln1_ddr_staged_branch_indices = [ln1_ddr_stage_source_branch_idx]
    stage_col = branch_stage_cols[ln1_ddr_stage_source_branch_idx]
    candidate_reserved_cols = {stage_col: 1}
    try:
        alloc_bup_cols, alloc_down_cols = _allocate_ffn_weight_mem_tile_cols_by_streams(
            up_stream_count,
            down_stream_count,
            reserved_output_load_by_col=candidate_reserved_cols,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to allocate shim output channels for LN1 DDR single-stream staging "
            f"(branches={effective_ffn_branches}, reserved_cols={candidate_reserved_cols})"
        ) from exc
    logging_module.info(
        "LN1 DDR single-stream staging: source_branch=%d stage_col=%d "
        "(broadcast to %d FFN branches, group_counts=%s)",
        ln1_ddr_stage_source_branch_idx,
        stage_col,
        effective_ffn_branches,
        ffn_col_group_counts,
    )

    if bup_split_enabled or bdown_split_enabled:
        if alloc_bup_cols is None or alloc_down_cols is None:
            raise ValueError("Missing precomputed FFN weight column allocation")
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
        if alloc_bup_cols is None or alloc_down_cols is None:
            raise ValueError("Missing precomputed FFN weight column allocation")
        branch_bup_cols, branch_down_b_cols = alloc_bup_cols, alloc_down_cols

    # In low-head/high-acc 4-branch DDR tails, duplicating B_Up staging on one
    # memtile can push that tile over the 48-BD limit once LN replay and FFN
    # down-acc staging are included. Spread duplicate B_Up streams onto spare
    # shim-output columns without exceeding the per-column 2-channel budget.
    if parallel_heads <= 4 and proj_acc_depth >= 8 and effective_ffn_branches >= 4:
        output_load = {col: 0 for col in range(8)}
        for col in (0, 1, 2, 3, 7):
            output_load[col] += 1
        stage_col = branch_stage_cols[ln1_ddr_stage_source_branch_idx]
        output_load[stage_col] += 1
        for col in branch_down_b_cols:
            output_load[col] += 1

        remapped_bup_cols = []
        for col in branch_bup_cols:
            mapped_col = col
            duplicate_bup_col = mapped_col in remapped_bup_cols
            if duplicate_bup_col:
                for candidate_col in (0, 7, 6, 5, 4, 3, 2, 1):
                    if candidate_col in remapped_bup_cols:
                        continue
                    if output_load[candidate_col] >= 2:
                        continue
                    mapped_col = candidate_col
                    break
            remapped_bup_cols.append(mapped_col)
            output_load[mapped_col] += 1
        branch_bup_cols = remapped_bup_cols

    return (
        branch_stage_cols,
        branch_bup_cols,
        branch_down_b_cols,
        ln1_ddr_staged_branch_indices,
    )


def build_ln1_to_ffn_up_path(
    *,
    o_ty,
    ffn_up_broadcast_depth,
    ffn_up_consumer_depth,
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
    del tile_ctor, _override_int_env, _override_bool_env, parallel_heads, proj_acc_depth
    del o_proj_acc_group_size, ffn_a_stage_mem_tile_cols
    mem_out_ln_cons = []
    ln1_out_stage_to_ddr = {}
    ln1_in_from_ddr = {}
    ln1_ddr_staged_branch_set = set(ln1_ddr_staged_branch_indices)

    ln1_broadcast = object_fifo_ctor(
        o_ty,
        name="outLNBroadcast",
        depth=ffn_up_broadcast_depth,
    )
    if sorted(ln1_ddr_staged_branch_set) != [0]:
        raise ValueError(
            "DDR mode requires one LN1 stage source branch for DDR staging "
            f"(staged={sorted(ln1_ddr_staged_branch_set)}, branches={effective_ffn_branches})"
        )
    ln1_stage_source = (
        ln1_broadcast.cons(depth=ffn_up_broadcast_depth)
        if ln1_ddr_staged_branch_set
        else None
    )
    # Single host refill stream, then broadcast to all FFN-up branches.
    ln1_refill_broadcast = object_fifo_ctor(
        o_ty,
        name="inLNFromDDR",
        depth=ffn_up_broadcast_depth,
    )
    ln1_out_stage_to_ddr[0] = ln1_stage_source
    ln1_in_from_ddr[0] = ln1_refill_broadcast
    for _ in range(effective_ffn_branches):
        mem_out_ln_cons.append(ln1_refill_broadcast.cons(depth=ffn_up_consumer_depth))

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
    proj_acc_depth,
    parallel_heads,
    ln1_replay_mem_tile_col,
):
    del ln1_replay_mem_tile_col
    # For low-head/high-acc DDR topologies with two FFN branches, keeping the
    # second FFN-down accumulation stream on col4 can collide with O-proj
    # accumulation BD allocation on memtile4. Move that second stream to a
    # different memtile column while preserving branch-0 placement.
    if (
        effective_ffn_branches == 2
        and parallel_heads <= 4
        and proj_acc_depth >= 8
        and len(ffn_down_acc_mem_tile_cols) >= 2
    ):
        # Keep col7 as a last resort because it already carries high-depth
        # residual/LN2 traffic in these topologies.
        fallback_cols = [2, 5, 6, 3, 1, 0, 4, 7]
        branch0_col = ffn_down_acc_mem_tile_cols[0]
        chosen_col = None
        for col in fallback_cols:
            if col == branch0_col:
                continue
            chosen_col = col
            break
        if chosen_col is not None:
            ffn_down_acc_mem_tile_cols[1] = chosen_col
    if effective_ffn_branches >= 4 and parallel_heads <= 4 and proj_acc_depth >= 8:
        # Col3 already carries W_O fanout in these topologies. Keep FFN down
        # accumulation staging off col3 to stay within memtile BD allocation.
        for idx, col in enumerate(ffn_down_acc_mem_tile_cols):
            if col != 3:
                continue
            for candidate_col in (6, 5, 4, 7, 2, 1, 0):
                if candidate_col not in ffn_down_acc_mem_tile_cols:
                    ffn_down_acc_mem_tile_cols[idx] = candidate_col
                    break
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
    if ln1_ddr_stage_enabled_branches != [0]:
        raise ValueError(
            "DDR mode expects a single LN1 stage source branch "
            f"(staged={ln1_ddr_stage_enabled_branches}, branches={effective_ffn_branches})"
        )
    if 0 not in ln1InFromDDR:
        raise ValueError(
            "LN1 DDR refill broadcast FIFO missing for staged branch 0 "
            f"(available_refills={sorted(ln1InFromDDR.keys())})"
        )
    return {
        "ln1_ddr_stage_enabled_branches": ln1_ddr_stage_enabled_branches,
        "ln1_ddr_stage_enabled": ln1_ddr_stage_enabled,
    }


def adjust_wait_ffn_weight_fill(
    *,
    wait_ffn_weight_fill,
    runtime_state,
    **_unused,
):
    del runtime_state
    return wait_ffn_weight_fill


def should_prefill_ffn_weights(
    *,
    runtime_state,
    **_unused,
):
    del runtime_state
    return False


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
        # Drain the previous tap output before waiting on the prior LN1 refill
        # task-group. Otherwise, FFN/LN2 backpressure can block those fills,
        # creating a wait cycle (refill waits on downstream progress while the
        # downstream drain is deferred until after refill completion).
        if pending_output_tap_idx is not None:
            schedule_final_output_for_tap(pending_output_tap_idx)
            pending_output_tap_idx = None
        rt.finish_task_group(pending_ln1_refill_tg)
        pending_ln1_refill_tg = None

    ln1_stage_region_offset = 2 * seq_len * embed_sz
    ln1_stage_base_offset = (
        ln1_stage_region_offset + col_group * proj_acc_depth * emb_tile
    )
    del runtime_state
    stage_source_branch = 0
    tg_ln1_drain = rt.task_group()
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
            "LN1 DDR staging transfer count mismatch: "
            f"staged={staged_branch_tokens} expected={expected_branch_tokens}"
        )
    stage_src = ln1OutStageToDDR[stage_source_branch]
    if hasattr(stage_src, "cons"):
        stage_src = stage_src.cons()
    rt.drain(
        stage_src,
        OR,
        tap=branch_stage_tap,
        placement=tile_ctor(col=ffn_a_stage_mem_tile_cols[stage_source_branch], row=0),
        task_group=tg_ln1_drain,
        wait=True,
    )
    rt.finish_task_group(tg_ln1_drain)

    tg_ln1_refill_and_weights = rt.task_group()
    rt.fill(
        ln1InFromDDR[stage_source_branch].prod(),
        OR,
        tap=branch_stage_tap,
        placement=tile_ctor(col=ffn_a_stage_mem_tile_cols[stage_source_branch], row=0),
        task_group=tg_ln1_refill_and_weights,
        wait=True,
    )
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
    if pending_output_tap_idx is not None:
        schedule_final_output_for_tap(pending_output_tap_idx)
        pending_output_tap_idx = None
    if pending_ln1_refill_tg is not None:
        rt.finish_task_group(pending_ln1_refill_tg)
        pending_ln1_refill_tg = None
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
