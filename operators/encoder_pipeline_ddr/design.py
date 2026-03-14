# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import math
import logging
from types import SimpleNamespace

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence, TensorAccessPattern
from aie.helpers.dialects.scf import if_, else_
import aie.dialects.index as index
from aie.dialects.aiex import *
from operators.encoder_pipeline_ddr.debug_modes import (
    STAGE_DOWN_PROJ,
    STAGE_LN1,
    STAGE_LN2,
    STAGE_O_PROJ,
    STAGE_PV,
    STAGE_QK,
    STAGE_SOFTMAX,
    STAGE_UP_PROJ,
    is_direct_verify_stage,
    stage_compute_enabled,
)
from operators.encoder_pipeline_ddr.mapping_validation import (
    find_ffn_layout,
    manhattan_distance,
)
from operators.encoder_pipeline_ddr.row_store import MemTileRowStore


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
    profile_stage: int | None = None,
    verify_stage: int | None = None,
    ln1_stage_mode: str | None = None,
    o_proj_acc_group_size: int = 1,
    ffn_group_split: str | None = None,
    runtime_serialize_tail_io: bool | None = None,
    runtime_serialize_q_prestage: bool | None = None,
    runtime_tail_wait_mode: str = "relax_ffn_weights",
):
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
    ln1_replay_row_store_compute_buffer_count = 2
    ln1_replay_row_store_compute_produce_buffer_count = 2
    ln1_replay_row_store_compute_consume_buffer_count = 1
    ln2_replay_row_store_compute_buffer_count = 2
    ln2_replay_row_store_compute_produce_buffer_count = 0
    ln2_replay_row_store_compute_consume_buffer_count = 0
    o_proj_acc_row_store_compute_buffer_count = 1
    o_proj_acc_row_store_compute_produce_buffer_count = 0
    o_proj_acc_row_store_compute_consume_buffer_count = 0
    ffn_down_acc_row_store_compute_buffer_count = 1
    ffn_down_acc_row_store_compute_produce_buffer_count = 0
    ffn_down_acc_row_store_compute_consume_buffer_count = 0

    num_q_seq_blocks = seq_len // seq_tile
    num_kv_seq_blocks = seq_len // kv_seq_tile
    num_qkv_head_block_per_parallel_head = heads // parallel_heads
    if nB_tiles_distributed <= 0:
        raise ValueError(
            f"nB_tiles_distributed must be > 0 (got {nB_tiles_distributed})"
        )
    if o_proj_acc_group_size <= 0:
        raise ValueError(
            f"o_proj_acc_group_size must be > 0 (got {o_proj_acc_group_size})"
        )
    if o_proj_acc_group_size > parallel_heads:
        raise ValueError(
            "o_proj_acc_group_size must be <= parallel_heads "
            f"({o_proj_acc_group_size} > {parallel_heads})"
        )
    if o_proj_acc_group_size not in (1, 2, 4):
        raise ValueError(
            "o_proj_acc_group_size currently supports values in {1, 2, 4} "
            f"(got {o_proj_acc_group_size})"
        )
    if parallel_heads % o_proj_acc_group_size != 0:
        raise ValueError(
            "parallel_heads must be divisible by o_proj_acc_group_size "
            f"({parallel_heads} % {o_proj_acc_group_size} != 0)"
        )
    num_o_proj_acc_groups = parallel_heads // o_proj_acc_group_size
    o_proj_group_idx = [i // o_proj_acc_group_size for i in range(parallel_heads)]
    o_proj_group_pos = [i % o_proj_acc_group_size for i in range(parallel_heads)]
    o_proj_stage_core_indices = [
        i
        for i in range(parallel_heads)
        if o_proj_group_pos[i] == o_proj_acc_group_size - 1
    ]
    o_proj_stage_order_by_core = {
        core_idx: stage_idx
        for stage_idx, core_idx in enumerate(o_proj_stage_core_indices)
    }
    o_proj_stage_core_set = set(o_proj_stage_core_indices)
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
    valid_stage_modes = {
        None,
        STAGE_QK,
        STAGE_SOFTMAX,
        STAGE_PV,
        STAGE_O_PROJ,
        STAGE_LN1,
        STAGE_UP_PROJ,
        STAGE_DOWN_PROJ,
        STAGE_LN2,
    }
    if profile_stage not in valid_stage_modes:
        raise ValueError(
            "profile_stage must be one of "
            f"{sorted(v for v in valid_stage_modes if v is not None)} or None "
            f"(got {profile_stage})"
        )
    if verify_stage not in valid_stage_modes:
        raise ValueError(
            "verify_stage must be one of "
            f"{sorted(v for v in valid_stage_modes if v is not None)} or None "
            f"(got {verify_stage})"
        )
    if profile_stage is not None and verify_stage is not None:
        raise ValueError(
            "profile_stage and verify_stage are mutually exclusive "
            f"(profile={profile_stage}, verify={verify_stage})"
        )
    ffn_col_groups = ffn_intermediate_size // emb_tile
    del ln1_stage_mode
    stage_ln1_to_ddr = True
    from operators.encoder_pipeline_ddr import hooks as ln1_mode_hooks

    ln1_mode_hooks = SimpleNamespace(
        choose_ln2_replay_mem_tile_col=getattr(
            ln1_mode_hooks,
            "choose_ln2_replay_mem_tile_col",
            lambda **kwargs: kwargs["default_col"],
        ),
        plan_branch_stage_configuration=getattr(
            ln1_mode_hooks,
            "plan_branch_stage_configuration",
            lambda *, branch_stage_cols, branch_bup_cols, branch_down_b_cols, **_unused: (
                branch_stage_cols,
                branch_bup_cols,
                branch_down_b_cols,
                [],
            ),
        ),
        build_runtime_state=getattr(
            ln1_mode_hooks,
            "build_runtime_state",
            lambda **_unused: {
                "ln1_ddr_stage_enabled_branches": [],
                "ln1_ddr_stage_enabled": False,
            },
        ),
        adjust_wait_ffn_weight_fill=getattr(
            ln1_mode_hooks,
            "adjust_wait_ffn_weight_fill",
            lambda *, wait_ffn_weight_fill, use_ln1_broadcast, effective_ffn_branches, **_unused: (
                True
                if use_ln1_broadcast and effective_ffn_branches > 1
                else wait_ffn_weight_fill
            ),
        ),
        should_prefill_ffn_weights=getattr(
            ln1_mode_hooks,
            "should_prefill_ffn_weights",
            lambda **_unused: True,
        ),
        use_bup_broadcast_priming=getattr(
            ln1_mode_hooks,
            "use_bup_broadcast_priming",
            lambda **_unused: False,
        ),
        schedule_runtime_tap=getattr(
            ln1_mode_hooks,
            "schedule_runtime_tap",
            lambda *, rt, schedule_final_output_for_tap, tap_idx, tg, tg_tail_fill, decouple_tail_fill, pending_ln1_refill_tg, pending_output_tap_idx, **_unused: (
                schedule_final_output_for_tap(tap_idx),
                rt.finish_task_group(tg),
                rt.finish_task_group(tg_tail_fill) if decouple_tail_fill else None,
                (pending_ln1_refill_tg, pending_output_tap_idx),
            )[
                -1
            ],
        ),
        finalize_runtime=getattr(
            ln1_mode_hooks,
            "finalize_runtime",
            lambda *, pending_ln1_refill_tg, pending_output_tap_idx, **_unused: (
                pending_ln1_refill_tg,
                pending_output_tap_idx,
            ),
        ),
        **{
            name: getattr(ln1_mode_hooks, name)
            for name in (
                "ln1_dram_stage_rows",
                "choose_ln1_replay_mem_tile_col",
                "build_ln1_to_ffn_up_path",
                "adjust_ffn_down_acc_mem_tile_cols",
            )
        },
    )
    qk_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_QK,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    softmax_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_SOFTMAX,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    pv_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_PV,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    oproj_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_O_PROJ,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    ln1_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_LN1,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    up_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_UP_PROJ,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    down_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_DOWN_PROJ,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    ln2_compute_enabled = stage_compute_enabled(
        stage_id=STAGE_LN2,
        profile_stage=profile_stage,
        verify_stage=verify_stage,
    )
    direct_verify_stage = verify_stage if is_direct_verify_stage(verify_stage) else None
    if profile_stage is None and verify_stage is None:
        ffn_stage_only = None
        addnorm1_debug_mode = -1
        addnorm2_debug_mode = -1
    elif (
        profile_stage
        in (
            STAGE_QK,
            STAGE_SOFTMAX,
            STAGE_PV,
            STAGE_O_PROJ,
        )
        or verify_stage == STAGE_O_PROJ
    ):
        ffn_stage_only = 3
        addnorm1_debug_mode = 0
        addnorm2_debug_mode = 1
    elif profile_stage == STAGE_LN1:
        ffn_stage_only = 4
        addnorm1_debug_mode = -1
        addnorm2_debug_mode = 1
    elif profile_stage == STAGE_UP_PROJ:
        ffn_stage_only = 0
        addnorm1_debug_mode = 0
        addnorm2_debug_mode = 1
    elif profile_stage == STAGE_DOWN_PROJ:
        ffn_stage_only = 1
        addnorm1_debug_mode = 0
        addnorm2_debug_mode = 0
    elif verify_stage == STAGE_DOWN_PROJ:
        ffn_stage_only = 1
        addnorm1_debug_mode = -1
        addnorm2_debug_mode = 0
    elif profile_stage == STAGE_LN2:
        ffn_stage_only = 2
        addnorm1_debug_mode = 0
        addnorm2_debug_mode = -1
    elif verify_stage == STAGE_LN2:
        ffn_stage_only = None
        addnorm1_debug_mode = -1
        addnorm2_debug_mode = -1
    else:
        raise ValueError(
            "Unhandled profile/verify stage combination "
            f"(profile={profile_stage}, verify={verify_stage})"
        )
    if not stage_ln1_to_ddr and parallel_heads == 4 and emb_tile <= 96:
        o_proj_acc_row_store_compute_produce_buffer_count = 2
        o_proj_acc_row_store_compute_consume_buffer_count = 1
    # Keep FFN replay topology identical across debug/non-debug configurations.
    # Stage-only modes already bypass compute inside core functions; changing
    # replay-group topology here can cause artificial resource/pruning behavior.
    profile_replay_groups = ffn_col_groups
    if nB_tiles_distributed > ffn_col_groups:
        raise ValueError(
            "nB_tiles_distributed must be <= ffn_col_groups "
            f"({nB_tiles_distributed} > {ffn_col_groups})"
        )
    pruning_disabled_for_this_config = o_proj_acc_group_size > 1
    num_o_col_groups = embed_sz // (emb_tile * proj_acc_depth)
    ln_tiles_per_q_block = num_o_col_groups * proj_acc_depth
    ffn_layout = find_ffn_layout(
        parallel_heads=parallel_heads,
        # The final O-proj stage hands off directly into AddNorm1 in DDR mode.
        # Keep LN1 adjacent to the final O-proj worker so that path stays local.
        mha_output_tile=(parallel_heads - 1, 5),
        require_mha_ln_neighbor=True,
        # For the no-merge topology, LN2 must always receive one reduced FFN stream
        # via a direct N/E/S neighboring down-proj connection.
        max_non_neighbor_down_to_ln2=0,
        restrict_down_reduction_to_nes=True,
        restrict_down_to_ln2_to_nes=True,
    )
    layout_down_edge_deltas = [
        (dst_col - src_col, dst_row - src_row)
        for ((src_col, src_row), (dst_col, dst_row)) in ffn_layout[
            "down_reduction_edges"
        ]
    ]
    layout_has_west_down_edge = any(
        delta_col < 0 for (delta_col, _delta_row) in layout_down_edge_deltas
    )

    if layout_has_west_down_edge:
        raise ValueError(
            "FFN layout produced westward down-reduction edge(s): "
            f"edges={ffn_layout['down_reduction_edges']} deltas={layout_down_edge_deltas}"
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
    selected_branch_indices = list(
        range(max_ffn_branches - requested_ffn_branches, max_ffn_branches)
    )
    if max_ffn_branches > 6:
        raise ValueError(
            "encoder_pipeline supports at most 6 FFN branches "
            f"(layout returned {max_ffn_branches})"
        )
    b_weight_split_requested = o_proj_acc_group_size > 1 and nB_tiles_distributed >= 6
    bup_split_requested = True
    bdown_split_requested = True
    b_weight_split_chunk_size_requested = (
        2 if (o_proj_acc_group_size > 1 and nB_tiles_distributed >= 6) else 3
    )

    def _is_nes_neighbor(
        src_tile: tuple[int, int],
        dst_tile: tuple[int, int],
    ) -> bool:
        delta = (dst_tile[0] - src_tile[0], dst_tile[1] - src_tile[1])
        return delta in ((0, 1), (1, 0), (0, -1))

    def _build_ffn_reduction_topology(
        active_branch_count: int,
        down_tiles: list[tuple[int, int]],
        ln2_compute_tile: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], list[int], str] | None:
        """
        Allowed topologies:
          1) Full chain across all down-proj cores (single LN2 FFN input).
          2) Chain across all-but-one down-proj cores + one bypass core
             (two LN2 FFN inputs: one N/E/S neighbor stream + one DMA stream).
        """
        if active_branch_count <= 0:
            raise ValueError(
                f"active_branch_count must be > 0 (got {active_branch_count})"
            )
        if len(down_tiles) != active_branch_count:
            raise ValueError(
                "down_tiles length must match active_branch_count "
                f"(tiles={len(down_tiles)}, active={active_branch_count})"
            )

        # Prefer full reduction through all down cores when LN2 can consume the
        # reduced stream from a direct N/E/S neighboring down core.
        full_chain_edges = [(i, i + 1) for i in range(active_branch_count - 1)]
        full_chain_outputs = [active_branch_count - 1]
        if _is_nes_neighbor(down_tiles[full_chain_outputs[0]], ln2_compute_tile):
            return full_chain_edges, full_chain_outputs, "all_cores"

        if active_branch_count < 2:
            return None

        # Fallback: all-down-cores-minus-one.
        # Keep exactly two LN2 FFN inputs:
        # - one N/E/S neighboring stream input
        # - one non-neighbor (DMA-routed) input
        candidates = [
            (
                [(i, i + 1) for i in range(active_branch_count - 2)],
                [active_branch_count - 2, active_branch_count - 1],
                "drop_last_from_chain",
            ),
            (
                [(i, i + 1) for i in range(1, active_branch_count - 1)],
                [0, active_branch_count - 1],
                "drop_first_from_chain",
            ),
        ]
        for edges, outputs, mode in candidates:
            neighbor_flags = [
                _is_nes_neighbor(down_tiles[branch_idx], ln2_compute_tile)
                for branch_idx in outputs
            ]
            if sum(1 for is_neighbor in neighbor_flags if is_neighbor) == 1:
                return edges, outputs, mode
        return None

    def _candidate_ffn_branch_group_counts(candidate_branches: int) -> list[int]:
        if candidate_branches <= 0:
            return []
        logical_ffn_parts = min(max(1, nB_tiles_distributed), profile_replay_groups)
        logical_part_base = profile_replay_groups // logical_ffn_parts
        logical_part_rem = profile_replay_groups % logical_ffn_parts
        logical_part_group_counts = [
            logical_part_base + (1 if i < logical_part_rem else 0)
            for i in range(logical_ffn_parts)
        ]
        parts_per_branch_base = logical_ffn_parts // candidate_branches
        parts_per_branch_rem = logical_ffn_parts % candidate_branches
        if parts_per_branch_base <= 0:
            return []
        candidate_group_counts = []
        running_part_offset = 0
        for branch_idx in range(candidate_branches):
            part_count = parts_per_branch_base + (
                1 if branch_idx < parts_per_branch_rem else 0
            )
            part_start = running_part_offset
            part_end = part_start + part_count
            candidate_group_counts.append(
                sum(logical_part_group_counts[part_start:part_end])
            )
            running_part_offset = part_end
        return candidate_group_counts

    def _estimate_b_weight_stream_count(candidate_branches: int) -> int:
        if candidate_branches <= 0:
            return 0
        candidate_group_counts = _candidate_ffn_branch_group_counts(candidate_branches)
        if not candidate_group_counts:
            return 2 * candidate_branches
        use_split = b_weight_split_requested
        if not use_split:
            return 2 * candidate_branches
        chunk_size = min(b_weight_split_chunk_size_requested, candidate_branches)
        chunk_size = min(chunk_size, candidate_branches)
        chunk_branches = [
            list(range(chunk_start, min(chunk_start + chunk_size, candidate_branches)))
            for chunk_start in range(0, candidate_branches, chunk_size)
        ]
        split_eligible = all(
            len({candidate_group_counts[idx] for idx in chunk}) == 1
            for chunk in chunk_branches
        )
        if not split_eligible:
            return 2 * candidate_branches
        chunk_count = len(chunk_branches)
        split_up = bup_split_requested
        split_down = bdown_split_requested
        up_streams = chunk_count if split_up else candidate_branches
        down_streams = chunk_count if split_down else candidate_branches
        return up_streams + down_streams

    def drop_non_root_branch(reason: str):
        non_root_selected = [i for i in selected_branch_indices if i != down_root_idx]
        if not non_root_selected:
            raise ValueError(
                f"Unable to prune FFN branches ({reason}): only root branch is selectable"
            )
        # Prefer a drop candidate that preserves a valid neighbor-chain
        # reduction order among the remaining selected down tiles.
        layout_edges = {
            (tuple(src), tuple(dst))
            for (src, dst) in ffn_layout.get("down_reduction_edges", [])
        }
        dropped_idx = None
        for candidate_idx in sorted(non_root_selected):
            remaining = [i for i in selected_branch_indices if i != candidate_idx]
            remaining_down_tiles = [all_ffn_down_tiles[i] for i in remaining]
            if all(
                (remaining_down_tiles[j], remaining_down_tiles[j + 1]) in layout_edges
                for j in range(len(remaining_down_tiles) - 1)
            ):
                dropped_idx = candidate_idx
                break
        if dropped_idx is None:
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

    def prune_or_fail(reason: str):
        # Even when grouped-O-proj pruning is disabled, hard compute-tile
        # capacity violations must still be pruned to produce a legal design.
        if pruning_disabled_for_this_config and not reason.startswith(
            "compute tile capacity exceeded"
        ):
            raise ValueError(
                "FFN branch pruning disabled for grouped O-proj "
                f"(o_proj_acc_group_size={o_proj_acc_group_size}); "
                f"cannot satisfy requested nB_tiles_distributed={nB_tiles_distributed}: "
                f"{reason}"
            )
        drop_non_root_branch(reason)

    layout_reduction_edges = {
        (tuple(src), tuple(dst))
        for (src, dst) in ffn_layout.get("down_reduction_edges", [])
    }
    ffn_reduction_edges: list[tuple[int, int]] | None = None
    ffn_down_output_branches: list[int] | None = None
    use_dual_ln2_ffn_inputs = False
    ffn_reduction_mode = "all_cores"
    # Mode-aware shim-output budget:
    # - base 5 streams: Q/K/V/W_O/R
    # - +FFN B-weight streams
    # - +1 stream in DDR mode for LN1 stage refill from host
    ln1_ddr_shim_output_streams = 1 if stage_ln1_to_ddr else 0
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
                prune_or_fail(
                    "selected FFN branches are not on a valid neighbor reduction chain "
                    f"(missing_edges={missing_chain_edges})"
                )
                continue
        topology = _build_ffn_reduction_topology(
            effective_candidate_branches,
            selected_down_tiles,
            ln2_tile,
        )
        if topology is None:
            prune_or_fail(
                "no valid FFN down reduction topology with LN2 N/E/S + DMA constraints "
                f"(branches={effective_candidate_branches}, down_tiles={selected_down_tiles}, ln2={ln2_tile})"
            )
            continue
        (
            candidate_reduction_edges,
            candidate_down_output_branches,
            candidate_reduction_mode,
        ) = topology
        candidate_use_dual_ln2_inputs = len(candidate_down_output_branches) == 2
        required_compute_tiles = (
            parallel_heads * 4
            + 2  # LN1 + LN2
            + 2 * effective_candidate_branches  # FFN up/down compute branches
        )
        if required_compute_tiles > 32:
            if len(selected_branch_indices) <= 1:
                raise ValueError(
                    "Configuration exceeds NPU2 compute tile capacity (32): "
                    f"needs {required_compute_tiles}"
                )
            prune_or_fail(
                f"compute tile capacity exceeded (needs {required_compute_tiles})"
            )
            continue
        # Shim output DMA budget:
        # - base streams: Q/K/V/W_O/R
        # - FFN B-weight streams (per-branch or split/packed)
        # - LN1 DDR stage refill stream in DDR mode
        estimated_b_weight_streams = _estimate_b_weight_stream_count(
            effective_candidate_branches
        )
        shim_output_stream_total = (
            5 + estimated_b_weight_streams + ln1_ddr_shim_output_streams
        )
        if shim_output_stream_total > 16:
            prune_or_fail(
                "shim output DMA budget exceeded for FFN weight streams "
                f"(branches={effective_candidate_branches}, "
                f"estimated_B_streams={estimated_b_weight_streams}, "
                f"ln1_ddr_streams={ln1_ddr_shim_output_streams}, "
                f"total_streams={shim_output_stream_total} > 16)"
            )
            continue
        # For wide-head/high-acc configurations, memtile BD-ID allocation for
        # FFN tail staging can fail even when nominal channel counts fit.
        # Keep up to 2 FFN branches (for practical parallelism), and prune
        # non-root branches only beyond that.
        if (
            not pruning_disabled_for_this_config
            and parallel_heads >= 6
            and proj_acc_depth >= 6
            and len(selected_branch_indices) > 2
        ):
            prune_or_fail(
                "memtile BD/channel budget exceeded for wide-head high-acc tail staging"
            )
            continue
        if (
            not pruning_disabled_for_this_config
            and parallel_heads >= 4
            and proj_acc_depth >= 8
            and len(selected_branch_indices) > 1
        ):
            prune_or_fail(
                "memtile BD/channel budget exceeded for high-acc tail staging"
            )
            continue
        ffn_reduction_edges = candidate_reduction_edges
        ffn_down_output_branches = candidate_down_output_branches
        use_dual_ln2_ffn_inputs = candidate_use_dual_ln2_inputs
        ffn_reduction_mode = candidate_reduction_mode
        break

    effective_ffn_branches = len(selected_branch_indices)
    if effective_ffn_branches <= 0:
        raise ValueError("No FFN branches selected")
    if ffn_reduction_edges is None or ffn_down_output_branches is None:
        raise ValueError("FFN reduction topology was not finalized")
    if selected_branch_indices[-1] != down_root_idx:
        logging.warning(
            "Using non-default FFN reduction root branch (selected tail idx=%d, layout root idx=%d)",
            selected_branch_indices[-1],
            down_root_idx,
        )

    # Keep requested nB distribution as the logical FFN partition count, then
    # map those logical parts onto the feasible physical branch count.
    logical_ffn_parts = min(max(1, nB_tiles_distributed), profile_replay_groups)
    logical_part_base = profile_replay_groups // logical_ffn_parts
    logical_part_rem = profile_replay_groups % logical_ffn_parts
    logical_part_group_counts = [
        logical_part_base + (1 if i < logical_part_rem else 0)
        for i in range(logical_ffn_parts)
    ]
    parts_per_branch_base = logical_ffn_parts // effective_ffn_branches
    parts_per_branch_rem = logical_ffn_parts % effective_ffn_branches
    ffn_col_group_counts = []
    ffn_col_group_offsets = []
    ffn_branch_logical_ranges = []
    running_group_offset = 0
    running_part_offset = 0
    for branch_idx in range(effective_ffn_branches):
        part_count = parts_per_branch_base + (
            1 if branch_idx < parts_per_branch_rem else 0
        )
        if part_count <= 0:
            raise ValueError(
                "Invalid logical-to-physical FFN partition mapping: "
                f"logical_parts={logical_ffn_parts}, physical_branches={effective_ffn_branches}"
            )
        part_start = running_part_offset
        part_end = part_start + part_count
        count = sum(logical_part_group_counts[part_start:part_end])
        ffn_col_group_counts.append(count)
        ffn_col_group_offsets.append(running_group_offset)
        ffn_branch_logical_ranges.append((part_start, part_end))
        running_group_offset += count
        running_part_offset = part_end
    if running_group_offset != profile_replay_groups:
        raise ValueError(
            "FFN logical partition mapping error: "
            f"mapped_groups={running_group_offset}, expected={profile_replay_groups}"
        )
    if running_part_offset != logical_ffn_parts:
        raise ValueError(
            "FFN logical partition mapping error: "
            f"mapped_parts={running_part_offset}, expected={logical_ffn_parts}"
        )
    # LN1 fanout is broadcast-only in both memtile and DDR modes.
    use_ln1_broadcast = True
    if len(set(ffn_col_group_counts)) != 1:
        raise ValueError(
            "FFN branches must be evenly partitioned for encoder_pipeline "
            f"(intermediate_size={ffn_intermediate_size}, emb_tile={emb_tile}, "
            f"requested_branches={nB_tiles_distributed}, effective_branches={effective_ffn_branches}, "
            f"branch_group_counts={ffn_col_group_counts})"
        )
    b_weight_split_enabled = b_weight_split_requested
    if b_weight_split_enabled and effective_ffn_branches <= 2:
        logging.info(
            "Disabling B-weight split mode for <=2 effective FFN branches "
            "(branches=%d)",
            effective_ffn_branches,
        )
        b_weight_split_enabled = False
    b_weight_split_chunk_size = b_weight_split_chunk_size_requested
    # Targeted low-head 6-way FFN memtile topology:
    # - 4pheads, 6pffn, 8pacc, 2opg
    # Use 2-way branch grouping for B-weight streams.
    use_low_head_6way_qkvo_split_group = (
        (not stage_ln1_to_ddr)
        and parallel_heads == 4
        and proj_acc_depth >= 8
        and o_proj_acc_group_size == 2
        and effective_ffn_branches >= 6
    )
    # DDR-specific variant:
    # - 4pheads, 6pffn, >=6pacc, 2opg
    # Split MHA ingress so W_O no longer concentrates on memtile col3.
    use_low_head_6way_ddr_mha_split = (
        stage_ln1_to_ddr
        and parallel_heads == 4
        and proj_acc_depth >= 8
        and o_proj_acc_group_size == 2
        and effective_ffn_branches >= 6
    )
    if use_low_head_6way_qkvo_split_group:
        b_weight_split_chunk_size = 2
    b_weight_split_chunk_size = min(b_weight_split_chunk_size, effective_ffn_branches)
    requested_split_chunks = [
        list(
            range(
                chunk_start,
                min(chunk_start + b_weight_split_chunk_size, effective_ffn_branches),
            )
        )
        for chunk_start in range(0, effective_ffn_branches, b_weight_split_chunk_size)
    ]
    if b_weight_split_enabled:
        split_eligible = all(
            len({ffn_col_group_counts[idx] for idx in chunk}) == 1
            for chunk in requested_split_chunks
        )
        if not split_eligible:
            logging.warning(
                "Disabling B-weight split mode: uneven per-chunk branch group counts=%s chunks=%s",
                ffn_col_group_counts,
                requested_split_chunks,
            )
            b_weight_split_enabled = False
    bup_split_enabled = b_weight_split_enabled and bup_split_requested
    bdown_split_enabled = b_weight_split_enabled and bdown_split_requested
    branch_split_chunks = (
        requested_split_chunks
        if b_weight_split_enabled
        else [[branch_idx] for branch_idx in range(effective_ffn_branches)]
    )
    logging.info(
        "FFN B-weight ingress: split=%s (up=%s down=%s) chunk_size=%d chunks=%s branch_group_counts=%s",
        b_weight_split_enabled,
        bup_split_enabled,
        bdown_split_enabled,
        b_weight_split_chunk_size,
        branch_split_chunks,
        ffn_col_group_counts,
    )
    ln1_broadcast_groups = max(ffn_col_group_counts)
    if ln1_broadcast_groups <= 0:
        raise ValueError(
            "LN1 broadcast groups must be > 0 "
            f"(got {ln1_broadcast_groups}, branch_counts={ffn_col_group_counts})"
        )
    final_ffn_branch_idx = effective_ffn_branches - 1
    ffn_down_acc_grouping_enabled = (
        not stage_ln1_to_ddr and proj_acc_depth >= 6 and effective_ffn_branches > 1
    )
    ffn_down_acc_group_ranges = [(0, effective_ffn_branches)]
    if ffn_down_acc_grouping_enabled:
        if effective_ffn_branches <= 3:
            ffn_down_acc_group_ranges = [(0, effective_ffn_branches)]
        else:
            first_group_size = (effective_ffn_branches + 1) // 2
            ffn_down_acc_group_ranges = [
                (0, first_group_size),
                (first_group_size, effective_ffn_branches),
            ]
    ffn_down_acc_group_size = max(
        group_end - group_start for group_start, group_end in ffn_down_acc_group_ranges
    )
    ffn_down_groups_have_uniform_counts = all(
        len(set(ffn_col_group_counts[group_start:group_end])) == 1
        for group_start, group_end in ffn_down_acc_group_ranges
    )
    grouped_ffn_down_root_input_overflow_risk = (
        emb_tile >= 128
        and parallel_heads <= 4
        and o_proj_acc_group_size > 1
        and effective_ffn_branches >= 6
    )
    if ffn_down_acc_grouping_enabled and not ffn_down_groups_have_uniform_counts:
        logging.info(
            "Disabling FFN down accumulation grouping because group branch counts are uneven: ranges=%s counts=%s",
            ffn_down_acc_group_ranges,
            ffn_col_group_counts,
        )
        ffn_down_acc_grouping_enabled = False
        ffn_down_acc_group_ranges = [(0, effective_ffn_branches)]
    if ffn_down_acc_grouping_enabled and grouped_ffn_down_root_input_overflow_risk:
        logging.info(
            "Disabling FFN down accumulation grouping because grouped roots would exceed compute-tile input DMA budget "
            "(emb_tile=%d parallel_heads=%d o_proj_acc_group_size=%d effective_ffn_branches=%d)",
            emb_tile,
            parallel_heads,
            o_proj_acc_group_size,
            effective_ffn_branches,
        )
        ffn_down_acc_grouping_enabled = False
        ffn_down_acc_group_ranges = [(0, effective_ffn_branches)]
    if ffn_down_acc_grouping_enabled:
        ffn_down_group_stage_branches = [
            group_end - 1 for group_start, group_end in ffn_down_acc_group_ranges
        ]
    else:
        ffn_down_group_stage_branches = list(range(effective_ffn_branches))
    ffn_down_group_stage_set = set(ffn_down_group_stage_branches)
    ffn_down_group_partner_srcs_by_stage = {}
    ffn_down_group_stage_by_src = {}
    if ffn_down_acc_grouping_enabled:
        for group_start, group_end in ffn_down_acc_group_ranges:
            stage_branch_idx = group_end - 1
            partner_sources = list(range(group_start, stage_branch_idx))
            ffn_down_group_partner_srcs_by_stage[stage_branch_idx] = partner_sources
            for src_branch_idx in partner_sources:
                ffn_down_group_stage_by_src[src_branch_idx] = stage_branch_idx
        ffn_reduction_edges = []
        ffn_down_output_branches = list(ffn_down_group_stage_branches)
    use_dual_ln2_ffn_inputs = len(ffn_down_output_branches) == 2
    ffn_reduction_sources = [src for (src, _) in ffn_reduction_edges]
    ffn_reduce_in_by_branch = {dst: src for (src, dst) in ffn_reduction_edges}
    ffn_reduce_dst_by_src = {src: dst for (src, dst) in ffn_reduction_edges}
    ffn_down_output_stream_idx_by_branch = {
        branch_idx: stream_idx
        for stream_idx, branch_idx in enumerate(ffn_down_output_branches)
    }
    stage_ffn_down_to_ddr = (
        len(ffn_down_output_branches) == 2 and not use_dual_ln2_ffn_inputs
    )
    staged_ffn_down_stream_idx = (
        1 if stage_ffn_down_to_ddr and len(ffn_down_output_branches) > 1 else None
    )
    ffn_down_ddr_stage_col = 0
    # Worker count:
    # - MHA path: 4 workers per parallel head
    # - Encoder tail:
    #   LN1(norm) + LN1(mul+resadd) + AddNorm2
    #   + (FFN up/down)*branches
    required_compute_tiles = parallel_heads * 4 + 3 + 2 * effective_ffn_branches
    if required_compute_tiles > 32:
        raise ValueError(
            "Configuration exceeds NPU2 compute tile capacity (32): "
            f"needs {required_compute_tiles}"
        )

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    if emulate_bf16_mmul_with_bfp16:
        r, s, t = 8, 8, 8
    else:
        r, s, t = 4, 8, 8

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
        "FFN distribution: groups=%d logical_parts=%d logical_part_counts=%s physical_counts=%s offsets=%s logical_ranges=%s",
        profile_replay_groups,
        logical_ffn_parts,
        logical_part_group_counts,
        ffn_col_group_counts,
        ffn_col_group_offsets,
        ffn_branch_logical_ranges,
    )
    logging.info(
        "FFN mapped placement: ln1=%s up(all)=%s down(all)=%s up(active)=%s down(active)=%s ln2=%s down_root=%s down_reduction_edges=%s down_to_ln2=%s non_neighbor_down_to_ln2=%s active_branch_indices=%s",
        ln1_tile,
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
        "FFN down-reduction direction policy: allow_west=%s edge_deltas=%s",
        False,
        layout_down_edge_deltas,
    )
    logging.info(
        "FFN reduction topology(active branch order): edges=%s down_output_branches=%s "
        "final_branch_idx=%d (down_tile=%s) dual_ln2_inputs=%s mode=%s "
        "stage_ffn_down_to_ddr=%s staged_stream_idx=%s stage_col=%d stage_ln1_to_ddr=%s",
        ffn_reduction_edges,
        ffn_down_output_branches,
        final_ffn_branch_idx,
        selected_down_tiles[final_ffn_branch_idx],
        use_dual_ln2_ffn_inputs,
        ffn_reduction_mode,
        stage_ffn_down_to_ddr,
        staged_ffn_down_stream_idx,
        ffn_down_ddr_stage_col,
        stage_ln1_to_ddr,
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

    dtype = bfloat16

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
    # Reserve tail space in OR for LN1 DDR staging:
    # [O region | residual R region | LN1 stage scratch region]
    ln1_dram_stage_rows = ln1_mode_hooks.ln1_dram_stage_rows(
        profile_replay_groups=ln1_broadcast_groups,
        seq_tile=seq_tile,
    )
    OR_ty = np.ndarray[
        (2 * seq_len + ln1_dram_stage_rows, embed_sz),
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
    matmul_init_kernel_o_proj = Kernel(
        f"matmul_init_bf16_bf16_o_proj",
        bin_name,
        [q_ty, wo_ty, o_ty],
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
    convert_stats_to_packet_kernel = Kernel(
        "pack_stats_f32_to_bf16_packet",
        bin_name,
        [sum_l1_ty, sum_l1_ty, o_ty, np.int32],
    )
    convert_packet_to_stats_kernel = Kernel(
        "unpack_stats_bf16_packet_to_f32",
        bin_name,
        [o_ty, sum_l1_ty, sum_l1_ty, np.int32],
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
    ffn_matmul_init_kernel_up_proj = Kernel(
        f"ffn_matmul_init_{dtype_str}_{dtype_str}_up_proj",
        bin_name,
        [o_ty, ffn_b_ty, o_ty],
    )
    ffn_matmul_init_kernel_down_proj = Kernel(
        f"ffn_matmul_init_{dtype_str}_{dtype_str}_down_proj",
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
    # Topology-specific split/group path:
    # - Split Q/K/V/W_O ingress into 2 streams.
    # - Pair with 2-way B-stream grouping for 6 FFN branches.
    use_two_way_mha_stream_split = (
        use_low_head_6way_qkvo_split_group or use_low_head_6way_ddr_mha_split
    )
    mha_stream_split_factor = 2 if use_two_way_mha_stream_split else 1
    if parallel_heads % mha_stream_split_factor != 0:
        raise ValueError(
            "parallel_heads must be divisible by MHA stream split factor "
            f"({parallel_heads} % {mha_stream_split_factor} != 0)"
        )
    heads_per_mha_stream = parallel_heads // mha_stream_split_factor
    if use_two_way_mha_stream_split:
        # Rebalance low-head 6-way FFN topologies so each shim column stays
        # within the 2-output-channel budget once B-weight streams are packed.
        q_stream_mem_cols = [0, 5]
        k_stream_mem_cols = [1, 2]
        v_stream_mem_cols = [0, 6]
        ow_stream_mem_cols = [3, 4]
    else:
        q_stream_mem_cols = [0]
        k_stream_mem_cols = [1]
        v_stream_mem_cols = [2]
        ow_stream_mem_cols = [3]
    stream_col_checks = [
        len(q_stream_mem_cols) != mha_stream_split_factor,
        len(k_stream_mem_cols) != mha_stream_split_factor,
        len(v_stream_mem_cols) != mha_stream_split_factor,
        len(ow_stream_mem_cols) != mha_stream_split_factor,
    ]
    if any(stream_col_checks):
        raise ValueError(
            "Invalid MHA stream memtile column mapping for split factor "
            f"{mha_stream_split_factor}: "
            f"Q={q_stream_mem_cols} K={k_stream_mem_cols} "
            f"V={v_stream_mem_cols} W_O={ow_stream_mem_cols}"
        )
    logging.info(
        "MHA ingress split: enabled=%s factor=%d Q=%s K=%s V=%s W_O=%s",
        use_two_way_mha_stream_split,
        mha_stream_split_factor,
        q_stream_mem_cols,
        k_stream_mem_cols,
        v_stream_mem_cols,
        ow_stream_mem_cols,
    )

    # Reduce BD pressure for the stressed 4pheads/6pffn/8pacc/2opg memtile case.
    mha_ingress_fifo_depth = 1 if use_two_way_mha_stream_split else of_depth
    inQ_streams = []
    memQ = []
    for split_idx in range(mha_stream_split_factor):
        inQ_streams.append(
            ObjectFifo(
                np.ndarray[(seq_tile, d * heads_per_mha_stream), np.dtype[dtype]],
                name="inQ" if split_idx == 0 else f"inQSplit{split_idx}",
                depth=mha_ingress_fifo_depth,
            )
        )
        memQ.extend(
            inQ_streams[split_idx]
            .cons()
            .split(
                offsets=[seq_tile * d * i for i in range(heads_per_mha_stream)],
                obj_types=[q_ty] * heads_per_mha_stream,
                names=[
                    f"memQ{split_idx * heads_per_mha_stream + i}"
                    for i in range(heads_per_mha_stream)
                ],
                dims_to_stream=[q_dims] * heads_per_mha_stream,
                depths=[mha_ingress_fifo_depth] * heads_per_mha_stream,
                placement=Tile(col=q_stream_mem_cols[split_idx], row=1),
            )
        )  # Split between N parallel blocks of sequences

    # K is stored in column-major order
    k_dims = [(kv_seq_tile // t, t * d), (d // s, s), (t, d), (s, 1)]
    inK_streams = []
    memK = []
    for split_idx in range(mha_stream_split_factor):
        inK_streams.append(
            ObjectFifo(
                np.ndarray[(kv_seq_tile, d * heads_per_mha_stream), np.dtype[dtype]],
                name="inK" if split_idx == 0 else f"inKSplit{split_idx}",
                depth=mha_ingress_fifo_depth,
            )
        )
        memK.extend(
            inK_streams[split_idx]
            .cons()
            .split(
                offsets=[kv_seq_tile * d * i for i in range(heads_per_mha_stream)],
                obj_types=[k_ty] * heads_per_mha_stream,
                names=[
                    f"memK{split_idx * heads_per_mha_stream + i}"
                    for i in range(heads_per_mha_stream)
                ],
                dims_to_stream=[k_dims] * heads_per_mha_stream,
                depths=[mha_ingress_fifo_depth] * heads_per_mha_stream,
                placement=Tile(col=k_stream_mem_cols[split_idx], row=1),
            )
        )  # Split between N parallel blocks of heads

    v_dims = [
        (kv_seq_tile // s, s * kv_seq_tile),
        (d // t, t),
        (s, kv_seq_tile),
        (t, 1),
    ]
    inV_streams = []
    memV = []
    for split_idx in range(mha_stream_split_factor):
        inV_streams.append(
            ObjectFifo(
                np.ndarray[(kv_seq_tile, d * heads_per_mha_stream), np.dtype[dtype]],
                name="inV" if split_idx == 0 else f"inVSplit{split_idx}",
                depth=mha_ingress_fifo_depth,
            )
        )
        memV.extend(
            inV_streams[split_idx]
            .cons()
            .split(
                offsets=[kv_seq_tile * d * i for i in range(heads_per_mha_stream)],
                obj_types=[v_ty] * heads_per_mha_stream,
                names=[
                    f"memV{split_idx * heads_per_mha_stream + i}"
                    for i in range(heads_per_mha_stream)
                ],
                dims_to_stream=[v_dims] * heads_per_mha_stream,
                depths=[mha_ingress_fifo_depth] * heads_per_mha_stream,
                placement=Tile(col=v_stream_mem_cols[split_idx], row=1),
            )
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
    if use_two_way_mha_stream_split:
        ow_fifo_depth = 1
    inOW_streams = []
    memOW = []
    for split_idx in range(mha_stream_split_factor):
        inOW_streams.append(
            ObjectFifo(
                np.ndarray[(d * heads_per_mha_stream, emb_tile), np.dtype[dtype]],
                name="inOW" if split_idx == 0 else f"inOWSplit{split_idx}",
                depth=ow_fifo_depth,
            )
        )
        memOW.extend(
            inOW_streams[split_idx]
            .cons()
            .split(
                offsets=[d * emb_tile * i for i in range(heads_per_mha_stream)],
                obj_types=[wo_ty] * heads_per_mha_stream,
                names=[
                    f"memOW{split_idx * heads_per_mha_stream + i}"
                    for i in range(heads_per_mha_stream)
                ],
                dims_to_stream=[ow_dims] * heads_per_mha_stream,
                depths=[ow_fifo_depth] * heads_per_mha_stream,
                placement=Tile(col=ow_stream_mem_cols[split_idx], row=1),
            )
        )  # Split between N parallel blocks of heads

    # Partial out proj tiles produced by per-head PV workers.
    outOProj = [
        ObjectFifo(q_ty, depth=of_depth, name=f"outOProj{i}")
        for i in range(parallel_heads)
    ]
    o_proj_worker_tiles = [Tile(col=i, row=5) for i in range(parallel_heads)]
    # Grouped O-proj accumulation staging.
    # For grouped mode, each group-boundary core (group_size-1, 2*group_size-1, ...)
    # performs memtile accumulation buffering. This matches per-group local
    # accumulation followed by cross-group reduction through neighbor links.
    o_proj_accum_core_indices = o_proj_stage_core_indices
    o_proj_accum_core_set = set(o_proj_accum_core_indices)
    outOProjAccumIn = [None] * parallel_heads
    outOProjAccumOut = [None] * parallel_heads
    use_o_proj_acc_row_store = parallel_heads <= 4

    def _allocate_o_proj_acc_row_store_channels(
        mem_tile_col: int, slot_idx: int
    ) -> tuple[int, int] | None:
        # Keep grouped O-proj row-store traffic off the heavily used low
        # channels on each memtile. Some columns, such as col3, are already
        # saturated by W_O/LN replay traffic in common topologies and still
        # need the old forwarded FIFO path.
        channel_pairs_by_col = {
            0: [(5, 5)],
            4: [(2, 2), (3, 3)],
            5: [(3, 3), (4, 4)],
            6: [(3, 3)],
            7: [(4, 4), (5, 5)],
        }
        col_pairs = channel_pairs_by_col.get(mem_tile_col)
        if col_pairs is None or slot_idx >= len(col_pairs):
            return None
        return col_pairs[slot_idx]

    # Memtile DMA/BD pressure model (per tile):
    # - col4: memBUp stream pair (low depth), can host 2 O-proj accum FIFOs.
    # - col5: memBDown stream pair (low depth), can host 2 O-proj accum FIFOs.
    # - col6: ffnDownAccum stream pair (depth=proj_acc_depth), host at most 1.
    # - col7: memR + memLN2 stream pairs, host at most 2 O-proj accum FIFOs.
    # Keep O-proj accum FIFOs off cols 6/7 for <=4 parallel heads so those
    # memtiles can absorb FFN/LN traffic at high seq/head configurations.
    if o_proj_acc_group_size > 1 and parallel_heads >= 6 and proj_acc_depth >= 6:
        # Grouped O-proj staging uses fewer accum streams; prioritize less
        # contended memtiles while avoiding col3 fanout pressure from W_O
        # split and reducing col5 pressure from LN1 replay + B streams.
        acc_mem_tile_order = (
            [5, 6, 4, 7]
            if effective_ffn_branches <= 2 and proj_acc_depth >= 16
            else [5, 6, 7, 4]
        )
    elif parallel_heads <= 4:
        if proj_acc_depth >= 8 and effective_ffn_branches > 1:
            # For 4-way grouped O-proj at high acc depth, placing the only
            # stage core on col4 can collide with FFN/LN replay staging.
            if (
                o_proj_acc_group_size >= 4
                and len(o_proj_accum_core_indices) == 1
                and effective_ffn_branches >= 4
            ):
                acc_mem_tile_order = [0, 6, 5, 7, 4]
            else:
                # For high-acc (depth=8) multi-branch tails, keep the base
                # O-proj staging order stable and remap LN replay / FFN
                # down-acc streams around it via mode-specific hooks.
                acc_mem_tile_order = [4, 5, 7, 6]
        else:
            # Keep col5 in the <=4-head map and avoid concentrating depth-8
            # O-proj accumulators on col6 where FFN tail staging is anchored.
            # For low-head grouped 6-way FFN (acc>=6), place the second
            # O-proj accum stream on col3 to avoid memtile-5 BD saturation.
            if (
                parallel_heads == 4
                and proj_acc_depth >= 6
                and o_proj_acc_group_size > 1
                and effective_ffn_branches >= 6
            ):
                # For low-head 6-way tails, keep grouped O-proj staging away
                # from col4/5 where LN replay and dual B_Up ingress can push
                # the memtile BD allocator over its limit.
                acc_mem_tile_order = [6, 3, 7, 4]
            else:
                acc_mem_tile_order = [4, 5, 7, 3]
    elif proj_acc_depth >= 6:
        # Keep high-head accumulator streams on 4/5/6/7. Col3 already carries
        # high fanout from W_O staging and can hit output-channel limits.
        acc_mem_tile_order = [4, 5, 6, 7, 6, 5, 7]
    else:
        # For wider MHA parallelism, spread across available memtiles.
        acc_mem_tile_order = [4, 5, 6, 7, 4, 5, 7]
    if len(o_proj_accum_core_indices) > len(acc_mem_tile_order):
        raise ValueError(
            "Unsupported o_proj accumulation group count for current memtile DMA budget: "
            f"{len(o_proj_accum_core_indices)} > {len(acc_mem_tile_order)}"
        )
    acc_mem_tile_cols = acc_mem_tile_order[: len(o_proj_accum_core_indices)]
    # Must match proj_acc_depth: O-proj kernels initialize/drain one token per
    # accumulation step before reuse. Shrinking this depth below proj_acc_depth
    # can deadlock (producer fills FIFO before consumer phase starts).
    o_proj_acc_fifo_depth = proj_acc_depth
    o_proj_acc_row_store_slots_by_col: dict[int, int] = {}
    for stage_idx, core_idx in enumerate(o_proj_accum_core_indices):
        acc_mem_col = acc_mem_tile_cols[stage_idx]
        row_store_slot_idx = o_proj_acc_row_store_slots_by_col.get(acc_mem_col, 0)
        row_store_channels = (
            _allocate_o_proj_acc_row_store_channels(acc_mem_col, row_store_slot_idx)
            if use_o_proj_acc_row_store
            else None
        )
        if row_store_channels is not None:
            memtile_ingress_channel, memtile_egress_channel = row_store_channels
            outOProjAccumIn[core_idx] = outOProjAccumOut[core_idx] = MemTileRowStore(
                obj_type=o_ty,
                compute_tile=o_proj_worker_tiles[core_idx],
                mem_tile=Tile(col=acc_mem_col, row=1),
                part_count=proj_acc_depth,
                buffer_count=2,
                compute_buffer_count=o_proj_acc_row_store_compute_buffer_count,
                compute_produce_buffer_count=(
                    o_proj_acc_row_store_compute_produce_buffer_count
                ),
                compute_consume_buffer_count=(
                    o_proj_acc_row_store_compute_consume_buffer_count
                ),
                name=f"outOProjAccum{core_idx}",
                compute_mm2s_channel=0,
                compute_s2mm_channel=1,
                memtile_ingress_channel=memtile_ingress_channel,
                memtile_egress_channel=memtile_egress_channel,
            )
            o_proj_acc_row_store_slots_by_col[acc_mem_col] = row_store_slot_idx + 1
            logging.debug(
                "Placed outOProjAccum[%d] row-store on mem tile (%d,1) "
                "with channels in=%d out=%d",
                core_idx,
                acc_mem_col,
                memtile_ingress_channel,
                memtile_egress_channel,
            )
        else:
            outOProjAccumOut[core_idx] = ObjectFifo(
                o_ty, depth=1, name=f"outOProjAccumOut{core_idx}"
            )
            outOProjAccumIn[core_idx] = (
                outOProjAccumOut[core_idx]
                .cons(depth=o_proj_acc_fifo_depth)
                .forward(
                    name=f"outOProjAccumIn{core_idx}",
                    depth=o_proj_acc_fifo_depth,
                    placement=Tile(col=acc_mem_col, row=1),
                )
            )
            logging.debug(
                "Placed outOProjAccum[%d] FIFO fallback on mem tile (%d,1) "
                "with fifo_depth=%d",
                core_idx,
                acc_mem_col,
                o_proj_acc_fifo_depth,
            )

    # Intra-group neighbor chain: each core forwards its partial contribution to
    # the next core in the same group each qkv/acc iteration.
    o_proj_group_chain_depth = (
        1 if (o_proj_acc_group_size > 1 and parallel_heads >= 6) else of_depth
    )
    outOGroupPart = [None] * (parallel_heads - 1)
    for i in range(parallel_heads - 1):
        if o_proj_acc_group_size > 1 or o_proj_group_idx[i] == o_proj_group_idx[i + 1]:
            edge_depth = o_proj_group_chain_depth
            if (
                emb_tile >= 128
                and o_proj_acc_group_size > 1
                and i in o_proj_accum_core_set
            ):
                edge_depth = 1
            outOGroupPart[i] = ObjectFifo(
                o_ty,
                depth=edge_depth,
                name=f"outOGroupPart{i}",
            )

    # Global chain across group stage cores to form the final O tile for LN1.
    # Grouped mode (size=2) carries the running partial across all O-proj cores
    # each qkv/acc iteration and only the final stage core emits to LN1, so no
    # additional stage-level chain is required there.
    # Large emb_tile layouts can exceed O-proj core L1 when this chain is
    # double-buffered; keep it shallow there.
    o_proj_stage_chain_depth = 1 if emb_tile >= 128 else of_depth
    outOPart = []
    if o_proj_acc_group_size == 1:
        for i in range(num_o_proj_acc_groups - 1):
            outOPart.append(
                ObjectFifo(o_ty, depth=o_proj_stage_chain_depth, name=f"outOPart{i}")
            )
    logging.debug(
        "O-proj accumulation grouping: group_size=%d groups=%d stage_cores=%s group_idx=%s",
        o_proj_acc_group_size,
        num_o_proj_acc_groups,
        o_proj_stage_core_indices,
        o_proj_group_idx,
    )

    # Keep LN residual staging shallow to avoid BD allocator over-subscription
    # in high-acc/high-parallel topologies.
    ln_fifo_depth = of_depth
    # FFN-up A-input depth controls:
    # - Broadcast FIFO at LN1 producer side.
    # - Per-branch consumer FIFO depth at FFN-up cores.
    # Keep consumer-side depth shallow for emb_tile=128 to stay within L1.
    ffn_up_broadcast_depth = 2 if (effective_ffn_branches >= 6) else 1
    ffn_up_consumer_depth = 1 if emb_tile >= 128 else 2
    # Use depth 2 for O-proj->LN1 input.
    ln_input_depth = 2
    # emb_tile=128 can exceed O-proj core L1 with double-buffered O->LN1.
    o_proj_input_depth = 1 if emb_tile >= 128 else 2
    ln1_replay_tile_bytes = seq_tile * emb_tile * 2
    # Fused LN1 now keeps these buffers resident on the LN1 tile:
    # - outLNBroadcast producer double buffer
    # - memR consumer double buffer
    # - ln1Replay destination slot
    ln1_fused_tile_resident_slots = 5
    ln1_replay_row_store_compute_produce_buffer_count = (
        3
        if ((ln1_fused_tile_resident_slots + 3) * ln1_replay_tile_bytes <= 60 * 1024)
        else 2
    )
    # O-proj stream into LN1 norm worker (no DMA layout transform).
    outOProjInput = ObjectFifo(
        o_ty,
        name="outOProjInput",
        depth=o_proj_input_depth,
    )
    oProjVerifyOut = (
        ObjectFifo(o_ty, name="oProjVerifyOut", depth=o_proj_input_depth)
        if direct_verify_stage == STAGE_O_PROJ
        else None
    )
    ln1_replay_mem_tile_col = ln1_mode_hooks.choose_ln1_replay_mem_tile_col(
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        effective_ffn_branches=effective_ffn_branches,
        o_proj_acc_group_size=o_proj_acc_group_size,
    )
    ln1_norm_tile_obj = Tile(col=ln1_tile[0], row=ln1_tile[1])
    ln1Replay = MemTileRowStore(
        obj_type=o_ty,
        compute_tile=ln1_norm_tile_obj,
        mem_tile=Tile(col=ln1_replay_mem_tile_col, row=1),
        part_count=ln_tiles_per_q_block,
        buffer_count=2,
        compute_buffer_count=ln1_replay_row_store_compute_buffer_count,
        compute_produce_buffer_count=(
            ln1_replay_row_store_compute_produce_buffer_count
        ),
        compute_consume_buffer_count=(
            ln1_replay_row_store_compute_consume_buffer_count
        ),
        name="ln1Replay",
    )
    r_dims = [
        (seq_tile // r, r * emb_tile),
        (emb_tile // s, s),
        (r, emb_tile),
        (s, 1),
    ]
    # AddNorm-2 consumes staged residual tiles from LN1 output path.
    ffn_residual_depth = proj_acc_depth
    ln_mem_tile_col = 7
    # AddNorm2 needs two logical passes (sum/sumsq pass + output pass).
    # Prefer LN2-side replay FIFO by default so FFN-down only emits one pass
    # unless explicitly overridden.
    high_pacc_direct_ln2_replay_default = proj_acc_depth >= 16 and (
        (parallel_heads >= 6 and effective_ffn_branches <= 2)
        or (parallel_heads == 4 and effective_ffn_branches >= 6)
    )
    emit_ln2_replay_from_down_default = (
        ffn_down_acc_grouping_enabled and use_dual_ln2_ffn_inputs
    ) or high_pacc_direct_ln2_replay_default
    emit_ln2_replay_from_down = emit_ln2_replay_from_down_default
    allow_high_pacc_dual_ln2_direct_replay = (
        use_dual_ln2_ffn_inputs and high_pacc_direct_ln2_replay_default
    )
    if (
        use_dual_ln2_ffn_inputs
        and emit_ln2_replay_from_down
        and not ffn_down_acc_grouping_enabled
        and not allow_high_pacc_dual_ln2_direct_replay
    ):
        emit_ln2_replay_from_down = False
    use_ln2_replay_store = ln2_compute_enabled and (not emit_ln2_replay_from_down)
    # Targeted remap for low-head/high-acc grouped-O-proj memtile topologies:
    # col4 can saturate BD budget with {O-proj accum + LN2 replay + FFN down}.
    # Route LN2 replay through col2 to keep per-memtile BD usage <= 48.
    needs_low_head_6way_ln2_replay_remap = (
        (not stage_ln1_to_ddr)
        and parallel_heads <= 4
        and proj_acc_depth >= 8
        and o_proj_acc_group_size > 1
        and effective_ffn_branches >= 6
    )
    if needs_low_head_6way_ln2_replay_remap:
        ln2_replay_mem_tile_default = 2
    else:
        ln2_replay_mem_tile_default = (
            3
            if (
                parallel_heads <= 2
                and proj_acc_depth >= 8
                and effective_ffn_branches > 1
            )
            else 4
        )
    ln2_replay_mem_tile_col = ln1_mode_hooks.choose_ln2_replay_mem_tile_col(
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        effective_ffn_branches=effective_ffn_branches,
        o_proj_acc_group_size=o_proj_acc_group_size,
        stage_ln1_to_ddr=stage_ln1_to_ddr,
        default_col=ln2_replay_mem_tile_default,
    )
    if (
        (not stage_ln1_to_ddr)
        and parallel_heads >= 6
        and proj_acc_depth >= 16
        and effective_ffn_branches <= 1
        and o_proj_acc_group_size > 1
        and ln2_replay_mem_tile_col == 4
    ):
        ln2_replay_mem_tile_col = 5
    # Enforce targeted remap for the known saturated memtile topology.
    if (
        (not stage_ln1_to_ddr)
        and parallel_heads == 4
        and proj_acc_depth >= 8
        and o_proj_acc_group_size == 2
        and nB_tiles_distributed >= 6
        and effective_ffn_branches >= 6
    ):
        ln2_replay_mem_tile_col = 2
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

    # LN2 output stream to shim.
    ln2_output_fifo_depth = of_depth
    o_dims = [(seq_tile // r, r * emb_tile), (r, s), (emb_tile // s, r * s), (s, 1)]

    def _allocate_ffn_weight_mem_tile_cols_by_streams(
        up_streams: int,
        down_streams: int,
        reserved_output_load_by_col: dict[int, int] | None = None,
    ) -> tuple[list[int], list[int]]:
        # Per-shim output-channel capacity is 2. Baseline encoder streams use:
        # col0(Q), col1(K), col2(V), col3(W_O), col7(R).
        shim_out_capacity = {col: 2 for col in range(8)}
        for used_col in (0, 1, 2, 3, 7):
            shim_out_capacity[used_col] -= 1
        if reserved_output_load_by_col:
            for col, reserved in reserved_output_load_by_col.items():
                shim_out_capacity[col] -= reserved
            overflow = {col: cap for col, cap in shim_out_capacity.items() if cap < 0}
            if overflow:
                raise ValueError(
                    "Insufficient shim output DMA channels after applying reserved streams "
                    f"(reserved={reserved_output_load_by_col}, overflow={overflow})"
                )

        # Col 7 already carries residual/LN2 traffic in most topologies.
        # Prefer B-weight ingress on non-col7 memtiles and use col7 only as fallback.
        preferred_cols = [6, 5, 4, 3, 2, 1, 0, 7]
        reserved_cols = (
            set(reserved_output_load_by_col.keys())
            if reserved_output_load_by_col
            else set()
        )
        # When LN1 DDR staging reserves shim output slots on stage columns,
        # prefer routing B_Down on non-stage columns to reduce tail-stage
        # contention on the same memtile/shim pair.
        preferred_down_cols = [c for c in preferred_cols if c not in reserved_cols] + [
            c for c in preferred_cols if c in reserved_cols
        ]
        up_cols = []
        down_cols = []
        for _ in range(up_streams):
            candidates = [c for c in preferred_cols if shim_out_capacity[c] > 0]
            if not candidates:
                raise ValueError(
                    "Insufficient shim output DMA channels for FFN weight streams "
                    f"(need up={up_streams}, down={down_streams}; "
                    f"available={sum(v for v in shim_out_capacity.values() if v > 0)})"
                )
            # Pick the column with the most remaining budget; tie-break by preference order.
            col = max(
                candidates,
                key=lambda c: (shim_out_capacity[c], -preferred_cols.index(c)),
            )
            up_cols.append(col)
            shim_out_capacity[col] -= 1
        for _ in range(down_streams):
            candidates = [c for c in preferred_down_cols if shim_out_capacity[c] > 0]
            if not candidates:
                raise ValueError(
                    "Insufficient shim output DMA channels for FFN weight streams "
                    f"(need up={up_streams}, down={down_streams}; "
                    f"available={sum(v for v in shim_out_capacity.values() if v > 0)})"
                )
            col = max(
                candidates,
                key=lambda c: (shim_out_capacity[c], -preferred_down_cols.index(c)),
            )
            down_cols.append(col)
            shim_out_capacity[col] -= 1
        return up_cols, down_cols

    def _allocate_ffn_weight_mem_tile_cols(
        branches: int,
    ) -> tuple[list[int], list[int]]:
        return _allocate_ffn_weight_mem_tile_cols_by_streams(branches, branches)

    # Keep FFN branch staging streams off LN replay (col 5) when MHA uses many
    # heads; col 5 already carries LN1 full-row replay traffic.
    if parallel_heads >= 6 and proj_acc_depth >= 6:
        branch_stage_cols = [6, 4, 5]
        high_head_grouped_two_branch = (
            o_proj_acc_group_size > 1 and effective_ffn_branches == 2
        )
        if high_head_grouped_two_branch:
            # Rebalance B-stream staging away from col4 so col4 can host
            # LN1 replay + O-proj accum + FFN-down accum without exceeding
            # memtile BD=48 in grouped high-acc topologies.
            branch_bup_cols = [5, 6, 5]
            branch_down_b_cols = [6, 5, 4]
        else:
            # Place both B_Up streams on col 5 in high-head/high-acc mode to keep
            # tail memtile block usage within the 48-block allocator budget.
            branch_bup_cols = [5, 5, 5]
            branch_down_b_cols = [4, 6, 5]
    elif parallel_heads >= 6:
        branch_stage_cols = [6, 4, 5]
        branch_bup_cols = [6, 4, 5]
        branch_down_b_cols = [4, 6, 5]
    else:
        # For low-head configurations, keep a wider default pool so FFN can
        # scale up to 6-way distribution when resources permit.
        branch_stage_cols = [6, 5, 4, 7, 3, 2]
        if bup_split_enabled or bdown_split_enabled:
            split_chunk_count = len(branch_split_chunks)
            up_stream_count = (
                split_chunk_count if bup_split_enabled else effective_ffn_branches
            )
            down_stream_count = (
                split_chunk_count if bdown_split_enabled else effective_ffn_branches
            )
            alloc_bup_cols, alloc_down_cols = (
                _allocate_ffn_weight_mem_tile_cols_by_streams(
                    up_stream_count,
                    down_stream_count,
                )
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
            # For low-head/high-acc 6-way FFN grouped-O-proj memtile layouts,
            # rebalance legacy 2-branch split chunks across memtiles.
            needs_low_head_6way_b_rebalance = (
                parallel_heads <= 4
                and proj_acc_depth >= 8
                and o_proj_acc_group_size > 1
                and effective_ffn_branches >= 6
                and bup_split_enabled
                and bdown_split_enabled
                and len(branch_split_chunks) == 3
                and all(len(chunk) == 2 for chunk in branch_split_chunks)
            )
            if needs_low_head_6way_b_rebalance:
                # Keep each shim column at <=2 output streams for the
                # 4pheads/6pffn/8pacc/2opg memtile topology.
                rebalance_up_cols = [6, 4, 5]
                rebalance_down_cols = [1, 2, 3]
                for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                    for branch_idx in chunk_branches:
                        branch_bup_cols[branch_idx] = rebalance_up_cols[chunk_idx]
                        branch_down_b_cols[branch_idx] = rebalance_down_cols[chunk_idx]
                logging.info(
                    "Applied low-head 6-way B-stream rebalance: B_Up=%s B_Down=%s",
                    branch_bup_cols,
                    branch_down_b_cols,
                )
            needs_low_head_6way_b_rebalance_acc6 = (
                parallel_heads == 4
                and proj_acc_depth >= 6
                and proj_acc_depth < 8
                and o_proj_acc_group_size > 1
                and effective_ffn_branches >= 6
                and bdown_split_enabled
                and len(branch_split_chunks) == 3
                and all(len(chunk) == 2 for chunk in branch_split_chunks)
            )
            if needs_low_head_6way_b_rebalance_acc6:
                if bup_split_enabled:
                    if use_low_head_6way_ddr_mha_split:
                        # DDR low-head 6-way split path already occupies:
                        # Q on cols0/5, K on cols1/2, V on cols0/6, W_O on
                        # cols3/4, LN1 refill on col6, and residual on col7.
                        # Pack B streams onto the remaining shim slots so no
                        # shim or memtile exceeds its output-channel budget.
                        rebalance_up_cols = [5, 4, 7]
                        rebalance_down_cols = [1, 2, 3]
                    else:
                        # For 4pheads/6pffn/6pacc/2opg with split B_Up/B_Down,
                        # keep cols4/5 below the memtile output-channel limit by
                        # shifting one B-stream chunk to col0 and one to col7.
                        rebalance_up_cols = [6, 5, 0]
                        rebalance_down_cols = [6, 1, 7]
                    for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                        for branch_idx in chunk_branches:
                            branch_bup_cols[branch_idx] = rebalance_up_cols[chunk_idx]
                            branch_down_b_cols[branch_idx] = rebalance_down_cols[
                                chunk_idx
                            ]
                else:
                    # When B_Up is unsplit, B_Up already saturates cols4/5/6
                    # at shim output. Move B_Down chunks off cols2/3 (which
                    # also carry V/W_O split fanout) and onto cols0/7/1.
                    # Also move one B_Up branch off col6 to reduce memtile-6
                    # BD/start pressure in 4pheads/6pffn/6pacc/2opg layouts.
                    if len(branch_bup_cols) >= 4:
                        branch_bup_cols[3] = 2
                    rebalance_down_cols = [0, 7, 1]
                    for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                        for branch_idx in chunk_branches:
                            branch_down_b_cols[branch_idx] = rebalance_down_cols[
                                chunk_idx
                            ]
                logging.info(
                    "Applied low-head 6-way B-stream rebalance(acc6): B_Up=%s B_Down=%s",
                    branch_bup_cols,
                    branch_down_b_cols,
                )
        else:
            branch_bup_cols, branch_down_b_cols = _allocate_ffn_weight_mem_tile_cols(
                effective_ffn_branches
            )
    if effective_ffn_branches > len(branch_stage_cols):
        raise ValueError(
            "Unsupported effective_ffn_branches for current memtile assignment "
            f"({effective_ffn_branches} > {len(branch_stage_cols)})"
        )
    (
        branch_stage_cols,
        branch_bup_cols,
        branch_down_b_cols,
        ln1_ddr_staged_branch_indices,
    ) = ln1_mode_hooks.plan_branch_stage_configuration(
        branch_stage_cols=branch_stage_cols,
        branch_bup_cols=branch_bup_cols,
        branch_down_b_cols=branch_down_b_cols,
        effective_ffn_branches=effective_ffn_branches,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        ln1_replay_mem_tile_col=ln1_replay_mem_tile_col,
        ffn_col_group_counts=ffn_col_group_counts,
        bup_split_enabled=bup_split_enabled,
        bdown_split_enabled=bdown_split_enabled,
        branch_split_chunks=branch_split_chunks,
        _allocate_ffn_weight_mem_tile_cols_by_streams=_allocate_ffn_weight_mem_tile_cols_by_streams,
        logging_module=logging,
    )
    ffn_a_stage_mem_tile_cols = branch_stage_cols[:effective_ffn_branches]

    ln1_path = ln1_mode_hooks.build_ln1_to_ffn_up_path(
        o_ty=o_ty,
        ffn_up_broadcast_depth=ffn_up_broadcast_depth,
        ffn_up_consumer_depth=ffn_up_consumer_depth,
        effective_ffn_branches=effective_ffn_branches,
        ffn_a_stage_mem_tile_cols=ffn_a_stage_mem_tile_cols,
        ln1_ddr_staged_branch_indices=ln1_ddr_staged_branch_indices,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        object_fifo_ctor=ObjectFifo,
        tile_ctor=Tile,
    )
    ln1Broadcast = ln1_path["ln1Broadcast"]
    memOutLNCons = ln1_path["memOutLNCons"]
    ln1OutStageToDDR = ln1_path["ln1OutStageToDDR"]
    ln1InFromDDR = ln1_path["ln1InFromDDR"]
    # FFN residual path for AddNorm-2. In DDR mode, reuse the staged LN1 output
    # in host memory instead of emitting a duplicate on-chip residual stream
    # from the LN1 worker.
    ffnRFromDDR = ObjectFifo(o_ty, name="ffnRFromDDR", depth=1)
    ffnRIn = ffnRFromDDR.cons(depth=ffn_residual_depth).forward(
        obj_type=o_ty,
        name="ffnRIn",
        depth=ffn_residual_depth,
        placement=Tile(col=ln_mem_tile_col, row=1),
    )

    # FFN weights (Up/Down projections)
    b_dims = [
        (emb_tile // s, s * emb_tile),
        (emb_tile // t, t),
        (s, emb_tile),
        (t, 1),
    ]
    # Large emb_tile (e.g. 128) can overflow FFN-up L1 when B-weight FIFOs are
    # double-buffered at the consumer; keep them shallow in that regime.
    ffn_weight_fifo_depth = 1 if emb_tile >= 128 else 2
    if use_two_way_mha_stream_split:
        ffn_weight_fifo_depth = 1
    ffn_bup_mem_tile_cols = branch_bup_cols[:effective_ffn_branches]
    ffn_bdown_mem_tile_cols = branch_down_b_cols[:effective_ffn_branches]
    logging.info(
        "FFN B-weight memtile columns: B_Up=%s B_Down=%s",
        ffn_bup_mem_tile_cols,
        ffn_bdown_mem_tile_cols,
    )

    def _choose_ln1_residual_refill_col(
        stage_fill_col: int,
        bup_fill_cols: list[int],
        bdown_fill_cols: list[int],
    ) -> int:
        shim_out_capacity = {col: 2 for col in range(8)}
        for used_col in (0, 1, 2, 3, 7):
            shim_out_capacity[used_col] -= 1
        shim_out_capacity[stage_fill_col] -= 1
        for col in bup_fill_cols:
            shim_out_capacity[col] -= 1
        for col in bdown_fill_cols:
            shim_out_capacity[col] -= 1
        preferred_cols = [6, 5, 4, 3, 2, 1, 0, 7]
        candidates = [col for col in preferred_cols if shim_out_capacity[col] > 0]
        if not candidates:
            raise ValueError(
                "No shim output DMA headroom left for LN1 residual refill "
                f"(stage={stage_fill_col}, bup={bup_fill_cols}, bdown={bdown_fill_cols}, "
                f"remaining={shim_out_capacity})"
            )
        return max(
            candidates,
            key=lambda c: (shim_out_capacity[c], -preferred_cols.index(c)),
        )

    ffn_bup_fill_cols = (
        [ffn_bup_mem_tile_cols[chunk[0]] for chunk in branch_split_chunks]
        if b_weight_split_enabled
        else ffn_bup_mem_tile_cols
    )
    ffn_bdown_fill_cols = (
        [ffn_bdown_mem_tile_cols[chunk[0]] for chunk in branch_split_chunks]
        if b_weight_split_enabled
        else ffn_bdown_mem_tile_cols
    )
    ffn_residual_fill_col = _choose_ln1_residual_refill_col(
        ffn_a_stage_mem_tile_cols[0],
        ffn_bup_fill_cols,
        ffn_bdown_fill_cols,
    )
    logging.info("LN1 residual refill shim column: %d", ffn_residual_fill_col)

    inBUp = []
    memBUp = [None] * effective_ffn_branches
    inBDown = []
    memBDown = [None] * effective_ffn_branches
    need_bup_weights = up_compute_enabled
    need_bdown_weights = down_compute_enabled
    ffn_bup_split_mem_tile_cols = []
    ffn_bdown_split_mem_tile_cols = []
    ffn_b_pack_elem_count = emb_tile * emb_tile
    if bup_split_enabled or bdown_split_enabled:
        ffn_weight_split_parent_depth = (
            1 if use_two_way_mha_stream_split else max(2, ffn_weight_fifo_depth)
        )
    else:
        ffn_weight_split_parent_depth = ffn_weight_fifo_depth

    if need_bup_weights:
        if bup_split_enabled:
            for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                chunk_size = len(chunk_branches)
                ffn_b_pack_ty = np.ndarray[
                    (chunk_size * ffn_b_pack_elem_count,),
                    np.dtype[dtype],
                ]
                inBUp.append(
                    ObjectFifo(
                        ffn_b_pack_ty,
                        name="inBUp" if chunk_idx == 0 else f"inBUpPack{chunk_idx}",
                        depth=ffn_weight_split_parent_depth,
                    )
                )
                ffn_bup_split_mem_tile_cols.append(
                    ffn_bup_mem_tile_cols[chunk_branches[0]]
                )
                mem_bup_chunk = (
                    inBUp[-1]
                    .cons()
                    .split(
                        offsets=[
                            local_idx * ffn_b_pack_elem_count
                            for local_idx in range(chunk_size)
                        ],
                        obj_types=[ffn_b_ty] * chunk_size,
                        names=[
                            "memBUp" if branch_idx == 0 else f"memBUp{branch_idx}"
                            for branch_idx in chunk_branches
                        ],
                        dims_to_stream=[b_dims] * chunk_size,
                        depths=[ffn_weight_fifo_depth] * chunk_size,
                        placement=Tile(col=ffn_bup_split_mem_tile_cols[-1], row=1),
                    )
                )
                for local_idx, branch_idx in enumerate(chunk_branches):
                    memBUp[branch_idx] = mem_bup_chunk[local_idx]
        else:
            for branch_idx in range(effective_ffn_branches):
                inBUp.append(
                    ObjectFifo(
                        ffn_b_ty,
                        name="inBUp" if branch_idx == 0 else f"inBUp{branch_idx}",
                        depth=ffn_weight_fifo_depth,
                    )
                )
                memBUp[branch_idx] = (
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

    if need_bdown_weights:
        if bdown_split_enabled:
            for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
                chunk_size = len(chunk_branches)
                ffn_b_pack_ty = np.ndarray[
                    (chunk_size * ffn_b_pack_elem_count,),
                    np.dtype[dtype],
                ]
                inBDown.append(
                    ObjectFifo(
                        ffn_b_pack_ty,
                        name="inBDown" if chunk_idx == 0 else f"inBDownPack{chunk_idx}",
                        depth=ffn_weight_split_parent_depth,
                    )
                )
                ffn_bdown_split_mem_tile_cols.append(
                    ffn_bdown_mem_tile_cols[chunk_branches[0]]
                )
                mem_bdown_chunk = (
                    inBDown[-1]
                    .cons()
                    .split(
                        offsets=[
                            local_idx * ffn_b_pack_elem_count
                            for local_idx in range(chunk_size)
                        ],
                        obj_types=[ffn_b_ty] * chunk_size,
                        names=[
                            "memBDown" if branch_idx == 0 else f"memBDown{branch_idx}"
                            for branch_idx in chunk_branches
                        ],
                        dims_to_stream=[b_dims] * chunk_size,
                        depths=[ffn_weight_fifo_depth] * chunk_size,
                        placement=Tile(col=ffn_bdown_split_mem_tile_cols[-1], row=1),
                    )
                )
                for local_idx, branch_idx in enumerate(chunk_branches):
                    memBDown[branch_idx] = mem_bdown_chunk[local_idx]
        else:
            for branch_idx in range(effective_ffn_branches):
                inBDown.append(
                    ObjectFifo(
                        ffn_b_ty,
                        name="inBDown" if branch_idx == 0 else f"inBDown{branch_idx}",
                        depth=ffn_weight_fifo_depth,
                    )
                )
                memBDown[branch_idx] = (
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
    if (need_bup_weights and any(fifo is None for fifo in memBUp)) or (
        need_bdown_weights and any(fifo is None for fifo in memBDown)
    ):
        raise ValueError("FFN B-weight FIFO assignment is incomplete")
    inBUpProd = [fifo.prod() for fifo in inBUp] if need_bup_weights else []
    inBDownProd = [fifo.prod() for fifo in inBDown] if need_bdown_weights else []

    # FFN internal pipelines and final encoder output
    ffnUpOut = []
    ffnDownPart = [None] * effective_ffn_branches
    ffnDownAccum = [None] * effective_ffn_branches
    ffn_down_worker_tiles = [
        Tile(
            col=selected_down_tiles[branch_idx][0],
            row=selected_down_tiles[branch_idx][1],
        )
        for branch_idx in range(effective_ffn_branches)
    ]
    use_ffn_down_acc_row_store = effective_ffn_branches <= 4 and (
        not stage_ln1_to_ddr or parallel_heads >= 4
    )

    def _allocate_ffn_down_acc_row_store_channels(
        mem_tile_col: int, slot_idx: int
    ) -> tuple[int, int] | None:
        # Keep FFN-down row-store traffic off channels already consumed by
        # LN1 replay and split B-weight ingress on the same memtile.
        # Col5 specifically cannot use channel 0 once LN1 replay is lowered
        # through memtile_row_store there in both memtile and ddr modes.
        channel_pairs_by_col = {
            1: [(4, 4)],
            2: [(4, 4)],
            4: [(2, 2), (3, 3)],
            5: [(2, 2), (4, 4)],
            6: [(2, 2), (3, 3)],
            7: [(0, 0), (1, 1)],
        }
        col_pairs = channel_pairs_by_col.get(mem_tile_col)
        if col_pairs is None or slot_idx >= len(col_pairs):
            return None
        return col_pairs[slot_idx]

    ffn_down_acc_mem_tile_cols = ln1_mode_hooks.adjust_ffn_down_acc_mem_tile_cols(
        ffn_down_acc_mem_tile_cols=list(branch_stage_cols[:effective_ffn_branches]),
        effective_ffn_branches=effective_ffn_branches,
        proj_acc_depth=proj_acc_depth,
        parallel_heads=parallel_heads,
        ln1_replay_mem_tile_col=ln1_replay_mem_tile_col,
    )
    logging.info(
        "FFN down-acc memtile columns: %s",
        ffn_down_acc_mem_tile_cols,
    )
    if ffn_down_acc_grouping_enabled:
        logging.info(
            "FFN down accumulation grouping: ranges=%s stage_branches=%s max_group_size=%d partner_map=%s",
            ffn_down_acc_group_ranges,
            ffn_down_group_stage_branches,
            ffn_down_acc_group_size,
            ffn_down_group_partner_srcs_by_stage,
        )
    ffn_down_acc_row_store_slots_by_col = dict(o_proj_acc_row_store_slots_by_col)
    for branch_idx in range(effective_ffn_branches):
        ffn_up_out_depth = 1 if emb_tile >= 128 else 2
        ffnUpOut.append(
            ObjectFifo(
                o_ty,
                name="ffnUpOut" if branch_idx == 0 else f"ffnUpOut{branch_idx}",
                depth=ffn_up_out_depth,
            )
        )
        if branch_idx not in ffn_down_group_stage_set:
            continue
        # Keep FFN-down accumulation in mem tile FIFO(s) so down-proj core L1 stays
        # within limits while replaying for LN2's two-pass consumption.
        ffn_down_acc_mem_col = ffn_down_acc_mem_tile_cols[branch_idx]
        row_store_slot_idx = ffn_down_acc_row_store_slots_by_col.get(
            ffn_down_acc_mem_col, 0
        )
        ffn_down_row_store_channels = (
            _allocate_ffn_down_acc_row_store_channels(
                ffn_down_acc_mem_col, row_store_slot_idx
            )
            if use_ffn_down_acc_row_store
            else None
        )
        if ffn_down_row_store_channels is not None:
            memtile_ingress_channel, memtile_egress_channel = (
                ffn_down_row_store_channels
            )
            ffnDownPart[branch_idx] = ffnDownAccum[branch_idx] = MemTileRowStore(
                obj_type=o_ty,
                compute_tile=ffn_down_worker_tiles[branch_idx],
                mem_tile=Tile(col=ffn_down_acc_mem_col, row=1),
                part_count=proj_acc_depth,
                buffer_count=2,
                compute_buffer_count=ffn_down_acc_row_store_compute_buffer_count,
                compute_produce_buffer_count=(
                    ffn_down_acc_row_store_compute_produce_buffer_count
                ),
                compute_consume_buffer_count=(
                    ffn_down_acc_row_store_compute_consume_buffer_count
                ),
                name="ffnDownAccum" if branch_idx == 0 else f"ffnDownAccum{branch_idx}",
                compute_mm2s_channel=0,
                compute_s2mm_channel=0,
                memtile_ingress_channel=memtile_ingress_channel,
                memtile_egress_channel=memtile_egress_channel,
            )
            ffn_down_acc_row_store_slots_by_col[ffn_down_acc_mem_col] = (
                row_store_slot_idx + 1
            )
        else:
            ffnDownPart[branch_idx] = ObjectFifo(
                o_ty,
                name="ffnDownPart" if branch_idx == 0 else f"ffnDownPart{branch_idx}",
                depth=1,
            )
            ffnDownAccum[branch_idx] = (
                ffnDownPart[branch_idx]
                .cons(depth=proj_acc_depth)
                .forward(
                    obj_type=o_ty,
                    name=(
                        "ffnDownAccum"
                        if branch_idx == 0
                        else f"ffnDownAccum{branch_idx}"
                    ),
                    depth=proj_acc_depth,
                    placement=Tile(col=ffn_down_acc_mem_col, row=1),
                )
            )
    # Large emb_tile (128) can overflow FFN-down L1 when reduction FIFOs are
    # double-buffered alongside B-down and accumulation buffers.
    ffn_down_reduce_depth = 1 if emb_tile >= 128 else 2
    stage_ffn_reduce_via_memtile = False
    ffnDownGroupPartial = [None] * effective_ffn_branches
    for src_branch_idx, _stage_branch_idx in ffn_down_group_stage_by_src.items():
        ffnDownGroupPartial[src_branch_idx] = ObjectFifo(
            o_ty,
            name=f"ffnDownGroup{src_branch_idx}",
            depth=ffn_down_reduce_depth,
        )
    ffnDownReduce = []
    ffnDownReduceSrc = []
    ffnDownReducePart = []
    ffn_reduction_link_idx_by_src = {
        src_branch_idx: link_idx
        for link_idx, src_branch_idx in enumerate(ffn_reduction_sources)
    }
    reduce_link_count = len(ffn_reduction_sources)
    if stage_ffn_reduce_via_memtile and reduce_link_count > 0:
        # One staged reduction link per configured reduction edge.
        # Place each link on the destination branch memtile column so the
        # consumer-side hop is local and producer->memtile provides decoupling.
        ffn_down_reduce_mem_tile_col_by_src = {}
        for src_branch_idx in ffn_reduction_sources:
            dst_branch_idx = ffn_reduce_dst_by_src[src_branch_idx]
            if dst_branch_idx >= len(ffn_down_acc_mem_tile_cols):
                raise ValueError(
                    "Unable to assign memtile columns for FFN reduction links: "
                    f"source={src_branch_idx} destination={dst_branch_idx} "
                    f"acc_cols={len(ffn_down_acc_mem_tile_cols)}"
                )
            ffn_down_reduce_mem_tile_col_by_src[src_branch_idx] = (
                ffn_down_acc_mem_tile_cols[dst_branch_idx]
            )
    for src_branch_idx in ffn_reduction_sources:
        if stage_ffn_reduce_via_memtile:
            ffnDownReducePart.append(
                ObjectFifo(
                    o_ty,
                    name=f"ffnDownReduce{src_branch_idx}Part",
                    depth=1,
                )
            )
            link_idx = len(ffnDownReducePart) - 1
            ffnDownReduce.append(
                ffnDownReducePart[link_idx]
                .cons(depth=ffn_down_reduce_depth)
                .forward(
                    obj_type=o_ty,
                    name=f"ffnDownReduce{src_branch_idx}",
                    depth=ffn_down_reduce_depth,
                    placement=Tile(
                        col=ffn_down_reduce_mem_tile_col_by_src[src_branch_idx], row=1
                    ),
                )
            )
            ffnDownReduceSrc.append(ffnDownReducePart[link_idx])
        else:
            link = ObjectFifo(
                o_ty,
                name=f"ffnDownReduce{src_branch_idx}",
                depth=ffn_down_reduce_depth,
            )
            ffnDownReduce.append(link)
            ffnDownReduceSrc.append(link)
    ffn_down_out_depth = 1 if emb_tile >= 128 else 2
    ffnDownOut = []
    ffnDownOutStageToDDR = None
    ffnDownInFromDDR = None
    ffnDownInFromDDRMem = None
    for stream_idx, _branch_idx in enumerate(ffn_down_output_branches):
        if (
            staged_ffn_down_stream_idx is not None
            and stream_idx == staged_ffn_down_stream_idx
        ):
            # Route this FFN-down output stream through shim/DDR to break
            # on-chip backpressure cycles when >2 branches are active.
            ffnDownOutPart = ObjectFifo(
                o_ty,
                name=(
                    "ffnDownOut"
                    if len(ffn_down_output_branches) == 1 and stream_idx == 0
                    else f"ffnDownOut{stream_idx}Part"
                ),
                depth=ffn_down_out_depth,
            )
            ffnDownOut.append(ffnDownOutPart)
            ffnDownOutStageToDDR = ffnDownOutPart.cons(
                depth=ffn_down_out_depth
            ).forward(
                obj_type=o_ty,
                name=f"ffnDownOut{stream_idx}",
                depth=ffn_down_out_depth,
                placement=Tile(col=ffn_down_ddr_stage_col, row=1),
            )
            ffnDownInFromDDR = ObjectFifo(
                o_ty,
                name=f"ffnDownInFromDDR{stream_idx}",
                depth=ffn_down_out_depth,
            )
            ffnDownInFromDDRMem = ffnDownInFromDDR.cons(
                depth=ffn_down_out_depth
            ).forward(
                obj_type=o_ty,
                name=f"ffnDownInFromDDRMem{stream_idx}",
                depth=ffn_down_out_depth,
                placement=Tile(col=ffn_down_ddr_stage_col, row=1),
            )
        else:
            ffnDownOut.append(
                ObjectFifo(
                    o_ty,
                    name=(
                        "ffnDownOut"
                        if len(ffn_down_output_branches) == 1 and stream_idx == 0
                        else f"ffnDownOut{stream_idx}"
                    ),
                    depth=ffn_down_out_depth,
                )
            )
    ln2ReplayPart = None
    ln2Replay = None
    ln2_replay_curr = None
    ln2_replay_new = None
    ln2_tile_obj = Tile(col=ln2_tile[0], row=ln2_tile[1])
    # The current LN2 row-store lowering is stable on the broader 4/6-head
    # encoder topologies with <=4 FFN branches. The 6-head and 6-FFN-branch
    # layouts still regress with the current compiler/runtime path, so keep
    # those on the forwarded FIFO path for now.
    use_ln2_row_store = (
        use_ln2_replay_store
        and (not stage_ln1_to_ddr)
        and parallel_heads == 4
        and effective_ffn_branches <= 4
    )
    if use_ln2_row_store:
        ln2Replay = MemTileRowStore(
            obj_type=o_ty,
            compute_tile=ln2_tile_obj,
            mem_tile=Tile(col=ln2_replay_mem_tile_col, row=1),
            part_count=proj_acc_depth,
            buffer_count=2,
            compute_buffer_count=ln2_replay_row_store_compute_buffer_count,
            compute_produce_buffer_count=(
                ln2_replay_row_store_compute_produce_buffer_count
            ),
            compute_consume_buffer_count=(
                ln2_replay_row_store_compute_consume_buffer_count
            ),
            name="ln2Replay",
        )
        ln2_replay_curr = ln2Replay.cons(depth=1)
        ln2_replay_new = ln2Replay.prod()
    elif use_ln2_replay_store:
        # Keep DDR mode on the forwarded FIFO path until the row-store verifier
        # accepts the current AddNorm2 placement there.
        ln2ReplayPart = ObjectFifo(o_ty, name="ln2ReplayPart", depth=1)
        ln2Replay = ln2ReplayPart.cons(depth=proj_acc_depth).forward(
            obj_type=o_ty,
            name="ln2Replay",
            depth=proj_acc_depth,
            placement=Tile(col=ln2_replay_mem_tile_col, row=1),
        )
        ln2_replay_curr = ln2Replay.cons(depth=1)
        ln2_replay_new = ln2ReplayPart.prod()
    outLN2 = ObjectFifo(o_ty, name="outLN2", depth=ln2_output_fifo_depth)
    memLN2 = outLN2.cons().forward(
        obj_type=o_ty,
        name="memLN2",
        dims_to_stream=o_dims,
        depth=ln2_output_fifo_depth,
        placement=Tile(col=ln_mem_tile_col, row=1),
    )

    def batched_matmul_qk(
        of_q,
        of_k,
        of_a_out,
        zero,
        matmul_QK,
        q_block_bias,
        idx_buffer,
        compute_enabled,
    ):

        for _ in range_(sys.maxsize):

            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                elem_in_q = of_q.acquire(1)

                for _ in range_(num_kv_seq_blocks):

                    elem_in_k = of_k.acquire(1)
                    elem_a_out = of_a_out.acquire(1)

                    if compute_enabled:
                        zero(elem_a_out)
                        matmul_QK(elem_in_q, elem_in_k, elem_a_out, idx_buffer)
                    else:
                        zero(elem_a_out)

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
        zero_qk,
        q_block_bias,
        idx_buffer,
        scale_buffer,
        scale_passthrough_buffer,
        compute_enabled,
        emit_scale,
    ):

        for _ in range_(sys.maxsize):

            idx_buffer[0] = 0
            idx_buffer[1] = 0

            for _ in range_(num_qkv_head_block_per_parallel_head):

                init_scale_buffer(scale_buffer, seq_tile)

                for _ in range_(num_kv_seq_blocks):

                    elt_of_out_p = of_out_p.acquire(1)
                    elt_of_in_a = of_in_a.acquire(1)
                    elt_of_out_scale = of_out_scale.acquire(1) if emit_scale else None

                    if compute_enabled:
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
                    else:
                        zero_qk(elt_of_out_p)
                    if emit_scale:
                        if compute_enabled:
                            memcopy_kernel_scale(
                                scale_buffer, elt_of_out_scale, 4 * seq_tile
                            )
                        else:
                            memcopy_kernel_scale(
                                scale_passthrough_buffer,
                                elt_of_out_scale,
                                4 * seq_tile,
                            )

                    of_in_a.release(1)
                    of_out_p.release(1)
                    if emit_scale:
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
        compute_enabled,
    ):

        for _ in range_(sys.maxsize):

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
                elt_of_out_scale = of_scale.acquire(1) if compute_enabled else None

                if compute_enabled:
                    matmul_PV(
                        elem_in_p,
                        elem_in_v,
                        elem_o_out,
                        elt_of_out_scale,
                        seq_tile,
                        0,
                        idx_buffer,
                    )
                else:
                    zero(elem_o_out)

                of_p.release(1)
                of_v.release(1)
                if compute_enabled:
                    of_scale.release(1)

                idx_buffer[0] += 0
                ###

                if num_kv_seq_blocks > 2:
                    for _ in range_(num_kv_seq_blocks - 2):
                        elem_in_p = of_p.acquire(1)
                        elem_in_v = of_v.acquire(1)
                        elt_of_out_scale2 = (
                            of_scale.acquire(1) if compute_enabled else None
                        )

                        if compute_enabled:
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
                        if compute_enabled:
                            of_scale.release(1)

                        idx_buffer[0] += 0

                ### Last iteration, final rescaling
                if num_kv_seq_blocks > 1:
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elt_of_out_scale3 = of_scale.acquire(1) if compute_enabled else None

                    if compute_enabled:
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
                    if compute_enabled:
                        of_scale.release(1)

                    idx_buffer[0] += 0
                # else:
                else:
                    if compute_enabled:
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
        of_o_row_store_parts,
        buffer_to_reduce,
        group_reduce_in,
        group_reduce_out,
        partial_o_scratch,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        zero,
        matmul_init,
        matmul,
        add,
        copy,
        is_group_staging_core,
        emit_final_output,
        emit_ln1_stats,
        use_grouped_chain,
        compute_enabled,
        core_idx,
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
            if not compute_enabled:
                if emit_ln1_stats:
                    zero_f32(stats_sum_buf, seq_tile)
                    zero_f32(stats_sumsq_buf, seq_tile)
                if not use_grouped_chain:
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
                            zero(elem_out_o_acc)
                            of_o_acc_out.release(1)
                            of_ow_in.release(1)
                            of_o_acc_in.release(1)
                        of_o_in.release(1)

                    tile_indices = (
                        range(proj_acc_depth)
                        if of_o_row_store_parts is not None
                        else range_(proj_acc_depth)
                    )
                    for tile_idx in tile_indices:
                        elem_in_o_acc = of_o_acc_in.acquire(1)
                        if buffer_to_reduce:
                            partial_o_acc = buffer_to_reduce.acquire(1)
                            buffer_to_reduce.release(1)
                        if of_o_row_store_parts is not None:
                            elem_out_o = of_o_row_store_parts[tile_idx].acquire(1)
                            zero(elem_out_o)
                            of_o_row_store_parts[tile_idx].release(1)
                        else:
                            elem_out_o = of_o_out.acquire(1)
                            zero(elem_out_o)
                            of_o_out.release(1)
                        of_o_acc_in.release(1)
                    if emit_ln1_stats:
                        elem_stats_pkt = of_o_out.acquire(1)
                        pack_stats(
                            stats_sum_buf, stats_sumsq_buf, elem_stats_pkt, seq_tile
                        )
                        of_o_out.release(1)
                else:
                    group_pos = core_idx % o_proj_acc_group_size
                    group_idx = core_idx // o_proj_acc_group_size
                    local_has_input = group_pos > 0
                    local_has_output = group_pos < (o_proj_acc_group_size - 1)
                    global_has_input = group_idx > 0
                    global_has_output = group_reduce_out is not None
                    if is_group_staging_core:
                        for _ in range_(proj_acc_depth):
                            elem_out_o_acc = of_o_acc_out.acquire(1)
                            zero(elem_out_o_acc)
                            of_o_acc_out.release(1)

                    for _ in range_(num_qkv_head_block_per_parallel_head):
                        elem_in_o = of_o_in.acquire(1)
                        for _ in range_(proj_acc_depth):
                            elem_in_ow = of_ow_in.acquire(1)
                            if local_has_input:
                                elem_group = group_reduce_in.acquire(1)
                                group_reduce_in.release(1)
                            if is_group_staging_core:
                                elem_in_o_acc = of_o_acc_in.acquire(1)
                                elem_out_o_acc = of_o_acc_out.acquire(1)
                                zero(elem_out_o_acc)
                                of_o_acc_out.release(1)
                                of_o_acc_in.release(1)
                            elif local_has_output:
                                elem_group_out = group_reduce_out.acquire(1)
                                zero(elem_group_out)
                                group_reduce_out.release(1)
                            of_ow_in.release(1)
                        of_o_in.release(1)

                    tile_indices = (
                        range(proj_acc_depth)
                        if of_o_row_store_parts is not None
                        else range_(proj_acc_depth)
                    )
                    for tile_idx in tile_indices:
                        if is_group_staging_core:
                            elem_in_o_acc = of_o_acc_in.acquire(1)
                            if global_has_input:
                                elem_group = group_reduce_in.acquire(1)
                                group_reduce_in.release(1)
                            if emit_final_output:
                                if of_o_row_store_parts is not None:
                                    elem_out_o = of_o_row_store_parts[tile_idx].acquire(
                                        1
                                    )
                                    zero(elem_out_o)
                                    of_o_row_store_parts[tile_idx].release(1)
                                else:
                                    elem_out_o = of_o_out.acquire(1)
                                    zero(elem_out_o)
                                    of_o_out.release(1)
                            else:
                                elem_group_out = group_reduce_out.acquire(1)
                                zero(elem_group_out)
                                group_reduce_out.release(1)
                            of_o_acc_in.release(1)
                        elif global_has_input:
                            elem_group = group_reduce_in.acquire(1)
                            elem_group_out = group_reduce_out.acquire(1)
                            zero(elem_group_out)
                            group_reduce_out.release(1)
                            group_reduce_in.release(1)
                    if emit_ln1_stats:
                        elem_stats_pkt = of_o_out.acquire(1)
                        pack_stats(
                            stats_sum_buf, stats_sumsq_buf, elem_stats_pkt, seq_tile
                        )
                        of_o_out.release(1)
                continue

            if not use_grouped_chain:
                # Baseline O-proj flow: each core accumulates in memtile, then
                # a stage-level chain reduces to the LN1 input.
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

                tile_indices = (
                    range(proj_acc_depth)
                    if of_o_row_store_parts is not None
                    else range_(proj_acc_depth)
                )
                for tile_idx in tile_indices:
                    elem_in_o_acc = of_o_acc_in.acquire(1)
                    if buffer_to_reduce:
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
                    if of_o_row_store_parts is not None:
                        elem_out_o = of_o_row_store_parts[tile_idx].acquire(1)
                        copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                        of_o_row_store_parts[tile_idx].release(1)
                    else:
                        elem_out_o = of_o_out.acquire(1)
                        copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                        of_o_out.release(1)
                    of_o_acc_in.release(1)
                if emit_ln1_stats:
                    elem_stats_pkt = of_o_out.acquire(1)
                    pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats_pkt, seq_tile)
                    of_o_out.release(1)
            else:
                # Grouped flow:
                #   1) reduce within each O-proj group and accumulate on group
                #      boundary cores in memtile;
                #   2) reduce staged group results across groups along the
                #      neighbor chain into the final boundary core.
                group_pos = core_idx % o_proj_acc_group_size
                group_idx = core_idx // o_proj_acc_group_size
                local_has_input = group_pos > 0
                local_has_output = group_pos < (o_proj_acc_group_size - 1)
                global_has_input = group_idx > 0
                global_has_output = group_reduce_out is not None
                if emit_ln1_stats:
                    zero_f32(stats_sum_buf, seq_tile)
                    zero_f32(stats_sumsq_buf, seq_tile)
                if is_group_staging_core:
                    for _ in range_(proj_acc_depth):
                        elem_out_o_acc = of_o_acc_out.acquire(1)
                        zero(elem_out_o_acc)
                        of_o_acc_out.release(1)

                for _ in range_(num_qkv_head_block_per_parallel_head):
                    elem_in_o = of_o_in.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_in_ow = of_ow_in.acquire(1)
                        matmul_init(elem_in_o, elem_in_ow, partial_o_scratch)
                        if local_has_input:
                            if group_reduce_in is None:
                                raise ValueError(
                                    "Grouped O-proj local reduction is missing input FIFO: "
                                    f"core={core_idx} group_pos={group_pos}"
                                )
                            elem_group = group_reduce_in.acquire(1)
                            add(
                                elem_group,
                                partial_o_scratch,
                                partial_o_scratch,
                                seq_tile * emb_tile,
                            )
                            group_reduce_in.release(1)
                        if is_group_staging_core:
                            elem_in_o_acc = of_o_acc_in.acquire(1)
                            elem_out_o_acc = of_o_acc_out.acquire(1)
                            add(
                                elem_in_o_acc,
                                partial_o_scratch,
                                elem_out_o_acc,
                                seq_tile * emb_tile,
                            )
                            of_o_acc_out.release(1)
                            of_o_acc_in.release(1)
                        elif local_has_output:
                            if group_reduce_out is None:
                                raise ValueError(
                                    "Grouped O-proj local reduction is missing output FIFO: "
                                    f"core={core_idx} group_pos={group_pos}"
                                )
                            elem_group_out = group_reduce_out.acquire(1)
                            copy(partial_o_scratch, elem_group_out, seq_tile * emb_tile)
                            group_reduce_out.release(1)
                        of_ow_in.release(1)
                    of_o_in.release(1)

                # Cross-group reduction after local group accumulation.
                # Stage cores provide local staged results, while non-stage
                # cores in later groups pass through global reductions.
                tile_indices = (
                    range(proj_acc_depth)
                    if of_o_row_store_parts is not None
                    else range_(proj_acc_depth)
                )
                for tile_idx in tile_indices:
                    if is_group_staging_core:
                        elem_in_o_acc = of_o_acc_in.acquire(1)
                        if global_has_input:
                            if group_reduce_in is None:
                                raise ValueError(
                                    "Grouped O-proj global reduction is missing input FIFO: "
                                    f"core={core_idx} group_idx={group_idx}"
                                )
                            elem_group = group_reduce_in.acquire(1)
                            add(
                                elem_group,
                                elem_in_o_acc,
                                elem_in_o_acc,
                                seq_tile * emb_tile,
                            )
                            group_reduce_in.release(1)
                        if emit_final_output:
                            if emit_ln1_stats:
                                calc_sum_sumsq(
                                    elem_in_o_acc, stats_sum_buf, stats_sumsq_buf
                                )
                            if of_o_row_store_parts is not None:
                                elem_out_o = of_o_row_store_parts[tile_idx].acquire(1)
                                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                                of_o_row_store_parts[tile_idx].release(1)
                            else:
                                elem_out_o = of_o_out.acquire(1)
                                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                                of_o_out.release(1)
                        else:
                            if not global_has_output:
                                raise ValueError(
                                    "Grouped O-proj global reduction is missing output FIFO: "
                                    f"core={core_idx} group_idx={group_idx}"
                                )
                            elem_group_out = group_reduce_out.acquire(1)
                            copy(elem_in_o_acc, elem_group_out, seq_tile * emb_tile)
                            group_reduce_out.release(1)
                        # Non-final grouped stage cores must still drain their
                        # staged accum tokens each outer iteration; otherwise
                        # the next zero/init pass can deadlock on full FIFO.
                        of_o_acc_in.release(1)
                    elif global_has_input:
                        if group_reduce_in is None or not global_has_output:
                            raise ValueError(
                                "Grouped O-proj pass-through core is missing global reduction FIFO(s): "
                                f"core={core_idx} group_idx={group_idx}"
                            )
                        elem_group = group_reduce_in.acquire(1)
                        elem_group_out = group_reduce_out.acquire(1)
                        copy(elem_group, elem_group_out, seq_tile * emb_tile)
                        group_reduce_out.release(1)
                        group_reduce_in.release(1)
                if emit_ln1_stats:
                    elem_stats_pkt = of_o_out.acquire(1)
                    pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats_pkt, seq_tile)
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
        of_out_residual,
        fused_add_layer_norm,
        fused_layer_norm,
        calc_sum_sumsq,
        unpack_stats,
        zero_f32,
        copy,
        ln_mul_add,
        compute_enabled,
        up_group_count,
        consume_group_count,
    ):
        if consume_group_count < up_group_count:
            raise ValueError(
                "LN1 consume_group_count must be >= up_group_count "
                f"(consume={consume_group_count}, up={up_group_count})"
            )
        use_packed_ln1_stats = compute_enabled and (emb_tile >= 2 * seq_tile)

        for _ in range_(sys.maxsize):
            if compute_enabled and not use_packed_ln1_stats:
                zero_f32(sum_buf, seq_tile)
                zero_f32(sumsq_buf, seq_tile)
            # Pass 1 on raw O-proj output: accumulate row-wise statistics and seed replay FIFO.
            for _ in range_(ln_tiles_per_q_block):
                elem_in = of_in_o_proj.acquire(1)
                if compute_enabled and not use_packed_ln1_stats:
                    calc_sum_sumsq(elem_in, sum_buf, sumsq_buf)
                elem_replay = of_replay_new.acquire(1)
                copy(elem_in, elem_replay, seq_tile * emb_tile)
                of_replay_new.release(1)
                of_in_o_proj.release(1)
            if use_packed_ln1_stats:
                elem_stats_pkt = of_in_o_proj.acquire(1)
                unpack_stats(elem_stats_pkt, sum_buf, sumsq_buf, seq_tile)
                of_in_o_proj.release(1)
            # Pass 2: compute AddNorm1 once for group 0, then replay the
            # post-LN1 output for later groups while still draining repeated
            # residual tokens to preserve the existing runtime fill contract.
            # Group 0: consume raw O-proj replay and produce the stage output.
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_replay_curr.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                elem_out_up = of_out_up.acquire(1)
                if compute_enabled:
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
                else:
                    copy(elem_in, elem_out_up, seq_tile * emb_tile)
                if of_out_residual:
                    elem_out_res = of_out_residual.acquire(1)
                    copy(elem_out_up, elem_out_res, seq_tile * emb_tile)
                    of_out_residual.release(1)
                if consume_group_count > 1:
                    elem_replay = of_replay_new.acquire(1)
                    copy(elem_out_up, elem_replay, seq_tile * emb_tile)
                    of_replay_new.release(1)
                of_out_up.release(1)
                of_replay_curr.release(1)
                of_in_residual.release(1)

            # Additional broadcast groups: replay the post-LN1 output and only
            # emit to FFN-up for the requested groups.
            for group_idx in range_(consume_group_count - 1):
                group_idx_i32 = index.casts(T.i32(), group_idx)
                emit_up_group = group_idx_i32 < (up_group_count - 1)
                replay_next_group = group_idx_i32 < (consume_group_count - 2)
                for _ in range_(ln_tiles_per_q_block):
                    elem_in = of_replay_curr.acquire(1)
                    elem_residual = of_in_residual.acquire(1)
                    with if_(emit_up_group):
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
        of_in_a_curr,
        of_in_b,
        of_out_c,
        of_out_residual,
        zero,
        matmul_init,
        matmul,
        gelu,
        copy,
        group_count,
        consume_group_count,
        compute_enabled,
        use_init_matmul,
    ):
        if consume_group_count < group_count:
            raise ValueError(
                "FFN-up consume_group_count must be >= group_count "
                f"(consume={consume_group_count}, group={group_count})"
            )
        for _ in range_(sys.maxsize):
            for group_idx in range_(consume_group_count):
                group_idx_i32 = index.casts(T.i32(), group_idx)
                with if_(group_idx_i32 < group_count) as if_group_produces:
                    elem_out_matmul = of_out_c.acquire(1)
                    if compute_enabled:
                        for acc_idx in range_(proj_acc_depth):
                            elem_in_a = of_in_a_curr.acquire(1)
                            if of_out_residual:
                                with if_(group_idx_i32 == 0):
                                    elem_out_res = of_out_residual.acquire(1)
                                    copy(elem_in_a, elem_out_res, seq_tile * emb_tile)
                                    of_out_residual.release(1)
                            elem_in_b = of_in_b.acquire(1)
                            acc_idx_i32 = index.casts(T.i32(), acc_idx)
                            if use_init_matmul:
                                with if_(acc_idx_i32 == 0) as if_first_acc:
                                    matmul_init(elem_in_a, elem_in_b, elem_out_matmul)
                                with else_(if_first_acc):
                                    matmul(elem_in_a, elem_in_b, elem_out_matmul)
                            else:
                                with if_(acc_idx_i32 == 0):
                                    zero(elem_out_matmul)
                                matmul(elem_in_a, elem_in_b, elem_out_matmul)
                            of_in_b.release(1)
                            of_in_a_curr.release(1)
                    else:
                        zero(elem_out_matmul)
                        for _ in range_(proj_acc_depth):
                            elem_in_a = of_in_a_curr.acquire(1)
                            if of_out_residual:
                                with if_(group_idx_i32 == 0):
                                    elem_out_res = of_out_residual.acquire(1)
                                    copy(elem_in_a, elem_out_res, seq_tile * emb_tile)
                                    of_out_residual.release(1)
                            of_in_a_curr.release(1)
                    if compute_enabled and gelu:
                        gelu(elem_out_matmul, elem_out_matmul, seq_tile * emb_tile)
                    of_out_c.release(1)
                with else_(if_group_produces):
                    for _ in range_(proj_acc_depth):
                        elem_in_a = of_in_a_curr.acquire(1)
                        if of_out_residual:
                            with if_(group_idx_i32 == 0):
                                elem_out_res = of_out_residual.acquire(1)
                                copy(elem_in_a, elem_out_res, seq_tile * emb_tile)
                                of_out_residual.release(1)
                        of_in_a_curr.release(1)

    def core_fn_ffn_down_proj(
        of_in_a,
        of_in_b,
        of_curr_acc,
        of_new_acc,
        of_out,
        zero,
        matmul_init,
        matmul,
        add,
        copy,
        group_count,
        buffer_to_reduce,
        is_final_branch,
        compute_enabled,
        emit_replay_pass,
        of_group_partial_in0,
        partner_group_count0,
        of_group_partial_in1,
        partner_group_count1,
        of_group_partial_out,
    ):
        down_enabled = compute_enabled
        grouped_leaf_mode = of_group_partial_out is not None
        grouped_stage_mode = (
            of_curr_acc is not None
            and of_new_acc is not None
            and (
                (of_group_partial_in0 is not None and partner_group_count0 > 0)
                or (of_group_partial_in1 is not None and partner_group_count1 > 0)
            )
        )
        partner_inputs = [
            (partner_fifo, partner_count)
            for partner_fifo, partner_count in (
                (of_group_partial_in0, partner_group_count0),
                (of_group_partial_in1, partner_group_count1),
            )
            if partner_fifo is not None and partner_count > 0
        ]
        if grouped_stage_mode and any(
            partner_group_count != group_count
            for _partner_fifo, partner_group_count in partner_inputs
        ):
            raise ValueError(
                "Grouped FFN-down stage requires uniform branch group counts: "
                f"group_count={group_count} partner_counts="
                f"{[partner_group_count for _, partner_group_count in partner_inputs]}"
            )

        for _ in range_(sys.maxsize):
            if group_count <= 0:
                continue

            if grouped_leaf_mode:
                for _ in range_(group_count):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_partial = of_group_partial_out.acquire(1)
                        if down_enabled:
                            elem_in_b = of_in_b.acquire(1)
                            matmul_init(elem_in_a, elem_in_b, elem_partial)
                            of_in_b.release(1)
                        else:
                            zero(elem_partial)
                        of_group_partial_out.release(1)
                    of_in_a.release(1)
                continue

            if grouped_stage_mode:
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_new_acc = of_new_acc.acquire(1)
                    if partner_inputs:
                        elem_partial = partner_inputs[0][0].acquire(1)
                        copy(elem_partial, elem_new_acc, seq_tile * emb_tile)
                        partner_inputs[0][0].release(1)
                        for partner_fifo, _partner_group_count in partner_inputs[1:]:
                            elem_partial = partner_fifo.acquire(1)
                            add(
                                elem_partial,
                                elem_new_acc,
                                elem_new_acc,
                                seq_tile * emb_tile,
                            )
                            partner_fifo.release(1)
                    if down_enabled:
                        elem_in_b = of_in_b.acquire(1)
                        if partner_inputs:
                            matmul(elem_in_a, elem_in_b, elem_new_acc, elem_new_acc)
                        else:
                            matmul_init(elem_in_a, elem_in_b, elem_new_acc)
                        of_in_b.release(1)
                    else:
                        zero(elem_new_acc)
                    of_new_acc.release(1)
                of_in_a.release(1)

                for _ in range_(group_count - 1):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        if partner_inputs:
                            elem_partial = partner_inputs[0][0].acquire(1)
                            copy(elem_partial, elem_new_acc, seq_tile * emb_tile)
                            partner_inputs[0][0].release(1)
                            for partner_fifo, _partner_group_count in partner_inputs[
                                1:
                            ]:
                                elem_partial = partner_fifo.acquire(1)
                                add(
                                    elem_partial,
                                    elem_new_acc,
                                    elem_new_acc,
                                    seq_tile * emb_tile,
                                )
                                partner_fifo.release(1)
                        if down_enabled:
                            elem_in_b = of_in_b.acquire(1)
                            if partner_inputs:
                                matmul(elem_in_a, elem_in_b, elem_new_acc, elem_new_acc)
                            else:
                                matmul_init(elem_in_a, elem_in_b, elem_new_acc)
                            of_in_b.release(1)
                        else:
                            zero(elem_new_acc)
                        add(
                            elem_new_acc,
                            elem_curr_acc,
                            elem_new_acc,
                            seq_tile * emb_tile,
                        )
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)
            else:
                # First FFN-down group seeds the accumulator directly.
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_new_acc = of_new_acc.acquire(1)
                    if down_enabled:
                        elem_in_b = of_in_b.acquire(1)
                        matmul_init(elem_in_a, elem_in_b, elem_new_acc)
                        of_in_b.release(1)
                    else:
                        zero(elem_new_acc)
                    of_new_acc.release(1)
                of_in_a.release(1)

                # Remaining groups accumulate onto the current partials.
                for _ in range_(group_count - 1):
                    elem_in_a = of_in_a.acquire(1)
                    for _ in range_(proj_acc_depth):
                        elem_curr_acc = of_curr_acc.acquire(1)
                        elem_new_acc = of_new_acc.acquire(1)
                        if down_enabled:
                            elem_in_b = of_in_b.acquire(1)
                            matmul(elem_in_a, elem_in_b, elem_curr_acc, elem_new_acc)
                            of_in_b.release(1)
                        else:
                            zero(elem_new_acc)
                        of_new_acc.release(1)
                        of_curr_acc.release(1)
                    of_in_a.release(1)

            for _ in range_(proj_acc_depth):
                # Acquire what's in L2, which is the final accumulated result for the tile.
                # Acquire output slot first so we don't consume upstream reduction
                # tokens while downstream (LN2) is backpressured.
                elem_out = of_out.acquire(1)
                elem_curr_acc = of_curr_acc.acquire(1)
                if buffer_to_reduce:
                    partial_acc = buffer_to_reduce.acquire(1)
                    if down_enabled:
                        add(
                            partial_acc,
                            elem_curr_acc,
                            elem_curr_acc,
                            seq_tile * emb_tile,
                        )
                    buffer_to_reduce.release(1)
                if down_enabled:
                    copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                else:
                    zero(elem_out)
                of_curr_acc.release(1)
                if is_final_branch and emit_replay_pass:
                    elem_new_acc = of_new_acc.acquire(1)
                    if down_enabled:
                        # Copy from the output slot after releasing the current-accumulator
                        # slot so FIFO-backed replay does not self-deadlock.
                        copy(elem_out, elem_new_acc, seq_tile * emb_tile)
                    else:
                        zero(elem_new_acc)
                    of_new_acc.release(1)
                of_out.release(1)

            if is_final_branch and emit_replay_pass:
                for _ in range_(proj_acc_depth):
                    elem_curr_acc = of_curr_acc.acquire(1)
                    elem_out = of_out.acquire(1)
                    if down_enabled:
                        copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                    else:
                        zero(elem_out)
                    of_curr_acc.release(1)
                    of_out.release(1)

    def core_fn_add_norm2(
        of_in1,
        of_in1_extra,
        of_in2,
        ffn_merge_buf,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        fused_add_layer_norm,
        calc_sum_sumsq,
        add_vec,
        zero_f32,
        copy,
        compute_enabled,
        verify_stage,
        expect_replay_pass,
        of_replay_curr,
        of_replay_new,
    ):
        replay_from_fifo = of_replay_curr is not None and of_replay_new is not None
        dual_ffn_inputs = of_in1_extra is not None
        if dual_ffn_inputs and ffn_merge_buf is None:
            raise ValueError(
                "AddNorm2 dual-input mode requires a merge buffer for FFN partial sums"
            )
        if expect_replay_pass and replay_from_fifo:
            raise ValueError(
                "AddNorm2 replay source must be either FFN-down replay pass or LN2 replay FIFO"
            )
        for _ in range_(sys.maxsize):
            if not compute_enabled:
                if expect_replay_pass:
                    for _ in range_(proj_acc_depth):
                        elem_in1 = of_in1.acquire(1)
                        if dual_ffn_inputs:
                            elem_in1_extra = of_in1_extra.acquire(1)
                            of_in1_extra.release(1)
                        of_in1.release(1)
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    if dual_ffn_inputs:
                        elem_in1_extra = of_in1_extra.acquire(1)
                        add_vec(
                            elem_in1,
                            elem_in1_extra,
                            ffn_merge_buf,
                            seq_tile * emb_tile,
                        )
                        elem_ffn = ffn_merge_buf
                    else:
                        elem_ffn = elem_in1
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    if verify_stage == STAGE_DOWN_PROJ:
                        copy(elem_ffn, elem_out, seq_tile * emb_tile)
                    else:
                        copy(elem_in2, elem_out, seq_tile * emb_tile)
                    of_out.release(1)
                    if dual_ffn_inputs:
                        of_in1_extra.release(1)
                    of_in1.release(1)
                    of_in2.release(1)
            else:
                if not expect_replay_pass and not replay_from_fifo:
                    raise ValueError(
                        "AddNorm2 normal mode requires FFN-down replay pass or LN2 replay FIFO"
                    )
                zero_f32(sum_buf, seq_tile)
                zero_f32(sumsq_buf, seq_tile)
                for _ in range_(proj_acc_depth):
                    elem_in1 = of_in1.acquire(1)
                    if dual_ffn_inputs:
                        elem_in1_extra = of_in1_extra.acquire(1)
                        add_vec(
                            elem_in1,
                            elem_in1_extra,
                            ffn_merge_buf,
                            seq_tile * emb_tile,
                        )
                        calc_sum_sumsq(ffn_merge_buf, sum_buf, sumsq_buf)
                    else:
                        calc_sum_sumsq(elem_in1, sum_buf, sumsq_buf)
                    if replay_from_fifo:
                        elem_replay = of_replay_new.acquire(1)
                        copy(
                            ffn_merge_buf if dual_ffn_inputs else elem_in1,
                            elem_replay,
                            seq_tile * emb_tile,
                        )
                        of_replay_new.release(1)
                    if dual_ffn_inputs:
                        of_in1_extra.release(1)
                    of_in1.release(1)

                for col_idx in range_(proj_acc_depth):
                    col_i32 = index.casts(T.i32(), col_idx)
                    if replay_from_fifo:
                        elem_ffn = of_replay_curr.acquire(1)
                    else:
                        elem_in1 = of_in1.acquire(1)
                        if dual_ffn_inputs:
                            elem_in1_extra = of_in1_extra.acquire(1)
                            add_vec(
                                elem_in1,
                                elem_in1_extra,
                                ffn_merge_buf,
                                seq_tile * emb_tile,
                            )
                            elem_ffn = ffn_merge_buf
                        else:
                            elem_ffn = elem_in1
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    fused_add_layer_norm(
                        elem_ffn,
                        elem_in2,
                        weights,
                        sum_buf,
                        sumsq_buf,
                        elem_out,
                        embed_sz,
                        col_i32,
                    )
                    of_out.release(1)
                    if replay_from_fifo:
                        of_replay_curr.release(1)
                    else:
                        if dual_ffn_inputs:
                            of_in1_extra.release(1)
                        of_in1.release(1)
                    of_in2.release(1)

    # Create worker from task
    matmul_workers = []
    softmax_workers = []
    matmul_pv_workers = []
    o_proj_workers = []
    build_softmax_workers = True
    build_pv_workers = True
    build_o_proj_workers = True
    build_ln1_worker = direct_verify_stage not in {STAGE_O_PROJ}
    build_ffn_up_workers = direct_verify_stage not in {STAGE_O_PROJ}
    build_ffn_down_workers = direct_verify_stage is None
    build_ln2_worker = direct_verify_stage is None
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
                    qk_compute_enabled,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=2),
                while_true=False,
            )
        )
        qk_input_cons = memA[i].cons() if build_softmax_workers else None
        idx_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_softmax_{i}",
        )
        scale_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(4 * seq_tile,), dtype=dtype),
            name=f"scale_buffer_softmax_{i}",
        )
        scale_passthrough_buffer = Buffer(
            initial_value=np.ones(shape=(4 * seq_tile,), dtype=dtype),
            name=f"scale_passthrough_buffer_{i}",
        )
        if build_softmax_workers:
            softmax_workers.append(
                Worker(
                    softmax,
                    fn_args=[
                        qk_input_cons,
                        memP[i].prod(),
                        scaleOF[i].prod(),
                        partial_softmax_kernel,
                        scale_buffer_init_kernel,
                        memcopy_kernel_scale,
                        zero_kernel,
                        i,
                        idx_buffer_softmax,
                        scale_buffer_softmax,
                        scale_passthrough_buffer,
                        softmax_compute_enabled,
                        pv_compute_enabled,
                    ],
                    stack_size=0xD00,
                    placement=Tile(col=i, row=3),
                    while_true=False,
                )
            )
        else:
            softmax_workers.append(None)
        softmax_input_cons = memP[i].cons() if build_pv_workers else None
        idx_buffer_pv = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_pv_{i}",
        )
        if build_pv_workers:
            matmul_pv_workers.append(
                Worker(
                    batched_matmul_pv,
                    fn_args=[
                        softmax_input_cons,
                        memV[i].cons(),
                        scaleOF[i].cons(),
                        outOProj[i].prod(),
                        zero_kernel,
                        matmul_PV,
                        rescale_O,
                        i,
                        idx_buffer_pv,
                        pv_compute_enabled,
                    ],
                    stack_size=0xD00,
                    placement=Tile(col=i, row=4),
                    while_true=False,
                )
            )
        else:
            matmul_pv_workers.append(None)
        group_reduce_in = (
            outOGroupPart[i - 1].cons()
            if i > 0 and outOGroupPart[i - 1] is not None
            else None
        )
        group_reduce_out = (
            outOGroupPart[i].prod()
            if i < parallel_heads - 1 and outOGroupPart[i] is not None
            else None
        )
        is_group_boundary_core = i in o_proj_stage_order_by_core
        is_group_accum_core = i in o_proj_accum_core_set
        if (
            o_proj_acc_group_size > 1
            and group_reduce_out is None
            and i < parallel_heads - 1
        ):
            raise ValueError(
                "Grouped O-proj core is missing neighbor forwarding FIFO: "
                f"core={i}, group={o_proj_group_idx[i]}"
            )
        if (
            not is_group_boundary_core
            and o_proj_acc_group_size == 1
            and group_reduce_out is None
        ):
            raise ValueError(
                "Non-stage O-proj core is missing neighbor forwarding FIFO: "
                f"core={i}, group={o_proj_group_idx[i]}"
            )
        o_proj_in = outOProj[i].cons() if build_o_proj_workers else None
        o_proj_ow = memOW[i].cons()
        o_proj_row_store_parts = None
        if is_group_boundary_core:
            stage_order = o_proj_stage_order_by_core[i]
            if o_proj_acc_group_size > 1:
                o_proj_acc_in = (
                    outOProjAccumIn[i].cons(depth=1) if is_group_accum_core else None
                )
                o_proj_acc_out = (
                    outOProjAccumOut[i].prod() if is_group_accum_core else None
                )
                o_proj_group_out = group_reduce_out
                o_proj_reduce_in = None
                emit_final_output = i == (parallel_heads - 1)
                o_proj_output = (
                    (
                        oProjVerifyOut.prod()
                        if direct_verify_stage == STAGE_O_PROJ
                        else outOProjInput.prod()
                    )
                    if emit_final_output
                    else None
                )
                emit_ln1_stats = (
                    emit_final_output
                    and ln1_compute_enabled
                    and (emb_tile >= 2 * seq_tile)
                )
            else:
                o_proj_acc_in = outOProjAccumIn[i].cons(depth=1)
                o_proj_acc_out = outOProjAccumOut[i].prod()
                o_proj_group_out = None
                o_proj_reduce_in = (
                    outOPart[stage_order - 1].cons() if stage_order > 0 else None
                )
                emit_final_output = True
                emit_ln1_stats = (
                    stage_order == (num_o_proj_acc_groups - 1)
                    and ln1_compute_enabled
                    and (emb_tile >= 2 * seq_tile)
                )
                if stage_order < num_o_proj_acc_groups - 1:
                    o_proj_output = outOPart[stage_order].prod()
                else:
                    o_proj_output = (
                        oProjVerifyOut.prod()
                        if direct_verify_stage == STAGE_O_PROJ
                        else outOProjInput.prod()
                    )
        else:
            stage_order = -1
            o_proj_acc_in = None
            o_proj_acc_out = None
            o_proj_group_out = group_reduce_out
            o_proj_reduce_in = None
            emit_final_output = False
            emit_ln1_stats = False
            o_proj_output = None
            o_proj_row_store_parts = None
        o_proj_partial_scratch = Buffer(type=o_ty, name=f"o_proj_partial_scratch_{i}")
        o_proj_stats_sum_buffer = Buffer(type=sum_l1_ty, name=f"o_proj_stats_sum_{i}")
        o_proj_stats_sumsq_buffer = Buffer(
            type=sum_l1_ty, name=f"o_proj_stats_sumsq_{i}"
        )
        if build_o_proj_workers:
            o_proj_worker = Worker(
                matmul_o_proj,
                fn_args=[
                    o_proj_in,
                    o_proj_ow,
                    o_proj_acc_in,
                    o_proj_acc_out,
                    o_proj_output,
                    o_proj_row_store_parts,
                    o_proj_reduce_in,
                    group_reduce_in,
                    o_proj_group_out,
                    o_proj_partial_scratch,
                    o_proj_stats_sum_buffer,
                    o_proj_stats_sumsq_buffer,
                    ln_zero_f32_kernel,
                    ln_calc_sum_sumsq_kernel,
                    convert_stats_to_packet_kernel,
                    zero_kernel_o_proj,
                    matmul_init_kernel_o_proj,
                    matmul_kernel_o_proj,
                    eltwise_add_vector,
                    mem_copy_o_proj,
                    is_group_accum_core,
                    emit_final_output,
                    emit_ln1_stats,
                    o_proj_acc_group_size > 1,
                    oproj_compute_enabled,
                    i,
                ],
                stack_size=0xD00,
                placement=o_proj_worker_tiles[i],
                while_true=False,
            )
            if o_proj_row_store_parts is not None:
                for handle in o_proj_row_store_parts:
                    handle.endpoint = o_proj_worker
                    o_proj_worker._fifos.append(handle)
            o_proj_workers.append(o_proj_worker)
        else:
            o_proj_workers.append(None)
        logging.debug(
            "Configured o_proj worker %d with acc_depth=%d "
            "(group=%d pos=%d stage_core=%s stage_order=%d emit_final=%s)",
            i,
            proj_acc_depth,
            o_proj_group_idx[i],
            o_proj_group_pos[i],
            is_group_boundary_core,
            stage_order,
            emit_final_output,
        )

    # Create a single fused LN1 worker (stats + add/norm + replay).
    ln1_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=static_ln1_weights,
        name="static_ln1_weights",
    )
    ln1_norm_sum_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sum_buffer")
    ln1_norm_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sumsq_buffer")
    ln1_input_cons = outOProjInput.cons() if build_ln1_worker else None
    o_proj_verify_cons = (
        oProjVerifyOut.cons() if direct_verify_stage == STAGE_O_PROJ else None
    )
    ln1_output_group_count = ln1_broadcast_groups
    ln1_consume_group_count = ln1_broadcast_groups

    ln1_worker = (
        Worker(
            core_fn_ln1_fused,
            fn_args=[
                ln1_input_cons,
                ln1Replay.cons(),
                ln1Replay.prod(),
                memR.cons(),
                ln1_norm_sum_buffer,
                ln1_norm_sumsq_buffer,
                ln1_weight_buffer,
                ln1Broadcast.prod(),
                None,
                ln_fused_add_layer_norm_kernel,
                ln_fused_layer_norm_kernel,
                ln_calc_sum_sumsq_kernel,
                convert_packet_to_stats_kernel,
                ln_zero_f32_kernel,
                mem_copy_o_proj,
                ln_mul_add_kernel,
                ln1_compute_enabled,
                ln1_output_group_count,
                ln1_consume_group_count,
            ],
            placement=ln1_norm_tile_obj,
            while_true=False,
        )
        if build_ln1_worker
        else None
    )

    ffn_up_workers = []
    ffn_up_output_cons = [None] * effective_ffn_branches
    ffn_up_use_init_matmul = True
    for branch_idx in range(effective_ffn_branches):
        if build_ffn_down_workers:
            ffn_up_output_cons[branch_idx] = ffnUpOut[branch_idx].cons()
        ffn_up_worker_args = [
            memOutLNCons[branch_idx],
            memBUp[branch_idx].cons() if memBUp[branch_idx] is not None else None,
            ffnUpOut[branch_idx].prod(),
            None,
            ffn_zero_kernel_up_proj,
            ffn_matmul_init_kernel_up_proj,
            ffn_matmul_kernel_up_proj,
            ffn_gelu_kernel,
            mem_copy_o_proj,
            ffn_col_group_counts[branch_idx],
            ln1_broadcast_groups,
            up_compute_enabled,
            ffn_up_use_init_matmul,
        ]
        if build_ffn_up_workers:
            ffn_up_workers.append(
                Worker(
                    core_fn_ffn_up_proj,
                    fn_args=ffn_up_worker_args,
                    placement=Tile(
                        col=selected_up_tiles[branch_idx][0],
                        row=selected_up_tiles[branch_idx][1],
                    ),
                    stack_size=0x700,
                    while_true=False,
                )
            )
        else:
            ffn_up_workers.append(None)

    ffn_down_workers = []
    for branch_idx in range(effective_ffn_branches):
        group_partner_srcs = ffn_down_group_partner_srcs_by_stage.get(branch_idx, [])
        group_partner_in0 = (
            ffnDownGroupPartial[group_partner_srcs[0]].cons()
            if len(group_partner_srcs) > 0
            else None
        )
        group_partner_in1 = (
            ffnDownGroupPartial[group_partner_srcs[1]].cons()
            if len(group_partner_srcs) > 1
            else None
        )
        group_partial_out = (
            ffnDownGroupPartial[branch_idx].prod()
            if branch_idx in ffn_down_group_stage_by_src
            else None
        )
        if branch_idx in ffn_down_group_stage_set:
            reduce_in_src_branch = ffn_reduce_in_by_branch.get(branch_idx)
            reduce_in = (
                ffnDownReduce[
                    ffn_reduction_link_idx_by_src[reduce_in_src_branch]
                ].cons()
                if reduce_in_src_branch is not None
                else None
            )
            output_stream_idx = ffn_down_output_stream_idx_by_branch.get(branch_idx)
            is_output_branch = output_stream_idx is not None
            if is_output_branch:
                reduce_out = ffnDownOut[output_stream_idx].prod(ffn_down_out_depth)
            else:
                reduce_out_link_idx = ffn_reduction_link_idx_by_src.get(branch_idx)
                if reduce_out_link_idx is None:
                    raise ValueError(
                        "FFN down stage branch has neither reduction output nor LN2 output stream: "
                        f"branch={branch_idx} edges={ffn_reduction_edges} "
                        f"ln2_output_branches={ffn_down_output_branches}"
                    )
                reduce_out = ffnDownReduceSrc[reduce_out_link_idx].prod()
        else:
            reduce_in = None
            reduce_out = None
            is_output_branch = False
        ffn_down_worker_args = [
            ffn_up_output_cons[branch_idx],
            memBDown[branch_idx].cons() if memBDown[branch_idx] is not None else None,
            (
                ffnDownAccum[branch_idx].cons(depth=1)
                if ffnDownAccum[branch_idx] is not None
                else None
            ),
            (
                ffnDownPart[branch_idx].prod()
                if ffnDownPart[branch_idx] is not None
                else None
            ),
            reduce_out,
            ffn_zero_kernel_down_proj,
            ffn_matmul_init_kernel_down_proj,
            ffn_matmul_kernel_down_proj,
            eltwise_add_vector if effective_ffn_branches > 1 else None,
            mem_copy_o_proj,
            ffn_col_group_counts[branch_idx],
            reduce_in,
            is_output_branch,
            down_compute_enabled,
            emit_ln2_replay_from_down and is_output_branch,
            group_partner_in0,
            (
                ffn_col_group_counts[group_partner_srcs[0]]
                if len(group_partner_srcs) > 0
                else 0
            ),
            group_partner_in1,
            (
                ffn_col_group_counts[group_partner_srcs[1]]
                if len(group_partner_srcs) > 1
                else 0
            ),
            group_partial_out,
        ]
        if build_ffn_down_workers:
            ffn_down_workers.append(
                Worker(
                    core_fn_ffn_down_proj,
                    fn_args=ffn_down_worker_args,
                    placement=ffn_down_worker_tiles[branch_idx],
                    stack_size=0xF00,
                    while_true=False,
                )
            )
        else:
            ffn_down_workers.append(None)

    ln2_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=static_ln2_weights,
        name="static_ln2_weights",
    )
    ln2_sum_buffer = Buffer(type=sum_l1_ty, name="ln2_sum_buffer")
    ln2_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln2_sumsq_buffer")
    if len(ffnDownOut) > 2:
        raise ValueError(
            "FFN reduction produced more than two LN2 input streams; "
            "merge-worker topologies are disabled "
            f"(streams={len(ffnDownOut)}, branches={effective_ffn_branches}, "
            f"edges={ffn_reduction_edges}, outputs={ffn_down_output_branches})"
        )

    def _ln2_ffn_input_cons(stream_idx: int):
        if staged_ffn_down_stream_idx == stream_idx:
            if ffnDownInFromDDRMem is None:
                raise ValueError(
                    "FFN DDR staged stream configured but no refill FIFO was created"
                )
            return ffnDownInFromDDRMem.cons()
        return ffnDownOut[stream_idx].cons()

    ln2_ffn_primary_cons = _ln2_ffn_input_cons(0)
    ln2_ffn_secondary_cons = (
        _ln2_ffn_input_cons(1)
        if use_dual_ln2_ffn_inputs and len(ffnDownOut) > 1
        else None
    )

    ln2_ffn_merge_buffer = (
        Buffer(type=o_ty, name="ln2_ffn_merge_buffer")
        if ln2_ffn_secondary_cons is not None
        else None
    )
    ln2_worker = (
        Worker(
            core_fn_add_norm2,
            fn_args=[
                ln2_ffn_primary_cons,
                ln2_ffn_secondary_cons,
                ffnRIn.cons(),
                ln2_ffn_merge_buffer,
                ln2_sum_buffer,
                ln2_sumsq_buffer,
                ln2_weight_buffer,
                outLN2.prod(),
                ln_fused_add_layer_norm_kernel,
                ln_calc_sum_sumsq_kernel,
                eltwise_add_vector if ln2_ffn_secondary_cons is not None else None,
                ln_zero_f32_kernel,
                mem_copy_o_proj,
                ln2_compute_enabled,
                verify_stage,
                emit_ln2_replay_from_down,
                ln2_replay_curr,
                ln2_replay_new,
            ],
            placement=ln2_tile_obj,
            stack_size=0xF00,
            while_true=False,
        )
        if build_ln2_worker
        else None
    )

    # Define tensor access patterns for inputs/outputs.
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

    # WO remains tiled by emb_tile rather than emb_tile * proj_acc_depth.
    WO_tiles = TensorTiler2D.group_tiler(
        (embed_sz, embed_sz),
        (d, emb_tile),
        (parallel_heads, embed_sz // emb_tile // num_o_col_groups),
    )
    # Flip the first two dimensions so splitting iterates over rows first.
    for tile in WO_tiles:
        tile._sizes = [tile._sizes[1], tile._sizes[0], tile._sizes[2], tile._sizes[3]]
        tile._strides = [
            tile._strides[1],
            tile._strides[0],
            tile._strides[2],
            tile._strides[3],
        ]

    def split_tap_along_axis(
        tap: TensorAccessPattern,
        tensor_shape: tuple[int, int],
        axis_idx: int,
        parts: int,
    ) -> list[TensorAccessPattern]:
        if parts <= 1:
            return [tap]
        sizes = [int(v) for v in tap.sizes]
        strides = [int(v) for v in tap.strides]
        axis_size = sizes[axis_idx]
        if axis_size % parts != 0:
            raise ValueError(
                "Cannot split tap axis evenly: "
                f"axis_size={axis_size}, parts={parts}, tap={tap}"
            )
        chunk = axis_size // parts
        taps = []
        for part_idx in range(parts):
            sizes_part = list(sizes)
            sizes_part[axis_idx] = chunk
            taps.append(
                TensorAccessPattern(
                    tensor_shape,
                    offset=int(tap.offset) + part_idx * chunk * strides[axis_idx],
                    sizes=sizes_part,
                    strides=strides,
                )
            )
        return taps

    Q_tiles_by_stream = [[] for _ in range(mha_stream_split_factor)]
    K_tiles_by_stream = [[] for _ in range(mha_stream_split_factor)]
    V_tiles_by_stream = [[] for _ in range(mha_stream_split_factor)]
    WO_tiles_by_stream = [[] for _ in range(mha_stream_split_factor)]
    for q_tap in Q_tiles:
        q_parts = split_tap_along_axis(
            q_tap, qkv_tensor_shape, axis_idx=1, parts=mha_stream_split_factor
        )
        for split_idx in range(mha_stream_split_factor):
            Q_tiles_by_stream[split_idx].append(q_parts[split_idx])
    for k_tap in K_tiles:
        k_parts = split_tap_along_axis(
            k_tap, qkv_tensor_shape, axis_idx=1, parts=mha_stream_split_factor
        )
        for split_idx in range(mha_stream_split_factor):
            K_tiles_by_stream[split_idx].append(k_parts[split_idx])
    for v_tap in V_tiles:
        v_parts = split_tap_along_axis(
            v_tap, qkv_tensor_shape, axis_idx=1, parts=mha_stream_split_factor
        )
        for split_idx in range(mha_stream_split_factor):
            V_tiles_by_stream[split_idx].append(v_parts[split_idx])
    for wo_tap in WO_tiles:
        wo_parts = split_tap_along_axis(
            wo_tap, (embed_sz, embed_sz), axis_idx=0, parts=mha_stream_split_factor
        )
        for split_idx in range(mha_stream_split_factor):
            WO_tiles_by_stream[split_idx].append(wo_parts[split_idx])

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
    or_tensor_shape = (2 * seq_len + ln1_dram_stage_rows, embed_sz)
    O_tiles = retarget_tas(o_tiles_base, or_tensor_shape, offset_delta=0)
    R_tiles = retarget_tas(
        r_tiles_base, or_tensor_shape, offset_delta=seq_len * embed_sz
    )
    for tile in R_tiles:
        if len(tile._sizes) < 1:
            raise ValueError(f"Unexpected R tile rank for replay: {tile._sizes}")
        tile._sizes[0] = ln1_broadcast_groups
        tile._strides[0] = 0
    up_verify_tensor_shape = (seq_len, ffn_intermediate_size)

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
    B_Up_split_tiles = []
    B_Down_split_tiles = []
    if b_weight_split_enabled:
        for chunk_branches in branch_split_chunks:
            chunk_size = len(chunk_branches)
            first_branch_idx = chunk_branches[0]
            first_group_offset = ffn_col_group_offsets[first_branch_idx]
            shared_group_count = ffn_col_group_counts[first_branch_idx]
            if any(
                ffn_col_group_counts[branch_idx] != shared_group_count
                for branch_idx in chunk_branches
            ):
                raise ValueError(
                    "B-weight split chunk requires uniform group count within each chunk "
                    f"(counts={ffn_col_group_counts}, chunk={chunk_branches})"
                )
            b_up_chunk_taps = []
            b_down_chunk_taps = []
            for col_group in range(num_o_col_groups):
                for local_group_idx in range(shared_group_count):
                    b_up_chunk_taps.append(
                        TensorAccessPattern(
                            (embed_sz, ffn_intermediate_size),
                            offset=(
                                col_group
                                * (proj_acc_depth * emb_tile * ffn_intermediate_size)
                                + (first_group_offset + local_group_idx) * emb_tile
                            ),
                            sizes=[proj_acc_depth, chunk_size, emb_tile, emb_tile],
                            strides=[
                                emb_tile * ffn_intermediate_size,
                                shared_group_count * emb_tile,
                                ffn_intermediate_size,
                                1,
                            ],
                        )
                    )
                    b_down_chunk_taps.append(
                        TensorAccessPattern(
                            (ffn_intermediate_size, embed_sz),
                            offset=(
                                col_group * (proj_acc_depth * emb_tile)
                                + (first_group_offset + local_group_idx)
                                * emb_tile
                                * embed_sz
                            ),
                            sizes=[proj_acc_depth, chunk_size, emb_tile, emb_tile],
                            strides=[
                                emb_tile,
                                shared_group_count * emb_tile * embed_sz,
                                embed_sz,
                                1,
                            ],
                        )
                    )
            B_Up_split_tiles.append(TensorAccessSequence.from_taps(b_up_chunk_taps))
            B_Down_split_tiles.append(TensorAccessSequence.from_taps(b_down_chunk_taps))

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
    if b_weight_split_enabled:
        for chunk_idx in range(len(branch_split_chunks)):
            legalize_tas(B_Up_split_tiles[chunk_idx])
            legalize_tas(B_Down_split_tiles[chunk_idx])

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
        ln1_broadcast_groups * proj_acc_depth,
        "R",
    )
    assert_all_taps_match_count(
        O_tiles,
        (seq_tile, emb_tile),
        proj_acc_depth,
        "O",
    )
    if b_weight_split_enabled:
        for chunk_idx, chunk_branches in enumerate(branch_split_chunks):
            chunk_size = len(chunk_branches)
            chunk_group_count = ffn_col_group_counts[chunk_branches[0]]
            expected_split_tap_count = num_o_col_groups * chunk_group_count
            if len(B_Up_split_tiles[chunk_idx]) != expected_split_tap_count:
                raise ValueError(
                    "B_Up split tap count mismatch for chunk "
                    f"{chunk_idx}: have={len(B_Up_split_tiles[chunk_idx])} "
                    f"expected={expected_split_tap_count}"
                )
            if len(B_Down_split_tiles[chunk_idx]) != expected_split_tap_count:
                raise ValueError(
                    "B_Down split tap count mismatch for chunk "
                    f"{chunk_idx}: have={len(B_Down_split_tiles[chunk_idx])} "
                    f"expected={expected_split_tap_count}"
                )
            assert_all_taps_match_count(
                B_Up_split_tiles[chunk_idx],
                (chunk_size * emb_tile * emb_tile,),
                proj_acc_depth,
                f"B_Up_split[{chunk_idx}]",
            )
            assert_all_taps_match_count(
                B_Down_split_tiles[chunk_idx],
                (chunk_size * emb_tile * emb_tile,),
                proj_acc_depth,
                f"B_Down_split[{chunk_idx}]",
            )
    else:
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
        # Default to relaxed FFN-weight waits to improve overlap while keeping
        # residual/output waits serialized unless explicitly overridden.
        serialize_tail_io = (
            True if runtime_serialize_tail_io is None else runtime_serialize_tail_io
        )
        serialize_q_prestage = serialize_tail_io
        if runtime_serialize_q_prestage is not None:
            serialize_q_prestage = runtime_serialize_q_prestage
        if not serialize_tail_io:
            serialize_q_prestage = False
        if b_weight_split_enabled and effective_ffn_branches > 2:
            serialize_q_prestage = False
        ffn_down_ddr_stage_enabled = (
            staged_ffn_down_stream_idx is not None
            and ffnDownOutStageToDDR is not None
            and ffnDownInFromDDR is not None
        )
        runtime_state = ln1_mode_hooks.build_runtime_state(
            effective_ffn_branches=effective_ffn_branches,
            ln1OutStageToDDR=ln1OutStageToDDR,
            ln1InFromDDR=ln1InFromDDR,
            ln1_ddr_staged_branch_indices=ln1_ddr_staged_branch_indices,
        )
        ffn_down_ddr_stage_passes = 2 if emit_ln2_replay_from_down else 1
        tail_wait_mode = runtime_tail_wait_mode.strip().lower()
        wait_residual_fill = serialize_tail_io
        wait_ffn_weight_fill = serialize_tail_io
        wait_output_drain = serialize_tail_io
        wait_ffn_weight_fill = ln1_mode_hooks.adjust_wait_ffn_weight_fill(
            wait_ffn_weight_fill=wait_ffn_weight_fill,
            runtime_state=runtime_state,
            use_ln1_broadcast=use_ln1_broadcast,
            effective_ffn_branches=effective_ffn_branches,
        )
        bup_broadcast_priming = ln1_mode_hooks.use_bup_broadcast_priming(
            use_ln1_broadcast=use_ln1_broadcast,
            effective_ffn_branches=effective_ffn_branches,
            runtime_state=runtime_state,
        )
        # For wider FFN branch counts, prioritize downstream reduction-chain
        # branches when issuing B_Down fills so consumers are ready before
        # upstream branches push partial sums.
        fill_bdown_reverse = effective_ffn_branches > 2
        fill_bup_reverse = False
        # Decoupling tail fills (R/B_Up/B_Down) from output drains can help
        # in wider-branch topologies, but adds runtime scheduling overhead for
        # small branch counts. Keep it enabled by default only when >2 branches.
        decouple_tail_fill = effective_ffn_branches > 2
        if b_weight_split_enabled and effective_ffn_branches > 2:
            decouple_tail_fill = False
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
                    "runtime_tail_wait_mode must be one of "
                    "{strict, relax_ffn_weights, relax_ffn_weights_residual, "
                    "relax_all_tail} "
                    f"(got '{tail_wait_mode}')"
                )
        # Split B-stream topologies rely on deterministic per-iteration
        # completion of FFN-weight fills; relaxing these waits can accumulate
        # cross-iteration backpressure and lead to runtime timeouts.
        if b_weight_split_enabled and use_ln1_broadcast and effective_ffn_branches > 1:
            wait_ffn_weight_fill = True
        if stage_ln1_to_ddr:
            wait_residual_fill = False
        residual_fill_max_groups = min(ln1_broadcast_groups, 22)
        b_weight_fill_max_groups_default = min(max(ffn_col_group_counts), 22)
        # For low-head 6-way split-B tails, avoid splitting each packed B_Down
        # tap into multiple DMA BDs (e.g. 8 -> 4+4 groups). Keeping one fill
        # per packed tap reduces runtime-sequence BD pressure.
        if (
            bdown_split_enabled
            and effective_ffn_branches >= 6
            and parallel_heads <= 4
            and proj_acc_depth >= 6
        ):
            max_chunk_groups = max(
                ffn_col_group_counts[chunk[0]] * len(chunk)
                for chunk in branch_split_chunks
            )
            b_weight_fill_max_groups_default = min(
                22, max(b_weight_fill_max_groups_default, max_chunk_groups)
            )
        # For low-pressure topologies, avoid fragmenting each B-weight tap into
        # multiple fills; this reduces runtime fill scheduling overhead.
        if (
            not b_weight_split_enabled
            and effective_ffn_branches <= 2
            and parallel_heads <= 2
        ):
            b_weight_fill_max_groups_default = max(ffn_col_group_counts)
        b_weight_fill_max_groups = b_weight_fill_max_groups_default

        def split_tap_dim0(
            tap: TensorAccessPattern,
            tensor_shape: tuple[int, int],
            max_groups: int,
        ) -> list[TensorAccessPattern]:
            sizes = [int(v) for v in tap.sizes]
            strides = [int(v) for v in tap.strides]
            if not sizes or sizes[0] <= max_groups:
                return [tap]
            parts = []
            remaining = sizes[0]
            offset = int(tap.offset)
            while remaining > 0:
                chunk = min(max_groups, remaining)
                parts.append(
                    TensorAccessPattern(
                        tensor_shape,
                        offset=offset,
                        sizes=[chunk, *sizes[1:]],
                        strides=strides,
                    )
                )
                if strides[0] != 0:
                    offset += chunk * strides[0]
                remaining -= chunk
            return parts

        def split_tap_dim0_prefix_rest(
            tap: TensorAccessPattern,
            tensor_shape: tuple[int, int],
            prefix_groups: int,
        ) -> tuple[TensorAccessPattern | None, TensorAccessPattern | None]:
            sizes = [int(v) for v in tap.sizes]
            strides = [int(v) for v in tap.strides]
            if not sizes or sizes[0] <= 0:
                return None, None
            prefix = min(prefix_groups, sizes[0])
            prefix_tap = TensorAccessPattern(
                tensor_shape,
                offset=int(tap.offset),
                sizes=[prefix, *sizes[1:]],
                strides=strides,
            )
            remaining = sizes[0] - prefix
            if remaining <= 0:
                return prefix_tap, None
            rest_offset = int(tap.offset)
            if strides[0] != 0:
                rest_offset += prefix * strides[0]
            rest_tap = TensorAccessPattern(
                tensor_shape,
                offset=rest_offset,
                sizes=[remaining, *sizes[1:]],
                strides=strides,
            )
            return prefix_tap, rest_tap

        def schedule_residual_fill(task_group, tap_idx):
            r_tap = R_tiles[tap_idx]
            for r_tap_part in split_tap_dim0(
                r_tap, or_tensor_shape, residual_fill_max_groups
            ):
                rt.fill(
                    inR.prod(),
                    OR,
                    tap=r_tap_part,
                    placement=Tile(col=ln_mem_tile_col, row=0),
                    task_group=task_group,
                    wait=wait_residual_fill,
                )

        def schedule_ffn_weight_fills(task_group, col_group_idx):
            # Fill all B_Up streams first so every FFN-up branch can start
            # before B_Down transfers contend for shim/memtile bandwidth.
            schedule_bup = up_compute_enabled
            schedule_bdown = down_compute_enabled
            if schedule_bup and bup_split_enabled:
                bup_chunk_order = (
                    list(range(len(branch_split_chunks) - 1, -1, -1))
                    if fill_bup_reverse
                    else list(range(len(branch_split_chunks)))
                )
                for chunk_idx in bup_chunk_order:
                    chunk_group_count = ffn_col_group_counts[
                        branch_split_chunks[chunk_idx][0]
                    ]
                    for local_group_idx in range(chunk_group_count):
                        b_up_tap = B_Up_split_tiles[chunk_idx][
                            col_group_idx * chunk_group_count + local_group_idx
                        ]
                        rt.fill(
                            inBUpProd[chunk_idx],
                            B_Up,
                            tap=b_up_tap,
                            placement=Tile(
                                col=ffn_bup_split_mem_tile_cols[chunk_idx], row=0
                            ),
                            task_group=task_group,
                            wait=wait_ffn_weight_fill,
                        )
            elif schedule_bup:
                bup_branch_order = (
                    list(range(effective_ffn_branches - 1, -1, -1))
                    if fill_bup_reverse
                    else list(range(effective_ffn_branches))
                )
                if bup_broadcast_priming:
                    remaining_bup_taps = {}
                    for branch_idx in bup_branch_order:
                        b_up_tap = B_Up_tiles[branch_idx][col_group_idx]
                        first_tap, rest_tap = split_tap_dim0_prefix_rest(
                            b_up_tap,
                            (embed_sz, ffn_intermediate_size),
                            1,
                        )
                        if first_tap is not None:
                            rt.fill(
                                inBUpProd[branch_idx],
                                B_Up,
                                tap=first_tap,
                                placement=Tile(
                                    col=ffn_bup_mem_tile_cols[branch_idx], row=0
                                ),
                                task_group=task_group,
                                wait=False,
                            )
                        if rest_tap is not None:
                            remaining_bup_taps[branch_idx] = rest_tap
                    for branch_idx in bup_branch_order:
                        b_up_tap = remaining_bup_taps.get(branch_idx)
                        if b_up_tap is None:
                            continue
                        for b_up_tap_part in split_tap_dim0(
                            b_up_tap,
                            (embed_sz, ffn_intermediate_size),
                            b_weight_fill_max_groups,
                        ):
                            rt.fill(
                                inBUpProd[branch_idx],
                                B_Up,
                                tap=b_up_tap_part,
                                placement=Tile(
                                    col=ffn_bup_mem_tile_cols[branch_idx], row=0
                                ),
                                task_group=task_group,
                                wait=wait_ffn_weight_fill,
                            )
                    # B_Up priming/fill for broadcast mode is complete.
                    # Continue to B_Down scheduling.
                else:
                    for branch_idx in bup_branch_order:
                        b_up_tap = B_Up_tiles[branch_idx][col_group_idx]
                        for b_up_tap_part in split_tap_dim0(
                            b_up_tap,
                            (embed_sz, ffn_intermediate_size),
                            b_weight_fill_max_groups,
                        ):
                            rt.fill(
                                inBUpProd[branch_idx],
                                B_Up,
                                tap=b_up_tap_part,
                                placement=Tile(
                                    col=ffn_bup_mem_tile_cols[branch_idx], row=0
                                ),
                                task_group=task_group,
                                wait=wait_ffn_weight_fill,
                            )

            if schedule_bdown and bdown_split_enabled:
                bdown_chunk_order = (
                    list(range(len(branch_split_chunks) - 1, -1, -1))
                    if fill_bdown_reverse
                    else list(range(len(branch_split_chunks)))
                )
                for chunk_idx in bdown_chunk_order:
                    chunk_group_count = ffn_col_group_counts[
                        branch_split_chunks[chunk_idx][0]
                    ]
                    for local_group_idx in range(chunk_group_count):
                        b_down_tap = B_Down_split_tiles[chunk_idx][
                            col_group_idx * chunk_group_count + local_group_idx
                        ]
                        rt.fill(
                            inBDownProd[chunk_idx],
                            B_Down,
                            tap=b_down_tap,
                            placement=Tile(
                                col=ffn_bdown_split_mem_tile_cols[chunk_idx],
                                row=0,
                            ),
                            task_group=task_group,
                            wait=wait_ffn_weight_fill,
                        )
            elif schedule_bdown:
                bdown_branch_order = (
                    list(range(effective_ffn_branches - 1, -1, -1))
                    if fill_bdown_reverse
                    else list(range(effective_ffn_branches))
                )
                for branch_idx in bdown_branch_order:
                    b_down_tap = B_Down_tiles[branch_idx][col_group_idx]
                    for b_down_tap_part in split_tap_dim0(
                        b_down_tap,
                        (ffn_intermediate_size, embed_sz),
                        b_weight_fill_max_groups,
                    ):
                        rt.fill(
                            inBDownProd[branch_idx],
                            B_Down,
                            tap=b_down_tap_part,
                            placement=Tile(
                                col=ffn_bdown_mem_tile_cols[branch_idx],
                                row=0,
                            ),
                            task_group=task_group,
                            wait=wait_ffn_weight_fill,
                        )

        def schedule_final_output_for_tap(tap_idx):
            if ffn_down_ddr_stage_enabled:
                # Stage one FFN-down stream through host memory to decouple
                # LN2 consumption from on-chip FFN down reduction progress.
                for _ in range(ffn_down_ddr_stage_passes):
                    tg_ffn_ddr_drain = rt.task_group()
                    rt.drain(
                        ffnDownOutStageToDDR.cons(),
                        OR,
                        tap=O_tiles[tap_idx],
                        placement=Tile(col=ffn_down_ddr_stage_col, row=0),
                        task_group=tg_ffn_ddr_drain,
                        wait=True,
                    )
                    rt.finish_task_group(tg_ffn_ddr_drain)
                    tg_ffn_ddr_fill = rt.task_group()
                    rt.fill(
                        ffnDownInFromDDR.prod(),
                        OR,
                        tap=O_tiles[tap_idx],
                        placement=Tile(col=ffn_down_ddr_stage_col, row=0),
                        task_group=tg_ffn_ddr_fill,
                        wait=True,
                    )
                    rt.finish_task_group(tg_ffn_ddr_fill)

            tg_out = rt.task_group()
            rt.drain(
                memLN2.cons(),
                OR,
                tap=O_tiles[tap_idx],
                placement=Tile(col=ln_mem_tile_col, row=0),
                task_group=tg_out,
                wait=wait_output_drain,
            )
            logging.debug(f"  O tap: {O_tiles[tap_idx]}")
            rt.finish_task_group(tg_out)

        def direct_verify_stage_source(handle):
            return handle.cons() if hasattr(handle, "cons") else handle

        def schedule_q_fill(task_group, q_block_idx, *, wait):
            for split_idx in range(mha_stream_split_factor):
                rt.fill(
                    inQ_streams[split_idx].prod(),
                    QKV,
                    tap=Q_tiles_by_stream[split_idx][q_block_idx],
                    placement=Tile(col=q_stream_mem_cols[split_idx], row=0),
                    task_group=task_group,
                    wait=wait,
                )

        def schedule_mha_head_fills(
            task_group,
            *,
            head_idx,
            col_group,
            include_v,
            include_wo,
            wait,
        ):
            wo_tap_idx = head_idx * num_o_col_groups + col_group
            for split_idx in range(mha_stream_split_factor):
                rt.fill(
                    inK_streams[split_idx].prod(),
                    QKV,
                    tap=K_tiles_by_stream[split_idx][head_idx],
                    placement=Tile(col=k_stream_mem_cols[split_idx], row=0),
                    task_group=task_group,
                    wait=wait,
                )
                if include_v:
                    rt.fill(
                        inV_streams[split_idx].prod(),
                        QKV,
                        tap=V_tiles_by_stream[split_idx][head_idx],
                        placement=Tile(col=v_stream_mem_cols[split_idx], row=0),
                        task_group=task_group,
                        wait=wait,
                    )
                if include_wo:
                    rt.fill(
                        inOW_streams[split_idx].prod(),
                        W_O,
                        tap=WO_tiles_by_stream[split_idx][wo_tap_idx],
                        placement=Tile(col=ow_stream_mem_cols[split_idx], row=0),
                        task_group=task_group,
                        wait=wait,
                    )

        stage_source_branch = 0
        direct_verify_output_col = ln_mem_tile_col

        pending_ln1_refill_tg = None
        pending_output_tap_idx = None

        if direct_verify_stage is not None:
            for i in range(parallel_heads):
                rt.start(matmul_workers[i])
                if build_softmax_workers:
                    rt.start(softmax_workers[i])
                if build_pv_workers:
                    rt.start(matmul_pv_workers[i])
                if build_o_proj_workers:
                    rt.start(o_proj_workers[i])
            if build_ln1_worker:
                rt.start(ln1_worker)
            if build_ffn_up_workers:
                for branch_idx in range(effective_ffn_branches):
                    rt.start(ffn_up_workers[branch_idx])
            if direct_verify_stage == STAGE_O_PROJ:
                for q_block_idx in range(num_q_seq_blocks):
                    for col_group in range(num_o_col_groups):
                        tg_q = rt.task_group()
                        schedule_q_fill(tg_q, q_block_idx, wait=False)
                        for head_idx in range(num_qkv_head_block_per_parallel_head):
                            tg_head = rt.task_group()
                            schedule_mha_head_fills(
                                tg_head,
                                head_idx=head_idx,
                                col_group=col_group,
                                include_v=True,
                                include_wo=True,
                                wait=False,
                            )
                            rt.finish_task_group(tg_head)
                        tap_idx = q_block_idx * num_o_col_groups + col_group
                        tg_out = rt.task_group()
                        rt.drain(
                            o_proj_verify_cons,
                            OR,
                            tap=O_tiles[tap_idx],
                            placement=Tile(col=direct_verify_output_col, row=0),
                            task_group=tg_out,
                            wait=True,
                        )
                        rt.finish_task_group(tg_out)
                        rt.finish_task_group(tg_q)
        else:
            for i in range(parallel_heads):
                rt.start(matmul_workers[i])
                rt.start(softmax_workers[i])
                rt.start(matmul_pv_workers[i])
                rt.start(o_proj_workers[i])
            rt.start(ln1_worker)
            for branch_idx in range(effective_ffn_branches):
                rt.start(ffn_up_workers[branch_idx])
                rt.start(ffn_down_workers[branch_idx])
            rt.start(ln2_worker)

            for q_block_idx in range(num_q_seq_blocks):
                for col_group in range(num_o_col_groups):
                    # Main fill group (Q when not pre-staged).
                    tg = rt.task_group()
                    # Tail fill group (R/B_Up/B_Down). Optionally keep this
                    # decoupled so tail drains can make progress before all FFN
                    # weight fills complete.
                    tg_tail_fill = rt.task_group() if decouple_tail_fill else tg
                    if serialize_q_prestage:
                        # Stage Q tiles first so head-0 compute cannot race ahead of Q load.
                        tg_q = rt.task_group()
                        for split_idx in range(mha_stream_split_factor):
                            rt.fill(
                                inQ_streams[split_idx].prod(),
                                QKV,
                                tap=Q_tiles_by_stream[split_idx][q_block_idx],
                                placement=Tile(col=q_stream_mem_cols[split_idx], row=0),
                                task_group=tg_q,
                            )
                        rt.finish_task_group(tg_q)
                    else:
                        for split_idx in range(mha_stream_split_factor):
                            rt.fill(
                                inQ_streams[split_idx].prod(),
                                QKV,
                                tap=Q_tiles_by_stream[split_idx][q_block_idx],
                                placement=Tile(col=q_stream_mem_cols[split_idx], row=0),
                                task_group=tg,
                                # Non-prestaged Q fills must be awaited at end-of-tap
                                # to avoid run-to-run drift in persistent worker loops.
                                wait=True,
                            )
                    logging.debug(
                        f"Scheduling fills for q block {q_block_idx}, col group {col_group} for QKV, W_O, and OR"
                    )
                    if mha_stream_split_factor == 1:
                        logging.debug(f"  Q tap: {Q_tiles_by_stream[0][q_block_idx]}")
                    else:
                        for split_idx in range(mha_stream_split_factor):
                            logging.debug(
                                "  Q tap split[%d]: %s",
                                split_idx,
                                Q_tiles_by_stream[split_idx][q_block_idx],
                            )
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
                            o_off = (
                                o_offsets[acc_idx] if acc_idx < len(o_offsets) else None
                            )
                            r_off = (
                                r_offsets[acc_idx] if acc_idx < len(r_offsets) else None
                            )
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
                        wo_tap_idx = head_idx * num_o_col_groups + col_group
                        for split_idx in range(mha_stream_split_factor):
                            rt.fill(
                                inK_streams[split_idx].prod(),
                                QKV,
                                tap=K_tiles_by_stream[split_idx][head_idx],
                                placement=Tile(col=k_stream_mem_cols[split_idx], row=0),
                                task_group=tg_head,
                                wait=True,
                            )
                            rt.fill(
                                inV_streams[split_idx].prod(),
                                QKV,
                                tap=V_tiles_by_stream[split_idx][head_idx],
                                placement=Tile(col=v_stream_mem_cols[split_idx], row=0),
                                task_group=tg_head,
                                wait=True,
                            )
                            rt.fill(
                                inOW_streams[split_idx].prod(),
                                W_O,
                                tap=WO_tiles_by_stream[split_idx][wo_tap_idx],
                                placement=Tile(
                                    col=ow_stream_mem_cols[split_idx], row=0
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                        rt.finish_task_group(tg_head)
                        if mha_stream_split_factor == 1:
                            logging.debug(
                                f"    K tap: {K_tiles_by_stream[0][head_idx]}"
                            )
                            logging.debug(
                                f"    V tap: {V_tiles_by_stream[0][head_idx]}"
                            )
                            logging.debug(
                                f"    W_O tap: {WO_tiles_by_stream[0][wo_tap_idx]}"
                            )
                        else:
                            for split_idx in range(mha_stream_split_factor):
                                logging.debug(
                                    "    K tap split[%d]: %s",
                                    split_idx,
                                    K_tiles_by_stream[split_idx][head_idx],
                                )
                                logging.debug(
                                    "    V tap split[%d]: %s",
                                    split_idx,
                                    V_tiles_by_stream[split_idx][head_idx],
                                )
                                logging.debug(
                                    "    W_O tap split[%d]: %s",
                                    split_idx,
                                    WO_tiles_by_stream[split_idx][wo_tap_idx],
                                )
                        wo_offsets = enumerate_outer_object_offsets(
                            WO_tiles[wo_tap_idx], inner_rank=2
                        )
                        logging.debug(
                            "    W_O outer offsets (head_idx=%d, q_block=%d, col_group=%d): %s",
                            head_idx,
                            q_block_idx,
                            col_group,
                            wo_offsets,
                        )

                    schedule_residual_fill(
                        tg_tail_fill,
                        q_block_idx * (num_o_col_groups) + col_group,
                    )
                    if ln1_mode_hooks.should_prefill_ffn_weights(
                        runtime_state=runtime_state,
                    ):
                        schedule_ffn_weight_fills(tg_tail_fill, col_group)

                    pending_ln1_refill_tg, pending_output_tap_idx = (
                        ln1_mode_hooks.schedule_runtime_tap(
                            rt=rt,
                            OR=OR,
                            or_tensor_shape=or_tensor_shape,
                            seq_len=seq_len,
                            seq_tile=seq_tile,
                            embed_sz=embed_sz,
                            profile_replay_groups=ln1_broadcast_groups,
                            proj_acc_depth=proj_acc_depth,
                            emb_tile=emb_tile,
                            runtime_state=runtime_state,
                            ln1OutStageToDDR=ln1OutStageToDDR,
                            ln1InFromDDR=ln1InFromDDR,
                            ffnRFromDDR=ffnRFromDDR,
                            ffn_a_stage_mem_tile_cols=ffn_a_stage_mem_tile_cols,
                            ffn_residual_fill_col=ffn_residual_fill_col,
                            transfer_count_for_fifo_obj=transfer_count_for_fifo_obj,
                            tensor_access_pattern_cls=TensorAccessPattern,
                            schedule_ffn_weight_fills=schedule_ffn_weight_fills,
                            schedule_final_output_for_tap=schedule_final_output_for_tap,
                            tap_idx=tap_idx,
                            col_group=col_group,
                            tg=tg,
                            tg_tail_fill=tg_tail_fill,
                            decouple_tail_fill=decouple_tail_fill,
                            pending_ln1_refill_tg=pending_ln1_refill_tg,
                            pending_output_tap_idx=pending_output_tap_idx,
                            tile_ctor=Tile,
                        )
                    )

            pending_ln1_refill_tg, pending_output_tap_idx = (
                ln1_mode_hooks.finalize_runtime(
                    rt=rt,
                    schedule_final_output_for_tap=schedule_final_output_for_tap,
                    pending_ln1_refill_tg=pending_ln1_refill_tg,
                    pending_output_tap_idx=pending_output_tap_idx,
                )
            )

    # Create the program from the device type and runtime
    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module
