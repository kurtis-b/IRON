# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.arith as arith
import aie.dialects.index as index
import aie.extras.types as T
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import if_, else_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from iron.operators.encoder_pipeline.topology import (
    load_encoder_pipeline_topology_placements,
    topology_from_fields,
)

BASE_DIR = Path(__file__).parent


def compute_num_q_seq_blocks(seq_len: int, seq_tile: int) -> int:
    if seq_len % seq_tile != 0:
        raise ValueError("seq_len must be divisible by seq_tile")
    return seq_len // seq_tile


def compute_q_blocks_per_lane(seq_len: int, seq_tile: int, parallel_seq: int) -> int:
    num_q_seq_blocks = compute_num_q_seq_blocks(seq_len, seq_tile)
    if num_q_seq_blocks % parallel_seq != 0:
        raise ValueError(
            "encoder_pipeline requires num_q_seq_blocks divisible by parallel_seq "
            f"({num_q_seq_blocks} % {parallel_seq} != 0)"
        )
    return num_q_seq_blocks // parallel_seq


def compute_num_kv_seq_blocks(seq_len: int, kv_seq_tile: int) -> int:
    if seq_len % kv_seq_tile != 0:
        raise ValueError("seq_len must be divisible by kv_seq_tile")
    return seq_len // kv_seq_tile


def compute_qkv_head_blocks_per_parallel_head(heads: int, parallel_heads: int) -> int:
    if heads % parallel_heads != 0:
        raise ValueError(
            "encoder_pipeline requires heads divisible by parallel_heads "
            f"({heads} % {parallel_heads} != 0)"
        )
    return heads // parallel_heads


def compute_ln1_broadcast_groups(ffn_intermediate_size: int, ffn_tile: int) -> int:
    if ffn_intermediate_size % ffn_tile != 0:
        raise ValueError(
            "ffn_intermediate_size must be divisible by ffn_tile "
            f"({ffn_intermediate_size} % {ffn_tile} != 0)"
        )
    return ffn_intermediate_size // ffn_tile


def compute_ffn_col_group_count(
    ffn_intermediate_size: int, ffn_tile: int, effective_ffn_branches: int
) -> int:
    ln1_broadcast_groups = compute_ln1_broadcast_groups(ffn_intermediate_size, ffn_tile)
    if ln1_broadcast_groups % effective_ffn_branches != 0:
        raise ValueError(
            "encoder_pipeline requires FFN branch count to divide "
            "ln1_broadcast_groups "
            f"({ln1_broadcast_groups} % {effective_ffn_branches} != 0)"
        )
    return ln1_broadcast_groups // effective_ffn_branches


def build_q_block_schedule(parallel_seq: int, q_blocks_per_lane: int) -> list[int]:
    return [
        lane_idx * q_blocks_per_lane + lane_block_idx
        for lane_block_idx in range(q_blocks_per_lane)
        for lane_idx in range(parallel_seq)
    ]


def count_tap_iterations(tap: TensorAccessPattern, obj_shape: tuple[int, ...]) -> int:
    return math.prod(int(s) for s in tap.sizes) // math.prod(obj_shape)


def assert_tap_iteration_count(
    tap: TensorAccessPattern,
    obj_shape: tuple[int, ...],
    expected: int,
    message: str,
) -> None:
    if count_tap_iterations(tap, obj_shape) != expected:
        raise ValueError(message)


def repeat_outer_tap(
    tap: TensorAccessPattern, tensor_shape: tuple[int, ...], repeat_count: int
) -> TensorAccessPattern:
    repeated_outer_size = int(tap.sizes[0]) * repeat_count
    return TensorAccessPattern(
        tensor_shape,
        offset=int(tap.offset),
        sizes=[repeated_outer_size, *[int(s) for s in tap.sizes[1:]]],
        strides=[0, *[int(s) for s in tap.strides[1:]]],
    )


def _resolve_topology_placement(
    heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    ffn_tile: int,
    parallel_seq: int,
    parallel_heads: int,
    proj_acc_depth: int,
    nB_tiles_distributed: int,
    ffn_intermediate_size: int,
    use_transport_groups: bool | None = None,
    use_unified_qr_split: bool | None = None,
    use_explicit_o_proj_stage_mem_cols: bool | None = None,
    use_explicit_ffn_down_acc_mem_cols: bool | None = None,
    use_explicit_ffn_down_stage_mem_cols: bool | None = None,
) -> tuple[dict, dict | None]:
    topology = topology_from_fields(
        num_heads=heads,
        seq_len=seq_len,
        d=d,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=1,
        parallel_ffn=nB_tiles_distributed,
        ffn_intermediate_size=ffn_intermediate_size,
    )
    topology_placements = load_encoder_pipeline_topology_placements()
    if topology.key not in topology_placements:
        raise ValueError(
            "encoder_pipeline only supports hardcoded placement topologies "
            f"{sorted(topology_placements)} (got placement key {topology.key})"
        )
    placement = deepcopy(topology_placements[topology.key])
    sequence_parallel = placement["sequence_parallel"]
    if sequence_parallel is None:
        if (
            use_transport_groups is not None
            or use_unified_qr_split is not None
            or use_explicit_o_proj_stage_mem_cols is not None
            or use_explicit_ffn_down_acc_mem_cols is not None
            or use_explicit_ffn_down_stage_mem_cols is not None
        ):
            raise ValueError(
                "encoder_pipeline data-movement overrides require a "
                "sequence-parallel placement"
            )
        return placement, sequence_parallel

    if use_transport_groups is not None:
        transport_groups = sequence_parallel.get("transport_groups")
        if transport_groups is None or len(transport_groups) <= 1:
            raise ValueError(
                "encoder_pipeline use_transport_groups override requires a "
                "sequence-parallel placement with multiple transport groups"
            )
        if not use_transport_groups:
            sequence_parallel = dict(sequence_parallel)
            sequence_parallel.pop("transport_groups", None)
            placement["sequence_parallel"] = sequence_parallel

    if use_unified_qr_split is not None:
        unified_qr_split = sequence_parallel.get("unified_qr_split")
        if unified_qr_split is None:
            raise ValueError(
                "encoder_pipeline use_unified_qr_split override requires a "
                "sequence-parallel placement that defines unified_qr_split"
            )
        if not use_unified_qr_split:
            sequence_parallel = dict(sequence_parallel)
            sequence_parallel.pop("unified_qr_split", None)
            placement["sequence_parallel"] = sequence_parallel

    if use_explicit_o_proj_stage_mem_cols is not None:
        explicit_cols = sequence_parallel.get("lane_o_proj_stage_mem_cols")
        if explicit_cols is None:
            raise ValueError(
                "encoder_pipeline use_explicit_o_proj_stage_mem_cols override "
                "requires a sequence-parallel placement that defines "
                "lane_o_proj_stage_mem_cols"
            )
        if not use_explicit_o_proj_stage_mem_cols:
            sequence_parallel = dict(sequence_parallel)
            sequence_parallel.pop("lane_o_proj_stage_mem_cols", None)
            placement["sequence_parallel"] = sequence_parallel

    if use_explicit_ffn_down_acc_mem_cols is not None:
        explicit_cols = sequence_parallel.get("lane_ffn_down_acc_mem_cols")
        if explicit_cols is None:
            raise ValueError(
                "encoder_pipeline use_explicit_ffn_down_acc_mem_cols override "
                "requires a sequence-parallel placement that defines "
                "lane_ffn_down_acc_mem_cols"
            )
        if not use_explicit_ffn_down_acc_mem_cols:
            sequence_parallel = dict(sequence_parallel)
            sequence_parallel.pop("lane_ffn_down_acc_mem_cols", None)
            placement["sequence_parallel"] = sequence_parallel

    if use_explicit_ffn_down_stage_mem_cols is not None:
        explicit_cols = sequence_parallel.get("lane_ffn_down_stage_mem_cols")
        if explicit_cols is None:
            raise ValueError(
                "encoder_pipeline use_explicit_ffn_down_stage_mem_cols override "
                "requires a sequence-parallel placement that defines "
                "lane_ffn_down_stage_mem_cols"
            )
        if not use_explicit_ffn_down_stage_mem_cols:
            sequence_parallel = dict(sequence_parallel)
            sequence_parallel.pop("lane_ffn_down_stage_mem_cols", None)
            placement["sequence_parallel"] = sequence_parallel

    return placement, placement["sequence_parallel"]


def _validate_sequence_parallel_placement(
    sequence_parallel: dict | None,
    parallel_seq: int,
) -> None:
    if sequence_parallel is None:
        return
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
    lane_o_proj_stage_mem_cols = sequence_parallel.get("lane_o_proj_stage_mem_cols")
    if (
        lane_o_proj_stage_mem_cols is not None
        and len(lane_o_proj_stage_mem_cols) != parallel_seq
    ):
        raise ValueError(
            "encoder_pipeline sequence-parallel placement must provide one "
            "O-proj replay memtile per sequence lane "
            f"({len(lane_o_proj_stage_mem_cols)} != {parallel_seq})"
        )
    transport_groups = sequence_parallel.get("transport_groups")
    if transport_groups is None:
        return
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


def _derive_encoder_pipeline_layout(
    heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    ffn_tile: int,
    proj_acc_depth: int,
    parallel_seq: int,
    parallel_heads: int,
    trace_size: int,
    sequence_parallel: dict | None,
    placement: dict,
    nB_tiles_distributed: int,
    ffn_intermediate_size: int,
    o_proj_acc_group_size: int,
    ffn_down_acc_group_size: int,
    emulate_bf16_mmul_with_bfp16: bool,
    weight_forward_depth: int | None = None,
    o_proj_fifo_depth: int | None = None,
    ffn_replay_fifo_depth: int | None = None,
    ffn_up_consumer_depth: int | None = None,
    ffn_up_out_depth: int | None = None,
    ffn_down_output_producer_depth: int | None = None,
) -> dict:
    if trace_size != 0:
        raise ValueError("encoder_pipeline does not support tracing")
    if o_proj_acc_group_size <= 0:
        raise ValueError(
            "encoder_pipeline requires o_proj_acc_group_size > 0 "
            f"(got {o_proj_acc_group_size})"
        )
    if ffn_down_acc_group_size <= 0:
        raise ValueError(
            "encoder_pipeline requires ffn_down_acc_group_size > 0 "
            f"(got {ffn_down_acc_group_size})"
        )
    if d != 64:
        raise ValueError(f"encoder_pipeline requires d=64 (got {d})")
    if not emulate_bf16_mmul_with_bfp16:
        raise ValueError("encoder_pipeline requires emulate_bf16_mmul_with_bfp16=True")

    embed_sz = heads * d
    num_q_seq_blocks = compute_num_q_seq_blocks(seq_len, seq_tile)
    num_kv_seq_blocks = compute_num_kv_seq_blocks(seq_len, kv_seq_tile)
    q_blocks_per_lane = compute_q_blocks_per_lane(seq_len, seq_tile, parallel_seq)
    num_qkv_head_block_per_parallel_head = compute_qkv_head_blocks_per_parallel_head(
        heads, parallel_heads
    )
    num_o_col_groups = embed_sz // (emb_tile * proj_acc_depth)
    if num_o_col_groups != 1:
        raise ValueError(
            "encoder_pipeline hardcoded path requires num_o_col_groups == 1 "
            f"(got {num_o_col_groups})"
        )
    if embed_sz != emb_tile * proj_acc_depth:
        raise ValueError(
            "emb_tile * proj_acc_depth must equal embed_sz "
            f"({emb_tile} * {proj_acc_depth} != {embed_sz})"
        )
    if ffn_tile % 16 != 0:
        raise ValueError(f"ffn_tile must be divisible by 16 ({ffn_tile} % 16 != 0)")
    if len(placement["mha_cols"]) != parallel_heads:
        raise ValueError(
            "encoder_pipeline hardcoded placement must provide one MHA column per "
            f"parallel head (cols={placement['mha_cols']}, parallel_heads={parallel_heads})"
        )

    dtype = bfloat16
    inv_scale = (1 / np.sqrt(d)) * 1.4453125
    of_depth = 2
    o_tile_bytes = seq_tile * emb_tile * np.dtype(dtype).itemsize
    large_activation_tile = emb_tile >= 128 or o_tile_bytes > 8192
    weight_forward_depth_default = (
        1 if large_activation_tile else (2 if ffn_tile <= 64 else 1)
    )
    o_proj_fifo_depth_default = 1 if large_activation_tile else of_depth
    ffn_replay_fifo_depth_default = 1 if large_activation_tile else 2
    ffn_up_consumer_depth_default = 1 if large_activation_tile else 2
    ffn_up_out_depth_default = 1 if large_activation_tile else 2
    ffn_down_output_producer_depth_default = 1 if large_activation_tile else 2
    weight_forward_depth = _resolve_positive_layout_override(
        "weight_forward_depth", weight_forward_depth, weight_forward_depth_default
    )
    o_proj_fifo_depth = _resolve_positive_layout_override(
        "o_proj_fifo_depth", o_proj_fifo_depth, o_proj_fifo_depth_default
    )
    o_proj_weight_consumer_depth = o_proj_fifo_depth
    ffn_replay_fifo_depth = _resolve_positive_layout_override(
        "ffn_replay_fifo_depth",
        ffn_replay_fifo_depth,
        ffn_replay_fifo_depth_default,
    )
    ffn_up_consumer_depth = _resolve_positive_layout_override(
        "ffn_up_consumer_depth",
        ffn_up_consumer_depth,
        ffn_up_consumer_depth_default,
    )
    ffn_up_out_depth = _resolve_positive_layout_override(
        "ffn_up_out_depth", ffn_up_out_depth, ffn_up_out_depth_default
    )
    ffn_down_output_producer_depth = _resolve_positive_layout_override(
        "ffn_down_output_producer_depth",
        ffn_down_output_producer_depth,
        ffn_down_output_producer_depth_default,
    )
    ln1_broadcast_groups = compute_ln1_broadcast_groups(ffn_intermediate_size, ffn_tile)
    effective_ffn_branches = len(placement["tail_tiles"]["ffn_up_by_branch"])
    if effective_ffn_branches != nB_tiles_distributed:
        raise ValueError(
            "encoder_pipeline hardcoded placement must provide one FFN branch per "
            "requested nB_tiles_distributed "
            f"(branches={effective_ffn_branches}, requested={nB_tiles_distributed})"
        )
    ffn_col_group_count = compute_ffn_col_group_count(
        ffn_intermediate_size, ffn_tile, effective_ffn_branches
    )

    if sequence_parallel is not None and parallel_heads > 1:
        if o_proj_acc_group_size not in (1, parallel_heads):
            raise ValueError(
                "encoder_pipeline seq-par currently supports "
                "o_proj_acc_group_size of 1 or parallel_heads "
                f"(got {o_proj_acc_group_size}, parallel_heads={parallel_heads})"
            )
    elif o_proj_acc_group_size != 1:
        raise ValueError(
            "encoder_pipeline currently supports o_proj_acc_group_size=1 "
            "outside the seq-par multi-head path "
            f"(got {o_proj_acc_group_size})"
        )
    if sequence_parallel is not None and effective_ffn_branches > 1:
        if ffn_down_acc_group_size not in (1, effective_ffn_branches):
            raise ValueError(
                "encoder_pipeline seq-par currently supports "
                "ffn_down_acc_group_size of 1 or effective_ffn_branches "
                f"(got {ffn_down_acc_group_size}, effective_ffn_branches={effective_ffn_branches})"
            )
    elif ffn_down_acc_group_size != 1:
        raise ValueError(
            "encoder_pipeline currently supports ffn_down_acc_group_size=1 "
            "outside the seq-par multi-branch path "
            f"(got {ffn_down_acc_group_size})"
        )

    _validate_sequence_parallel_placement(sequence_parallel, parallel_seq)

    return {
        "embed_sz": embed_sz,
        "num_q_seq_blocks": num_q_seq_blocks,
        "num_kv_seq_blocks": num_kv_seq_blocks,
        "q_blocks_per_lane": q_blocks_per_lane,
        "num_qkv_head_block_per_parallel_head": num_qkv_head_block_per_parallel_head,
        "num_o_col_groups": num_o_col_groups,
        "dtype": dtype,
        "inv_scale": inv_scale,
        "of_depth": of_depth,
        "weight_forward_depth": weight_forward_depth,
        "o_proj_weight_consumer_depth": o_proj_weight_consumer_depth,
        "o_proj_fifo_depth": o_proj_fifo_depth,
        "ffn_replay_fifo_depth": ffn_replay_fifo_depth,
        "ffn_up_consumer_depth": ffn_up_consumer_depth,
        "ffn_up_out_depth": ffn_up_out_depth,
        "ffn_down_output_producer_depth": ffn_down_output_producer_depth,
        "ln1_broadcast_groups": ln1_broadcast_groups,
        "effective_ffn_branches": effective_ffn_branches,
        "ffn_col_group_count": ffn_col_group_count,
        "ln_tiles_per_q_block": proj_acc_depth,
        "ln1_dram_stage_rows": (
            seq_len if sequence_parallel is not None else proj_acc_depth * seq_tile
        ),
    }


def _normalize_slot_coords(value, expected_count, name):
    if expected_count == 1:
        if isinstance(value[0], int):
            return [tuple(value)]
        if len(value) != 1:
            raise ValueError(f"{name} must provide exactly one placement (got {value})")
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


def _normalize_slot_mem_cols(value, expected_count, name):
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


def _resolve_positive_layout_override(name, value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(
            f"encoder_pipeline requires {name} to be an integer > 0 (got {value})"
        )
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"encoder_pipeline requires {name} > 0 (got {normalized})")
    return normalized


def _normalize_base_placement(placement: dict) -> dict:
    return {
        "qk_tiles": [Tile(col=col, row=2) for col in placement["mha_cols"]],
        "softmax_tiles": [Tile(col=col, row=3) for col in placement["mha_cols"]],
        "pv_tiles": [Tile(col=col, row=4) for col in placement["mha_cols"]],
        "o_proj_tiles": [Tile(col=col, row=5) for col in placement["mha_cols"]],
        "ln1_tile": Tile(*placement["tail_tiles"]["ln1"]),
        "ffn_up_tiles": [
            Tile(*tile) for tile in placement["tail_tiles"]["ffn_up_by_branch"]
        ],
        "ffn_down_tiles": [
            Tile(*tile) for tile in placement["tail_tiles"]["ffn_down_by_branch"]
        ],
        "ln2_tile": Tile(*placement["tail_tiles"]["ln2"]),
        "accumulation_mem_tiles": placement["accumulation_mem_tiles"],
        "weight_mem_tiles": placement["weight_mem_tiles"],
        "q_mem_col": placement["mem_tiles"]["q"],
        "k_mem_col": placement["mem_tiles"]["k"],
        "v_mem_col": placement["mem_tiles"]["v"],
        "ow_mem_col": placement["mem_tiles"]["w_o"],
        "o_proj_stage_mem_col": placement["accumulation_mem_tiles"]["o_proj_stage"],
        "ln1_stage_col": placement["mem_tiles"]["ln1_stage"],
        "residual_mem_col": placement["mem_tiles"]["residual"],
        "ffn_residual_mem_col": placement["mem_tiles"]["ffn_residual"],
        "output_mem_col": placement["mem_tiles"]["output"],
        "q_shim_col": placement["shim_tiles"]["q"],
        "k_shim_col": placement["shim_tiles"]["k"],
        "v_shim_col": placement["shim_tiles"]["v"],
        "ow_shim_col": placement["shim_tiles"]["w_o"],
        "bdown_shim_col": placement["shim_tiles"]["b_down"],
        "bup_shim_col": placement["shim_tiles"]["b_up"],
        "ln1_stage_shim_col": placement["shim_tiles"]["ln1_stage"],
        "residual_shim_col": placement["shim_tiles"]["residual"],
        "output_shim_col": placement["shim_tiles"]["output"],
    }


def _load_ln_weights(
    embed_sz: int, ln1_weight_file: str | None, ln2_weight_file: str | None
) -> tuple[np.ndarray, np.ndarray]:
    def _load_weight_file(path: str | None) -> np.ndarray:
        if path is None:
            return np.ones(embed_sz, dtype=bfloat16)
        raw = np.load(path)
        if raw.dtype == np.dtype("V2"):
            raw = raw.view(bfloat16)
        return np.asarray(raw, dtype=bfloat16).reshape(embed_sz)

    static_ln1_weights = _load_weight_file(ln1_weight_file)
    static_ln2_weights = _load_weight_file(ln2_weight_file)
    return static_ln1_weights, static_ln2_weights


def _load_staged_projection_biases(
    *,
    heads: int,
    d: int,
    q_proj_bias_file: str | None,
    k_proj_bias_file: str | None,
    v_proj_bias_file: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _load_bias_file(path: str | None, expected_shape: tuple[int]) -> np.ndarray:
        if path is None:
            return np.zeros(expected_shape, dtype=bfloat16)
        raw = np.load(path)
        if raw.dtype == np.dtype("V2"):
            raw = raw.view(bfloat16)
        return np.asarray(raw, dtype=bfloat16).reshape(expected_shape)

    return (
        _load_bias_file(q_proj_bias_file, (heads * d,)),
        _load_bias_file(k_proj_bias_file, (heads * d,)),
        _load_bias_file(v_proj_bias_file, (heads * d,)),
    )


def _pack_ln_weights_for_runtime(weights: np.ndarray) -> np.ndarray:
    return np.asarray(weights, dtype=np.float32).view(np.int32).copy()


def _emit_runtime_buffer_values(buffer: Buffer, values: np.ndarray) -> None:
    flat_values = np.asarray(values).reshape(-1)
    shape = tuple(int(dim) for dim in getattr(buffer, "shape", (len(flat_values),)))
    for idx, value in enumerate(flat_values):
        if len(shape) == 1:
            buffer[idx] = int(value)
        else:
            buffer[np.unravel_index(idx, shape)] = int(value)


def _emit_program(rt: Runtime):
    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


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
    argparser.add_argument("--ffn-down-acc-group-size", type=int, default=1)
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
    argparser.add_argument("--weight-forward-depth", type=int, default=None)
    argparser.add_argument("--o-proj-fifo-depth", type=int, default=None)
    argparser.add_argument("--ffn-replay-fifo-depth", type=int, default=None)
    argparser.add_argument("--ffn-up-consumer-depth", type=int, default=None)
    argparser.add_argument("--ffn-up-out-depth", type=int, default=None)
    argparser.add_argument("--ffn-down-output-producer-depth", type=int, default=None)
    argparser.add_argument(
        "--use-fused-replayed-addnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    argparser.add_argument(
        "--use-transport-groups",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    argparser.add_argument(
        "--use-unified-qr-split",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    argparser.add_argument(
        "--use-explicit-o-proj-stage-mem-cols",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    argparser.add_argument(
        "--use-explicit-ffn-down-acc-mem-cols",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    argparser.add_argument(
        "--use-explicit-ffn-down-stage-mem-cols",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    argparser.add_argument(
        "--qkv-projection-mode",
        type=str,
        default="packed_input",
    )
    argparser.add_argument("--q-proj-bias-file", type=str, default=None)
    argparser.add_argument("--k-proj-bias-file", type=str, default=None)
    argparser.add_argument("--v-proj-bias-file", type=str, default=None)
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
        ffn_down_acc_group_size=args.ffn_down_acc_group_size,
        weight_forward_depth=args.weight_forward_depth,
        o_proj_fifo_depth=args.o_proj_fifo_depth,
        ffn_replay_fifo_depth=args.ffn_replay_fifo_depth,
        ffn_up_consumer_depth=args.ffn_up_consumer_depth,
        ffn_up_out_depth=args.ffn_up_out_depth,
        ffn_down_output_producer_depth=args.ffn_down_output_producer_depth,
        use_fused_replayed_addnorm=args.use_fused_replayed_addnorm,
        use_transport_groups=args.use_transport_groups,
        use_unified_qr_split=args.use_unified_qr_split,
        use_explicit_o_proj_stage_mem_cols=args.use_explicit_o_proj_stage_mem_cols,
        use_explicit_ffn_down_acc_mem_cols=args.use_explicit_ffn_down_acc_mem_cols,
        use_explicit_ffn_down_stage_mem_cols=args.use_explicit_ffn_down_stage_mem_cols,
        qkv_projection_mode=args.qkv_projection_mode,
        q_proj_bias_file=args.q_proj_bias_file,
        k_proj_bias_file=args.k_proj_bias_file,
        v_proj_bias_file=args.v_proj_bias_file,
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
    ffn_down_acc_group_size: int = 1,
    weight_forward_depth: int | None = None,
    o_proj_fifo_depth: int | None = None,
    ffn_replay_fifo_depth: int | None = None,
    ffn_up_consumer_depth: int | None = None,
    ffn_up_out_depth: int | None = None,
    ffn_down_output_producer_depth: int | None = None,
    use_fused_replayed_addnorm: bool = False,
    use_transport_groups: bool | None = None,
    use_unified_qr_split: bool | None = None,
    use_explicit_o_proj_stage_mem_cols: bool | None = None,
    use_explicit_ffn_down_acc_mem_cols: bool | None = None,
    use_explicit_ffn_down_stage_mem_cols: bool | None = None,
    qkv_projection_mode: str = "packed_input",
    q_proj_bias_file: str | None = None,
    k_proj_bias_file: str | None = None,
    v_proj_bias_file: str | None = None,
    use_runtime_ln_weights: bool = False,
):
    if qkv_projection_mode not in {"packed_input", "staged_hidden_states"}:
        raise ValueError(
            "encoder_pipeline qkv_projection_mode must be one of "
            "{'packed_input', 'staged_hidden_states'} "
            f"(got {qkv_projection_mode!r})"
        )
    if ffn_intermediate_size is None:
        ffn_intermediate_size = 4 * heads * d
    if ffn_tile is None:
        ffn_tile = emb_tile

    placement, sequence_parallel = _resolve_topology_placement(
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
        nB_tiles_distributed,
        ffn_intermediate_size,
        use_transport_groups,
        use_unified_qr_split,
        use_explicit_o_proj_stage_mem_cols,
        use_explicit_ffn_down_acc_mem_cols,
        use_explicit_ffn_down_stage_mem_cols,
    )
    use_staged_hidden_state_projection = qkv_projection_mode == "staged_hidden_states"
    if (
        use_staged_hidden_state_projection
        and sequence_parallel is not None
        and (parallel_heads > 2 or nB_tiles_distributed != 1)
    ):
        raise ValueError(
            "encoder_pipeline staged_hidden_states sequence-parallel projection "
            "currently requires parallel_heads <= 2 and nB_tiles_distributed=1 "
            f"(got parallel_heads={parallel_heads}, "
            f"nB_tiles_distributed={nB_tiles_distributed})"
        )
    if use_staged_hidden_state_projection and parallel_heads > 2:
        raise ValueError(
            "encoder_pipeline staged_hidden_states QKV projection currently "
            "supports parallel_heads <= 2 "
            f"(got parallel_heads={parallel_heads})"
        )
    layout = _derive_encoder_pipeline_layout(
        heads,
        seq_len,
        d,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        ffn_tile,
        proj_acc_depth,
        parallel_seq,
        parallel_heads,
        trace_size,
        sequence_parallel,
        placement,
        nB_tiles_distributed,
        ffn_intermediate_size,
        o_proj_acc_group_size,
        ffn_down_acc_group_size,
        emulate_bf16_mmul_with_bfp16,
        weight_forward_depth,
        o_proj_fifo_depth,
        ffn_replay_fifo_depth,
        ffn_up_consumer_depth,
        ffn_up_out_depth,
        ffn_down_output_producer_depth,
    )
    embed_sz = layout["embed_sz"]
    num_q_seq_blocks = layout["num_q_seq_blocks"]
    num_kv_seq_blocks = layout["num_kv_seq_blocks"]
    q_blocks_per_lane = layout["q_blocks_per_lane"]
    num_qkv_head_block_per_parallel_head = layout[
        "num_qkv_head_block_per_parallel_head"
    ]
    num_o_col_groups = layout["num_o_col_groups"]
    dtype = layout["dtype"]
    inv_scale = layout["inv_scale"]
    of_depth = layout["of_depth"]
    weight_forward_depth = layout["weight_forward_depth"]
    o_proj_weight_consumer_depth = layout["o_proj_weight_consumer_depth"]
    o_proj_fifo_depth = layout["o_proj_fifo_depth"]
    ffn_replay_fifo_depth = layout["ffn_replay_fifo_depth"]
    ffn_up_consumer_depth = layout["ffn_up_consumer_depth"]
    ffn_up_out_depth = layout["ffn_up_out_depth"]
    ffn_down_output_producer_depth = layout["ffn_down_output_producer_depth"]
    ln1_broadcast_groups = layout["ln1_broadcast_groups"]
    effective_ffn_branches = layout["effective_ffn_branches"]
    ffn_col_group_count = layout["ffn_col_group_count"]
    ln_tiles_per_q_block = layout["ln_tiles_per_q_block"]
    ln1_dram_stage_rows = layout["ln1_dram_stage_rows"]
    use_staged_hidden_state_kv_cache = False
    # The staged hidden-state path currently keeps direct K/V streaming for all
    # supported topologies. The earlier OR-backed K/V cache path is pruned
    # because it is not functionally stable at longer sequence lengths.
    or_rows_before_ln1_stage = (
        3 * seq_len
        if use_staged_hidden_state_kv_cache
        else (
            4 * seq_len
            if (use_staged_hidden_state_projection and sequence_parallel is not None)
            else (seq_len if use_staged_hidden_state_projection else 2 * seq_len)
        )
    )
    or_tensor_shape = (or_rows_before_ln1_stage + ln1_dram_stage_rows, embed_sz)

    static_ln1_weights_bf16, static_ln2_weights_bf16 = _load_ln_weights(
        embed_sz, ln1_weight_file, ln2_weight_file
    )
    static_ln1_weights = _pack_ln_weights_for_runtime(static_ln1_weights_bf16)
    static_ln2_weights = _pack_ln_weights_for_runtime(static_ln2_weights_bf16)
    static_q_proj_biases = None
    static_k_proj_biases = None
    static_v_proj_biases = None
    if use_staged_hidden_state_projection:
        (
            static_q_proj_biases,
            static_k_proj_biases,
            static_v_proj_biases,
        ) = _load_staged_projection_biases(
            heads=heads,
            d=d,
            q_proj_bias_file=q_proj_bias_file,
            k_proj_bias_file=k_proj_bias_file,
            v_proj_bias_file=v_proj_bias_file,
        )

    normalized_placement = _normalize_base_placement(placement)
    qk_tiles = normalized_placement["qk_tiles"]
    softmax_tiles = normalized_placement["softmax_tiles"]
    pv_tiles = normalized_placement["pv_tiles"]
    o_proj_tiles = normalized_placement["o_proj_tiles"]
    ln1_tile = normalized_placement["ln1_tile"]
    ffn_up_tiles = normalized_placement["ffn_up_tiles"]
    ffn_down_tiles = normalized_placement["ffn_down_tiles"]
    ln2_tile = normalized_placement["ln2_tile"]
    accumulation_mem_tiles = normalized_placement["accumulation_mem_tiles"]
    weight_mem_tiles = normalized_placement["weight_mem_tiles"]
    q_mem_col = normalized_placement["q_mem_col"]
    k_mem_col = normalized_placement["k_mem_col"]
    v_mem_col = normalized_placement["v_mem_col"]
    ow_mem_col = normalized_placement["ow_mem_col"]
    ln1_stage_col = normalized_placement["ln1_stage_col"]
    residual_mem_col = normalized_placement["residual_mem_col"]
    ffn_residual_mem_col = normalized_placement["ffn_residual_mem_col"]
    output_mem_col = normalized_placement["output_mem_col"]
    q_shim_col = normalized_placement["q_shim_col"]
    k_shim_col = normalized_placement["k_shim_col"]
    v_shim_col = normalized_placement["v_shim_col"]
    ow_shim_col = normalized_placement["ow_shim_col"]
    bdown_shim_col = normalized_placement["bdown_shim_col"]
    bup_shim_col = normalized_placement["bup_shim_col"]
    ln1_stage_shim_col = normalized_placement["ln1_stage_shim_col"]
    residual_shim_col = normalized_placement["residual_shim_col"]
    output_shim_col = normalized_placement["output_shim_col"]
    use_memtile_o_proj_replay = o_proj_fifo_depth == 1 and parallel_seq > 1

    W_O_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    W_ATTN_ty = np.ndarray[(4 * embed_sz, embed_sz), np.dtype[dtype]]
    X_ty = np.ndarray[(seq_len, embed_sz), np.dtype[dtype]]
    W_Q_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    W_K_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    W_V_ty = np.ndarray[(embed_sz, embed_sz), np.dtype[dtype]]
    QKV_ty = np.ndarray[(3 * seq_len, embed_sz), np.dtype[dtype]]
    OR_ty = np.ndarray[or_tensor_shape, np.dtype[dtype]]
    B_Up_ty = np.ndarray[(embed_sz * ffn_intermediate_size,), np.dtype[dtype]]
    B_Down_ty = np.ndarray[(ffn_intermediate_size * embed_sz,), np.dtype[dtype]]
    q_stream_ty = np.ndarray[(seq_tile, d * parallel_heads), np.dtype[dtype]]
    kv_stream_ty = np.ndarray[(kv_seq_tile, d * parallel_heads), np.dtype[dtype]]
    wo_stream_ty = np.ndarray[(d * parallel_heads, emb_tile), np.dtype[dtype]]

    q_ty = np.ndarray[(seq_tile, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, kv_seq_tile), np.dtype[dtype]]
    qk_ty = np.ndarray[(seq_tile, kv_seq_tile), np.dtype[dtype]]
    v_ty = np.ndarray[(kv_seq_tile, d), np.dtype[dtype]]
    bias_matrix_ty = np.ndarray[
        (num_qkv_head_block_per_parallel_head, d), np.dtype[dtype]
    ]
    xq_ty = np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]]
    xkv_ty = np.ndarray[(kv_seq_tile, emb_tile), np.dtype[dtype]]
    q_proj_w_ty = np.ndarray[(emb_tile, d), np.dtype[dtype]]
    k_proj_w_ty = np.ndarray[(emb_tile, d), np.dtype[dtype]]
    v_proj_w_ty = np.ndarray[(emb_tile, d), np.dtype[dtype]]
    staged_proj_group_w_ty = np.ndarray[(emb_tile, d * parallel_heads), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * seq_tile,), np.dtype[dtype]]
    wo_ty = np.ndarray[(d, emb_tile), np.dtype[dtype]]
    o_ty = np.ndarray[(seq_tile, emb_tile), np.dtype[dtype]]
    ffn_up_ty = np.ndarray[(seq_tile, ffn_tile), np.dtype[dtype]]
    ffn_b_up_ty = np.ndarray[(emb_tile, ffn_tile), np.dtype[dtype]]
    ffn_b_down_ty = np.ndarray[(ffn_tile, emb_tile), np.dtype[dtype]]
    ln_weights_ty = np.ndarray[(embed_sz,), np.dtype[np.int32]]
    sum_l1_ty = np.ndarray[(seq_tile,), np.dtype[np.float32]]

    zero_kernel = Kernel("zero_bf16", kernel_archive, [qk_ty])
    zero_kernel_q = Kernel("ln_zero_bf16", kernel_archive, [q_ty, np.int32])
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
    if use_staged_hidden_state_projection:
        matmul_init_q_proj_kernel = Kernel(
            "matmul_init_bf16_bf16_q_proj",
            kernel_archive,
            [xq_ty, q_proj_w_ty, q_ty],
        )
        matmul_q_proj_kernel = Kernel(
            "matmul_with_acc_bf16_bf16_q_proj",
            kernel_archive,
            [xq_ty, q_proj_w_ty, q_ty, q_ty],
        )
        matmul_init_k_proj_kernel = Kernel(
            "matmul_init_bf16_bf16_k_proj",
            kernel_archive,
            [xkv_ty, k_proj_w_ty, k_ty],
        )
        matmul_k_proj_kernel = Kernel(
            "matmul_with_acc_bf16_bf16_k_proj",
            kernel_archive,
            [xkv_ty, k_proj_w_ty, k_ty, k_ty],
        )
        matmul_init_v_proj_kernel = Kernel(
            "matmul_init_bf16_bf16_v_proj",
            kernel_archive,
            [xkv_ty, v_proj_w_ty, v_ty],
        )
        matmul_v_proj_kernel = Kernel(
            "matmul_with_acc_bf16_bf16_v_proj",
            kernel_archive,
            [xkv_ty, v_proj_w_ty, v_ty, v_ty],
        )
        eltwise_add_q_kernel = Kernel(
            "eltwise_add_bf16_tile_bias_matrix_q_proj",
            kernel_archive,
            [q_ty, bias_matrix_ty, q_ty, np.int32, np.int32, np.int32],
        )
        eltwise_add_k_kernel = Kernel(
            "eltwise_add_bf16_tile_bias_matrix_kv_proj",
            kernel_archive,
            [k_ty, bias_matrix_ty, k_ty, np.int32, np.int32, np.int32],
        )
        eltwise_add_v_kernel = eltwise_add_k_kernel
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
        [q_ty, s_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
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
    ln_zero_f32_kernel = Kernel("ln_zero_f32", kernel_archive, [sum_l1_ty, np.int32])
    ln_calc_sum_sumsq_kernel = Kernel(
        "ln_calc_sum_sumsq",
        kernel_archive,
        [o_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_add_calc_sum_sumsq_kernel = Kernel(
        "ln_add_calc_sum_sumsq",
        kernel_archive,
        [o_ty, o_ty, sum_l1_ty, sum_l1_ty],
    )
    ln_fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_1outs_fp32weights",
        kernel_archive,
        [o_ty, o_ty, ln_weights_ty, sum_l1_ty, sum_l1_ty, o_ty, np.int32, np.int32],
    )
    ln_fused_add_layer_norm_from_inputs_kernel = Kernel(
        "fused_add_layer_norm_1outs_from_inputs_fp32weights",
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
        (kv_seq_tile // 8, 8 * d),
        (d // 8, 8),
        (8, d),
        (8, 1),
    ]
    ow_dims = [(d // 8, 8 * emb_tile), (emb_tile // 8, 8), (8, emb_tile), (8, 1)]
    o_dims = [(seq_tile // 8, 8 * emb_tile), (8, 8), (emb_tile // 8, 8 * 8), (8, 1)]
    residual_dims = [
        (seq_tile // 8, 8 * emb_tile),
        (emb_tile // 8, 8),
        (8, emb_tile),
        (8, 1),
    ]
    # Some direct non-seq DMA paths can only carry 3 wrap dimensions.
    # These are layout-equivalent 3-D forms of the 4-D O/residual patterns above.
    o_dma_dims = [
        (seq_tile // 8, 8 * emb_tile),
        (emb_tile // 8, 64),
        (64, 1),
    ]
    residual_dma_dims = [
        (seq_tile // 8, 8 * emb_tile),
        (8, emb_tile),
        (emb_tile, 1),
    ]
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
    projQOut = None
    projKOut = None
    projVOut = None
    inXQ = None
    inXK = None
    inXV = None
    inWQ = None
    inWK = None
    inWV = None
    inWKShared = None
    inWVShared = None
    if use_staged_hidden_state_projection:
        inXQ = [
            ObjectFifo(xq_ty, name=f"inXQ{i}", depth=1) for i in range(parallel_heads)
        ]
        inXK = [
            ObjectFifo(xkv_ty, name=f"inXK{i}", depth=1) for i in range(parallel_heads)
        ]
        inXV = [
            ObjectFifo(xkv_ty, name=f"inXV{i}", depth=1) for i in range(parallel_heads)
        ]
        inWQ = [
            ObjectFifo(q_proj_w_ty, name=f"inWQ{i}", depth=1)
            for i in range(parallel_heads)
        ]
        inWK = [
            ObjectFifo(k_proj_w_ty, name=f"inWK{i}", depth=1)
            for i in range(parallel_heads)
        ]
        inWV = [
            ObjectFifo(k_proj_w_ty, name=f"inWV{i}", depth=1)
            for i in range(parallel_heads)
        ]
        if parallel_heads > 1:
            # Multi-head staged projection cannot afford one shim producer per
            # head for every weight ingress stream. Group the staged K/V
            # weights, then split them on-chip back to per-head worker inputs.
            inWKShared = ObjectFifo(staged_proj_group_w_ty, name="inWKAll", depth=1)
            inWK = list(
                inWKShared.cons().split(
                    offsets=[emb_tile * d * i for i in range(parallel_heads)],
                    obj_types=[k_proj_w_ty] * parallel_heads,
                    names=[f"inWK{i}" for i in range(parallel_heads)],
                    depths=[1] * parallel_heads,
                    placement=Tile(col=k_mem_col, row=1),
                )
            )
            inWVShared = ObjectFifo(staged_proj_group_w_ty, name="inWVAll", depth=1)
            inWV = list(
                inWVShared.cons().split(
                    offsets=[emb_tile * d * i for i in range(parallel_heads)],
                    obj_types=[v_proj_w_ty] * parallel_heads,
                    names=[f"inWV{i}" for i in range(parallel_heads)],
                    depths=[1] * parallel_heads,
                    placement=Tile(col=ow_mem_col, row=1),
                )
            )
        projQOut = [
            ObjectFifo(q_ty, name=f"projQOut{i}", depth=1)
            for i in range(parallel_heads)
        ]
        projKOut = [
            ObjectFifo(k_ty, name=f"projKOut{i}", depth=2)
            for i in range(parallel_heads)
        ]
        projVOut = [
            ObjectFifo(v_ty, name=f"projVOut{i}", depth=2)
            for i in range(parallel_heads)
        ]
        memQ = [
            projQOut[i]
            .cons()
            .forward(
                obj_type=q_ty,
                name=f"stagedMemQ{i}",
                depth=of_depth,
                placement=Tile(col=q_mem_col, row=1),
            )
            for i in range(parallel_heads)
        ]
        if not use_staged_hidden_state_kv_cache:
            memK = [
                projKOut[i]
                .cons()
                .forward(
                    obj_type=k_ty,
                    name=f"stagedMemK{i}",
                    depth=max(of_depth, min(q_blocks_per_lane, 4)),
                    placement=Tile(col=k_mem_col, row=1),
                )
                for i in range(parallel_heads)
            ]
            memV = [
                projVOut[i]
                .cons()
                .forward(
                    obj_type=v_ty,
                    name=f"stagedMemV{i}",
                    depth=max(of_depth, min(q_blocks_per_lane, 4)),
                    placement=Tile(col=v_mem_col, row=1),
                )
                for i in range(parallel_heads)
            ]
    inOW = ObjectFifo(wo_stream_ty, name="inOW", depth=of_depth)
    memOW = inOW.cons().split(
        offsets=[d * emb_tile * i for i in range(parallel_heads)],
        obj_types=[wo_ty] * parallel_heads,
        names=[f"memOW{i}" for i in range(parallel_heads)],
        dims_to_stream=[ow_dims] * parallel_heads,
        depths=[o_proj_weight_consumer_depth] * parallel_heads,
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
    outOProjAccumIn = []
    o_proj_stage_chain_depth = o_proj_fifo_depth
    outOPart = [
        ObjectFifo(o_ty, depth=o_proj_stage_chain_depth, name=f"outOPart{i}")
        for i in range(parallel_heads - 1)
    ]
    if use_memtile_o_proj_replay:
        outOProjInput = None
        for i in range(parallel_heads):
            acc_mem_tile = Tile(
                col=accumulation_mem_tiles["o_proj_acc_by_head"][i],
                row=1,
            )
            replay_mem_tile = Tile(col=accumulation_mem_tiles["o_proj_stage"], row=1)
            acc_cons = outOProjAccumOut[i].cons(depth=proj_acc_depth)
            if i == (parallel_heads - 1):
                outOProjAccumIn_i, outOProjInput = acc_cons.split(
                    offsets=[0, 0],
                    placement=replay_mem_tile,
                    depths=[proj_acc_depth, o_proj_fifo_depth],
                    obj_types=[o_ty, o_ty],
                    names=[f"outOProjAccumIn{i}", "outOProjInput"],
                )
            else:
                outOProjAccumIn_i = acc_cons.forward(
                    obj_type=o_ty,
                    name=f"outOProjAccumIn{i}",
                    depth=proj_acc_depth,
                    placement=acc_mem_tile,
                )
            outOProjAccumIn.append(outOProjAccumIn_i)
        if outOProjInput is None:
            raise ValueError(
                "Failed to construct outOProjInput for final O-proj replay"
            )
    else:
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
        outOProjInput = ObjectFifo(
            o_ty,
            name="outOProjInput",
            depth=o_proj_fifo_depth,
        )
    inR = ObjectFifo(o_ty, name="inR", depth=of_depth)
    memR = inR.cons().forward(
        obj_type=o_ty,
        name="memR",
        dims_to_stream=residual_dims,
        depth=of_depth,
        placement=Tile(col=residual_mem_col, row=1),
    )
    outLNBroadcast = ObjectFifo(
        o_ty,
        name="outLNBroadcast",
        depth=1,
        dims_to_stream=o_dma_dims,
    )
    ln1StageOut = outLNBroadcast.cons(depth=1)
    inLNFromDDR = ObjectFifo(o_ty, name="inLNFromDDR", depth=1)
    memLNStage = inLNFromDDR.cons().forward(
        obj_type=o_ty,
        name="memLNStage",
        dims_to_stream=residual_dma_dims,
        depth=2,
        placement=Tile(col=ln1_stage_col, row=1),
    )
    memOutLN = [
        memLNStage.cons(depth=ffn_up_consumer_depth)
        for _ in range(effective_ffn_branches)
    ]

    ffnRFromDDR = ObjectFifo(o_ty, name="ffnRFromDDR", depth=ffn_replay_fifo_depth)
    ffnRIn = ffnRFromDDR.cons(depth=ffn_replay_fifo_depth).forward(
        obj_type=o_ty,
        name="ffnRIn",
        dims_to_stream=residual_dma_dims,
        depth=ffn_replay_fifo_depth,
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
                depth=ffn_up_out_depth,
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
    ffn_down_reduce_depth = ffn_replay_fifo_depth
    ffnDownReduce = [
        ObjectFifo(
            o_ty,
            name=f"ffnDownReduce{branch_idx}",
            depth=ffn_down_reduce_depth,
        )
        for branch_idx in range(effective_ffn_branches - 1)
    ]
    ffnDownOut = ObjectFifo(o_ty, name="ffnDownOut", depth=ffn_replay_fifo_depth)

    outLN2 = ObjectFifo(o_ty, name="outLN2", depth=ffn_replay_fifo_depth)
    memLN2 = outLN2.cons().forward(
        obj_type=o_ty,
        name="memLN2",
        dims_to_stream=o_dims,
        depth=ffn_replay_fifo_depth,
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

    q_tile_elems = seq_tile * d

    def batched_matmul_pv(
        of_p,
        of_v,
        of_scale,
        of_o_out,
        zero,
        zero_size,
        matmul_pv,
        rescale_o,
        idx_buffer,
    ):
        for _ in range_(sys.maxsize):
            idx_buffer[0] = 0
            idx_buffer[1] = 0
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_out_o = of_o_out.acquire(1)
                zero(elem_out_o, zero_size)
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
        add,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
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
                elem_out_o = of_o_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                of_o_out.release(1)
                of_o_acc_in.release(1)

    def matmul_o_proj_emit_twice(
        of_o_in,
        of_ow_in,
        of_o_acc_in,
        of_o_acc_out,
        buffer_to_reduce,
        of_o_out,
        add,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
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
                elem_out_o_acc = of_o_acc_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o_acc, seq_tile * emb_tile)
                of_o_acc_out.release(1)
                if of_o_out is not None:
                    elem_out_o = of_o_out.acquire(1)
                    copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                    of_o_out.release(1)
                of_o_acc_in.release(1)
            for _ in range_(proj_acc_depth):
                elem_in_o_acc = of_o_acc_in.acquire(1)
                if of_o_out is not None:
                    elem_out_o = of_o_out.acquire(1)
                    copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                    of_o_out.release(1)
                of_o_acc_in.release(1)

    def matmul_o_proj_group_head_first(
        of_o_in,
        of_ow_in,
        of_group_acc_in,
        of_partial_out,
        matmul,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_in_o = of_o_in.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_group_acc = of_group_acc_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_partial = of_partial_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_group_acc, elem_out_partial)
                    of_partial_out.release(1)
                    of_ow_in.release(1)
                    of_group_acc_in.release(1)
                of_o_in.release(1)

    def matmul_o_proj_group_head_middle(
        of_o_in,
        of_ow_in,
        of_partial_in,
        of_partial_out,
        matmul,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_in_o = of_o_in.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_partial_in = of_partial_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_partial = of_partial_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_partial_in, elem_out_partial)
                    of_partial_out.release(1)
                    of_ow_in.release(1)
                    of_partial_in.release(1)
                of_o_in.release(1)

    def matmul_o_proj_group_head_final_stage_then_emit_stats(
        of_o_in,
        of_ow_in,
        of_partial_in,
        of_group_acc_out,
        of_stage_out,
        of_stats_out,
        stats_sum_buf,
        stats_sumsq_buf,
        zero_f32,
        calc_sum_sumsq,
        pack_stats,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(stats_sum_buf, seq_tile)
            zero_f32(stats_sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_out_o_acc = of_group_acc_out.acquire(1)
                zero(elem_out_o_acc)
                of_group_acc_out.release(1)
            for _ in range_(num_qkv_head_block_per_parallel_head - 1):
                elem_in_o = of_o_in.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_partial_in = of_partial_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_o_acc = of_group_acc_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_partial_in, elem_out_o_acc)
                    of_group_acc_out.release(1)
                    of_ow_in.release(1)
                    of_partial_in.release(1)
                of_o_in.release(1)

            elem_in_o = of_o_in.acquire(1)
            for _ in range_(proj_acc_depth):
                elem_partial_in = of_partial_in.acquire(1)
                elem_in_ow = of_ow_in.acquire(1)
                elem_stage = of_stage_out.acquire(1)
                matmul(elem_in_o, elem_in_ow, elem_partial_in, elem_stage)
                calc_sum_sumsq(elem_stage, stats_sum_buf, stats_sumsq_buf)
                of_stage_out.release(1)
                of_ow_in.release(1)
                of_partial_in.release(1)
            of_o_in.release(1)

            elem_stats = of_stats_out.acquire(1)
            pack_stats(stats_sum_buf, stats_sumsq_buf, elem_stats, seq_tile)
            of_stats_out.release(1)

    def matmul_o_proj_group_head_final_emit_twice(
        of_o_in,
        of_ow_in,
        of_group_acc_in,
        of_group_acc_out,
        of_partial_in,
        of_o_out,
        zero,
        matmul,
        copy,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(proj_acc_depth):
                elem_out_o_acc = of_group_acc_out.acquire(1)
                zero(elem_out_o_acc)
                of_group_acc_out.release(1)
            for _ in range_(num_qkv_head_block_per_parallel_head):
                elem_in_o = of_o_in.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_partial_in = of_partial_in.acquire(1)
                    elem_in_ow = of_ow_in.acquire(1)
                    elem_out_o_acc = of_group_acc_out.acquire(1)
                    matmul(elem_in_o, elem_in_ow, elem_partial_in, elem_out_o_acc)
                    of_group_acc_out.release(1)
                    of_ow_in.release(1)
                    of_partial_in.release(1)
                of_o_in.release(1)
            for _ in range_(proj_acc_depth):
                elem_in_o_acc = of_group_acc_in.acquire(1)
                elem_out_o_acc = of_group_acc_out.acquire(1)
                copy(elem_in_o_acc, elem_out_o_acc, seq_tile * emb_tile)
                of_group_acc_out.release(1)
                if of_o_out is not None:
                    elem_out_o = of_o_out.acquire(1)
                    copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                    of_o_out.release(1)
                of_group_acc_in.release(1)
            for _ in range_(proj_acc_depth):
                elem_in_o_acc = of_group_acc_in.acquire(1)
                if of_o_out is not None:
                    elem_out_o = of_o_out.acquire(1)
                    copy(elem_in_o_acc, elem_out_o, seq_tile * emb_tile)
                    of_o_out.release(1)
                of_group_acc_in.release(1)

    def core_fn_ln1_from_replayed_inputs(
        of_in_o_proj,
        of_in_residual,
        sum_buf,
        sumsq_buf,
        weights,
        of_out_stage,
        add,
        zero_f32,
        calc_sum_sumsq,
        fused_add_layer_norm,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for col_idx in range_(ln_tiles_per_q_block):
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                add(elem_residual, elem_in, elem_in, seq_tile * emb_tile)
                calc_sum_sumsq(elem_in, sum_buf, sumsq_buf)
                of_in_o_proj.release(1)
                of_in_residual.release(1)
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                add(elem_residual, elem_in, elem_in, seq_tile * emb_tile)
                elem_out = of_out_stage.acquire(1)
                fused_add_layer_norm(
                    elem_in,
                    elem_in,
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

    def core_fn_ln1_from_replayed_inputs_fused_add(
        of_in_o_proj,
        of_in_residual,
        sum_buf,
        sumsq_buf,
        weights,
        of_out_stage,
        zero_f32,
        add_calc_sum_sumsq,
        fused_add_layer_norm_from_inputs,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for col_idx in range_(ln_tiles_per_q_block):
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                add_calc_sum_sumsq(elem_in, elem_residual, sum_buf, sumsq_buf)
                of_in_o_proj.release(1)
                of_in_residual.release(1)
            for col_idx in range_(ln_tiles_per_q_block):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_in = of_in_o_proj.acquire(1)
                elem_residual = of_in_residual.acquire(1)
                elem_out = of_out_stage.acquire(1)
                fused_add_layer_norm_from_inputs(
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

    def core_fn_ffn_down_proj_emit_twice(
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
                elem_curr_acc = of_curr_acc.acquire(1)
                if reduce_in is not None:
                    elem_reduce = reduce_in.acquire(1)
                    add(elem_reduce, elem_curr_acc, elem_curr_acc, seq_tile * emb_tile)
                    reduce_in.release(1)
                elem_new_acc = of_new_acc.acquire(1)
                copy(elem_curr_acc, elem_new_acc, seq_tile * emb_tile)
                of_new_acc.release(1)
                of_curr_acc.release(1)
            for _ in range_(proj_acc_depth):
                elem_curr_acc = of_curr_acc.acquire(1)
                elem_new_acc = of_new_acc.acquire(1)
                copy(elem_curr_acc, elem_new_acc, seq_tile * emb_tile)
                of_new_acc.release(1)
                elem_out = of_out.acquire(1)
                copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                of_curr_acc.release(1)
                of_out.release(1)
            for _ in range_(proj_acc_depth):
                elem_curr_acc = of_curr_acc.acquire(1)
                elem_out = of_out.acquire(1)
                copy(elem_curr_acc, elem_out, seq_tile * emb_tile)
                of_curr_acc.release(1)
                of_out.release(1)

    def core_fn_ffn_down_proj_group_first(
        of_in_a,
        of_in_b,
        of_group_acc_in,
        reduce_out,
        matmul,
        group_count,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(group_count):
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_group_acc = of_group_acc_in.acquire(1)
                    elem_in_b = of_in_b.acquire(1)
                    elem_reduce_out = reduce_out.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_group_acc, elem_reduce_out)
                    reduce_out.release(1)
                    of_in_b.release(1)
                    of_group_acc_in.release(1)
                of_in_a.release(1)

    def core_fn_ffn_down_proj_group_middle(
        of_in_a,
        of_in_b,
        reduce_in,
        reduce_out,
        matmul,
        group_count,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(group_count):
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_reduce_in = reduce_in.acquire(1)
                    elem_in_b = of_in_b.acquire(1)
                    elem_reduce_out = reduce_out.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_reduce_in, elem_reduce_out)
                    reduce_out.release(1)
                    of_in_b.release(1)
                    reduce_in.release(1)
                of_in_a.release(1)

    def core_fn_ffn_down_proj_group_final_emit_twice(
        of_in_a,
        of_in_b,
        reduce_in,
        of_group_acc_out,
        of_stage_out,
        of_stage_in,
        of_out,
        matmul,
        zero,
        copy,
        group_count,
    ):
        for _ in range_(sys.maxsize):
            for _ in range_(proj_acc_depth):
                elem_group_acc = of_group_acc_out.acquire(1)
                zero(elem_group_acc)
                of_group_acc_out.release(1)

            for _ in range_(group_count - 1):
                elem_in_a = of_in_a.acquire(1)
                for _ in range_(proj_acc_depth):
                    elem_reduce_in = reduce_in.acquire(1)
                    elem_in_b = of_in_b.acquire(1)
                    elem_group_acc = of_group_acc_out.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_reduce_in, elem_group_acc)
                    of_group_acc_out.release(1)
                    of_in_b.release(1)
                    reduce_in.release(1)
                of_in_a.release(1)

            elem_in_a = of_in_a.acquire(1)
            for _ in range_(proj_acc_depth):
                elem_reduce_in = reduce_in.acquire(1)
                elem_in_b = of_in_b.acquire(1)
                elem_stage = of_stage_out.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_reduce_in, elem_stage)
                of_stage_out.release(1)
                of_in_b.release(1)
                reduce_in.release(1)
            of_in_a.release(1)
            for _ in range_(proj_acc_depth):
                elem_stage = of_stage_in.acquire(1)
                elem_stage_out = of_stage_out.acquire(1)
                copy(elem_stage, elem_stage_out, seq_tile * emb_tile)
                of_stage_out.release(1)
                elem_out = of_out.acquire(1)
                copy(elem_stage, elem_out, seq_tile * emb_tile)
                of_stage_in.release(1)
                of_out.release(1)
            for _ in range_(proj_acc_depth):
                elem_stage = of_stage_in.acquire(1)
                elem_out = of_out.acquire(1)
                copy(elem_stage, elem_out, seq_tile * emb_tile)
                of_stage_in.release(1)
                of_out.release(1)

    def core_fn_add_norm2_from_replayed_inputs(
        of_in1,
        of_in2,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        add,
        zero_f32,
        calc_sum_sumsq,
        fused_add_layer_norm,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_ffn = of_in1.acquire(1)
                elem_residual = of_in2.acquire(1)
                add(elem_residual, elem_ffn, elem_ffn, seq_tile * emb_tile)
                calc_sum_sumsq(elem_ffn, sum_buf, sumsq_buf)
                of_in2.release(1)
                of_in1.release(1)
            for col_idx in range_(proj_acc_depth):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_ffn = of_in1.acquire(1)
                elem_residual = of_in2.acquire(1)
                add(elem_residual, elem_ffn, elem_ffn, seq_tile * emb_tile)
                elem_out = of_out.acquire(1)
                fused_add_layer_norm(
                    elem_ffn,
                    elem_ffn,
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

    def core_fn_add_norm2_from_replayed_inputs_fused_add(
        of_in1,
        of_in2,
        sum_buf,
        sumsq_buf,
        weights,
        of_out,
        zero_f32,
        add_calc_sum_sumsq,
        fused_add_layer_norm_from_inputs,
    ):
        for _ in range_(sys.maxsize):
            zero_f32(sum_buf, seq_tile)
            zero_f32(sumsq_buf, seq_tile)
            for _ in range_(proj_acc_depth):
                elem_ffn = of_in1.acquire(1)
                elem_residual = of_in2.acquire(1)
                add_calc_sum_sumsq(elem_ffn, elem_residual, sum_buf, sumsq_buf)
                of_in2.release(1)
                of_in1.release(1)
            for col_idx in range_(proj_acc_depth):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_ffn = of_in1.acquire(1)
                elem_residual = of_in2.acquire(1)
                elem_out = of_out.acquire(1)
                fused_add_layer_norm_from_inputs(
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

    def project_q_from_hidden_states(
        of_x_in,
        of_w_in,
        of_q_out,
        matmul_init,
        matmul_acc,
        add_bias,
        bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                elem_out_q = of_q_out.acquire(1)
                elem_in_x = of_x_in.acquire(1)
                elem_in_w = of_w_in.acquire(1)
                matmul_init(elem_in_x, elem_in_w, elem_out_q)
                of_w_in.release(1)
                of_x_in.release(1)
                for _ in range_(proj_acc_depth - 1):
                    elem_in_x = of_x_in.acquire(1)
                    elem_in_w = of_w_in.acquire(1)
                    matmul_acc(elem_in_x, elem_in_w, elem_out_q, elem_out_q)
                    of_w_in.release(1)
                    of_x_in.release(1)
                add_bias(
                    elem_out_q,
                    bias_buffer,
                    elem_out_q,
                    head_group_i32,
                    seq_tile,
                    d,
                )
                of_q_out.release(1)

    def project_k_from_hidden_states(
        of_x_in,
        of_w_in,
        of_k_out,
        matmul_init_k,
        matmul_acc_k,
        add_k_bias,
        k_bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                for _ in range_(num_kv_seq_blocks):
                    elem_out_k = of_k_out.acquire(1)
                    for acc_idx in range_(proj_acc_depth):
                        elem_in_x = of_x_in.acquire(1)
                        elem_in_wk = of_w_in.acquire(1)
                        if acc_idx == 0:
                            matmul_init_k(elem_in_x, elem_in_wk, elem_out_k)
                        else:
                            matmul_acc_k(elem_in_x, elem_in_wk, elem_out_k, elem_out_k)
                        of_w_in.release(1)
                        of_x_in.release(1)
                    add_k_bias(
                        elem_out_k,
                        k_bias_buffer,
                        elem_out_k,
                        head_group_i32,
                        kv_seq_tile,
                        d,
                    )
                    of_k_out.release(1)

    def project_v_from_hidden_states(
        of_x_in,
        of_w_in,
        of_v_out,
        matmul_init_v,
        matmul_acc_v,
        add_v_bias,
        v_bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                for _ in range_(num_kv_seq_blocks):
                    elem_out_v = of_v_out.acquire(1)
                    for acc_idx in range_(proj_acc_depth):
                        elem_in_x = of_x_in.acquire(1)
                        elem_in_wv = of_w_in.acquire(1)
                        if acc_idx == 0:
                            matmul_init_v(elem_in_x, elem_in_wv, elem_out_v)
                        else:
                            matmul_acc_v(elem_in_x, elem_in_wv, elem_out_v, elem_out_v)
                        of_w_in.release(1)
                        of_x_in.release(1)
                    add_v_bias(
                        elem_out_v,
                        v_bias_buffer,
                        elem_out_v,
                        head_group_i32,
                        kv_seq_tile,
                        d,
                    )
                    of_v_out.release(1)

    use_fused_replayed_addnorm_for_winners = use_fused_replayed_addnorm

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
        if use_staged_hidden_state_projection:
            if parallel_heads > 2 or effective_ffn_branches != 1:
                raise ValueError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection currently supports only parallel_heads <= 2 and "
                    "nB_tiles_distributed=1 "
                    f"(got parallel_heads={parallel_heads}, "
                    f"nB_tiles_distributed={effective_ffn_branches})"
                )
            if unified_qr_split is not None:
                raise ValueError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection does not yet support unified_qr_split"
                )
            if group_count != 1:
                raise ValueError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection currently supports only the default single "
                    f"transport group (got group_count={group_count})"
                )
            if tuple(transport_groups[0]["lanes"]) != tuple(range(parallel_seq)):
                raise ValueError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection currently requires the default contiguous lane "
                    f"order (got lanes={transport_groups[0]['lanes']})"
                )
            if parallel_seq > 2:
                raise ValueError(
                    "encoder_pipeline staged_hidden_states sequence-parallel "
                    "projection currently supports only parallel_seq <= 2 "
                    f"(got parallel_seq={parallel_seq})"
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
        ln_stage_base_offset = or_rows_before_ln1_stage * embed_sz
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
                    "qk": _normalize_slot_coords(
                        lane_cfg["qk"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].qk",
                    ),
                    "softmax": _normalize_slot_coords(
                        lane_cfg["softmax"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].softmax",
                    ),
                    "pv": _normalize_slot_coords(
                        lane_cfg["pv"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].pv",
                    ),
                    "o_proj": _normalize_slot_coords(
                        lane_cfg["o_proj"],
                        parallel_heads,
                        f"sequence_parallel.lane_tiles[{lane_idx}].o_proj",
                    ),
                }
            )
            lane_tail_tiles.append(
                {
                    "ln1": tuple(lane_cfg["ln1"]),
                    "ffn_up": _normalize_slot_coords(
                        lane_cfg["ffn_up"],
                        effective_ffn_branches,
                        f"sequence_parallel.lane_tiles[{lane_idx}].ffn_up",
                    ),
                    "ffn_down": _normalize_slot_coords(
                        lane_cfg["ffn_down"],
                        effective_ffn_branches,
                        f"sequence_parallel.lane_tiles[{lane_idx}].ffn_down",
                    ),
                    "ln2": tuple(lane_cfg["ln2"]),
                }
            )
        lane_o_proj_acc_mem_cols = [
            _normalize_slot_mem_cols(
                cols,
                parallel_heads,
                f"sequence_parallel.lane_o_proj_acc_mem_cols[{lane_idx}]",
            )
            for lane_idx, cols in enumerate(
                sequence_parallel["lane_o_proj_acc_mem_cols"]
            )
        ]
        lane_tail_mem_cols = sequence_parallel["lane_tail_mem_cols"]
        lane_o_proj_stage_mem_cols = sequence_parallel.get(
            "lane_o_proj_stage_mem_cols", lane_tail_mem_cols
        )
        lane_o_proj_stage_mem_cols = [int(col) for col in lane_o_proj_stage_mem_cols]
        if len(lane_o_proj_stage_mem_cols) != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one "
                "O-proj replay memtile per sequence lane "
                f"({len(lane_o_proj_stage_mem_cols)} != {parallel_seq})"
            )
        lane_ffn_down_acc_mem_cols = sequence_parallel.get(
            "lane_ffn_down_acc_mem_cols", lane_tail_mem_cols
        )
        lane_ffn_down_acc_mem_cols = [
            _normalize_slot_mem_cols(
                cols,
                effective_ffn_branches,
                f"sequence_parallel.lane_ffn_down_acc_mem_cols[{lane_idx}]",
            )
            for lane_idx, cols in enumerate(lane_ffn_down_acc_mem_cols)
        ]
        lane_ffn_down_stage_mem_cols = sequence_parallel.get(
            "lane_ffn_down_stage_mem_cols",
            (
                [cols[0] for cols in lane_o_proj_acc_mem_cols]
                if parallel_heads > 1
                else lane_tail_mem_cols
            ),
        )
        lane_ffn_down_stage_mem_cols = [
            int(col) for col in lane_ffn_down_stage_mem_cols
        ]
        if len(lane_ffn_down_stage_mem_cols) != parallel_seq:
            raise ValueError(
                "encoder_pipeline sequence-parallel placement must provide one "
                "FFN-down stage memtile per sequence lane "
                f"({len(lane_ffn_down_stage_mem_cols)} != {parallel_seq})"
            )
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
        lane_ffn_down_stage_part = [None] * parallel_seq
        lane_ffn_down_stage = [None] * parallel_seq
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
        lane_ln1_weight_buffers = [None] * parallel_seq
        lane_ln2_weight_buffers = [None] * parallel_seq

        lane_outLN2 = [None] * parallel_seq
        staged_seq_inXQ = None
        staged_seq_inXQAll = None
        staged_seq_inWQAll = None
        staged_seq_inWQ = None
        staged_seq_projQOut = None
        staged_seq_inXKAll = None
        staged_seq_inXK = None
        staged_seq_inWK = None
        staged_seq_projKOut = None
        staged_seq_inXVAll = None
        staged_seq_inXV = None
        staged_seq_inWV = None
        staged_seq_projVOut = None
        staged_seq_q_proj_workers = []
        staged_seq_k_proj_workers = []
        staged_seq_v_proj_workers = []
        q_proj_bias_buffers = None
        k_proj_bias_buffers = None
        v_proj_bias_buffers = None
        if use_staged_hidden_state_projection:
            staged_q_bias_rows = static_q_proj_biases.reshape(
                num_qkv_head_block_per_parallel_head, parallel_heads, d
            )
            staged_k_bias_rows = static_k_proj_biases.reshape(
                num_qkv_head_block_per_parallel_head, parallel_heads, d
            )
            staged_v_bias_rows = static_v_proj_biases.reshape(
                num_qkv_head_block_per_parallel_head, parallel_heads, d
            )
            q_proj_bias_buffers = [
                [
                    Buffer(
                        type=bias_matrix_ty,
                        initial_value=np.ascontiguousarray(
                            staged_q_bias_rows[:, head_idx, :]
                        ),
                        name=f"static_q_proj_bias_seq_l{lane_idx}_h{head_idx}",
                    )
                    for head_idx in range(parallel_heads)
                ]
                for lane_idx in range(parallel_seq)
            ]
            k_proj_bias_buffers = [
                Buffer(
                    type=bias_matrix_ty,
                    initial_value=np.ascontiguousarray(
                        staged_k_bias_rows[:, head_idx, :]
                    ),
                    name=f"static_k_proj_bias_seq_h{head_idx}",
                )
                for head_idx in range(parallel_heads)
            ]
            v_proj_bias_buffers = [
                Buffer(
                    type=bias_matrix_ty,
                    initial_value=np.ascontiguousarray(
                        staged_v_bias_rows[:, head_idx, :]
                    ),
                    name=f"static_v_proj_bias_seq_h{head_idx}",
                )
                for head_idx in range(parallel_heads)
            ]
            if parallel_heads == 1:
                staged_seq_inWQ = [
                    [
                        ObjectFifo(
                            q_proj_w_ty,
                            name=f"inWQSeqL{lane_idx}H0",
                            depth=1,
                        )
                    ]
                    for lane_idx in range(parallel_seq)
                ]
            else:
                staged_seq_inWQAll = [None] * parallel_heads
            staged_seq_projQOut = [
                [
                    ObjectFifo(
                        q_ty,
                        name=f"projQOutSeqL{lane_idx}H{head_idx}",
                        depth=1,
                    )
                    for head_idx in range(parallel_heads)
                ]
                for lane_idx in range(parallel_seq)
            ]
            staged_seq_inXKAll = ObjectFifo(xkv_ty, name="inXKSeqAll", depth=1)
            staged_seq_inWK = [
                ObjectFifo(k_proj_w_ty, name=f"inWKSeqH{head_idx}", depth=1)
                for head_idx in range(parallel_heads)
            ]
            staged_seq_projKOut = [
                ObjectFifo(k_ty, name=f"projKOutSeqH{head_idx}", depth=2)
                for head_idx in range(parallel_heads)
            ]
            staged_seq_inXVAll = ObjectFifo(xkv_ty, name="inXVSeqAll", depth=1)
            staged_seq_inWV = [
                ObjectFifo(v_proj_w_ty, name=f"inWVSeqH{head_idx}", depth=1)
                for head_idx in range(parallel_heads)
            ]
            staged_seq_projVOut = [
                ObjectFifo(v_ty, name=f"projVOutSeqH{head_idx}", depth=2)
                for head_idx in range(parallel_heads)
            ]
        grouped_o_proj_acc = (
            parallel_heads > 1 and o_proj_acc_group_size == parallel_heads
        )
        grouped_ffn_down_acc = (
            effective_ffn_branches > 1
            and ffn_down_acc_group_size == effective_ffn_branches
        )
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
                dims_to_stream=[residual_dims] * parallel_seq,
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
                depth=o_proj_weight_consumer_depth,
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

            if use_staged_hidden_state_projection and staged_seq_inXQAll is None:
                staged_seq_inXQAll = ObjectFifo(
                    np.ndarray[(parallel_seq * seq_tile, emb_tile), np.dtype[dtype]],
                    name="inXQSeqAll",
                    depth=1,
                )
                if parallel_heads == 1:
                    staged_seq_inXQ = [
                        [fifo]
                        for fifo in staged_seq_inXQAll.cons().split(
                            offsets=[
                                lane_idx * seq_tile * emb_tile
                                for lane_idx in range(parallel_seq)
                            ],
                            obj_types=[xq_ty] * parallel_seq,
                            names=[
                                f"inXQSeqL{lane_idx}"
                                for lane_idx in range(parallel_seq)
                            ],
                            depths=[1] * parallel_seq,
                            placement=Tile(
                                col=transport_group["joined_q_mem_col"], row=1
                            ),
                        )
                    ]
                else:
                    staged_seq_inXQ_split = list(
                        staged_seq_inXQAll.cons().split(
                            offsets=[
                                lane_idx * seq_tile * emb_tile
                                for lane_idx in range(parallel_seq)
                                for _ in range(parallel_heads)
                            ],
                            obj_types=[xq_ty] * (parallel_seq * parallel_heads),
                            names=[
                                f"inXQSeqL{lane_idx}H{head_idx}"
                                for lane_idx in range(parallel_seq)
                                for head_idx in range(parallel_heads)
                            ],
                            depths=[1] * (parallel_seq * parallel_heads),
                            placement=Tile(col=7, row=1),
                        )
                    )
                    staged_seq_inXQ = [
                        [
                            staged_seq_inXQ_split[lane_idx * parallel_heads + head_idx]
                            for head_idx in range(parallel_heads)
                        ]
                        for lane_idx in range(parallel_seq)
                    ]
                    staged_seq_inWQ = [
                        [None] * parallel_heads for _ in range(parallel_seq)
                    ]
                    for head_idx in range(parallel_heads):
                        if head_idx == 0:
                            for lane_idx in range(parallel_seq):
                                staged_seq_inWQ[lane_idx][head_idx] = ObjectFifo(
                                    q_proj_w_ty,
                                    name=f"inWQSeqL{lane_idx}H{head_idx}",
                                    depth=1,
                                )
                            continue
                        staged_seq_inWQAll[head_idx] = ObjectFifo(
                            q_proj_w_ty,
                            name=f"inWQSeqAllH{head_idx}",
                            depth=1,
                        )
                        staged_seq_inWQ_lane = list(
                            staged_seq_inWQAll[head_idx]
                            .cons()
                            .split(
                                offsets=[0] * parallel_seq,
                                obj_types=[q_proj_w_ty] * parallel_seq,
                                names=[
                                    f"inWQSeqL{lane_idx}H{head_idx}"
                                    for lane_idx in range(parallel_seq)
                                ],
                                depths=[1] * parallel_seq,
                                placement=Tile(
                                    col=6,
                                    row=1,
                                ),
                            )
                        )
                        for lane_idx in range(parallel_seq):
                            staged_seq_inWQ[lane_idx][head_idx] = staged_seq_inWQ_lane[
                                lane_idx
                            ]
                if parallel_heads == 1:
                    staged_seq_inXK = [staged_seq_inXKAll]
                    staged_seq_inXV = [staged_seq_inXVAll]
                else:
                    staged_seq_inXK = list(
                        staged_seq_inXKAll.cons().split(
                            offsets=[0] * parallel_heads,
                            obj_types=[xkv_ty] * parallel_heads,
                            names=[
                                f"inXKSeqH{head_idx}"
                                for head_idx in range(parallel_heads)
                            ],
                            depths=[1] * parallel_heads,
                            placement=Tile(
                                col=transport_group["shared_ingress_cols"]["k"], row=1
                            ),
                        )
                    )
                    staged_seq_inXV = list(
                        staged_seq_inXVAll.cons().split(
                            offsets=[0] * parallel_heads,
                            obj_types=[xkv_ty] * parallel_heads,
                            names=[
                                f"inXVSeqH{head_idx}"
                                for head_idx in range(parallel_heads)
                            ],
                            depths=[1] * parallel_heads,
                            placement=Tile(
                                col=transport_group["shared_ingress_cols"]["v"], row=1
                            ),
                        )
                    )

            if unified_qr_split is None:
                if use_staged_hidden_state_projection:
                    group_inQSeq.append(None)
                    for lane_idx in lanes:
                        for head_idx in range(parallel_heads):
                            if parallel_heads > 1:
                                if parallel_seq > 1 and head_idx == 1:
                                    # For the staged seq-par 2-head branch, feed the
                                    # second projected-Q head directly into QK instead
                                    # of relaying it through another memtile. This
                                    # removes the last hot memtile on the staged Q path.
                                    memQ_by_lane[lane_idx][head_idx] = (
                                        staged_seq_projQOut[lane_idx][head_idx]
                                    )
                                else:
                                    memQ_by_lane[lane_idx][head_idx] = (
                                        staged_seq_projQOut[lane_idx][head_idx]
                                        .cons()
                                        .forward(
                                            obj_type=q_ty,
                                            name=f"stagedMemQSeqL{lane_idx}H{head_idx}",
                                            depth=of_depth,
                                            placement=Tile(col=6 + head_idx, row=1),
                                        )
                                    )
                            else:
                                memQ_by_lane[lane_idx][head_idx] = (
                                    staged_seq_projQOut[lane_idx][head_idx]
                                    .cons()
                                    .forward(
                                        obj_type=q_ty,
                                        name=f"stagedMemQSeqL{lane_idx}H{head_idx}",
                                        depth=of_depth,
                                        placement=Tile(
                                            col=transport_group["joined_q_mem_col"],
                                            row=1,
                                        ),
                                    )
                                )
                else:
                    inQSeq = ObjectFifo(
                        q_batch_ty, name=f"inQSeq{suffix}", depth=of_depth
                    )
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

            if use_staged_hidden_state_projection and parallel_heads > 1:
                group_inKSeq.append(None)
                group_memKSeq.append(
                    [
                        staged_seq_projKOut[head_idx]
                        .cons()
                        .forward(
                            obj_type=k_ty,
                            name=f"stagedMemKSeq{suffix}H{head_idx}",
                            depth=shared_forward_depth,
                            placement=Tile(
                                col=transport_group["shared_ingress_cols"]["k"], row=1
                            ),
                        )
                        for head_idx in range(parallel_heads)
                    ]
                )
            else:
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

            if use_staged_hidden_state_projection and parallel_heads > 1:
                group_inVSeq.append(None)
                group_memVSeq.append(
                    [
                        staged_seq_projVOut[head_idx]
                        .cons()
                        .forward(
                            obj_type=v_ty,
                            name=f"stagedMemVSeq{suffix}H{head_idx}",
                            depth=shared_forward_depth,
                            placement=Tile(
                                col=transport_group["shared_ingress_cols"]["v"], row=1
                            ),
                        )
                        for head_idx in range(parallel_heads)
                    ]
                )
            else:
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
                        depth=o_proj_weight_consumer_depth,
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
                        depths=[o_proj_weight_consumer_depth] * parallel_heads,
                        placement=Tile(
                            col=transport_group["shared_ingress_cols"]["w_o"], row=1
                        ),
                    )
                    group_memOWSeq.append(list(memOWSeq))
            else:
                group_inOWSeq.append(None)
                group_memOWSeq.append([unified_memOWSeq])

            weight_b_up_cols = _normalize_slot_mem_cols(
                transport_group["weight_mem_cols"]["b_up"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].weight_mem_cols.b_up",
            )
            weight_b_down_cols = _normalize_slot_mem_cols(
                transport_group["weight_mem_cols"]["b_down"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].weight_mem_cols.b_down",
            )
            weight_b_up_shim_cols = _normalize_slot_mem_cols(
                group_shim_cols["b_up"],
                effective_ffn_branches,
                f"sequence_parallel.transport_groups[{group_idx}].shim_cols.b_up",
            )
            weight_b_down_shim_cols = _normalize_slot_mem_cols(
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
                    dims_to_stream=[residual_dims] * group_size,
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
                dims_to_stream=[residual_dims] * group_size,
                depths=[ffn_up_consumer_depth] * group_size,
                placement=Tile(
                    col=transport_group["joined_or_mem_cols"]["ln1_refill"], row=1
                ),
            )
            for local_idx, lane_idx in enumerate(lanes):
                lane_memOutLN[lane_idx] = lane_memOutLN_group[local_idx]

            ffnRFromDDRSeq = ObjectFifo(
                joined_o_ty,
                name=f"ffnRFromDDRSeq{suffix}",
                depth=ffn_replay_fifo_depth,
            )
            group_ffnRFromDDRSeq.append(ffnRFromDDRSeq)
            lane_ffn_r_in_group = ffnRFromDDRSeq.cons().split(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"ffnRInSeqL{lane_idx}" for lane_idx in lanes],
                dims_to_stream=[residual_dims] * group_size,
                depths=[ffn_replay_fifo_depth] * group_size,
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
                depth=ffn_replay_fifo_depth,
                dims_to_stream=o_dims,
            )
            group_memLN2Joined.append(memLN2Joined)
            lane_outLN2_group = memLN2Joined.prod().join(
                offsets=joined_offsets,
                obj_types=[o_ty] * group_size,
                names=[f"outLN2SeqL{lane_idx}" for lane_idx in lanes],
                depths=[ffn_replay_fifo_depth] * group_size,
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
                    depth=o_proj_fifo_depth,
                )
            )
            lane_ffn_down_out[lane_idx] = ObjectFifo(
                o_ty,
                name=f"ffnDownOutSeqL{lane_idx}",
                depth=ffn_replay_fifo_depth,
            )
            if grouped_ffn_down_acc:
                lane_ffn_down_stage_part[lane_idx] = ObjectFifo(
                    o_ty,
                    depth=1,
                    name=f"ffnDownStagePartSeqL{lane_idx}",
                )
                lane_ffn_down_stage[lane_idx] = (
                    lane_ffn_down_stage_part[lane_idx]
                    .cons(depth=proj_acc_depth)
                    .forward(
                        obj_type=o_ty,
                        name=f"ffnDownStageSeqL{lane_idx}",
                        depth=proj_acc_depth,
                        placement=Tile(
                            col=lane_ffn_down_stage_mem_cols[lane_idx],
                            row=1,
                        ),
                    )
                )
            if grouped_o_proj_acc:
                final_head_idx = parallel_heads - 1
                lane_acc_fifo = ObjectFifo(
                    o_ty,
                    depth=1,
                    name=f"outOProjAccumOutSeqL{lane_idx}H{final_head_idx}",
                )
                lane_o_proj_acc_out[lane_idx][final_head_idx] = lane_acc_fifo
                lane_o_proj_acc_in[lane_idx][final_head_idx] = lane_acc_fifo.cons(
                    depth=proj_acc_depth
                ).forward(
                    obj_type=o_ty,
                    name=f"outOProjAccumInSeqL{lane_idx}H{final_head_idx}",
                    depth=proj_acc_depth,
                    placement=Tile(
                        col=lane_o_proj_acc_mem_cols[lane_idx][final_head_idx],
                        row=1,
                    ),
                )
            if grouped_ffn_down_acc:
                final_branch_idx = effective_ffn_branches - 1
                lane_ffn_down_part[lane_idx][final_branch_idx] = ObjectFifo(
                    o_ty,
                    name=f"ffnDownPartSeqL{lane_idx}B{final_branch_idx}",
                    depth=1,
                )
                lane_ffn_down_acc[lane_idx][final_branch_idx] = (
                    lane_ffn_down_part[lane_idx][final_branch_idx]
                    .cons(depth=proj_acc_depth)
                    .forward(
                        obj_type=o_ty,
                        name=f"ffnDownAccumSeqL{lane_idx}B{final_branch_idx}",
                        depth=proj_acc_depth,
                        placement=Tile(
                            col=lane_ffn_down_acc_mem_cols[lane_idx][final_branch_idx],
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
                if (not grouped_o_proj_acc) or head_idx != (parallel_heads - 1):
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
                            zero_kernel_q,
                            q_tile_elems,
                            matmul_pv_kernel,
                            rescale_o_kernel,
                            idx_buffer_pv,
                        ],
                        placement=Tile(*lane_front_tiles[lane_idx]["pv"][head_idx]),
                        stack_size=0xD00,
                        while_true=False,
                    )
                )
                if grouped_o_proj_acc:
                    if head_idx == 0:
                        lane_o_proj_workers.append(
                            Worker(
                                matmul_o_proj_group_head_first,
                                fn_args=[
                                    lane_outOProj[lane_idx][head_idx].cons(),
                                    group_memOWSeq[lane_group_idx][head_idx].cons(),
                                    lane_o_proj_acc_in[lane_idx][
                                        parallel_heads - 1
                                    ].cons(depth=1),
                                    lane_o_proj_partial[lane_idx][head_idx].prod(),
                                    matmul_kernel_o_proj,
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
                                matmul_o_proj_group_head_final_emit_twice,
                                fn_args=[
                                    lane_outOProj[lane_idx][head_idx].cons(),
                                    group_memOWSeq[lane_group_idx][head_idx].cons(),
                                    lane_o_proj_acc_in[lane_idx][head_idx].cons(
                                        depth=1
                                    ),
                                    lane_o_proj_acc_out[lane_idx][head_idx].prod(),
                                    lane_o_proj_partial[lane_idx][head_idx - 1].cons(),
                                    lane_ln1_input[lane_idx].prod(),
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
                                matmul_o_proj_group_head_middle,
                                fn_args=[
                                    lane_outOProj[lane_idx][head_idx].cons(),
                                    group_memOWSeq[lane_group_idx][head_idx].cons(),
                                    lane_o_proj_partial[lane_idx][head_idx - 1].cons(),
                                    lane_o_proj_partial[lane_idx][head_idx].prod(),
                                    matmul_kernel_o_proj,
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
                            matmul_o_proj_emit_twice,
                            fn_args=(
                                [
                                    lane_outOProj[lane_idx][head_idx].cons(),
                                    group_memOWSeq[lane_group_idx][head_idx].cons(),
                                    lane_o_proj_acc_in[lane_idx][head_idx].cons(
                                        depth=1
                                    ),
                                    lane_o_proj_acc_out[lane_idx][head_idx].prod(),
                                    (
                                        lane_o_proj_partial[lane_idx][
                                            head_idx - 1
                                        ].cons()
                                        if head_idx > 0
                                        else None
                                    ),
                                    lane_ln1_input[lane_idx].prod(),
                                    eltwise_add_vector_kernel,
                                    zero_kernel_o_proj,
                                    matmul_kernel_o_proj,
                                    mem_copy_o_proj,
                                ]
                            ),
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
                                (
                                    lane_o_proj_partial[lane_idx][head_idx - 1].cons()
                                    if head_idx > 0
                                    else None
                                ),
                                lane_o_proj_partial[lane_idx][head_idx].prod(),
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

            lane_ln1_weight_buffer = Buffer(
                type=ln_weights_ty,
                use_write_rtp=use_runtime_ln_weights,
                initial_value=None if use_runtime_ln_weights else static_ln1_weights,
                name=f"static_ln1_weights_seq_l{lane_idx}",
            )
            lane_ln2_weight_buffer = Buffer(
                type=ln_weights_ty,
                use_write_rtp=use_runtime_ln_weights,
                initial_value=None if use_runtime_ln_weights else static_ln2_weights,
                name=f"static_ln2_weights_seq_l{lane_idx}",
            )
            lane_ln1_weight_buffers[lane_idx] = lane_ln1_weight_buffer
            lane_ln2_weight_buffers[lane_idx] = lane_ln2_weight_buffer
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

            if lane_ln1_input[lane_idx] is None:
                raise ValueError(
                    f"Failed to construct LN1 replay FIFO for sequence lane {lane_idx}"
                )
            lane_ln1_workers.append(
                Worker(
                    (
                        core_fn_ln1_from_replayed_inputs_fused_add
                        if use_fused_replayed_addnorm_for_winners
                        else core_fn_ln1_from_replayed_inputs
                    ),
                    fn_args=(
                        [
                            lane_ln1_input[lane_idx].cons(),
                            lane_in_r[lane_idx].cons(depth=1),
                            lane_ln1_norm_sum_buffer,
                            lane_ln1_norm_sumsq_buffer,
                            lane_ln1_weight_buffer,
                            lane_ln1_stage_out[lane_idx].prod(),
                            ln_zero_f32_kernel,
                            ln_add_calc_sum_sumsq_kernel,
                            ln_fused_add_layer_norm_from_inputs_kernel,
                        ]
                        if use_fused_replayed_addnorm_for_winners
                        else [
                            lane_ln1_input[lane_idx].cons(),
                            lane_in_r[lane_idx].cons(depth=1),
                            lane_ln1_norm_sum_buffer,
                            lane_ln1_norm_sumsq_buffer,
                            lane_ln1_weight_buffer,
                            lane_ln1_stage_out[lane_idx].prod(),
                            eltwise_add_vector_kernel,
                            ln_zero_f32_kernel,
                            ln_calc_sum_sumsq_kernel,
                            ln_fused_add_layer_norm_kernel,
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
                    depth=ffn_up_out_depth,
                )
                needs_lane_ffn_down_acc = (not grouped_ffn_down_acc) or (
                    branch_idx == (effective_ffn_branches - 1)
                )
                if (
                    needs_lane_ffn_down_acc
                    and lane_ffn_down_part[lane_idx][branch_idx] is None
                ):
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
                            lane_memOutLN[lane_idx].cons(depth=ffn_up_consumer_depth),
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

                if grouped_ffn_down_acc:
                    if branch_idx == 0:
                        lane_ffn_down_workers.append(
                            Worker(
                                core_fn_ffn_down_proj_group_first,
                                fn_args=[
                                    lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                    group_memBDownSeq[lane_group_idx][
                                        branch_idx
                                    ].cons(),
                                    lane_ffn_down_acc[lane_idx][
                                        effective_ffn_branches - 1
                                    ].cons(depth=1),
                                    lane_ffn_down_reduce[lane_idx][branch_idx].prod(),
                                    ffn_matmul_kernel_down_proj,
                                    ffn_col_group_count,
                                ],
                                placement=Tile(
                                    *lane_tail_tiles[lane_idx]["ffn_down"][branch_idx]
                                ),
                                stack_size=0xF00,
                                while_true=False,
                            )
                        )
                    elif branch_idx == (effective_ffn_branches - 1):
                        lane_ffn_down_workers.append(
                            Worker(
                                core_fn_ffn_down_proj_group_final_emit_twice,
                                fn_args=[
                                    lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                    group_memBDownSeq[lane_group_idx][
                                        branch_idx
                                    ].cons(),
                                    lane_ffn_down_reduce[lane_idx][
                                        branch_idx - 1
                                    ].cons(),
                                    lane_ffn_down_part[lane_idx][branch_idx].prod(),
                                    lane_ffn_down_stage_part[lane_idx].prod(),
                                    lane_ffn_down_stage[lane_idx].cons(depth=1),
                                    lane_ffn_down_out[lane_idx].prod(
                                        ffn_down_output_producer_depth
                                    ),
                                    ffn_matmul_kernel_down_proj,
                                    ffn_zero_kernel_down_proj,
                                    mem_copy_o_proj,
                                    ffn_col_group_count,
                                ],
                                placement=Tile(
                                    *lane_tail_tiles[lane_idx]["ffn_down"][branch_idx]
                                ),
                                stack_size=0xF00,
                                while_true=False,
                            )
                        )
                    else:
                        lane_ffn_down_workers.append(
                            Worker(
                                core_fn_ffn_down_proj_group_middle,
                                fn_args=[
                                    lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                    group_memBDownSeq[lane_group_idx][
                                        branch_idx
                                    ].cons(),
                                    lane_ffn_down_reduce[lane_idx][
                                        branch_idx - 1
                                    ].cons(),
                                    lane_ffn_down_reduce[lane_idx][branch_idx].prod(),
                                    ffn_matmul_kernel_down_proj,
                                    ffn_col_group_count,
                                ],
                                placement=Tile(
                                    *lane_tail_tiles[lane_idx]["ffn_down"][branch_idx]
                                ),
                                stack_size=0xF00,
                                while_true=False,
                            )
                        )
                else:
                    lane_ffn_down_workers.append(
                        Worker(
                            (
                                core_fn_ffn_down_proj_emit_twice
                                if branch_idx == (effective_ffn_branches - 1)
                                else core_fn_ffn_down_proj
                            ),
                            fn_args=(
                                [
                                    lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                    group_memBDownSeq[lane_group_idx][
                                        branch_idx
                                    ].cons(),
                                    lane_ffn_down_acc[lane_idx][branch_idx].cons(
                                        depth=1
                                    ),
                                    lane_ffn_down_part[lane_idx][branch_idx].prod(),
                                    (
                                        lane_ffn_down_reduce[lane_idx][
                                            branch_idx - 1
                                        ].cons()
                                        if branch_idx > 0
                                        else None
                                    ),
                                    lane_ffn_down_out[lane_idx].prod(
                                        ffn_down_output_producer_depth
                                    ),
                                    ffn_matmul_init_kernel_down_proj,
                                    ffn_matmul_kernel_down_proj,
                                    eltwise_add_vector_kernel,
                                    mem_copy_o_proj,
                                    ffn_col_group_count,
                                ]
                                if branch_idx == (effective_ffn_branches - 1)
                                else [
                                    lane_ffn_up_out[lane_idx][branch_idx].cons(),
                                    group_memBDownSeq[lane_group_idx][
                                        branch_idx
                                    ].cons(),
                                    lane_ffn_down_acc[lane_idx][branch_idx].cons(
                                        depth=1
                                    ),
                                    lane_ffn_down_part[lane_idx][branch_idx].prod(),
                                    (
                                        lane_ffn_down_reduce[lane_idx][
                                            branch_idx - 1
                                        ].cons()
                                        if branch_idx > 0
                                        else None
                                    ),
                                    (
                                        lane_ffn_down_reduce[lane_idx][
                                            branch_idx
                                        ].prod()
                                        if branch_idx < (effective_ffn_branches - 1)
                                        else lane_ffn_down_out[lane_idx].prod(
                                            ffn_down_output_producer_depth
                                        )
                                    ),
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
                    (
                        core_fn_add_norm2_from_replayed_inputs_fused_add
                        if use_fused_replayed_addnorm_for_winners
                        else core_fn_add_norm2_from_replayed_inputs
                    ),
                    fn_args=(
                        [
                            lane_ffn_down_out[lane_idx].cons(),
                            lane_ffn_r_in[lane_idx].cons(),
                            lane_ln2_sum_buffer,
                            lane_ln2_sumsq_buffer,
                            lane_ln2_weight_buffer,
                            lane_outLN2[lane_idx].prod(),
                            ln_zero_f32_kernel,
                            ln_add_calc_sum_sumsq_kernel,
                            ln_fused_add_layer_norm_from_inputs_kernel,
                        ]
                        if use_fused_replayed_addnorm_for_winners
                        else [
                            lane_ffn_down_out[lane_idx].cons(),
                            lane_ffn_r_in[lane_idx].cons(),
                            lane_ln2_sum_buffer,
                            lane_ln2_sumsq_buffer,
                            lane_ln2_weight_buffer,
                            lane_outLN2[lane_idx].prod(),
                            eltwise_add_vector_kernel,
                            ln_zero_f32_kernel,
                            ln_calc_sum_sumsq_kernel,
                            ln_fused_add_layer_norm_kernel,
                        ]
                    ),
                    placement=Tile(*lane_tail_tiles[lane_idx]["ln2"]),
                    stack_size=0xF00,
                    while_true=False,
                )
            )

        if use_staged_hidden_state_projection:
            if parallel_heads == 1:
                staged_seq_wq_shim_cols = [k_shim_col, ln1_stage_shim_col][
                    :parallel_seq
                ]
                staged_seq_shared_wq_shim_cols = None
            else:
                staged_seq_wq_shim_cols = [k_shim_col, v_shim_col][:parallel_seq]
                staged_seq_shared_wq_shim_cols = [None, ln1_stage_shim_col][
                    :parallel_heads
                ]
            # In staged seq-par mode, route projected-X ingress away from the
            # residual/output shim so the front-end does not exhaust shim DMA
            # resources before the rest of the lane graph is even considered.
            staged_seq_xq_shim_cols = [q_shim_col, ln1_stage_shim_col][:parallel_seq]
            staged_seq_xk_shim_cols = [v_shim_col, residual_shim_col][:parallel_heads]
            staged_seq_wk_shim_cols = [ow_shim_col, output_shim_col][:parallel_heads]
            staged_seq_xv_shim_cols = [ln1_stage_shim_col, k_shim_col][:parallel_heads]
            staged_seq_wv_shim_cols = [bdown_shim_col, bup_shim_col][:parallel_heads]

            if parallel_heads == 1:
                q_proj_tiles = [
                    [Tile(col=2 + lane_idx, row=2)] for lane_idx in range(parallel_seq)
                ]
                k_proj_tiles = [Tile(col=6, row=2)]
                v_proj_tiles = [Tile(col=7, row=2)]
            else:
                q_proj_tiles = [
                    [Tile(col=3, row=2), Tile(col=3, row=3)],
                    [Tile(col=7, row=2), Tile(col=7, row=3)],
                ][:parallel_seq]
                k_proj_tiles = [Tile(col=2, row=4), Tile(col=6, row=4)][:parallel_heads]
                v_proj_tiles = [Tile(col=3, row=5), Tile(col=7, row=5)][:parallel_heads]

            for lane_idx in range(parallel_seq):
                for head_idx in range(parallel_heads):
                    staged_seq_q_proj_workers.append(
                        Worker(
                            project_q_from_hidden_states,
                            fn_args=[
                                staged_seq_inXQ[lane_idx][head_idx].cons(),
                                staged_seq_inWQ[lane_idx][head_idx].cons(),
                                staged_seq_projQOut[lane_idx][head_idx].prod(),
                                matmul_init_q_proj_kernel,
                                matmul_q_proj_kernel,
                                eltwise_add_q_kernel,
                                q_proj_bias_buffers[lane_idx][head_idx],
                            ],
                            placement=q_proj_tiles[lane_idx][head_idx],
                            stack_size=0xD00,
                            while_true=False,
                        )
                    )
            for head_idx in range(parallel_heads):
                staged_seq_k_proj_workers.append(
                    Worker(
                        project_k_from_hidden_states,
                        fn_args=[
                            staged_seq_inXK[head_idx].cons(),
                            staged_seq_inWK[head_idx].cons(),
                            staged_seq_projKOut[head_idx].prod(),
                            matmul_init_k_proj_kernel,
                            matmul_k_proj_kernel,
                            eltwise_add_k_kernel,
                            k_proj_bias_buffers[head_idx],
                        ],
                        placement=k_proj_tiles[head_idx],
                        stack_size=0xD00,
                        while_true=False,
                    )
                )
                staged_seq_v_proj_workers.append(
                    Worker(
                        project_v_from_hidden_states,
                        fn_args=[
                            staged_seq_inXV[head_idx].cons(),
                            staged_seq_inWV[head_idx].cons(),
                            staged_seq_projVOut[head_idx].prod(),
                            matmul_init_v_proj_kernel,
                            matmul_v_proj_kernel,
                            eltwise_add_v_kernel,
                            v_proj_bias_buffers[head_idx],
                        ],
                        placement=v_proj_tiles[head_idx],
                        stack_size=0xD00,
                        while_true=False,
                    )
                )

        qkv_tensor_shape = (3 * seq_len, embed_sz)
        staged_seq_q_stage_base_offset = seq_len * embed_sz
        staged_seq_k_stage_base_offset = 2 * seq_len * embed_sz
        staged_seq_v_stage_base_offset = 3 * seq_len * embed_sz
        staged_hidden_state_tensor_shape = (seq_len, embed_sz)
        attn_weight_tensor_shape = (4 * embed_sz, embed_sz)
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
        staged_hidden_state_q_tiles_base = None
        staged_hidden_state_q_batch_tiles_base = None
        staged_hidden_state_q_tiles = None
        staged_hidden_state_q_batch_tiles = None
        staged_seq_q_tiles = None
        staged_seq_q_batch_tiles = None
        if use_staged_hidden_state_projection:
            staged_hidden_state_q_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (seq_tile, emb_tile),
                (1, proj_acc_depth),
            )
            staged_hidden_state_q_batch_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (parallel_seq * seq_tile, emb_tile),
                (1, proj_acc_depth),
            )
            staged_seq_q_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (seq_tile, d),
                (1, parallel_heads),
            )
            staged_hidden_state_q_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        staged_hidden_state_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in staged_hidden_state_q_tiles_base
                ]
            )
            staged_hidden_state_q_batch_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        staged_hidden_state_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in staged_hidden_state_q_batch_tiles_base
                ]
            )
            staged_seq_q_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + staged_seq_q_stage_base_offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in staged_seq_q_tiles_base
                ]
            )
            staged_seq_q_batch_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + staged_seq_q_stage_base_offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in q_batch_tiles_base
                ]
            )
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
        staged_hidden_state_kv_tiles = None
        staged_seq_k_tiles = None
        staged_seq_v_tiles = None
        staged_seq_wq_tiles = None
        staged_seq_wk_tiles = None
        staged_seq_wv_tiles = None
        staged_seq_repeated_wk_tiles = None
        staged_seq_repeated_wv_tiles = None
        staged_seq_hidden_state_kv_fill_taps = None
        staged_seq_wk_fill_taps = None
        staged_seq_wv_fill_taps = None
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
        if use_staged_hidden_state_projection:
            staged_hidden_state_kv_tiles_base = TensorTiler2D.group_tiler(
                (seq_len, embed_sz),
                (kv_seq_tile, emb_tile),
                (num_kv_seq_blocks, proj_acc_depth),
            )
            staged_hidden_state_kv_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        staged_hidden_state_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in staged_hidden_state_kv_tiles_base
                ]
            )
            staged_seq_wq_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        attn_weight_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in TensorTiler2D.group_tiler(
                        (embed_sz, embed_sz),
                        (emb_tile, d),
                        (proj_acc_depth, 1),
                    )
                ]
            )
            staged_seq_wk_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        attn_weight_tensor_shape,
                        offset=tap.offset + embed_sz * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in TensorTiler2D.group_tiler(
                        (embed_sz, embed_sz),
                        (emb_tile, d),
                        (proj_acc_depth, 1),
                    )
                ]
            )
            staged_seq_wv_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        attn_weight_tensor_shape,
                        offset=tap.offset + 2 * embed_sz * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in TensorTiler2D.group_tiler(
                        (embed_sz, embed_sz),
                        (emb_tile, d),
                        (proj_acc_depth, 1),
                    )
                ]
            )
            staged_seq_repeated_wk_tiles = [
                repeat_outer_tap(
                    staged_seq_wk_tiles[head_group_idx],
                    attn_weight_tensor_shape,
                    num_kv_seq_blocks,
                )
                for head_group_idx in range(len(staged_seq_wk_tiles))
            ]
            staged_seq_repeated_wv_tiles = [
                repeat_outer_tap(
                    staged_seq_wv_tiles[head_group_idx],
                    attn_weight_tensor_shape,
                    num_kv_seq_blocks,
                )
                for head_group_idx in range(len(staged_seq_wv_tiles))
            ]
            staged_seq_k_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + staged_seq_k_stage_base_offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in k_tiles_base
                ]
            )
            staged_seq_v_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + staged_seq_v_stage_base_offset,
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
                    offset=(
                        tap.offset + ln_stage_base_offset
                        if use_staged_hidden_state_projection
                        else tap.offset + 2 * seq_len * embed_sz
                    ),
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in joined_o_tiles_base
            ]
        )
        if use_staged_hidden_state_projection:
            joined_r_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        staged_hidden_state_tensor_shape,
                        offset=tap.offset,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in joined_r_tiles_base
                ]
            )
        else:
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
            if use_staged_hidden_state_projection:
                unified_joined_r_tiles = TensorAccessSequence.from_taps(
                    [
                        TensorAccessPattern(
                            staged_hidden_state_tensor_shape,
                            offset=tap.offset,
                            sizes=tap.sizes,
                            strides=tap.strides,
                        )
                        for tap in unified_joined_r_tiles_base
                    ]
                )
            else:
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

        def split_fill_tap_prefix_rest(
            tap: TensorAccessPattern,
            tensor_shape: tuple[int, ...],
            prefix_outer_dim: int,
        ) -> tuple[TensorAccessPattern | None, TensorAccessPattern | None]:
            outer_size = int(tap.sizes[0])
            if outer_size <= 0:
                return None, None
            prefix_len = min(prefix_outer_dim, outer_size)
            prefix_tap = TensorAccessPattern(
                tensor_shape,
                offset=int(tap.offset),
                sizes=[prefix_len, *[int(s) for s in tap.sizes[1:]]],
                strides=[int(s) for s in tap.strides],
            )
            if prefix_len == outer_size:
                return prefix_tap, None
            rest_tap = TensorAccessPattern(
                tensor_shape,
                offset=int(tap.offset) + prefix_len * int(tap.strides[0]),
                sizes=[outer_size - prefix_len, *[int(s) for s in tap.sizes[1:]]],
                strides=[int(s) for s in tap.strides],
            )
            return prefix_tap, rest_tap

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
        if use_staged_hidden_state_projection:
            for tas in (
                staged_hidden_state_q_tiles,
                staged_hidden_state_q_batch_tiles,
                staged_hidden_state_kv_tiles,
                staged_seq_q_tiles,
                staged_seq_q_batch_tiles,
                staged_seq_k_tiles,
                staged_seq_v_tiles,
                staged_seq_wq_tiles,
                staged_seq_wk_tiles,
                staged_seq_wv_tiles,
                staged_seq_repeated_wk_tiles,
                staged_seq_repeated_wv_tiles,
            ):
                legalize_tas(tas)
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
        if use_staged_hidden_state_projection:
            staged_seq_hidden_state_kv_fill_taps = split_fill_tap_on_outer_dim(
                staged_hidden_state_kv_tiles[0],
                staged_hidden_state_tensor_shape,
                max_host_fill_outer_dim,
            )
            staged_seq_wk_fill_taps = [
                split_fill_tap_on_outer_dim(
                    tap,
                    attn_weight_tensor_shape,
                    max_host_fill_outer_dim,
                )
                for tap in staged_seq_repeated_wk_tiles
            ]
            staged_seq_wv_fill_taps = [
                split_fill_tap_on_outer_dim(
                    tap,
                    attn_weight_tensor_shape,
                    max_host_fill_outer_dim,
                )
                for tap in staged_seq_repeated_wv_tiles
            ]

        assert_tap_iteration_count(
            q_batch_tiles[0],
            (group_size * seq_tile, d * parallel_heads),
            1,
            "Sequence-parallel Q batch tap count mismatch",
        )
        if unified_q_batch_tiles is not None:
            assert_tap_iteration_count(
                unified_q_batch_tiles[0],
                (parallel_seq * seq_tile, d * parallel_heads),
                1,
                "Sequence-parallel unified Q batch tap count mismatch",
            )
        assert_tap_iteration_count(
            k_tiles[0],
            (kv_seq_tile, d * parallel_heads),
            num_kv_seq_blocks,
            "Sequence-parallel K tap count mismatch",
        )
        assert_tap_iteration_count(
            v_tiles[0],
            (kv_seq_tile, d * parallel_heads),
            num_kv_seq_blocks,
            "Sequence-parallel V tap count mismatch",
        )
        assert_tap_iteration_count(
            wo_tiles[0],
            (d * parallel_heads, emb_tile),
            proj_acc_depth,
            "Sequence-parallel W_O tap count mismatch",
        )
        assert_tap_iteration_count(
            joined_r_tiles[0],
            (group_size * seq_tile, emb_tile),
            proj_acc_depth,
            "Sequence-parallel residual tap count mismatch",
        )
        if unified_joined_r_tiles is not None:
            assert_tap_iteration_count(
                unified_joined_r_tiles[0],
                (parallel_seq * seq_tile, emb_tile),
                proj_acc_depth,
                "Sequence-parallel unified residual tap count mismatch",
            )
        assert_tap_iteration_count(
            joined_o_tiles[0],
            (group_size * seq_tile, emb_tile),
            proj_acc_depth,
            "Sequence-parallel output tap count mismatch",
        )
        assert_tap_iteration_count(
            joined_i_tiles[0],
            (group_size * seq_tile, emb_tile),
            proj_acc_depth,
            "Sequence-parallel intermediate tap count mismatch",
        )
        assert_tap_iteration_count(
            joined_refill_taps[0],
            (group_size * seq_tile, emb_tile),
            ffn_col_group_count * proj_acc_depth,
            "Sequence-parallel LN1 refill tap count mismatch",
        )

        rt = Runtime()
        sequence_types = (W_O_ty, QKV_ty, OR_ty, B_Up_ty, B_Down_ty)
        if use_staged_hidden_state_projection:
            sequence_types = (W_ATTN_ty, X_ty, OR_ty, B_Up_ty, B_Down_ty)
        with rt.sequence(*sequence_types) as sequence_args:
            if use_staged_hidden_state_projection:
                W_ATTN, X, OR, B_Up, B_Down = sequence_args
            else:
                W_O, QKV, OR, B_Up, B_Down = sequence_args
            if use_runtime_ln_weights:
                for lane_idx in range(parallel_seq):
                    rt.inline_ops(
                        _emit_runtime_buffer_values,
                        [lane_ln1_weight_buffers[lane_idx], static_ln1_weights],
                    )
                    rt.inline_ops(
                        _emit_runtime_buffer_values,
                        [lane_ln2_weight_buffers[lane_idx], static_ln2_weights],
                    )
            if use_staged_hidden_state_projection:
                for worker in staged_seq_q_proj_workers:
                    rt.start(worker)
                for worker in staged_seq_k_proj_workers:
                    rt.start(worker)
                for worker in staged_seq_v_proj_workers:
                    rt.start(worker)
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
                    if (
                        use_staged_hidden_state_projection
                        and parallel_heads == 1
                        and lane_batch_idx == 0
                    ):
                        tg_kv_stage = rt.task_group()
                        for hidden_state_kv_tap in staged_seq_hidden_state_kv_fill_taps:
                            rt.fill(
                                staged_seq_inXK[0].prod(),
                                X,
                                tap=hidden_state_kv_tap,
                                placement=Tile(col=v_shim_col, row=0),
                                task_group=tg_kv_stage,
                                wait=True,
                            )
                        for wk_tap in staged_seq_wk_fill_taps[head_group_idx]:
                            rt.fill(
                                staged_seq_inWK[0].prod(),
                                W_ATTN,
                                tap=wk_tap,
                                placement=Tile(col=ow_shim_col, row=0),
                                task_group=tg_kv_stage,
                                wait=True,
                            )
                        rt.drain(
                            staged_seq_projKOut[0].cons(),
                            OR,
                            tap=staged_seq_k_tiles[head_group_idx],
                            placement=Tile(col=v_shim_col, row=0),
                            task_group=tg_kv_stage,
                            wait=True,
                        )
                        for hidden_state_kv_tap in staged_seq_hidden_state_kv_fill_taps:
                            rt.fill(
                                staged_seq_inXV[0].prod(),
                                X,
                                tap=hidden_state_kv_tap,
                                placement=Tile(col=ln1_stage_shim_col, row=0),
                                task_group=tg_kv_stage,
                                wait=True,
                            )
                        for wv_tap in staged_seq_wv_fill_taps[head_group_idx]:
                            rt.fill(
                                staged_seq_inWV[0].prod(),
                                W_ATTN,
                                tap=wv_tap,
                                placement=Tile(col=ln1_stage_shim_col, row=0),
                                task_group=tg_kv_stage,
                                wait=True,
                            )
                        rt.drain(
                            staged_seq_projVOut[0].cons(),
                            OR,
                            tap=staged_seq_v_tiles[head_group_idx],
                            placement=Tile(col=v_shim_col, row=0),
                            task_group=tg_kv_stage,
                            wait=True,
                        )
                        rt.finish_task_group(tg_kv_stage)

                    if use_staged_hidden_state_projection:
                        tg_q_stage = rt.task_group()
                        rt.fill(
                            staged_seq_inXQAll.prod(),
                            X,
                            tap=staged_hidden_state_q_batch_tiles[lane_batch_idx],
                            placement=Tile(col=q_shim_col, row=0),
                            task_group=tg_q_stage,
                            wait=True,
                        )
                        if parallel_heads == 1:
                            for lane_idx in transport_groups[0]["lanes"]:
                                rt.fill(
                                    staged_seq_inWQ[lane_idx][0].prod(),
                                    W_ATTN,
                                    tap=staged_seq_wq_tiles[head_group_idx],
                                    placement=Tile(
                                        col=staged_seq_wq_shim_cols[lane_idx], row=0
                                    ),
                                    task_group=tg_q_stage,
                                    wait=True,
                                )
                        else:
                            for head_idx in range(parallel_heads):
                                global_head_idx = (
                                    head_group_idx * parallel_heads + head_idx
                                )
                                if staged_seq_inWQAll[head_idx] is None:
                                    for lane_idx in transport_groups[0]["lanes"]:
                                        rt.fill(
                                            staged_seq_inWQ[lane_idx][head_idx].prod(),
                                            W_ATTN,
                                            tap=staged_seq_wq_tiles[global_head_idx],
                                            placement=Tile(
                                                col=staged_seq_wq_shim_cols[lane_idx],
                                                row=0,
                                            ),
                                            task_group=tg_q_stage,
                                            wait=True,
                                        )
                                else:
                                    rt.fill(
                                        staged_seq_inWQAll[head_idx].prod(),
                                        W_ATTN,
                                        tap=staged_seq_wq_tiles[global_head_idx],
                                        placement=Tile(
                                            col=staged_seq_shared_wq_shim_cols[
                                                head_idx
                                            ],
                                            row=0,
                                        ),
                                        task_group=tg_q_stage,
                                        wait=True,
                                    )
                        rt.finish_task_group(tg_q_stage)

                    if use_staged_hidden_state_projection and parallel_heads > 1:
                        tg_kv_direct = rt.task_group()
                        for hidden_state_kv_tap in staged_seq_hidden_state_kv_fill_taps:
                            rt.fill(
                                staged_seq_inXKAll.prod(),
                                X,
                                tap=hidden_state_kv_tap,
                                placement=Tile(col=staged_seq_xk_shim_cols[0], row=0),
                                task_group=tg_kv_direct,
                                wait=True,
                            )
                        for hidden_state_kv_tap in staged_seq_hidden_state_kv_fill_taps:
                            rt.fill(
                                staged_seq_inXVAll.prod(),
                                X,
                                tap=hidden_state_kv_tap,
                                placement=Tile(col=staged_seq_xv_shim_cols[0], row=0),
                                task_group=tg_kv_direct,
                                wait=True,
                            )
                        for head_idx in range(parallel_heads):
                            global_head_idx = head_group_idx * parallel_heads + head_idx
                            for wk_tap in staged_seq_wk_fill_taps[global_head_idx]:
                                rt.fill(
                                    staged_seq_inWK[head_idx].prod(),
                                    W_ATTN,
                                    tap=wk_tap,
                                    placement=Tile(
                                        col=staged_seq_wk_shim_cols[head_idx], row=0
                                    ),
                                    task_group=tg_kv_direct,
                                    wait=True,
                                )
                            for wv_tap in staged_seq_wv_fill_taps[global_head_idx]:
                                rt.fill(
                                    staged_seq_inWV[head_idx].prod(),
                                    W_ATTN,
                                    tap=wv_tap,
                                    placement=Tile(
                                        col=staged_seq_wv_shim_cols[head_idx], row=0
                                    ),
                                    task_group=tg_kv_direct,
                                    wait=True,
                                )
                        rt.finish_task_group(tg_kv_direct)

                    tg_head = rt.task_group()
                    if unified_qr_split is not None:
                        rt.fill(
                            unified_inQSeq.prod(),
                            OR if use_staged_hidden_state_projection else QKV,
                            tap=(
                                staged_seq_q_batch_tiles[
                                    lane_batch_idx
                                    * num_qkv_head_block_per_parallel_head
                                    + head_group_idx
                                ]
                                if use_staged_hidden_state_projection
                                else unified_q_batch_tiles[
                                    lane_batch_idx
                                    * num_qkv_head_block_per_parallel_head
                                    + head_group_idx
                                ]
                            ),
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
                        if (
                            unified_qr_split is None
                            and not use_staged_hidden_state_projection
                        ):
                            rt.fill(
                                group_inQSeq[group_idx].prod(),
                                OR if use_staged_hidden_state_projection else QKV,
                                tap=(
                                    staged_seq_q_batch_tiles[
                                        group_batch_idx
                                        * num_qkv_head_block_per_parallel_head
                                        + head_group_idx
                                    ]
                                    if use_staged_hidden_state_projection
                                    else q_batch_tiles[
                                        group_batch_idx
                                        * num_qkv_head_block_per_parallel_head
                                        + head_group_idx
                                    ]
                                ),
                                placement=Tile(
                                    col=transport_groups[group_idx].get(
                                        "shim_cols", default_group_shim_cols
                                    )["q"],
                                    row=0,
                                ),
                                task_group=tg_head,
                                wait=True,
                            )
                        if not (
                            use_staged_hidden_state_projection and parallel_heads > 1
                        ):
                            for k_tap in (
                                [staged_seq_k_tiles[head_group_idx]]
                                if use_staged_hidden_state_projection
                                else K_fill_taps[head_group_idx]
                            ):
                                rt.fill(
                                    group_inKSeq[group_idx].prod(),
                                    OR if use_staged_hidden_state_projection else QKV,
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
                            for v_tap in (
                                [staged_seq_v_tiles[head_group_idx]]
                                if use_staged_hidden_state_projection
                                else V_fill_taps[head_group_idx]
                            ):
                                rt.fill(
                                    group_inVSeq[group_idx].prod(),
                                    OR if use_staged_hidden_state_projection else QKV,
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
                                W_ATTN if use_staged_hidden_state_projection else W_O,
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
                        X if use_staged_hidden_state_projection else OR,
                        tap=unified_joined_r_tiles[lane_batch_idx],
                        placement=Tile(
                            col=unified_qr_split["residual_shim_col"],
                            row=0,
                        ),
                        task_group=tg_residual,
                        wait=False,
                    )
                    rt.fill(
                        unified_inRSeq.prod(),
                        X if use_staged_hidden_state_projection else OR,
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
                            X if use_staged_hidden_state_projection else OR,
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
                        rt.fill(
                            group_inRSeq[group_idx].prod(),
                            X if use_staged_hidden_state_projection else OR,
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
                use_seq_b_up_priming = (
                    effective_ffn_branches == 1 and ffn_col_group_count > 1
                )
                seq_b_up_prime_groups = 1 if use_seq_b_up_priming else 0
                if unified_inBUpSeq is not None:
                    if use_seq_b_up_priming:
                        first_tap, rest_tap = split_fill_tap_prefix_rest(
                            b_up_tiles[0],
                            (embed_sz, ffn_intermediate_size),
                            seq_b_up_prime_groups,
                        )
                        if first_tap is not None:
                            rt.fill(
                                unified_inBUpSeq.prod(),
                                B_Up,
                                tap=first_tap,
                                placement=Tile(
                                    col=unified_shared_streams["b_up"]["shim_col"],
                                    row=0,
                                ),
                                task_group=tg_tail,
                                wait=False,
                            )
                        if rest_tap is not None:
                            rt.fill(
                                unified_inBUpSeq.prod(),
                                B_Up,
                                tap=rest_tap,
                                placement=Tile(
                                    col=unified_shared_streams["b_up"]["shim_col"],
                                    row=0,
                                ),
                                task_group=tg_tail,
                                wait=True,
                            )
                    else:
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
                            if use_seq_b_up_priming:
                                first_tap, rest_tap = split_fill_tap_prefix_rest(
                                    b_up_tiles[branch_idx],
                                    (embed_sz, ffn_intermediate_size),
                                    seq_b_up_prime_groups,
                                )
                                if first_tap is not None:
                                    rt.fill(
                                        group_inBUpSeq[group_idx][branch_idx].prod(),
                                        B_Up,
                                        tap=first_tap,
                                        placement=Tile(
                                            col=group_weight_b_up_shim_cols[group_idx][
                                                branch_idx
                                            ],
                                            row=0,
                                        ),
                                        task_group=tg_tail,
                                        wait=False,
                                    )
                                if rest_tap is not None:
                                    rt.fill(
                                        group_inBUpSeq[group_idx][branch_idx].prod(),
                                        B_Up,
                                        tap=rest_tap,
                                        placement=Tile(
                                            col=group_weight_b_up_shim_cols[group_idx][
                                                branch_idx
                                            ],
                                            row=0,
                                        ),
                                        task_group=tg_tail,
                                        wait=True,
                                    )
                            else:
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
                    for branch_idx in range(effective_ffn_branches):
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

        return _emit_program(rt)

    def project_q_from_hidden_states(
        of_x_in,
        of_w_in,
        of_q_out,
        matmul_init,
        matmul_acc,
        add_bias,
        bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                elem_out_q = of_q_out.acquire(1)
                elem_in_x = of_x_in.acquire(1)
                elem_in_w = of_w_in.acquire(1)
                matmul_init(elem_in_x, elem_in_w, elem_out_q)
                of_w_in.release(1)
                of_x_in.release(1)
                for _ in range_(proj_acc_depth - 1):
                    elem_in_x = of_x_in.acquire(1)
                    elem_in_w = of_w_in.acquire(1)
                    matmul_acc(elem_in_x, elem_in_w, elem_out_q, elem_out_q)
                    of_w_in.release(1)
                    of_x_in.release(1)
                add_bias(
                    elem_out_q,
                    bias_buffer,
                    elem_out_q,
                    head_group_i32,
                    seq_tile,
                    d,
                )
                of_q_out.release(1)

    def project_k_from_hidden_states(
        of_x_in,
        of_w_in,
        of_k_out,
        matmul_init_k,
        matmul_acc_k,
        add_k_bias,
        k_bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                for _ in range_(num_kv_seq_blocks):
                    elem_out_k = of_k_out.acquire(1)
                    for acc_idx in range_(proj_acc_depth):
                        elem_in_x = of_x_in.acquire(1)
                        elem_in_wk = of_w_in.acquire(1)
                        if acc_idx == 0:
                            matmul_init_k(elem_in_x, elem_in_wk, elem_out_k)
                        else:
                            matmul_acc_k(elem_in_x, elem_in_wk, elem_out_k, elem_out_k)
                        of_w_in.release(1)
                        of_x_in.release(1)
                    add_k_bias(
                        elem_out_k,
                        k_bias_buffer,
                        elem_out_k,
                        head_group_i32,
                        kv_seq_tile,
                        d,
                    )
                    of_k_out.release(1)

    def project_v_from_hidden_states(
        of_x_in,
        of_w_in,
        of_v_out,
        matmul_init_v,
        matmul_acc_v,
        add_v_bias,
        v_bias_buffer,
    ):
        for _ in range_(sys.maxsize):
            for head_group_idx in range_(num_qkv_head_block_per_parallel_head):
                head_group_i32 = arith.index_castui(T.i32(), head_group_idx)
                for _ in range_(num_kv_seq_blocks):
                    elem_out_v = of_v_out.acquire(1)
                    for acc_idx in range_(proj_acc_depth):
                        elem_in_x = of_x_in.acquire(1)
                        elem_in_wv = of_w_in.acquire(1)
                        if acc_idx == 0:
                            matmul_init_v(elem_in_x, elem_in_wv, elem_out_v)
                        else:
                            matmul_acc_v(elem_in_x, elem_in_wv, elem_out_v, elem_out_v)
                        of_w_in.release(1)
                        of_x_in.release(1)
                    add_v_bias(
                        elem_out_v,
                        v_bias_buffer,
                        elem_out_v,
                        head_group_i32,
                        kv_seq_tile,
                        d,
                    )
                    of_v_out.release(1)

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
    ln1_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=None if use_runtime_ln_weights else static_ln1_weights,
        name="static_ln1_weights",
        use_write_rtp=use_runtime_ln_weights,
    )
    ln2_weight_buffer = Buffer(
        type=ln_weights_ty,
        initial_value=None if use_runtime_ln_weights else static_ln2_weights,
        name="static_ln2_weights",
        use_write_rtp=use_runtime_ln_weights,
    )
    ln1_norm_sum_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sum_buffer")
    ln1_norm_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln1_norm_sumsq_buffer")
    ffn_down_sum_buffer = Buffer(type=sum_l1_ty, name="ffn_down_sum_buffer")
    ffn_down_sumsq_buffer = Buffer(type=sum_l1_ty, name="ffn_down_sumsq_buffer")
    ln2_sum_buffer = Buffer(type=sum_l1_ty, name="ln2_sum_buffer")
    ln2_sumsq_buffer = Buffer(type=sum_l1_ty, name="ln2_sumsq_buffer")
    q_proj_bias_buffer = None
    k_proj_bias_buffer = None
    v_proj_bias_buffer = None
    if use_staged_hidden_state_projection:
        staged_q_bias_rows = static_q_proj_biases.reshape(
            num_qkv_head_block_per_parallel_head, parallel_heads, d
        )
        staged_k_bias_rows = static_k_proj_biases.reshape(
            num_qkv_head_block_per_parallel_head, parallel_heads, d
        )
        staged_v_bias_rows = static_v_proj_biases.reshape(
            num_qkv_head_block_per_parallel_head, parallel_heads, d
        )
        q_proj_bias_buffer = [
            Buffer(
                type=bias_matrix_ty,
                initial_value=np.ascontiguousarray(staged_q_bias_rows[:, head_idx, :]),
                name=f"static_q_proj_bias_{head_idx}",
            )
            for head_idx in range(parallel_heads)
        ]
        k_proj_bias_buffer = [
            Buffer(
                type=bias_matrix_ty,
                initial_value=np.ascontiguousarray(staged_k_bias_rows[:, head_idx, :]),
                name=f"static_k_proj_bias_{head_idx}",
            )
            for head_idx in range(parallel_heads)
        ]
        v_proj_bias_buffer = [
            Buffer(
                type=bias_matrix_ty,
                initial_value=np.ascontiguousarray(staged_v_bias_rows[:, head_idx, :]),
                name=f"static_v_proj_bias_{head_idx}",
            )
            for head_idx in range(parallel_heads)
        ]

    q_proj_workers = []
    k_proj_workers = []
    v_proj_workers = []
    if use_staged_hidden_state_projection:
        if parallel_heads == 1:
            q_proj_tiles = [Tile(col=2, row=2)]
            k_proj_tiles = [Tile(col=3, row=2)]
            v_proj_tiles = [Tile(col=4, row=2)]
        else:
            proj_col_start = parallel_heads
            q_proj_tiles = [
                Tile(col=proj_col_start + head_idx, row=2)
                for head_idx in range(parallel_heads)
            ]
            k_proj_tiles = [
                Tile(col=proj_col_start + head_idx, row=3)
                for head_idx in range(parallel_heads)
            ]
            v_proj_tiles = [
                Tile(col=proj_col_start + parallel_heads + head_idx, row=3)
                for head_idx in range(parallel_heads)
            ]
        for head_idx in range(parallel_heads):
            q_proj_workers.append(
                Worker(
                    project_q_from_hidden_states,
                    fn_args=[
                        inXQ[head_idx].cons(),
                        inWQ[head_idx].cons(),
                        projQOut[head_idx].prod(),
                        matmul_init_q_proj_kernel,
                        matmul_q_proj_kernel,
                        eltwise_add_q_kernel,
                        q_proj_bias_buffer[head_idx],
                    ],
                    placement=q_proj_tiles[head_idx],
                    stack_size=0xD00,
                    while_true=False,
                )
            )
            k_proj_workers.append(
                Worker(
                    project_k_from_hidden_states,
                    fn_args=[
                        inXK[head_idx].cons(),
                        inWK[head_idx].cons(),
                        projKOut[head_idx].prod(),
                        matmul_init_k_proj_kernel,
                        matmul_k_proj_kernel,
                        eltwise_add_k_kernel,
                        k_proj_bias_buffer[head_idx],
                    ],
                    placement=k_proj_tiles[head_idx],
                    stack_size=0xD00,
                    while_true=False,
                )
            )
            v_proj_workers.append(
                Worker(
                    project_v_from_hidden_states,
                    fn_args=[
                        inXV[head_idx].cons(),
                        inWV[head_idx].cons(),
                        projVOut[head_idx].prod(),
                        matmul_init_v_proj_kernel,
                        matmul_v_proj_kernel,
                        eltwise_add_v_kernel,
                        v_proj_bias_buffer[head_idx],
                    ],
                    placement=v_proj_tiles[head_idx],
                    stack_size=0xD00,
                    while_true=False,
                )
            )

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
                    zero_kernel_q,
                    q_tile_elems,
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
                    matmul_o_proj_emit_twice
                    if i == (parallel_heads - 1)
                    else matmul_o_proj
                ),
                fn_args=(
                    [
                        outOProj[i].cons(),
                        memOW[i].cons(),
                        outOProjAccumIn[i].cons(depth=1),
                        outOProjAccumOut[i].prod(),
                        outOPart[i - 1].cons() if i > 0 else None,
                        outOProjInput.prod() if not use_memtile_o_proj_replay else None,
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
                        eltwise_add_vector_kernel,
                        zero_kernel_o_proj,
                        matmul_kernel_o_proj,
                        mem_copy_o_proj,
                    ]
                ),
                placement=o_proj_tiles[i],
                stack_size=0xD00,
                while_true=False,
            )
        )
    ln1_worker = Worker(
        core_fn_ln1_from_replayed_inputs,
        fn_args=[
            outOProjInput.cons(),
            memR.cons(depth=1),
            ln1_norm_sum_buffer,
            ln1_norm_sumsq_buffer,
            ln1_weight_buffer,
            outLNBroadcast.prod(1),
            eltwise_add_vector_kernel,
            ln_zero_f32_kernel,
            ln_calc_sum_sumsq_kernel,
            ln_fused_add_layer_norm_kernel,
        ],
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
            else ffnDownOut.prod(ffn_down_output_producer_depth)
        )
        ffn_down_workers.append(
            Worker(
                (
                    core_fn_ffn_down_proj_emit_twice
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
                        ffnDownOut.prod(ffn_down_output_producer_depth),
                        ffn_matmul_init_kernel_down_proj,
                        ffn_matmul_kernel_down_proj,
                        eltwise_add_vector_kernel,
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
        core_fn_add_norm2_from_replayed_inputs,
        fn_args=[
            ffnDownOut.cons(),
            ffnRIn.cons(),
            ln2_sum_buffer,
            ln2_sumsq_buffer,
            ln2_weight_buffer,
            outLN2.prod(),
            eltwise_add_vector_kernel,
            ln_zero_f32_kernel,
            ln_calc_sum_sumsq_kernel,
            ln_fused_add_layer_norm_kernel,
        ],
        placement=ln2_tile,
        stack_size=0xF00,
        while_true=False,
    )

    qkv_tensor_shape = (3 * seq_len, embed_sz)
    attn_weight_tensor_shape = (4 * embed_sz, embed_sz)
    ln_stage_base_offset = or_rows_before_ln1_stage * embed_sz
    hidden_state_tensor_shape = (seq_len, embed_sz)
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
    if use_staged_hidden_state_projection:
        hidden_state_q_tiles_base = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (seq_tile, emb_tile),
            (1, proj_acc_depth),
        )
        hidden_state_q_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    hidden_state_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in hidden_state_q_tiles_base
            ]
        )
        hidden_state_kv_tiles_base = TensorTiler2D.group_tiler(
            (seq_len, embed_sz),
            (kv_seq_tile, emb_tile),
            (num_kv_seq_blocks, proj_acc_depth),
        )
        hidden_state_kv_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    hidden_state_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in hidden_state_kv_tiles_base
            ]
        )
        wq_tiles_base = TensorTiler2D.group_tiler(
            (embed_sz, embed_sz),
            (emb_tile, d),
            (proj_acc_depth, 1),
        )
        WQ_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    attn_weight_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in wq_tiles_base
            ]
        )
        wk_tiles_base = TensorTiler2D.group_tiler(
            (embed_sz, embed_sz),
            (emb_tile, d),
            (proj_acc_depth, 1),
        )
        WK_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    attn_weight_tensor_shape,
                    offset=tap.offset + embed_sz * embed_sz,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in wk_tiles_base
            ]
        )
        wv_tiles_base = TensorTiler2D.group_tiler(
            (embed_sz, embed_sz),
            (emb_tile, d),
            (proj_acc_depth, 1),
        )
        WV_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    attn_weight_tensor_shape,
                    offset=tap.offset + 2 * embed_sz * embed_sz,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in wv_tiles_base
            ]
        )
        grouped_k_weight_tiles = None
        grouped_v_weight_tiles = None
        if parallel_heads > 1:
            wk_group_tiles_base = TensorTiler2D.group_tiler(
                (embed_sz, embed_sz),
                (emb_tile, d * parallel_heads),
                (proj_acc_depth, 1),
            )
            grouped_k_weight_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        attn_weight_tensor_shape,
                        offset=tap.offset + embed_sz * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in wk_group_tiles_base
                ]
            )
            wv_group_tiles_base = TensorTiler2D.group_tiler(
                (embed_sz, embed_sz),
                (emb_tile, d * parallel_heads),
                (proj_acc_depth, 1),
            )
            grouped_v_weight_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        attn_weight_tensor_shape,
                        offset=tap.offset + 2 * embed_sz * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in wv_group_tiles_base
                ]
            )
    wo_tiles_base = TensorTiler2D.group_tiler(
        (embed_sz, embed_sz),
        (d, emb_tile),
        (parallel_heads, embed_sz // emb_tile),
    )
    WO_tiles = TensorAccessSequence.from_taps(
        [
            TensorAccessPattern(
                (
                    attn_weight_tensor_shape
                    if use_staged_hidden_state_projection
                    else (embed_sz, embed_sz)
                ),
                offset=(
                    tap.offset + 3 * embed_sz * embed_sz
                    if use_staged_hidden_state_projection
                    else tap.offset
                ),
                sizes=tap.sizes,
                strides=tap.strides,
            )
            for tap in wo_tiles_base
        ]
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
    R_tiles = None
    if not use_staged_hidden_state_projection:
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
    if use_staged_hidden_state_projection:
        hidden_state_residual_tiles = TensorAccessSequence.from_taps(
            [
                TensorAccessPattern(
                    hidden_state_tensor_shape,
                    offset=tap.offset,
                    sizes=tap.sizes,
                    strides=tap.strides,
                )
                for tap in r_tiles_base
            ]
        )
        if use_staged_hidden_state_kv_cache:
            cached_k_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + seq_len * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in k_tiles_base
                ]
            )
            cached_v_tiles = TensorAccessSequence.from_taps(
                [
                    TensorAccessPattern(
                        or_tensor_shape,
                        offset=tap.offset + 2 * seq_len * embed_sz,
                        sizes=tap.sizes,
                        strides=tap.strides,
                    )
                    for tap in v_tiles_base
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

    legalized_tas = [
        Q_tiles,
        K_tiles,
        V_tiles,
        WO_tiles,
        O_tiles,
        B_Up_tiles,
        B_Down_tiles,
    ]
    if R_tiles is not None:
        legalized_tas.append(R_tiles)
    if use_staged_hidden_state_projection:
        legalized_tas.extend(
            [
                hidden_state_q_tiles,
                hidden_state_kv_tiles,
                hidden_state_residual_tiles,
                WQ_tiles,
                WK_tiles,
                WV_tiles,
            ]
        )
        if use_staged_hidden_state_kv_cache:
            legalized_tas.extend([cached_k_tiles, cached_v_tiles])
    for tas in legalized_tas:
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
    staged_hidden_state_kv_fill_taps = None
    staged_k_weight_fill_taps = None
    staged_v_weight_fill_taps = None
    staged_grouped_k_weight_fill_taps = None
    staged_grouped_v_weight_fill_taps = None

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
        assert_tap_iteration_count(tap_seq[0], obj_shape, expected, message)
    if R_tiles is not None:
        assert_tap_iteration_count(
            R_tiles[0],
            (seq_tile, emb_tile),
            proj_acc_depth,
            "Residual tap count does not match LN1 runtime contract",
        )

    if use_staged_hidden_state_projection:
        repeated_k_weight_tiles = [
            repeat_outer_tap(
                WK_tiles[head_group_idx], attn_weight_tensor_shape, num_kv_seq_blocks
            )
            for head_group_idx in range(len(WK_tiles))
        ]
        repeated_v_weight_tiles = [
            repeat_outer_tap(
                WV_tiles[head_group_idx], attn_weight_tensor_shape, num_kv_seq_blocks
            )
            for head_group_idx in range(len(WV_tiles))
        ]
        repeated_grouped_k_weight_tiles = None
        repeated_grouped_v_weight_tiles = None
        if parallel_heads > 1:
            repeated_grouped_k_weight_tiles = [
                repeat_outer_tap(
                    grouped_k_weight_tiles[head_group_idx],
                    attn_weight_tensor_shape,
                    num_kv_seq_blocks,
                )
                for head_group_idx in range(len(grouped_k_weight_tiles))
            ]
            repeated_grouped_v_weight_tiles = [
                repeat_outer_tap(
                    grouped_v_weight_tiles[head_group_idx],
                    attn_weight_tensor_shape,
                    num_kv_seq_blocks,
                )
                for head_group_idx in range(len(grouped_v_weight_tiles))
            ]
        assert_tap_iteration_count(
            hidden_state_q_tiles[0],
            (seq_tile, emb_tile),
            proj_acc_depth,
            "XQ tap count does not match proj_acc_depth",
        )
        assert_tap_iteration_count(
            hidden_state_kv_tiles[0],
            (kv_seq_tile, emb_tile),
            num_kv_seq_blocks * proj_acc_depth,
            "XKV tap count does not match staged KV projection loop count",
        )
        assert_tap_iteration_count(
            hidden_state_residual_tiles[0],
            (seq_tile, emb_tile),
            proj_acc_depth,
            "staged residual tap count does not match LN1 runtime contract",
        )
        assert_tap_iteration_count(
            WQ_tiles[0],
            (emb_tile, d),
            proj_acc_depth,
            "W_Q tap count does not match proj_acc_depth",
        )
        assert_tap_iteration_count(
            repeated_k_weight_tiles[0],
            (emb_tile, d),
            num_kv_seq_blocks * proj_acc_depth,
            "W_K tap count does not match staged KV projection loop count",
        )
        assert_tap_iteration_count(
            repeated_v_weight_tiles[0],
            (emb_tile, d),
            num_kv_seq_blocks * proj_acc_depth,
            "W_V tap count does not match staged KV projection loop count",
        )
        if parallel_heads > 1:
            assert_tap_iteration_count(
                repeated_grouped_k_weight_tiles[0],
                (emb_tile, d * parallel_heads),
                num_kv_seq_blocks * proj_acc_depth,
                "grouped W_K tap count does not match staged KV projection loop count",
            )
            assert_tap_iteration_count(
                repeated_grouped_v_weight_tiles[0],
                (emb_tile, d * parallel_heads),
                num_kv_seq_blocks * proj_acc_depth,
                "grouped W_V tap count does not match staged KV projection loop count",
            )
        if use_staged_hidden_state_kv_cache:
            assert_tap_iteration_count(
                cached_k_tiles[0],
                (kv_seq_tile, d),
                num_kv_seq_blocks,
                "cached K tap count does not match KV block count",
            )
            assert_tap_iteration_count(
                cached_v_tiles[0],
                (kv_seq_tile, d),
                num_kv_seq_blocks,
                "cached V tap count does not match KV block count",
            )
        staged_hidden_state_kv_fill_taps = split_fill_tap_on_outer_dim(
            hidden_state_kv_tiles[0],
            hidden_state_tensor_shape,
            max_host_fill_outer_dim,
        )
        staged_k_weight_fill_taps = [
            split_fill_tap_on_outer_dim(
                tap,
                attn_weight_tensor_shape,
                max_host_fill_outer_dim,
            )
            for tap in repeated_k_weight_tiles
        ]
        staged_v_weight_fill_taps = [
            split_fill_tap_on_outer_dim(
                tap,
                attn_weight_tensor_shape,
                max_host_fill_outer_dim,
            )
            for tap in repeated_v_weight_tiles
        ]
        if parallel_heads > 1:
            staged_grouped_k_weight_fill_taps = [
                split_fill_tap_on_outer_dim(
                    tap,
                    attn_weight_tensor_shape,
                    max_host_fill_outer_dim,
                )
                for tap in repeated_grouped_k_weight_tiles
            ]
            staged_grouped_v_weight_fill_taps = [
                split_fill_tap_on_outer_dim(
                    tap,
                    attn_weight_tensor_shape,
                    max_host_fill_outer_dim,
                )
                for tap in repeated_grouped_v_weight_tiles
            ]
    rt = Runtime()
    staged_xq_shim_cols = None
    staged_wq_shim_cols = None
    staged_xk_shim_cols = None
    staged_wk_shim_cols = None
    staged_xv_shim_cols = None
    staged_wv_shim_cols = None
    if use_staged_hidden_state_projection:
        staged_xq_shim_cols = [q_shim_col] * parallel_heads
        staged_wq_shim_cols = [k_shim_col] * parallel_heads
        staged_xk_shim_cols = [v_shim_col] * parallel_heads
        staged_wk_shim_cols = [bdown_shim_col] * parallel_heads
        staged_xv_shim_cols = [ow_shim_col] * parallel_heads
        if parallel_heads == 2:
            staged_xv_shim_cols = [ow_shim_col, residual_shim_col]
        staged_wv_shim_cols = [bup_shim_col] * parallel_heads
        if effective_ffn_branches > 1:
            # Keep staged K/V weight ingress off the FFN-tail shims entirely.
            # The multi-branch tail already saturates the branch-weight columns,
            # and the LN/output shims are consumed by OR staging. For staged
            # multi-branch bring-up, isolate the projection weights on the
            # projection-side shim group instead.
            staged_wk_shim_cols = [k_shim_col] * parallel_heads
            staged_wv_shim_cols = [q_shim_col] * parallel_heads
    sequence_types = (W_O_ty, QKV_ty, OR_ty, B_Up_ty, B_Down_ty)
    if use_staged_hidden_state_projection:
        sequence_types = (W_ATTN_ty, X_ty, OR_ty, B_Up_ty, B_Down_ty)
    with rt.sequence(*sequence_types) as sequence_args:
        if use_staged_hidden_state_projection:
            (
                W_ATTN,
                X,
                OR,
                B_Up,
                B_Down,
            ) = sequence_args
        else:
            W_O, QKV, OR, B_Up, B_Down = sequence_args
        if use_runtime_ln_weights:
            rt.inline_ops(
                _emit_runtime_buffer_values,
                [ln1_weight_buffer, static_ln1_weights],
            )
            rt.inline_ops(
                _emit_runtime_buffer_values,
                [ln2_weight_buffer, static_ln2_weights],
            )
        if use_staged_hidden_state_projection:
            for worker in q_proj_workers:
                rt.start(worker)
            for worker in k_proj_workers:
                rt.start(worker)
            for worker in v_proj_workers:
                rt.start(worker)
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
        if use_staged_hidden_state_kv_cache:
            for head_group_idx in range(num_qkv_head_block_per_parallel_head):
                tg_kv_cache = rt.task_group()
                if parallel_heads > 1:
                    rt.fill(
                        inWKShared.prod(),
                        W_ATTN,
                        tap=repeated_grouped_k_weight_tiles[head_group_idx],
                        placement=Tile(col=staged_wk_shim_cols[0], row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                    rt.fill(
                        inWVShared.prod(),
                        W_ATTN,
                        tap=repeated_grouped_v_weight_tiles[head_group_idx],
                        placement=Tile(col=staged_wv_shim_cols[0], row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                for head_idx in range(parallel_heads):
                    global_head_idx = head_group_idx * parallel_heads + head_idx
                    rt.fill(
                        inXK[head_idx].prod(),
                        X,
                        tap=hidden_state_kv_tiles[0],
                        placement=Tile(col=staged_xk_shim_cols[head_idx], row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                    if parallel_heads == 1:
                        rt.fill(
                            inWK[head_idx].prod(),
                            W_ATTN,
                            tap=repeated_k_weight_tiles[global_head_idx],
                            placement=Tile(col=staged_wk_shim_cols[head_idx], row=0),
                            task_group=tg_kv_cache,
                            wait=True,
                        )
                    rt.fill(
                        inXV[head_idx].prod(),
                        X,
                        tap=hidden_state_kv_tiles[0],
                        placement=Tile(col=staged_xv_shim_cols[head_idx], row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                    if parallel_heads == 1:
                        rt.fill(
                            inWV[head_idx].prod(),
                            W_ATTN,
                            tap=repeated_v_weight_tiles[global_head_idx],
                            placement=Tile(col=staged_wv_shim_cols[head_idx], row=0),
                            task_group=tg_kv_cache,
                            wait=True,
                        )
                    rt.drain(
                        projKOut[head_idx].cons(),
                        OR,
                        tap=cached_k_tiles[global_head_idx],
                        placement=Tile(col=k_shim_col, row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                    rt.drain(
                        projVOut[head_idx].cons(),
                        OR,
                        tap=cached_v_tiles[global_head_idx],
                        placement=Tile(col=v_shim_col, row=0),
                        task_group=tg_kv_cache,
                        wait=True,
                    )
                rt.finish_task_group(tg_kv_cache)
        pending_ln1_refill_tg = None
        pending_output_tap_idx = None
        q_block_schedule = build_q_block_schedule(parallel_seq, q_blocks_per_lane)
        if len(q_block_schedule) != num_q_seq_blocks or sorted(
            q_block_schedule
        ) != list(range(num_q_seq_blocks)):
            raise ValueError(
                "Q block schedule must visit each sequence block exactly once"
            )

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
                if use_staged_hidden_state_projection and parallel_heads > 1:
                    if not use_staged_hidden_state_kv_cache:
                        for grouped_k_weight_tap in staged_grouped_k_weight_fill_taps[
                            head_group_idx
                        ]:
                            rt.fill(
                                inWKShared.prod(),
                                W_ATTN,
                                tap=grouped_k_weight_tap,
                                placement=Tile(col=staged_wk_shim_cols[0], row=0),
                                task_group=tg_head,
                                wait=True,
                            )
                        for grouped_v_weight_tap in staged_grouped_v_weight_fill_taps[
                            head_group_idx
                        ]:
                            rt.fill(
                                inWVShared.prod(),
                                W_ATTN,
                                tap=grouped_v_weight_tap,
                                placement=Tile(col=staged_wv_shim_cols[0], row=0),
                                task_group=tg_head,
                                wait=True,
                            )
                if use_staged_hidden_state_projection:
                    for head_idx in range(parallel_heads):
                        global_head_idx = head_group_idx * parallel_heads + head_idx
                        if use_staged_hidden_state_kv_cache:
                            rt.fill(
                                inK.prod(),
                                OR,
                                tap=cached_k_tiles[global_head_idx],
                                placement=Tile(col=k_shim_col, row=0),
                                task_group=tg_head,
                                wait=False,
                            )
                            rt.fill(
                                inV.prod(),
                                OR,
                                tap=cached_v_tiles[global_head_idx],
                                placement=Tile(col=v_shim_col, row=0),
                                task_group=tg_head,
                                wait=False,
                            )
                        else:
                            for hidden_state_kv_tap in staged_hidden_state_kv_fill_taps:
                                rt.fill(
                                    inXK[head_idx].prod(),
                                    X,
                                    tap=hidden_state_kv_tap,
                                    placement=Tile(
                                        col=staged_xk_shim_cols[head_idx], row=0
                                    ),
                                    task_group=tg_head,
                                    wait=True,
                                )
                            if parallel_heads == 1:
                                for k_weight_tap in staged_k_weight_fill_taps[
                                    global_head_idx
                                ]:
                                    rt.fill(
                                        inWK[head_idx].prod(),
                                        W_ATTN,
                                        tap=k_weight_tap,
                                        placement=Tile(
                                            col=staged_wk_shim_cols[head_idx], row=0
                                        ),
                                        task_group=tg_head,
                                        wait=True,
                                    )
                            for hidden_state_kv_tap in staged_hidden_state_kv_fill_taps:
                                rt.fill(
                                    inXV[head_idx].prod(),
                                    X,
                                    tap=hidden_state_kv_tap,
                                    placement=Tile(
                                        col=staged_xv_shim_cols[head_idx], row=0
                                    ),
                                    task_group=tg_head,
                                    wait=True,
                                )
                            if parallel_heads == 1:
                                for v_weight_tap in staged_v_weight_fill_taps[
                                    global_head_idx
                                ]:
                                    rt.fill(
                                        inWV[head_idx].prod(),
                                        W_ATTN,
                                        tap=v_weight_tap,
                                        placement=Tile(
                                            col=staged_wv_shim_cols[head_idx], row=0
                                        ),
                                        task_group=tg_head,
                                        wait=True,
                                    )
                        rt.fill(
                            inXQ[head_idx].prod(),
                            X,
                            tap=hidden_state_q_tiles[tap_idx],
                            placement=Tile(col=staged_xq_shim_cols[head_idx], row=0),
                            task_group=tg_head,
                            wait=True,
                        )
                        rt.fill(
                            inWQ[head_idx].prod(),
                            W_ATTN,
                            tap=WQ_tiles[global_head_idx],
                            placement=Tile(col=staged_wq_shim_cols[head_idx], row=0),
                            task_group=tg_head,
                            wait=True,
                        )
                else:
                    rt.fill(
                        inQ.prod(),
                        QKV,
                        tap=Q_tiles[
                            tap_idx * num_qkv_head_block_per_parallel_head
                            + head_group_idx
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
                    W_ATTN if use_staged_hidden_state_projection else W_O,
                    tap=WO_tiles[head_group_idx],
                    placement=Tile(col=ow_shim_col, row=0),
                    task_group=tg_head,
                    wait=not use_staged_hidden_state_kv_cache,
                )
                rt.finish_task_group(tg_head)

            tg_tail = rt.task_group()
            initial_residual_source = X if use_staged_hidden_state_projection else OR
            initial_residual_tap = (
                hidden_state_residual_tiles[tap_idx]
                if use_staged_hidden_state_projection
                else R_tiles[tap_idx]
            )
            rt.fill(
                inR.prod(),
                initial_residual_source,
                tap=initial_residual_tap,
                placement=Tile(col=residual_shim_col, row=0),
                task_group=tg_tail,
                wait=False,
            )
            rt.fill(
                inR.prod(),
                initial_residual_source,
                tap=initial_residual_tap,
                placement=Tile(col=residual_shim_col, row=0),
                task_group=tg_tail,
                wait=False,
            )

            branch_stage_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln_stage_base_offset,
                sizes=[1, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            branch_refill_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln_stage_base_offset,
                sizes=[ffn_col_group_count, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            residual_stage_tap = TensorAccessPattern(
                or_tensor_shape,
                offset=ln_stage_base_offset,
                sizes=[1, proj_acc_depth, seq_tile, emb_tile],
                strides=[0, emb_tile, embed_sz, 1],
            )
            assert_tap_iteration_count(
                branch_stage_tap,
                (seq_tile, emb_tile),
                proj_acc_depth,
                "LN1 stage tap count mismatch",
            )
            assert_tap_iteration_count(
                branch_refill_tap,
                (seq_tile, emb_tile),
                ffn_col_group_count * proj_acc_depth,
                "LN1 branch refill tap count mismatch",
            )
            assert_tap_iteration_count(
                residual_stage_tap,
                (seq_tile, emb_tile),
                proj_acc_depth,
                "LN1 residual stage tap count mismatch",
            )

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

    return _emit_program(rt)


if __name__ == "__main__":
    main()
