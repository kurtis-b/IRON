#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.common.test_utils import run_test
from iron.operators.encoder_pipeline.design import (
    _normalize_base_placement,
    _normalize_slot_coords,
    _normalize_slot_mem_cols,
    _validate_sequence_parallel_placement,
    build_q_block_schedule,
    compute_ffn_col_group_count,
    compute_num_kv_seq_blocks,
    compute_q_blocks_per_lane,
    compute_qkv_head_blocks_per_parallel_head,
)
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline
from iron.operators.encoder_pipeline.reference import generate_golden_reference
from iron.operators.encoder_pipeline.topology import (
    all_supported_topologies,
    filter_topologies_by_alias_or_id,
    load_encoder_pipeline_topology_placements,
    topology_from_key,
)

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
SCALED_SEQ_LENS = tuple(1 << exp for exp in range(6, 15))
BASE_TOPOLOGY = (64, 12, 3072, 32, 64, 96, 64)
TOPOLOGY_CASES = (
    ("", (1, 1, 8, 1, 1, 1)),
    ("_2ps", (2, 1, 8, 1, 1, 1)),
    ("_2ps_2ph", (2, 2, 8, 1, 1, 1)),
    ("_2ps_2pffn", (2, 1, 8, 1, 1, 2)),
    ("_2ps_2ph_2pffn", (2, 2, 8, 1, 1, 2)),
    ("_2ps_4pffn", (2, 1, 8, 1, 1, 4)),
    ("_4ps", (4, 1, 8, 1, 1, 1)),
    ("_2ph", (1, 2, 8, 1, 1, 1)),
    ("_4ph", (1, 4, 8, 1, 1, 1)),
    ("_2pffn", (1, 1, 8, 1, 1, 2)),
    ("_2ph_2pffn", (1, 2, 8, 1, 1, 2)),
    ("_4pffn", (1, 1, 8, 1, 1, 4)),
    ("_2ph_4pffn", (1, 2, 8, 1, 1, 4)),
)


def topology_name(suffix: str) -> str:
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        BASE_TOPOLOGY
    )
    return (
        f"{d}d_{num_heads}h_{ffn_intermediate_size}ffn_{seq_tile}q_"
        f"{kv_seq_tile}kv_{emb_tile}emb_{ffn_tile}ffnt{suffix}"
    )


def generate_test_params():
    params = []
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        BASE_TOPOLOGY
    )
    for topology_suffix, runtime_topology in TOPOLOGY_CASES:
        (
            parallel_seq,
            parallel_heads,
            proj_acc_depth,
            o_proj_acc_group_size,
            ffn_down_acc_group_size,
            nB_tiles_distributed,
        ) = runtime_topology
        for seq_len in SCALED_SEQ_LENS:
            if seq_len % seq_tile != 0 or (seq_len // seq_tile) % parallel_seq != 0:
                continue
            params.append(
                pytest.param(
                    seq_len,
                    d,
                    num_heads,
                    ffn_intermediate_size,
                    seq_tile,
                    kv_seq_tile,
                    emb_tile,
                    ffn_tile,
                    parallel_seq,
                    parallel_heads,
                    proj_acc_depth,
                    o_proj_acc_group_size,
                    ffn_down_acc_group_size,
                    nB_tiles_distributed,
                    id=f"encoder_pipeline_{seq_len}seq_{topology_name(topology_suffix)}",
                )
            )
    return params


all_params = generate_test_params()


def test_topology_key_round_trip_exposes_canonical_and_legacy_ids():
    key = (12, 64, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072)

    topology = topology_from_key(key)

    assert topology.key == key
    assert topology.family_id == "seq32_kv64"
    assert topology.legacy_alias == "2ps_4pffn"
    assert topology.topology_id == "seq32_kv64__ps2_ph1_pffn4"
    assert topology.cache_signature == (2, 1, 4, 32, 64, 96, 64, 8, 1, 3072)


def test_filter_topologies_by_alias_or_id_accepts_legacy_and_canonical_tokens():
    target = topology_from_key((12, 64, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072))
    alternatives = [
        target,
        topology_from_key((12, 64, 64, 32, 64, 96, 64, 1, 1, 8, 1, 1, 3072)),
    ]

    assert filter_topologies_by_alias_or_id(alternatives, {target.legacy_alias}) == [
        target
    ]
    assert filter_topologies_by_alias_or_id(alternatives, {target.topology_id}) == [
        target
    ]


def test_mirrored_64_32_family_is_registered_for_supported_shapes():
    mirrored = topology_from_key((12, 128, 64, 64, 32, 96, 64, 2, 1, 8, 1, 4, 3072))
    large = topology_from_key((16, 64, 64, 64, 32, 128, 64, 1, 1, 8, 1, 1, 4096))
    supported = {topology.key for topology in all_supported_topologies()}

    assert mirrored.key in supported
    assert large.key in supported


def test_topology_ids_are_unique_across_32_64_and_64_32_families():
    dual_family = [
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 2
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 4
    ]

    assert {topology.family_id for topology in dual_family} == {
        "seq32_kv64",
        "seq64_kv32",
    }
    assert {topology.topology_id for topology in dual_family} == {
        "seq32_kv64__ps2_ph1_pffn4",
        "seq64_kv32__ps2_ph1_pffn4",
    }


def test_topology_compute_tile_count_tracks_full_and_underfilled_layouts():
    one_ps = next(
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 1
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
    )
    four_ps = next(
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 4
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
    )

    assert one_ps.compute_tile_count == 8
    assert four_ps.compute_tile_count == 32
    assert one_ps.utilization_fraction == pytest.approx(0.25)
    assert four_ps.utilization_fraction == pytest.approx(1.0)


def test_compute_q_blocks_per_lane_requires_exact_lane_partition():
    assert compute_q_blocks_per_lane(256, 32, 4) == 2

    with pytest.raises(ValueError, match="num_q_seq_blocks divisible by parallel_seq"):
        compute_q_blocks_per_lane(192, 64, 2)


def test_compute_num_kv_seq_blocks_requires_exact_kv_tiling():
    assert compute_num_kv_seq_blocks(256, 64) == 4

    with pytest.raises(ValueError, match="seq_len must be divisible by kv_seq_tile"):
        compute_num_kv_seq_blocks(160, 64)


def test_compute_qkv_head_blocks_requires_parallel_head_partition():
    assert compute_qkv_head_blocks_per_parallel_head(16, 4) == 4

    with pytest.raises(ValueError, match="heads divisible by parallel_heads"):
        compute_qkv_head_blocks_per_parallel_head(12, 5)


def test_compute_ffn_col_group_count_requires_exact_branch_partition():
    assert compute_ffn_col_group_count(3072, 64, 4) == 12

    with pytest.raises(ValueError, match="divide ln1_broadcast_groups"):
        compute_ffn_col_group_count(3072, 64, 5)


def test_build_q_block_schedule_visits_each_q_block_once():
    q_blocks_per_lane = compute_q_blocks_per_lane(256, 32, 4)

    schedule = build_q_block_schedule(4, q_blocks_per_lane)

    assert schedule == [0, 2, 4, 6, 1, 3, 5, 7]
    assert sorted(schedule) == list(range(8))


def test_normalize_slot_coords_supports_single_and_multi_slot_layouts():
    assert _normalize_slot_coords((5, 6), 1, "slot") == [(5, 6)]
    assert _normalize_slot_coords([(1, 2), (3, 4)], 2, "slot") == [
        (1, 2),
        (3, 4),
    ]

    with pytest.raises(ValueError, match="must provide 2 placements"):
        _normalize_slot_coords((1, 2), 2, "slot")


def test_normalize_slot_mem_cols_supports_single_and_multi_slot_layouts():
    assert _normalize_slot_mem_cols(7, 1, "slot") == [7]
    assert _normalize_slot_mem_cols([8, 9], 2, "slot") == [8, 9]

    with pytest.raises(ValueError, match="must provide 2 memtile columns"):
        _normalize_slot_mem_cols(7, 2, "slot")


def test_validate_sequence_parallel_placement_checks_lane_coverage_and_group_size():
    valid = {
        "lane_tiles": [{}, {}, {}, {}],
        "lane_o_proj_acc_mem_cols": [1, 2, 3, 4],
        "lane_tail_mem_cols": [5, 6, 7, 8],
        "transport_groups": [{"lanes": (0, 1)}, {"lanes": (2, 3)}],
    }

    _validate_sequence_parallel_placement(valid, 4)

    with pytest.raises(ValueError, match="cover every lane exactly once"):
        _validate_sequence_parallel_placement(
            {
                **valid,
                "transport_groups": [{"lanes": (0, 1)}, {"lanes": (1, 2)}],
            },
            4,
        )

    with pytest.raises(ValueError, match="uniform size"):
        _validate_sequence_parallel_placement(
            {
                **valid,
                "transport_groups": [{"lanes": (0,)}, {"lanes": (1, 2, 3)}],
            },
            4,
        )


def test_normalize_base_placement_preserves_known_tile_and_mem_columns():
    topology = topology_from_key((12, 128, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072))
    placement = load_encoder_pipeline_topology_placements()[topology.key]

    normalized = _normalize_base_placement(placement)

    assert [tile.col for tile in normalized["qk_tiles"]] == list(placement["mha_cols"])
    assert [tile.row for tile in normalized["qk_tiles"]] == [2]
    assert [tile.col for tile in normalized["o_proj_tiles"]] == list(
        placement["mha_cols"]
    )
    assert len(normalized["ffn_up_tiles"]) == topology.parallel_ffn
    assert len(normalized["ffn_down_tiles"]) == topology.parallel_ffn
    assert normalized["q_mem_col"] == placement["mem_tiles"]["q"]
    assert normalized["output_shim_col"] == placement["shim_tiles"]["output"]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,ffn_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,ffn_down_acc_group_size,nB_tiles_distributed",
    all_params,
)
def test_encoder_pipeline(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=d,
        intermediate_size=ffn_intermediate_size,
        seq_tile=seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
    )

    operator = AIEEncoderPipeline(
        num_heads=num_heads,
        seq_len=seq_len,
        d=d,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        ffn_down_acc_group_size=ffn_down_acc_group_size,
        nB_tiles_distributed=nB_tiles_distributed,
        ffn_intermediate_size=ffn_intermediate_size,
        ln1_weight=golden_ref["ln1_weight"],
        ln2_weight=golden_ref["ln2_weight"],
        context=aie_context,
    )

    input_buffers = {
        "QKV": golden_ref["QKV"],
        "OR": golden_ref["OR"],
        "W_O": golden_ref["W_O"],
        "B_Up": golden_ref["B_Up"],
        "B_Down": golden_ref["B_Down"],
    }
    output_buffers = {"O": golden_ref["O"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=10,
        timed_iters=100,
    )

    max_acceptable_errors = int(seq_len * d * num_heads * ERROR_THRESHOLD)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    print(
        f"({len(errors.get('O', []))} errors out of {max_acceptable_errors} max allowable)"
    )

    assert len(errors.get("O", [])) <= max_acceptable_errors
