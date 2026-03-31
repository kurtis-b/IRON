#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.mha_out_proj.topology import (
    _block2_stage_working_sets_fit,
    _block2_practical_sort_key,
    mha_out_proj_practical_topologies,
    mha_out_proj_theoretical_topologies,
    mha_out_proj_topologies,
)
from iron.operators.mha_out_proj.op import (
    AIEMHAOutProj,
    _pack_block3_residual_output,
    _unpack_block3_attention_output,
)
from iron.operators.mha_out_proj.reference import generate_golden_reference
from iron.common.test_utils import run_test

DEBUG_MODE = 0


def generate_test_params():
    workloads = [
        (64, 64, 1),
        (512, 64, 1),
        (64, 64, 12),
        (512, 64, 12),
        (64, 64, 16),
        (512, 64, 16),
        (1984, 64, 16),
        (2048, 64, 16),
    ]
    params = []
    for seq_len, head_dim, num_heads in workloads:
        supported_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        }
        for topology in mha_out_proj_practical_topologies(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ):
            topology_id = str(topology["topology_id"])
            marks = ()
            if topology_id not in supported_ids:
                marks = pytest.mark.skip(
                    reason="Block 2 practical topology is not runtime-supported yet"
                )
            params.append(
                pytest.param(
                    seq_len,
                    head_dim,
                    num_heads,
                    topology_id,
                    id=(
                        f"mha_out_proj_{num_heads}heads_{seq_len}seq_{head_dim}hdim_"
                        f"{topology_id}"
                    ),
                    marks=marks,
                )
            )
    return params


regular_params = generate_test_params()


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,head_dim,num_heads,topology_id",
    regular_params,
)
def test_mha_out_proj(
    seq_len,
    head_dim,
    num_heads,
    topology_id,
    aie_context,
):
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
        debug=DEBUG_MODE,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id=topology_id,
        debug=DEBUG_MODE,
        context=aie_context,
    )

    input_buffers = {
        "Q": golden_ref["Q"].flatten(),
        "K": golden_ref["K"].flatten(),
        "V": golden_ref["V"].flatten(),
        "W_O": golden_ref["W_O"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
    )

    error_threshold = 0.005
    max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    assert operator.topology_id == topology_id
    assert operator.topology_family == "fused_mha_out_proj"
    if errors:
        print(
            "({} errors out of {} max allowable)".format(
                len(errors["O"]), max_acceptable_errors
            )
        )
        assert (
            len(errors["O"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['O'])} errors (max allowable: {max_acceptable_errors})"


def test_theoretical_block2_topologies_cover_supported_surface():
    for seq_len, num_heads in ((64, 1), (512, 12), (2048, 16)):
        supported_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=64,
            )
        }
        theoretical_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_theoretical_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=64,
            )
        }
        assert supported_ids <= theoretical_ids


def test_packed_block3_output_layout_reserves_a_half_and_places_residual_half():
    seq_len = 64
    embed_sz = 768
    q_seq_tile = 32
    emb_tile = 96
    parallel_seq = 4
    packed_rows = 128

    residual = np.arange(seq_len * embed_sz, dtype=np.float32).reshape(
        seq_len, embed_sz
    )
    packed = _pack_block3_residual_output(
        residual,
        seq_len=seq_len,
        packed_rows=packed_rows,
        embed_sz=embed_sz,
        q_seq_tile=q_seq_tile,
        emb_tile=emb_tile,
        parallel_seq=parallel_seq,
    )
    unpacked = _unpack_block3_attention_output(
        packed,
        seq_len=seq_len,
        packed_rows=packed_rows,
        embed_sz=embed_sz,
        q_seq_tile=q_seq_tile,
        emb_tile=emb_tile,
        parallel_seq=parallel_seq,
    )

    assert packed.shape == (2 * packed_rows * embed_sz,)
    assert np.count_nonzero(unpacked) == 0
    tile_elems = q_seq_tile * emb_tile
    first_tile_residual = packed[tile_elems : 2 * tile_elems].reshape(
        q_seq_tile, emb_tile
    )
    assert np.array_equal(first_tile_residual, residual[:q_seq_tile, :emb_tile])


def test_block2_packed_output_uses_distinct_artifacts(aie_context):
    plain = AIEMHAOutProj(
        num_heads=12,
        seq_len=64,
        d=64,
        topology_id="q32_kv64_e96_ps1_ph1_acc1",
        context=aie_context,
    )
    packed = AIEMHAOutProj(
        num_heads=12,
        seq_len=64,
        d=64,
        topology_id="q32_kv64_e96_ps1_ph1_acc1",
        context=aie_context,
        packed_output_parallel_seq=1,
        packed_output_rows=64,
    )

    aie_context.compile_all()

    assert plain.xclbin_artifact.path.name != packed.xclbin_artifact.path.name
    assert plain.insts_artifact.path.name != packed.insts_artifact.path.name


def test_supported_block2_topologies_include_promoted_runtime_variants():
    topology_ids_1_seq64 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=64,
            num_heads=1,
            head_dim=64,
        )
    }
    topology_ids_1 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=512,
            num_heads=1,
            head_dim=64,
        )
    }
    topology_ids_12 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=512,
            num_heads=12,
            head_dim=64,
        )
    }
    topology_ids_12_seq384 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=384,
            num_heads=12,
            head_dim=64,
        )
    }
    topology_ids_16 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=512,
            num_heads=16,
            head_dim=64,
        )
    }
    topology_ids_16_seq384 = {
        str(topology["topology_id"])
        for topology in mha_out_proj_topologies(
            seq_len=384,
            num_heads=16,
            head_dim=64,
        )
    }

    assert "q32_kv64_e64_ps2_ph1_acc1" in topology_ids_1_seq64
    assert "q32_kv64_e64_ps2_ph1_acc1" not in topology_ids_1
    assert "q32_kv64_e64_ps4_ph1_acc1" not in topology_ids_1
    assert "q32_kv64_e96_ps1_ph2_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps1_ph4_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps1_ph6_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps2_ph1_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps4_ph1_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps8_ph1_acc1" in topology_ids_12
    assert "q32_kv64_e96_ps6_ph1_acc1" in topology_ids_12_seq384
    assert "q32_kv64_e128_ps1_ph2_acc1" in topology_ids_16
    assert "q32_kv64_e128_ps1_ph4_acc1" in topology_ids_16
    assert "q32_kv64_e128_ps2_ph1_acc1" in topology_ids_16
    assert "q32_kv64_e128_ps4_ph1_acc1" in topology_ids_16
    assert "q32_kv64_e128_ps8_ph1_acc1" in topology_ids_16
    assert "q32_kv64_e128_ps6_ph1_acc1" in topology_ids_16_seq384
    assert "q32_kv64_e128_ps1_ph8_acc1" not in topology_ids_16


def test_block2_accepts_flat_qkv_without_head_major_reshape(aie_context):
    seq_len = 64
    head_dim = 64
    num_heads = 12
    topology_id = str(
        mha_out_proj_topologies(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )[0]["topology_id"]
    )
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
        seed=17,
        debug=DEBUG_MODE,
    )
    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id=topology_id,
        debug=DEBUG_MODE,
        context=aie_context,
    )
    aie_context.compile_all()
    aie_context.prepare_runtime()

    q_head = golden_ref["Q"].view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    k_head = golden_ref["K"].view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    v_head = golden_ref["V"].view(seq_len, num_heads, head_dim).permute(1, 0, 2)

    o_head = operator.forward(q_head, k_head, v_head, golden_ref["W_O"])
    o_flat = operator.forward(
        golden_ref["Q"],
        golden_ref["K"],
        golden_ref["V"],
        golden_ref["W_O"],
    )

    assert torch.equal(o_head, o_flat)


def test_block2_packed_output_matches_dense_output(aie_context):
    seq_len = 64
    head_dim = 64
    num_heads = 12
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
        seed=29,
        debug=DEBUG_MODE,
    )

    plain = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id="q32_kv64_e96_ps1_ph1_acc1",
        context=aie_context,
    )
    packed = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id="q32_kv64_e96_ps1_ph1_acc1",
        context=aie_context,
        packed_output_parallel_seq=1,
        packed_output_rows=seq_len,
    )

    aie_context.compile_all()
    aie_context.prepare_runtime()

    dense_out = plain.forward(
        golden_ref["Q"],
        golden_ref["K"],
        golden_ref["V"],
        golden_ref["W_O"],
    )
    packed_out = packed.forward(
        golden_ref["Q"],
        golden_ref["K"],
        golden_ref["V"],
        golden_ref["W_O"],
    )

    assert torch.equal(dense_out, packed_out)


def test_theoretical_block2_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in mha_out_proj_theoretical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    }
    assert "q32_kv64_e128_ps1_ph1_acc1" in topology_ids
    assert "q32_kv64_e128_ps1_ph2_acc1" in topology_ids
    assert "q32_kv64_e128_ps1_ph1_acc2" in topology_ids
    assert "q32_kv64_e128_ps1_ph1_acc8" in topology_ids
    assert "q32_kv128_e64_ps1_ph1_acc1" in topology_ids


def test_theoretical_block2_topologies_exclude_nonrunnable_l1_overflows():
    topology_ids = {
        str(topology["topology_id"])
        for topology in mha_out_proj_theoretical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    }
    assert "q64_kv64_e128_ps1_ph1_acc1" not in topology_ids
    assert "q32_kv64_e128_ps1_ph8_acc1" not in topology_ids
    assert "q32_kv64_e64_ps1_ph1_acc16" not in topology_ids


def test_theoretical_block2_topologies_are_unique_and_contract_valid():
    topologies = mha_out_proj_theoretical_topologies(
        seq_len=2048,
        num_heads=16,
        head_dim=64,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_heads = int(topology["parallel_heads"])
        q_seq_tile = int(topology["q_seq_tile"])
        kv_seq_tile = int(topology["kv_seq_tile"])
        emb_tile = int(topology["emb_tile"])
        o_proj_acc_depth = int(topology["o_proj_acc_depth"])

        assert parallel_seq in (1, 2, 4, 6, 8)
        assert 2048 % (parallel_seq * q_seq_tile) == 0
        assert 2048 % kv_seq_tile == 0
        assert 16 % parallel_heads == 0
        assert parallel_seq * parallel_heads <= 8
        assert 1024 % (emb_tile * o_proj_acc_depth) == 0
        assert _block2_stage_working_sets_fit(
            q_seq_tile=q_seq_tile,
            kv_seq_tile=kv_seq_tile,
            emb_tile=emb_tile,
            head_dim=64,
        )


def test_practical_block2_topologies_are_subset_of_theoretical_surface():
    practical = mha_out_proj_practical_topologies(
        seq_len=2048,
        num_heads=16,
        head_dim=64,
    )
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in mha_out_proj_theoretical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    }
    practical_ids = [str(topology["topology_id"]) for topology in practical]

    assert practical
    assert len(practical) <= 64
    assert len(practical) < len(theoretical_ids)
    assert set(practical_ids) <= theoretical_ids


def test_practical_block2_topologies_include_retained_runtime_surface():
    practical_ids = {
        str(topology["topology_id"])
        for topology in mha_out_proj_practical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    }
    assert "q32_kv64_e128_ps1_ph1_acc1" in practical_ids
    assert "q32_kv64_e128_ps1_ph2_acc1" in practical_ids


def test_practical_block2_topologies_prioritize_real_lowered_axes():
    practical_ids = [
        str(topology["topology_id"])
        for topology in mha_out_proj_practical_topologies(
            seq_len=2048,
            num_heads=16,
            head_dim=64,
        )
    ]

    promoted_parallel_head = "q32_kv64_e128_ps1_ph2_acc1"
    unreal_parallel_seq = "q32_kv128_e128_ps8_ph1_acc1"

    assert promoted_parallel_head in practical_ids
    if unreal_parallel_seq in practical_ids:
        assert practical_ids.index(promoted_parallel_head) < practical_ids.index(
            unreal_parallel_seq
        )


def test_practical_block2_topologies_are_ranked_and_pruned():
    practical = mha_out_proj_practical_topologies(
        seq_len=2048,
        num_heads=16,
        head_dim=64,
    )

    assert practical == sorted(
        practical,
        key=lambda candidate: _block2_practical_sort_key(candidate, head_dim=64),
        reverse=True,
    )
    for topology in practical:
        lane_parallelism = int(topology["parallel_heads"]) * int(
            topology["parallel_seq"]
        )
        sequence_chunk = int(topology["q_seq_tile"]) * int(topology["parallel_seq"])
        if str(topology["topology_id"]) != "q32_kv64_e128_ps1_ph1_acc1":
            assert int(topology["q_seq_tile"]) >= 32
            assert int(topology["kv_seq_tile"]) >= 64
            assert int(topology["emb_tile"]) >= 64
            assert lane_parallelism >= 2
            assert sequence_chunk >= 32
        assert topology["topology_family"] == "fused_mha_out_proj_practical"

    practical_ids = {str(topology["topology_id"]) for topology in practical}
    assert "q64_kv64_e128_ps1_ph1_acc1" not in practical_ids


def test_practical_block2_sort_key_favors_higher_acc_depth():
    lower_acc = {
        "parallel_seq": 1,
        "q_seq_tile": 32,
        "kv_seq_tile": 64,
        "emb_tile": 128,
        "parallel_heads": 1,
        "o_proj_acc_depth": 1,
    }
    higher_acc = {
        **lower_acc,
        "o_proj_acc_depth": 8,
    }

    assert _block2_practical_sort_key(
        higher_acc, head_dim=64
    ) > _block2_practical_sort_key(lower_acc, head_dim=64)


@pytest.mark.parametrize(
    "seq_len,num_heads,topology_id",
    (
        (64, 1, "q32_kv64_e64_ps2_ph1_acc1"),
        (512, 12, "q32_kv64_e96_ps2_ph1_acc1"),
        (512, 12, "q32_kv64_e96_ps4_ph1_acc1"),
        (512, 12, "q32_kv64_e96_ps8_ph1_acc1"),
        (384, 12, "q32_kv64_e96_ps6_ph1_acc1"),
        (512, 16, "q32_kv64_e128_ps8_ph1_acc1"),
        (384, 16, "q32_kv64_e128_ps6_ph1_acc1"),
    ),
)
def test_block2_sequence_parallel_topologies_run_numerically(
    seq_len, num_heads, topology_id, aie_context
):
    head_dim = 64
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=head_dim,
        heads=num_heads,
        seed=23,
        debug=DEBUG_MODE,
    )

    operator = AIEMHAOutProj(
        num_heads=num_heads,
        seq_len=seq_len,
        d=head_dim,
        topology_id=topology_id,
        debug=DEBUG_MODE,
        context=aie_context,
    )
    input_buffers = {
        "Q": golden_ref["Q"].flatten(),
        "K": golden_ref["K"].flatten(),
        "V": golden_ref["V"].flatten(),
        "W_O": golden_ref["W_O"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, _, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
    )
    max_acceptable_errors = int(seq_len * head_dim * num_heads * 0.005)
    if errors:
        assert len(errors["O"]) <= max_acceptable_errors
