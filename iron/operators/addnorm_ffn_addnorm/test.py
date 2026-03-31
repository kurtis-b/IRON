#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.addnorm_ffn_addnorm.topology import (
    _block3_practical_sort_key,
    _is_block3_practical_candidate,
    addnorm_ffn_addnorm_design,
    addnorm_ffn_addnorm_practical_topologies,
    addnorm_ffn_addnorm_theoretical_topologies,
    addnorm_ffn_addnorm_topologies,
)
from iron.operators.addnorm_ffn_addnorm.op import AIEAddNormFFNAddNorm
from iron.operators.addnorm_ffn_addnorm.reference import generate_golden_reference


def _count_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rel_tol: float,
    abs_tol: float,
) -> int:
    return int((~torch.isclose(actual, expected, rtol=rel_tol, atol=abs_tol)).sum())


def generate_test_params():
    workloads = [
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
        (64, 2048, 8192),
        (512, 2048, 8192),
    ]
    params = []
    for seq_len, hidden_size, intermediate_size in workloads:
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            topology_id = str(topology["topology_id"])
            params.append(
                pytest.param(
                    seq_len,
                    hidden_size,
                    intermediate_size,
                    topology_id,
                    id=(
                        f"block3_{seq_len}x{hidden_size}x{intermediate_size}_"
                        f"{topology_id}"
                    ),
                )
            )
    return params


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id",
    generate_test_params(),
)
def test_addnorm_ffn_addnorm(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    aie_context,
):
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    golden = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=7,
    )

    operator = AIEAddNormFFNAddNorm(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
    operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
    operator.ln1_weight = golden["ln1_weight"].contiguous()
    operator.ln2_weight = golden["ln2_weight"].contiguous()
    aie_context.compile_all()
    aie_context.prepare_runtime()

    output = operator.forward(
        golden["hidden_states"],
        golden["residual"],
    )
    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    max_acceptable_errors = int(seq_len * hidden_size * error_threshold)

    assert operator.topology_id == topology_id
    assert operator.topology_family == "pipelined_addnorm_ffn_addnorm"
    assert golden["hidden_states"].shape == (seq_len, hidden_size)
    assert golden["residual"].shape == (seq_len, hidden_size)
    assert golden["ffn_up_weight"].shape == (hidden_size, intermediate_size)
    assert golden["ffn_down_weight"].shape == (intermediate_size, hidden_size)
    assert golden["ln1_weight"].shape == (hidden_size,)
    assert golden["ln2_weight"].shape == (hidden_size,)
    assert golden["output"].shape == (seq_len, hidden_size)
    assert output.shape == golden["output"].shape
    print(f"\nOutput errors: {output_errors} / {max_acceptable_errors}")
    assert output_errors <= max_acceptable_errors


def test_packed_hidden_residual_round_trips(aie_context):
    operator = AIEAddNormFFNAddNorm(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        context=aie_context,
    )
    hidden_states = generate_golden_reference(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        seed=11,
    )
    hidden_padded = hidden_states["hidden_states"].float().numpy()
    residual_padded = hidden_states["residual"].float().numpy()
    packed = operator._pack_hidden_residual(
        hidden_padded,
        residual_padded,
    )
    unpacked_hidden, unpacked_residual = operator._unpack_hidden_residual(packed, 64)

    assert packed.shape == (2 * 64 * 768,)
    assert unpacked_hidden.shape == (64, 768)
    assert unpacked_residual.shape == (64, 768)
    assert np.array_equal(unpacked_hidden, hidden_padded)
    assert np.array_equal(unpacked_residual, residual_padded)


def test_forward_packed_uses_bound_static_weights(aie_context):
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    seq_len = 64
    hidden_size = 768
    intermediate_size = 3072
    topology_id = "m32_k96_n64_ps2_pi6_d8_g1"

    golden = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=17,
    )
    operator = AIEAddNormFFNAddNorm(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
    operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
    operator.ln1_weight = golden["ln1_weight"].contiguous()
    operator.ln2_weight = golden["ln2_weight"].contiguous()

    packed = torch.cat((golden["hidden_states"], golden["residual"]), dim=0)

    aie_context.compile_all()
    aie_context.prepare_runtime()
    output = operator.forward_packed(packed)

    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    max_acceptable_errors = int(seq_len * hidden_size * error_threshold)

    assert output.shape == golden["output"].shape
    assert output_errors <= max_acceptable_errors


def test_theoretical_block3_topologies_cover_supported_surface():
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
        (64, 2048, 8192),
        (512, 2048, 8192),
    ):
        supported_ids = {
            (
                int(topology["tile_m"]),
                int(topology["tile_k"]),
                int(topology["tile_n"]),
                int(topology["down_proj_depth"]),
                int(topology["num_aie_columns"]),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
                int(topology["gelu_stage"]),
            )
            for topology in addnorm_ffn_addnorm_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        theoretical_ids = {
            (
                int(topology["tile_m"]),
                int(topology["tile_k"]),
                int(topology["tile_n"]),
                int(topology["down_proj_depth"]),
                int(topology["num_aie_columns"]),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
                int(topology["gelu_stage"]),
            )
            for topology in addnorm_ffn_addnorm_theoretical_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        assert supported_ids <= theoretical_ids


def test_supported_block3_topologies_generalize_beyond_retained_families():
    topology_ids_1536 = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=64,
            hidden_size=1536,
            intermediate_size=6144,
        )
    }
    topology_ids_960 = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=64,
            hidden_size=960,
            intermediate_size=3840,
        )
    }

    assert topology_ids_1536
    assert topology_ids_960
    assert len(topology_ids_1536) <= 8
    assert len(topology_ids_960) <= 8
    assert "m16_k192_n64_ps4_pi3_d8_g1" in topology_ids_1536
    assert "m16_k120_n160_ps4_pi3_d8_g1" in topology_ids_960


def test_supported_block3_topologies_promote_generated_retained_variants():
    topology_ids_64_768 = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    topology_ids_512_1024 = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=512,
            hidden_size=1024,
            intermediate_size=4096,
        )
    }

    assert len(topology_ids_64_768) == 8
    assert len(topology_ids_512_1024) == 8
    assert "m32_k96_n64_ps2_pi6_d8_g1" in topology_ids_64_768
    assert "m16_k128_n128_ps4_pi3_d6_g1" in topology_ids_64_768
    assert "m32_k128_n64_ps2_pi4_d8_g1" in topology_ids_512_1024
    assert "m64_k64_n128_ps8_pi1_d8_g1" in topology_ids_512_1024


def test_generalized_block3_runtime_topology_runs_numerically(aie_context):
    seq_len = 64
    hidden_size = 1536
    intermediate_size = 6144
    topology_id = str(
        addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )[0]["topology_id"]
    )
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    golden = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=19,
    )

    operator = AIEAddNormFFNAddNorm(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
    operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
    operator.ln1_weight = golden["ln1_weight"].contiguous()
    operator.ln2_weight = golden["ln2_weight"].contiguous()
    aie_context.compile_all()
    aie_context.prepare_runtime()

    output = operator.forward(
        golden["hidden_states"],
        golden["residual"],
    )
    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    max_acceptable_errors = int(seq_len * hidden_size * error_threshold)

    assert operator.topology_id == topology_id
    assert output_errors <= max_acceptable_errors


def test_theoretical_block3_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert "m32_k96_n64_c8_ps2_pi6_d8_g1" in topology_ids
    assert "m32_k96_n64_c8_ps2_pi6_d8_g0" in topology_ids
    assert "m32_k96_n64_c8_ps4_pi3_d8_g1" in topology_ids
    assert "m32_k192_n64_c8_ps4_pi3_d4_g1" in topology_ids
    assert "m32_k96_n128_c8_ps2_pi6_d8_g1" in topology_ids


def test_theoretical_block3_topologies_are_unique_and_contract_valid():
    topologies = addnorm_ffn_addnorm_theoretical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        down_proj_depth = int(topology["down_proj_depth"])
        num_aie_columns = int(topology["num_aie_columns"])
        parallel_seq = int(topology["parallel_seq"])
        parallel_int_dim = int(topology["parallel_int_dim"])
        gelu_stage = int(topology["gelu_stage"])

        assert 512 % (parallel_seq * tile_m) == 0
        assert 768 % tile_k == 0
        assert (768 // tile_k) % down_proj_depth == 0
        assert 3072 % tile_n == 0
        assert 3072 % parallel_int_dim == 0
        assert 3072 % (tile_n * parallel_int_dim) == 0
        assert 1 <= num_aie_columns <= 8
        assert parallel_seq * (2 + 2 * parallel_int_dim) <= num_aie_columns * 4
        assert gelu_stage in (0, 1)


def test_default_block3_design_chooses_highest_core_runtime_topology():
    runtime_topologies = addnorm_ffn_addnorm_topologies(
        seq_len=512,
        hidden_size=2048,
        intermediate_size=8192,
    )
    default_2048 = addnorm_ffn_addnorm_design(
        seq_len=512,
        hidden_size=2048,
        intermediate_size=8192,
    )

    expected = max(
        runtime_topologies,
        key=lambda topology: (
            int(topology["parallel_seq"]) * (2 + 2 * int(topology["parallel_int_dim"])),
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
        ),
    )

    assert default_2048["topology_id"] == str(expected["topology_id"])


def test_practical_block3_topologies_are_subset_of_theoretical_surface():
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    practical_ids = [str(topology["topology_id"]) for topology in practical]

    assert practical
    assert len(practical) <= 16
    assert len(practical) < len(theoretical_ids)
    assert set(practical_ids) <= theoretical_ids


def test_practical_block3_topologies_keep_validated_gelu_stage():
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )
    assert practical
    assert {int(topology["gelu_stage"]) for topology in practical} == {0, 1}


def test_practical_block3_topologies_keep_parallel_seq_buckets_when_feasible():
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
        (64, 2048, 8192),
        (512, 2048, 8192),
    ):
        practical = addnorm_ffn_addnorm_practical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        counts_by_parallel_seq = {1: 0, 2: 0, 4: 0, 8: 0}
        feasible_counts_by_parallel_seq = {1: 0, 2: 0, 4: 0, 8: 0}
        runtime_counts_by_parallel_seq = {1: 0, 2: 0, 4: 0, 8: 0}

        for topology in practical:
            counts_by_parallel_seq[int(topology["parallel_seq"])] += 1

        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            ps = int(topology["parallel_seq"])
            if ps in runtime_counts_by_parallel_seq:
                runtime_counts_by_parallel_seq[ps] += 1

        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            if _is_block3_practical_candidate(
                topology,
                seq_len=seq_len,
                hidden_size=hidden_size,
            ):
                ps = int(topology["parallel_seq"])
                if ps in feasible_counts_by_parallel_seq:
                    feasible_counts_by_parallel_seq[ps] += 1

        for parallel_seq in (1, 2, 4, 8):
            if feasible_counts_by_parallel_seq[parallel_seq] > 0:
                assert counts_by_parallel_seq[parallel_seq] > 0
            assert counts_by_parallel_seq[parallel_seq] <= (
                4 + runtime_counts_by_parallel_seq[parallel_seq]
            )


def test_practical_block3_topologies_are_ranked_and_pruned():
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )

    assert practical == sorted(practical, key=_block3_practical_sort_key, reverse=True)
    practical_counts_by_parallel_seq = {1: 0, 2: 0, 4: 0, 8: 0}
    runtime_counts_by_parallel_seq = {1: 0, 2: 0, 4: 0, 8: 0}
    for topology in addnorm_ffn_addnorm_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    ):
        parallel_seq = int(topology["parallel_seq"])
        runtime_counts_by_parallel_seq[parallel_seq] = (
            runtime_counts_by_parallel_seq.get(parallel_seq, 0) + 1
        )
    for topology in practical:
        parallel_seq = int(topology["parallel_seq"])
        practical_counts_by_parallel_seq[parallel_seq] = (
            practical_counts_by_parallel_seq.get(parallel_seq, 0) + 1
        )
        assert int(topology["tile_m"]) in (16, 32, 64)
        assert parallel_seq in (1, 2, 4, 8)
        assert int(topology["tile_k"]) >= 16
        assert int(topology["tile_n"]) >= 16
        assert int(topology["num_aie_columns"]) == 8
        assert parallel_seq * int(topology["parallel_int_dim"]) >= 8
        assert topology["topology_family"] == "pipelined_addnorm_ffn_addnorm_practical"
    assert all(
        practical_counts_by_parallel_seq[parallel_seq]
        <= 4 + runtime_counts_by_parallel_seq[parallel_seq]
        for parallel_seq in practical_counts_by_parallel_seq
    )


def test_practical_block3_topologies_keep_only_highest_depth_per_shape():
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
        (64, 2048, 8192),
        (512, 2048, 8192),
    ):
        practical = addnorm_ffn_addnorm_practical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        runtime_shape_depths = {
            (
                int(topology["tile_m"]),
                int(topology["tile_k"]),
                int(topology["tile_n"]),
                int(topology["num_aie_columns"]),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
                int(topology["gelu_stage"]),
            ): int(topology["down_proj_depth"])
            for topology in addnorm_ffn_addnorm_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        feasible_max_depth_by_shape: dict[tuple[int, ...], int] = {}
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            if not _is_block3_practical_candidate(
                topology,
                seq_len=seq_len,
                hidden_size=hidden_size,
            ):
                continue
            shape_key = (
                int(topology["tile_m"]),
                int(topology["tile_k"]),
                int(topology["tile_n"]),
                int(topology["num_aie_columns"]),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
                int(topology["gelu_stage"]),
            )
            feasible_max_depth_by_shape[shape_key] = max(
                feasible_max_depth_by_shape.get(shape_key, 0),
                int(topology["down_proj_depth"]),
            )

        seen_shapes: set[tuple[int, ...]] = set()
        for topology in practical:
            shape_key = (
                int(topology["tile_m"]),
                int(topology["tile_k"]),
                int(topology["tile_n"]),
                int(topology["num_aie_columns"]),
                int(topology["parallel_seq"]),
                int(topology["parallel_int_dim"]),
                int(topology["gelu_stage"]),
            )
            assert shape_key not in seen_shapes
            seen_shapes.add(shape_key)
            actual_depth = int(topology["down_proj_depth"])
            runtime_depth = runtime_shape_depths.get(shape_key)
            max_depth = feasible_max_depth_by_shape.get(shape_key)
            if max_depth is None:
                assert runtime_depth is not None
                assert actual_depth == runtime_depth
            else:
                assert actual_depth == max_depth or actual_depth == runtime_depth
