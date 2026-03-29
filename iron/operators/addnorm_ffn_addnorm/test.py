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


def _block3_runtime_signature(topology: dict[str, int | str]) -> tuple[int, ...]:
    return (
        int(topology["compile_rows"]),
        int(topology["tile_m"]),
        int(topology["tile_k"]),
        int(topology["tile_n"]),
        int(topology["down_proj_depth"]),
        int(topology["num_aie_columns"]),
        int(topology["parallel_seq"]),
        int(topology["parallel_int_dim"]),
        int(topology["gelu_stage"]),
    )


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
    padded_rows = ((64 + operator.M - 1) // operator.M) * operator.M
    hidden_padded = np.zeros((padded_rows, 768), dtype=np.float32)
    residual_padded = np.zeros((padded_rows, 768), dtype=np.float32)
    hidden_padded[:64, :] = hidden_states["hidden_states"].float().numpy()
    residual_padded[:64, :] = hidden_states["residual"].float().numpy()
    packed = operator._pack_hidden_residual(
        hidden_padded,
        residual_padded,
    )
    unpacked_hidden, unpacked_residual = operator._unpack_hidden_residual(
        packed, padded_rows
    )

    assert packed.shape == (2 * padded_rows * 768,)
    assert unpacked_hidden.shape == (padded_rows, 768)
    assert unpacked_residual.shape == (padded_rows, 768)
    assert np.array_equal(
        unpacked_hidden[:64, :],
        hidden_padded[:64, :],
    )
    assert np.array_equal(
        unpacked_residual[:64, :],
        residual_padded[:64, :],
    )


def test_theoretical_block3_topologies_cover_supported_surface():
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
        (512, 1024, 4096),
    ):
        supported_ids = {
            (
                int(topology["compile_rows"]),
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
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        theoretical_ids = {
            (
                int(topology["compile_rows"]),
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


def test_theoretical_block3_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert "cr128_m32_k96_n64_c8_ps2_pi6_d8_g1" in topology_ids
    assert "cr128_m32_k96_n64_c8_ps2_pi6_d8_g0" in topology_ids
    assert "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1" in topology_ids
    assert "cr128_m32_k192_n64_c8_ps4_pi3_d4_g1" in topology_ids
    assert "cr128_m32_k96_n128_c8_ps2_pi6_d8_g1" in topology_ids


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
        compile_rows = int(topology["compile_rows"])
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        down_proj_depth = int(topology["down_proj_depth"])
        num_aie_columns = int(topology["num_aie_columns"])
        parallel_seq = int(topology["parallel_seq"])
        parallel_int_dim = int(topology["parallel_int_dim"])
        gelu_stage = int(topology["gelu_stage"])

        assert compile_rows % (parallel_seq * tile_m) == 0
        assert 768 % tile_k == 0
        assert down_proj_depth == 768 // tile_k
        assert 3072 % tile_n == 0
        assert 3072 % parallel_int_dim == 0
        assert 3072 % (tile_n * parallel_int_dim) == 0
        assert 1 <= num_aie_columns <= 8
        assert parallel_seq * (1 + 2 * parallel_int_dim) <= num_aie_columns * 4
        assert gelu_stage in (0, 1)


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
    assert len(practical) <= 64
    assert len(practical) < len(theoretical_ids)
    assert set(practical_ids) <= theoretical_ids


def test_practical_block3_topologies_include_retained_runtime_surface():
    practical_signatures = {
        (
            int(topology["compile_rows"]),
            int(topology["tile_m"]),
            int(topology["tile_k"]),
            int(topology["tile_n"]),
            int(topology["down_proj_depth"]),
            int(topology["num_aie_columns"]),
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
            int(topology["gelu_stage"]),
        )
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert (128, 32, 96, 64, 8, 8, 2, 6, 1) in practical_signatures
    assert (128, 32, 96, 64, 8, 8, 4, 3, 1) in practical_signatures


def test_practical_block3_topologies_keep_validated_gelu_stage():
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )
    assert practical
    assert {int(topology["gelu_stage"]) for topology in practical} == {1}


def test_practical_block3_topologies_keep_validated_runtime_signature_families():
    expected_768 = {
        (96, 64, 8, 8, 2, 6, 1),
        (96, 64, 8, 8, 4, 3, 1),
    }
    actual_768 = {
        (
            int(topology["tile_k"]),
            int(topology["tile_n"]),
            int(topology["down_proj_depth"]),
            int(topology["num_aie_columns"]),
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
            int(topology["gelu_stage"]),
        )
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    expected_1024 = {
        (128, 32, 8, 8, 4, 2, 1),
    }
    actual_1024 = {
        (
            int(topology["tile_k"]),
            int(topology["tile_n"]),
            int(topology["down_proj_depth"]),
            int(topology["num_aie_columns"]),
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
            int(topology["gelu_stage"]),
        )
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=512,
            hidden_size=1024,
            intermediate_size=4096,
        )
    }

    assert actual_768 <= expected_768
    assert actual_1024 <= expected_1024


def test_practical_block3_topologies_are_ranked_and_pruned():
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
    )

    assert practical == sorted(practical, key=_block3_practical_sort_key, reverse=True)
    retained_signatures = {
        (128, 32, 96, 64, 8, 8, 2, 6, 1),
        (128, 32, 96, 64, 8, 8, 4, 3, 1),
    }
    for topology in practical:
        signature = (
            int(topology["compile_rows"]),
            int(topology["tile_m"]),
            int(topology["tile_k"]),
            int(topology["tile_n"]),
            int(topology["down_proj_depth"]),
            int(topology["num_aie_columns"]),
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
            int(topology["gelu_stage"]),
        )
        if signature not in retained_signatures:
            assert int(topology["compile_rows"]) >= 128
            assert int(topology["tile_m"]) >= 32
            assert int(topology["tile_k"]) >= 64
            assert int(topology["tile_n"]) >= 32
            assert int(topology["num_aie_columns"]) >= 4
            assert (
                int(topology["parallel_seq"]) * int(topology["parallel_int_dim"]) >= 8
            )
        assert topology["topology_family"] == "pipelined_addnorm_ffn_addnorm_practical"
