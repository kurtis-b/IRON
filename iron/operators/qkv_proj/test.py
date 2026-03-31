#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.qkv_proj.topology import (
    _block1_compute_tile_working_set_fits,
    _block1_compute_tile_shape_score,
    _block1_practical_topologies,
    _block1_practical_sort_key,
    _block1_theoretical_topologies,
    qkv_proj_design,
    qkv_proj_topologies,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import generate_golden_reference


def _count_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rel_tol: float,
    abs_tol: float,
) -> int:
    return int((~torch.isclose(actual, expected, rtol=rel_tol, atol=abs_tol)).sum())


def _flatten_head_major(tensor: torch.Tensor) -> torch.Tensor:
    num_heads, seq_len, head_dim = tensor.shape
    return tensor.permute(1, 0, 2).contiguous().view(seq_len, num_heads * head_dim)


def _first_block1_runtime_topology_id(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    predicate,
) -> str:
    for topology in qkv_proj_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
    ):
        if predicate(topology):
            return str(topology["topology_id"])
    raise AssertionError("expected matching Block 1 runtime topology")


def generate_test_params():
    workloads = [
        (64, 12, 64),
        (512, 12, 64),
        (64, 16, 64),
        (512, 16, 64),
    ]
    params = []
    for seq_len, num_heads, head_dim in workloads:
        for topology in qkv_proj_topologies(
            seq_len=seq_len,
            hidden_size=num_heads * head_dim,
            num_heads=num_heads,
        ):
            topology_id = str(topology["topology_id"])
            params.append(
                pytest.param(
                    seq_len,
                    num_heads,
                    head_dim,
                    topology_id,
                    id=f"block1_{seq_len}_{num_heads}_{head_dim}_{topology_id}",
                )
            )
    return params


@pytest.mark.parametrize(
    "seq_len,num_heads,head_dim,topology_id",
    generate_test_params(),
)
def test_qkv_proj(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    topology_id: str,
    aie_context,
):
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=7,
    )

    operator = AIEQKVProj(
        seq_len=seq_len,
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.q_proj.weight = golden_ref["q_proj_weight"].contiguous()
    operator.k_proj.weight = golden_ref["k_proj_weight"].contiguous()
    operator.v_proj.weight = golden_ref["v_proj_weight"].contiguous()
    aie_context.compile_all()
    aie_context.prepare_runtime()

    q, k, v = operator.forward(golden_ref["hidden_states"])
    max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)
    q_errors = _count_errors(q, golden_ref["q"], rel_tol=rel_tol, abs_tol=abs_tol)
    k_errors = _count_errors(k, golden_ref["k"], rel_tol=rel_tol, abs_tol=abs_tol)
    v_errors = _count_errors(v, golden_ref["v"], rel_tol=rel_tol, abs_tol=abs_tol)

    assert operator.topology_id == topology_id
    assert operator.topology_family == "shared_runtime_qkv_proj"
    assert q.shape == golden_ref["q"].shape
    assert k.shape == golden_ref["k"].shape
    assert v.shape == golden_ref["v"].shape
    print(f"\nQ errors: {q_errors} / {max_acceptable_errors}")
    print(f"K errors: {k_errors} / {max_acceptable_errors}")
    print(f"V errors: {v_errors} / {max_acceptable_errors}")
    assert q_errors <= max_acceptable_errors
    assert k_errors <= max_acceptable_errors
    assert v_errors <= max_acceptable_errors


@pytest.mark.parametrize(
    "seq_len,num_heads,head_dim",
    [
        (64, 24, 64),
        (64, 12, 80),
    ],
)
def test_qkv_proj_generalized_runtime_workloads(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    aie_context,
):
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    hidden_size = num_heads * head_dim
    topology_id = str(
        qkv_proj_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )[0]["topology_id"]
    )
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=11,
    )

    operator = AIEQKVProj(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.q_proj.weight = golden_ref["q_proj_weight"].contiguous()
    operator.k_proj.weight = golden_ref["k_proj_weight"].contiguous()
    operator.v_proj.weight = golden_ref["v_proj_weight"].contiguous()
    aie_context.compile_all()
    aie_context.prepare_runtime()

    q, k, v = operator.forward(golden_ref["hidden_states"])
    max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)
    q_errors = _count_errors(q, golden_ref["q"], rel_tol=rel_tol, abs_tol=abs_tol)
    k_errors = _count_errors(k, golden_ref["k"], rel_tol=rel_tol, abs_tol=abs_tol)
    v_errors = _count_errors(v, golden_ref["v"], rel_tol=rel_tol, abs_tol=abs_tol)

    assert q_errors <= max_acceptable_errors
    assert k_errors <= max_acceptable_errors
    assert v_errors <= max_acceptable_errors


def test_qkv_proj_known_runtime_k24_n32_topology_compiles_and_runs(aie_context):
    seq_len = 64
    num_heads = 24
    head_dim = 64
    hidden_size = num_heads * head_dim
    topology_id = "m16_k24_n32_c8_ps4_pe8"
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    error_threshold = 0.005
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=17,
    )

    operator = AIEQKVProj(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        topology_id=topology_id,
        context=aie_context,
    )
    operator.q_proj.weight = golden_ref["q_proj_weight"].contiguous()
    operator.k_proj.weight = golden_ref["k_proj_weight"].contiguous()
    operator.v_proj.weight = golden_ref["v_proj_weight"].contiguous()
    aie_context.compile_all()
    aie_context.prepare_runtime()

    q, k, v = operator.forward(golden_ref["hidden_states"])
    max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)
    q_errors = _count_errors(q, golden_ref["q"], rel_tol=rel_tol, abs_tol=abs_tol)
    k_errors = _count_errors(k, golden_ref["k"], rel_tol=rel_tol, abs_tol=abs_tol)
    v_errors = _count_errors(v, golden_ref["v"], rel_tol=rel_tol, abs_tol=abs_tol)

    assert q_errors <= max_acceptable_errors
    assert k_errors <= max_acceptable_errors
    assert v_errors <= max_acceptable_errors


def test_qkv_proj_forward_flat_matches_reference_without_head_major_reshape(
    aie_context,
):
    seq_len = 64
    num_heads = 12
    head_dim = 64
    hidden_size = num_heads * head_dim
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=19,
    )
    topology_ids = (
        "m32_k256_n24_c8_ps1_pe1",
        _first_block1_runtime_topology_id(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            predicate=lambda topology: int(topology["parallel_emb"]) > 1,
        ),
    )

    operators = []
    for topology_id in topology_ids:
        operator = AIEQKVProj(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            topology_id=topology_id,
            context=aie_context,
        )
        operator.q_proj.weight = golden_ref["q_proj_weight"].contiguous()
        operator.k_proj.weight = golden_ref["k_proj_weight"].contiguous()
        operator.v_proj.weight = golden_ref["v_proj_weight"].contiguous()
        operators.append(operator)

    aie_context.compile_all()
    aie_context.prepare_runtime()

    expected_q = _flatten_head_major(golden_ref["q"])
    expected_k = _flatten_head_major(golden_ref["k"])
    expected_v = _flatten_head_major(golden_ref["v"])
    rel_tol = 4.0e-2
    abs_tol = 1.5e-1
    max_acceptable_errors = int(seq_len * hidden_size * 0.005)

    for operator in operators:
        q, k, v = operator.forward_flat(golden_ref["hidden_states"])
        assert operator.get_bo("Q") is not None
        assert operator.get_bo("K") is not None
        assert operator.get_bo("V") is not None
        assert (
            _count_errors(q, expected_q, rel_tol=rel_tol, abs_tol=abs_tol)
            <= max_acceptable_errors
        )
        assert (
            _count_errors(k, expected_k, rel_tol=rel_tol, abs_tol=abs_tol)
            <= max_acceptable_errors
        )
        assert (
            _count_errors(v, expected_v, rel_tol=rel_tol, abs_tol=abs_tol)
            <= max_acceptable_errors
        )


def test_theoretical_block1_topologies_cover_supported_surface():
    for seq_len, hidden_size, num_heads in (
        (64, 768, 12),
        (512, 768, 12),
        (512, 1024, 16),
        (512, 1536, 24),
        (512, 960, 12),
    ):
        supported_ids = {
            str(topology["topology_id"])
            for topology in qkv_proj_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        theoretical_ids = {
            str(topology["topology_id"])
            for topology in _block1_theoretical_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        assert supported_ids <= theoretical_ids


def test_runtime_block1_topologies_are_subset_of_practical_surface():
    for seq_len, hidden_size, num_heads in (
        (64, 768, 12),
        (512, 768, 12),
        (64, 1024, 16),
        (512, 1024, 16),
        (512, 1536, 24),
        (512, 960, 12),
    ):
        runtime_ids = {
            str(topology["topology_id"])
            for topology in qkv_proj_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        practical_ids = {
            str(topology["topology_id"])
            for topology in _block1_practical_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        assert runtime_ids <= practical_ids


def test_runtime_block1_topologies_lower_parallel_axes_honestly():
    for seq_len, hidden_size, num_heads in (
        (64, 768, 12),
        (512, 768, 12),
        (64, 1024, 16),
        (512, 1024, 16),
        (512, 1536, 24),
        (512, 960, 12),
    ):
        runtime_topologies = qkv_proj_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        assert runtime_topologies
        assert any(int(topology["parallel_seq"]) > 1 for topology in runtime_topologies)
        assert any(int(topology["parallel_emb"]) > 1 for topology in runtime_topologies)
        for topology in runtime_topologies:
            assert int(topology["parallel_seq"]) in (1, 2, 4)
            assert int(topology["parallel_emb"]) >= 1
            assert hidden_size % int(topology["parallel_emb"]) == 0
            assert int(topology["num_aie_columns"]) % int(topology["parallel_emb"]) == 0


def test_runtime_block1_topologies_are_pruned_and_generalized():
    retained_runtime = qkv_proj_topologies(
        seq_len=512,
        hidden_size=768,
        num_heads=12,
    )
    generalized_runtime = qkv_proj_topologies(
        seq_len=512,
        hidden_size=1536,
        num_heads=24,
    )

    assert 0 < len(retained_runtime) <= 12
    assert 0 < len(generalized_runtime) <= 12
    assert all(
        int(topology["num_aie_columns"]) in (6, 8) for topology in retained_runtime
    )
    assert all(
        int(topology["num_aie_columns"]) in (6, 8) for topology in generalized_runtime
    )
    assert any(int(topology["parallel_seq"]) > 1 for topology in retained_runtime)
    assert any(int(topology["parallel_emb"]) > 1 for topology in generalized_runtime)


def test_runtime_block1_topologies_can_select_c6_for_768_family():
    runtime_topologies = qkv_proj_topologies(
        seq_len=64,
        hidden_size=768,
        num_heads=12,
    )

    assert any(int(topology["num_aie_columns"]) == 6 for topology in runtime_topologies)


def test_block1_design_accepts_canonical_runtime_topology_ids():
    config = qkv_proj_design(
        seq_len=64,
        hidden_size=768,
        num_heads=12,
        topology_id="m32_k256_n24_c8_ps1_pe1",
    )

    assert config["topology_id"] == "m32_k256_n24_c8_ps1_pe1"
    assert config["num_aie_columns"] == 8
    assert config["tile_n"] == 24


def test_block1_design_accepts_canonical_parallel_emb_runtime_ids():
    config = qkv_proj_design(
        seq_len=64,
        hidden_size=768,
        num_heads=12,
        topology_id="m32_k256_n24_c8_ps1_pe4",
    )

    assert config["topology_id"] == "m32_k256_n24_c8_ps1_pe4"
    assert config["parallel_emb"] == 4


def test_block1_design_accepts_c6_runtime_topology_ids():
    topology_id = _first_block1_runtime_topology_id(
        seq_len=64,
        hidden_size=768,
        num_heads=12,
        predicate=lambda topology: int(topology["num_aie_columns"]) == 6,
    )

    config = qkv_proj_design(
        seq_len=64,
        hidden_size=768,
        num_heads=12,
        topology_id=topology_id,
    )

    assert config["topology_id"] == topology_id
    assert config["num_aie_columns"] == 6


def test_theoretical_block1_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in _block1_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            num_heads=12,
        )
    }
    assert "m64_k64_n16_c8_ps1_pe1" in topology_ids
    assert "m32_k64_n16_c8_ps1_pe1" in topology_ids
    assert "m64_k64_n24_c8_ps1_pe1" in topology_ids
    assert "m64_k64_n16_c4_ps1_pe1" in topology_ids
    assert "m64_k64_n16_c8_ps2_pe1" in topology_ids


def test_theoretical_block1_topologies_exclude_nonrunnable_l1_overflows():
    topology_ids = {
        str(topology["topology_id"])
        for topology in _block1_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            num_heads=12,
        )
    }

    assert "m32_k384_n48_c8_ps1_pe1" not in topology_ids
    assert "m32_k256_n64_c8_ps1_pe1" not in topology_ids


def test_theoretical_block1_topologies_are_unique_and_contract_valid():
    topologies = _block1_theoretical_topologies(
        seq_len=512,
        hidden_size=768,
        num_heads=12,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_emb = int(topology["parallel_emb"])
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        num_aie_columns = int(topology["num_aie_columns"])

        assert parallel_seq in (1, 2, 4, 6, 8)
        assert 512 % (parallel_seq * tile_m) == 0
        assert 768 % parallel_emb == 0
        assert num_aie_columns % parallel_emb == 0
        assert _block1_compute_tile_working_set_fits(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
        )


def test_practical_block1_topologies_are_subset_of_theoretical_surface():
    practical = _block1_practical_topologies(
        seq_len=512,
        hidden_size=768,
        num_heads=12,
    )
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in _block1_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            num_heads=12,
        )
    }
    practical_ids = [str(topology["topology_id"]) for topology in practical]

    assert practical
    assert len(practical) <= 20
    assert len(practical) < len(theoretical_ids)
    assert set(practical_ids) <= theoretical_ids


def test_practical_block1_topologies_support_generalized_workloads():
    seq_aware = _block1_practical_topologies(
        seq_len=512,
        hidden_size=1536,
        num_heads=24,
    )
    seq_independent = qkv_proj_topologies(hidden_size=1536, num_heads=24)

    assert seq_aware
    assert seq_independent
    assert len(seq_aware) <= 20
    assert len(seq_independent) <= 12


def test_practical_block1_topologies_are_ranked_and_pruned():
    practical = _block1_practical_topologies(
        seq_len=512,
        hidden_size=768,
        num_heads=12,
    )

    assert practical == sorted(practical, key=_block1_practical_sort_key, reverse=True)
    for topology in practical:
        assert int(topology["tile_m"]) % 8 == 0
        assert int(topology["tile_k"]) % 8 == 0
        assert int(topology["tile_n"]) % 8 == 0
        assert int(topology["num_aie_columns"]) in (6, 8)
        assert topology["topology_family"] == "shared_runtime_qkv_proj_practical"

    practical_ids = {str(topology["topology_id"]) for topology in practical}
    assert "m32_k384_n48_c8_ps1_pe1" not in practical_ids


def test_block1_compute_tile_shape_score_prefers_fuller_square_tiles():
    assert _block1_compute_tile_shape_score(tile_m=64, tile_k=64, tile_n=64) > (
        _block1_compute_tile_shape_score(tile_m=32, tile_k=256, tile_n=24)
    )
    assert _block1_compute_tile_shape_score(tile_m=64, tile_k=96, tile_n=48) > (
        _block1_compute_tile_shape_score(tile_m=32, tile_k=256, tile_n=24)
    )
