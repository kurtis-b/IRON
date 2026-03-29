#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.qkv_proj.design import (
    qkv_proj_theoretical_topologies,
    qkv_proj_topologies,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import generate_golden_reference


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

    assert operator.topology_id == topology_id
    assert operator.topology_family == "shared_runtime_qkv_proj"
    assert q.shape == golden_ref["q"].shape
    assert k.shape == golden_ref["k"].shape
    assert v.shape == golden_ref["v"].shape
    assert torch.allclose(q, golden_ref["q"], rtol=4.0e-2, atol=1.5e-1)
    assert torch.allclose(k, golden_ref["k"], rtol=4.0e-2, atol=1.5e-1)
    assert torch.allclose(v, golden_ref["v"], rtol=4.0e-2, atol=1.5e-1)


def test_theoretical_block1_topologies_cover_supported_surface():
    for seq_len, hidden_size, num_heads in (
        (64, 768, 12),
        (512, 768, 12),
        (512, 1024, 16),
    ):
        supported_ids = {
            str(topology["topology_id"])
            for topology in qkv_proj_topologies(
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        theoretical_ids = {
            str(topology["topology_id"]).replace("_c8", "")
            for topology in qkv_proj_theoretical_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        }
        assert supported_ids <= theoretical_ids


def test_theoretical_block1_topologies_include_nondefault_valid_variants():
    topology_ids = {
        str(topology["topology_id"])
        for topology in qkv_proj_theoretical_topologies(
            seq_len=512,
            hidden_size=768,
            num_heads=12,
        )
    }
    assert "m64_k64_n16_c8_ps1_ph1_pd1" in topology_ids
    assert "m32_k64_n16_c8_ps1_ph1_pd1" in topology_ids
    assert "m64_k64_n24_c8_ps1_ph1_pd1" in topology_ids
    assert "m64_k64_n16_c4_ps1_ph1_pd1" in topology_ids
    assert "m64_k64_n16_c8_ps2_ph1_pd1" in topology_ids


def test_theoretical_block1_topologies_are_unique_and_contract_valid():
    topologies = qkv_proj_theoretical_topologies(
        seq_len=512,
        hidden_size=768,
        num_heads=12,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_heads = int(topology["parallel_heads"])
        parallel_head_dim = int(topology["parallel_head_dim"])
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        num_aie_columns = int(topology["num_aie_columns"])

        assert parallel_seq in (1, 2, 4, 6, 8)
        assert 512 % (parallel_seq * tile_m) == 0
        assert 12 % parallel_heads == 0
        assert 64 % parallel_head_dim == 0
        assert parallel_seq * parallel_heads * parallel_head_dim <= 8
        assert 2304 % (tile_n * num_aie_columns) == 0
