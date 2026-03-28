#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import generate_golden_reference


def generate_test_params():
    params = [
        (64, 12, 64, "m64_k64_n16_ps1_ph1_pd1"),
        (512, 12, 64, "m64_k64_n16_ps1_ph1_pd1"),
        (512, 16, 64, "m64_k64_n16_ps1_ph1_pd1"),
    ]
    names = [
        f"block1_{seq_len}_{num_heads}_{head_dim}_{topology_id}"
        for seq_len, num_heads, head_dim, topology_id in params
    ]
    return params, names


regular_params, regular_names = generate_test_params()


@pytest.mark.parametrize(
    "seq_len,num_heads,head_dim,topology_id",
    [
        pytest.param(*params, id=name)
        for params, name in zip(regular_params, regular_names)
    ],
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
