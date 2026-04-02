#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.addnorm_ffn_addnorm.op import AIEAddNormFFNAddNorm
from iron.operators.addnorm_ffn_addnorm.reference import generate_golden_reference
from iron.operators.addnorm_ffn_addnorm.topology import (
    addnorm_ffn_addnorm_design,
    addnorm_ffn_addnorm_practical_topologies,
    addnorm_ffn_addnorm_theoretical_topologies,
    addnorm_ffn_addnorm_topologies,
)


def _count_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rel_tol: float,
    abs_tol: float,
) -> int:
    return int((~torch.isclose(actual, expected, rtol=rel_tol, atol=abs_tol)).sum())


def _configure_operator(
    operator: AIEAddNormFFNAddNorm, golden: dict[str, torch.Tensor]
):
    operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
    operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
    operator.ln1_weight = golden["ln1_weight"].contiguous()
    operator.ln2_weight = golden["ln2_weight"].contiguous()


def _run_block3_case(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    topology_id: str,
    stage_only: int | None,
    aie_context,
) -> tuple[AIEAddNormFFNAddNorm, torch.Tensor, dict[str, torch.Tensor]]:
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
        stage_only=stage_only,
        context=aie_context,
    )
    _configure_operator(operator, golden)

    aie_context.compile_all()
    aie_context.prepare_runtime()
    output = operator.forward(golden["hidden_states"], golden["residual"])
    return operator, output, golden


def _first_block3_runtime_topology_id(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    predicate=lambda topology: True,
) -> str:
    for topology in addnorm_ffn_addnorm_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ):
        if predicate(topology):
            return str(topology["topology_id"])
    raise AssertionError("expected matching Block 3 runtime topology")


def _retained_runtime_block3_cases() -> list[pytest.ParamSpec]:
    cases: list[pytest.ParamSpec] = []
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
    ):
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            topology_id = str(topology["topology_id"])
            cases.append(
                pytest.param(
                    seq_len,
                    hidden_size,
                    intermediate_size,
                    topology_id,
                    id=f"block3_{seq_len}x{hidden_size}x{intermediate_size}_{topology_id}",
                )
            )
    return cases


_SINGLE_CASE = [pytest.param(None, id="default")]


def _topology_ids(topologies: list[dict[str, int | str]]) -> list[str]:
    return [str(topology["topology_id"]) for topology in topologies]


_BLOCK3_TOPOLOGY_CONTRACT_CASES = [
    pytest.param(16, 96, 96, id="contract_16x96x96"),
    pytest.param(32, 96, 96, id="contract_32x96x96"),
    pytest.param(64, 96, 192, id="contract_64x96x192"),
    pytest.param(64, 96, 384, id="contract_64x96x384"),
    pytest.param(64, 768, 96, id="contract_64x768x96"),
    pytest.param(64, 768, 3072, id="contract_64x768x3072"),
    pytest.param(64, 96, 576, id="contract_64x96x576"),
    pytest.param(128, 96, 96, id="contract_128x96x96"),
    pytest.param(512, 768, 3072, id="contract_512x768x3072"),
]


_BLOCK3_RETAINED_RUNTIME_EXPECTED_IDS = {
    (96, 96): [
        "m16_k96_n96_ps1_pi1_d1_g0",
        "m16_k96_n96_ps1_pi1_d1_g1",
        "m16_k96_n96_ps2_pi1_d1_g1",
        "m16_k96_n96_ps4_pi1_d1_g1",
    ],
    (96, 192): [
        "m16_k96_n96_ps1_pi1_d1_g1",
        "m16_k96_n96_ps1_pi2_d1_g1",
    ],
    (96, 384): [
        "m16_k96_n96_ps1_pi1_d1_g1",
        "m16_k96_n96_ps1_pi4_d1_g1",
    ],
    (96, 768): [
        "m16_k96_n96_ps1_pi1_d1_g1",
    ],
    (192, 96): [
        "m16_k96_n96_ps1_pi1_d2_g1",
    ],
    (384, 96): [
        "m16_k96_n96_ps1_pi1_d4_g1",
    ],
    (768, 96): [
        "m16_k96_n96_ps1_pi1_d8_g1",
    ],
    (768, 3072): [
        "m16_k96_n96_ps4_pi2_d8_g1",
    ],
}


_BLOCK3_GENERALIZED_RUNTIME_NUMERIC_CASES = [
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        lambda topology: int(topology["parallel_seq"]) == 2
        and int(topology["parallel_int_dim"]) == 1
        and int(topology["gelu_stage"]) == 1,
        id="block3_generalized_64x96x96_m16_k96_n96_ps2_pi1_d1_g1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        lambda topology: int(topology["parallel_seq"]) == 1
        and int(topology["parallel_int_dim"]) == 2
        and int(topology["gelu_stage"]) == 1,
        id="block3_generalized_32x96x192_m16_k96_n96_ps1_pi2_d1_g1",
    ),
    pytest.param(
        512,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        lambda topology: int(topology["parallel_seq"]) == 1
        and int(topology["parallel_int_dim"]) == 1
        and int(topology["down_proj_depth"]) == 8,
        id="block3_retained_512x768x96_m16_k96_n96_ps1_pi1_d8_g1",
    ),
]


_ACTIVE_BRINGUP_CASES = [
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        -1,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_stageonly-1",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        0,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_stageonly0",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        1,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_stageonly1",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        2,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_stageonly2",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        3,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_stageonly3",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g0",
        None,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g0_full",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        16,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="baseline_16x96x96_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleM_32x96x96_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleM_64x96x96_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        128,
        96,
        96,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleM_128x96x96_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        -1,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_stageonly-1",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        0,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_stageonly0",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        1,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_stageonly1",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        2,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_stageonly2",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        3,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_stageonly3",
    ),
    pytest.param(
        32,
        192,
        96,
        "m16_k96_n96_ps1_pi1_d2_g1",
        None,
        id="scaleK_32x192x96_m16_k96_n96_ps1_pi1_d2_g1_full",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        -1,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_stageonly-1",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        0,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_stageonly0",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        1,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_stageonly1",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        2,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_stageonly2",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        3,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_stageonly3",
    ),
    pytest.param(
        32,
        384,
        96,
        "m16_k96_n96_ps1_pi1_d4_g1",
        None,
        id="scaleK_32x384x96_m16_k96_n96_ps1_pi1_d4_g1_full",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        -1,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_stageonly-1",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        0,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_stageonly0",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        1,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_stageonly1",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        2,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_stageonly2",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        3,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_stageonly3",
    ),
    pytest.param(
        32,
        768,
        96,
        "m16_k96_n96_ps1_pi1_d8_g1",
        None,
        id="scaleK_32x768x96_m16_k96_n96_ps1_pi1_d8_g1_full",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleN_32x96x192_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleN_32x96x384_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        768,
        "m16_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleN_32x96x768_m16_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        -1,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        0,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        1,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        2,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        3,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps2_pi1_d1_g1",
        None,
        id="scalePs_64x96x96_m16_k96_n96_ps2_pi1_d1_g1_full",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        -1,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        0,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        1,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        2,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        3,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        64,
        96,
        96,
        "m16_k96_n96_ps4_pi1_d1_g1",
        None,
        id="scalePs_64x96x96_m16_k96_n96_ps4_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        -1,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        0,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        1,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        2,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        3,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        192,
        "m16_k96_n96_ps1_pi2_d1_g1",
        None,
        id="scalePi_32x96x192_m16_k96_n96_ps1_pi2_d1_g1_full",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        -1,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        0,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        1,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        2,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        3,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        384,
        "m16_k96_n96_ps1_pi4_d1_g1",
        None,
        id="scalePi_32x96x384_m16_k96_n96_ps1_pi4_d1_g1_full",
    ),
]


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id,stage_only",
    _ACTIVE_BRINGUP_CASES,
)
def test_addnorm_ffn_addnorm_bringup(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    stage_only,
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
        stage_only=stage_only,
        context=aie_context,
    )
    operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
    operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
    operator.ln1_weight = golden["ln1_weight"].contiguous()
    operator.ln2_weight = golden["ln2_weight"].contiguous()

    aie_context.compile_all()
    aie_context.prepare_runtime()

    output = operator.forward(golden["hidden_states"], golden["residual"])

    assert operator.topology_id == topology_id
    assert operator.topology_family == "pipelined_addnorm_ffn_addnorm"
    assert output.shape == golden["output"].shape

    if stage_only is not None:
        return

    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    max_acceptable_errors = int(seq_len * hidden_size * error_threshold)
    assert output_errors <= max_acceptable_errors


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id",
    _retained_runtime_block3_cases(),
)
def test_block3_runtime_topologies_run_numerically(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    aie_context,
):
    operator, output, golden = _run_block3_case(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=topology_id,
        stage_only=None,
        aie_context=aie_context,
    )

    assert operator.topology_id == topology_id
    assert operator.topology_family == "pipelined_addnorm_ffn_addnorm"
    assert output.shape == golden["output"].shape

    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
    )
    max_acceptable_errors = int(seq_len * hidden_size * 0.005)
    assert output_errors <= max_acceptable_errors


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id",
    [
        pytest.param(
            64,
            96,
            768,
            "m16_k96_n96_ps1_pi1_d1_g1",
            id="block3_generalized_64x96x768_m16_k96_n96_ps1_pi1_d1_g1",
        )
    ],
)
def test_generalized_block3_runtime_topology_runs_numerically(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    aie_context,
):
    resolved_topology_id = _first_block3_runtime_topology_id(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        predicate=lambda topology: int(topology["parallel_seq"]) == 1
        and int(topology["parallel_int_dim"]) == 1,
    )
    assert resolved_topology_id == topology_id

    operator, output, golden = _run_block3_case(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=resolved_topology_id,
        stage_only=None,
        aie_context=aie_context,
    )

    assert operator.topology_id == resolved_topology_id
    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
    )
    max_acceptable_errors = int(seq_len * hidden_size * 0.005)
    assert output_errors <= max_acceptable_errors


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block3_design_accepts_canonical_runtime_topology_ids(_case):
    config = addnorm_ffn_addnorm_design(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        topology_id="m16_k96_n96_ps4_pi2_d8_g1",
    )
    assert config["topology_id"] == "m16_k96_n96_ps4_pi2_d8_g1"
    assert config["topology_family"] == "pipelined_addnorm_ffn_addnorm"


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block3_design_accepts_canonical_parallel_runtime_ids(_case):
    config = addnorm_ffn_addnorm_design(
        seq_len=64,
        hidden_size=96,
        intermediate_size=96,
        topology_id="m16_k96_n96_ps4_pi1_d1_g1",
    )
    assert config["topology_id"] == "m16_k96_n96_ps4_pi1_d1_g1"
    assert config["topology_family"] == "pipelined_addnorm_ffn_addnorm"


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_block3_design_rejects_unknown_runtime_topology_ids(_case):
    with pytest.raises(ValueError):
        addnorm_ffn_addnorm_design(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            topology_id="m32_k96_n64_ps2_pi6_d8_g1",
        )


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_theoretical_block3_topologies_cover_supported_surface(_case):
    for seq_len, hidden_size, intermediate_size in (
        (64, 96, 96),
        (64, 96, 384),
        (64, 768, 3072),
    ):
        supported_ids = {
            str(topology["topology_id"])
            for topology in addnorm_ffn_addnorm_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        theoretical_ids = {
            str(topology["topology_id"])
            for topology in addnorm_ffn_addnorm_theoretical_topologies(
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        }
        assert supported_ids <= theoretical_ids


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size",
    _BLOCK3_TOPOLOGY_CONTRACT_CASES,
)
def test_theoretical_block3_topologies_cover_supported_surface_across_matrix(
    seq_len,
    hidden_size,
    intermediate_size,
):
    supported_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    assert supported_ids <= theoretical_ids


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_practical_block3_topologies_are_subset_of_theoretical_surface(_case):
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert practical_ids <= theoretical_ids


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size",
    _BLOCK3_TOPOLOGY_CONTRACT_CASES,
)
def test_practical_block3_topologies_are_subset_of_theoretical_surface_across_matrix(
    seq_len,
    hidden_size,
    intermediate_size,
):
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    theoretical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    assert practical_ids <= theoretical_ids


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_runtime_block3_topologies_are_subset_of_practical_surface(_case):
    runtime_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert runtime_ids <= practical_ids


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size",
    _BLOCK3_TOPOLOGY_CONTRACT_CASES,
)
def test_runtime_block3_topologies_are_subset_of_practical_surface_across_matrix(
    seq_len,
    hidden_size,
    intermediate_size,
):
    runtime_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    assert runtime_ids <= practical_ids


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,expected_ids",
    [
        pytest.param(
            hidden_size,
            intermediate_size,
            expected_ids,
            id=f"retained_{hidden_size}x{intermediate_size}",
        )
        for (hidden_size, intermediate_size), expected_ids in (
            _BLOCK3_RETAINED_RUNTIME_EXPECTED_IDS.items()
        )
    ],
)
def test_retained_runtime_block3_topologies_match_promoted_surface_exactly(
    hidden_size,
    intermediate_size,
    expected_ids,
):
    runtime_ids = _topology_ids(
        addnorm_ffn_addnorm_topologies(
            seq_len=64,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    )
    assert runtime_ids == expected_ids


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,expected_ids",
    [
        pytest.param(
            hidden_size,
            intermediate_size,
            expected_ids,
            id=f"retained_{hidden_size}x{intermediate_size}",
        )
        for (hidden_size, intermediate_size), expected_ids in (
            _BLOCK3_RETAINED_RUNTIME_EXPECTED_IDS.items()
        )
    ],
)
def test_practical_block3_topologies_include_retained_runtime_surface(
    hidden_size,
    intermediate_size,
    expected_ids,
):
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=64,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    }
    assert set(expected_ids) <= practical_ids


@pytest.mark.parametrize(
    "seq_len,expected_ids",
    [
        pytest.param(
            16,
            ["m16_k96_n96_ps1_pi1_d1_g0", "m16_k96_n96_ps1_pi1_d1_g1"],
            id="seq16",
        ),
        pytest.param(
            32,
            [
                "m16_k96_n96_ps1_pi1_d1_g0",
                "m16_k96_n96_ps1_pi1_d1_g1",
                "m16_k96_n96_ps2_pi1_d1_g1",
            ],
            id="seq32",
        ),
        pytest.param(
            48,
            ["m16_k96_n96_ps1_pi1_d1_g0", "m16_k96_n96_ps1_pi1_d1_g1"],
            id="seq48",
        ),
        pytest.param(
            64,
            [
                "m16_k96_n96_ps1_pi1_d1_g0",
                "m16_k96_n96_ps1_pi1_d1_g1",
                "m16_k96_n96_ps2_pi1_d1_g1",
                "m16_k96_n96_ps4_pi1_d1_g1",
            ],
            id="seq64",
        ),
        pytest.param(
            128,
            [
                "m16_k96_n96_ps1_pi1_d1_g0",
                "m16_k96_n96_ps1_pi1_d1_g1",
                "m16_k96_n96_ps2_pi1_d1_g1",
                "m16_k96_n96_ps4_pi1_d1_g1",
            ],
            id="seq128",
        ),
    ],
)
def test_runtime_block3_topologies_change_with_seq_len_for_96x96(
    seq_len,
    expected_ids,
):
    runtime_ids = _topology_ids(
        addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=96,
            intermediate_size=96,
        )
    )
    assert runtime_ids == expected_ids


@pytest.mark.parametrize(
    "seq_len,expected_ids",
    [
        pytest.param(32, [], id="seq32"),
        pytest.param(64, ["m16_k96_n96_ps4_pi2_d8_g1"], id="seq64"),
        pytest.param(512, ["m16_k96_n96_ps4_pi2_d8_g1"], id="seq512"),
    ],
)
def test_runtime_block3_topologies_change_with_seq_len_for_768x3072(
    seq_len,
    expected_ids,
):
    runtime_ids = _topology_ids(
        addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=768,
            intermediate_size=3072,
        )
    )
    assert runtime_ids == expected_ids


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_retained_runtime_block3_topologies_do_not_fallback_when_promoted_ids_are_invalid(
    _case,
):
    runtime_ids = _topology_ids(
        addnorm_ffn_addnorm_topologies(
            seq_len=32,
            hidden_size=768,
            intermediate_size=3072,
        )
    )
    practical_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_practical_topologies(
            seq_len=32,
            hidden_size=768,
            intermediate_size=3072,
        )
    }

    assert runtime_ids == []
    assert practical_ids
    assert "m16_k96_n96_ps1_pi4_d8_g1" in practical_ids


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,topology_id,predicate",
    _BLOCK3_GENERALIZED_RUNTIME_NUMERIC_CASES,
)
def test_additional_block3_runtime_topologies_run_numerically(
    seq_len,
    hidden_size,
    intermediate_size,
    topology_id,
    predicate,
    aie_context,
):
    resolved_topology_id = _first_block3_runtime_topology_id(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        predicate=predicate,
    )
    assert resolved_topology_id == topology_id

    operator, output, golden = _run_block3_case(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topology_id=resolved_topology_id,
        stage_only=None,
        aie_context=aie_context,
    )

    assert operator.topology_id == resolved_topology_id
    output_errors = _count_errors(
        output,
        golden["output"],
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
    )
    max_acceptable_errors = int(seq_len * hidden_size * 0.005)
    assert output_errors <= max_acceptable_errors


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_runtime_block3_topologies_lower_parallel_axes_honestly(_case):
    runtime_topologies = addnorm_ffn_addnorm_topologies(
        seq_len=64,
        hidden_size=96,
        intermediate_size=576,
    )
    assert runtime_topologies
    assert any(int(topology["parallel_seq"]) > 1 for topology in runtime_topologies)
    assert any(int(topology["parallel_int_dim"]) > 1 for topology in runtime_topologies)
    for topology in runtime_topologies:
        assert int(topology["parallel_seq"]) in (1, 2, 4)
        assert int(topology["parallel_int_dim"]) in (1, 2, 4)
        assert int(topology["num_aie_columns"]) == 2 * max(
            int(topology["parallel_seq"]),
            int(topology["parallel_int_dim"]),
        )


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_runtime_block3_topologies_are_pruned_and_generalized(_case):
    retained_runtime = addnorm_ffn_addnorm_topologies(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
    )
    generalized_runtime = addnorm_ffn_addnorm_topologies(
        seq_len=64,
        hidden_size=96,
        intermediate_size=768,
    )

    assert [str(topology["topology_id"]) for topology in retained_runtime] == [
        "m16_k96_n96_ps4_pi2_d8_g1"
    ]
    assert generalized_runtime
    assert all(
        str(topology["topology_family"]) == "pipelined_addnorm_ffn_addnorm"
        for topology in generalized_runtime
    )


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_theoretical_block3_topologies_include_nondefault_valid_variants(_case):
    topology_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert "m16_k96_n96_ps4_pi2_d8_g1" in topology_ids
    assert "m16_k96_n96_ps4_pi1_d8_g1" in topology_ids
    assert "m16_k96_n96_ps2_pi4_d8_g1" in topology_ids
    assert "m16_k96_n96_ps1_pi4_d8_g1" in topology_ids


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_theoretical_block3_topologies_exclude_nonrunnable_core_overflows(_case):
    topology_ids = {
        str(topology["topology_id"])
        for topology in addnorm_ffn_addnorm_theoretical_topologies(
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
        )
    }
    assert "m16_k96_n96_ps4_pi4_d8_g1" not in topology_ids


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_theoretical_block3_topologies_are_unique_and_contract_valid(_case):
    topologies = addnorm_ffn_addnorm_theoretical_topologies(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_int_dim = int(topology["parallel_int_dim"])
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        down_proj_depth = int(topology["down_proj_depth"])
        num_aie_columns = int(topology["num_aie_columns"])

        assert (
            topology["topology_family"] == "pipelined_addnorm_ffn_addnorm_theoretical"
        )
        assert parallel_seq in (1, 2, 4)
        assert parallel_int_dim in (1, 2, 4)
        assert tile_m == 16
        assert tile_k == 96
        assert tile_n == 96
        assert down_proj_depth in (1, 2, 4, 8)
        assert 768 == tile_k * down_proj_depth
        assert 3072 % (tile_n * parallel_int_dim) == 0
        assert 64 % (tile_m * parallel_seq) == 0
        assert num_aie_columns == 2 * max(parallel_seq, parallel_int_dim)


@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size",
    _BLOCK3_TOPOLOGY_CONTRACT_CASES,
)
def test_theoretical_block3_topologies_are_unique_and_contract_valid_across_matrix(
    seq_len,
    hidden_size,
    intermediate_size,
):
    topologies = addnorm_ffn_addnorm_theoretical_topologies(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    assert topologies

    topology_ids = [str(topology["topology_id"]) for topology in topologies]
    assert len(topology_ids) == len(set(topology_ids))

    for topology in topologies:
        parallel_seq = int(topology["parallel_seq"])
        parallel_int_dim = int(topology["parallel_int_dim"])
        tile_m = int(topology["tile_m"])
        tile_k = int(topology["tile_k"])
        tile_n = int(topology["tile_n"])
        down_proj_depth = int(topology["down_proj_depth"])
        num_aie_columns = int(topology["num_aie_columns"])

        assert (
            topology["topology_family"] == "pipelined_addnorm_ffn_addnorm_theoretical"
        )
        assert parallel_seq in (1, 2, 4)
        assert parallel_int_dim in (1, 2, 4)
        assert tile_m == 16
        assert tile_k == 96
        assert tile_n == 96
        assert down_proj_depth in (1, 2, 4, 8)
        assert hidden_size == tile_k * down_proj_depth
        assert intermediate_size % (tile_n * parallel_int_dim) == 0
        assert seq_len % (tile_m * parallel_seq) == 0
        assert num_aie_columns == 2 * max(parallel_seq, parallel_int_dim)


@pytest.mark.parametrize("_case", _SINGLE_CASE)
def test_practical_block3_topologies_are_ranked_and_pruned(_case):
    practical = addnorm_ffn_addnorm_practical_topologies(
        seq_len=64,
        hidden_size=96,
        intermediate_size=768,
        max_candidates=2,
    )
    assert len(practical) == 2
    assert str(practical[0]["topology_id"]) == "m16_k96_n96_ps4_pi2_d1_g1"
    assert practical[0]["topology_family"] == "pipelined_addnorm_ffn_addnorm_practical"
