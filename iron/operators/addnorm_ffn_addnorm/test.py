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


def _count_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rel_tol: float,
    abs_tol: float,
) -> int:
    return int((~torch.isclose(actual, expected, rtol=rel_tol, atol=abs_tol)).sum())


_ACTIVE_BRINGUP_CASES = [
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        -1,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        0,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_stageonly0",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        1,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_stageonly1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        2,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_stageonly2",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        3,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_stageonly3",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g0",
        None,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g0_full",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        0,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        1,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        2,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        3,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        None,
        id="baseline_32x96x96_m32_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        64,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleM_64x96x96_m32_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        128,
        96,
        96,
        "m32_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleM_128x96x96_m32_k96_n96_ps1_pi1_d1_g1_full",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        -1,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_stageonly-1",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        0,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_stageonly0",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        1,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_stageonly1",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        2,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_stageonly2",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        3,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_stageonly3",
    ),
    pytest.param(
        32,
        192,
        96,
        "m32_k96_n96_ps1_pi1_d2_g1",
        None,
        id="scaleK_32x192x96_m32_k96_n96_ps1_pi1_d2_g1_full",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        -1,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_stageonly-1",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        0,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_stageonly0",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        1,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_stageonly1",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        2,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_stageonly2",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        3,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_stageonly3",
    ),
    pytest.param(
        32,
        768,
        96,
        "m32_k96_n96_ps1_pi1_d8_g1",
        None,
        id="scaleK_32x768x96_m32_k96_n96_ps1_pi1_d8_g1_full",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        -1,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_stageonly-1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        0,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_stageonly0",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        1,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_stageonly1",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        2,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_stageonly2",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        3,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_stageonly3",
    ),
    pytest.param(
        32,
        96,
        192,
        "m32_k96_n96_ps1_pi1_d1_g1",
        None,
        id="scaleN_32x96x192_m32_k96_n96_ps1_pi1_d1_g1_full",
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
    stage_preadd, stage_ln1_out = operator.read_staged_buffers()

    assert operator.topology_id == topology_id
    assert operator.topology_family == "pipelined_addnorm_ffn_addnorm"
    assert output.shape == golden["output"].shape

    if stage_only in (0, None):
        preadd_errors = _count_errors(
            stage_preadd,
            golden["preadd"],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        ln1_errors = _count_errors(
            stage_ln1_out,
            golden["ln1_out"],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        max_acceptable_errors = int(seq_len * hidden_size * error_threshold)
        assert preadd_errors <= max_acceptable_errors
        assert ln1_errors <= max_acceptable_errors

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
