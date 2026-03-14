#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.common.test_utils import nearly_equal
from operators.common.utils import torch_to_numpy
from operators.encoder_pipeline_ddr.debug_modes import (
    PROFILE_DEBUG_CHOICES,
    VERIFY_DEBUG_CHOICES,
    resolve_pipeline_debug_modes,
    stage_name,
)
from operators.encoder_pipeline_ddr.op import AIEEncoderPipeline
from operators.encoder_pipeline_ddr.reference import generate_golden_reference

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
DEBUG_CASE = (64, 64, 12, 3072, 32, 64, 96, 4, 4, 8, 4)


@lru_cache(maxsize=32)
def _cached_reference(debug_mode):
    seq_len, d, heads, intermediate_size, *_ = DEBUG_CASE
    return generate_golden_reference(
        seq_len=seq_len,
        d=d,
        heads=heads,
        intermediate_size=intermediate_size,
        seed=42,
        debug=debug_mode,
    )


def _build_operator(debug_mode, ref, aie_context):
    (
        seq_len,
        d,
        heads,
        intermediate_size,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        parallel_ffn,
        proj_acc_depth,
        o_proj_acc_group_size,
    ) = DEBUG_CASE
    return AIEEncoderPipeline(
        seq_len=seq_len,
        d=d,
        num_heads=heads,
        seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        ffn_intermediate_size=intermediate_size,
        nB_tiles_distributed=parallel_ffn,
        debug=debug_mode,
        ln1_weight=ref["ln1_weight"].clone(),
        ln2_weight=ref["ln2_weight"].clone(),
        context=aie_context,
    )


def _run_forward(op, ref):
    op.context.compile_all()
    op.context.prepare_runtime()
    op.write_buffer("O", np.zeros(op.buffers["O"], dtype=np.uint8))
    op.write_buffer("QKV", torch_to_numpy(ref["QKV"]))
    op.write_buffer("OR", torch_to_numpy(ref["OR"]))
    op.write_buffer("W_O", torch_to_numpy(ref["W_O"]))
    op.write_buffer("B_Up", torch_to_numpy(ref["B_Up"]))
    op.write_buffer("B_Down", torch_to_numpy(ref["B_Down"]))
    op.run_runlist()
    output_shape = op._debug_output_shape()
    return op.read_buffer("O", shape=output_shape)


def _assert_matches(actual, expected):
    actual_np = np.asarray(actual).reshape((-1,))
    expected_np = torch_to_numpy(expected).reshape((-1,))
    assert actual_np.shape == expected_np.shape
    errors = []
    for idx, (a, e) in enumerate(zip(actual_np, expected_np, strict=True)):
        if not nearly_equal(float(a), float(e), rel_tol=REL_TOL, abs_tol=ABS_TOL):
            errors.append(idx)
            if len(errors) >= 32:
                break
    max_errors = int(expected_np.size * ERROR_THRESHOLD)
    print(f"({len(errors)} errors out of {max_errors} max allowable)")
    assert len(errors) <= max_errors


PROFILE_DEBUG_CASES = [
    pytest.param(debug_mode, id=stage_name(debug_mode))
    for debug_mode in PROFILE_DEBUG_CHOICES
]
VERIFY_DEBUG_CASES = [
    pytest.param(
        debug_mode, id=stage_name(resolve_pipeline_debug_modes(debug_mode).verify_stage)
    )
    for debug_mode in VERIFY_DEBUG_CHOICES
]


@pytest.mark.parametrize("debug_mode", PROFILE_DEBUG_CASES)
def test_stage_profile_modes(debug_mode, aie_context):
    ref = _cached_reference(debug_mode)
    op = _build_operator(debug_mode, ref, aie_context)
    result = _run_forward(op, ref)
    expected_shape = (DEBUG_CASE[0], DEBUG_CASE[1] * DEBUG_CASE[2])
    assert tuple(result.shape) == expected_shape
    assert np.isfinite(np.asarray(result).astype(np.float32)).all()


@pytest.mark.parametrize("debug_mode", VERIFY_DEBUG_CASES)
def test_stage_verify_modes(debug_mode, aie_context):
    ref = _cached_reference(debug_mode)
    op = _build_operator(debug_mode, ref, aie_context)
    result = _run_forward(op, ref)
    _assert_matches(result, ref["O"])
