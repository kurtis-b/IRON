# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch

from operators.encoder_pipeline_ddr.debug_modes import (
    STAGE_DOWN_PROJ,
    STAGE_LN2,
    STAGE_O_PROJ,
    STAGE_PV,
    STAGE_QK,
    STAGE_SOFTMAX,
    resolve_pipeline_debug_modes,
    stage_compute_enabled,
)


def _layer_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )


def generate_golden_reference(
    heads: int = 3,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seed: int = 42,
    debug: int = -1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    intermediate_size = 4 * embed_sz if intermediate_size is None else intermediate_size
    dtype = torch.bfloat16
    val_range = 4

    debug_cfg = resolve_pipeline_debug_modes(debug)
    profile_stage = debug_cfg.profile_stage
    verify_stage = debug_cfg.verify_stage

    q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    w_o = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range
    r1 = torch.rand(seq_len, embed_sz, dtype=dtype) * val_range
    b_up = torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
    b_down = torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range
    ln1_w = torch.rand(embed_sz, dtype=dtype)
    ln2_w = torch.rand(embed_sz, dtype=dtype)

    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores / math.sqrt(d), dim=-1)
    attn = torch.matmul(probs, v)
    pv = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    o_proj = torch.matmul(pv, w_o)
    ln1 = _layer_norm(o_proj, ln1_w) + r1
    up = torch.matmul(ln1, b_up)
    up_gelu = torch.nn.functional.gelu(up)
    down = torch.matmul(up_gelu, b_down)
    y = _layer_norm(down, ln2_w) + ln1

    if verify_stage == STAGE_O_PROJ:
        o = o_proj
    elif verify_stage == STAGE_DOWN_PROJ:
        o = down
    elif verify_stage == STAGE_LN2:
        o = y
    elif profile_stage is None:
        o = y
    else:
        # Profile modes do not have a meaningful stage-native host output contract.
        # Keep the golden output shape stable for any debug utility that still asks
        # for a reference in that mode.
        o = torch.zeros(seq_len, embed_sz, dtype=dtype)

    q_2d = q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    k_2d = k.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    v_2d = v.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    qkv = torch.cat((q_2d, k_2d, v_2d), dim=0)
    or_buf = torch.cat((torch.zeros_like(r1), r1), dim=0)
    ar = torch.cat((down, ln1), dim=0)

    return {
        "Q": q_2d,
        "K": k_2d,
        "V": v_2d,
        "QKV": qkv,
        "W_O": w_o,
        "R1": r1,
        "OR": or_buf,
        "ln1_weight": ln1_w,
        "mha_to_an_out": ln1,
        "AR": ar,
        "B_Up": b_up,
        "B_Down": b_down,
        "ln2_weight": ln2_w,
        "O": o,
        "stage_compute_enabled": {
            stage_id: stage_compute_enabled(
                stage_id=stage_id,
                profile_stage=profile_stage,
                verify_stage=verify_stage,
            )
            for stage_id in (
                STAGE_QK,
                STAGE_SOFTMAX,
                STAGE_PV,
                STAGE_O_PROJ,
                STAGE_DOWN_PROJ,
                STAGE_LN2,
            )
        },
    }
