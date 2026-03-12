# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch

from operators.encoder_pipeline.debug_modes import (
    ADDNORM_DEBUG_INPUT,
    ADDNORM_DEBUG_RESIDUAL,
    DEBUG_ADDNORM1_ONLY,
    DEBUG_ADDNORM1_POST_ONLY,
    DEBUG_ADDNORM1_STATS_ONLY,
    DEBUG_FFN_ADDNORM_ONLY,
    DEBUG_FFN_DOWN_ONLY,
    DEBUG_FFN_UP_ONLY,
    DEBUG_MHA_ONLY,
    DEBUG_MHA_INPUT_PATH,
    DEBUG_RESIDUAL_PATH,
    DEBUG_SELF_ATTN,
    resolve_pipeline_debug_modes,
)


def _apply_addnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, mode: int
):
    if mode == ADDNORM_DEBUG_INPUT:
        return x.clone()
    if mode == ADDNORM_DEBUG_RESIDUAL:
        return residual.clone()
    y = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )
    return y + residual


def _apply_addnorm_stats_only(x: torch.Tensor, weight: torch.Tensor):
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )


def _apply_addnorm_post_only(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
):
    return x * weight + residual


def generate_golden_reference(
    heads: int = 3,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seed: int = 42,
    debug: int = -1,
):
    """Golden reference for encoder pipeline (mha_to_an + ffn_addnorm)."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    intermediate_size = 4 * embed_sz if intermediate_size is None else intermediate_size
    dtype = torch.bfloat16
    val_range = 4

    mha_debug, ffn_stage_only, stage1_mode, stage2_mode = resolve_pipeline_debug_modes(
        debug
    )

    # Stage-1 inputs/weights follow mha_to_an reference behavior.
    if debug == DEBUG_SELF_ATTN:
        q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        r1 = torch.zeros(seq_len, embed_sz, dtype=dtype)
        ln1_w = torch.ones(embed_sz, dtype=dtype)
    elif debug == DEBUG_RESIDUAL_PATH:
        q = torch.zeros((heads, seq_len, d), dtype=dtype)
        k = torch.zeros((heads, seq_len, d), dtype=dtype)
        v = torch.zeros((heads, seq_len, d), dtype=dtype)
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        r1 = torch.arange(seq_len * embed_sz, dtype=dtype).reshape(seq_len, embed_sz)
        ln1_w = torch.ones(embed_sz, dtype=dtype)
    elif debug in (
        DEBUG_MHA_INPUT_PATH,
        DEBUG_FFN_UP_ONLY,
        DEBUG_FFN_DOWN_ONLY,
        DEBUG_FFN_ADDNORM_ONLY,
        DEBUG_MHA_ONLY,
        DEBUG_ADDNORM1_ONLY,
        DEBUG_ADDNORM1_STATS_ONLY,
        DEBUG_ADDNORM1_POST_ONLY,
    ):
        base = torch.eye(max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=dtype)
        q2d = base[:seq_len, :embed_sz]
        k2d = base[:seq_len, :embed_sz]
        v2d = torch.arange(seq_len * embed_sz, dtype=dtype).reshape(seq_len, embed_sz)
        q = q2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        k = k2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        v = v2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        r1 = torch.zeros(seq_len, embed_sz, dtype=dtype)
        ln1_w = torch.ones(embed_sz, dtype=dtype)
    else:
        q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        w_o = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range
        r1 = torch.rand(seq_len, embed_sz, dtype=dtype) * val_range
        ln1_w = torch.rand(embed_sz, dtype=dtype)

    # Stage-2 weights follow ffn_addnorm full/debug generation style.
    if debug < 0:
        b_up = torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
        b_down = torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range
        ln2_w = torch.rand(embed_sz, dtype=dtype)
    else:
        b_up = torch.eye(embed_sz, intermediate_size, dtype=dtype)
        b_down = torch.eye(intermediate_size, embed_sz, dtype=dtype)
        ln2_w = torch.ones(embed_sz, dtype=dtype)

    # Stage 1: MHA -> out-proj -> AddNorm1.
    if mha_debug == -1:
        # Kernel debug path bypasses softmax and uses linear QK/PV flow.
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.matmul(attn_scores, v)
    else:
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=(1.0 / math.sqrt(d)),
        )

    attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    o_proj = torch.matmul(attn_2d, w_o)
    if debug == DEBUG_ADDNORM1_STATS_ONLY:
        h1 = _apply_addnorm_stats_only(o_proj, ln1_w)
    elif debug == DEBUG_ADDNORM1_POST_ONLY:
        h1 = _apply_addnorm_post_only(o_proj, r1, ln1_w)
    else:
        h1 = _apply_addnorm(o_proj, r1, ln1_w, stage1_mode)

    # Stage 2: FFN -> AddNorm2.
    run_up = ffn_stage_only in (None, 0)
    run_down = ffn_stage_only in (None, 1)
    run_addnorm2 = ffn_stage_only in (None, 2)

    if run_up:
        up = torch.matmul(h1, b_up)
        gelu = torch.nn.functional.gelu(up)
    else:
        gelu = torch.zeros(seq_len, intermediate_size, dtype=dtype)

    if run_down:
        down = torch.matmul(gelu, b_down)
    else:
        down = torch.zeros(seq_len, embed_sz, dtype=dtype)

    if run_addnorm2:
        y = _apply_addnorm(down, h1, ln2_w, stage2_mode)
    else:
        # Match encoder design stage-only behavior: bypass LN2 to residual.
        y = h1.clone()

    q_2d = q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    k_2d = k.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    v_2d = v.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    qkv = torch.cat((q_2d, k_2d, v_2d), dim=0)
    or_buf = torch.cat((torch.zeros_like(r1), r1), dim=0)
    ar = torch.cat((down, h1), dim=0)

    return {
        "Q": q_2d,
        "K": k_2d,
        "V": v_2d,
        "QKV": qkv,
        "W_O": w_o,
        "R1": r1,
        "OR": or_buf,
        "ln1_weight": ln1_w,
        "mha_to_an_out": h1,
        "AR": ar,
        "B_Up": b_up,
        "B_Down": b_down,
        "ln2_weight": ln2_w,
        "O": y,
    }
