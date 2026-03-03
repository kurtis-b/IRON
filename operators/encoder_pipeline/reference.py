# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch
from operators.encoder_pipeline.constants import (
    ADDNORM_DEBUG_DISABLED,
    ADDNORM_DEBUG_INPUT,
    ADDNORM_DEBUG_RESIDUAL,
    DEBUG_FULL,
    DEBUG_SELF_ATTN,
    DEBUG_O_PROJ,
    resolve_addnorm_modes,
    resolve_ffn_stage_only,
    resolve_mha_debug_mode,
)


def _apply_addnorm_stage(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    ln_weight: torch.Tensor,
    addnorm_mode: int,
):
    if addnorm_mode == ADDNORM_DEBUG_INPUT:
        return input_tensor.clone()
    if addnorm_mode == ADDNORM_DEBUG_RESIDUAL:
        return residual_tensor.clone()
    return (
        torch.nn.functional.layer_norm(
            input_tensor,
            normalized_shape=(input_tensor.shape[-1],),
            weight=ln_weight,
            bias=None,
        )
        + residual_tensor
    )


def _generate_mha_inputs(
    heads: int,
    seq_len: int,
    d: int,
    embed_sz: int,
    val_range: int,
    mha_debug_mode: int,
):
    dtype = torch.bfloat16

    if mha_debug_mode == DEBUG_FULL:
        # Keep full-mode distribution aligned with encoder_pipeline historical
        # reference so baseline tests remain stable.
        q = torch.rand(heads, seq_len, d, dtype=dtype) * val_range
        k = torch.rand(heads, seq_len, d, dtype=dtype) * val_range
        v = torch.rand(heads, seq_len, d, dtype=dtype) * val_range
    elif mha_debug_mode == DEBUG_SELF_ATTN:
        # Keep self-attention debug inputs bounded so softmax numerics in the
        # hardware pipeline remain comparable to PyTorch reference.
        self_attn_range = min(val_range, 0.5)
        q = torch.randn(heads, seq_len, d, dtype=dtype) * self_attn_range
        k = torch.randn(heads, seq_len, d, dtype=dtype) * self_attn_range
        v = torch.randn(heads, seq_len, d, dtype=dtype) * self_attn_range
    else:
        q = torch.eye(max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=dtype)
        k = torch.eye(max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=dtype)
        v = torch.eye(max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=dtype)
        q = q[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()
        k = k[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()
        v = v[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()
    return q, k, v


def _resolve_modes(
    debug: int,
    addnorm_debug_mode: int,
    addnorm1_debug_mode: int | None,
    addnorm2_debug_mode: int | None,
):
    mha_debug_mode = resolve_mha_debug_mode(debug)
    ffn_stage_only = resolve_ffn_stage_only(debug)
    _, addnorm1_mode, addnorm2_mode = resolve_addnorm_modes(
        addnorm_debug_mode=addnorm_debug_mode,
        addnorm1_debug_mode=addnorm1_debug_mode,
        addnorm2_debug_mode=addnorm2_debug_mode,
    )
    return (
        mha_debug_mode,
        ffn_stage_only,
        addnorm1_mode,
        addnorm2_mode,
    )


def generate_golden_reference(
    heads: int = 3,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seed: int = 42,
    debug: int = DEBUG_FULL,
    addnorm_debug_mode: int = ADDNORM_DEBUG_DISABLED,
    addnorm1_debug_mode: int | None = None,
    addnorm2_debug_mode: int | None = None,
):
    """Golden reference for encoder_pipeline with stage-level debug controls.

    `debug` modes:
      0: full pipeline.
      1: MHA self-attention focused mode (matches mha_to_an).
      2: MHA output-projection focused mode (matches mha_to_an).
      3: FFN up-proj-only stage isolation (matches ffn_addnorm stage_only=0),
         with deterministic MHA/O-proj feeder inputs.
      4: FFN down-proj-only stage isolation (matches ffn_addnorm stage_only=1),
         with deterministic MHA/O-proj feeder inputs.
      5: FFN AddNorm2-only stage isolation (matches ffn_addnorm stage_only=2),
         with deterministic MHA/O-proj feeder inputs.
    """
    (
        mha_debug_mode,
        ffn_stage_only,
        addnorm1_mode,
        addnorm2_mode,
    ) = _resolve_modes(
        debug=debug,
        addnorm_debug_mode=addnorm_debug_mode,
        addnorm1_debug_mode=addnorm1_debug_mode,
        addnorm2_debug_mode=addnorm2_debug_mode,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    intermediate_size = 4 * embed_sz if intermediate_size is None else intermediate_size
    val_range = 4
    dtype = torch.bfloat16

    is_full_debug = debug == DEBUG_FULL
    run_reference_self_attention = mha_debug_mode in (DEBUG_FULL, DEBUG_SELF_ATTN)

    q, k, v = _generate_mha_inputs(
        heads, seq_len, d, embed_sz, val_range, mha_debug_mode
    )

    if mha_debug_mode == DEBUG_O_PROJ:
        w_o = (
            ((torch.arange(embed_sz * embed_sz, dtype=torch.float32) % 256.0) - 128.0)
            .reshape(embed_sz, embed_sz)
            .to(torch.bfloat16)
        )
    elif is_full_debug:
        w_o = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range
    else:
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)

    if run_reference_self_attention:
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=(1.0 / math.sqrt(d)),
        )
    else:
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.matmul(attn_scores, v)

    if is_full_debug:
        # Keep AddNorm scales aligned with mha_to_an full-mode ranges.
        ln1_w = torch.rand(embed_sz, dtype=dtype)
        ln2_w = torch.rand(embed_sz, dtype=dtype)
        r1 = torch.rand(seq_len, embed_sz, dtype=dtype)
        b_up = torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
        b_down = torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range
    else:
        ln1_w = torch.ones(embed_sz, dtype=dtype)
        ln2_w = torch.ones(embed_sz, dtype=dtype)
        r1 = torch.zeros(seq_len, embed_sz, dtype=dtype)
        b_up = torch.eye(embed_sz, intermediate_size, dtype=dtype)
        b_down = torch.eye(intermediate_size, embed_sz, dtype=dtype)

    attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    o_proj = torch.matmul(attn_2d, w_o)
    h1 = _apply_addnorm_stage(o_proj, r1, ln1_w, addnorm1_mode)

    q_2d = q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    k_2d = k.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    v_2d = v.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    qkv = torch.cat((q_2d, k_2d, v_2d), dim=0)
    or_buf = torch.cat((torch.zeros_like(r1), r1), dim=0)

    run_up = ffn_stage_only in (None, 0)
    run_down = ffn_stage_only in (None, 1)
    run_addnorm2 = ffn_stage_only in (None, 2)

    if run_up:
        up = torch.matmul(h1, b_up)
        gelu = torch.nn.functional.gelu(up)
    else:
        up = torch.zeros(seq_len, intermediate_size, dtype=dtype)
        gelu = up

    if run_down:
        down = torch.matmul(gelu, b_down)
    else:
        down = torch.zeros(seq_len, embed_sz, dtype=dtype)

    if run_addnorm2:
        y = _apply_addnorm_stage(down, h1, ln2_w, addnorm2_mode)
    else:
        # Match stage-only behavior: bypass LN2 and keep residual path visible.
        y = h1.clone()

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
