# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

DEBUG_SELF_ATTN = 0
DEBUG_MHA_INPUT_PATH = 1
DEBUG_RESIDUAL_PATH = 2

VALID_DEBUG_MODES = (DEBUG_SELF_ATTN, DEBUG_MHA_INPUT_PATH, DEBUG_RESIDUAL_PATH)


def generate_golden_reference(
    heads=1,
    seq_len=256,
    d=256,
    seed=42,
    debug=-1,
):
    """
    Golden reference for MHA + O-proj + AddNorm.

    debug:
      -1: disabled (full reference behavior)
       0: self-attention debug, AddNorm input path
       1: MHA-input-path debug (deterministic), AddNorm input path
       2: residual-path debug, AddNorm residual path
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if debug not in (-1, *VALID_DEBUG_MODES):
        raise ValueError(f"debug must be one of {{-1, 0, 1, 2}} (got {debug})")

    dtype = torch.bfloat16
    val_range = 4
    embed_sz = d * heads

    if debug == DEBUG_SELF_ATTN:
        q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=(1.0 / np.sqrt(d)),
        )
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        o = torch.matmul(attn_2d, w_o)
        ln_weight = torch.ones(embed_sz, dtype=dtype)
        residual = torch.zeros(seq_len, embed_sz, dtype=dtype)

    elif debug == DEBUG_MHA_INPUT_PATH:
        base = torch.eye(
            max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=torch.bfloat16
        )
        q2d = base[:seq_len, :embed_sz]
        k2d = base[:seq_len, :embed_sz]
        v2d = torch.arange(seq_len * embed_sz, dtype=dtype).reshape(seq_len, embed_sz)
        q = q2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        k = k2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        v = v2d.view(seq_len, heads, d).transpose(0, 1).contiguous()
        # Match MHA debug path used by kernel (linear QK/PV flow).
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.matmul(attn_scores, v)
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        o = torch.matmul(attn_2d, w_o)
        ln_weight = torch.ones(embed_sz, dtype=dtype)
        residual = torch.zeros(seq_len, embed_sz, dtype=dtype)

    elif debug == DEBUG_RESIDUAL_PATH:
        # Keep MHA input stream zero so residual path validation is unambiguous.
        q = torch.zeros((heads, seq_len, d), dtype=dtype)
        k = torch.zeros((heads, seq_len, d), dtype=dtype)
        v = torch.zeros((heads, seq_len, d), dtype=dtype)
        w_o = torch.eye(embed_sz, embed_sz, dtype=dtype)
        ln_weight = torch.ones(embed_sz, dtype=dtype)
        residual = torch.arange(seq_len * embed_sz, dtype=dtype).reshape(
            seq_len, embed_sz
        )
        o = residual.clone()

    else:
        # Match the original full-path distribution used by passing tests.
        # Centered random inputs reduce saturation in softmax/accumulation and
        # better match hardware numerics at current tolerances.
        q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=(1.0 / np.sqrt(d)),
        )
        w_o = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range
        attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        o = torch.matmul(attn_2d, w_o)
        ln_weight = torch.rand(embed_sz, dtype=dtype)
        residual = torch.rand(seq_len, embed_sz, dtype=dtype)
        o = torch.nn.functional.layer_norm(
            o, normalized_shape=(embed_sz,), weight=ln_weight, bias=None
        )
        o = o + residual

    q2d = q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    k2d = k.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    v2d = v.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    qkv = torch.cat((q2d, k2d, v2d), dim=0)
    or_buf = torch.cat((torch.zeros_like(residual), residual), dim=0)

    return {
        "W_O": w_o,
        "QKV": qkv,
        "OR": or_buf,
        "ln_weight": ln_weight,
        "O": o,
    }
