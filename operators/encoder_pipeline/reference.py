# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch
from operators.mha_to_an.reference import (
    ADDNORM_DEBUG_DISABLED,
    ADDNORM_DEBUG_INPUT,
    ADDNORM_DEBUG_RESIDUAL,
)


def generate_golden_reference(
    heads: int = 3,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seed: int = 42,
    addnorm_debug_mode: int = ADDNORM_DEBUG_DISABLED,
):
    """Model-parameter-driven reference for the fused encoder pipeline.

    Stage 1 (MHA path):
      H1 = AddNorm1((MHA(Q,K,V) @ W_O), R1)
    Stage 2 (FFN path):
      Y = AddNorm2(GeLU(H1 @ B_Up) @ B_Down, H1)
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    intermediate_size = 4 * embed_sz if intermediate_size is None else intermediate_size

    val_range = 4
    dtype = torch.bfloat16

    Q = torch.rand(heads, seq_len, d, dtype=dtype) * val_range
    K = torch.rand(heads, seq_len, d, dtype=dtype) * val_range
    V = torch.rand(heads, seq_len, d, dtype=dtype) * val_range

    # Align with stabilized mha_to_an full-mode behavior.
    W_O = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range

    ln1_w = torch.rand(embed_sz, dtype=dtype) * val_range
    R1 = torch.rand(seq_len, embed_sz, dtype=dtype) * val_range

    B_Up = torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
    B_Down = torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range

    # Current encoder_pipeline hardware provides a single LN-weight stream and
    # reuses it for both AddNorm stages.
    ln2_w = ln1_w

    # Stage 1: MHA -> OutProj -> AddNorm1
    attn = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, dropout_p=0.0, is_causal=False, scale=(1.0 / math.sqrt(d))
    )
    attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    o_proj = torch.matmul(attn_2d, W_O)
    if addnorm_debug_mode == ADDNORM_DEBUG_INPUT:
        H1 = o_proj.clone()
    elif addnorm_debug_mode == ADDNORM_DEBUG_RESIDUAL:
        H1 = R1.clone()
    elif addnorm_debug_mode in (None, ADDNORM_DEBUG_DISABLED):
        ln1 = torch.nn.functional.layer_norm(
            o_proj, normalized_shape=(embed_sz,), weight=ln1_w, bias=None
        )
        H1 = ln1 + R1
    else:
        raise ValueError(
            "Invalid addnorm_debug_mode. Expected one of {-1, 0, 1}, "
            f"got {addnorm_debug_mode}."
        )

    # Host-visible packed buffers used by current stage implementations.
    Q_2d = Q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    K_2d = K.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    V_2d = V.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    QKV = torch.cat((Q_2d, K_2d, V_2d), dim=0)
    OR = torch.cat((torch.zeros_like(R1), R1), dim=0)

    # Stage 2: replay H1 as both FFN input and residual
    up = torch.matmul(H1, B_Up)
    gelu = torch.nn.functional.gelu(up)
    down = torch.matmul(gelu, B_Down)
    if addnorm_debug_mode == ADDNORM_DEBUG_INPUT:
        Y = down.clone()
    elif addnorm_debug_mode == ADDNORM_DEBUG_RESIDUAL:
        Y = H1.clone()
    else:
        ln2 = torch.nn.functional.layer_norm(
            down, normalized_shape=(embed_sz,), weight=ln2_w, bias=None
        )
        Y = ln2 + H1

    # AR represents [input_to_addnorm2; residual_to_addnorm2].
    AR = torch.cat((down, H1), dim=0)

    return {
        "Q": Q_2d,
        "K": K_2d,
        "V": V_2d,
        "QKV": QKV,
        "W_O": W_O,
        "R1": R1,
        "OR": OR,
        "ln1_weight": ln1_w,
        "mha_to_an_out": H1,
        "AR": AR,
        "B_Up": B_Up,
        "B_Down": B_Down,
        "ln2_weight": ln2_w,
        "O": Y,
    }
