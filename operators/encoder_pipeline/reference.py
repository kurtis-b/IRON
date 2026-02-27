# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch
from operators.mha_to_an.reference import (
    generate_golden_reference as generate_mha_reference,
)


def _expect_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}; expected {expected}")


def generate_golden_reference(
    heads: int = 3,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seed: int = 42,
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    w_o: torch.Tensor | None = None,
    r1: torch.Tensor | None = None,
    b_up: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
    ln1_weight: torch.Tensor | None = None,
    ln2_weight: torch.Tensor | None = None,
):
    """Model-parameter-driven reference for a fused encoder pipeline path.

    Pipeline model:
    1) MHA + output projection + AddNorm1: `H1 = LN((MHA(Q,K,V) @ W_O)) + R1`
    2) FFN + AddNorm2 with streamed replay of H1: `Y = LN(GeLU(H1 @ B_Up) @ B_Down) + H1`

    The reference intentionally does not take hardware tiling parameters.
    Inputs are model parameters/tensors; when tensors are omitted, they are generated.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    if intermediate_size is None:
        intermediate_size = 4 * embed_sz

    val_range = 4
    dtype = torch.bfloat16

    stage1_inputs_provided = any(x is not None for x in (q, k, v, w_o, r1, ln1_weight))

    if not stage1_inputs_provided:
        # Keep stage-1 aligned with mha_to_an reference behavior.
        stage1 = generate_mha_reference(
            heads=heads,
            seq_len=seq_len,
            d=d,
            seed=seed,
            debug=0,
        )
        Q_2d = stage1["Q"]
        K_2d = stage1["K"]
        V_2d = stage1["V"]
        QKV = stage1["QKV"]
        W_O = stage1["W_O"]
        R1 = stage1["R"]
        OR = stage1["OR"]
        ln1_w = stage1["ln_weight"]
        H1 = stage1["O"]
    else:
        # Stage 1 inputs from model parameters/tensors.
        Q = (
            q
            if q is not None
            else torch.rand(heads, seq_len, d, dtype=dtype) * val_range
        )
        K = (
            k
            if k is not None
            else torch.rand(heads, seq_len, d, dtype=dtype) * val_range
        )
        V = (
            v
            if v is not None
            else torch.rand(heads, seq_len, d, dtype=dtype) * val_range
        )
        W_O = (
            w_o
            if w_o is not None
            else torch.rand(embed_sz, embed_sz, dtype=dtype) * val_range
        )
        ln1_w = (
            ln1_weight
            if ln1_weight is not None
            else torch.rand(embed_sz, dtype=dtype) * val_range
        )
        R1 = (
            r1
            if r1 is not None
            else torch.rand(seq_len, embed_sz, dtype=dtype) * val_range
        )

        _expect_shape(Q, (heads, seq_len, d), "q")
        _expect_shape(K, (heads, seq_len, d), "k")
        _expect_shape(V, (heads, seq_len, d), "v")
        _expect_shape(W_O, (embed_sz, embed_sz), "w_o")
        _expect_shape(R1, (seq_len, embed_sz), "r1")
        _expect_shape(ln1_w, (embed_sz,), "ln1_weight")

        # Stage 1: MHA -> OutProj -> AddNorm1
        attn = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, is_causal=False, scale=(1.0 / math.sqrt(d))
        )
        attn_2d = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        o_proj = torch.matmul(attn_2d, W_O)
        ln1 = torch.nn.functional.layer_norm(
            o_proj, normalized_shape=(embed_sz,), weight=ln1_w, bias=None
        )
        H1 = ln1 + R1

        # Host-visible packed buffers used by current stage implementations.
        Q_2d = Q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        K_2d = K.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        V_2d = V.transpose(0, 1).contiguous().view(seq_len, embed_sz)
        QKV = torch.cat((Q_2d, K_2d, V_2d), dim=0)
        OR = torch.cat((torch.zeros_like(R1), R1), dim=0)

    # Stage 2 inputs
    B_Up = (
        b_up
        if b_up is not None
        else torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
    )
    B_Down = (
        b_down
        if b_down is not None
        else torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range
    )
    ln2_w = ln2_weight if ln2_weight is not None else ln1_w

    # Validate model parameter shapes.
    _expect_shape(W_O, (embed_sz, embed_sz), "w_o")
    _expect_shape(R1, (seq_len, embed_sz), "r1")
    _expect_shape(B_Up, (embed_sz, intermediate_size), "b_up")
    _expect_shape(B_Down, (intermediate_size, embed_sz), "b_down")
    _expect_shape(ln1_w, (embed_sz,), "ln1_weight")
    _expect_shape(ln2_w, (embed_sz,), "ln2_weight")

    # Stage 2: replay H1 as both FFN input and residual
    up = torch.matmul(H1, B_Up)
    gelu = torch.nn.functional.gelu(up)
    down = torch.matmul(gelu, B_Down)
    ln2 = torch.nn.functional.layer_norm(
        down, normalized_shape=(embed_sz,), weight=ln2_w, bias=None
    )
    Y = ln2 + H1

    AR = torch.cat((H1, H1), dim=0)

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
