# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
import logging


def generate_golden_reference(
    heads=1,
    seq_len=256,
    d=256,
    seed=42,
    debug=0,
):
    """
    Generate golden reference data for MHA (Multi-Head Attention) + Add & Norm.
    Note that this function generates data without causal masking, as the
    implementation is for a BERT encoder.

    Parameters:
        heads: Number of query heads
        seq_len: Sequence length for query (Q), key (K), and value (V)
        d: Embedding dimension per head
        seed: Random seed
        debug: Debug particular operations
            - 0: No debug
            - 1: Self attention (QK^T, softmax, AV)
            - 2: MHA output projection
    Returns:
        dict: Contains:
              - 'W_O' (output projection weights)
              - 'QKV' (combined query/key/value buffer, shape [3*seq_len, embed_sz])
              - 'OR' (combined output/residual buffer, shape [2*seq_len, embed_sz])
              - 'ln_weight' (layer norm weights)
              - 'O' (expected output)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_range = 4

    embed_sz = d * heads
    # Need to adjust inputs for debugging NPU execution
    if debug not in (0, 2):
        out_proj_weights = torch.eye(embed_sz, embed_sz, dtype=torch.bfloat16)
    else:
        if debug == 2:
            out_proj_weights = torch.arange(
                embed_sz * embed_sz, dtype=torch.bfloat16
            ).reshape(embed_sz, embed_sz)
        else:
            out_proj_weights = (
                torch.rand(embed_sz, embed_sz, dtype=torch.bfloat16) * val_range
            )

    if debug not in (0, 1):
        Q = torch.eye(
            max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=torch.bfloat16
        )
        K = torch.eye(
            max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=torch.bfloat16
        )
        V = torch.eye(
            max(seq_len, embed_sz), max(seq_len, embed_sz), dtype=torch.bfloat16
        )
        Q = Q[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()
        K = K[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()
        V = V[:seq_len, :embed_sz].view(seq_len, heads, d).transpose(0, 1).contiguous()

        # Skip softmax computation when not debugging self attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        O = torch.matmul(attn_scores, V)
    else:
        if debug == 1:
            # Q = torch.ones(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            # K = torch.ones(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            # V = torch.ones(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            Q = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            K = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            V = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
        else:
            Q = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            K = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
            V = torch.rand(heads, seq_len, d, dtype=torch.bfloat16) * val_range
        # MHA from PyTorch
        inv_scale = 1 / np.sqrt(K.shape[-1])
        O = torch.nn.functional.scaled_dot_product_attention(
            Q.to(torch.bfloat16),
            K.to(torch.bfloat16),
            V.to(torch.bfloat16),
            dropout_p=0.0,
            is_causal=False,
            scale=inv_scale,
        )

    # Apply output projection
    attn_output = O.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    O = torch.matmul(attn_output, out_proj_weights)

    # Generate layer norm weights and residual for Add & Norm stage
    if debug != 0:
        ln_weight = torch.ones(embed_sz, dtype=torch.bfloat16)
        R = torch.zeros(seq_len, embed_sz, dtype=torch.bfloat16)
    else:
        ln_weight = torch.rand(embed_sz, dtype=torch.bfloat16)
        R = torch.rand(seq_len, embed_sz, dtype=torch.bfloat16)

    # Apply layer norm + residual add
    layer_norm_output = torch.nn.functional.layer_norm(
        O, normalized_shape=(embed_sz,), weight=ln_weight, bias=None
    )
    O = layer_norm_output + R

    # Reshape for NPU format
    Q = Q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    K = K.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    V = V.transpose(0, 1).contiguous().view(seq_len, embed_sz)

    # Combine host-facing buffers:
    # QKV packs [Q; K; V], OR packs [output_region; residual_input].
    QKV = torch.cat((Q, K, V), dim=0)
    OR = torch.cat((torch.zeros_like(R), R), dim=0)

    # Log shapes for debugging
    logging.debug(
        f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}, QKV shape: {QKV.shape}, OR shape: {OR.shape}, O shape: {O.shape}"
    )

    return {
        "W_O": out_proj_weights,
        "QKV": QKV,
        "OR": OR,
        "Q": Q,
        "K": K,
        "V": V,
        "ln_weight": ln_weight,
        "R": R,
        "O": O,
    }
