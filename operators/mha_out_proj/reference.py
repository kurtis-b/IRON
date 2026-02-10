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
):
    """
    Generate golden reference data for MHA (Multi-Head Attention). Note that this function generates data without
    causal masking, as the implementation is for a BERT encoder.

    Parameters:
        heads: Number of query heads
        seq_len: Sequence length for query (Q), key (K), and value (V)
        d: Embedding dimension per head
        seed: Random seed

    Returns:
        dict: Contains 'W_O' (output projection weights), 'Q' (query), 'K' (key), 'V' (value), 'O' (output)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_range = 4

    embed_dim = d * heads
    out_proj_weights = (
        torch.randn(embed_dim, embed_dim, dtype=torch.bfloat16) * val_range
    )

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
    attn_output = O.transpose(0, 1).contiguous().view(seq_len, embed_dim)
    # O = torch.matmul(attn_output, out_proj_weights)

    # Log shapes for debugging
    logging.debug(
        f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}, O shape: {O.shape}"
    )

    return {
        "W_O": out_proj_weights,
        "Q": Q,
        "K": K,
        "V": V,
        "O": O,
    }
