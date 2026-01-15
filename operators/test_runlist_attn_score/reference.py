# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from operators.common.utils import torch_dtype_map


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    dtype="bf16",
    use_sep_gemms=True,
    seed=42,
):
    """
    Generate golden reference for BERT attention scores layer.
    If use_sep_gemms is True, then the individual attention scores will be calculated using unbatched gemm kernel.
    If use_sep_gemms is False, then the attention scores will be calculated using batched gemm kernel.
    """
    torch.manual_seed(seed)
    val_range = 0.05
    dtype_torch = torch_dtype_map[dtype]

    # Generate input tensor (seq_len, hidden_size)
    input_tensor = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range

    # MHA weights (separate Q/K/V weights)
    head_dim = hidden_size // num_heads
    q_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    k_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range

    # Forward pass simulation
    # 1. Multi-Head Attention (separate Q/K/V projections)
    q = torch.matmul(input_tensor, q_weight)
    k = torch.matmul(input_tensor, k_weight)

    # Reshape for multi-head attention
    q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
    k = k.transpose(-2, -1)

    # Attention scores
    attn_scores = torch.matmul(q, k)

    if use_sep_gemms:
        buffer_q = {
            f"q_{i}": q[i].contiguous() for i in range(num_heads)
        }
        buffer_k = {
            f"k_{i}": k[i].contiguous() for i in range(num_heads)
        }
        buffer_attn_scores = {
            f"attn_scores_{i}": attn_scores[i].contiguous() for i in range(num_heads)
        }
        return buffer_q | buffer_k | buffer_attn_scores
    else:
        return {
            "q": q,
            "k": k,
            "attn_scores": attn_scores,
        }
