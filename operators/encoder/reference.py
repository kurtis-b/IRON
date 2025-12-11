# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from operators.common.utils import torch_dtype_map


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype="bf16",
    seed=42,
):
    """
    Generate golden reference for BERT Encoder Layer.

    A BERT Encoder Layer consists of:
    1. Multi-Head Self-Attention (MHA)
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network (FFN): up-projection -> GeLU -> down-projection
    4. Add & Norm (residual connection + layer normalization)

    Args:
        seq_len: Sequence length (M)
        hidden_size: Hidden dimension size (K)
        intermediate_size: Intermediate FFN size (N), typically 4*hidden_size
        num_heads: Number of attention heads
        dtype: Data type for tensors
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - input: Input tensor (seq_len, hidden_size)
            - attention_mask: Attention mask (seq_len, seq_len)
            - output: Final output after encoder layer (seq_len, hidden_size)
            - weights: Dictionary of all weight matrices
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Generate input tensor (seq_len, hidden_size)
    input_tensor = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range

    # Generate attention mask (seq_len, seq_len)
    attention_mask = torch.ones(seq_len, seq_len, dtype=dtype_torch)

    # MHA weights (separate Q/K/V weights)
    head_dim = hidden_size // num_heads
    q_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    k_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    v_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    attn_output_weight = (
        torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    )

    # Layer norm 1 weights
    ln1_weight = torch.ones(hidden_size, dtype=dtype_torch)
    ln1_bias = torch.zeros(hidden_size, dtype=dtype_torch)

    # FFN weights
    ffn_up_weight = (
        torch.randn(hidden_size, intermediate_size, dtype=dtype_torch) * val_range
    )
    ffn_down_weight = (
        torch.randn(intermediate_size, hidden_size, dtype=dtype_torch) * val_range
    )

    # Layer norm 2 weights
    ln2_weight = torch.ones(hidden_size, dtype=dtype_torch)
    ln2_bias = torch.zeros(hidden_size, dtype=dtype_torch)

    # Forward pass simulation
    # 1. Multi-Head Attention (separate Q/K/V projections)
    q = torch.matmul(input_tensor, q_weight)
    k = torch.matmul(input_tensor, k_weight)
    v = torch.matmul(input_tensor, v_weight)

    # Reshape for multi-head attention
    q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

    # Attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    # Reshape back
    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, attn_output_weight)

    # 2. Add & Norm 1
    hidden_states = input_tensor + attn_output
    hidden_states = torch.nn.functional.layer_norm(
        hidden_states, (hidden_size,), ln1_weight, ln1_bias
    )

    # 3. Feed-Forward Network
    # Up-projection
    intermediate = torch.matmul(hidden_states, ffn_up_weight)
    # GeLU activation
    intermediate = torch.nn.functional.gelu(intermediate)
    # Down-projection
    ffn_output = torch.matmul(intermediate, ffn_down_weight)

    # 4. Add & Norm 2
    output = hidden_states + ffn_output
    output = torch.nn.functional.layer_norm(
        output, (hidden_size,), ln2_weight, ln2_bias
    )

    weights = {
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "attn_output_weight": attn_output_weight,
        "ln1_weight": ln1_weight,
        "ln1_bias": ln1_bias,
        "ffn_up_weight": ffn_up_weight,
        "ffn_down_weight": ffn_down_weight,
        "ln2_weight": ln2_weight,
        "ln2_bias": ln2_bias,
    }

    return {
        "input": input_tensor,
        "attention_mask": attention_mask,
        "output": output,
        "weights": weights,
    }
