# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.utils import torch_dtype_map


def _layer_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )


def _add_then_norm(
    residual: torch.Tensor, update: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return _layer_norm(update + residual, weight)


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype="bf16",
    seed=42,
):
    """
    Generate golden reference for the post-projection encoder layer.

    The encoder runlist workload consists of:
    1. Multi-Head Self-Attention (MHA) from supplied Q/K/V
    2. Output projection
    3. Add & Norm (residual connection followed by layer normalization)
    4. Feed-Forward Network (FFN): up-projection -> GeLU -> down-projection
    5. Add & Norm (residual connection followed by layer normalization)

    Args:
        seq_len: Sequence length (M)
        hidden_size: Hidden dimension size (K)
        intermediate_size: Intermediate FFN size (N), typically 4*hidden_size
        num_heads: Number of attention heads
        dtype: Data type for tensors
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - Q: Query tensor (seq_len, hidden_size)
            - K: Key tensor (seq_len, hidden_size)
            - V: Value tensor (seq_len, hidden_size)
            - R: Residual input tensor (seq_len, hidden_size)
            - output: Final output after encoder layer (seq_len, hidden_size)
            - weights: Dictionary of downstream weight matrices
    """
    torch.manual_seed(seed)
    val_range = 0.05
    dtype_torch = torch_dtype_map[dtype]

    # Generate post-projection inputs (seq_len, hidden_size)
    head_dim = hidden_size // num_heads
    q_input = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range
    k_input = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range
    v_input = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range
    residual_input = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range

    # Downstream encoder weights
    attn_output_weight = (
        torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    )

    # Layer norm 1 weights
    ln1_weight = torch.rand(hidden_size, dtype=dtype_torch)
    # FFN weights
    ffn_up_weight = (
        torch.randn(hidden_size, intermediate_size, dtype=dtype_torch) * val_range
    )
    ffn_down_weight = (
        torch.randn(intermediate_size, hidden_size, dtype=dtype_torch) * val_range
    )

    # Layer norm 2 weights
    ln2_weight = torch.rand(hidden_size, dtype=dtype_torch)

    # Forward pass simulation from supplied Q/K/V/R
    q = q_input.view(seq_len, num_heads, head_dim).transpose(0, 1)
    k = k_input.view(seq_len, num_heads, head_dim).transpose(0, 1)
    v = v_input.view(seq_len, num_heads, head_dim).transpose(0, 1)

    # Attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    # Reshape back
    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, attn_output_weight)

    # 2. Norm & Add 1
    hidden_states = _add_then_norm(residual_input, attn_output, ln1_weight)

    # 3. Feed-Forward Network
    # Up-projection
    intermediate = torch.matmul(hidden_states, ffn_up_weight)
    # GeLU activation
    intermediate = torch.nn.functional.gelu(intermediate)
    # Down-projection
    ffn_output = torch.matmul(intermediate, ffn_down_weight)

    # 4. Norm & Add 2
    output = _add_then_norm(hidden_states, ffn_output, ln2_weight)

    weights = {
        "attn_output_weight": attn_output_weight,
        "ln1_weight": ln1_weight,
        "ffn_up_weight": ffn_up_weight,
        "ffn_down_weight": ffn_down_weight,
        "ln2_weight": ln2_weight,
    }

    return {
        "Q": q_input,
        "K": k_input,
        "V": v_input,
        "R": residual_input,
        "output": output,
        "weights": weights,
    }
