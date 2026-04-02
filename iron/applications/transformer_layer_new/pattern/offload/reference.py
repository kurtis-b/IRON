# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from iron.common.utils import torch_dtype_map


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype="bf16",
    seed=42,
):
    torch.manual_seed(seed)
    val_range = 0.05
    dtype_torch = torch_dtype_map[dtype]

    input_tensor = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range
    attention_mask = torch.ones(seq_len, seq_len, dtype=dtype_torch)

    head_dim = hidden_size // num_heads
    q_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    k_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    v_weight = torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    attn_output_weight = (
        torch.randn(hidden_size, hidden_size, dtype=dtype_torch) * val_range
    )

    ln1_weight = torch.rand(hidden_size, dtype=dtype_torch)
    ln1_bias = torch.zeros(hidden_size, dtype=dtype_torch)

    ffn_up_weight = (
        torch.randn(hidden_size, intermediate_size, dtype=dtype_torch) * val_range
    )
    ffn_down_weight = (
        torch.randn(intermediate_size, hidden_size, dtype=dtype_torch) * val_range
    )

    ln2_weight = torch.rand(hidden_size, dtype=dtype_torch)
    ln2_bias = torch.zeros(hidden_size, dtype=dtype_torch)

    q = torch.matmul(input_tensor, q_weight)
    k = torch.matmul(input_tensor, k_weight)
    v = torch.matmul(input_tensor, v_weight)

    q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, attn_output_weight)

    hidden_states = torch.nn.functional.layer_norm(
        attn_output + input_tensor, (hidden_size,), ln1_weight, ln1_bias
    )

    intermediate = torch.matmul(hidden_states, ffn_up_weight)
    intermediate = torch.nn.functional.gelu(intermediate)
    ffn_output = torch.matmul(intermediate, ffn_down_weight)

    output = torch.nn.functional.layer_norm(
        ffn_output + hidden_states, (hidden_size,), ln2_weight, ln2_bias
    )

    weights = {
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "attn_output_weight": attn_output_weight,
        "ln1_weight": ln1_weight,
        "ffn_up_weight": ffn_up_weight,
        "ffn_down_weight": ffn_down_weight,
        "ln2_weight": ln2_weight,
    }

    return {
        "input": input_tensor,
        "attention_mask": attention_mask,
        "output": output,
        "weights": weights,
    }
