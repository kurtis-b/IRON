# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.common.utils import torch_dtype_map


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype: str = "bf16",
    seed: int = 42,
    *,
    workload_variant: str = "encoder_bert",
    include_output: bool = True,
    include_attention_mask: bool | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
    """
    Generate shared synthetic tensors for the transformer-layer patterns.

    When ``include_output`` is false, the helper skips materializing the full
    attention pipeline output so the long sequence ladder stays practical for the
    study runner.
    """

    torch.manual_seed(seed)
    val_range = 0.05
    dtype_torch = torch_dtype_map[dtype]

    if include_attention_mask is None:
        include_attention_mask = include_output

    input_tensor = torch.randn(seq_len, hidden_size, dtype=dtype_torch) * val_range
    if include_attention_mask:
        if workload_variant == "decoder_gpt2":
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype_torch))
        else:
            attention_mask = torch.ones(seq_len, seq_len, dtype=dtype_torch)
    else:
        attention_mask = None

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

    output = None
    if include_output:
        if workload_variant == "encoder_bert":
            attn_input = input_tensor
            residual_after_attention = input_tensor
            use_causal_attention = False
            apply_post_attention_norm = True
        elif workload_variant == "decoder_gpt2":
            attn_input = torch.nn.functional.layer_norm(
                input_tensor,
                (hidden_size,),
                ln1_weight,
                ln1_bias,
            )
            residual_after_attention = input_tensor
            apply_post_attention_norm = False
            use_causal_attention = True
        else:
            raise ValueError(f"Unsupported workload_variant: {workload_variant}")

        q = torch.matmul(attn_input, q_weight)
        k = torch.matmul(attn_input, k_weight)
        v = torch.matmul(attn_input, v_weight)

        q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
        k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
        v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        if use_causal_attention:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=attn_scores.device
                ),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0), float("-inf")
            )
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
        )
        attn_output = torch.matmul(attn_output, attn_output_weight)

        residual_hidden_states = attn_output + residual_after_attention
        if apply_post_attention_norm:
            hidden_states = torch.nn.functional.layer_norm(
                residual_hidden_states,
                (hidden_size,),
                ln1_weight,
                ln1_bias,
            )
        else:
            hidden_states = residual_hidden_states

        ffn_input = (
            torch.nn.functional.layer_norm(
                residual_hidden_states,
                (hidden_size,),
                ln2_weight,
                ln2_bias,
            )
            if workload_variant == "decoder_gpt2"
            else hidden_states
        )

        intermediate = torch.matmul(ffn_input, ffn_up_weight)
        intermediate = torch.nn.functional.gelu(intermediate)
        ffn_output = torch.matmul(intermediate, ffn_down_weight)

        if workload_variant == "decoder_gpt2":
            output = ffn_output + residual_hidden_states
        else:
            output = torch.nn.functional.layer_norm(
                ffn_output + hidden_states,
                (hidden_size,),
                ln2_weight,
                ln2_bias,
            )

    return {
        "input": input_tensor,
        "attention_mask": attention_mask,
        "output": output,
        "weights": weights,
    }


__all__ = [
    "generate_golden_reference",
]
