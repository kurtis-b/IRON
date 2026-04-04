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
    attention_mask = (
        torch.ones(seq_len, seq_len, dtype=dtype_torch)
        if include_attention_mask
        else None
    )

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
        q = torch.matmul(input_tensor, q_weight)
        k = torch.matmul(input_tensor, k_weight)
        v = torch.matmul(input_tensor, v_weight)

        q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
        k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
        v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seq_len, hidden_size)
        )
        attn_output = torch.matmul(attn_output, attn_output_weight)

        hidden_states = torch.nn.functional.layer_norm(
            attn_output + input_tensor,
            (hidden_size,),
            ln1_weight,
            ln1_bias,
        )

        intermediate = torch.matmul(hidden_states, ffn_up_weight)
        intermediate = torch.nn.functional.gelu(intermediate)
        ffn_output = torch.matmul(intermediate, ffn_down_weight)

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


def derive_offload_inputs(
    reference: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    *,
    num_heads: int,
) -> dict[str, torch.Tensor]:
    input_tensor = reference["input"]
    weights = reference["weights"]
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("reference['input'] must be a tensor")
    if not isinstance(weights, dict):
        raise ValueError("reference['weights'] must be a tensor mapping")

    hidden_size = int(input_tensor.shape[-1])
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
        )
    head_dim = hidden_size // num_heads

    q = torch.matmul(input_tensor, weights["q_weight"])
    k = torch.matmul(input_tensor, weights["k_weight"])
    v = torch.matmul(input_tensor, weights["v_weight"])

    return {
        "q": q.view(input_tensor.shape[0], num_heads, head_dim)
        .transpose(0, 1)
        .contiguous(),
        "k": k.view(input_tensor.shape[0], num_heads, head_dim)
        .transpose(0, 1)
        .contiguous(),
        "v": v.view(input_tensor.shape[0], num_heads, head_dim)
        .transpose(0, 1)
        .contiguous(),
        "residual": input_tensor.contiguous(),
    }


def derive_gemm_only_inputs(
    reference: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    *,
    num_heads: int,
) -> dict[str, torch.Tensor]:
    return derive_offload_inputs(reference, num_heads=num_heads)


__all__ = [
    "derive_offload_inputs",
    "derive_gemm_only_inputs",
    "generate_golden_reference",
]
