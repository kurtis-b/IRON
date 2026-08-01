# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch


def _chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    query_block_size: int,
    is_causal: bool,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    k_t = k.transpose(-1, -2)
    for start in range(0, q.shape[1], query_block_size):
        end = min(start + query_block_size, q.shape[1])
        q_block = q[:, start:end, :]
        scores = torch.matmul(q_block, k_t) * scale
        if is_causal:
            q_positions = torch.arange(
                start,
                end,
                device=scores.device,
            ).unsqueeze(1)
            kv_positions = torch.arange(
                k.shape[1],
                device=scores.device,
            ).unsqueeze(0)
            causal_mask = kv_positions > q_positions
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        blocks.append(torch.matmul(probs, v))
    return torch.cat(blocks, dim=1)


def generate_golden_reference(heads=1, seq_len=256, d=64, seed=42, is_causal=False):
    torch.manual_seed(seed)
    val_range = 4

    embed_sz = heads * d
    input_q = torch.rand(seq_len, embed_sz, dtype=torch.bfloat16) * val_range
    input_k = torch.rand(seq_len, embed_sz, dtype=torch.bfloat16) * val_range
    input_v = torch.rand(seq_len, embed_sz, dtype=torch.bfloat16) * val_range
    input_w_o = torch.rand(embed_sz, embed_sz, dtype=torch.bfloat16) * val_range

    q = input_q.view(seq_len, heads, d).transpose(0, 1).contiguous()
    k = input_k.view(seq_len, heads, d).transpose(0, 1).contiguous()
    v = input_v.view(seq_len, heads, d).transpose(0, 1).contiguous()

    scale = 1 / math.sqrt(d)
    if seq_len >= 16384:
        attention = _chunked_attention(
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            scale=scale,
            query_block_size=256,
            is_causal=is_causal,
        ).to(torch.bfloat16)
    else:
        attention = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
    attention = attention.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    output = torch.matmul(attention, input_w_o)

    return {
        "input_q": input_q,
        "input_k": input_k,
        "input_v": input_v,
        "input_w_o": input_w_o,
        "output": output,
    }
