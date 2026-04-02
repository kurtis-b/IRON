# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch


def generate_golden_reference(heads=1, seq_len=256, d=64, seed=42):
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

    attention = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=1 / math.sqrt(d),
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
