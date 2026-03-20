# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch


def _layer_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )


def generate_golden_reference(
    heads: int = 12,
    seq_len: int = 64,
    d: int = 64,
    intermediate_size: int | None = None,
    seq_tile: int = 32,
    emb_tile: int = 96,
    ffn_tile: int | None = None,
    parallel_seq: int = 1,
    ln1_staging_design: str = "ddr",
    seed: int = 42,
):
    """Generate full-pipeline encoder reference tensors.

    This is the full-pipeline reference only. It intentionally omits the
    debug/profile branching that exists in the DDR bring-up module.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_sz = heads * d
    intermediate_size = 4 * embed_sz if intermediate_size is None else intermediate_size
    if ffn_tile is None:
        ffn_tile = emb_tile
    dtype = torch.bfloat16
    val_range = 4

    q = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    k = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    v = torch.randn(heads, seq_len, d, dtype=dtype) * val_range
    w_o = torch.randn(embed_sz, embed_sz, dtype=dtype) * val_range
    r1 = torch.rand(seq_len, embed_sz, dtype=dtype) * val_range
    b_up = torch.randn(embed_sz, intermediate_size, dtype=dtype) * val_range
    b_down = torch.randn(intermediate_size, embed_sz, dtype=dtype) * val_range
    ln1_w = torch.rand(embed_sz, dtype=dtype)
    ln2_w = torch.rand(embed_sz, dtype=dtype)

    inv_scale = 1 / math.sqrt(d)
    attn = torch.nn.functional.scaled_dot_product_attention(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        dropout_p=0.0,
        is_causal=False,
        scale=inv_scale,
    )
    pv = attn.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    o_proj = torch.matmul(pv, w_o)
    ln1 = _layer_norm(o_proj, ln1_w) + r1
    up = torch.matmul(ln1, b_up)
    up_gelu = torch.nn.functional.gelu(up)
    down = torch.matmul(up_gelu, b_down)
    y = _layer_norm(down, ln2_w) + ln1

    q_2d = q.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    k_2d = k.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    v_2d = v.transpose(0, 1).contiguous().view(seq_len, embed_sz)
    qkv = torch.cat((q_2d, k_2d, v_2d), dim=0)
    ln1_staging_mode = ln1_staging_design.strip().lower()
    if ln1_staging_mode in {"dram", "host"}:
        ln1_staging_mode = "ddr"
    elif ln1_staging_mode in {"mt", "onchip"}:
        ln1_staging_mode = "memtile"
    elif ln1_staging_mode in {"mix", "mixed", "memtile-ddr", "memtile_dram"}:
        ln1_staging_mode = "hybrid"
    elif ln1_staging_mode not in {"ddr", "memtile", "hybrid"}:
        raise ValueError(
            "generate_golden_reference ln1_staging_design must be one of "
            "{ddr, dram, host, memtile, mt, onchip, hybrid, mix, mixed} "
            f"(got {ln1_staging_design!r})"
        )
    if ln1_staging_mode == "memtile":
        ln1_stage_rows = 0
    elif parallel_seq > 1:
        if seq_len % seq_tile != 0:
            raise ValueError(
                "generate_golden_reference requires seq_len divisible by seq_tile"
            )
        num_q_seq_blocks = seq_len // seq_tile
        if num_q_seq_blocks % parallel_seq != 0:
            raise ValueError(
                "generate_golden_reference requires num_q_seq_blocks divisible by "
                f"parallel_seq ({num_q_seq_blocks} % {parallel_seq} != 0)"
            )
        ln1_stage_rows = seq_len
    else:
        ln1_stage_rows = (embed_sz // emb_tile) * seq_tile
    or_buf = torch.cat(
        (
            torch.zeros_like(r1),
            r1,
            torch.zeros((ln1_stage_rows, embed_sz), dtype=dtype),
        ),
        dim=0,
    )

    return {
        "Q": q_2d,
        "K": k_2d,
        "V": v_2d,
        "QKV": qkv,
        "W_O": w_o,
        "R1": r1,
        "OR": or_buf,
        "ln1_weight": ln1_w,
        "B_Up": b_up,
        "B_Down": b_down,
        "ln2_weight": ln2_w,
        "O": y,
    }
