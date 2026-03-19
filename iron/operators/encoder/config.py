# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_MODEL_FAMILIES = ("bert", "roberta", "distilbert")


@dataclass(frozen=True)
class EncoderArchitecture:
    seq_len: int = 512
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 1
    model_type: str = "bert"
    seq_tile: int = 32
    kv_seq_tile: int = 64
    emb_tile: int = 96
    ffn_tile: int = 64
    parallel_seq: int = 1
    parallel_heads: int = 1
    proj_acc_depth: int | None = None
    o_proj_acc_group_size: int = 1
    ffn_down_acc_group_size: int = 1
    nB_tiles_distributed: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def normalize_model_family(model_type: str | None) -> str:
    if model_type is None:
        return "bert"

    normalized = str(model_type).strip().lower().replace("_", "-")
    aliases = {
        "bert": "bert",
        "roberta": "roberta",
        "distilbert": "distilbert",
        "distil-bert": "distilbert",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Supported: {SUPPORTED_MODEL_FAMILIES}"
        )
    return aliases[normalized]


def canonicalize_architecture(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_hidden_layers: int = 1,
    model_type: str = "bert",
    seq_tile: int = 32,
    kv_seq_tile: int = 64,
    emb_tile: int = 96,
    ffn_tile: int = 64,
    parallel_seq: int = 1,
    parallel_heads: int = 1,
    proj_acc_depth: int | None = None,
    o_proj_acc_group_size: int = 1,
    ffn_down_acc_group_size: int = 1,
    nB_tiles_distributed: int = 1,
) -> EncoderArchitecture:
    model_family = normalize_model_family(model_type)
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads "
            f"({hidden_size} % {num_attention_heads} != 0)"
        )
    head_dim = hidden_size // num_attention_heads
    if head_dim != 64:
        raise ValueError(
            f"encoder_pipeline only supports head_dim=64 today (got {head_dim})"
        )
    if proj_acc_depth is None:
        if hidden_size % emb_tile != 0:
            raise ValueError(
                f"emb_tile must divide hidden_size ({hidden_size} % {emb_tile} != 0)"
            )
        proj_acc_depth = hidden_size // emb_tile

    return EncoderArchitecture(
        seq_len=int(seq_len),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_hidden_layers=int(num_hidden_layers),
        model_type=model_family,
        seq_tile=int(seq_tile),
        kv_seq_tile=int(kv_seq_tile),
        emb_tile=int(emb_tile),
        ffn_tile=int(ffn_tile),
        parallel_seq=int(parallel_seq),
        parallel_heads=int(parallel_heads),
        proj_acc_depth=int(proj_acc_depth),
        o_proj_acc_group_size=int(o_proj_acc_group_size),
        ffn_down_acc_group_size=int(ffn_down_acc_group_size),
        nB_tiles_distributed=int(nB_tiles_distributed),
    )
