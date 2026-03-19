# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import torch

from .config import (
    EncoderArchitecture,
    canonicalize_architecture,
    normalize_model_family,
)


def _layer_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=None,
    )


def _random_weights(shape: tuple[int, ...], dtype: torch.dtype, val_range: float):
    return torch.randn(*shape, dtype=dtype) * val_range


def _layer_weight_key_map(model_type: str, layer_idx: int) -> dict[str, str]:
    family = normalize_model_family(model_type)
    if family in ("bert", "roberta"):
        prefix = f"encoder.layer.{layer_idx}"
        return {
            "query.weight": f"{prefix}.attention.self.query.weight",
            "query.bias": f"{prefix}.attention.self.query.bias",
            "key.weight": f"{prefix}.attention.self.key.weight",
            "key.bias": f"{prefix}.attention.self.key.bias",
            "value.weight": f"{prefix}.attention.self.value.weight",
            "value.bias": f"{prefix}.attention.self.value.bias",
            "o_proj.weight": f"{prefix}.attention.output.dense.weight",
            "ln1.weight": f"{prefix}.attention.output.LayerNorm.weight",
            "up.weight": f"{prefix}.intermediate.dense.weight",
            "down.weight": f"{prefix}.output.dense.weight",
            "ln2.weight": f"{prefix}.output.LayerNorm.weight",
        }

    prefix = f"transformer.layer.{layer_idx}"
    return {
        "query.weight": f"{prefix}.attention.q_lin.weight",
        "query.bias": f"{prefix}.attention.q_lin.bias",
        "key.weight": f"{prefix}.attention.k_lin.weight",
        "key.bias": f"{prefix}.attention.k_lin.bias",
        "value.weight": f"{prefix}.attention.v_lin.weight",
        "value.bias": f"{prefix}.attention.v_lin.bias",
        "o_proj.weight": f"{prefix}.attention.out_lin.weight",
        "ln1.weight": f"{prefix}.sa_layer_norm.weight",
        "up.weight": f"{prefix}.ffn.lin1.weight",
        "down.weight": f"{prefix}.ffn.lin2.weight",
        "ln2.weight": f"{prefix}.output_layer_norm.weight",
    }


def generate_golden_reference(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
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
    dtype: str = "bf16",
    seed: int = 42,
):
    """Generate random BERT-family encoder-stack inputs, weights, and output.

    The reference matches the encoder_pipeline execution contract:
    q/k/v projections use CPU-side linear layers with bias, while O-proj, FFN,
    and layer-norm use weight-only pipeline semantics.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    architecture = canonicalize_architecture(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        model_type=model_type,
        seq_tile=seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        ffn_down_acc_group_size=ffn_down_acc_group_size,
        nB_tiles_distributed=nB_tiles_distributed,
    )

    if dtype != "bf16":
        raise ValueError(f"Unsupported dtype {dtype!r}; only 'bf16' is supported")

    dtype_torch = torch.bfloat16
    val_range = 0.05
    input_tensor = _random_weights(
        (architecture.seq_len, architecture.hidden_size), dtype_torch, val_range
    )
    hidden_states = input_tensor
    weights: dict[str, torch.Tensor] = {}

    for layer_idx in range(architecture.num_hidden_layers):
        keys = _layer_weight_key_map(architecture.model_type, layer_idx)
        query_weight = _random_weights(
            (architecture.hidden_size, architecture.hidden_size), dtype_torch, val_range
        )
        key_weight = _random_weights(
            (architecture.hidden_size, architecture.hidden_size), dtype_torch, val_range
        )
        value_weight = _random_weights(
            (architecture.hidden_size, architecture.hidden_size), dtype_torch, val_range
        )
        query_bias = _random_weights(
            (architecture.hidden_size,), dtype_torch, val_range
        )
        key_bias = _random_weights((architecture.hidden_size,), dtype_torch, val_range)
        value_bias = _random_weights(
            (architecture.hidden_size,), dtype_torch, val_range
        )
        o_proj_weight = _random_weights(
            (architecture.hidden_size, architecture.hidden_size), dtype_torch, val_range
        )
        ln1_weight = torch.rand(architecture.hidden_size, dtype=dtype_torch)
        up_weight = _random_weights(
            (architecture.intermediate_size, architecture.hidden_size),
            dtype_torch,
            val_range,
        )
        down_weight = _random_weights(
            (architecture.hidden_size, architecture.intermediate_size),
            dtype_torch,
            val_range,
        )
        ln2_weight = torch.rand(architecture.hidden_size, dtype=dtype_torch)

        weights[keys["query.weight"]] = query_weight
        weights[keys["query.bias"]] = query_bias
        weights[keys["key.weight"]] = key_weight
        weights[keys["key.bias"]] = key_bias
        weights[keys["value.weight"]] = value_weight
        weights[keys["value.bias"]] = value_bias
        weights[keys["o_proj.weight"]] = o_proj_weight
        weights[keys["ln1.weight"]] = ln1_weight
        weights[keys["up.weight"]] = up_weight
        weights[keys["down.weight"]] = down_weight
        weights[keys["ln2.weight"]] = ln2_weight

        q = torch.nn.functional.linear(hidden_states, query_weight, query_bias)
        k = torch.nn.functional.linear(hidden_states, key_weight, key_bias)
        v = torch.nn.functional.linear(hidden_states, value_weight, value_bias)

        q = q.view(
            architecture.seq_len,
            architecture.num_attention_heads,
            architecture.head_dim,
        ).transpose(0, 1)
        k = k.view(
            architecture.seq_len,
            architecture.num_attention_heads,
            architecture.head_dim,
        ).transpose(0, 1)
        v = v.view(
            architecture.seq_len,
            architecture.num_attention_heads,
            architecture.head_dim,
        ).transpose(0, 1)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=1 / math.sqrt(architecture.head_dim),
        )
        attn_2d = (
            attn.transpose(0, 1)
            .contiguous()
            .view(architecture.seq_len, architecture.hidden_size)
        )
        o_proj = torch.nn.functional.linear(attn_2d, o_proj_weight)
        ln1 = _layer_norm(o_proj, ln1_weight) + hidden_states
        up = torch.nn.functional.linear(ln1, up_weight)
        up_gelu = torch.nn.functional.gelu(up)
        down = torch.nn.functional.linear(up_gelu, down_weight)
        hidden_states = _layer_norm(down, ln2_weight) + ln1

    return {
        "architecture": architecture,
        "input": input_tensor,
        "output": hidden_states,
        "weights": weights,
    }
