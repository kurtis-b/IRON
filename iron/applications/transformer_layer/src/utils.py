# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec


def dtype_from_name(dtype_name: str) -> torch.dtype:
    return TransformerLayerSpec(dtype=dtype_name).torch_dtype


def make_synthetic_layer_inputs(
    spec: TransformerLayerSpec,
    *,
    seed: int = 0,
) -> TransformerLayerInputs:
    generator = torch.Generator().manual_seed(seed)
    qkv_shape = (
        spec.batch_size,
        spec.num_attention_heads,
        spec.seq_len,
        spec.attention_head_size,
    )
    residual_shape = (spec.batch_size, spec.seq_len, spec.hidden_size)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        ).to(spec.torch_dtype)

    return TransformerLayerInputs(
        q=randn(*qkv_shape),
        k=randn(*qkv_shape),
        v=randn(*qkv_shape),
        r=randn(*residual_shape),
    )


def make_synthetic_layer_weights(
    spec: TransformerLayerSpec,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    dtype = spec.torch_dtype
    hidden = spec.hidden_size
    intermediate = spec.intermediate_size

    def randn(*shape: int) -> torch.Tensor:
        return (0.02 * torch.randn(shape, generator=generator, dtype=torch.float32)).to(
            dtype
        )

    weights = {
        "out_proj_weight": randn(hidden, hidden),
        "ffn_up_weight": randn(intermediate, hidden),
        "ffn_down_weight": randn(hidden, intermediate),
        "ln1_weight": torch.ones(hidden, dtype=dtype),
        "ln2_weight": torch.ones(hidden, dtype=dtype),
    }
    if spec.use_bias:
        weights.update(
            {
                "out_proj_bias": randn(hidden),
                "ffn_up_bias": randn(intermediate),
                "ffn_down_bias": randn(hidden),
            }
        )
    return weights


def require_keys(
    weights: Mapping[str, torch.Tensor],
    required_keys: list[str],
) -> None:
    missing = [key for key in required_keys if key not in weights]
    if missing:
        raise KeyError(f"Missing required weight keys: {', '.join(missing)}")
