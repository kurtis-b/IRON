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
    hidden_shape = (spec.batch_size, spec.seq_len, spec.hidden_size)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        ).to(spec.torch_dtype)

    return TransformerLayerInputs(hidden_states=randn(*hidden_shape))


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
        "q_proj_weight": randn(hidden, hidden),
        "k_proj_weight": randn(hidden, hidden),
        "v_proj_weight": randn(hidden, hidden),
        "out_proj_weight": randn(hidden, hidden),
        "ffn_up_weight": randn(intermediate, hidden),
        "ffn_down_weight": randn(hidden, intermediate),
        "ln1_weight": torch.ones(hidden, dtype=dtype),
        "ln2_weight": torch.ones(hidden, dtype=dtype),
    }
    if spec.use_bias:
        weights.update(
            {
                "q_proj_bias": randn(hidden),
                "k_proj_bias": randn(hidden),
                "v_proj_bias": randn(hidden),
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


def select_mha_out_proj_emb_tile(hidden_size: int) -> int:
    if hidden_size % 128 == 0:
        return 128
    if hidden_size % 96 == 0:
        return 96
    raise ValueError(
        "Block 2 currently requires hidden_size divisible by 128 or 96 for output-projection tiling"
    )


def host_project_qkv_head_major(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    spec: TransformerLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.matmul(hidden_states, weights["q_proj_weight"].T.contiguous())
    k = torch.matmul(hidden_states, weights["k_proj_weight"].T.contiguous())
    v = torch.matmul(hidden_states, weights["v_proj_weight"].T.contiguous())
    return (
        q.view(spec.seq_len, spec.num_attention_heads, spec.attention_head_size)
        .permute(1, 0, 2)
        .contiguous(),
        k.view(spec.seq_len, spec.num_attention_heads, spec.attention_head_size)
        .permute(1, 0, 2)
        .contiguous(),
        v.view(spec.seq_len, spec.num_attention_heads, spec.attention_head_size)
        .permute(1, 0, 2)
        .contiguous(),
    )


def host_attention_output(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    spec: TransformerLayerSpec,
) -> torch.Tensor:
    q, k, v = host_project_qkv_head_major(hidden_states, weights, spec)
    attn_scores = torch.matmul(q, k.transpose(-1, -2))
    attn_scores = attn_scores * (spec.attention_head_size**-0.5)
    attn_probs = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(q.dtype)
    attn_context = torch.matmul(attn_probs, v)
    attn_context = (
        attn_context.transpose(0, 1).contiguous().view(spec.seq_len, spec.hidden_size)
    )
    return torch.matmul(attn_context, weights["out_proj_weight"].T.contiguous())


def bind_qkv_proj_weights(block, weights: Mapping[str, torch.Tensor]) -> None:
    block.q_proj.weight = weights["q_proj_weight"].contiguous()
    block.k_proj.weight = weights["k_proj_weight"].contiguous()
    block.v_proj.weight = weights["v_proj_weight"].contiguous()


def bind_mha_out_proj_weights(block, weights: Mapping[str, torch.Tensor]) -> None:
    block.w_o_proj = weights["out_proj_weight"].contiguous()


def bind_addnorm_ffn_addnorm_weights(
    block, weights: Mapping[str, torch.Tensor]
) -> None:
    block.block.weight_up_proj = weights["ffn_up_weight"].contiguous()
    block.block.weight_down_proj = weights["ffn_down_weight"].contiguous()
    block.block.ln2_weight = weights["ln2_weight"].contiguous()


def compile_setup_time_ms(compile_setup_time_sec: float | None) -> float | None:
    if compile_setup_time_sec is None:
        return None
    return compile_setup_time_sec * 1000.0


def make_in_process_npu_metadata(
    *,
    compile_setup_time_sec: float | None,
    dispatch_count: int,
    unique_instruction_binary_count: int,
    unique_xclbin_count: int,
) -> dict[str, object]:
    return {
        "compile_setup_time_ms": compile_setup_time_ms(compile_setup_time_sec),
        "npu_dispatch_count": dispatch_count,
        "npu_unique_instruction_binary_count": unique_instruction_binary_count,
        "npu_unique_xclbin_count": unique_xclbin_count,
        "process_model": "in_process",
    }
