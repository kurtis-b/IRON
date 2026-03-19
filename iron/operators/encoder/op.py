# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline

from .config import (
    EncoderArchitecture,
    canonicalize_architecture,
    normalize_model_family,
)


@dataclass(frozen=True)
class EncoderLayerWeights:
    query_weight: torch.Tensor
    query_bias: torch.Tensor
    key_weight: torch.Tensor
    key_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor
    attn_output_weight: torch.Tensor
    ln1_weight: torch.Tensor
    ffn_up_weight: torch.Tensor
    ffn_down_weight: torch.Tensor
    ln2_weight: torch.Tensor


class BertLikeQKVProjection(nn.Module):
    def __init__(self, architecture: EncoderArchitecture, dtype=torch.bfloat16):
        super().__init__()
        hidden_size = architecture.hidden_size
        self.num_attention_heads = architecture.num_attention_heads
        self.attention_head_size = architecture.head_dim
        self.query = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        return query_layer, key_layer, value_layer

    def assign_weights(self, layer_weights: EncoderLayerWeights, dtype: torch.dtype):
        self.query.weight = nn.Parameter(layer_weights.query_weight.to(dtype).clone())
        self.query.bias = nn.Parameter(layer_weights.query_bias.to(dtype).clone())
        self.key.weight = nn.Parameter(layer_weights.key_weight.to(dtype).clone())
        self.key.bias = nn.Parameter(layer_weights.key_bias.to(dtype).clone())
        self.value.weight = nn.Parameter(layer_weights.value_weight.to(dtype).clone())
        self.value.bias = nn.Parameter(layer_weights.value_bias.to(dtype).clone())


class AIETransformerEncoderLayer(nn.Module):
    """One BERT-family encoder layer backed by AIEEncoderPipeline."""

    def __init__(
        self,
        architecture: EncoderArchitecture,
        layer_idx: int,
        context=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.architecture = architecture
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.qkv_projection = BertLikeQKVProjection(architecture, dtype=dtype)
        self.encoder_pipeline = AIEEncoderPipeline(
            num_heads=architecture.num_attention_heads,
            seq_len=architecture.seq_len,
            d=architecture.head_dim,
            artifact_prefix=f"encoder_pipeline_layer{layer_idx}",
            seq_tile=architecture.seq_tile,
            kv_seq_tile=architecture.kv_seq_tile,
            emb_tile=architecture.emb_tile,
            ffn_tile=architecture.ffn_tile,
            parallel_seq=architecture.parallel_seq,
            parallel_heads=architecture.parallel_heads,
            proj_acc_depth=architecture.proj_acc_depth,
            o_proj_acc_group_size=architecture.o_proj_acc_group_size,
            ffn_down_acc_group_size=architecture.ffn_down_acc_group_size,
            nB_tiles_distributed=architecture.nB_tiles_distributed,
            ffn_intermediate_size=architecture.intermediate_size,
            static_weights=True,
            ln1_weight=torch.ones(architecture.hidden_size, dtype=dtype),
            ln2_weight=torch.ones(architecture.hidden_size, dtype=dtype),
            context=context,
        )

    def assign_weights(self, layer_weights: EncoderLayerWeights):
        self.qkv_projection.assign_weights(layer_weights, self.dtype)
        self.encoder_pipeline.ln1_weight = layer_weights.ln1_weight.to(
            self.dtype
        ).clone()
        self.encoder_pipeline.ln2_weight = layer_weights.ln2_weight.to(
            self.dtype
        ).clone()
        self.encoder_pipeline.w_o_proj = (
            layer_weights.attn_output_weight.to(self.dtype).contiguous().clone()
        )
        self.encoder_pipeline.weight_up_proj = (
            layer_weights.ffn_up_weight.to(self.dtype).contiguous().clone()
        )
        self.encoder_pipeline.weight_down_proj = (
            layer_weights.ffn_down_weight.to(self.dtype).contiguous().clone()
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        if attention_mask is not None:
            raise RuntimeError(
                "AIETransformerEncoderLayer requires attention_mask=None for encoder_pipeline"
            )

        if hidden_states.ndim == 2:
            batch_dim = False
            hidden_states_2d = hidden_states
            hidden_states_3d = hidden_states.unsqueeze(0)
        elif hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            batch_dim = True
            hidden_states_2d = hidden_states.squeeze(0)
            hidden_states_3d = hidden_states
        else:
            raise RuntimeError(
                "AIETransformerEncoderLayer supports [seq, hidden] or [1, seq, hidden] input"
            )

        query_layer, key_layer, value_layer = self.qkv_projection(hidden_states_3d)
        output = self.encoder_pipeline(
            query_layer.squeeze(0),
            key_layer.squeeze(0),
            value_layer.squeeze(0),
            r=hidden_states_2d,
        )
        return output.unsqueeze(0) if batch_dim else output


class AIEBERTEncoder(nn.Module):
    """General BERT-family encoder stack built from encoder_pipeline layers.

    The operator currently targets BERT-like architectures whose hidden size and
    attention head count imply head_dim=64, matching encoder_pipeline.
    Supported model families for weight resolution are bert, roberta, and
    distilbert.

    By default, each layer gets its own AIEContext. That is the reliable mode on
    the current toolchain when stacking multiple encoder_pipeline operators with
    different compiled layer-norm weights.
    """

    def __init__(
        self,
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
        context=None,
        shared_context: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
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
        self.architecture = architecture
        self.dtype = dtype
        self.shared_context = bool(shared_context)
        self._owned_contexts = []
        self.contexts = []
        layers = []

        for layer_idx in range(architecture.num_hidden_layers):
            layer_context = self._make_layer_context(context)
            layers.append(
                AIETransformerEncoderLayer(
                    architecture=architecture,
                    layer_idx=layer_idx,
                    context=layer_context,
                    dtype=dtype,
                )
            )
            self.contexts.append(layer_context)

        self.layer = nn.ModuleList(layers)
        if not self.layer:
            raise ValueError("AIEBERTEncoder requires num_hidden_layers > 0")
        self.context = self.contexts[0]

    def _make_layer_context(self, context):
        if self.shared_context:
            if context is None:
                context = AIEContext()
                self._owned_contexts = [context]
            elif not self._owned_contexts:
                self._owned_contexts = []
            return context

        if context is None:
            layer_context = AIEContext()
        else:
            layer_context = AIEContext(
                use_runlist=context.use_runlist,
                mlir_verbose=context.mlir_verbose,
            )
        self._owned_contexts.append(layer_context)
        return layer_context

    def compile_all(self):
        seen = set()
        for context in self.contexts:
            if id(context) in seen:
                continue
            context.compile_all()
            seen.add(id(context))

    def prepare_runtime(self):
        seen = set()
        for context in self.contexts:
            if id(context) in seen:
                continue
            context.prepare_runtime()
            seen.add(id(context))

    def cleanup(self):
        for context in self._owned_contexts:
            context.cleanup()

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states

    def assign_weights(self, combined_weights: dict[str, torch.Tensor]):
        for layer_idx, layer in enumerate(self.layer):
            layer.assign_weights(
                resolve_layer_weights(
                    combined_weights,
                    layer_idx=layer_idx,
                    model_type=self.architecture.model_type,
                )
            )


def _resolve_weight(
    combined_weights: dict[str, torch.Tensor], *keys: str
) -> torch.Tensor:
    for key in keys:
        if key in combined_weights:
            return combined_weights[key]
    raise KeyError(f"Missing required encoder weight. Tried: {keys}")


def resolve_layer_weights(
    combined_weights: dict[str, torch.Tensor],
    layer_idx: int,
    model_type: str = "bert",
) -> EncoderLayerWeights:
    family = normalize_model_family(model_type)
    canonical_prefix = f"encoder.layer.{layer_idx}"
    distil_prefix = f"transformer.layer.{layer_idx}"

    def resolve(
        canonical_suffix: str, distil_suffix: str | None = None
    ) -> torch.Tensor:
        candidates = [f"{canonical_prefix}.{canonical_suffix}"]
        if distil_suffix is not None and family == "distilbert":
            candidates.append(f"{distil_prefix}.{distil_suffix}")
        return _resolve_weight(combined_weights, *candidates)

    return EncoderLayerWeights(
        query_weight=resolve("attention.self.query.weight", "attention.q_lin.weight"),
        query_bias=resolve("attention.self.query.bias", "attention.q_lin.bias"),
        key_weight=resolve("attention.self.key.weight", "attention.k_lin.weight"),
        key_bias=resolve("attention.self.key.bias", "attention.k_lin.bias"),
        value_weight=resolve("attention.self.value.weight", "attention.v_lin.weight"),
        value_bias=resolve("attention.self.value.bias", "attention.v_lin.bias"),
        attn_output_weight=resolve(
            "attention.output.dense.weight", "attention.out_lin.weight"
        ),
        ln1_weight=resolve("attention.output.LayerNorm.weight", "sa_layer_norm.weight"),
        ffn_up_weight=resolve("intermediate.dense.weight", "ffn.lin1.weight"),
        ffn_down_weight=resolve("output.dense.weight", "ffn.lin2.weight"),
        ln2_weight=resolve("output.LayerNorm.weight", "output_layer_norm.weight"),
    )
