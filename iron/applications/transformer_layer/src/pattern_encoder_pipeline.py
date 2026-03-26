# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline

from .layer_spec import TransformerLayerSpec
from .utils import require_keys


class EncoderPipelinePattern(nn.Module):
    """Single-layer encoder_pipeline wrapper with layer-centric inputs."""

    pattern_label = "encoder_pipeline"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "encoder_pipeline thesis pattern currently requires use_bias=False"
            )
        hidden = spec.hidden_size
        dtype = spec.torch_dtype
        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self._weights_assigned = False
        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.encoder_pipeline = AIEEncoderPipeline(
            num_heads=spec.num_attention_heads,
            seq_len=spec.seq_len,
            d=spec.attention_head_size,
            seq_tile=32,
            kv_seq_tile=64,
            emb_tile=96,
            ffn_tile=64,
            parallel_seq=1,
            parallel_heads=1,
            proj_acc_depth=max(1, hidden // 96),
            o_proj_acc_group_size=1,
            nB_tiles_distributed=1,
            ffn_intermediate_size=spec.intermediate_size,
            static_weights=True,
            ln1_weight=torch.ones(hidden, dtype=dtype),
            ln2_weight=torch.ones(hidden, dtype=dtype),
            context=self.context,
        )

    def _prepare_runtime(self) -> None:
        if self._runtime_ready:
            return
        if not self._weights_assigned:
            raise RuntimeError("assign_weights() must be called before execution")
        self.context.compile_all()
        self.context.prepare_runtime()
        self._runtime_ready = True

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if self._runtime_ready:
            raise RuntimeError(
                "assign_weights() after runtime preparation is not supported"
            )
        require_keys(
            weights,
            [
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln1_weight",
                "ln2_weight",
            ],
        )
        with torch.no_grad():
            self.q_proj.weight.copy_(weights["q_proj_weight"])
            self.k_proj.weight.copy_(weights["k_proj_weight"])
            self.v_proj.weight.copy_(weights["v_proj_weight"])
        self.encoder_pipeline.w_o_proj = weights["out_proj_weight"].contiguous()
        self.encoder_pipeline.weight_up_proj = weights["ffn_up_weight"].contiguous()
        self.encoder_pipeline.weight_down_proj = weights["ffn_down_weight"].contiguous()
        self.encoder_pipeline.ln1_weight = weights["ln1_weight"].contiguous()
        self.encoder_pipeline.ln2_weight = weights["ln2_weight"].contiguous()
        self._weights_assigned = True

    def forward_with_stage_timings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "encoder_pipeline thesis pattern currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError(
                "encoder_pipeline thesis pattern currently supports batch_size=1"
            )
        self._prepare_runtime()

        qkv_start = time.perf_counter()
        batch, seq_len, _ = hidden_states.shape
        heads = self.spec.num_attention_heads
        head_dim = self.spec.attention_head_size
        query = (
            self.q_proj(hidden_states)
            .view(batch, seq_len, heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, seq_len, heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, seq_len, heads, head_dim)
            .transpose(1, 2)
        )
        qkv_end = time.perf_counter()

        start = time.perf_counter()
        output = self.encoder_pipeline(
            query.squeeze(0),
            key.squeeze(0),
            value.squeeze(0),
            r=hidden_states.squeeze(0),
        ).unsqueeze(0)
        end = time.perf_counter()
        return output, {
            "qkv_projection_sec": qkv_end - qkv_start,
            "pattern_sec": end - start,
        }

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            hidden_states, attention_mask=attention_mask
        )
        return output
