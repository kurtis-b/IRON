# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from ..utils import require_keys


class ReferenceTransformerLayer(nn.Module):
    """Single encoder-style transformer layer used as correctness reference."""

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_size
        intermediate = spec.intermediate_size
        dtype = spec.torch_dtype
        self.q_proj = nn.Linear(hidden, hidden, bias=spec.use_bias, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, bias=spec.use_bias, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, bias=spec.use_bias, dtype=dtype)
        self.out_proj = nn.Linear(hidden, hidden, bias=spec.use_bias, dtype=dtype)
        self.ffn_up = nn.Linear(hidden, intermediate, bias=spec.use_bias, dtype=dtype)
        self.ffn_down = nn.Linear(intermediate, hidden, bias=spec.use_bias, dtype=dtype)
        self.ln1 = nn.LayerNorm(
            hidden, eps=spec.layer_norm_eps, elementwise_affine=True
        )
        self.ln2 = nn.LayerNorm(
            hidden, eps=spec.layer_norm_eps, elementwise_affine=True
        )
        self.ln1.to(dtype=dtype)
        self.ln2.to(dtype=dtype)

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        required = [
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "out_proj_weight",
            "ffn_up_weight",
            "ffn_down_weight",
            "ln1_weight",
            "ln2_weight",
        ]
        if self.spec.use_bias:
            required.extend(
                [
                    "q_proj_bias",
                    "k_proj_bias",
                    "v_proj_bias",
                    "out_proj_bias",
                    "ffn_up_bias",
                    "ffn_down_bias",
                ]
            )
        require_keys(weights, required)

        with torch.no_grad():
            self.q_proj.weight.copy_(weights["q_proj_weight"])
            self.k_proj.weight.copy_(weights["k_proj_weight"])
            self.v_proj.weight.copy_(weights["v_proj_weight"])
            self.out_proj.weight.copy_(weights["out_proj_weight"])
            self.ffn_up.weight.copy_(weights["ffn_up_weight"])
            self.ffn_down.weight.copy_(weights["ffn_down_weight"])
            self.ln1.weight.copy_(weights["ln1_weight"])
            self.ln2.weight.copy_(weights["ln2_weight"])
            if self.spec.use_bias:
                self.q_proj.bias.copy_(weights["q_proj_bias"])
                self.k_proj.bias.copy_(weights["k_proj_bias"])
                self.v_proj.bias.copy_(weights["v_proj_bias"])
                self.out_proj.bias.copy_(weights["out_proj_bias"])
                self.ffn_up.bias.copy_(weights["ffn_up_bias"])
                self.ffn_down.bias.copy_(weights["ffn_down_bias"])
            else:
                self.ln1.bias.zero_()
                self.ln2.bias.zero_()

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise RuntimeError(
                "attention_mask is not implemented in the initial thesis app"
            )

        layer_inputs.validate(self.spec)
        batch = self.spec.batch_size
        seq_len = self.spec.seq_len
        hidden = self.spec.hidden_size
        head_dim = self.spec.attention_head_size
        hidden_states = layer_inputs.hidden_states
        q = self.q_proj(hidden_states).view(
            batch,
            seq_len,
            self.spec.num_attention_heads,
            head_dim,
        )
        k = self.k_proj(hidden_states).view(
            batch,
            seq_len,
            self.spec.num_attention_heads,
            head_dim,
        )
        v = self.v_proj(hidden_states).view(
            batch,
            seq_len,
            self.spec.num_attention_heads,
            head_dim,
        )
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        residual = hidden_states

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores * (head_dim**-0.5)
        attn_probs = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(q.dtype)
        attn_context = torch.matmul(attn_probs, v)
        attn_context = (
            attn_context.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        )

        attention_output = self.out_proj(attn_context)
        attention_output = self.ln1(attention_output + residual)
        ffn_hidden = F.gelu(self.ffn_up(attention_output))
        return self.ln2(self.ffn_down(ffn_hidden) + attention_output)
