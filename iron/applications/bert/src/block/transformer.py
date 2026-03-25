# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import torch
import torch.nn as nn

from iron.operators.encoder_pipeline.op import AIEEncoderPipeline
from ..utils import assign

combine_qkv_projection_parameters = AIEEncoderPipeline.combine_qkv_projection_parameters


class BertQKVProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.model_config.hidden_size
        dtype = config.aie_config.dtype
        self.num_attention_heads = config.model_config.num_attention_heads
        if hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.query = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, hidden_states):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        return query_layer, key_layer, value_layer

    def assign_weights(self, layer_idx, combined_weights, dtype):
        self.query.weight = assign(
            self.query.weight,
            combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.query.weight"
            ].to(dtype),
            f"encoder.layer.{layer_idx}.attention.self.query.weight",
        )
        self.query.bias = assign(
            self.query.bias,
            combined_weights[f"encoder.layer.{layer_idx}.attention.self.query.bias"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.attention.self.query.bias",
        )
        self.key.weight = assign(
            self.key.weight,
            combined_weights[f"encoder.layer.{layer_idx}.attention.self.key.weight"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.attention.self.key.weight",
        )
        self.key.bias = assign(
            self.key.bias,
            combined_weights[f"encoder.layer.{layer_idx}.attention.self.key.bias"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.attention.self.key.bias",
        )
        self.value.weight = assign(
            self.value.weight,
            combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.value.weight"
            ].to(dtype),
            f"encoder.layer.{layer_idx}.attention.self.value.weight",
        )
        self.value.bias = assign(
            self.value.bias,
            combined_weights[f"encoder.layer.{layer_idx}.attention.self.value.bias"].to(
                dtype
            ),
            f"encoder.layer.{layer_idx}.attention.self.value.bias",
        )


class BertEncoderPipelineLayer(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        hidden_size = config.model_config.hidden_size
        num_heads = config.model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        aie_cfg = config.aie_config
        emb_tile = int(getattr(aie_cfg, "encoder_pipeline_emb_tile", 96))
        ffn_tile = int(getattr(aie_cfg, "encoder_pipeline_ffn_tile", 64))
        proj_acc_depth = int(
            getattr(aie_cfg, "encoder_pipeline_proj_acc_depth", hidden_size // emb_tile)
        )
        self.qkv_projection_mode = str(
            getattr(aie_cfg, "encoder_pipeline_qkv_projection_mode", "split_host")
        )
        if self.qkv_projection_mode not in {
            "split_host",
            "packed_host",
            "staged_npu",
        }:
            raise ValueError(
                "encoder_pipeline_qkv_projection_mode must be one of "
                f"{{'split_host', 'packed_host', 'staged_npu'}} "
                f"(got {self.qkv_projection_mode!r})"
            )

        self.qkv_projection = (
            BertQKVProjection(config)
            if self.qkv_projection_mode == "split_host"
            else None
        )
        self.encoder_pipeline = AIEEncoderPipeline(
            num_heads=num_heads,
            seq_len=seq_len,
            d=head_dim,
            seq_tile=int(getattr(aie_cfg, "encoder_pipeline_seq_tile", 32)),
            kv_seq_tile=int(getattr(aie_cfg, "encoder_pipeline_kv_seq_tile", 64)),
            emb_tile=emb_tile,
            ffn_tile=ffn_tile,
            parallel_seq=int(getattr(aie_cfg, "encoder_pipeline_parallel_seq", 1)),
            parallel_heads=int(getattr(aie_cfg, "encoder_pipeline_parallel_heads", 1)),
            proj_acc_depth=proj_acc_depth,
            o_proj_acc_group_size=int(
                getattr(aie_cfg, "encoder_pipeline_o_proj_acc_group_size", 1)
            ),
            nB_tiles_distributed=int(
                getattr(aie_cfg, "encoder_pipeline_parallel_ffn", 1)
            ),
            ffn_intermediate_size=int(
                getattr(
                    aie_cfg,
                    "encoder_pipeline_ffn_intermediate_size",
                    config.model_config.intermediate_size,
                )
            ),
            weight_forward_depth=getattr(
                aie_cfg, "encoder_pipeline_weight_forward_depth", None
            ),
            o_proj_fifo_depth=getattr(
                aie_cfg, "encoder_pipeline_o_proj_fifo_depth", None
            ),
            ffn_replay_fifo_depth=getattr(
                aie_cfg, "encoder_pipeline_ffn_replay_fifo_depth", None
            ),
            use_fused_replayed_addnorm=getattr(
                aie_cfg, "encoder_pipeline_use_fused_replayed_addnorm", None
            ),
            use_transport_groups=getattr(
                aie_cfg, "encoder_pipeline_use_transport_groups", None
            ),
            use_unified_qr_split=getattr(
                aie_cfg, "encoder_pipeline_use_unified_qr_split", None
            ),
            qkv_projection_mode=(
                "staged_hidden_states"
                if self.qkv_projection_mode == "staged_npu"
                else "packed_input"
            ),
            static_weights=True,
            ln1_weight=torch.ones(hidden_size, dtype=config.aie_config.dtype),
            ln2_weight=torch.ones(hidden_size, dtype=config.aie_config.dtype),
        )
        self.attn_output_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            raise RuntimeError(
                "encoder_pipeline benchmark path currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError("encoder_pipeline benchmark path supports batch size 1")
        hidden_states_2d = hidden_states.squeeze(0)

        qkv_start = time.perf_counter()
        if self.qkv_projection_mode == "packed_host":
            packed_qkv = self.encoder_pipeline.project_hidden_states_to_packed_qkv(
                hidden_states_2d
            )
        elif self.qkv_projection_mode == "split_host":
            query_layer, key_layer, value_layer = self.qkv_projection(hidden_states)
        qkv_end = time.perf_counter()

        pipeline_start = time.perf_counter()
        if self.qkv_projection_mode == "packed_host":
            output = self.encoder_pipeline.forward_packed_qkv(
                packed_qkv,
                r=hidden_states_2d,
            ).unsqueeze(0)
        elif self.qkv_projection_mode == "staged_npu":
            output = self.encoder_pipeline.forward_hidden_states(
                hidden_states_2d
            ).unsqueeze(0)
        else:
            output = self.encoder_pipeline(
                query_layer.squeeze(0),
                key_layer.squeeze(0),
                value_layer.squeeze(0),
                r=hidden_states_2d,
            ).unsqueeze(0)
        pipeline_end = time.perf_counter()
        return output, {
            "qkv_projection_sec": qkv_end - qkv_start,
            "encoder_pipeline_sec": pipeline_end - pipeline_start,
        }

    def _forward_output_only(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            raise RuntimeError(
                "encoder_pipeline benchmark path currently requires attention_mask=None"
            )
        if hidden_states.shape[0] != 1:
            raise RuntimeError("encoder_pipeline benchmark path supports batch size 1")
        hidden_states_2d = hidden_states.squeeze(0)
        if self.qkv_projection_mode == "split_host":
            query_layer, key_layer, value_layer = self.qkv_projection(hidden_states)
            return self.encoder_pipeline.forward_split_qkv(
                query_layer.squeeze(0),
                key_layer.squeeze(0),
                value_layer.squeeze(0),
                r=hidden_states_2d,
            ).unsqueeze(0)
        if self.qkv_projection_mode == "staged_npu":
            return self.encoder_pipeline.forward_hidden_states(
                hidden_states_2d
            ).unsqueeze(0)
        return self.encoder_pipeline.forward_hidden_states(
            hidden_states_2d,
            r=hidden_states_2d,
        ).unsqueeze(0)

    def forward(self, hidden_states, attention_mask=None):
        return self._forward_output_only(
            hidden_states,
            attention_mask=attention_mask,
        )

    def assign_weights(self, layer_idx, combined_weights, dtype):
        if self.qkv_projection_mode == "split_host":
            self.qkv_projection.assign_weights(layer_idx, combined_weights, dtype)
        elif self.qkv_projection_mode == "packed_host":
            q_weight = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.query.weight"
            ].to(dtype)
            k_weight = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.key.weight"
            ].to(dtype)
            v_weight = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.value.weight"
            ].to(dtype)
            q_bias = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.query.bias"
            ].to(dtype)
            k_bias = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.key.bias"
            ].to(dtype)
            v_bias = combined_weights[
                f"encoder.layer.{layer_idx}.attention.self.value.bias"
            ].to(dtype)
            combined_qkv_weight, combined_qkv_bias = combine_qkv_projection_parameters(
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
            )
            self.encoder_pipeline.qkv_proj_weight = combined_qkv_weight
            self.encoder_pipeline.qkv_proj_bias = combined_qkv_bias
        else:
            self.encoder_pipeline.q_proj_weight = (
                combined_weights[
                    f"encoder.layer.{layer_idx}.attention.self.query.weight"
                ]
                .to(dtype)
                .T.contiguous()
            )
            self.encoder_pipeline.k_proj_weight = (
                combined_weights[f"encoder.layer.{layer_idx}.attention.self.key.weight"]
                .to(dtype)
                .T.contiguous()
            )
            self.encoder_pipeline.v_proj_weight = (
                combined_weights[
                    f"encoder.layer.{layer_idx}.attention.self.value.weight"
                ]
                .to(dtype)
                .T.contiguous()
            )
            self.encoder_pipeline.q_proj_bias = (
                combined_weights[f"encoder.layer.{layer_idx}.attention.self.query.bias"]
                .to(dtype)
                .contiguous()
            )
            self.encoder_pipeline.k_proj_bias = (
                combined_weights[f"encoder.layer.{layer_idx}.attention.self.key.bias"]
                .to(dtype)
                .contiguous()
            )
            self.encoder_pipeline.v_proj_bias = (
                combined_weights[f"encoder.layer.{layer_idx}.attention.self.value.bias"]
                .to(dtype)
                .contiguous()
            )
        self.attn_output_weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.output.dense.weight"
        ].to(dtype)
        self.ffn_up_weight = combined_weights[
            f"encoder.layer.{layer_idx}.intermediate.dense.weight"
        ].to(dtype)
        self.ffn_down_weight = combined_weights[
            f"encoder.layer.{layer_idx}.output.dense.weight"
        ].to(dtype)
        self.encoder_pipeline.ln1_weight = combined_weights[
            f"encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"
        ].to(dtype)
        self.encoder_pipeline.ln2_weight = combined_weights[
            f"encoder.layer.{layer_idx}.output.LayerNorm.weight"
        ].to(dtype)
        self.encoder_pipeline.w_o_proj = self.attn_output_weight.contiguous()
        self.encoder_pipeline.weight_up_proj = self.ffn_up_weight.contiguous()
        self.encoder_pipeline.weight_down_proj = self.ffn_down_weight.contiguous()


class BertEncoder(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                BertEncoderPipelineLayer(config, seq_len=seq_len)
                for _ in range(config.model_config.num_hidden_layers)
            ]
        )

    def forward_with_stage_timings(self, hidden_states, attention_mask=None):
        qkv_projection_sec = 0.0
        encoder_pipeline_sec = 0.0
        for layer_module in self.layer:
            hidden_states, stage_timings = layer_module.forward_with_stage_timings(
                hidden_states,
                attention_mask,
            )
            qkv_projection_sec += stage_timings["qkv_projection_sec"]
            encoder_pipeline_sec += stage_timings["encoder_pipeline_sec"]
        return hidden_states, {
            "qkv_projection_sec": qkv_projection_sec,
            "encoder_pipeline_sec": encoder_pipeline_sec,
        }

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

    def assign_weights(self, combined_weights):
        dtype = self.config.aie_config.dtype
        for layer_idx, layer in enumerate(self.layer):
            layer.assign_weights(layer_idx, combined_weights, dtype)
