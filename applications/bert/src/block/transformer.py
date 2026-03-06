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
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference used:
# https://medium.com/@alexmriggio/bert-for-sequence-classification-from-scratch-code-and-theory-fb88053800fa

import torch
import torch.nn as nn
from .feed_forward import BertFeedForward
from .mha import BertAttention
from .mha import BertSelfAttention
from operators import AIEBERTEncoder
from operators import AIEEncoderPipeline
from operators import AIEMHAOutProj


def _cfg_bool(config, name):
    return bool(getattr(config, name, False))


class BertLayer(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.attention = BertAttention(config, seq_len=seq_len)
        self.ffn = BertFeedForward(config, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        attention_mask,
    ) -> tuple[torch.Tensor]:
        self_attention_output, _ = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_output
        ffn_output = self.ffn(attention_output)
        return ffn_output


class BertMhaToAnLayer(nn.Module):
    """BERT layer path using mha_to_an for MHA+O-proj+AddNorm1."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        hidden_size = config.model_config.hidden_size
        num_heads = config.model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        aie_cfg = config.aie_config

        emb_tile = int(getattr(aie_cfg, "mha_to_an_emb_tile", 96))
        o_proj_acc_depth = int(
            getattr(aie_cfg, "mha_to_an_o_proj_acc_depth", hidden_size // emb_tile)
        )
        self.self_attention = BertSelfAttention(config, seq_len=seq_len)
        self.ffn = BertFeedForward(config, seq_len=seq_len)
        self.mha_to_an = AIEMHAOutProj(
            num_heads=num_heads,
            seq_len=seq_len,
            d=head_dim,
            seq_tile=int(getattr(aie_cfg, "mha_to_an_seq_tile", 32)),
            kv_seq_tile=int(getattr(aie_cfg, "mha_to_an_kv_seq_tile", 64)),
            emb_tile=emb_tile,
            parallel_heads=int(getattr(aie_cfg, "mha_to_an_parallel_heads", 1)),
            o_proj_acc_depth=o_proj_acc_depth,
            debug=int(getattr(aie_cfg, "mha_to_an_debug", -1)),
            ln_weight=torch.ones(hidden_size, dtype=config.aie_config.dtype),
        )
        self.attn_output_weight = None

    def forward(self, hidden_states, attention_mask):
        if attention_mask is not None:
            raise RuntimeError("mha_to_an path currently requires attention_mask=None")
        query_layer, key_layer, value_layer = self.self_attention.project_qkv(
            hidden_states
        )
        if query_layer.shape[0] != 1:
            raise RuntimeError("mha_to_an path currently supports batch size 1")
        attention_output = self.mha_to_an(
            query_layer.squeeze(0),
            key_layer.squeeze(0),
            value_layer.squeeze(0),
            r=hidden_states.squeeze(0),
            w_o=self.attn_output_weight,
        ).unsqueeze(0)
        return self.ffn(attention_output)

    def assign_weights(self, l, combined_weights, dtype):
        self.self_attention.assign_weights(
            l,
            combined_weights[f"bert.encoder.layer.{l}.attention.self.query.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.query.bias"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.key.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.key.bias"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.value.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.value.bias"].to(
                dtype
            ),
        )
        self.attn_output_weight = combined_weights[
            f"bert.encoder.layer.{l}.attention.output.dense.weight"
        ].to(dtype)
        self.mha_to_an.ln_weight = combined_weights[
            f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
        ].to(dtype)
        self.ffn.assign_weights(
            l,
            combined_weights[f"bert.encoder.layer.{l}.intermediate.dense.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.intermediate.dense.bias"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.output.dense.weight"].to(dtype),
            combined_weights[f"bert.encoder.layer.{l}.output.dense.bias"].to(dtype),
            combined_weights[f"bert.encoder.layer.{l}.output.LayerNorm.gamma"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.output.LayerNorm.beta"].to(dtype),
        )


class BertEncoderPipelineLayer(nn.Module):
    """BERT layer path using encoder_pipeline for MHA+AN1+FFN+AN2."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        hidden_size = config.model_config.hidden_size
        num_heads = config.model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        aie_cfg = config.aie_config

        emb_tile = int(getattr(aie_cfg, "encoder_pipeline_emb_tile", 96))
        proj_acc_depth = int(
            getattr(aie_cfg, "encoder_pipeline_proj_acc_depth", hidden_size // emb_tile)
        )
        self.self_attention = BertSelfAttention(config, seq_len=seq_len)
        self.encoder_pipeline = AIEEncoderPipeline(
            num_heads=num_heads,
            seq_len=seq_len,
            d=head_dim,
            seq_tile=int(getattr(aie_cfg, "encoder_pipeline_seq_tile", 32)),
            kv_seq_tile=int(getattr(aie_cfg, "encoder_pipeline_kv_seq_tile", 64)),
            emb_tile=emb_tile,
            parallel_heads=int(getattr(aie_cfg, "encoder_pipeline_parallel_heads", 1)),
            proj_acc_depth=proj_acc_depth,
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
            debug=int(getattr(aie_cfg, "encoder_pipeline_debug", -1)),
            ln1_staging_design=getattr(
                aie_cfg, "encoder_pipeline_ln1_staging_design", None
            ),
            ln1_weight=torch.ones(hidden_size, dtype=config.aie_config.dtype),
            ln2_weight=torch.ones(hidden_size, dtype=config.aie_config.dtype),
        )
        self.attn_output_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None

    def forward(self, hidden_states, attention_mask):
        if attention_mask is not None:
            raise RuntimeError(
                "encoder_pipeline path currently requires attention_mask=None"
            )
        query_layer, key_layer, value_layer = self.self_attention.project_qkv(
            hidden_states
        )
        if query_layer.shape[0] != 1:
            raise RuntimeError("encoder_pipeline path currently supports batch size 1")
        return self.encoder_pipeline(
            query_layer.squeeze(0),
            key_layer.squeeze(0),
            value_layer.squeeze(0),
            r=hidden_states.squeeze(0),
            w_o=self.attn_output_weight,
            b_up=self.ffn_up_weight,
            b_down=self.ffn_down_weight,
        ).unsqueeze(0)

    def assign_weights(self, l, combined_weights, dtype):
        self.self_attention.assign_weights(
            l,
            combined_weights[f"bert.encoder.layer.{l}.attention.self.query.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.query.bias"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.key.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.key.bias"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.value.weight"].to(
                dtype
            ),
            combined_weights[f"bert.encoder.layer.{l}.attention.self.value.bias"].to(
                dtype
            ),
        )
        self.attn_output_weight = combined_weights[
            f"bert.encoder.layer.{l}.attention.output.dense.weight"
        ].to(dtype)
        self.ffn_up_weight = combined_weights[
            f"bert.encoder.layer.{l}.intermediate.dense.weight"
        ].to(dtype)
        self.ffn_down_weight = combined_weights[
            f"bert.encoder.layer.{l}.output.dense.weight"
        ].to(dtype)
        self.encoder_pipeline.ln1_weight = combined_weights[
            f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
        ].to(dtype)
        self.encoder_pipeline.ln2_weight = combined_weights[
            f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
        ].to(dtype)


class BertEncoder(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.use_aie_bert_encoder = _cfg_bool(config.aie_config, "use_aie_bert_encoder")
        self.use_aie_mha_to_an = _cfg_bool(config.aie_config, "use_aie_mha_to_an")
        self.use_aie_encoder_pipeline = _cfg_bool(
            config.aie_config, "use_aie_encoder_pipeline"
        )
        self.use_aie_ffn_addnorm = _cfg_bool(
            config.aie_config, "use_aie_ffn_addnorm"
        ) or _cfg_bool(config.aie_config, "use_aie_addnorm_ffn")
        self.config.aie_config.use_aie_addnorm_ffn = self.use_aie_ffn_addnorm

        full_layer_paths_enabled = sum(
            [
                self.use_aie_bert_encoder,
                self.use_aie_mha_to_an,
                self.use_aie_encoder_pipeline,
                self.use_aie_ffn_addnorm,
            ]
        )
        assert (
            full_layer_paths_enabled <= 1
        ), "Select only one of {use_aie_bert_encoder,use_aie_mha_to_an,use_aie_ffn_addnorm,use_aie_encoder_pipeline}."

        offload_individual_operator = [
            config.aie_config.use_aie_gemm == True,
            config.aie_config.use_aie_gelu == True,
            config.aie_config.use_aie_softmax == True,
            config.aie_config.use_aie_layernorm == True,
            config.aie_config.use_aie_elementwise_add == True,
            config.aie_config.use_aie_elementwise_mul == True,
            config.aie_config.use_aie_transpose == True,
            config.aie_config.use_aie_ffn == True,
            config.aie_config.use_aie_addandnorm == True,
        ]
        assert not (
            self.use_aie_bert_encoder or self.use_aie_encoder_pipeline
        ) or not any(
            offload_individual_operator
        ), "Cannot mix full encoder offload with individual AIE operators."
        offload_individual_ffn_operator = [
            # Don't check for aie gemm here since it's used in attention too
            config.aie_config.use_aie_gelu
            == True,
        ]
        assert (
            self.config.aie_config.use_aie_ffn
            and not any(offload_individual_ffn_operator)
            or not self.config.aie_config.use_aie_ffn
        ), "Cannot mix pipelined FFN with individual AIE operators."
        offload_individual_addandnorm_operator = [
            config.aie_config.use_aie_layernorm == True,
            config.aie_config.use_aie_elementwise_add == True,
        ]
        assert (
            self.config.aie_config.use_aie_addandnorm
            and not any(offload_individual_addandnorm_operator)
            or not self.config.aie_config.use_aie_addandnorm
        ), "Cannot mix pipelined Add & Norm with individual AIE operators."
        if self.use_aie_bert_encoder:
            self.layer = [
                AIEBERTEncoder(
                    seq_len=seq_len,
                    hidden_size=config.model_config.hidden_size,
                    intermediate_size=config.model_config.intermediate_size,
                    num_heads=config.model_config.num_attention_heads,
                )
                for _ in range(config.model_config.num_hidden_layers)
            ]
        elif self.use_aie_encoder_pipeline:
            self.layer = nn.ModuleList(
                [
                    BertEncoderPipelineLayer(config, seq_len=seq_len)
                    for _ in range(config.model_config.num_hidden_layers)
                ]
            )
        elif self.use_aie_mha_to_an:
            self.layer = nn.ModuleList(
                [
                    BertMhaToAnLayer(config, seq_len=seq_len)
                    for _ in range(config.model_config.num_hidden_layers)
                ]
            )
        else:
            self.layer = nn.ModuleList(
                [
                    BertLayer(config, seq_len=seq_len)
                    for _ in range(config.model_config.num_hidden_layers)
                ]
            )

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
            )

        return hidden_states

    def assign_weights(self, combined_weights):
        if self.use_aie_bert_encoder:
            for l in range(self.config.model_config.num_hidden_layers):
                self.layer[l].q_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.query.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].k_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.key.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].v_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.self.value.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].attn_output_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.output.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ln1_weight = combined_weights[
                    f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ffn_up_weight = combined_weights[
                    f"bert.encoder.layer.{l}.intermediate.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ffn_down_weight = combined_weights[
                    f"bert.encoder.layer.{l}.output.dense.weight"
                ].to(self.config.aie_config.dtype)
                self.layer[l].ln2_weight = combined_weights[
                    f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                ].to(self.config.aie_config.dtype)
        elif self.use_aie_encoder_pipeline or self.use_aie_mha_to_an:
            for l in range(self.config.model_config.num_hidden_layers):
                self.layer[l].assign_weights(
                    l,
                    combined_weights=combined_weights,
                    dtype=self.config.aie_config.dtype,
                )
        else:
            # Operators here could be offloaded to AIE
            for l in range(self.config.model_config.num_hidden_layers):
                self.layer[l].attention.output.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.weight"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.bias"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                )
                self.layer[l].attention.self.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.query.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.query.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.key.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.key.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.value.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.attention.self.value.bias"
                    ].to(self.config.aie_config.dtype),
                )
                self.layer[l].ffn.assign_weights(
                    l,
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.weight"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[
                        f"bert.encoder.layer.{l}.intermediate.dense.bias"
                    ].to(self.config.aie_config.dtype),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.weight"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[f"bert.encoder.layer.{l}.output.dense.bias"].to(
                        self.config.aie_config.dtype
                    ),
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.gamma"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # weight
                    combined_weights[
                        f"bert.encoder.layer.{l}.output.LayerNorm.beta"
                    ].to(
                        self.config.aie_config.dtype
                    ),  # bias
                )
