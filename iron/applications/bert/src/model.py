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

from model_support import model_uses_token_type_ids, normalize_model_family
from .block.transformer import BertEncoder
from .utils import assign


class EncoderEmbeddings(nn.Module):
    """Construct encoder input embeddings for the NPU benchmark."""

    def __init__(self, config):
        super().__init__()
        self.model_family = normalize_model_family(config.model_config.model_type)
        self.padding_idx = int(config.model_config.pad_token_id)
        self.word_embeddings = nn.Embedding(
            config.model_config.vocab_size,
            config.model_config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.position_embeddings = nn.Embedding(
            config.model_config.max_position_embeddings,
            config.model_config.hidden_size,
        )
        self.use_token_type_embeddings = model_uses_token_type_ids(config.model_config)
        if self.use_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(
                config.model_config.type_vocab_size,
                config.model_config.hidden_size,
            )
        else:
            self.token_type_embeddings = None
        self.LayerNorm = nn.LayerNorm(
            config.model_config.hidden_size,
            eps=config.model_config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.model_config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.model_config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        inputs_embeds = self.word_embeddings(input_ids)
        if self.model_family == "roberta":
            mask = input_ids.ne(self.padding_idx).long()
            position_ids = (torch.cumsum(mask, dim=1) * mask) + self.padding_idx
        else:
            position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EncoderBackbone(nn.Module):
    """Minimal encoder backbone used by the NPU benchmark."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.embeddings = EncoderEmbeddings(config)
        self.encoder = BertEncoder(config, seq_len=seq_len)
        self.dtype = config.aie_config.dtype

    def forward_with_stage_timings(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        embedding_start = time.perf_counter()
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        embedding_end = time.perf_counter()
        encoder_output, encoder_timings = self.encoder.forward_with_stage_timings(
            embedding_output.to(self.dtype),
            attention_mask=attention_mask,
        )
        return encoder_output.to(embedding_output.dtype), {
            "embedding_sec": embedding_end - embedding_start,
            **encoder_timings,
        }

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        output, _ = self.forward_with_stage_timings(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return output

    def assign_backbone_weights(self, combined_weights):
        self.embeddings.word_embeddings.weight = assign(
            self.embeddings.word_embeddings.weight,
            combined_weights["embeddings.word_embeddings.weight"],
            "embeddings.word_embeddings.weight",
        )
        self.embeddings.position_embeddings.weight = assign(
            self.embeddings.position_embeddings.weight,
            combined_weights["embeddings.position_embeddings.weight"],
            "embeddings.position_embeddings.weight",
        )
        if self.embeddings.token_type_embeddings is not None:
            self.embeddings.token_type_embeddings.weight = assign(
                self.embeddings.token_type_embeddings.weight,
                combined_weights["embeddings.token_type_embeddings.weight"],
                "embeddings.token_type_embeddings.weight",
            )
        self.embeddings.LayerNorm.bias = assign(
            self.embeddings.LayerNorm.bias,
            combined_weights["embeddings.LayerNorm.bias"],
            "embeddings.LayerNorm.bias",
        )
        self.embeddings.LayerNorm.weight = assign(
            self.embeddings.LayerNorm.weight,
            combined_weights["embeddings.LayerNorm.weight"],
            "embeddings.LayerNorm.weight",
        )
        self.encoder.assign_weights(combined_weights)


BertEmbeddings = EncoderEmbeddings
BertEncoderBackbone = EncoderBackbone
