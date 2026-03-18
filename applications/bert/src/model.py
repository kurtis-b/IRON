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

import torch
import torch.nn as nn

from .block.transformer import BertEncoder
from .utils import assign


class BertEmbeddings(nn.Module):
    """Construct BERT input embeddings for the NPU encoder benchmark."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.model_config.vocab_size,
            config.model_config.hidden_size,
            padding_idx=config.model_config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.model_config.max_position_embeddings,
            config.model_config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.model_config.type_vocab_size,
            config.model_config.hidden_size,
        )
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

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoderBackbone(nn.Module):
    """Minimal BERT backbone used by the NPU encoder benchmark."""

    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, seq_len=seq_len)
        self.dtype = config.aie_config.dtype

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        return self.encoder(
            embedding_output.to(self.dtype),
            attention_mask=attention_mask,
        ).to(embedding_output.dtype)

    def assign_backbone_weights(self, combined_weights):
        self.embeddings.word_embeddings.weight = assign(
            self.embeddings.word_embeddings.weight,
            combined_weights["bert.embeddings.word_embeddings.weight"],
            "bert.embeddings.word_embeddings.weight",
        )
        self.embeddings.position_embeddings.weight = assign(
            self.embeddings.position_embeddings.weight,
            combined_weights["bert.embeddings.position_embeddings.weight"],
            "bert.embeddings.position_embeddings.weight",
        )
        self.embeddings.token_type_embeddings.weight = assign(
            self.embeddings.token_type_embeddings.weight,
            combined_weights["bert.embeddings.token_type_embeddings.weight"],
            "bert.embeddings.token_type_embeddings.weight",
        )
        self.embeddings.LayerNorm.bias = assign(
            self.embeddings.LayerNorm.bias,
            combined_weights["bert.embeddings.LayerNorm.beta"],
            "bert.embeddings.LayerNorm.beta",
        )
        self.embeddings.LayerNorm.weight = assign(
            self.embeddings.LayerNorm.weight,
            combined_weights["bert.embeddings.LayerNorm.gamma"],
            "bert.embeddings.LayerNorm.gamma",
        )
        self.encoder.assign_weights(combined_weights)
