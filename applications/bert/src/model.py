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
from .block.transformer import BertEncoder


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.model_config.vocab_size,
            config.model_config.hidden_size,
            padding_idx=config.model_config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.model_config.max_position_embeddings, config.model_config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.model_config.type_vocab_size, config.model_config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(
            config.model_config.hidden_size, eps=config.model_config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.model_config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.model_config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.model_config.hidden_size, config.model_config.hidden_size
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, seq_len=seq_len)
        self.pooler = BertPooler(config)
        self.dtype = config.aie_config.dtype

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        encoder_outputs = self.encoder(
            embedding_output.to(self.dtype), attention_mask=attention_mask
        ).to(embedding_output.dtype)
        pooled_output = self.pooler(encoder_outputs)

        return encoder_outputs, pooled_output


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, seq_len=512):
        super().__init__()
        self.config = config

        self.bert = BertModel(config, seq_len=seq_len)
        self.dropout = nn.Dropout(config.model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.model_config.hidden_size, config.model_config.num_labels
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
    ):
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
