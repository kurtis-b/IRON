# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from .layer_spec import TransformerLayerSpec


@dataclass(frozen=True)
class TransformerLayerInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor

    def validate(self, spec: TransformerLayerSpec) -> None:
        expected_qkv = (
            spec.batch_size,
            spec.num_attention_heads,
            spec.seq_len,
            spec.attention_head_size,
        )
        expected_r = (spec.batch_size, spec.seq_len, spec.hidden_size)
        if tuple(self.q.shape) != expected_qkv:
            raise ValueError(
                f"Expected q shape {expected_qkv}, got {tuple(self.q.shape)}"
            )
        if tuple(self.k.shape) != expected_qkv:
            raise ValueError(
                f"Expected k shape {expected_qkv}, got {tuple(self.k.shape)}"
            )
        if tuple(self.v.shape) != expected_qkv:
            raise ValueError(
                f"Expected v shape {expected_qkv}, got {tuple(self.v.shape)}"
            )
        if tuple(self.r.shape) != expected_r:
            raise ValueError(
                f"Expected r shape {expected_r}, got {tuple(self.r.shape)}"
            )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TransformerLayerInputs":
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return TransformerLayerInputs(
            q=self.q.to(**kwargs),
            k=self.k.to(**kwargs),
            v=self.v.to(**kwargs),
            r=self.r.to(**kwargs),
        )
