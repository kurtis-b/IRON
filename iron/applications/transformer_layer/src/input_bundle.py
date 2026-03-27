# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from .layer_spec import TransformerLayerSpec


@dataclass(frozen=True)
class TransformerLayerInputs:
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    r: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None

    def validate(self, spec: TransformerLayerSpec) -> None:
        expected_r = (spec.batch_size, spec.seq_len, spec.hidden_size)
        if self.r is None or tuple(self.r.shape) != expected_r:
            raise ValueError(
                f"Expected r shape {expected_r}, got "
                f"{None if self.r is None else tuple(self.r.shape)}"
            )
        if spec.input_boundary == "hidden_states":
            if (
                self.hidden_states is None
                or tuple(self.hidden_states.shape) != expected_r
            ):
                raise ValueError(
                    f"Expected hidden_states shape {expected_r}, got "
                    f"{None if self.hidden_states is None else tuple(self.hidden_states.shape)}"
                )
            return

        expected_qkv = (
            spec.batch_size,
            spec.num_attention_heads,
            spec.seq_len,
            spec.attention_head_size,
        )
        if self.q is None or tuple(self.q.shape) != expected_qkv:
            raise ValueError(
                f"Expected q shape {expected_qkv}, got "
                f"{None if self.q is None else tuple(self.q.shape)}"
            )
        if self.k is None or tuple(self.k.shape) != expected_qkv:
            raise ValueError(
                f"Expected k shape {expected_qkv}, got "
                f"{None if self.k is None else tuple(self.k.shape)}"
            )
        if self.v is None or tuple(self.v.shape) != expected_qkv:
            raise ValueError(
                f"Expected v shape {expected_qkv}, got "
                f"{None if self.v is None else tuple(self.v.shape)}"
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
            q=None if self.q is None else self.q.to(**kwargs),
            k=None if self.k is None else self.k.to(**kwargs),
            v=None if self.v is None else self.v.to(**kwargs),
            r=None if self.r is None else self.r.to(**kwargs),
            hidden_states=(
                None if self.hidden_states is None else self.hidden_states.to(**kwargs)
            ),
        )
