# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from .layer_spec import TransformerLayerSpec


@dataclass(frozen=True)
class TransformerLayerInputs:
    hidden_states: torch.Tensor | None = None

    def validate(self, spec: TransformerLayerSpec) -> None:
        expected_hidden = (spec.batch_size, spec.seq_len, spec.hidden_size)
        if (
            self.hidden_states is None
            or tuple(self.hidden_states.shape) != expected_hidden
        ):
            raise ValueError(
                f"Expected hidden_states shape {expected_hidden}, got "
                f"{None if self.hidden_states is None else tuple(self.hidden_states.shape)}"
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
            hidden_states=(
                None if self.hidden_states is None else self.hidden_states.to(**kwargs)
            ),
        )
