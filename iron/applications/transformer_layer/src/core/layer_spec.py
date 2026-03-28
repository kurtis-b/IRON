# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass

import torch

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class TransformerLayerSpec:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    batch_size: int = 1
    seq_len: int = 128
    dtype: str = "bfloat16"
    activation: str = "gelu"
    use_bias: bool = False
    layer_norm_eps: float = 1.0e-12
    attention_mask_mode: str = "none"
    weights_source: str = "synthetic"
    source_model_name: str | None = None
    source_layer_index: int | None = None

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.activation != "gelu":
            raise ValueError("Only gelu is currently supported")
        if self.attention_mask_mode not in {"none"}:
            raise ValueError("Only attention_mask_mode='none' is currently supported")
        if self.weights_source not in {"synthetic", "imported"}:
            raise ValueError("weights_source must be synthetic or imported")
        if self.weights_source == "synthetic":
            if (
                self.source_model_name is not None
                or self.source_layer_index is not None
            ):
                raise ValueError(
                    "Synthetic weights_source cannot include source_model_name/source_layer_index"
                )
        else:
            if self.source_model_name is None or self.source_layer_index is None:
                raise ValueError(
                    "Imported weights_source requires source_model_name and source_layer_index"
                )

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_BY_NAME[self.dtype]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TransformerLayerSpec":
        return cls(**payload)
