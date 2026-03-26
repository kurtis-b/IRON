# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.operators.encoder_runlist.op import AIEEncoderRunlist

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from .utils import require_keys


class OperatorRunlistPattern(nn.Module):
    pattern_label = "operator_runlist"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "operator_runlist thesis pattern currently requires use_bias=False"
            )
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError(
                "operator_runlist thesis pattern currently requires dtype=bfloat16"
            )

        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        self.encoder_runlist = AIEEncoderRunlist(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_heads=spec.num_attention_heads,
            ln1_weight=torch.ones(spec.hidden_size, dtype=spec.torch_dtype),
            ln2_weight=torch.ones(spec.hidden_size, dtype=spec.torch_dtype),
            context=self.context,
        )

    def _prepare_runtime(self) -> None:
        if self._runtime_ready:
            return
        if not self._weights_assigned:
            raise RuntimeError("assign_weights() must be called before execution")
        start = time.perf_counter()
        self.context.compile_all()
        self.context.prepare_runtime()
        self.compile_setup_time_sec = time.perf_counter() - start
        self._runtime_ready = True

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if self._runtime_ready:
            raise RuntimeError(
                "assign_weights() after runtime preparation is not supported"
            )
        require_keys(
            weights,
            [
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln1_weight",
                "ln2_weight",
            ],
        )
        self.encoder_runlist.assign_weights(
            w_o=weights["out_proj_weight"],
            b_up=weights["ffn_up_weight"],
            b_down=weights["ffn_down_weight"],
            ln1_weight=weights["ln1_weight"],
            ln2_weight=weights["ln2_weight"],
        )
        self._weights_assigned = True

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "operator_runlist thesis pattern currently requires attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        if layer_inputs.q.shape[0] != 1:
            raise RuntimeError(
                "operator_runlist thesis pattern currently supports batch_size=1"
            )
        self._prepare_runtime()

        operator_runlist_start = time.perf_counter()
        output = self.encoder_runlist(
            layer_inputs.q.squeeze(0),
            layer_inputs.k.squeeze(0),
            layer_inputs.v.squeeze(0),
            layer_inputs.r.squeeze(0),
        ).unsqueeze(0)
        operator_runlist_end = time.perf_counter()

        return output, {
            "operator_runlist_sec": operator_runlist_end - operator_runlist_start,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        unique_insts = {
            str(insts_artifact.path)
            for _, insts_artifact in self.encoder_runlist.component_artifacts.values()
        }
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": len(self.encoder_runlist.runlist),
            "npu_unique_instruction_binary_count": len(unique_insts),
            "process_model": "in_process",
        }

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            layer_inputs, attention_mask=attention_mask
        )
        return output
