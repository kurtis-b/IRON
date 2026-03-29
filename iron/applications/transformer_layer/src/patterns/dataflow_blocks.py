# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.common.utils import numpy_to_torch, torch_to_numpy
from iron.operators.addnorm_ffn_addnorm.op import AIEAddNormFFNAddNorm
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.qkv_proj.op import AIEQKVProj

from ..core.input_bundle import TransformerLayerInputs
from ..core.layer_spec import TransformerLayerSpec
from ..utils import (
    bind_addnorm_ffn_addnorm_weights,
    bind_mha_out_proj_weights,
    bind_qkv_proj_weights,
    make_in_process_npu_metadata,
    host_attention_output,
    host_project_qkv_head_major,
    require_keys,
)


class _BaseBlockPattern(nn.Module):
    def __init__(self, spec: TransformerLayerSpec) -> None:
        super().__init__()
        if spec.use_bias:
            raise ValueError("Dataflow block patterns currently require use_bias=False")
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError("Dataflow block patterns currently require dtype=bfloat16")
        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self.compile_setup_time_sec: float | None = None

    def _prepare_runtime(self) -> None:
        if self._runtime_ready:
            return
        start = time.perf_counter()
        self.context.compile_all()
        self.context.prepare_runtime()
        self.compile_setup_time_sec = time.perf_counter() - start
        self._runtime_ready = True


class Block1QKVProjPattern(_BaseBlockPattern):
    pattern_label = "block1_qkv_proj"

    def __init__(self, spec: TransformerLayerSpec) -> None:
        super().__init__(spec)
        self.block = AIEQKVProj(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            num_heads=spec.num_attention_heads,
            topology_id=spec.block1_topology_id,
            context=self.context,
        )

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        require_keys(weights, ["q_proj_weight", "k_proj_weight", "v_proj_weight"])
        bind_qkv_proj_weights(self.block, weights)

    def prepare_benchmark_inputs(self, layer_inputs: TransformerLayerInputs) -> None:
        layer_inputs.validate(self.spec)

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "Dataflow block studies currently require attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        self._prepare_runtime()
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        start = time.perf_counter()
        q, k, v = self.block.forward(hidden_states)
        end = time.perf_counter()
        return torch.stack((q, k, v), dim=0), {"block1_qkv_proj_sec": end - start}

    def get_benchmark_metadata(self) -> dict[str, object]:
        return make_in_process_npu_metadata(
            compile_setup_time_sec=self.compile_setup_time_sec,
            dispatch_count=1,
            unique_instruction_binary_count=1,
            unique_xclbin_count=1,
            extra_fields={
                "block1_topology_id": self.block.topology_id,
                "block1_topology_family": self.block.topology_family,
            },
        )

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            layer_inputs,
            attention_mask=attention_mask,
        )
        return output


class Block2MHAOutProjPattern(_BaseBlockPattern):
    pattern_label = "block2_mha_out_proj"

    def __init__(self, spec: TransformerLayerSpec) -> None:
        super().__init__(spec)
        if spec.attention_head_size != 64:
            raise ValueError("Block 2 currently supports attention_head_size=64 only")
        self.block = AIEMHAOutProj(
            num_heads=spec.num_attention_heads,
            seq_len=spec.seq_len,
            d=spec.attention_head_size,
            topology_id=spec.block2_topology_id,
            static_weights=True,
            context=self.context,
        )
        self._weights: dict[str, torch.Tensor] | None = None
        self._cached_q: torch.Tensor | None = None
        self._cached_k: torch.Tensor | None = None
        self._cached_v: torch.Tensor | None = None

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        require_keys(
            weights,
            ["q_proj_weight", "k_proj_weight", "v_proj_weight", "out_proj_weight"],
        )
        self._weights = weights
        bind_mha_out_proj_weights(self.block, weights)

    def prepare_benchmark_inputs(self, layer_inputs: TransformerLayerInputs) -> None:
        layer_inputs.validate(self.spec)
        if self._weights is None:
            raise RuntimeError("assign_weights() must be called before execution")
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        self._cached_q, self._cached_k, self._cached_v = host_project_qkv_head_major(
            hidden_states, self._weights, self.spec
        )

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "Dataflow block studies currently require attention_mask=None"
            )
        if self._cached_q is None or self._cached_k is None or self._cached_v is None:
            self.prepare_benchmark_inputs(layer_inputs)
        self._prepare_runtime()
        start = time.perf_counter()
        output = self.block(self._cached_q, self._cached_k, self._cached_v)
        end = time.perf_counter()
        return output.unsqueeze(0), {"block2_mha_out_proj_sec": end - start}

    def get_benchmark_metadata(self) -> dict[str, object]:
        return make_in_process_npu_metadata(
            compile_setup_time_sec=self.compile_setup_time_sec,
            dispatch_count=len(self.block.runlist),
            unique_instruction_binary_count=1,
            unique_xclbin_count=1,
            extra_fields={
                "block2_topology_id": self.block.topology_id,
                "block2_topology_family": self.block.topology_family,
            },
        )

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            layer_inputs,
            attention_mask=attention_mask,
        )
        return output


class Block3AddNormFFNAddNormPattern(_BaseBlockPattern):
    pattern_label = "block3_addnorm_ffn_addnorm"

    def __init__(self, spec: TransformerLayerSpec) -> None:
        super().__init__(spec)
        self.block = AIEAddNormFFNAddNorm(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            topology_id=spec.block3_topology_id,
            context=self.context,
        )
        self._weights: dict[str, torch.Tensor] | None = None
        self._cached_packed_hidden_residual: torch.Tensor | None = None

    def assign_weights(self, weights: dict[str, torch.Tensor]) -> None:
        require_keys(
            weights,
            [
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln1_weight",
                "ln2_weight",
            ],
        )
        self._weights = weights
        bind_addnorm_ffn_addnorm_weights(self.block, weights)

    def prepare_benchmark_inputs(self, layer_inputs: TransformerLayerInputs) -> None:
        layer_inputs.validate(self.spec)
        if self._weights is None:
            raise RuntimeError("assign_weights() must be called before execution")
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        attention_output = host_attention_output(
            hidden_states, self._weights, self.spec
        )
        padded_rows = (
            (attention_output.shape[0] + self.block.M - 1) // self.block.M
        ) * self.block.M
        attention_output_padded = torch.zeros(
            (padded_rows, attention_output.shape[1]),
            dtype=attention_output.dtype,
        )
        hidden_states_padded = torch.zeros(
            (padded_rows, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
        )
        attention_output_padded[: attention_output.shape[0], :] = attention_output
        hidden_states_padded[: hidden_states.shape[0], :] = hidden_states
        self._cached_packed_hidden_residual = numpy_to_torch(
            self.block._pack_hidden_residual(
                torch_to_numpy(attention_output_padded),
                torch_to_numpy(hidden_states_padded),
            )
        )

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "Dataflow block studies currently require attention_mask=None"
            )
        if self._cached_packed_hidden_residual is None:
            self.prepare_benchmark_inputs(layer_inputs)
        self._prepare_runtime()
        start = time.perf_counter()
        output = self.block.forward_packed(self._cached_packed_hidden_residual)
        end = time.perf_counter()
        return output.unsqueeze(0), {"block3_addnorm_ffn_addnorm_sec": end - start}

    def get_benchmark_metadata(self) -> dict[str, object]:
        return make_in_process_npu_metadata(
            compile_setup_time_sec=self.compile_setup_time_sec,
            dispatch_count=len(self.block.runlist),
            unique_instruction_binary_count=1,
            unique_xclbin_count=1,
            extra_fields={
                "block3_topology_id": self.block.topology_id,
                "block3_topology_family": self.block.topology_family,
            },
        )

    def forward(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_stage_timings(
            layer_inputs,
            attention_mask=attention_mask,
        )
        return output
