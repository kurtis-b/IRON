# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
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
    require_keys,
    resolve_mha_out_proj_topology,
)


class DataflowPattern(nn.Module):
    pattern_label = "dataflow"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "dataflow thesis pattern currently requires use_bias=False"
            )
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError(
                "dataflow thesis pattern currently requires dtype=bfloat16"
            )
        if spec.attention_head_size != 64:
            raise ValueError("dataflow thesis pattern currently supports head_dim=64")
        block2_topology = resolve_mha_out_proj_topology(spec)
        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        self.block1 = AIEQKVProj(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            num_heads=spec.num_attention_heads,
            context=self.context,
        )
        self.block2 = AIEMHAOutProj(
            num_heads=spec.num_attention_heads,
            seq_len=spec.seq_len,
            d=spec.attention_head_size,
            topology_id=str(block2_topology["topology_id"]),
            static_weights=True,
            context=self.context,
        )
        self.block3 = AIEAddNormFFNAddNorm(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
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
        require_keys(
            weights,
            [
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "out_proj_weight",
                "ffn_up_weight",
                "ffn_down_weight",
                "ln2_weight",
            ],
        )
        bind_qkv_proj_weights(self.block1, weights)
        bind_mha_out_proj_weights(self.block2, weights)
        bind_addnorm_ffn_addnorm_weights(self.block3, weights)
        self._weights_assigned = True

    def prepare_benchmark_inputs(self, layer_inputs: TransformerLayerInputs) -> None:
        layer_inputs.validate(self.spec)

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "dataflow thesis pattern currently requires attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        if layer_inputs.hidden_states.shape[0] != 1:
            raise RuntimeError(
                "dataflow thesis pattern currently supports batch_size=1"
            )
        self._prepare_runtime()

        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        block1_start = time.perf_counter()
        q, k, v = self.block1.forward(hidden_states)
        block1_end = time.perf_counter()

        block2_start = block1_end
        attention_output = self.block2(q, k, v)
        block2_end = time.perf_counter()

        block3_start = block2_end
        output = self.block3.forward(attention_output, hidden_states)
        block3_end = time.perf_counter()

        return output.unsqueeze(0), {
            "block1_qkv_proj_sec": block1_end - block1_start,
            "block2_mha_out_proj_sec": block2_end - block2_start,
            "block3_addnorm_ffn_addnorm_sec": block3_end - block3_start,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        unique_insts = set()
        unique_xclbins = set()

        def maybe_add_artifact_path(
            artifact_set: set[str],
            artifact,
        ) -> None:
            if artifact is not None and getattr(artifact, "path", None) is not None:
                artifact_set.add(str(artifact.path))

        for gemm_op in (self.block1.q_proj, self.block1.k_proj, self.block1.v_proj):
            maybe_add_artifact_path(unique_insts, gemm_op.insts_artifact)
            maybe_add_artifact_path(
                unique_xclbins,
                gemm_op.runtime_xclbin_artifact or gemm_op.xclbin_artifact,
            )
        maybe_add_artifact_path(unique_insts, self.block2.insts_artifact)
        maybe_add_artifact_path(unique_xclbins, self.block2.xclbin_artifact)
        maybe_add_artifact_path(unique_insts, self.block3.block.insts_artifact)
        maybe_add_artifact_path(unique_xclbins, self.block3.block.xclbin_artifact)
        return make_in_process_npu_metadata(
            compile_setup_time_sec=self.compile_setup_time_sec,
            dispatch_count=3
            + len(self.block2.runlist)
            + len(self.block3.block.runlist),
            unique_instruction_binary_count=len(unique_insts),
            unique_xclbin_count=len(unique_xclbins),
            extra_fields={
                "block1_topology_id": self.block1.topology_id,
                "block1_topology_family": self.block1.topology_family,
                "block2_topology_id": self.block2.topology_id,
                "block2_topology_family": self.block2.topology_family,
                "block3_topology_id": self.block3.topology_id,
                "block3_topology_family": self.block3.topology_family,
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
