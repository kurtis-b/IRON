# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn

from iron.common import AIEContext
from iron.operators.encoder_runlist.op import AIEEncoderRunlist
from iron.operators.gemm.op import AIEGEMM

from ..core.input_bundle import TransformerLayerInputs
from ..core.layer_spec import TransformerLayerSpec
from ..utils import require_keys


class OperatorRunlistPattern(nn.Module):
    pattern_label = "runlist"

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError("runlist thesis pattern currently requires use_bias=False")
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError("runlist thesis pattern currently requires dtype=bfloat16")

        self.spec = spec
        self.context = AIEContext(use_runlist=True)
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        projection_common = {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
        self.q_proj = AIEGEMM(
            M=spec.seq_len,
            K=spec.hidden_size,
            N=spec.hidden_size,
            use_static_weight=True,
            context=self.context,
            **projection_common,
        )
        self.k_proj = AIEGEMM(
            M=spec.seq_len,
            K=spec.hidden_size,
            N=spec.hidden_size,
            use_static_weight=True,
            context=self.context,
            **projection_common,
        )
        self.v_proj = AIEGEMM(
            M=spec.seq_len,
            K=spec.hidden_size,
            N=spec.hidden_size,
            use_static_weight=True,
            context=self.context,
            **projection_common,
        )
        self._bind_projection_artifacts()
        self.encoder_runlist = AIEEncoderRunlist(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_heads=spec.num_attention_heads,
            ln1_weight=torch.ones(spec.hidden_size, dtype=spec.torch_dtype),
            ln2_weight=torch.ones(spec.hidden_size, dtype=spec.torch_dtype),
            context=self.context,
        )

    def _projection_ops(self) -> list[tuple[str, AIEGEMM]]:
        return [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
        ]

    def _bind_projection_artifacts(self) -> None:
        shared_xclbin = self.q_proj.get_runtime_xclbin_artifact(
            prefix=(
                "runlist_projection_"
                f"{self.spec.seq_len}x{self.spec.hidden_size}_runtime_"
            )
        )
        shared_xclbin.kernel_name = "runlist_projection"
        for workload_name, gemm_op in self._projection_ops():
            insts_artifact = gemm_op.get_insts_artifact(
                prefix=(
                    "runlist_projection_"
                    f"{self.spec.seq_len}x{self.spec.hidden_size}_{workload_name}_"
                ),
                xclbin_input=shared_xclbin,
                kernel_name=shared_xclbin.kernel_name,
            )
            gemm_op.bind_artifacts(
                shared_xclbin,
                insts_artifact,
                runtime_xclbin_artifact=shared_xclbin,
                runtime_kernel_name=shared_xclbin.kernel_name,
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
        self.q_proj.weight = weights["q_proj_weight"].contiguous()
        self.k_proj.weight = weights["k_proj_weight"].contiguous()
        self.v_proj.weight = weights["v_proj_weight"].contiguous()
        self.encoder_runlist.assign_weights(
            w_o=weights["out_proj_weight"],
            b_up=weights["ffn_up_weight"],
            b_down=weights["ffn_down_weight"],
            ln1_weight=weights["ln1_weight"],
            ln2_weight=weights["ln2_weight"],
        )
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
                "runlist thesis pattern currently requires attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        batch_size = layer_inputs.hidden_states.shape[0]
        if batch_size != 1:
            raise RuntimeError("runlist thesis pattern currently supports batch_size=1")
        self._prepare_runtime()

        npu_projection_start = time.perf_counter()
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        residual = hidden_states
        q = self.q_proj(hidden_states)[:, : self.spec.hidden_size].contiguous()
        k = self.k_proj(hidden_states)[:, : self.spec.hidden_size].contiguous()
        v = self.v_proj(hidden_states)[:, : self.spec.hidden_size].contiguous()
        npu_projection_end = time.perf_counter()
        runlist_start = time.perf_counter()
        output = self.encoder_runlist(
            q,
            k,
            v,
            residual,
        ).unsqueeze(0)
        runlist_end = time.perf_counter()

        return output, {
            "npu_projection_sec": npu_projection_end - npu_projection_start,
            "runlist_sec": runlist_end - runlist_start,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        unique_insts = {
            str(insts_artifact.path)
            for _, insts_artifact in self.encoder_runlist.component_artifacts.values()
        }
        unique_xclbins = {
            str(xclbin_artifact.path)
            for xclbin_artifact, _ in self.encoder_runlist.component_artifacts.values()
        }
        for _, gemm_op in self._projection_ops():
            unique_insts.add(str(gemm_op.insts_artifact.path))
            unique_xclbins.add(
                str((gemm_op.runtime_xclbin_artifact or gemm_op.xclbin_artifact).path)
            )
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": len(self.encoder_runlist.runlist) + 3,
            "npu_unique_instruction_binary_count": len(unique_insts),
            "npu_unique_xclbin_count": len(unique_xclbins),
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


RunlistPattern = OperatorRunlistPattern
