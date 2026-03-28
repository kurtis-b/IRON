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

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from .utils import require_keys, select_mha_out_proj_emb_tile


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
        mha_emb_tile = select_mha_out_proj_emb_tile(spec.hidden_size)
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
            emb_tile=mha_emb_tile,
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
        self.block1.q_proj.weight = weights["q_proj_weight"].contiguous()
        self.block1.k_proj.weight = weights["k_proj_weight"].contiguous()
        self.block1.v_proj.weight = weights["v_proj_weight"].contiguous()
        self.block2.w_o_proj = weights["out_proj_weight"].contiguous()
        self.block3.block.weight_up_proj = weights["ffn_up_weight"].contiguous()
        self.block3.block.weight_down_proj = weights["ffn_down_weight"].contiguous()
        self.block3.block.ln2_weight = weights["ln2_weight"].contiguous()
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
        for gemm_op in (self.block1.q_proj, self.block1.k_proj, self.block1.v_proj):
            unique_insts.add(str(gemm_op.insts_artifact.path))
            unique_xclbins.add(
                str((gemm_op.runtime_xclbin_artifact or gemm_op.xclbin_artifact).path)
            )
        unique_insts.add(str(self.block2.insts_artifact.path))
        unique_xclbins.add(str(self.block2.xclbin_artifact.path))
        unique_insts.add(str(self.block3.block.insts_artifact.path))
        unique_xclbins.add(str(self.block3.block.xclbin_artifact.path))
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": 3
            + len(self.block2.runlist)
            + len(self.block3.block.runlist),
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
            layer_inputs,
            attention_mask=attention_mask,
        )
        return output
