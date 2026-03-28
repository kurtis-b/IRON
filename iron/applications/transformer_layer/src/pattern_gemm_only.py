# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext
from iron.operators.gemm.op import AIEGEMM

from .input_bundle import TransformerLayerInputs
from .layer_spec import TransformerLayerSpec
from .utils import require_keys


def _layer_norm_no_bias(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=weight,
        bias=None,
        eps=eps,
    )


class GemmOnlyPattern(nn.Module):
    pattern_label = "gemm_offload"

    @staticmethod
    def _resolve_query_block_size(seq_len: int) -> int:
        if seq_len >= 16384:
            return 256
        return seq_len

    def __init__(self, spec: TransformerLayerSpec):
        super().__init__()
        if spec.use_bias:
            raise ValueError(
                "gemm_offload thesis pattern currently requires use_bias=False"
            )
        if spec.torch_dtype != torch.bfloat16:
            raise ValueError(
                "gemm_offload thesis pattern currently requires dtype=bfloat16"
            )

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        head_dim = spec.attention_head_size
        runtime_seq_len = self._resolve_query_block_size(spec.seq_len)
        self.query_block_size = runtime_seq_len
        self.uses_query_blocking = runtime_seq_len < spec.seq_len
        gemm_common = {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }

        self.spec = spec
        self.context = AIEContext(use_runlist=False)
        self._runtime_ready = False
        self._weights_assigned = False
        self.compile_setup_time_sec: float | None = None
        self.q_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self.k_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self.v_proj = AIEGEMM(
            M=spec.seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self.attn_scores = AIEGEMM(
            M=runtime_seq_len,
            K=head_dim,
            N=spec.seq_len,
            force_batched_design=True,
            batch_A=(heads, 1),
            batch_B=(heads, 1),
            batch_C=(heads, 0),
            context=self.context,
            **gemm_common,
        )
        self.attn_output = AIEGEMM(
            M=runtime_seq_len,
            K=spec.seq_len,
            N=head_dim,
            force_batched_design=True,
            batch_A=(heads, 0),
            batch_B=(heads, 1),
            batch_C=(heads, 1),
            context=self.context,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=runtime_seq_len,
            K=hidden,
            N=hidden,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self.ffn_up = AIEGEMM(
            M=runtime_seq_len,
            K=hidden,
            N=spec.intermediate_size,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self.ffn_down = AIEGEMM(
            M=runtime_seq_len,
            K=spec.intermediate_size,
            N=hidden,
            use_static_weight=True,
            force_batched_design=True,
            context=self.context,
            **gemm_common,
        )
        self._bind_shared_gemm_artifacts()
        self.scale = head_dim**-0.5
        self.ln1_weight = nn.Parameter(
            torch.ones(hidden, dtype=spec.torch_dtype), requires_grad=False
        )
        self.ln2_weight = nn.Parameter(
            torch.ones(hidden, dtype=spec.torch_dtype), requires_grad=False
        )

    def _gemm_ops(self) -> list[tuple[str, AIEGEMM]]:
        return [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("attn_scores", self.attn_scores),
            ("attn_output", self.attn_output),
            ("out_proj", self.out_proj),
            ("ffn_up", self.ffn_up),
            ("ffn_down", self.ffn_down),
        ]

    def _artifact_case_prefix(self) -> str:
        return (
            "gemm_offload_case_"
            f"{self.spec.seq_len}x{self.spec.hidden_size}x"
            f"{self.spec.intermediate_size}x{self.spec.num_attention_heads}_"
        )

    def _bind_shared_gemm_artifacts(self) -> None:
        case_prefix = self._artifact_case_prefix()
        shared_xclbin = self.attn_scores.get_runtime_xclbin_artifact(
            prefix=f"{case_prefix}runtime_"
        )
        shared_xclbin.kernel_name = "gemm_offload_runtime"
        for workload_name, gemm_op in self._gemm_ops():
            insts_artifact = gemm_op.get_insts_artifact(
                prefix=f"{case_prefix}{workload_name}_",
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
        self.out_proj.weight = weights["out_proj_weight"].contiguous()
        self.ffn_up.weight = weights["ffn_up_weight"].contiguous()
        self.ffn_down.weight = weights["ffn_down_weight"].contiguous()
        self.ln1_weight.data.copy_(weights["ln1_weight"])
        self.ln2_weight.data.copy_(weights["ln2_weight"])
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
                "gemm_offload thesis pattern currently requires attention_mask=None"
            )
        layer_inputs.validate(self.spec)
        batch_size = layer_inputs.hidden_states.shape[0]
        if batch_size != 1:
            raise RuntimeError(
                "gemm_offload thesis pattern currently supports batch_size=1"
            )
        self._prepare_runtime()

        host_preprocess_start = time.perf_counter()
        residual = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        host_preprocess_end = time.perf_counter()
        npu_projection_total = 0.0
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        projection_start = time.perf_counter()
        q_matrix = self.q_proj(hidden_states).contiguous()
        k_matrix = self.k_proj(hidden_states).contiguous()
        v_matrix = self.v_proj(hidden_states).contiguous()
        npu_projection_total = time.perf_counter() - projection_start
        query_heads = q_matrix.view(
            self.spec.seq_len,
            self.spec.num_attention_heads,
            self.spec.attention_head_size,
        ).contiguous()
        key_heads = (
            k_matrix.view(
                self.spec.seq_len,
                self.spec.num_attention_heads,
                self.spec.attention_head_size,
            )
            .permute(2, 1, 0)
            .contiguous()
        )
        value_heads = v_matrix.view(
            self.spec.seq_len,
            self.spec.num_attention_heads,
            self.spec.attention_head_size,
        ).contiguous()

        npu_gemm_start = time.perf_counter()
        query_block_size = self.query_block_size
        attention_blocks = []
        host_softmax_total = 0.0
        npu_attention_total = 0.0
        for block_start in range(0, self.spec.seq_len, query_block_size):
            block_end = min(block_start + query_block_size, self.spec.seq_len)
            block_query = query_heads[block_start:block_end].contiguous()
            block_npu_start = time.perf_counter()
            attn_scores = self.attn_scores(block_query, key_heads)
            host_softmax_start = time.perf_counter()
            attn_probs = torch.softmax(
                attn_scores.to(torch.float32) * self.scale, dim=-1
            ).to(self.spec.torch_dtype)
            host_softmax_end = time.perf_counter()
            host_softmax_total += host_softmax_end - host_softmax_start
            npu_resume_start = time.perf_counter()
            attn_context = (
                self.attn_output(attn_probs, value_heads)
                .contiguous()
                .view(block_end - block_start, self.spec.hidden_size)
            )
            attention_block = self.out_proj(attn_context)
            attention_blocks.append(attention_block)
            npu_attention_total += (host_softmax_start - block_npu_start) + (
                time.perf_counter() - npu_resume_start
            )
        npu_gemm_after_attention = time.perf_counter()
        host_postprocess_start = npu_gemm_after_attention
        attention_output = torch.cat(attention_blocks, dim=0)
        attention_output = _layer_norm_no_bias(
            attention_output + residual,
            self.ln1_weight,
            self.spec.layer_norm_eps,
        )
        host_after_ln1 = time.perf_counter()
        npu_gemm_ffn_start = host_after_ln1
        output_blocks = []
        for block_start in range(0, self.spec.seq_len, query_block_size):
            block_end = min(block_start + query_block_size, self.spec.seq_len)
            attention_block = attention_output[block_start:block_end]
            ffn_up = self.ffn_up(attention_block)
            ffn_down = self.ffn_down(F.gelu(ffn_up))
            output_blocks.append(ffn_down)
        npu_gemm_end = time.perf_counter()
        host_postprocess_resume_start = npu_gemm_end
        ffn_down = torch.cat(output_blocks, dim=0)
        output = _layer_norm_no_bias(
            ffn_down + attention_output,
            self.ln2_weight,
            self.spec.layer_norm_eps,
        ).unsqueeze(0)
        host_postprocess_end = time.perf_counter()
        return output, {
            "host_preprocess_sec": host_preprocess_end - host_preprocess_start,
            "npu_projection_sec": npu_projection_total,
            "npu_gemm_sec": npu_attention_total + (npu_gemm_end - npu_gemm_ffn_start),
            "host_postprocess_sec": (
                host_softmax_total
                + (host_after_ln1 - host_postprocess_start)
                + (host_postprocess_end - host_postprocess_resume_start)
            ),
            "device_sync_sec": 0.0,
        }

    def get_benchmark_metadata(self) -> dict[str, object]:
        gemm_ops = [gemm_op for _, gemm_op in self._gemm_ops()]
        unique_insts = {
            str(op.insts_artifact.path)
            for op in gemm_ops
            if getattr(op, "insts_artifact", None) is not None
        }
        unique_xclbins = {
            str((op.runtime_xclbin_artifact or op.xclbin_artifact).path)
            for op in gemm_ops
            if (
                getattr(op, "runtime_xclbin_artifact", None) is not None
                or getattr(op, "xclbin_artifact", None) is not None
            )
        }
        block_count = (
            self.spec.seq_len + self.query_block_size - 1
        ) // self.query_block_size
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": 3 + (5 * block_count),
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


GemmOffloadPattern = GemmOnlyPattern
