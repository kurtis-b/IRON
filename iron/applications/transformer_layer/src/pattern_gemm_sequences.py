# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch

from .input_bundle import TransformerLayerInputs
from .pattern_gemm_only import GemmOffloadPattern


class GemmOffloadGemmSequencePattern(GemmOffloadPattern):
    pattern_label = "gemm_offload_gemm_sequence"

    def __init__(self, spec):
        super().__init__(spec)
        self._cached_attn_probs_blocks: list[torch.Tensor] = []
        self._cached_value_heads: torch.Tensor | None = None

    def prepare_benchmark_inputs(self, layer_inputs: TransformerLayerInputs) -> None:
        super().prepare_benchmark_inputs(layer_inputs)
        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        q_matrix = torch.matmul(hidden_states, self.q_proj.weight.T.contiguous())
        k_matrix = torch.matmul(hidden_states, self.k_proj.weight.T.contiguous())
        v_matrix = torch.matmul(hidden_states, self.v_proj.weight.T.contiguous())
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
        self._cached_value_heads = v_matrix.view(
            self.spec.seq_len,
            self.spec.num_attention_heads,
            self.spec.attention_head_size,
        ).contiguous()
        self._cached_attn_probs_blocks = []
        for block_start in range(0, self.spec.seq_len, self.query_block_size):
            block_end = min(block_start + self.query_block_size, self.spec.seq_len)
            block_query = query_heads[block_start:block_end].contiguous()
            attn_scores = torch.einsum("qhd,dhk->qhk", block_query, key_heads)
            attn_probs = torch.softmax(
                attn_scores.to(torch.float32) * self.scale, dim=-1
            ).to(self.spec.torch_dtype)
            self._cached_attn_probs_blocks.append(
                attn_probs.permute(1, 0, 2).contiguous()
            )

    def forward_with_stage_timings(
        self,
        layer_inputs: TransformerLayerInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if attention_mask is not None:
            raise RuntimeError(
                "gemm sequence studies currently require attention_mask=None"
            )
        if not self._cached_attn_probs_blocks or self._cached_value_heads is None:
            self.prepare_benchmark_inputs(layer_inputs)
        layer_inputs.validate(self.spec)
        self._prepare_runtime()

        hidden_states = layer_inputs.hidden_states.squeeze(0).to(self.spec.torch_dtype)
        start = time.perf_counter()
        q_matrix = self.q_proj(hidden_states).contiguous()
        k_matrix = self.k_proj(hidden_states).contiguous()
        v_matrix = self.v_proj(hidden_states).contiguous()
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

        output_blocks = []
        for block_index, block_start in enumerate(
            range(0, self.spec.seq_len, self.query_block_size)
        ):
            block_end = min(block_start + self.query_block_size, self.spec.seq_len)
            block_query = query_heads[block_start:block_end].contiguous()
            _ = self.attn_scores(block_query, key_heads)
            attn_context = (
                self.attn_output(
                    self._cached_attn_probs_blocks[block_index], value_heads
                )
                .contiguous()
                .view(block_end - block_start, self.spec.hidden_size)
            )
            attention_block = self.out_proj(attn_context)
            ffn_up = self.ffn_up(attention_block)
            output_blocks.append(self.ffn_down(ffn_up))
        end = time.perf_counter()
        return torch.cat(output_blocks, dim=0).unsqueeze(0), {
            "gemm_sequence_sec": end - start
        }


class RunlistGemmSequencePattern(GemmOffloadGemmSequencePattern):
    pattern_label = "runlist_gemm_sequence"

    def _bind_shared_gemm_artifacts(self) -> None:
        case_prefix = self._artifact_case_prefix().replace(
            "gemm_offload_case_", "runlist_gemm_sequence_case_"
        )
        for workload_name, gemm_op in self._gemm_ops():
            xclbin_artifact, insts_artifact = gemm_op.get_artifacts(
                prefix=f"{case_prefix}{workload_name}_"
            )
            xclbin_artifact.kernel_name = f"{workload_name}_kernel"
            insts_artifact.kernel_name = xclbin_artifact.kernel_name
            gemm_op.bind_artifacts(xclbin_artifact, insts_artifact)
