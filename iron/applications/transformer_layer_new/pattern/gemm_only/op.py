# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext, AIEOperatorConstraintError
from iron.operators.gemm.op import AIEGEMM


def default_gemm_only_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
):
    del hidden_size, intermediate_size, num_heads
    if seq_len <= 128:
        tile_m = 16
        tile_k = 64
        tile_n = 8
        num_aie_columns = 8
    else:
        tile_m = 64
        tile_k = 64
        tile_n = 16
    return {
        "shared_gemm": {
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "num_aie_columns": num_aie_columns,
            "b_col_maj": False,
            "c_col_maj": False,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
    }


def resolve_gemm_only_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
    operator_config=None,
):
    resolved = {
        name: dict(config)
        for name, config in default_gemm_only_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            num_aie_columns=num_aie_columns,
        ).items()
    }
    if operator_config is None:
        return resolved
    for name, overrides in operator_config.items():
        if name not in resolved:
            raise ValueError(f"Unsupported gemm_only operator config: {name}")
        resolved[name].update(overrides)
    return resolved


def _layer_norm_no_bias(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=weight,
        bias=None,
    )


class AIETransformerGemmOnly(nn.Module):
    pattern_label = "gemm_only"

    @staticmethod
    def _resolve_attn_scores_partition_n(seq_len: int) -> int:
        if seq_len <= 8192:
            return 1
        if seq_len % 4096 != 0:
            raise AIEOperatorConstraintError(
                "AIETransformerGemmOnly long attention-score partitioning "
                "requires seq_len divisible by 4096; "
                f"got seq_len={seq_len}"
            )
        return seq_len // 4096

    @staticmethod
    def _resolve_query_block_size(seq_len: int) -> int:
        if seq_len >= 16384:
            return 256
        return seq_len

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        *,
        num_aie_columns=8,
        ln1_weight=None,
        ln2_weight=None,
        operator_config=None,
        context=None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attn_output_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.operator_config = resolve_gemm_only_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            num_aie_columns=num_aie_columns,
            operator_config=operator_config,
        )

        self.context = context if context is not None else AIEContext(use_runlist=False)
        self.context.use_runlist = False

        runtime_seq_len = self._resolve_query_block_size(seq_len)
        attn_scores_partition_n = self._resolve_attn_scores_partition_n(seq_len)
        self.query_block_size = runtime_seq_len
        self.uses_query_blocking = runtime_seq_len < seq_len
        gemm_common = dict(self.operator_config["shared_gemm"])
        gemm_common["context"] = self.context

        self.attn_scores = AIEGEMM(
            M=runtime_seq_len,
            K=self.head_dim,
            N=seq_len,
            partition_N=attn_scores_partition_n,
            **gemm_common,
        )
        self.attn_output = AIEGEMM(
            M=runtime_seq_len,
            K=seq_len,
            N=self.head_dim,
            **gemm_common,
        )
        self.out_proj = AIEGEMM(
            M=runtime_seq_len,
            K=hidden_size,
            N=hidden_size,
            use_static_weight=True,
            **gemm_common,
        )
        self.ffn_up = AIEGEMM(
            M=runtime_seq_len,
            K=hidden_size,
            N=intermediate_size,
            use_static_weight=True,
            **gemm_common,
        )
        self.ffn_down = AIEGEMM(
            M=runtime_seq_len,
            K=intermediate_size,
            N=hidden_size,
            use_static_weight=True,
            **gemm_common,
        )

        self.gemm_ops = [
            ("attn_scores", self.attn_scores),
            ("attn_output", self.attn_output),
            ("out_proj", self.out_proj),
            ("ffn_up", self.ffn_up),
            ("ffn_down", self.ffn_down),
        ]

        self.shared_xclbin_artifact = None
        self._artifacts_ready = False
        self._compiled = False
        self._runtime_ready = False
        self.compile_setup_time_sec: float | None = None
        self.scale = self.head_dim**-0.5

    def _artifact_case_prefix(self) -> str:
        return (
            "gemm_only_case_"
            f"{self.seq_len}x{self.hidden_size}x"
            f"{self.intermediate_size}x{self.num_heads}_"
        )

    def _bind_shared_gemm_artifacts(self) -> None:
        case_prefix = self._artifact_case_prefix()
        shared_xclbin = self.out_proj.get_runtime_xclbin_artifact(
            prefix=f"{case_prefix}runtime_"
        )
        shared_xclbin.kernel_name = "gemm_only_runtime"
        self.shared_xclbin_artifact = shared_xclbin
        for workload_name, gemm_op in self.gemm_ops:
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

    def _validate_required_weights(self) -> None:
        required_weights = {
            "attn_output_weight": self.attn_output_weight,
            "ffn_up_weight": self.ffn_up_weight,
            "ffn_down_weight": self.ffn_down_weight,
            "ln1_weight": self.ln1_weight,
            "ln2_weight": self.ln2_weight,
        }
        missing = [name for name, value in required_weights.items() if value is None]
        if missing:
            raise AIEOperatorConstraintError(
                "AIETransformerGemmOnly: missing required weights " + ", ".join(missing)
            )

    def _bind_static_weights(self) -> None:
        self.out_proj.weight = self.attn_output_weight.T.contiguous()
        self.ffn_up.weight = self.ffn_up_weight.T.contiguous()
        self.ffn_down.weight = self.ffn_down_weight.T.contiguous()

    def prepare_runtime(self) -> None:
        self._validate_required_weights()
        if self._runtime_ready:
            return
        if not self._artifacts_ready:
            self._bind_shared_gemm_artifacts()
            self._artifacts_ready = True
        if not self._compiled:
            self._bind_static_weights()
            started = time.perf_counter()
            self.context.compile_all()
            self.context.prepare_runtime()
            self.compile_setup_time_sec = time.perf_counter() - started
            self._compiled = True
        else:
            self.context.prepare_runtime(describe_runtime=False)
        self._runtime_ready = True

    def release_runtime(self) -> None:
        try:
            self.context.reset_runtime()
        except Exception:
            pass
        self._runtime_ready = False

    def invalidate_runtime(self) -> None:
        self.release_runtime()
        self.context.static_data_pool = {}
        for op in self.context.operators:
            op.kernels = {}
            op.buffers = {}
            op.buffer_static_data = {}
            op.buffer_aliases = {}
            op.runlist = []
            op.buffer_bos = {}
            op.xrt_kernels = {}
            op.xrt_runlist = None
        self._compiled = False
        self._artifacts_ready = False
        self._runtime_ready = False

    def get_benchmark_metadata(self) -> dict[str, object]:
        inst_keys = {
            str(gemm_op.insts_artifact.path)
            for _, gemm_op in self.gemm_ops
            if getattr(gemm_op, "insts_artifact", None) is not None
        }
        xclbin_keys = (
            {str(self.shared_xclbin_artifact.path)}
            if self.shared_xclbin_artifact is not None
            else set()
        )
        block_count = (
            self.seq_len + self.query_block_size - 1
        ) // self.query_block_size
        return {
            "compile_setup_time_ms": (
                None
                if self.compile_setup_time_sec is None
                else self.compile_setup_time_sec * 1000.0
            ),
            "npu_dispatch_count": ((2 * self.num_heads) + 3) * block_count,
            "npu_unique_instruction_binary_count": len(inst_keys),
            "npu_unique_xclbin_count": len(xclbin_keys),
            "process_model": "in_process",
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise AIEOperatorConstraintError(
                "AIETransformerGemmOnly currently requires attention_mask=None"
            )
        if not self._runtime_ready:
            raise AIEOperatorConstraintError(
                "AIETransformerGemmOnly: prepare_runtime() must be called before forward"
            )

        expected_qkv = (self.num_heads, self.seq_len, self.head_dim)
        if tuple(q.shape) != expected_qkv:
            raise AIEOperatorConstraintError(
                f"AIETransformerGemmOnly: expected q shape {expected_qkv}"
            )
        if tuple(k.shape) != expected_qkv:
            raise AIEOperatorConstraintError(
                f"AIETransformerGemmOnly: expected k shape {expected_qkv}"
            )
        if tuple(v.shape) != expected_qkv:
            raise AIEOperatorConstraintError(
                f"AIETransformerGemmOnly: expected v shape {expected_qkv}"
            )
        expected_residual = (self.seq_len, self.hidden_size)
        if tuple(residual.shape) != expected_residual:
            raise AIEOperatorConstraintError(
                f"AIETransformerGemmOnly: expected residual shape {expected_residual}"
            )

        query_heads = q.contiguous()
        key_heads = k.transpose(-1, -2).contiguous()
        value_heads = v.contiguous()
        residual = residual.contiguous()

        query_block_size = self.query_block_size
        attention_blocks = []
        for block_start in range(0, self.seq_len, query_block_size):
            block_end = min(block_start + query_block_size, self.seq_len)
            block_query = query_heads[:, block_start:block_end, :].contiguous()
            attn_scores = torch.stack(
                [
                    self.attn_scores(block_query[head], key_heads[head])
                    for head in range(self.num_heads)
                ],
                dim=0,
            )
            attn_probs = torch.softmax(
                attn_scores.to(torch.float32) * self.scale, dim=-1
            ).to(query_heads.dtype)
            attn_context = (
                torch.stack(
                    [
                        self.attn_output(attn_probs[head], value_heads[head])
                        for head in range(self.num_heads)
                    ],
                    dim=1,
                )
                .contiguous()
                .view(block_end - block_start, self.hidden_size)
            )
            attention_blocks.append(self.out_proj(attn_context))

        attention_output = torch.cat(attention_blocks, dim=0)
        attention_output = _layer_norm_no_bias(
            attention_output + residual,
            self.ln1_weight,
        )

        output_blocks = []
        for block_start in range(0, self.seq_len, query_block_size):
            block_end = min(block_start + query_block_size, self.seq_len)
            attention_block = attention_output[block_start:block_end]
            ffn_up = self.ffn_up(attention_block)
            output_blocks.append(self.ffn_down(F.gelu(ffn_up)))

        ffn_down = torch.cat(output_blocks, dim=0)
        output = _layer_norm_no_bias(ffn_down + attention_output, self.ln2_weight)
        return output
