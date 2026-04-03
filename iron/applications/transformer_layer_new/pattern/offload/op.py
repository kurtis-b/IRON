# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from iron.common import AIEContext, AIEOperatorConstraintError
from iron.operators.gemm.op import AIEGEMM


def default_offload_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
):
    del seq_len, hidden_size, intermediate_size, num_heads
    return {
        "shared_gemm": {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": num_aie_columns,
            "b_col_maj": False,
            "c_col_maj": False,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
    }


def resolve_offload_operator_config(
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
        for name, config in default_offload_operator_config(
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
            raise ValueError(f"Unsupported offload operator config: {name}")
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


class AIETransformerOffload(nn.Module):
    """Transformer layer with all GEMM stages offloaded to AIE and non-GEMM stages on host."""

    pattern_label = "offload"

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
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
        self.num_aie_columns = num_aie_columns
        self.head_dim = hidden_size // num_heads

        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.operator_config = resolve_offload_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            num_aie_columns=num_aie_columns,
            operator_config=operator_config,
        )

        self.context = context if context is not None else AIEContext(use_runlist=False)
        self.context.use_runlist = False
        self.attn_context = AIEContext(use_runlist=False)
        self.attn_context.use_runlist = False
        self.attn_context.build_dir = self.context.build_dir
        self.post_context = AIEContext(use_runlist=False)
        self.post_context.use_runlist = False
        self.post_context.build_dir = self.context.build_dir

        self.shared_xclbin_artifact = None
        self.compile_setup_time_sec = None
        self._artifacts_ready = False

        hidden_gemm_args = dict(self.operator_config["shared_gemm"])
        hidden_gemm_args["context"] = self.context
        attn_gemm_args = dict(self.operator_config["shared_gemm"])
        attn_gemm_args["context"] = self.attn_context
        post_gemm_args = dict(self.operator_config["shared_gemm"])
        post_gemm_args["context"] = self.post_context

        self.q_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **hidden_gemm_args,
        )
        self.k_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **hidden_gemm_args,
        )
        self.v_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **hidden_gemm_args,
        )
        self.attn_scores_gemm = AIEGEMM(
            M=self.seq_len,
            K=self.head_dim,
            N=self.seq_len,
            **attn_gemm_args,
        )
        self.attn_output_gemm = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.head_dim,
            **attn_gemm_args,
        )
        self.o_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **post_gemm_args,
        )
        self.ffn_up_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.intermediate_size,
            use_static_weight=True,
            **post_gemm_args,
        )
        self.ffn_down_proj = AIEGEMM(
            M=self.seq_len,
            K=self.intermediate_size,
            N=self.hidden_size,
            use_static_weight=True,
            **post_gemm_args,
        )

        self.gemm_ops = [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("attn_scores", self.attn_scores_gemm),
            ("attn_output", self.attn_output_gemm),
            ("o_proj", self.o_proj),
            ("ffn_up", self.ffn_up_proj),
            ("ffn_down", self.ffn_down_proj),
        ]

    def _all_contexts(self):
        return (self.context, self.attn_context, self.post_context)

    def _artifact_case_prefix(self) -> str:
        return (
            "encoder_offload_case_"
            f"{self.seq_len}x{self.hidden_size}x"
            f"{self.intermediate_size}x{self.num_heads}_"
        )

    def _shared_runtime_dims(self) -> tuple[int, int, int]:
        config = self.operator_config["shared_gemm"]
        tile_m = int(config["tile_m"])
        tile_k = int(config["tile_k"])
        tile_n = int(config["tile_n"])
        num_cols = int(config["num_aie_columns"])
        return tile_m * 4, tile_k, tile_n * num_cols

    def _bind_shared_gemm_artifacts(self) -> None:
        case_prefix = self._artifact_case_prefix()
        if self.shared_xclbin_artifact is None:
            runtime_M, runtime_K, runtime_N = self._shared_runtime_dims()
            shared_builder = AIEGEMM(
                M=runtime_M,
                K=runtime_K,
                N=runtime_N,
                context=self.context,
                skip_add_to_list=True,
                **self.operator_config["shared_gemm"],
            )
            shared_xclbin, _ = shared_builder.get_artifacts(
                prefix=f"{case_prefix}runtime_"
            )
            self.shared_xclbin_artifact = shared_xclbin

        for workload_name, gemm_op in self.gemm_ops:
            prefix = f"{case_prefix}{workload_name}_"

            def _set_up_shared_artifacts(
                op, *, _prefix=prefix, _shared=self.shared_xclbin_artifact
            ):
                if op.xclbin_artifact is _shared and op.insts_artifact is not None:
                    return
                _, insts_artifact = op.get_artifacts(prefix=_prefix)
                op.xclbin_artifact = _shared
                op.insts_artifact = insts_artifact
                op.add_artifacts([_shared, insts_artifact])

            gemm_op.set_up_artifacts = MethodType(_set_up_shared_artifacts, gemm_op)

    def _validate_required_weights(self) -> None:
        required_weights = {
            "q_weight": self.q_weight,
            "k_weight": self.k_weight,
            "v_weight": self.v_weight,
            "attn_output_weight": self.attn_output_weight,
            "ffn_up_weight": self.ffn_up_weight,
            "ffn_down_weight": self.ffn_down_weight,
            "ln1_weight": self.ln1_weight,
            "ln2_weight": self.ln2_weight,
        }
        missing = [name for name, value in required_weights.items() if value is None]
        if missing:
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: missing required weights " + ", ".join(missing)
            )

    def prepare_runtime(self):
        if self._artifacts_ready:
            return

        self._validate_required_weights()
        self._bind_shared_gemm_artifacts()

        self.q_proj.weight = self.q_weight.T.contiguous()
        self.k_proj.weight = self.k_weight.T.contiguous()
        self.v_proj.weight = self.v_weight.T.contiguous()
        self.o_proj.weight = self.attn_output_weight.T.contiguous()
        self.ffn_up_proj.weight = self.ffn_up_weight.T.contiguous()
        self.ffn_down_proj.weight = self.ffn_down_weight.T.contiguous()

        compile_started = time.perf_counter()
        for context in self._all_contexts():
            context.compile_all()
        self.compile_setup_time_sec = time.perf_counter() - compile_started
        self._artifacts_ready = True

    @staticmethod
    def _clear_runtime_descriptors(context):
        for op in context.operators:
            op.kernels = {}
            op.buffers = {}
            op.buffer_static_data = {}
            op.buffer_aliases = {}
            op.runlist = []
            op.buffer_bos = {}
            op.xrt_kernels = {}
            op.xrt_runlist = None

    def _run_stage(self, context, stage_fn):
        context.prepare_runtime()
        try:
            return stage_fn()
        finally:
            context.reset_runtime()
            self._clear_runtime_descriptors(context)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            del attention_mask

        if tuple(x.shape) != (self.seq_len, self.hidden_size):
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: expected input shape "
                f"{(self.seq_len, self.hidden_size)}"
            )

        self.prepare_runtime()

        def run_qkv():
            return self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q, k, v = self._run_stage(self.context, run_qkv)

        q_heads = q.view(self.seq_len, self.num_heads, self.head_dim)
        k_heads = k.view(self.seq_len, self.num_heads, self.head_dim)
        v_heads = v.view(self.seq_len, self.num_heads, self.head_dim)

        scale = self.head_dim**-0.5

        def run_attention():
            attention_heads = []
            for head in range(self.num_heads):
                q_head = q_heads[:, head, :]
                k_head_t = k_heads[:, head, :].transpose(0, 1).contiguous()
                v_head = v_heads[:, head, :]

                attn_scores = self.attn_scores_gemm(q_head, k_head_t)
                attn_probs = torch.softmax(
                    attn_scores.to(torch.float32) * scale, dim=-1
                ).to(q.dtype)
                attention_heads.append(self.attn_output_gemm(attn_probs, v_head))
            return attention_heads

        attention_heads = self._run_stage(self.attn_context, run_attention)

        attn_output = (
            torch.stack(attention_heads, dim=1)
            .contiguous()
            .view(self.seq_len, self.hidden_size)
        )

        def run_post():
            projected = self.o_proj(attn_output)
            hidden_states = _layer_norm_no_bias(projected + x, self.ln1_weight)
            ffn_up = self.ffn_up_proj(hidden_states)
            ffn_down = self.ffn_down_proj(F.gelu(ffn_up))
            return _layer_norm_no_bias(ffn_down + hidden_states, self.ln2_weight)

        return self._run_stage(self.post_context, run_post)
