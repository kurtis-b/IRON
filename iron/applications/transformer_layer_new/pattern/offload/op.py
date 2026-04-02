# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from iron.common import AIEOperatorBase, AIEOperatorConstraintError
from iron.operators.gemm.op import AIEGEMM


class AIETransformerOffload(AIEOperatorBase):
    """Transformer layer with GEMM stages offloaded to AIE and non-GEMM stages on host."""

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=8,
        ln1_weight=None,
        ln2_weight=None,
        context=None,
    ):
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

        self.shared_xclbin_artifact = None
        self.shared_kernel_name = "encoder_offload_gemm"

        AIEOperatorBase.__init__(self, context=context)

        common_gemm_args = {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": self.num_aie_columns,
            "b_col_maj": False,
            "c_col_maj": False,
            "partition_N": 1,
            "emulate_bf16_mmul_with_bfp16": True,
            "context": self.context,
        }

        self.q_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **common_gemm_args,
        )
        self.k_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **common_gemm_args,
        )
        self.v_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **common_gemm_args,
        )
        self.attn_scores_gemm = AIEGEMM(
            M=self.seq_len,
            K=self.head_dim,
            N=self.seq_len,
            **common_gemm_args,
        )
        self.attn_output_gemm = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.head_dim,
            **common_gemm_args,
        )
        self.o_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            use_static_weight=True,
            **common_gemm_args,
        )
        self.ffn_up_proj = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_size,
            N=self.intermediate_size,
            use_static_weight=True,
            **common_gemm_args,
        )
        self.ffn_down_proj = AIEGEMM(
            M=self.seq_len,
            K=self.intermediate_size,
            N=self.hidden_size,
            use_static_weight=True,
            **common_gemm_args,
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

    def set_up_artifacts(self):
        if self.shared_xclbin_artifact is None:
            shared_builder = AIEGEMM(
                M=self.seq_len,
                K=self.hidden_size,
                N=self.hidden_size,
                tile_m=64,
                tile_k=64,
                tile_n=16,
                num_aie_columns=self.num_aie_columns,
                b_col_maj=False,
                c_col_maj=False,
                partition_N=1,
                emulate_bf16_mmul_with_bfp16=True,
                context=self.context,
                skip_add_to_list=True,
            )
            self.shared_xclbin_artifact = shared_builder.get_runtime_xclbin_artifact(
                prefix="encoder_offload_runtime_"
            )
            self.shared_xclbin_artifact.kernel_name = self.shared_kernel_name

        for name, gemm_op in self.gemm_ops:
            if (
                gemm_op.runtime_xclbin_artifact is self.shared_xclbin_artifact
                and gemm_op.insts_artifact is not None
            ):
                continue
            insts_artifact = gemm_op.get_insts_artifact(
                prefix=f"encoder_offload_{name}_",
                xclbin_input=self.shared_xclbin_artifact,
                kernel_name=self.shared_kernel_name,
            )
            gemm_op.bind_artifacts(
                self.shared_xclbin_artifact,
                insts_artifact,
                runtime_xclbin_artifact=self.shared_xclbin_artifact,
                runtime_kernel_name=self.shared_kernel_name,
            )

    def set_up_runtime(self):
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

        self.q_proj.weight = self.q_weight.T.contiguous()
        self.k_proj.weight = self.k_weight.T.contiguous()
        self.v_proj.weight = self.v_weight.T.contiguous()
        self.o_proj.weight = self.attn_output_weight.T.contiguous()
        self.ffn_up_proj.weight = self.ffn_up_weight.T.contiguous()
        self.ffn_down_proj.weight = self.ffn_down_weight.T.contiguous()

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            del attention_mask

        if tuple(x.shape) != (self.seq_len, self.hidden_size):
            raise AIEOperatorConstraintError(
                "AIETransformerOffload: expected input shape "
                f"{(self.seq_len, self.hidden_size)}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = q.view(self.seq_len, self.num_heads, self.head_dim)
        k_heads = k.view(self.seq_len, self.num_heads, self.head_dim)
        v_heads = v.view(self.seq_len, self.num_heads, self.head_dim)

        scale = math.sqrt(self.head_dim)
        attn_head_outputs = []
        for head in range(self.num_heads):
            q_head = q_heads[:, head, :]
            k_head_t = k_heads[:, head, :].transpose(0, 1).contiguous()
            v_head = v_heads[:, head, :]

            attn_scores = self.attn_scores_gemm(q_head, k_head_t)
            attn_probs = torch.nn.functional.softmax(attn_scores / scale, dim=-1)
            attn_head_outputs.append(self.attn_output_gemm(attn_probs, v_head))

        attn_output = (
            torch.stack(attn_head_outputs, dim=1)
            .contiguous()
            .view(self.seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        ln1_bias = torch.zeros_like(self.ln1_weight)
        hidden_states = torch.nn.functional.layer_norm(
            attn_output + x,
            (self.hidden_size,),
            self.ln1_weight,
            ln1_bias,
        )

        intermediate = self.ffn_up_proj(hidden_states)
        intermediate = torch.nn.functional.gelu(intermediate)
        ffn_output = self.ffn_down_proj(intermediate)

        ln2_bias = torch.zeros_like(self.ln2_weight)
        output = torch.nn.functional.layer_norm(
            ffn_output + hidden_states,
            (self.hidden_size,),
            self.ln2_weight,
            ln2_bias,
        )

        return output
