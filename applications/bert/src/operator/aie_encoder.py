# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import numpy as np
from ml_dtypes import bfloat16
import logging

from .aie_base import AIEOperatorBase
from ..compilation import (
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from ..utils import torch_to_numpy, numpy_to_torch
from .aie_gemm import AIEGEMM
from .aie_gelu import AIEGeLU
from .aie_elementwise_add import AIEElementwiseAdd
from .aie_layer_norm import AIELayerNorm
from .aie_softmax import AIESoftmax


class AIEEncoder(AIEOperatorBase):

    def __init__(self, seq_len, embedding_dim, hidden_dim, num_attention_heads):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = embedding_dim // num_attention_heads
        # weights to be set by user
        self.query_w = None
        self.key_w = None
        self.value_w = None
        self.att_out_w = None
        self.att_layernorm_w = None
        self.intermediate_dense_w = None
        self.output_dense_w = None
        self.output_layernorm_w = None

        # TODO: biases to be set by user
        # self.query_b = None
        # self.key_b = None
        # self.value_b = None
        # self.att_out_b = None
        # self.att_layernorm_b = None
        # self.intermediate_dense_b = None
        # self.output_dense_b = None
        # self.output_layernorm_b = None
        super().__init__()

    def set_up(self):
        # Artifact setup
        # ---
        artifacts = []
        device_str = self.device_manager.device_str()

        # GEMM for QKV projections
        gemm_query = AIEGEMM(
            M=self.seq_len,
            K=self.embedding_dim,
            N=self.embedding_dim,
            do_set_up=False,
        )
        gemm_query_xclbin, gemm_query_insts = gemm_query.get_artifacts(prefix="encoder_gemm_q_")
        gemm_query_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gemm_q",
            "--xclbin-kernel-id=0x901",
        ]
        gemm_query_xclbin.kernel_name = "encoder_gemm_q"
        artifacts.append(
            gemm_query_insts
        )  # xclbin artifact will be pulled in as a dependency of last xclbin

        gemm_key = AIEGEMM(
            M=self.seq_len,
            K=self.embedding_dim,
            N=self.embedding_dim,
            do_set_up=False,
        )
        gemm_key_xclbin, gemm_key_insts = gemm_key.get_artifacts(prefix="encoder_gemm_k_")
        gemm_key_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gemm_k",
            "--xclbin-kernel-id=0x902",
        ]
        gemm_key_xclbin.kernel_name = "encoder_gemm_k"
        artifacts.append(
            gemm_key_insts
        )

        gemm_value = AIEGEMM(
            M=self.seq_len,
            K=self.embedding_dim,
            N=self.embedding_dim,
            do_set_up=False,
        )
        gemm_value_xclbin, gemm_value_insts = gemm_value.get_artifacts(prefix="encoder_gemm_v_")
        gemm_value_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gemm_v",
            "--xclbin-kernel-id=0x903",
        ]
        gemm_value_xclbin.kernel_name = "encoder_gemm_v"
        artifacts.append(
            gemm_value_insts
        )

        # GEMM for attention weights and attention scores
        gemm_attn_weights = AIEGEMM(
            M=self.seq_len,
            K=self.head_dim,
            N=self.seq_len,
            do_set_up=False,
        )
        gemm_attn_weights_xclbin, gemm_attn_weights_insts = gemm_attn_weights.get_artifacts(
            prefix="encoder_gemm_attn_weights_"
        )
        gemm_attn_weights_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gemm_attn_weights",
            "--xclbin-kernel-id=0x904",
        ]
        gemm_attn_weights_xclbin.kernel_name = "encoder_gemm_attn_weights"
        gemm_attn_weights_xclbin.depends += [
            gemm_query_xclbin,
            gemm_key_xclbin,
        ]
        artifacts.append(gemm_attn_weights_insts)

        softmax = AIESoftmax(
            size=self.seq_len * self.seq_len,
            num_columns=8,
            num_channels=2,
            tile_size=self.seq_len,
            do_set_up=False,
        )
        softmax_xclbin, softmax_insts = softmax.get_artifacts(prefix="encoder_softmax_")
        softmax_xclbin.xclbin_input = gemm_attn_weights_xclbin
        softmax_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_softmax",
            "--xclbin-kernel-id=0x905",
        ]
        softmax_xclbin.kernel_name = "encoder_softmax"
        softmax_xclbin.depends += [gemm_attn_weights_xclbin]
        artifacts.append(softmax_insts)

        gemm_attn_scores = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.head_dim,
            do_set_up=False,
        )
        gemm_attn_scores_xclbin, gemm_attn_scores_insts = gemm_attn_scores.get_artifacts(
            prefix="encoder_gemm_attn_scores_"
        )
        gemm_attn_scores_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gemm_attn_scores",
            "--xclbin-kernel-id=0x906",
        ]
        gemm_attn_scores_xclbin.kernel_name = "encoder_gemm_attn_scores"
        gemm_attn_scores_xclbin.depends += [softmax_xclbin]
        artifacts.append(gemm_attn_scores_insts)

        silu = AIESiLU(
            size=self.seq_len * self.hidden_dim,
            num_columns=8,
            num_channels=2,
            tile_size=self.hidden_dim,
            do_set_up=False,
        )
        silu_xclbin, silu_insts = silu.get_artifacts(prefix="swiglu_silu_")
        silu_xclbin.xclbin_input = gemm_1_xclbin
        silu_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_silu",
            "--xclbin-kernel-id=0x902",
        ]
        silu_xclbin.kernel_name = "swiglu_silu"
        silu_xclbin.depends += [gemm_1_xclbin]
        artifacts.append(silu_insts)

        eltwise_mul = AIEElementwiseMul(
            size=self.seq_len * self.hidden_dim,
            num_columns=8,
            num_channels=2,
            tile_size=self.hidden_dim,
            do_set_up=False,
        )
        eltwise_mul_xclbin, eltwise_mul_insts = eltwise_mul.get_artifacts(
            prefix="swiglu_eltwise_mul_"
        )
        eltwise_mul_xclbin.xclbin_input = silu_xclbin
        eltwise_mul_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_eltwise_mul",
            "--xclbin-kernel-id=0x903",
        ]
        eltwise_mul_xclbin.kernel_name = "swiglu_eltwise_mul"
        eltwise_mul_xclbin.depends += [silu_xclbin]
        artifacts.append(eltwise_mul_insts)

        gemm_2 = AIEGEMM(
            M=self.seq_len,
            K=self.hidden_dim,
            N=self.embedding_dim,
            do_set_up=False,
        )
        gemm_2_xclbin, gemm_2_insts = gemm_2.get_artifacts(prefix="swiglu_gemm_2_")
        gemm_2_xclbin.xclbin_input = eltwise_mul_xclbin
        gemm_2_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_gemm_2",
            "--xclbin-kernel-id=0x904",
        ]
        gemm_2_xclbin.kernel_name = "swiglu_gemm_2"
        gemm_2_xclbin.depends += [eltwise_mul_xclbin]
        artifacts.append(gemm_2_xclbin)
        artifacts.append(gemm_2_insts)

        self.add_artifacts(artifacts)

        # Runtime setup
        # ---
        combined_xclbin = gemm_2_xclbin
        self.add_buffer("input", self.seq_len * self.embedding_dim)
        self.add_buffer(
            "weights_1",
            self.embedding_dim * self.hidden_dim,
            static_data=torch_to_numpy(self.weights_1.T),
        )
        self.add_buffer(
            "weights_2",
            self.embedding_dim * self.hidden_dim,
            static_data=torch_to_numpy(self.weights_2.T),
        )
        self.add_buffer(
            "weights_3",
            self.hidden_dim * self.embedding_dim,
            static_data=torch_to_numpy(self.weights_3.T),
        )
        self.add_buffer("left", self.seq_len * self.hidden_dim)
        self.add_buffer("left_swished", self.seq_len * self.hidden_dim)
        self.add_buffer("right", self.seq_len * self.hidden_dim)
        self.add_buffer("intermediate", self.seq_len * self.hidden_dim)
        self.add_buffer("output", self.seq_len * self.embedding_dim)
        self.add_kernel(
            "swiglu_gemm_1", combined_xclbin, gemm_1_xclbin.kernel_name, gemm_1_insts
        )
        self.add_kernel(
            "swiglu_silu", combined_xclbin, silu_xclbin.kernel_name, silu_insts
        )
        self.add_kernel(
            "swiglu_eltwise_mul",
            combined_xclbin,
            eltwise_mul_xclbin.kernel_name,
            eltwise_mul_insts,
        )
        self.add_kernel(
            "swiglu_gemm_2", combined_xclbin, gemm_2_xclbin.kernel_name, gemm_2_insts
        )
        self.add_to_runlist("swiglu_gemm_1", "input", "weights_1", "left")
        self.add_to_runlist("swiglu_gemm_1", "input", "weights_2", "right")
        self.add_to_runlist("swiglu_silu", "left", "left_swished")
        self.add_to_runlist(
            "swiglu_eltwise_mul", "left_swished", "right", "intermediate"
        )
        self.add_to_runlist("swiglu_gemm_2", "intermediate", "weights_3", "output")

    def forward(self, x):
        """Forward pass for SwiGLU operation"""

        # Always flatten to [batch, orig_size]
        original_shape = x.shape
        batch = x.shape[0] if x.dim() > 1 else 1
        x_flat = x.reshape(batch, -1)

        out = self._execute_aie_operation(x_flat)

        # Restore original shape
        out = out.reshape(*original_shape)

        return out

    def _execute_aie_operation(self, x):
        # x is [batch, size]
        batch = x.shape[0] if x.dim() > 1 else 1

        # Flatten inputs for AIE processing
        x_flat = x.view(-1)
        x_np = torch_to_numpy(x_flat)

        self.write_buffer("input", x_np)
        test_pattern = np.zeros(len(x_np), dtype=bfloat16)
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=x_np.shape, dtype=bfloat16)

        return result
