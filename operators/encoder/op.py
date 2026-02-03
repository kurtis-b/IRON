# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import numpy as np
from ml_dtypes import bfloat16
import math

from operators.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from operators.gemm.op import AIEGEMM
from operators.softmax.op import AIESoftmax
from operators.elementwise_mul.op import AIEElementwiseMul
from operators.layer_norm.op import AIELayerNorm
from operators.elementwise_add.op import AIEElementwiseAdd
from operators.gelu.op import AIEGELU
from operators.transpose.op import AIETranspose
from operators.ffn.op import AIEFFN
from operators.addnorm.op import AIEAddAndNorm
from operators.mha.op import AIEMHA
from operators.addnorm_ffn.op import AIEANFFN
from operators.common.utils import torch_to_numpy


class AIEBERTEncoder(AIEOperatorBase):
    """
    AIE-accelerated BERT Encoder Layer using runlist-based implementation.

    A BERT Encoder Layer consists of:
    1. Q/K/V projection (GEMM with Q/K/V weights)
    2. K transpose (for attention score calculations)
    3. Attention score calculations (GEMM per Q/K heads)
    4. Attention score scaling (Multiplication per attention score)
    5. Attention weight calculations (Softmax per attention score)
    6. Output head calculations (GEMM per attention weights/V heads)
    7. Output projection (GEMM with O weight)
    8. Layer normalization
    9. Residual connection (Eltwise add)
    10. Up projection (GEMM with Up projection weight)
    11. Activation function (GeLU)
    12. Down projection (GEMM with Down projection weight)
    13. Layer normalization
    14. Residual connection (Eltwise add)
    """

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=8,
        use_pip_ffn=False,
        use_pip_addnorm=False,
        use_pip_mha=True,
        use_pip_an_ffn=True,
        ln1_weight=None,
        ln2_weight=None,
        context=None,
    ):
        if use_pip_an_ffn and use_pip_ffn:
            raise ValueError(
                "Pipelined Add & Norm -> FFN -> Add & Norm cannot be used with standalone pipelined FFN"
            )
        if use_pip_an_ffn and use_pip_addnorm:
            raise ValueError(
                "Pipelined Add & Norm -> FFN -> Add & Norm cannot be used with standalone pipelined Add & Norm"
            )
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_aie_columns = num_aie_columns  # NOTE: This value isn't used for GEMMs to generate the output heads since N=64 there

        # Derived dimensions
        self.head_dim = hidden_size // num_heads

        # Weights to be set by user (separate Q/K/V weights)
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ln1_weight = ln1_weight
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln2_weight = ln2_weight
        self.use_pip_ffn = use_pip_ffn
        self.use_pip_addnorm = use_pip_addnorm
        self.use_pip_mha = use_pip_mha
        self.use_pip_an_ffn = use_pip_an_ffn

        # Artifacts created by set_up_artifacts() - one per layer
        self.combined_xclbin = None
        # Q/K/V/O projections
        self.qkvo_proj_xclbin = None
        self.qkvo_proj_insts = None
        # Attention
        if self.use_pip_mha:
            self.mha_xclbin = None
            self.mha_insts = None
        else:
            self.k_transpose_xclbin = None
            self.k_transpose_insts = None
            self.attn_scores_xclbin = None
            self.attn_scores_insts = None
            self.attn_scale_xclbin = None
            self.attn_scale_insts = None
            self.attn_softmax_xclbin = None
            self.attn_softmax_insts = None
            self.attn_output_xclbin = None
            self.attn_output_insts = None
        if self.use_pip_an_ffn:
            self.anffn_xclbin = None
            self.anffn_insts = None
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm
                self.add_norm1_xclbin = None
                self.add_norm1_insts = None
                self.add_norm2_xclbin = None
                self.add_norm2_insts = None
            else:
                # Residual connection
                self.add_xclbin = None
                self.add_insts = None
                # Layer normalization
                self.ln1_xclbin = None
                self.ln1_insts = None
                self.ln2_xclbin = None
                self.ln2_insts = None
            if self.use_pip_ffn:
                # Pipelined FFN
                self.ffn_xclbin = None
                self.ffn_insts = None
            else:
                # Up projection
                self.up_proj_xclbin = None
                self.up_proj_insts = None
                # GeLU activation
                self.gelu_xclbin = None
                self.gelu_insts = None
                # Down projection
                self.down_proj_xclbin = None
                self.down_proj_insts = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        """Set up artifacts for the encoder layer components using 13 individual layers."""
        artifacts = []
        device_str = self.context.device_manager.device_str()

        kernel_id = 0x801

        eltwise_mul_tile_size = (self.seq_len * self.seq_len * self.num_heads) // (
            self.num_aie_columns * 2
        )
        if not self.use_pip_addnorm or not self.use_pip_an_ffn:
            eltwise_add_tile_size = (self.seq_len * self.hidden_size) // (
                self.num_aie_columns * 2
            )
        if not self.use_pip_ffn or not self.use_pip_an_ffn:
            gelu_tile_size = (self.seq_len * self.intermediate_size) // (
                self.num_aie_columns * 2
            )

        prefix_base = f"encoder_{int(self.use_pip_addnorm)}_{int(self.use_pip_ffn)}_{int(self.use_pip_mha)}_{int(self.use_pip_an_ffn)}_"
        # Q/K/V/O projection kernel
        qkvo_proj = AIEGEMM(  # L1 utilization = 54 KB with double buffering
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            tile_m=64,
            tile_k=96,
            tile_n=48,  # N=768 processed across 8 columns with n=48
            num_aie_columns=self.num_aie_columns,
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
            skip_add_to_list=True,
        )
        self.qkvo_proj_xclbin, self.qkvo_proj_insts = qkvo_proj.get_artifacts(
            prefix=f"{prefix_base}qkvo_proj_"
        )
        self.qkvo_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_qkvo_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.qkvo_proj_xclbin.kernel_name = "encoder_qkvo_proj"
        artifacts.append(self.qkvo_proj_insts)
        kernel_id += 1

        if self.use_pip_mha:
            self.mha_xclbin, self.mha_insts = AIEMHA(
                num_heads=self.num_heads,
                seq_len=self.seq_len,
                d=self.head_dim,
                num_KV_heads=self.num_heads,
                num_of_pipelines=8,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}mha_")
            self.mha_xclbin.xclbin_input = self.qkvo_proj_xclbin
            self.mha_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_mha",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.mha_xclbin.kernel_name = "encoder_mha"
            self.mha_xclbin.depends += [self.qkvo_proj_xclbin]
            artifacts.append(self.mha_insts)
            kernel_id += 1
            next_dep = self.mha_xclbin
        else:
            # K Transpose kernel (transpose K matrix)
            k_transpose = AIETranspose(
                M=self.seq_len,
                N=self.hidden_size,
                num_aie_columns=self.num_aie_columns,
                num_channels=2,
                m=64,
                n=96,
                s=8,
                skip_add_to_list=True,
            )
            self.k_transpose_xclbin, self.k_transpose_insts = k_transpose.get_artifacts(
                prefix=f"{prefix_base}k_transpose_"
            )
            self.k_transpose_xclbin.xclbin_input = self.qkvo_proj_xclbin
            self.k_transpose_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_k_transpose",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.k_transpose_xclbin.kernel_name = "encoder_k_transpose"
            self.k_transpose_xclbin.depends += [self.qkvo_proj_xclbin]
            artifacts.append(self.k_transpose_insts)
            kernel_id += 1

            # Attention score calculations (GEMM for Q*K^T, batched across heads)
            self.attn_scores_xclbin, self.attn_scores_insts = AIEGEMM(
                M=self.seq_len,
                K=self.head_dim,
                N=self.seq_len,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                num_aie_columns=self.num_aie_columns,
                batch_A=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_C=(self.num_heads, 0),  # Batch across heads, batch dim first
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_scores_")
            self.attn_scores_xclbin.xclbin_input = self.k_transpose_xclbin
            self.attn_scores_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_scores",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_scores_xclbin.kernel_name = "encoder_attn_scores"
            self.attn_scores_xclbin.depends += [
                self.k_transpose_xclbin,
            ]
            artifacts.append(self.attn_scores_insts)
            kernel_id += 1

            # Attention score scaling (Multiplication per attention score)
            self.attn_scale_xclbin, self.attn_scale_insts = AIEElementwiseMul(
                size=self.seq_len * self.seq_len * self.num_heads,
                num_aie_columns=self.num_aie_columns,
                num_channels=2,
                tile_size=min(
                    math.gcd(4096, eltwise_mul_tile_size), eltwise_mul_tile_size
                ),
                scalar_broadcast=math.sqrt(1.0 / self.head_dim),
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_scale_")
            self.attn_scale_xclbin.xclbin_input = self.attn_scores_xclbin
            self.attn_scale_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_scale",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_scale_xclbin.kernel_name = "encoder_attn_scale"
            self.attn_scale_xclbin.depends += [self.attn_scores_xclbin]
            artifacts.append(self.attn_scale_insts)
            kernel_id += 1

            # Attention weight calculations (Softmax per attention score)
            self.attn_softmax_xclbin, self.attn_softmax_insts = AIESoftmax(
                rows=self.seq_len * self.num_heads,
                cols=self.seq_len,
                num_aie_columns=self.num_aie_columns,
                num_channels=2,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_softmax_")
            self.attn_softmax_xclbin.xclbin_input = self.attn_scale_xclbin
            self.attn_softmax_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_softmax",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_softmax_xclbin.kernel_name = "encoder_attn_softmax"
            self.attn_softmax_xclbin.depends += [self.attn_scale_xclbin]
            artifacts.append(self.attn_softmax_insts)
            kernel_id += 1

            # Output head calculations (GEMM per attention weights/V heads, batched across heads)
            self.attn_output_xclbin, self.attn_output_insts = AIEGEMM(
                M=self.seq_len,
                K=self.seq_len,
                N=self.head_dim,
                tile_m=64,
                tile_k=64,
                tile_n=16,
                num_aie_columns=4,
                batch_A=(self.num_heads, 0),  # Batch across heads, batch dim first
                batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
                batch_C=(self.num_heads, 1),  # Batch across heads, batch dim first
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
                skip_add_to_list=True,
            ).get_artifacts(prefix=f"{prefix_base}attn_output_")
            self.attn_output_xclbin.xclbin_input = self.attn_softmax_xclbin
            self.attn_output_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_attn_output",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.attn_output_xclbin.kernel_name = "encoder_attn_output"
            self.attn_output_xclbin.depends += [
                self.attn_softmax_xclbin,
            ]
            artifacts.append(self.attn_output_insts)
            kernel_id += 1
            next_dep = self.attn_output_xclbin

        if self.use_pip_an_ffn:
            aie_anffn_config = {
                "emulate_bf16_mmul_with_bfp16": True,
                "nA_tiles_distributed": 4,
                "nB_tiles_distributed": 2,
                "stage_only": None,
                "gelu_stage": 1,
            }
            # Second Layer normalization kernel
            self.anffn_xclbin, self.anffn_insts = AIEANFFN(
                M=self.seq_len,
                K=self.hidden_size,
                N=self.intermediate_size,
                tile_m=16,
                tile_k=128,
                tile_n=96,
                down_proj_depth=6,
                num_aie_columns=self.num_aie_columns,
                ln1_weight=self.ln1_weight,
                ln2_weight=self.ln2_weight,
                debug_mode=-1,
                **aie_anffn_config,
            ).get_artifacts(prefix=f"{prefix_base}anffn_")
            self.anffn_xclbin.xclbin_input = next_dep
            self.anffn_xclbin.extra_flags += [
                "--xclbin-instance-name=encoder_anffn",
                f"--xclbin-kernel-id={hex(kernel_id)}",
            ]
            self.anffn_xclbin.kernel_name = "encoder_anffn"
            self.anffn_xclbin.depends += [next_dep]
            artifacts.append(self.anffn_xclbin)
            artifacts.append(self.anffn_insts)
            # Store final xclbin
            self.combined_xclbin = self.anffn_xclbin
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm kernel
                self.add_norm1_xclbin, self.add_norm1_insts = AIEAddAndNorm(
                    size=self.seq_len * self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    tile_size=self.hidden_size,
                    weights=self.ln1_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_norm1_")
                self.add_norm1_xclbin.xclbin_input = next_dep
                self.add_norm1_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add_norm1",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_norm1_xclbin.kernel_name = "encoder_add_norm1"
                self.add_norm1_xclbin.depends += [
                    self.qkvo_proj_xclbin,
                    next_dep,
                ]
                artifacts.append(self.add_norm1_insts)
                next_dep = self.add_norm1_xclbin
                kernel_id += 1
            else:
                # Layer normalization kernel
                self.ln1_xclbin, self.ln1_insts = AIELayerNorm(
                    size=self.seq_len * self.hidden_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln1_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ln1_")
                self.ln1_xclbin.xclbin_input = next_dep
                self.ln1_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ln1",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ln1_xclbin.kernel_name = "encoder_ln1"
                self.ln1_xclbin.depends += [self.qkvo_proj_xclbin, next_dep]
                artifacts.append(self.ln1_insts)
                kernel_id += 1

                # Residual connection kernel (Eltwise add)
                self.add_xclbin, self.add_insts = AIEElementwiseAdd(
                    size=self.seq_len * self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(
                        math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size
                    ),
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_")
                self.add_xclbin.xclbin_input = self.ln1_xclbin
                self.add_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_xclbin.kernel_name = "encoder_add"
                self.add_xclbin.depends += [self.ln1_xclbin]
                artifacts.append(self.add_insts)
                next_dep = self.add_xclbin
                kernel_id += 1
            if self.use_pip_ffn:
                aie_ffn_config = {
                    "b_col_maj": False,
                    "c_col_maj": False,
                    "emulate_bf16_mmul_with_bfp16": True,
                    "n_a_tiles_distributed": 4,
                    "n_b_tiles_distributed": 4,
                    "stage_only": None,
                    "gelu_stage": 1,
                }
                self.ffn_xclbin, self.ffn_insts = AIEFFN(
                    M=self.seq_len,
                    K=self.hidden_size,
                    N=self.intermediate_size,
                    tile_m=64,
                    tile_k=48,
                    tile_n=96,
                    down_proj_depth=8,
                    num_aie_columns=self.num_aie_columns,
                    **aie_ffn_config,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ffn_")
                self.ffn_xclbin.xclbin_input = next_dep
                self.ffn_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ffn",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ffn_xclbin.kernel_name = "encoder_ffn"
                self.ffn_xclbin.depends += [next_dep]
                artifacts.append(self.ffn_insts)
                next_dep = self.ffn_xclbin
                kernel_id += 1
            else:
                # Up projection (GEMM with Up projection weight)
                self.up_proj_xclbin, self.up_proj_insts = (
                    AIEGEMM(  # L1 utilization = 54 KB with double buffering
                        M=self.seq_len,
                        K=self.hidden_size,
                        N=self.intermediate_size,
                        tile_m=64,
                        tile_k=48,
                        tile_n=96,
                        num_aie_columns=self.num_aie_columns,
                        prio_accuracy=False,
                        emulate_bf16_mmul_with_bfp16=True,
                        skip_add_to_list=True,
                    ).get_artifacts(prefix=f"{prefix_base}up_proj_")
                )
                self.up_proj_xclbin.xclbin_input = next_dep
                self.up_proj_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_up_proj",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.up_proj_xclbin.kernel_name = "encoder_up_proj"
                self.up_proj_xclbin.depends += [next_dep]
                artifacts.append(self.up_proj_insts)
                kernel_id += 1

                # Activation function (GeLU)
                self.gelu_xclbin, self.gelu_insts = AIEGELU(
                    size=self.seq_len * self.intermediate_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    tile_size=min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}gelu_")
                self.gelu_xclbin.xclbin_input = self.up_proj_xclbin
                self.gelu_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_gelu",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.gelu_xclbin.kernel_name = "encoder_gelu"
                self.gelu_xclbin.depends += [self.up_proj_xclbin]
                artifacts.append(self.gelu_insts)
                kernel_id += 1

                # Down projection (GEMM with Down projection weight)
                self.down_proj_xclbin, self.down_proj_insts = (
                    AIEGEMM(  # L1 utilization = 54 KB with double buffering
                        M=self.seq_len,
                        K=self.intermediate_size,
                        N=self.hidden_size,
                        tile_m=64,
                        tile_k=96,
                        tile_n=48,  # N=768 processed across 8 columns with n=48
                        num_aie_columns=self.num_aie_columns,
                        prio_accuracy=False,
                        emulate_bf16_mmul_with_bfp16=True,
                        skip_add_to_list=True,
                    ).get_artifacts(prefix=f"{prefix_base}down_proj_")
                )
                self.down_proj_xclbin.xclbin_input = self.gelu_xclbin
                self.down_proj_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_down_proj",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.down_proj_xclbin.kernel_name = "encoder_down_proj"
                self.down_proj_xclbin.depends += [self.gelu_xclbin]
                artifacts.append(self.down_proj_insts)
                next_dep = self.down_proj_xclbin
                kernel_id += 1
            if self.use_pip_addnorm:
                # Second Pipelined add & norm kernel
                self.add_norm2_xclbin, self.add_norm2_insts = AIEAddAndNorm(
                    size=self.seq_len * self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    tile_size=self.hidden_size,
                    weights=self.ln2_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}add_norm2_")
                self.add_norm2_xclbin.xclbin_input = next_dep
                self.add_norm2_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_add_norm2",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.add_norm2_xclbin.kernel_name = "encoder_add_norm2"
                self.add_norm2_xclbin.depends += [
                    next_dep,
                ]
                artifacts.append(self.add_norm2_xclbin)
                artifacts.append(self.add_norm2_insts)
                # Store final xclbin
                self.combined_xclbin = self.add_norm2_xclbin
            else:
                # Second Layer normalization kernel
                self.ln2_xclbin, self.ln2_insts = AIELayerNorm(
                    size=self.seq_len * self.hidden_size,
                    tile_size=self.hidden_size,
                    num_aie_columns=self.num_aie_columns,
                    num_channels=2,
                    weights=self.ln2_weight,
                    skip_add_to_list=True,
                ).get_artifacts(prefix=f"{prefix_base}ln2_")
                self.ln2_xclbin.xclbin_input = next_dep
                self.ln2_xclbin.extra_flags += [
                    "--xclbin-instance-name=encoder_ln2",
                    f"--xclbin-kernel-id={hex(kernel_id)}",
                ]
                self.ln2_xclbin.kernel_name = "encoder_ln2"
                self.ln2_xclbin.depends += [next_dep]
                artifacts.append(self.ln2_xclbin)
                artifacts.append(self.ln2_insts)
                # Store final xclbin
                self.combined_xclbin = self.ln2_xclbin

        self.add_artifacts(artifacts)
        logging.info(f"Finished setting up {len(artifacts)} BERT Encoder artifacts.")

    def set_up_runtime(self):
        """Set up runtime buffers and kernels for all 13 layers."""
        act_size = self.seq_len * self.hidden_size

        # Input buffer
        self.add_buffer("input", act_size)

        # Weight buffers (separate Q/K/V weights)
        self.add_buffer(
            "q_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.q_weight.T) if self.q_weight is not None else None
            ),
        )
        self.add_buffer(
            "k_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.k_weight.T) if self.k_weight is not None else None
            ),
        )
        self.add_buffer(
            "v_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.v_weight.T) if self.v_weight is not None else None
            ),
        )
        self.add_buffer(
            "attn_output_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.attn_output_weight.T)
                if self.attn_output_weight is not None
                else None
            ),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.add_buffer(
                "ln1_weight",
                self.hidden_size,
                static_data=(
                    torch_to_numpy(self.ln1_weight)
                    if self.ln1_weight is not None
                    else None
                ),
            )
        self.add_buffer(
            "ffn_up_weight",
            self.hidden_size * self.intermediate_size,
            static_data=(
                torch_to_numpy(self.ffn_up_weight.T)
                if self.ffn_up_weight is not None
                else None
            ),
        )
        self.add_buffer(
            "ffn_down_weight",
            self.intermediate_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.ffn_down_weight.T)
                if self.ffn_down_weight is not None
                else None
            ),
        )
        if self.use_pip_addnorm or self.use_pip_an_ffn:
            self.add_buffer(
                "ln2_weight",
                self.hidden_size,
                static_data=(
                    torch_to_numpy(self.ln2_weight)
                    if self.ln2_weight is not None
                    else None
                ),
            )

        # Intermediate buffers for all layers
        self.add_buffer("q_output", act_size)  # After layer 1a
        self.add_buffer("k_output", act_size)  # After layer 1b
        self.add_buffer("v_output", act_size)  # After layer 1c
        if self.use_pip_mha:
            self.add_buffer("mha_output", act_size)  # After layer 5
        else:
            self.add_buffer("k_transposed", act_size)  # After K transpose
            self.add_buffer(
                "attn_scores_output", self.seq_len * self.seq_len * self.num_heads
            )  # After layer 2
            self.add_buffer(
                "attn_scaled_output", self.seq_len * self.seq_len * self.num_heads
            )  # After layer 3
            self.add_buffer(
                "attn_weights_output", self.seq_len * self.seq_len * self.num_heads
            )  # After layer 4
            self.add_buffer("attn_heads_output", act_size)  # After layer 5
        self.add_buffer("output_proj_output", act_size)  # After layer 6
        if not self.use_pip_an_ffn:
            if self.use_pip_addnorm:
                self.add_buffer("add_norm1_output", act_size)  # After layer 7
            else:
                self.add_buffer("ln1_output", act_size)  # After layer 7
                self.add_buffer("add1_output", act_size)  # After layer 8
            if self.use_pip_ffn:
                self.add_buffer("ffn_output", act_size)  # After layer 9-12
            else:
                self.add_buffer(
                    "up_proj_output", self.seq_len * self.intermediate_size
                )  # After layer 9
                self.add_buffer(
                    "gelu_output", self.seq_len * self.intermediate_size
                )  # After layer 10
                self.add_buffer("down_proj_output", act_size)  # After layer 11
            if not self.use_pip_addnorm:
                self.add_buffer("ln2_output", act_size)  # After layer 12

        # Output buffer
        self.add_buffer("output", act_size)
        logging.info(
            f"Finished setting up {len(self.buffers)} BERT Encoder runtime buffers."
        )

        # Add kernels for all layers
        self.add_kernel(
            "encoder_qkvo_proj",
            self.combined_xclbin,
            self.qkvo_proj_xclbin.kernel_name,
            self.qkvo_proj_insts,
        )
        if self.use_pip_mha:
            self.add_kernel(
                "encoder_mha",
                self.combined_xclbin,
                self.mha_xclbin.kernel_name,
                self.mha_insts,
            )
        else:
            self.add_kernel(
                "encoder_k_transpose",
                self.combined_xclbin,
                self.k_transpose_xclbin.kernel_name,
                self.k_transpose_insts,
            )
            self.add_kernel(
                "encoder_attn_scores",
                self.combined_xclbin,
                self.attn_scores_xclbin.kernel_name,
                self.attn_scores_insts,
            )
            self.add_kernel(
                "encoder_attn_scale",
                self.combined_xclbin,
                self.attn_scale_xclbin.kernel_name,
                self.attn_scale_insts,
            )
            self.add_kernel(
                "encoder_attn_softmax",
                self.combined_xclbin,
                self.attn_softmax_xclbin.kernel_name,
                self.attn_softmax_insts,
            )
            self.add_kernel(
                "encoder_attn_output",
                self.combined_xclbin,
                self.attn_output_xclbin.kernel_name,
                self.attn_output_insts,
            )
        if self.use_pip_an_ffn:
            self.add_kernel(
                "encoder_anffn",
                self.combined_xclbin,
                self.anffn_xclbin.kernel_name,
                self.anffn_insts,
            )
        else:
            if self.use_pip_addnorm:
                self.add_kernel(
                    "encoder_add_norm1",
                    self.combined_xclbin,
                    self.add_norm1_xclbin.kernel_name,
                    self.add_norm1_insts,
                )
            else:
                self.add_kernel(
                    "encoder_ln1",
                    self.combined_xclbin,
                    self.ln1_xclbin.kernel_name,
                    self.ln1_insts,
                )
                self.add_kernel(
                    "encoder_add",
                    self.combined_xclbin,
                    self.add_xclbin.kernel_name,
                    self.add_insts,
                )
            if self.use_pip_ffn:
                self.add_kernel(
                    "encoder_ffn",
                    self.combined_xclbin,
                    self.ffn_xclbin.kernel_name,
                    self.ffn_insts,
                )
            else:
                self.add_kernel(
                    "encoder_up_proj",
                    self.combined_xclbin,
                    self.up_proj_xclbin.kernel_name,
                    self.up_proj_insts,
                )
                self.add_kernel(
                    "encoder_gelu",
                    self.combined_xclbin,
                    self.gelu_xclbin.kernel_name,
                    self.gelu_insts,
                )
                self.add_kernel(
                    "encoder_down_proj",
                    self.combined_xclbin,
                    self.down_proj_xclbin.kernel_name,
                    self.down_proj_insts,
                )
            if self.use_pip_addnorm:
                self.add_kernel(
                    "encoder_add_norm2",
                    self.combined_xclbin,
                    self.add_norm2_xclbin.kernel_name,
                    self.add_norm2_insts,
                )
            else:
                self.add_kernel(
                    "encoder_ln2",
                    self.combined_xclbin,
                    self.ln2_xclbin.kernel_name,
                    self.ln2_insts,
                )
        logging.info(
            f"Finished setting up {len(self.kernels)} BERT Encoder runtime kernels."
        )

        # Build runlist for all layers
        # Q projection
        self.add_to_runlist("encoder_qkvo_proj", "input", "q_weight", "q_output")
        # K projection
        self.add_to_runlist("encoder_qkvo_proj", "input", "k_weight", "k_output")
        # V projection
        self.add_to_runlist("encoder_qkvo_proj", "input", "v_weight", "v_output")
        if self.use_pip_mha:
            self.add_to_runlist(
                "encoder_mha", "q_output", "k_output", "v_output", "mha_output"
            )
            next_output = "mha_output"
        else:
            # Transpose K matrix
            self.add_to_runlist("encoder_k_transpose", "k_output", "k_transposed")
            # Attention score calculations
            self.add_to_runlist(
                "encoder_attn_scores", "q_output", "k_transposed", "attn_scores_output"
            )
            # Attention score scaling
            self.add_to_runlist(
                "encoder_attn_scale",
                "attn_scores_output",
                "attn_scaled_output",
            )
            # Attention weight calculations (Softmax)
            self.add_to_runlist(
                "encoder_attn_softmax", "attn_scaled_output", "attn_weights_output"
            )
            # Output head calculations
            self.add_to_runlist(
                "encoder_attn_output",
                "attn_weights_output",
                "v_output",
                "attn_heads_output",
            )
            next_output = "attn_heads_output"
        # Output projection
        self.add_to_runlist(
            "encoder_qkvo_proj",
            next_output,
            "attn_output_weight",
            "output_proj_output",
        )
        if self.use_pip_an_ffn:
            # Pipelined AN-FFN, 2nd input is for residual connection
            self.add_to_runlist(
                "encoder_anffn",
                "output_proj_output",
                "input",
                "ffn_up_weight",
                "ffn_down_weight",
                "output",
            )
        else:
            if self.use_pip_addnorm:
                # Pipelined add & norm, 2nd input is for residual connection
                self.add_to_runlist(
                    "encoder_add_norm1",
                    "output_proj_output",
                    "input",
                    "add_norm1_output",
                )
                next_output = "add_norm1_output"
            else:
                # Layer normalization
                self.add_to_runlist("encoder_ln1", "output_proj_output", "ln1_output")
                # Residual connection
                self.add_to_runlist("encoder_add", "input", "ln1_output", "add1_output")
                next_output = "add1_output"
            if self.use_pip_ffn:
                # Pipelined FFN
                self.add_to_runlist(
                    "encoder_ffn",
                    next_output,
                    "ffn_up_weight",
                    "ffn_down_weight",
                    "ffn_output",
                )
                next_output = "ffn_output"
            else:
                # Up projection
                self.add_to_runlist(
                    "encoder_up_proj", next_output, "ffn_up_weight", "up_proj_output"
                )
                # GeLU activation
                self.add_to_runlist("encoder_gelu", "up_proj_output", "gelu_output")
                # Down projection
                self.add_to_runlist(
                    "encoder_down_proj",
                    "gelu_output",
                    "ffn_down_weight",
                    "down_proj_output",
                )
                next_output = "down_proj_output"
            if self.use_pip_addnorm:
                # Second Pipelined add & norm, 2nd input is for residual connection
                self.add_to_runlist(
                    "encoder_add_norm2", next_output, "add_norm1_output", "output"
                )
            else:
                # Layer normalization
                self.add_to_runlist("encoder_ln2", next_output, "ln2_output")
                # Residual connection
                self.add_to_runlist(
                    "encoder_add", "add1_output", "ln2_output", "output"
                )

        logging.info(f"Finished setting up {len(self.runlist)} BERT Encoder runlist.")

    def forward(self, x, attention_mask=None):
        """
        Forward pass through BERT encoder layer.

        Args:
            x: Input tensor of shape (seq_len, hidden_size)
            attention_mask: Optional attention mask (not used for now)

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        # x is [batch, size]
        batch = x.shape[0] if x.dim() > 1 else 1

        # Flatten inputs for AIE processing
        x_flat = x.view(-1)

        # Verify input size matches expected dimensions
        expected_size = batch * self.seq_len * self.hidden_size
        assert x_flat.shape[0] == expected_size

        self.write_buffer("input", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output",
            (self.seq_len, self.hidden_size),
            dtype=bfloat16,
        ).view(x.shape)

        return result
