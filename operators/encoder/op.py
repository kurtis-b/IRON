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
    8. Residual connection (Eltwise add)
    9. Layer normalization
    10. Up projection (GEMM with Up projection weight)
    11. Activation function (GeLU)
    12. Down projection (GEMM with Down projection weight)
    13. Residual connection (Eltwise add)
    14. Layer normalization
    """

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        context=None,
        num_aie_columns=8,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_aie_columns = num_aie_columns

        # Derived dimensions
        self.head_dim = hidden_size // num_heads

        # Weights to be set by user (separate Q/K/V weights)
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ln1_weight = None
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln2_weight = None

        # Artifacts created by set_up_artifacts() - one per layer
        self.combined_xclbin = None
        # Layer 1: Q/K/V projections (separate GEMMs)
        self.q_proj_xclbin = None
        self.q_proj_insts = None
        self.k_proj_xclbin = None
        self.k_proj_insts = None
        self.k_transpose_xclbin = None
        self.k_transpose_insts = None
        self.v_proj_xclbin = None
        self.v_proj_insts = None
        # Layers 2-5: Attention (per head, simplified to single ops for now)
        self.attn_scores_xclbin = None
        self.attn_scores_insts = None
        self.attn_scale_xclbin = None
        self.attn_scale_insts = None
        self.attn_softmax_xclbin = None
        self.attn_softmax_insts = None
        self.attn_output_xclbin = None
        self.attn_output_insts = None
        # Layer 6: Output projection
        self.output_proj_xclbin = None
        self.output_proj_insts = None
        # Layer 7: Residual connection 1
        self.add1_xclbin = None
        self.add1_insts = None
        # Layer 8: Layer normalization 1
        self.ln1_xclbin = None
        self.ln1_insts = None
        # Layer 9: Up projection
        self.up_proj_xclbin = None
        self.up_proj_insts = None
        # Layer 10: GeLU activation
        self.gelu_xclbin = None
        self.gelu_insts = None
        # Layer 11: Down projection
        self.down_proj_xclbin = None
        self.down_proj_insts = None
        # Layer 12: Residual connection 2
        self.add2_xclbin = None
        self.add2_insts = None
        # Layer 13: Layer normalization 2
        self.ln2_xclbin = None
        self.ln2_insts = None

        super().__init__()

    def set_up_artifacts(self):
        """Set up artifacts for the encoder layer components using 13 individual layers."""
        artifacts = []
        device_str = self.context.device_manager.device_str()

        kernel_id = 0x801

        gelu_tile_size = (self.seq_len * self.intermediate_size) // (
            self.num_aie_columns * 2
        )
        eltwise_mul_tile_size = (self.seq_len * self.seq_len * self.num_heads) // (
            self.num_aie_columns * 2
        )
        eltwise_add_tile_size = (self.seq_len * self.hidden_size) // (
            self.num_aie_columns * 2
        )

        # Layer 1a: Q projection
        q_proj = AIEGEMM(  # L1 utilization = 54 KB with double buffering
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            tile_m=64,
            tile_k=96,
            tile_n=48,  # N=768 processed across 8 columns with n=48
            num_aie_columns=self.num_aie_columns,
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
        )
        self.q_proj_xclbin, self.q_proj_insts = q_proj.get_artifacts(
            prefix="encoder_q_proj_"
        )
        self.q_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_q_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.q_proj_xclbin.kernel_name = "encoder_q_proj"
        artifacts.append(self.q_proj_insts)
        kernel_id += 1

        # Layer 1b: K projection (separate GEMM)
        k_proj = AIEGEMM(  # L1 utilization = 54 KB with double buffering
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            tile_m=64,
            tile_k=96,
            tile_n=48,  # N=768 processed across 8 columns with n=48
            num_aie_columns=self.num_aie_columns,
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
        )
        self.k_proj_xclbin, self.k_proj_insts = k_proj.get_artifacts(
            prefix="encoder_k_proj_"
        )
        self.k_proj_xclbin.xclbin_input = self.q_proj_xclbin
        self.k_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_k_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.k_proj_xclbin.kernel_name = "encoder_k_proj"
        self.k_proj_xclbin.depends += [self.q_proj_xclbin]
        artifacts.append(self.k_proj_insts)
        kernel_id += 1

        # K Transpose layer (transpose K matrix)
        k_transpose = AIETranspose(
            M=self.seq_len,
            N=self.hidden_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            m=64,
            n=96,
            s=8,
        )
        self.k_transpose_xclbin, self.k_transpose_insts = k_transpose.get_artifacts(
            prefix="encoder_k_transpose_"
        )
        self.k_transpose_xclbin.xclbin_input = self.k_proj_xclbin
        self.k_transpose_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_k_transpose",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.k_transpose_xclbin.kernel_name = "encoder_k_transpose"
        self.k_transpose_xclbin.depends += [self.k_proj_xclbin]
        artifacts.append(self.k_transpose_insts)
        kernel_id += 1

        # Layer 1c: V projection (separate GEMM)
        v_proj = AIEGEMM(  # L1 utilization = 54 KB with double buffering
            M=self.seq_len,
            K=self.hidden_size,
            N=self.hidden_size,
            tile_m=64,
            tile_k=96,
            tile_n=48,  # N=768 processed across 8 columns with n=48
            num_aie_columns=self.num_aie_columns,
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
        )
        self.v_proj_xclbin, self.v_proj_insts = v_proj.get_artifacts(
            prefix="encoder_v_proj_"
        )
        self.v_proj_xclbin.xclbin_input = self.k_transpose_xclbin
        self.v_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_v_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.v_proj_xclbin.kernel_name = "encoder_v_proj"
        self.v_proj_xclbin.depends += [self.k_transpose_xclbin]
        artifacts.append(self.v_proj_insts)
        kernel_id += 1

        # Layer 2: Attention score calculations (GEMM for Q*K^T, batched across heads)
        self.attn_scores_xclbin, self.attn_scores_insts = AIEGEMM(
            M=self.seq_len,
            K=self.head_dim,
            N=self.seq_len,
            num_aie_columns=self.num_aie_columns,
            batch_A=(self.num_heads, 1),  # Batch across heads, batch dim first
            batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
            batch_C=(self.num_heads, 0),  # Batch across heads, batch dim first
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
        ).get_artifacts(prefix="encoder_attn_scores_")
        self.attn_scores_xclbin.xclbin_input = self.v_proj_xclbin
        self.attn_scores_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_scores",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_scores_xclbin.kernel_name = "encoder_attn_scores"
        self.attn_scores_xclbin.depends += [
            self.q_proj_xclbin,
            self.k_transpose_xclbin,
            self.v_proj_xclbin,
        ]
        artifacts.append(self.attn_scores_insts)
        kernel_id += 1

        # Layer 3: Attention score scaling (Multiplication per attention score)
        self.attn_scale_xclbin, self.attn_scale_insts = AIEElementwiseMul(
            size=self.seq_len * self.seq_len * self.num_heads,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            tile_size=min(math.gcd(4096, eltwise_mul_tile_size), eltwise_mul_tile_size),
        ).get_artifacts(prefix="encoder_attn_scale_")
        self.attn_scale_xclbin.xclbin_input = self.attn_scores_xclbin
        self.attn_scale_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_scale",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_scale_xclbin.kernel_name = "encoder_attn_scale"
        self.attn_scale_xclbin.depends += [self.attn_scores_xclbin]
        artifacts.append(self.attn_scale_insts)
        kernel_id += 1

        # Layer 4: Attention weight calculations (Softmax per attention score)
        self.attn_softmax_xclbin, self.attn_softmax_insts = AIESoftmax(
            rows=self.seq_len * self.num_heads,
            cols=self.seq_len,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
        ).get_artifacts(prefix="encoder_attn_softmax_")
        self.attn_softmax_xclbin.xclbin_input = self.attn_scale_xclbin
        self.attn_softmax_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_softmax",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_softmax_xclbin.kernel_name = "encoder_attn_softmax"
        self.attn_softmax_xclbin.depends += [self.attn_scale_xclbin]
        artifacts.append(self.attn_softmax_insts)
        kernel_id += 1

        # Layer 5: Output head calculations (GEMM per attention weights/V heads, batched across heads)
        self.attn_output_xclbin, self.attn_output_insts = AIEGEMM(
            M=self.seq_len,
            K=self.seq_len,
            N=self.head_dim,
            num_aie_columns=self.num_aie_columns,
            batch_A=(self.num_heads, 0),  # Batch across heads, batch dim first
            batch_B=(self.num_heads, 1),  # Batch across heads, batch dim first
            batch_C=(self.num_heads, 1),  # Batch across heads, batch dim first
            prio_accuracy=False,
            emulate_bf16_mmul_with_bfp16=True,
        ).get_artifacts(prefix="encoder_attn_output_")
        self.attn_output_xclbin.xclbin_input = self.attn_softmax_xclbin
        self.attn_output_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_output",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_output_xclbin.kernel_name = "encoder_attn_output"
        self.attn_output_xclbin.depends += [
            self.attn_softmax_xclbin,
            self.v_proj_xclbin,
        ]
        artifacts.append(self.attn_output_insts)
        kernel_id += 1

        # Layer 6: Output projection (GEMM with O weight)
        self.output_proj_xclbin, self.output_proj_insts = (
            AIEGEMM(  # L1 utilization = 54 KB with double buffering
                M=self.seq_len,
                K=self.hidden_size,
                N=self.hidden_size,
                tile_m=64,
                tile_k=96,
                tile_n=48,  # N=768 processed across 8 columns with n=48
                num_aie_columns=self.num_aie_columns,
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
            ).get_artifacts(prefix="encoder_output_proj_")
        )
        self.output_proj_xclbin.xclbin_input = self.attn_output_xclbin
        self.output_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_output_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.output_proj_xclbin.kernel_name = "encoder_output_proj"
        self.output_proj_xclbin.depends += [self.attn_output_xclbin]
        artifacts.append(self.output_proj_insts)
        kernel_id += 1

        # Layer 7: Residual connection (Eltwise add)
        self.add1_xclbin, self.add1_insts = AIEElementwiseAdd(
            size=self.seq_len * self.hidden_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            tile_size=min(math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size),
        ).get_artifacts(prefix="encoder_add1_")
        self.add1_xclbin.xclbin_input = self.output_proj_xclbin
        self.add1_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_add1",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add1_xclbin.kernel_name = "encoder_add1"
        self.add1_xclbin.depends += [self.output_proj_xclbin]
        artifacts.append(self.add1_insts)
        kernel_id += 1

        # Layer 8: Layer normalization
        self.ln1_xclbin, self.ln1_insts = AIELayerNorm(
            size=self.seq_len * self.hidden_size,
            tile_size=self.hidden_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            weighted=True,
        ).get_artifacts(prefix="encoder_ln1_")
        self.ln1_xclbin.xclbin_input = self.add1_xclbin
        self.ln1_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_ln1",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.ln1_xclbin.kernel_name = "encoder_ln1"
        self.ln1_xclbin.depends += [self.add1_xclbin]
        artifacts.append(self.ln1_insts)
        kernel_id += 1

        # Layer 9: Up projection (GEMM with Up projection weight)
        self.up_proj_xclbin, self.up_proj_insts = (
            AIEGEMM(  # L1 utilization = 54 KB with double buffering
                M=self.seq_len,
                K=self.hidden_size,
                N=self.intermediate_size,
                tile_m=64,
                tile_k=96,
                tile_n=48,  # N=768 processed across 8 columns with n=48
                num_aie_columns=self.num_aie_columns,
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
            ).get_artifacts(prefix="encoder_up_proj_")
        )
        self.up_proj_xclbin.xclbin_input = self.ln1_xclbin
        self.up_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_up_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.up_proj_xclbin.kernel_name = "encoder_up_proj"
        self.up_proj_xclbin.depends += [self.ln1_xclbin]
        artifacts.append(self.up_proj_insts)
        kernel_id += 1

        # Layer 10: Activation function (GeLU)
        self.gelu_xclbin, self.gelu_insts = AIEGELU(
            size=self.seq_len * self.intermediate_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            tile_size=min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
        ).get_artifacts(prefix="encoder_gelu_")
        self.gelu_xclbin.xclbin_input = self.up_proj_xclbin
        self.gelu_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_gelu",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.gelu_xclbin.kernel_name = "encoder_gelu"
        self.gelu_xclbin.depends += [self.up_proj_xclbin]
        artifacts.append(self.gelu_insts)
        kernel_id += 1

        # Layer 11: Down projection (GEMM with Down projection weight)
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
            ).get_artifacts(prefix="encoder_down_proj_")
        )
        self.down_proj_xclbin.xclbin_input = self.gelu_xclbin
        self.down_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_down_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.down_proj_xclbin.kernel_name = "encoder_down_proj"
        self.down_proj_xclbin.depends += [self.gelu_xclbin]
        artifacts.append(self.down_proj_insts)
        kernel_id += 1

        # Layer 12: Residual connection (Eltwise add)
        self.add2_xclbin, self.add2_insts = AIEElementwiseAdd(
            size=self.seq_len * self.hidden_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            tile_size=min(math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size),
        ).get_artifacts(prefix="encoder_add2_")
        self.add2_xclbin.xclbin_input = self.down_proj_xclbin
        self.add2_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_add2",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add2_xclbin.kernel_name = "encoder_add2"
        self.add2_xclbin.depends += [self.down_proj_xclbin, self.ln1_xclbin]
        artifacts.append(self.add2_insts)
        kernel_id += 1

        # Layer 13: Layer normalization
        self.ln2_xclbin, self.ln2_insts = AIELayerNorm(
            size=self.seq_len * self.hidden_size,
            tile_size=self.hidden_size,
            num_aie_columns=self.num_aie_columns,
            num_channels=2,
            weighted=True,
        ).get_artifacts(prefix="encoder_ln2_")
        self.ln2_xclbin.xclbin_input = self.add2_xclbin
        self.ln2_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_ln2",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.ln2_xclbin.kernel_name = "encoder_ln2"
        self.ln2_xclbin.depends += [self.add2_xclbin]
        artifacts.append(self.ln2_xclbin)
        artifacts.append(self.ln2_insts)

        # Store final xclbin
        self.combined_xclbin = self.ln2_xclbin

        self.add_artifacts(artifacts)
        logging.info(f"Finished setting up {len(artifacts)} BERT Encoder artifacts.")

    def set_up_runtime(self):
        """Set up runtime buffers and kernels for all 13 layers."""
        act_size = self.seq_len * self.hidden_size

        # Input/output buffers
        self.add_buffer("input", act_size)
        self.add_buffer("output", act_size)

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
        self.add_buffer(
            "ln1_weight",
            self.hidden_size,
            static_data=(
                torch_to_numpy(self.ln1_weight) if self.ln1_weight is not None else None
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
        self.add_buffer(
            "ln2_weight",
            self.hidden_size,
            static_data=(
                torch_to_numpy(self.ln2_weight) if self.ln2_weight is not None else None
            ),
        )

        # Intermediate buffers for all layers
        self.add_buffer("q_output", act_size)  # After layer 1a
        self.add_buffer("k_output", act_size)  # After layer 1b
        self.add_buffer("k_transposed", act_size)  # After K transpose
        self.add_buffer("v_output", act_size)  # After layer 1c
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
        self.add_buffer("add1_output", act_size)  # After layer 7
        self.add_buffer("ln1_output", act_size)  # After layer 8
        self.add_buffer(
            "up_proj_output", self.seq_len * self.intermediate_size
        )  # After layer 9
        self.add_buffer(
            "gelu_output", self.seq_len * self.intermediate_size
        )  # After layer 10
        self.add_buffer("down_proj_output", act_size)  # After layer 11
        self.add_buffer("add2_output", act_size)  # After layer 12
        # Layer 13 outputs to "output" buffer

        # Scaling factor for attention
        # TODO: Should be scalar, but for now is a matrix
        self.add_buffer(
            "attn_scale_factor", self.seq_len * self.seq_len * self.num_heads
        )
        logging.info(
            f"Finished setting up {len(self.buffers)} BERT Encoder runtime buffers."
        )

        # Add kernels for all layers
        self.add_kernel(
            "encoder_q_proj",
            self.combined_xclbin,
            self.q_proj_xclbin.kernel_name,
            self.q_proj_insts,
        )
        self.add_kernel(
            "encoder_k_proj",
            self.combined_xclbin,
            self.k_proj_xclbin.kernel_name,
            self.k_proj_insts,
        )
        self.add_kernel(
            "encoder_k_transpose",
            self.combined_xclbin,
            self.k_transpose_xclbin.kernel_name,
            self.k_transpose_insts,
        )
        self.add_kernel(
            "encoder_v_proj",
            self.combined_xclbin,
            self.v_proj_xclbin.kernel_name,
            self.v_proj_insts,
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
        self.add_kernel(
            "encoder_output_proj",
            self.combined_xclbin,
            self.output_proj_xclbin.kernel_name,
            self.output_proj_insts,
        )
        self.add_kernel(
            "encoder_add1",
            self.combined_xclbin,
            self.add1_xclbin.kernel_name,
            self.add1_insts,
        )
        self.add_kernel(
            "encoder_ln1",
            self.combined_xclbin,
            self.ln1_xclbin.kernel_name,
            self.ln1_insts,
        )
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
        self.add_kernel(
            "encoder_add2",
            self.combined_xclbin,
            self.add2_xclbin.kernel_name,
            self.add2_insts,
        )
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
        # Layer 1a: Q projection
        self.add_to_runlist("encoder_q_proj", "input", "q_weight", "q_output")
        # Layer 1b: K projection
        self.add_to_runlist("encoder_k_proj", "input", "k_weight", "k_output")
        # K Transpose: Transpose K matrix
        self.add_to_runlist("encoder_k_transpose", "k_output", "k_transposed")
        # Layer 1c: V projection
        self.add_to_runlist("encoder_v_proj", "input", "v_weight", "v_output")
        # Layer 2: Attention score calculations
        self.add_to_runlist(
            "encoder_attn_scores", "q_output", "k_transposed", "attn_scores_output"
        )
        # Layer 3: Attention score scaling
        self.add_to_runlist(
            "encoder_attn_scale",
            "attn_scores_output",
            "attn_scale_factor",
            "attn_scaled_output",
        )
        # Layer 4: Attention weight calculations (Softmax)
        self.add_to_runlist(
            "encoder_attn_softmax", "attn_scaled_output", "attn_weights_output"
        )
        # Layer 5: Output head calculations
        self.add_to_runlist(
            "encoder_attn_output",
            "attn_weights_output",
            "v_output",
            # "attn_heads_output",
            "output",
        )
        # Layer 6: Output projection
        # self.add_to_runlist(
        #     "encoder_output_proj",
        #     "attn_heads_output",
        #     "attn_output_weight",
        #     "output_proj_output",
        # )
        # # Layer 7: Residual connection
        # self.add_to_runlist(
        #     "encoder_add1", "input", "output_proj_output", "add1_output"
        # )
        # # Layer 8: Layer normalization
        # self.add_to_runlist("encoder_ln1", "add1_output", "ln1_weight", "ln1_output")
        # # Layer 9: Up projection
        # self.add_to_runlist(
        #     "encoder_up_proj", "ln1_output", "ffn_up_weight", "up_proj_output"
        # )
        # # Layer 10: GeLU activation
        # self.add_to_runlist("encoder_gelu", "up_proj_output", "gelu_output")
        # # Layer 11: Down projection
        # self.add_to_runlist(
        #     "encoder_down_proj", "gelu_output", "ffn_down_weight", "down_proj_output"
        # )
        # # Layer 12: Residual connection
        # self.add_to_runlist(
        #     "encoder_add2", "ln1_output", "down_proj_output", "add2_output"
        # )
        # # Layer 13: Layer normalization
        # self.add_to_runlist("encoder_ln2", "add2_output", "ln2_weight", "output")

        logging.info(f"Finished setting up {len(self.runlist)} BERT Encoder runlist.")

    def forward(self, x, attention_mask=None):
        """
        Forward pass through BERT encoder layer.

        Args:
            x: Input tensor of shape (seq_len, hidden_size)
            attention_mask: Optional attention mask (not used in this simplified version)

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        # Flatten input and drop batch dimension if present
        x_flat = x.reshape(-1)

        assert (
            x_flat.shape[0] == self.seq_len * self.hidden_size
        ), f"Expected input size {self.seq_len * self.hidden_size}, got {x_flat.shape[0]}"

        self.write_buffer("input", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output",
            (self.seq_len, self.hidden_size),
        ).view_as(x)

        return result
