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
from operators.common.utils import torch_to_numpy


class AIEAttnScores(AIEOperatorBase):
    """
    AIE-accelerated attention score calculation using runlist-based implementation.
    """

    def __init__(
        self,
        seq_len,
        hidden_size,
        num_heads,
        use_sep_gemms=True,
        context=None,
        num_aie_columns=8,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_aie_columns = num_aie_columns
        self.use_sep_gemms = use_sep_gemms

        # Derived dimensions
        self.head_dim = hidden_size // num_heads

        # Artifacts created by set_up_artifacts() - one per layer
        self.combined_xclbin = None
        # Attention
        self.attn_scores_xclbin = None
        self.attn_scores_insts = None

        super().__init__()

    def set_up_artifacts(self):
        """Set up artifacts for the encoder layer components using 13 individual layers."""
        artifacts = []
        device_str = self.context.device_manager.device_str()

        kernel_id = 0x801

        # Attention score calculations (GEMM for Q*K^T, batched across heads)
        if self.use_sep_gemms:
            self.attn_scores_xclbin, self.attn_scores_insts = AIEGEMM(
                M=self.seq_len,
                K=self.head_dim,
                N=self.seq_len,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                num_aie_columns=self.num_aie_columns,
                prio_accuracy=False,
                emulate_bf16_mmul_with_bfp16=True,
            ).get_artifacts(prefix="encoder_attn_scores_")
        else:
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
            ).get_artifacts(prefix="encoder_attn_scores_")
        self.attn_scores_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_scores",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_scores_xclbin.kernel_name = "encoder_attn_scores"
        artifacts.append(self.attn_scores_insts)
        artifacts.append(self.attn_scores_xclbin)

        # Store final xclbin
        self.combined_xclbin = self.attn_scores_xclbin

        self.add_artifacts(artifacts)
        logging.info(f"Finished setting up {len(artifacts)} BERT Encoder artifacts.")

    def set_up_runtime(self):
        act_size = self.seq_len * self.hidden_size
        if self.use_sep_gemms:
            for i in range(self.num_heads):
                # Input buffer
                self.add_buffer(f"q_{i}", act_size)
                self.add_buffer(f"k_{i}", act_size)
                # Output buffer
                self.add_buffer(f"attn_scores_{i}", act_size)
        else:
            # Input buffer
            self.add_buffer("q", act_size)
            self.add_buffer("k", act_size)
            # Output buffer
            self.add_buffer("attn_scores", act_size)

        logging.info(
            f"Finished setting up {len(self.buffers)} BERT Encoder runtime buffers."
        )

        # Add kernels for all layers
        self.add_kernel(
            "encoder_attn_scores",
            self.combined_xclbin,
            self.attn_scores_xclbin.kernel_name,
            self.attn_scores_insts,
        )
        logging.info(
            f"Finished setting up {len(self.kernels)} BERT Encoder runtime kernels."
        )

        # Build runlist for all layers
        # Attention score calculations
        if self.use_sep_gemms:
            for i in range(self.num_heads):
                self.add_to_runlist(
                    "encoder_attn_scores",
                    f"q_{i}",
                    f"k_{i}",
                    f"attn_scores_{i}",
                )
        else:
            self.add_to_runlist("encoder_attn_scores", "q", "k", "attn_scores")

        logging.info(f"Finished setting up {len(self.runlist)} BERT Encoder runlist.")

    def forward(self, x, attention_mask=None):
        """
        Only using this to test whether runlist of individual gemms is slower than batched gemm.
        """
        pass
