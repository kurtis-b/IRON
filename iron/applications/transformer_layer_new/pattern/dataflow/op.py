# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import numpy as np
from ml_dtypes import bfloat16
import math

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.ffn.op import AIEFFN
from iron.common.utils import torch_to_numpy


def default_dataflow_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
):
    head_dim = hidden_size // num_heads
    return {
        "qkv_proj": {
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "tile_m": 32,
            "tile_k": 64,
            "tile_n": 16,
            "parallel_seq": 1,
            "parallel_emb": num_aie_columns,
        },
        "mha_out_proj": {
            "num_heads": num_heads,
            "seq_len": seq_len,
            "d": head_dim,
            "parallel_seq": 1,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": head_dim,
            "parallel_heads": 1,
            "o_proj_acc_depth": 1,
        },
        "add_norm1": {
            "size": seq_len * hidden_size,
            "num_aie_columns": num_aie_columns,
            "tile_size": hidden_size,
        },
        "ffn": {
            "M": seq_len,
            "K": hidden_size,
            "N": intermediate_size,
            "tile_m": 64,
            "tile_k": 48,
            "tile_n": 96,
            "down_proj_depth": 8,
            "num_aie_columns": num_aie_columns,
            "b_col_maj": False,
            "c_col_maj": False,
            "emulate_bf16_mmul_with_bfp16": True,
            "n_a_tiles_distributed": 4,
            "n_b_tiles_distributed": 4,
            "stage_only": None,
            "gelu_stage": 1,
        },
        "add_norm2": {
            "size": seq_len * hidden_size,
            "num_aie_columns": num_aie_columns,
            "tile_size": hidden_size,
        },
    }


def resolve_dataflow_operator_config(
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
        for name, config in default_dataflow_operator_config(
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
            raise ValueError(f"Unsupported dataflow operator config: {name}")
        resolved[name].update(overrides)
    return resolved


class AIETransformerDataflow(AIEOperatorBase):
    """
    AIE-accelerated Transformer Layer using a runlist dataflow implementations.

    A Transformer Layer consists of:
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
        ln1_weight=None,
        ln2_weight=None,
        operator_config=None,
        context=None,
    ):
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
        self.operator_config = resolve_dataflow_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            num_aie_columns=num_aie_columns,
            operator_config=operator_config,
        )

        # Artifacts created by set_up_artifacts() - one per layer
        self.combined_xclbin = None
        # Q/K/V/O projections
        self.qkvo_proj_xclbin = None
        self.qkvo_proj_insts = None
        # Attention
        self.mha_out_proj_xclbin = None
        self.mha_out_proj_insts = None
        # Pipelined add & norm
        self.add_norm1_xclbin = None
        self.add_norm1_insts = None
        self.add_norm2_xclbin = None
        self.add_norm2_insts = None
        # Pipelined FFN
        self.ffn_xclbin = None
        self.ffn_insts = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        """Set up artifacts for the encoder layer components using 13 individual layers."""
        artifacts = []
        device_str = self.context.device_manager.device_str()

        kernel_id = 0x801

        eltwise_mul_tile_size = (self.seq_len * self.seq_len * self.num_heads) // (
            self.num_aie_columns * 2
        )

        prefix_base = f"encoder_dataflow_"
        # Q/K/V/O projection kernel
        qkvo_proj_kwargs = dict(self.operator_config["qkv_proj"])
        qkvo_proj_kwargs.update(
            {
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        qkvo_proj = AIEQKVProj(**qkvo_proj_kwargs)
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

        mha_out_proj_kwargs = dict(self.operator_config["mha_out_proj"])
        mha_out_proj_kwargs.update(
            {
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        mha_out_proj = AIEMHAOutProj(**mha_out_proj_kwargs)
        mha_out_proj.set_up_artifacts()
        self.mha_out_proj_xclbin = mha_out_proj.xclbin_artifact
        self.mha_out_proj_insts = mha_out_proj.insts_artifact
        self.mha_out_proj_xclbin.xclbin_input = self.qkvo_proj_xclbin
        self.mha_out_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_mha_out_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.mha_out_proj_xclbin.kernel_name = "encoder_mha_out_proj"
        self.mha_out_proj_xclbin.depends += [self.qkvo_proj_xclbin]
        artifacts.append(self.mha_out_proj_insts)
        kernel_id += 1
        next_dep = self.mha_out_proj_xclbin

        # Pipelined add & norm kernel
        add_norm1_kwargs = dict(self.operator_config["add_norm1"])
        add_norm1_kwargs.update(
            {
                "weights": self.ln1_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.add_norm1_xclbin, self.add_norm1_insts = AIEAddAndNorm(
            **add_norm1_kwargs
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

        ffn_kwargs = dict(self.operator_config["ffn"])
        ffn_kwargs.update(
            {
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.ffn_xclbin, self.ffn_insts = AIEFFN(
            **ffn_kwargs,
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

        # Second Pipelined add & norm kernel
        add_norm2_kwargs = dict(self.operator_config["add_norm2"])
        add_norm2_kwargs.update(
            {
                "weights": self.ln2_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.add_norm2_xclbin, self.add_norm2_insts = AIEAddAndNorm(
            **add_norm2_kwargs
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

        self.add_artifacts(artifacts)
        logging.info(f"Finished setting up {len(artifacts)} BERT Encoder artifacts.")

    def set_up_runtime(self):
        """Set up runtime buffers and kernels for all 13 layers."""
        act_size = self.seq_len * self.hidden_size

        # Input buffer
        self.add_buffer("input", act_size)

        # Weight buffers (separate Q/K/V weights)
        self.add_buffer(
            "qkv_weight",
            self.hidden_size * 3 * self.hidden_size,
            static_data=(
                torch_to_numpy(
                    torch.cat([self.q_weight, self.k_weight, self.v_weight], dim=1)
                )
                if self.q_weight is not None
                else None
            ),
        )
        self.add_buffer(
            "attn_output_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.attn_output_weight)
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
                torch_to_numpy(self.ffn_up_weight)
                if self.ffn_up_weight is not None
                else None
            ),
        )
        self.add_buffer(
            "ffn_down_weight",
            self.intermediate_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.ffn_down_weight)
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
        self.add_buffer("v_output", act_size)  # After layer 1c
        self.add_buffer("mha_out_proj_output", act_size)  # After layer 5
        self.add_buffer("add_norm1_output", act_size)  # After layer 7
        self.add_buffer("ffn_output", act_size)  # After layer 9-12

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
        self.add_kernel(
            "encoder_mha_out_proj",
            self.combined_xclbin,
            self.mha_out_proj_xclbin.kernel_name,
            self.mha_out_proj_insts,
        )
        self.add_kernel(
            "encoder_add_norm1",
            self.combined_xclbin,
            self.add_norm1_xclbin.kernel_name,
            self.add_norm1_insts,
        )
        self.add_kernel(
            "encoder_ffn",
            self.combined_xclbin,
            self.ffn_xclbin.kernel_name,
            self.ffn_insts,
        )
        self.add_kernel(
            "encoder_add_norm2",
            self.combined_xclbin,
            self.add_norm2_xclbin.kernel_name,
            self.add_norm2_insts,
        )
        logging.info(
            f"Finished setting up {len(self.kernels)} BERT Encoder runtime kernels."
        )

        # Build runlist for all layers
        # Q/K/V projection
        self.add_to_runlist(
            "encoder_qkvo_proj",
            "input",
            "qkv_weight",
            "q_output",
            "k_output",
            "v_output",
        )
        self.add_to_runlist(
            "encoder_mha_out_proj",
            "attn_output_weight",
            "q_output",
            "k_output",
            "v_output",
            "mha_out_proj_output",
        )
        next_output = "mha_out_proj_output"
        # Pipelined add & norm, 2nd input is for residual connection
        self.add_to_runlist(
            "encoder_add_norm1",
            next_output,
            "input",
            "add_norm1_output",
        )
        next_output = "add_norm1_output"
        # Pipelined FFN
        self.add_to_runlist(
            "encoder_ffn",
            next_output,
            "ffn_up_weight",
            "ffn_down_weight",
            "ffn_output",
        )
        next_output = "ffn_output"
        # Second Pipelined add & norm, 2nd input is for residual connection
        self.add_to_runlist(
            "encoder_add_norm2", next_output, "add_norm1_output", "output"
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
        # Flatten inputs for AIE processing
        x_flat = x.view(-1)

        # Verify input size matches expected dimensions
        expected_size = self.seq_len * self.hidden_size
        assert x_flat.shape[0] == expected_size

        self.write_buffer("input", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output",
            (self.seq_len, self.hidden_size),
            dtype=bfloat16,
        ).view(x.shape)

        return result
