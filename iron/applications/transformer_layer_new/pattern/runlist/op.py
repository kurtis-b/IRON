# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import numpy as np
from ml_dtypes import bfloat16
import math

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.gemm.op import AIEGEMM
from iron.operators.softmax.op import AIESoftmax
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.gelu.op import AIEGELU
from iron.operators.transpose.op import AIETranspose
from iron.common.utils import torch_to_numpy


def default_runlist_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    num_aie_columns=8,
):
    head_dim = hidden_size // num_heads
    query_block_size = _resolve_query_block_size(seq_len)
    attn_scratch_size = query_block_size * seq_len * num_heads
    eltwise_mul_tile_size = attn_scratch_size // (num_aie_columns * 2)
    eltwise_add_tile_size = (seq_len * hidden_size) // (num_aie_columns * 2)
    gelu_tile_size = (seq_len * intermediate_size) // (num_aie_columns * 2)
    return {
        "qkvo_proj": {
            "M": seq_len,
            "K": hidden_size,
            "N": hidden_size,
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": num_aie_columns,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "k_transpose": {
            "M": seq_len,
            "N": hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
            "m": 64,
            "n": 96,
            "s": 8,
        },
        "attn_scores": {
            "M": query_block_size,
            "K": head_dim,
            "N": seq_len,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "num_aie_columns": num_aie_columns,
            "batch_A": (num_heads, 1),
            "batch_B": (num_heads, 1),
            "batch_C": (num_heads, 0),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "attn_scale": {
            "size": attn_scratch_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
            "tile_size": min(
                math.gcd(4096, eltwise_mul_tile_size), eltwise_mul_tile_size
            ),
            "scalar_broadcast": math.sqrt(1.0 / head_dim),
        },
        "attn_softmax": {
            "rows": query_block_size * num_heads,
            "cols": seq_len,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
        },
        "attn_output": {
            "M": query_block_size,
            "K": seq_len,
            "N": head_dim,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 4,
            "batch_A": (num_heads, 0),
            "batch_B": (num_heads, 1),
            "batch_C": (num_heads, 1),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "ln1": {
            "size": seq_len * hidden_size,
            "tile_size": hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
        },
        "add": {
            "size": seq_len * hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
            "tile_size": min(
                math.gcd(4096, eltwise_add_tile_size), eltwise_add_tile_size
            ),
        },
        "up_proj": {
            "M": seq_len,
            "K": hidden_size,
            "N": intermediate_size,
            "tile_m": 64,
            "tile_k": 48,
            "tile_n": 96,
            "num_aie_columns": num_aie_columns,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "gelu": {
            "size": seq_len * intermediate_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
            "tile_size": min(math.gcd(4096, gelu_tile_size), gelu_tile_size),
        },
        "down_proj": {
            "M": seq_len,
            "K": intermediate_size,
            "N": hidden_size,
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": num_aie_columns,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "ln2": {
            "size": seq_len * hidden_size,
            "tile_size": hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": 2,
        },
    }


def _use_blocked_attention(seq_len: int) -> bool:
    return seq_len >= 16384


def _resolve_query_block_size(seq_len: int) -> int:
    if seq_len == 16384:
        return 4096
    return 256 if _use_blocked_attention(seq_len) else seq_len


def resolve_runlist_operator_config(
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
        for name, config in default_runlist_operator_config(
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
            raise ValueError(f"Unsupported runlist operator config: {name}")
        resolved[name].update(overrides)
    return resolved


class AIETransformerRunlist(AIEOperatorBase):
    """
    AIE-accelerated Transformer Layer using runlist-based of separate nodes.

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
        self.use_blocked_attention = _use_blocked_attention(seq_len)
        self.use_long_seq_fallback = False
        self.query_block_size = _resolve_query_block_size(seq_len)
        if self.use_blocked_attention and seq_len % self.query_block_size != 0:
            raise AIEOperatorConstraintError(
                "Blocked runlist attention requires seq_len divisible by "
                f"{self.query_block_size}; got seq_len={seq_len}"
            )
        self.query_block_count = seq_len // self.query_block_size

        # Weights to be set by user (separate Q/K/V weights)
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ln1_weight = ln1_weight
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln2_weight = ln2_weight
        self.operator_config = resolve_runlist_operator_config(
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
        # Residual connection
        self.add_xclbin = None
        self.add_insts = None
        # Layer normalization
        self.ln1_xclbin = None
        self.ln1_insts = None
        self.ln2_xclbin = None
        self.ln2_insts = None
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

    def _iter_query_blocks(self):
        for block_index, q_start in enumerate(
            range(0, self.seq_len, self.query_block_size)
        ):
            yield block_index, q_start

    @staticmethod
    def _block_suffix(block_index):
        return f"block_{block_index:03d}"

    def set_up_artifacts(self):
        """Set up artifacts for the encoder layer components using 13 individual layers."""
        artifacts = []
        device_str = self.context.device_manager.device_str()

        kernel_id = 0x801

        prefix_base = f"encoder_runlist_"
        # Q/K/V/O projection kernel
        qkvo_proj_kwargs = dict(self.operator_config["qkvo_proj"])
        qkvo_proj_kwargs.update({"context": self.context, "skip_add_to_list": True})
        qkvo_proj = AIEGEMM(**qkvo_proj_kwargs)
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

        # K Transpose kernel (transpose K matrix)
        k_transpose_kwargs = dict(self.operator_config["k_transpose"])
        k_transpose_kwargs.update({"context": self.context, "skip_add_to_list": True})
        k_transpose = AIETranspose(**k_transpose_kwargs)
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
        attn_scores_kwargs = dict(self.operator_config["attn_scores"])
        attn_scores_kwargs.update({"context": self.context, "skip_add_to_list": True})
        if self.use_blocked_attention:
            attn_scores_kwargs.update(
                {"input_a_buffer_shape": (self.seq_len, self.hidden_size)}
            )
        attn_scores = AIEGEMM(**attn_scores_kwargs)
        self.attn_scores_xclbin, self.attn_scores_insts = attn_scores.get_artifacts(
            prefix=f"{prefix_base}attn_scores_"
        )
        self.attn_scores_xclbin.xclbin_input = self.k_transpose_xclbin
        self.attn_scores_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_scores",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_scores_xclbin.kernel_name = "encoder_attn_scores"
        self.attn_scores_xclbin.depends += [self.k_transpose_xclbin]
        if self.use_blocked_attention:
            setattr(self, "attn_scores_block_000_insts", self.attn_scores_insts)
            artifacts.append(self.attn_scores_insts)
            for block_index, q_start in self._iter_query_blocks():
                if block_index == 0:
                    continue
                block_suffix = self._block_suffix(block_index)
                block_attn_scores = AIEGEMM(
                    **{
                        **attn_scores_kwargs,
                        "input_a_offset": q_start * self.hidden_size,
                    }
                )
                block_insts = block_attn_scores.get_insts_artifact(
                    prefix=f"{prefix_base}attn_scores_{block_suffix}_",
                    xclbin_input=self.attn_scores_xclbin,
                    kernel_name=self.attn_scores_xclbin.kernel_name,
                )
                setattr(self, f"attn_scores_{block_suffix}_insts", block_insts)
                artifacts.append(block_insts)
        else:
            artifacts.append(self.attn_scores_insts)
        kernel_id += 1

        # Attention score scaling (Multiplication per attention score)
        attn_scale_kwargs = dict(self.operator_config["attn_scale"])
        attn_scale_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.attn_scale_xclbin, self.attn_scale_insts = AIEElementwiseMul(
            **attn_scale_kwargs
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
        attn_softmax_kwargs = dict(self.operator_config["attn_softmax"])
        attn_softmax_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.attn_softmax_xclbin, self.attn_softmax_insts = AIESoftmax(
            **attn_softmax_kwargs
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
        attn_output_kwargs = dict(self.operator_config["attn_output"])
        attn_output_kwargs.update({"context": self.context, "skip_add_to_list": True})
        if self.use_blocked_attention:
            attn_output_kwargs.update(
                {"output_c_buffer_shape": (self.seq_len, self.hidden_size)}
            )
        attn_output = AIEGEMM(**attn_output_kwargs)
        self.attn_output_xclbin, self.attn_output_insts = attn_output.get_artifacts(
            prefix=f"{prefix_base}attn_output_"
        )
        self.attn_output_xclbin.xclbin_input = self.attn_softmax_xclbin
        self.attn_output_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_attn_output",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.attn_output_xclbin.kernel_name = "encoder_attn_output"
        self.attn_output_xclbin.depends += [self.attn_softmax_xclbin]
        if self.use_blocked_attention:
            setattr(self, "attn_output_block_000_insts", self.attn_output_insts)
            artifacts.append(self.attn_output_insts)
            for block_index, q_start in self._iter_query_blocks():
                if block_index == 0:
                    continue
                block_suffix = self._block_suffix(block_index)
                block_attn_output = AIEGEMM(
                    **{
                        **attn_output_kwargs,
                        "output_c_offset": q_start * self.hidden_size,
                    }
                )
                block_insts = block_attn_output.get_insts_artifact(
                    prefix=f"{prefix_base}attn_output_{block_suffix}_",
                    xclbin_input=self.attn_output_xclbin,
                    kernel_name=self.attn_output_xclbin.kernel_name,
                )
                setattr(self, f"attn_output_{block_suffix}_insts", block_insts)
                artifacts.append(block_insts)
        else:
            artifacts.append(self.attn_output_insts)
        kernel_id += 1
        next_dep = self.attn_output_xclbin

        # Layer normalization kernel
        ln1_kwargs = dict(self.operator_config["ln1"])
        ln1_kwargs.update(
            {
                "weights": self.ln1_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.ln1_xclbin, self.ln1_insts = AIELayerNorm(**ln1_kwargs).get_artifacts(
            prefix=f"{prefix_base}ln1_"
        )
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
        add_kwargs = dict(self.operator_config["add"])
        add_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.add_xclbin, self.add_insts = AIEElementwiseAdd(**add_kwargs).get_artifacts(
            prefix=f"{prefix_base}add_"
        )
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
        # Up projection (GEMM with Up projection weight)
        up_proj_kwargs = dict(self.operator_config["up_proj"])
        up_proj_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.up_proj_xclbin, self.up_proj_insts = AIEGEMM(
            **up_proj_kwargs
        ).get_artifacts(prefix=f"{prefix_base}up_proj_")
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
        gelu_kwargs = dict(self.operator_config["gelu"])
        gelu_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.gelu_xclbin, self.gelu_insts = AIEGELU(**gelu_kwargs).get_artifacts(
            prefix=f"{prefix_base}gelu_"
        )
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
        down_proj_kwargs = dict(self.operator_config["down_proj"])
        down_proj_kwargs.update({"context": self.context, "skip_add_to_list": True})
        self.down_proj_xclbin, self.down_proj_insts = AIEGEMM(
            **down_proj_kwargs
        ).get_artifacts(prefix=f"{prefix_base}down_proj_")
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
        # Second Layer normalization kernel
        ln2_kwargs = dict(self.operator_config["ln2"])
        ln2_kwargs.update(
            {
                "weights": self.ln2_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.ln2_xclbin, self.ln2_insts = AIELayerNorm(**ln2_kwargs).get_artifacts(
            prefix=f"{prefix_base}ln2_"
        )
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
        attn_scratch_size = self.query_block_size * self.seq_len * self.num_heads

        # Input buffer
        self.add_buffer("input", act_size)

        # Weight buffers (separate Q/K/V weights)
        self.add_buffer(
            "q_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.q_weight) if self.q_weight is not None else None
            ),
        )
        self.add_buffer(
            "k_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.k_weight) if self.k_weight is not None else None
            ),
        )
        self.add_buffer(
            "v_weight",
            self.hidden_size * self.hidden_size,
            static_data=(
                torch_to_numpy(self.v_weight) if self.v_weight is not None else None
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

        # Intermediate buffers for all layers
        self.add_buffer("q_output", act_size)  # After layer 1a
        self.add_buffer("k_output", act_size)  # After layer 1b
        self.add_buffer("v_output", act_size)  # After layer 1c
        self.add_buffer("k_transposed", act_size)  # After K transpose
        self.add_buffer("attn_scores_output", attn_scratch_size)  # After layer 2
        self.add_buffer("attn_scaled_output", attn_scratch_size)  # After layer 3
        self.add_buffer("attn_weights_output", attn_scratch_size)  # After layer 4
        self.add_buffer("attn_heads_output", act_size)  # After layer 5
        self.add_buffer("output_proj_output", act_size)  # After layer 6
        self.add_buffer("add1_output", act_size)  # After residual add 1
        self.add_buffer("ln1_output", act_size)  # After layer norm 1
        self.add_buffer(
            "up_proj_output", self.seq_len * self.intermediate_size
        )  # After layer 9
        self.add_buffer(
            "gelu_output", self.seq_len * self.intermediate_size
        )  # After layer 10
        self.add_buffer("down_proj_output", act_size)  # After layer 11
        self.add_buffer("add2_output", act_size)  # After residual add 2

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
            "encoder_k_transpose",
            self.combined_xclbin,
            self.k_transpose_xclbin.kernel_name,
            self.k_transpose_insts,
        )
        if self.use_blocked_attention:
            for block_index, _ in self._iter_query_blocks():
                block_suffix = self._block_suffix(block_index)
                self.add_kernel(
                    f"encoder_attn_scores_{block_suffix}",
                    self.combined_xclbin,
                    self.attn_scores_xclbin.kernel_name,
                    getattr(self, f"attn_scores_{block_suffix}_insts"),
                )
        else:
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
        if self.use_blocked_attention:
            for block_index, _ in self._iter_query_blocks():
                block_suffix = self._block_suffix(block_index)
                self.add_kernel(
                    f"encoder_attn_output_{block_suffix}",
                    self.combined_xclbin,
                    self.attn_output_xclbin.kernel_name,
                    getattr(self, f"attn_output_{block_suffix}_insts"),
                )
        else:
            self.add_kernel(
                "encoder_attn_output",
                self.combined_xclbin,
                self.attn_output_xclbin.kernel_name,
                self.attn_output_insts,
            )
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
        # Transpose K matrix
        self.add_to_runlist("encoder_k_transpose", "k_output", "k_transposed")
        if self.use_blocked_attention:
            for block_index, _ in self._iter_query_blocks():
                block_suffix = self._block_suffix(block_index)
                self.add_to_runlist(
                    f"encoder_attn_scores_{block_suffix}",
                    "q_output",
                    "k_transposed",
                    "attn_scores_output",
                )
                self.add_to_runlist(
                    "encoder_attn_scale",
                    "attn_scores_output",
                    "attn_scaled_output",
                )
                self.add_to_runlist(
                    "encoder_attn_softmax",
                    "attn_scaled_output",
                    "attn_weights_output",
                )
                self.add_to_runlist(
                    f"encoder_attn_output_{block_suffix}",
                    "attn_weights_output",
                    "v_output",
                    "attn_heads_output",
                )
        else:
            # Attention score calculations
            self.add_to_runlist(
                "encoder_attn_scores",
                "q_output",
                "k_transposed",
                "attn_scores_output",
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
        # Output projection
        self.add_to_runlist(
            "encoder_qkvo_proj",
            "attn_heads_output",
            "attn_output_weight",
            "output_proj_output",
        )
        # Residual connection
        self.add_to_runlist("encoder_add", "input", "output_proj_output", "add1_output")
        # Layer normalization
        self.add_to_runlist("encoder_ln1", "add1_output", "ln1_output")
        next_output = "ln1_output"
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
        # Residual connection
        self.add_to_runlist(
            "encoder_add", "ln1_output", "down_proj_output", "add2_output"
        )
        # Layer normalization
        self.add_to_runlist("encoder_ln2", "add2_output", "output")

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
        return self.read_buffer_as_torch(
            "output",
            (self.seq_len, self.hidden_size),
            dtype=bfloat16,
        ).view(x.shape)
