# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import torch
from ml_dtypes import bfloat16

from iron.common import AIEOperatorBase
from iron.common.utils import torch_to_numpy
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.ffn.op import AIEFFN
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.qkv_proj.op import AIEQKVProj


def default_hybrid_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    workload_variant="encoder_bert",
    num_aie_columns=8,
):
    head_dim = hidden_size // num_heads
    ffn_config = {
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
    }
    qkv_proj_config = {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "tile_m": 32,
        "tile_k": 64,
        "tile_n": 16,
        "parallel_seq": 1,
        "parallel_emb": num_aie_columns,
    }
    if workload_variant == "decoder_gpt2":
        eltwise_add_tile_size = (seq_len * hidden_size) // (num_aie_columns * 2)
        return {
            "ln1": {
                "size": seq_len * hidden_size,
                "tile_size": hidden_size,
                "num_aie_columns": num_aie_columns,
                "num_channels": 2,
            },
            "qkv_proj": qkv_proj_config,
            "mha_out_proj": {
                "num_heads": num_heads,
                "seq_len": seq_len,
                "d": head_dim,
                "parallel_seq": 1,
                "q_seq_tile": 32,
                "kv_seq_tile": 32,
                "emb_tile": head_dim,
                "parallel_heads": 1,
                "o_proj_acc_depth": 1,
                "is_causal": True,
            },
            "add_norm": {
                "size": seq_len * hidden_size,
                "num_aie_columns": num_aie_columns,
                "tile_size": hidden_size,
            },
            "ffn": ffn_config,
            "add": {
                "size": seq_len * hidden_size,
                "num_aie_columns": num_aie_columns,
                "num_channels": 2,
                "tile_size": min(
                    math.gcd(4096, eltwise_add_tile_size),
                    eltwise_add_tile_size,
                ),
            },
        }
    if workload_variant != "encoder_bert":
        raise ValueError(f"Unsupported hybrid workload_variant: {workload_variant}")
    return {
        "qkv_proj": qkv_proj_config,
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
        "ffn": ffn_config,
        "add_norm2": {
            "size": seq_len * hidden_size,
            "num_aie_columns": num_aie_columns,
            "tile_size": hidden_size,
        },
    }


def resolve_hybrid_operator_config(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    *,
    workload_variant="encoder_bert",
    num_aie_columns=8,
    operator_config=None,
):
    resolved = {
        name: dict(config)
        for name, config in default_hybrid_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            workload_variant=workload_variant,
            num_aie_columns=num_aie_columns,
        ).items()
    }
    if operator_config is None:
        return resolved
    for name, overrides in operator_config.items():
        if name not in resolved:
            raise ValueError(f"Unsupported hybrid operator config: {name}")
        resolved[name].update(overrides)
    return resolved


class AIETransformerHybrid(AIEOperatorBase):
    """AIE-accelerated transformer block using the hybrid implementation."""

    def __init__(
        self,
        seq_len,
        hidden_size,
        intermediate_size,
        num_heads,
        num_aie_columns=8,
        ln1_weight=None,
        ln2_weight=None,
        workload_variant="encoder_bert",
        operator_config=None,
        context=None,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_aie_columns = num_aie_columns
        self.workload_variant = workload_variant
        self.head_dim = hidden_size // num_heads

        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.attn_output_weight = None
        self.ln1_weight = ln1_weight
        self.ffn_up_weight = None
        self.ffn_down_weight = None
        self.ln2_weight = ln2_weight
        self.operator_config = resolve_hybrid_operator_config(
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            workload_variant=workload_variant,
            num_aie_columns=num_aie_columns,
            operator_config=operator_config,
        )

        self.combined_xclbin = None
        self.qkvo_proj_xclbin = None
        self.qkvo_proj_insts = None
        self.mha_out_proj_xclbin = None
        self.mha_out_proj_insts = None
        self.add_norm1_xclbin = None
        self.add_norm1_insts = None
        self.add_norm2_xclbin = None
        self.add_norm2_insts = None
        self.add_norm_xclbin = None
        self.add_norm_insts = None
        self.ffn_xclbin = None
        self.ffn_insts = None
        self.ln1_xclbin = None
        self.ln1_insts = None
        self.add_xclbin = None
        self.add_insts = None
        self.reset_buffer_names = ()

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        if self.workload_variant == "decoder_gpt2":
            self._set_up_decoder_artifacts()
            return
        self._set_up_encoder_artifacts()

    def _set_up_encoder_artifacts(self):
        artifacts = []
        kernel_id = 0x801
        prefix_base = "encoder_hybrid_"

        qkvo_proj = AIEQKVProj(
            **{
                **self.operator_config["qkv_proj"],
                "context": self.context,
                "skip_add_to_list": True,
            }
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

        mha_out_proj = AIEMHAOutProj(
            **{
                **self.operator_config["mha_out_proj"],
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
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

        self.add_norm1_xclbin, self.add_norm1_insts = AIEAddAndNorm(
            **{
                **self.operator_config["add_norm1"],
                "weights": self.ln1_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}add_norm1_")
        self.add_norm1_xclbin.xclbin_input = next_dep
        self.add_norm1_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_add_norm1",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add_norm1_xclbin.kernel_name = "encoder_add_norm1"
        self.add_norm1_xclbin.depends += [self.qkvo_proj_xclbin, next_dep]
        artifacts.append(self.add_norm1_insts)
        next_dep = self.add_norm1_xclbin
        kernel_id += 1

        self.ffn_xclbin, self.ffn_insts = AIEFFN(
            **{
                **self.operator_config["ffn"],
                "context": self.context,
                "skip_add_to_list": True,
            }
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

        self.add_norm2_xclbin, self.add_norm2_insts = AIEAddAndNorm(
            **{
                **self.operator_config["add_norm2"],
                "weights": self.ln2_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}add_norm2_")
        self.add_norm2_xclbin.xclbin_input = next_dep
        self.add_norm2_xclbin.extra_flags += [
            "--xclbin-instance-name=encoder_add_norm2",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add_norm2_xclbin.kernel_name = "encoder_add_norm2"
        self.add_norm2_xclbin.depends += [next_dep]
        artifacts.append(self.add_norm2_xclbin)
        artifacts.append(self.add_norm2_insts)
        self.combined_xclbin = self.add_norm2_xclbin

        self.add_artifacts(artifacts)
        logging.info("Finished setting up %d encoder hybrid artifacts.", len(artifacts))

    def _set_up_decoder_artifacts(self):
        artifacts = []
        kernel_id = 0x801
        prefix_base = "decoder_hybrid_"

        self.ln1_xclbin, self.ln1_insts = AIELayerNorm(
            **{
                **self.operator_config["ln1"],
                "weights": self.ln1_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}ln1_")
        self.ln1_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_ln1",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.ln1_xclbin.kernel_name = "decoder_ln1"
        artifacts.append(self.ln1_insts)
        kernel_id += 1

        qkvo_proj = AIEQKVProj(
            **{
                **self.operator_config["qkv_proj"],
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        self.qkvo_proj_xclbin, self.qkvo_proj_insts = qkvo_proj.get_artifacts(
            prefix=f"{prefix_base}qkvo_proj_"
        )
        self.qkvo_proj_xclbin.xclbin_input = self.ln1_xclbin
        self.qkvo_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_qkvo_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.qkvo_proj_xclbin.kernel_name = "decoder_qkvo_proj"
        self.qkvo_proj_xclbin.depends += [self.ln1_xclbin]
        artifacts.append(self.qkvo_proj_insts)
        kernel_id += 1

        mha_out_proj = AIEMHAOutProj(
            **{
                **self.operator_config["mha_out_proj"],
                "context": self.context,
                "skip_add_to_list": True,
            }
        )
        mha_out_proj.set_up_artifacts()
        self.mha_out_proj_xclbin = mha_out_proj.xclbin_artifact
        self.mha_out_proj_insts = mha_out_proj.insts_artifact
        self.mha_out_proj_xclbin.xclbin_input = self.qkvo_proj_xclbin
        self.mha_out_proj_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_mha_out_proj",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.mha_out_proj_xclbin.kernel_name = "decoder_mha_out_proj"
        self.mha_out_proj_xclbin.depends += [self.qkvo_proj_xclbin]
        artifacts.append(self.mha_out_proj_insts)
        kernel_id += 1
        next_dep = self.mha_out_proj_xclbin

        self.add_norm_xclbin, self.add_norm_insts = AIEAddAndNorm(
            **{
                **self.operator_config["add_norm"],
                "weights": self.ln2_weight,
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}add_norm_")
        self.add_norm_xclbin.xclbin_input = next_dep
        self.add_norm_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_add_norm",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add_norm_xclbin.kernel_name = "decoder_add_norm"
        self.add_norm_xclbin.depends += [self.qkvo_proj_xclbin, next_dep]
        artifacts.append(self.add_norm_insts)
        next_dep = self.add_norm_xclbin
        kernel_id += 1

        self.ffn_xclbin, self.ffn_insts = AIEFFN(
            **{
                **self.operator_config["ffn"],
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}ffn_")
        self.ffn_xclbin.xclbin_input = next_dep
        self.ffn_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_ffn",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.ffn_xclbin.kernel_name = "decoder_ffn"
        self.ffn_xclbin.depends += [next_dep]
        artifacts.append(self.ffn_insts)
        next_dep = self.ffn_xclbin
        kernel_id += 1

        self.add_xclbin, self.add_insts = AIEElementwiseAdd(
            **{
                **self.operator_config["add"],
                "context": self.context,
                "skip_add_to_list": True,
            }
        ).get_artifacts(prefix=f"{prefix_base}add_")
        self.add_xclbin.xclbin_input = next_dep
        self.add_xclbin.extra_flags += [
            "--xclbin-instance-name=decoder_add",
            f"--xclbin-kernel-id={hex(kernel_id)}",
        ]
        self.add_xclbin.kernel_name = "decoder_add"
        self.add_xclbin.depends += [next_dep]
        artifacts.append(self.add_xclbin)
        artifacts.append(self.add_insts)
        self.combined_xclbin = self.add_xclbin

        self.add_artifacts(artifacts)
        logging.info("Finished setting up %d decoder hybrid artifacts.", len(artifacts))

    def set_up_runtime(self):
        if self.workload_variant == "decoder_gpt2":
            self._set_up_decoder_runtime()
            return
        self._set_up_encoder_runtime()

    def _add_shared_weight_buffers(self):
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

    def _set_up_encoder_runtime(self):
        act_size = self.seq_len * self.hidden_size
        self.add_buffer("input", act_size)
        self._add_shared_weight_buffers()
        self.add_buffer(
            "ln1_weight",
            self.hidden_size,
            static_data=(
                torch_to_numpy(self.ln1_weight) if self.ln1_weight is not None else None
            ),
        )
        self.add_buffer(
            "ln2_weight",
            self.hidden_size,
            static_data=(
                torch_to_numpy(self.ln2_weight) if self.ln2_weight is not None else None
            ),
        )
        self.add_buffer("q_output", act_size)
        self.add_buffer("k_output", act_size)
        self.add_buffer("v_output", act_size)
        self.add_buffer("mha_out_proj_output", act_size)
        self.add_buffer("add_norm1_output", act_size)
        self.add_buffer("ffn_output", act_size)
        self.add_buffer("output", act_size)

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
        self.add_to_runlist(
            "encoder_add_norm1",
            "mha_out_proj_output",
            "input",
            "add_norm1_output",
        )
        self.add_to_runlist(
            "encoder_ffn",
            "add_norm1_output",
            "ffn_up_weight",
            "ffn_down_weight",
            "ffn_output",
        )
        self.add_to_runlist(
            "encoder_add_norm2",
            "ffn_output",
            "add_norm1_output",
            "output",
        )
        self.reset_buffer_names = (
            "q_output",
            "k_output",
            "v_output",
            "mha_out_proj_output",
            "ffn_output",
        )

    def _set_up_decoder_runtime(self):
        act_size = self.seq_len * self.hidden_size
        self.add_buffer("input", act_size)
        self._add_shared_weight_buffers()
        self.add_buffer("ln1_output", act_size)
        self.add_buffer("q_output", act_size)
        self.add_buffer("k_output", act_size)
        self.add_buffer("v_output", act_size)
        self.add_buffer("mha_out_proj_output", act_size)
        self.add_buffer("residual_output", act_size)
        self.add_buffer("add_norm_output", act_size)
        self.add_buffer("ffn_output", act_size)
        self.add_buffer("output", act_size)

        self.add_kernel(
            "decoder_ln1",
            self.combined_xclbin,
            self.ln1_xclbin.kernel_name,
            self.ln1_insts,
        )
        self.add_kernel(
            "decoder_qkvo_proj",
            self.combined_xclbin,
            self.qkvo_proj_xclbin.kernel_name,
            self.qkvo_proj_insts,
        )
        self.add_kernel(
            "decoder_mha_out_proj",
            self.combined_xclbin,
            self.mha_out_proj_xclbin.kernel_name,
            self.mha_out_proj_insts,
        )
        self.add_kernel(
            "decoder_add_norm",
            self.combined_xclbin,
            self.add_norm_xclbin.kernel_name,
            self.add_norm_insts,
        )
        self.add_kernel(
            "decoder_ffn",
            self.combined_xclbin,
            self.ffn_xclbin.kernel_name,
            self.ffn_insts,
        )
        self.add_kernel(
            "decoder_add",
            self.combined_xclbin,
            self.add_xclbin.kernel_name,
            self.add_insts,
        )

        self.add_to_runlist("decoder_ln1", "input", "ln1_output")
        self.add_to_runlist(
            "decoder_qkvo_proj",
            "ln1_output",
            "qkv_weight",
            "q_output",
            "k_output",
            "v_output",
        )
        self.add_to_runlist(
            "decoder_mha_out_proj",
            "attn_output_weight",
            "q_output",
            "k_output",
            "v_output",
            "mha_out_proj_output",
        )
        self.add_to_runlist(
            "decoder_add",
            "mha_out_proj_output",
            "input",
            "residual_output",
        )
        self.add_to_runlist(
            "decoder_add_norm",
            "mha_out_proj_output",
            "input",
            "add_norm_output",
        )
        self.add_to_runlist(
            "decoder_ffn",
            "add_norm_output",
            "ffn_up_weight",
            "ffn_down_weight",
            "ffn_output",
        )
        self.add_to_runlist(
            "decoder_add",
            "residual_output",
            "ffn_output",
            "output",
        )
        self.reset_buffer_names = (
            "ln1_output",
            "q_output",
            "k_output",
            "v_output",
            "mha_out_proj_output",
            "residual_output",
            "add_norm_output",
            "ffn_output",
        )

    def forward(self, x, attention_mask=None):
        del attention_mask
        x_flat = x.view(-1)
        expected_size = self.seq_len * self.hidden_size
        assert x_flat.shape[0] == expected_size

        self.write_buffer("input", x_flat)
        self.run_runlist()
        return self.read_buffer_as_torch(
            "output",
            (self.seq_len, self.hidden_size),
            dtype=bfloat16,
        ).view(x.shape)
