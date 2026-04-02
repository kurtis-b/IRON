# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from iron.common.utils import numpy_to_torch, torch_to_numpy
from iron.operators.addnorm_ffn_addnorm.topology import addnorm_ffn_addnorm_design


def _append_debug_flag(extra_flags: list[str], debug_mode: int | None) -> None:
    if debug_mode in (0, 1):
        extra_flags.append(f"-DDEBUG_AIE_KERNELS={debug_mode}")


class AIEAddNormFFNAddNorm(AIEOperatorBase):
    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        context,
        topology_id: str | None = None,
        stage_only: int | None = None,
        debug_mode: int = -1,
    ) -> None:
        config = addnorm_ffn_addnorm_design(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topology_id=topology_id,
        )

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.M = seq_len
        self.K = hidden_size
        self.N = intermediate_size

        self.topology_id = str(config["topology_id"])
        self.topology_family = str(config["topology_family"])
        self.parallel_seq = int(config["parallel_seq"])
        self.parallel_int_dim = int(config["parallel_int_dim"])
        self.tile_m = int(config["tile_m"])
        self.tile_k = int(config["tile_k"])
        self.tile_n = int(config["tile_n"])
        self.down_proj_depth = int(config["down_proj_depth"])
        self.gelu_stage = int(config["gelu_stage"])
        self.num_aie_columns = int(config["num_aie_columns"])
        self.stage_only = stage_only

        self.weight_up_proj = torch.zeros((self.K, self.N), dtype=torch.bfloat16).T
        self.weight_down_proj = torch.zeros((self.N, self.K), dtype=torch.bfloat16).T
        self.ln1_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
        self.ln2_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
        self.debug_mode = debug_mode

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix: str = "addnorm_ffn_addnorm_"):
        operator_dir = Path(__file__).parent
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        ln1_weight_np = torch_to_numpy(self.ln1_weight)
        ln2_weight_np = torch_to_numpy(self.ln2_weight)
        weights_fingerprint = hashlib.sha1(
            ln1_weight_np.tobytes() + ln2_weight_np.tobytes()
        ).hexdigest()[:10]

        file_name_total_base = (
            f"{prefix}{self.M}x{self.K}x{self.N}_"
            f"{self.tile_m}x{self.tile_k}x{self.tile_n}_"
            f"{self.down_proj_depth}_"
            f"{self.parallel_seq}_"
            f"{self.parallel_int_dim}_"
            f"{self.gelu_stage}_"
            f"{self.stage_only}_"
            f"{weights_fingerprint}"
        )

        ln1_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln1_weight.npy"
        )
        np.save(ln1_weight_file_name, ln1_weight_np)
        ln2_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln2_weight.npy"
        )
        np.save(ln2_weight_file_name, ln2_weight_np)

        kernel_archive = (
            f"staged_block3_{self.tile_m}x{self.tile_k}x{self.tile_n}"
            f"_ps{self.parallel_seq}_pi{self.parallel_int_dim}.a"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_total_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_addnorm_ffn_addnorm",
            callback_kwargs={
                "dev": device_str,
                "M": self.M,
                "K": self.K,
                "N": self.N,
                "m": self.tile_m,
                "k": self.tile_k,
                "n": self.tile_n,
                "down_proj_depth": self.down_proj_depth,
                "nA_tiles_distributed": self.parallel_seq,
                "nB_tiles_distributed": self.parallel_int_dim,
                "dtype_in_str": "bf16",
                "dtype_out_str": "bf16",
                "emulate_bf16_mmul_with_bfp16": True,
                "trace_size": 0,
                "gelu_stage": self.gelu_stage,
                "stage_only": self.stage_only,
                "ln1_weight_file": ln1_weight_file_name,
                "ln2_weight_file": ln2_weight_file_name,
                "archive": kernel_archive,
                "n_aie_cols": self.num_aie_columns,
                "debug_mode": self.debug_mode,
            },
            requires_context=False,
        )

        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
        ]
        _append_debug_flag(encoder_kernel_flags, self.debug_mode)
        ln1_layer_norm_rename_symbols = {
            "layer_norm": "block3_ln1_layer_norm",
            "layer_norm_rows": "block3_ln1_layer_norm_rows",
        }
        ln1_mul_rename_symbols = {
            "eltwise_mul_bf16_scalar": "block3_ln1_eltwise_mul_bf16_scalar",
            "eltwise_mul_bf16_vector": "block3_ln1_eltwise_mul_bf16_vector",
            "eltwise_mul_bf16_vector_rows": "block3_ln1_eltwise_mul_bf16_vector_rows",
            "eltwise_mul_bf16_broadcasted_scalar": "block3_ln1_eltwise_mul_bf16_broadcasted_scalar",
        }
        ln1_add_rename_symbols = {
            "eltwise_add_bf16_scalar": "block3_ln1_eltwise_add_bf16_scalar",
            "eltwise_add_bf16_vector": "block3_ln1_eltwise_add_bf16_vector",
        }

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_total_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            f"fused_encoder_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "encoder.cc"
                                )
                            ],
                            extra_flags=encoder_kernel_flags,
                        ),
                        KernelObjectArtifact.new(
                            f"block3_ln1_layer_norm_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "layer_norm.cc"
                                )
                            ],
                            rename_symbols=ln1_layer_norm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"block3_ln1_mul_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "generic" / "mul.cc"
                                )
                            ],
                            rename_symbols=ln1_mul_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"block3_ln1_add_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "generic" / "add.cc"
                                )
                            ],
                            rename_symbols=ln1_add_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"ffn_passThrough_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[
                                SourceArtifact.new(
                                    base_dir
                                    / "aie_kernels"
                                    / "generic"
                                    / "passThrough.cc"
                                )
                            ],
                            rename_symbols={
                                "passThroughLine": "ffn_passThroughLine",
                                "passThroughTile": "ffn_passThroughTile",
                            },
                        ),
                    ],
                ),
            ],
            extra_flags=["--dynamic-objFifos"],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_total_base}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        if self.xclbin_artifact is None or self.insts_artifact is None:
            self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        self.add_kernel(
            "addnorm_ffn_addnorm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("A", self.M * self.K)
        self.add_buffer("R", self.M * self.K)
        self.add_buffer("B_stacked", 2 * self.K * self.N)
        self.add_buffer("stage_stacked", 2 * self.M * self.K)
        self.add_buffer("C", self.M * self.K)
        self.add_to_runlist(
            "addnorm_ffn_addnorm",
            "A",
            "R",
            "B_stacked",
            "stage_stacked",
            "C",
        )

    def forward(
        self,
        attention_output: torch.Tensor,
        residual: torch.Tensor,
        B_Up: torch.Tensor | None = None,
        B_Down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B_Up_shape = B_Up.shape if B_Up is not None else self.weight_up_proj.T.shape
        B_Down_shape = (
            B_Down.shape if B_Down is not None else self.weight_down_proj.T.shape
        )
        expected_output_shape = attention_output.shape
        if expected_output_shape != residual.shape:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: attention_output and residual must have the same shape"
            )

        if len(attention_output.shape) > 2:
            attention_output = attention_output.view(-1, attention_output.shape[-1])
        if len(residual.shape) > 2:
            residual = residual.view(-1, residual.shape[-1])
        if B_Up is not None and len(B_Up.shape) > 2:
            B_Up = B_Up.view(-1, B_Up_shape[-1])
        if B_Down is not None and len(B_Down.shape) > 2:
            B_Down = B_Down.view(-1, B_Down_shape[-1])

        M, K = attention_output.shape
        K2, N = B_Up_shape
        N2, K3 = B_Down_shape
        applicable = (
            M == self.M
            and K == self.K
            and K == K2
            and K == K3
            and N == self.N
            and N == N2
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: incompatible tensor shape(s)"
            )

        result_np = self._execute_aie_operation(
            torch_to_numpy(attention_output),
            torch_to_numpy(residual),
            torch_to_numpy(B_Up) if B_Up is not None else None,
            torch_to_numpy(B_Down) if B_Down is not None else None,
        )
        return numpy_to_torch(result_np).view(expected_output_shape)

    def _execute_aie_operation(
        self,
        attention_output_np: np.ndarray,
        residual_np: np.ndarray,
        B_Up_np: np.ndarray | None = None,
        B_Down_np: np.ndarray | None = None,
    ) -> np.ndarray:
        self.write_buffer("A", attention_output_np.reshape(-1))
        self.write_buffer("R", residual_np.reshape(-1))
        up_proj_np = (
            B_Up_np if B_Up_np is not None else torch_to_numpy(self.weight_up_proj.T)
        )
        down_proj_np = (
            B_Down_np
            if B_Down_np is not None
            else torch_to_numpy(self.weight_down_proj.T)
        )
        self.write_buffer(
            "B_stacked",
            np.concatenate([up_proj_np.reshape(-1), down_proj_np.reshape(-1)]),
        )
        self.run_runlist()
        result_np = self.read_buffer("C", shape=(self.M, self.K), dtype=bfloat16)
        if self.stage_only is None and np.isnan(result_np).any():
            raise RuntimeError("AIE execution returned NaN values.")
        return result_np

    def read_staged_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        stage_np = self.read_buffer(
            "stage_stacked",
            shape=(2, self.M, self.K),
            dtype=bfloat16,
        )
        return (
            numpy_to_torch(stage_np[0].copy()),
            numpy_to_torch(stage_np[1].copy()),
        )
