# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


class AIEAddNormFFNAddNorm(AIEOperatorBase):
    """
    Thesis-v2 Block 3 operator.

    This block uses the pipelined AddNorm+FFN design, which pipelines the
    first Add & Norm stage into the up-projection workers.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        context,
        topology_id: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        config = addnorm_ffn_addnorm_design(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topology_id=topology_id,
        )
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
        self.compile_rows = int(config["compile_rows"])

        self.M = self.compile_rows
        self.K = hidden_size
        self.N = intermediate_size

        self.weight_up_proj = torch.zeros((self.K, self.N), dtype=torch.bfloat16).T
        self.weight_down_proj = torch.zeros((self.N, self.K), dtype=torch.bfloat16).T
        self.ln2_weight = torch.ones(hidden_size, dtype=torch.bfloat16)

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix: str = "addnorm_ffn_addnorm_"):
        operator_dir = Path(__file__).parent
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        file_name_total_base = (
            f"{prefix}{self.M}x{self.K}x{self.N}_"
            f"{self.tile_m}x{self.tile_k}x{self.tile_n}_"
            f"{self.down_proj_depth}_"
            f"{self.parallel_seq}_"
            f"{self.parallel_int_dim}_"
            f"None_"
            f"{self.gelu_stage}"
        )

        ln2_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln2_weight_{self.K}.npy"
        )
        np.save(ln2_weight_file_name, torch_to_numpy(self.ln2_weight))

        kernel_archive = f"anffn_{self.tile_m}x{self.tile_k}x{self.tile_n}.a"

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
                "ln2_weight_file": ln2_weight_file_name,
                "stage_only": None,
                "archive": kernel_archive,
                "generate_taps": False,
                "n_aie_cols": self.num_aie_columns,
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
                        KernelObjectArtifact.new(
                            f"ln_passThrough_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
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
                                "passThroughLine": "ln_passThroughLine",
                                "passThroughTile": "ln_passThroughTile",
                            },
                        ),
                    ],
                ),
            ],
            extra_flags=["--dynamic-objFifos", "--profile", "-v", "--progress"],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_total_base}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos", "--profile", "-v", "--progress"],
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

        static_weights_up_proj = torch_to_numpy(self.weight_up_proj.T)
        static_weights_down_proj = torch_to_numpy(self.weight_down_proj.T)

        self.add_buffer("AR", 2 * self.M * self.K)
        self.add_buffer("B_Up", self.K * self.N, static_data=static_weights_up_proj)
        self.add_buffer(
            "B_Down",
            self.K * self.N,
            static_data=static_weights_down_proj,
        )
        self.add_buffer("C", self.M * self.K)
        self.add_to_runlist("addnorm_ffn_addnorm", "AR", "B_Up", "B_Down", "C")

    def forward(
        self,
        attention_output: torch.Tensor,
        residual: torch.Tensor,
        B_Up: torch.Tensor | None = None,
        B_Down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Block 3."""
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
            K == K2 and K == K3 and N == N2 and M > 0 and K <= self.K and N <= self.N
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: incompatible tensor shape(s)"
            )

        attention_output_padded = self._pad_A(torch_to_numpy(attention_output))
        residual_padded = self._pad_A(torch_to_numpy(residual))
        if B_Up is not None:
            B_Up_padded = self._pad_B(torch_to_numpy(B_Up), b_col_maj=False)
        else:
            B_Up_padded = None
        if B_Down is not None:
            B_Down_padded = self._pad_B(torch_to_numpy(B_Down), b_col_maj=True)
        else:
            B_Down_padded = None

        logging.debug(
            "Executing Block 3 for dimensions M=%s, K=%s, N=%s using compiled operator "
            "with M=%s, K=%s, N=%s",
            M,
            K,
            N,
            self.M,
            self.K,
            self.N,
        )

        result_padded = np.zeros((M, self.K), dtype=attention_output_padded.dtype)
        for M_lo in range(0, M, self.M):
            attention_output_part = attention_output_padded[M_lo : M_lo + self.M, :]
            residual_part = residual_padded[M_lo : M_lo + self.M, :]
            result_part = self._execute_aie_operation(
                attention_output_part,
                residual_part,
                B_Up_padded,
                B_Down_padded,
            )
            max_M = min(M_lo + self.M, M)
            result_padded[M_lo:max_M, :] = result_part[: max_M - M_lo, :]

        result = numpy_to_torch(result_padded[:M, :K])
        return result.view(expected_output_shape)

    def _pad_A(self, A_np: np.ndarray) -> np.ndarray:
        M, K = A_np.shape
        if M % self.M == 0 and K == self.K:
            return A_np

        M_multiple = (M + self.M - 1) // self.M * self.M
        A_padded = np.zeros((M_multiple, self.K), dtype=A_np.dtype)
        A_padded[:M, :K] = A_np
        return A_padded

    def _pad_B(self, B_np: np.ndarray, b_col_maj: bool) -> np.ndarray:
        if b_col_maj:
            K, N = B_np.shape
            dim1_end = N
            dim2_end = K
        else:
            N, K = B_np.shape
            dim1_end = K
            dim2_end = N
        if K == self.K and N == self.N:
            return B_np

        B_padded = np.zeros((dim1_end, dim2_end), dtype=B_np.dtype)
        B_padded[:dim1_end, :dim2_end] = B_np
        return B_padded

    def _execute_aie_operation(
        self,
        attention_output_np: np.ndarray,
        residual_np: np.ndarray,
        B_Up_np: np.ndarray | None = None,
        B_Down_np: np.ndarray | None = None,
    ) -> np.ndarray:
        M, K = attention_output_np.shape
        K2, N = B_Up_np.shape if B_Up_np is not None else self.weight_up_proj.T.shape
        N2, K3 = (
            B_Down_np.shape if B_Down_np is not None else self.weight_down_proj.T.shape
        )

        assert M == self.M
        assert K == K2 and K == K3 and K == self.K
        assert N == N2 and N == self.N

        ar_np = np.concatenate((attention_output_np, residual_np), axis=0)
        self.write_buffer("AR", ar_np)
        if B_Up_np is not None:
            self.write_buffer("B_Up", B_Up_np)
        if B_Down_np is not None:
            self.write_buffer("B_Down", B_Down_np)
        self.run_runlist()
        result_np = self.read_buffer("C", shape=(M, K), dtype=bfloat16)

        if np.isnan(result_np).any():
            nan_count = np.isnan(result_np).sum()
            total_count = result_np.size
            raise RuntimeError(
                f"AIE execution returned {nan_count}/{total_count} NaN values."
            )

        return result_np
