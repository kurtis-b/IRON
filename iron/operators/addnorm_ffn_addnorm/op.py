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


class AIEAddNormFFNAddNorm(AIEOperatorBase):
    """
    Thesis-v2 Block 3 operator.

    This block uses a pipelined AddNorm+FFN design with dedicated first-stage
    AddNorm workers that feed the up-projection stage.
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
        self.ln1_weight = torch.ones(hidden_size, dtype=torch.bfloat16)
        self.ln2_weight = torch.ones(hidden_size, dtype=torch.bfloat16)

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
            f"None_"
            f"{self.gelu_stage}_"
            f"{weights_fingerprint}"
        )

        ln1_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln1_weight_{self.K}.npy"
        )
        np.save(ln1_weight_file_name, ln1_weight_np)
        ln2_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln2_weight_{self.K}.npy"
        )
        np.save(ln2_weight_file_name, ln2_weight_np)

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
                "ln1_weight_file": ln1_weight_file_name,
                "ln2_weight_file": ln2_weight_file_name,
                "archive": kernel_archive,
                "n_aie_cols": self.num_aie_columns,
            },
            requires_context=False,
        )

        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            "-DBUILD_ADDNORM_REPLAY_FASTPATH",
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
                        KernelObjectArtifact.new(
                            f"ln_passThrough_f32_{self.tile_m}x{self.tile_k}x{self.tile_n}.o",
                            extra_flags=["-DBIT_WIDTH=32"],
                            depends=[
                                SourceArtifact.new(
                                    base_dir
                                    / "aie_kernels"
                                    / "generic"
                                    / "passThrough.cc"
                                )
                            ],
                            rename_symbols={
                                "passThroughLine": "ln_passThroughLine_f32",
                                "passThroughTile": "ln_passThroughTile_f32",
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

        self.add_buffer("packed_hidden_residual", 2 * self.M * self.K)
        self.add_buffer("B_Up", self.K * self.N, static_data=static_weights_up_proj)
        self.add_buffer(
            "B_Down",
            self.K * self.N,
            static_data=static_weights_down_proj,
        )
        self.add_buffer("C", self.M * self.K)
        self.add_to_runlist(
            "addnorm_ffn_addnorm",
            "packed_hidden_residual",
            "B_Up",
            "B_Down",
            "C",
        )

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

        attention_output_np = self._pad_A(torch_to_numpy(attention_output))
        residual_np = self._pad_A(torch_to_numpy(residual))
        padded_rows = attention_output_np.shape[0]
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

        result_padded = self._execute_chunked_hidden_residual(
            attention_output_np,
            residual_np,
            B_Up_padded,
            B_Down_padded,
        )

        result = numpy_to_torch(result_padded[:M, :K])
        return result.view(expected_output_shape)

    def forward_packed(
        self,
        packed_hidden_residual: torch.Tensor,
        B_Up: torch.Tensor | None = None,
        B_Down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed_hidden_residual_np, M = self._canonicalize_packed_hidden_residual(
            packed_hidden_residual
        )
        expected_output_shape = (M, self.K)
        M, K = expected_output_shape
        if B_Up is not None and len(B_Up.shape) > 2:
            B_Up = B_Up.view(-1, B_Up.shape[-1])
        if B_Down is not None and len(B_Down.shape) > 2:
            B_Down = B_Down.view(-1, B_Down.shape[-1])
        B_Up_shape = B_Up.shape if B_Up is not None else self.weight_up_proj.T.shape
        B_Down_shape = (
            B_Down.shape if B_Down is not None else self.weight_down_proj.T.shape
        )
        K2, N = B_Up_shape
        N2, K3 = B_Down_shape
        applicable = (
            K == K2 and K == K3 and N == N2 and M > 0 and K <= self.K and N <= self.N
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: incompatible tensor shape(s)"
            )

        packed_hidden_residual_padded, padded_rows = self._pad_packed_hidden_residual(
            packed_hidden_residual_np,
            M,
        )
        attention_output_padded, residual_padded = self._unpack_hidden_residual(
            packed_hidden_residual_padded,
            padded_rows,
        )
        if B_Up is not None:
            B_Up_padded = self._pad_B(torch_to_numpy(B_Up), b_col_maj=False)
        else:
            B_Up_padded = None
        if B_Down is not None:
            B_Down_padded = self._pad_B(torch_to_numpy(B_Down), b_col_maj=True)
        else:
            B_Down_padded = None

        result_padded = self._execute_chunked_hidden_residual(
            attention_output_padded,
            residual_padded,
            B_Up_padded,
            B_Down_padded,
        )

        result = numpy_to_torch(result_padded[:M, :K])
        return result.view(expected_output_shape)

    def _execute_chunked_hidden_residual(
        self,
        attention_output_padded: np.ndarray,
        residual_padded: np.ndarray,
        B_Up_np: np.ndarray | None = None,
        B_Down_np: np.ndarray | None = None,
    ) -> np.ndarray:
        padded_rows, hidden_size = attention_output_padded.shape
        if residual_padded.shape != (padded_rows, hidden_size):
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: attention_output and residual must have the same padded shape"
            )
        if hidden_size != self.K or padded_rows % self.M != 0:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: invalid padded hidden/residual chunking for execution"
            )

        result_padded = np.zeros(
            (padded_rows, self.K), dtype=attention_output_padded.dtype
        )
        for M_lo in range(0, padded_rows, self.M):
            chunk_hidden = attention_output_padded[M_lo : M_lo + self.M, :]
            chunk_residual = residual_padded[M_lo : M_lo + self.M, :]
            packed_hidden_residual_part = self._pack_hidden_residual(
                chunk_hidden,
                chunk_residual,
            )
            result_part = self._execute_aie_operation(
                packed_hidden_residual_part,
                B_Up_np,
                B_Down_np,
            )
            result_padded[M_lo : M_lo + self.M, :] = result_part
        return result_padded

    def _pad_A(self, A_np: np.ndarray) -> np.ndarray:
        M, K = A_np.shape
        if M % self.M == 0 and K == self.K:
            return A_np

        M_multiple = (M + self.M - 1) // self.M * self.M
        A_padded = np.zeros((M_multiple, self.K), dtype=A_np.dtype)
        A_padded[:M, :K] = A_np
        return A_padded

    def _canonicalize_packed_hidden_residual(
        self,
        packed_hidden_residual: torch.Tensor,
    ) -> tuple[np.ndarray, int]:
        if packed_hidden_residual.ndim == 1:
            packed_hidden_residual = packed_hidden_residual.contiguous()
            if packed_hidden_residual.numel() % (2 * self.K) != 0:
                raise AIEOperatorConstraintError(
                    "AIEAddNormFFNAddNorm: invalid flat packed_hidden_residual length"
                )
            padded_rows = packed_hidden_residual.numel() // (2 * self.K)
            if padded_rows < self.seq_len:
                raise AIEOperatorConstraintError(
                    "AIEAddNormFFNAddNorm: packed_hidden_residual does not cover seq_len"
                )
            return (
                torch_to_numpy(packed_hidden_residual),
                self.seq_len,
            )
        if packed_hidden_residual.ndim == 3 and packed_hidden_residual.shape[0] == 2:
            packed_hidden_residual = packed_hidden_residual.contiguous()
            if packed_hidden_residual.shape[2] != self.K:
                raise AIEOperatorConstraintError(
                    "AIEAddNormFFNAddNorm: incompatible packed_hidden_residual hidden size"
                )
            return (
                self._pack_hidden_residual(
                    torch_to_numpy(packed_hidden_residual[0]),
                    torch_to_numpy(packed_hidden_residual[1]),
                ),
                packed_hidden_residual.shape[1],
            )
        if (
            packed_hidden_residual.ndim == 2
            and packed_hidden_residual.shape[0] % 2 == 0
        ):
            rows = packed_hidden_residual.shape[0] // 2
            packed_hidden_residual = packed_hidden_residual.contiguous().view(
                2, rows, packed_hidden_residual.shape[1]
            )
            if packed_hidden_residual.shape[2] != self.K:
                raise AIEOperatorConstraintError(
                    "AIEAddNormFFNAddNorm: incompatible packed_hidden_residual hidden size"
                )
            return (
                self._pack_hidden_residual(
                    torch_to_numpy(packed_hidden_residual[0]),
                    torch_to_numpy(packed_hidden_residual[1]),
                ),
                rows,
            )
        raise AIEOperatorConstraintError(
            "AIEAddNormFFNAddNorm: expected packed_hidden_residual as flat tile-packed "
            "buffer or legacy shape (2, seq_len, hidden_size)"
        )

    def _pack_hidden_residual(
        self,
        attention_output_np: np.ndarray,
        residual_np: np.ndarray,
    ) -> np.ndarray:
        M, K = attention_output_np.shape
        if residual_np.shape != (M, K):
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: attention_output and residual must have the same shape"
            )
        if K != self.K:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: incompatible packed_hidden_residual hidden size"
            )
        if M % self.tile_m != 0:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: packed Block 3 handoff requires rows divisible by tile_m"
            )

        num_row_tiles = M // self.tile_m
        if num_row_tiles % self.parallel_seq != 0:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: packed Block 3 handoff requires "
                "(rows / tile_m) divisible by parallel_seq"
            )

        row_iters_per_a_tile = num_row_tiles // self.parallel_seq
        k_div_tile = K // self.tile_k
        tile_elems = self.tile_m * self.tile_k
        packed_hidden_residual_np = np.zeros(
            (2 * M * K,), dtype=attention_output_np.dtype
        )

        for row_tile_idx in range(num_row_tiles):
            a_tile = row_tile_idx % self.parallel_seq
            row_iter = row_tile_idx // self.parallel_seq
            row_start = row_tile_idx * self.tile_m
            for k_tile_idx in range(k_div_tile):
                col_start = k_tile_idx * self.tile_k
                tile_index = (
                    a_tile * row_iters_per_a_tile + row_iter
                ) * k_div_tile + k_tile_idx
                tile_offset = tile_index * (2 * tile_elems)
                packed_hidden_residual_np[tile_offset : tile_offset + tile_elems] = (
                    attention_output_np[
                        row_start : row_start + self.tile_m,
                        col_start : col_start + self.tile_k,
                    ].reshape(tile_elems)
                )
                packed_hidden_residual_np[
                    tile_offset + tile_elems : tile_offset + (2 * tile_elems)
                ] = residual_np[
                    row_start : row_start + self.tile_m,
                    col_start : col_start + self.tile_k,
                ].reshape(
                    tile_elems
                )

        return packed_hidden_residual_np

    def _unpack_hidden_residual(
        self,
        packed_hidden_residual_np: np.ndarray,
        rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rows % self.tile_m != 0:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: packed Block 3 handoff requires rows divisible by tile_m"
            )

        num_row_tiles = rows // self.tile_m
        if num_row_tiles % self.parallel_seq != 0:
            raise AIEOperatorConstraintError(
                "AIEAddNormFFNAddNorm: packed Block 3 handoff requires "
                "(rows / tile_m) divisible by parallel_seq"
            )

        row_iters_per_a_tile = num_row_tiles // self.parallel_seq
        k_div_tile = self.K // self.tile_k
        tile_elems = self.tile_m * self.tile_k
        attention_output_np = np.zeros(
            (rows, self.K), dtype=packed_hidden_residual_np.dtype
        )
        residual_np = np.zeros((rows, self.K), dtype=packed_hidden_residual_np.dtype)

        for row_tile_idx in range(num_row_tiles):
            a_tile = row_tile_idx % self.parallel_seq
            row_iter = row_tile_idx // self.parallel_seq
            row_start = row_tile_idx * self.tile_m
            for k_tile_idx in range(k_div_tile):
                col_start = k_tile_idx * self.tile_k
                tile_index = (
                    a_tile * row_iters_per_a_tile + row_iter
                ) * k_div_tile + k_tile_idx
                tile_offset = tile_index * (2 * tile_elems)
                attention_output_np[
                    row_start : row_start + self.tile_m,
                    col_start : col_start + self.tile_k,
                ] = packed_hidden_residual_np[
                    tile_offset : tile_offset + tile_elems
                ].reshape(
                    self.tile_m, self.tile_k
                )
                residual_np[
                    row_start : row_start + self.tile_m,
                    col_start : col_start + self.tile_k,
                ] = packed_hidden_residual_np[
                    tile_offset + tile_elems : tile_offset + (2 * tile_elems)
                ].reshape(
                    self.tile_m, self.tile_k
                )

        return attention_output_np, residual_np

    def _pad_packed_hidden_residual(
        self,
        packed_hidden_residual_np: np.ndarray,
        rows: int,
    ) -> tuple[np.ndarray, int]:
        if packed_hidden_residual_np.size != 2 * rows * self.K:
            existing_rows = packed_hidden_residual_np.size // (2 * self.K)
            if (
                packed_hidden_residual_np.size != 2 * existing_rows * self.K
                or existing_rows < rows
            ):
                raise AIEOperatorConstraintError(
                    "AIEAddNormFFNAddNorm: invalid packed_hidden_residual buffer size"
                )
            if existing_rows % self.M == 0:
                return packed_hidden_residual_np, existing_rows
        if rows % self.M == 0:
            return packed_hidden_residual_np, rows

        padded_rows = (rows + self.M - 1) // self.M * self.M
        attention_output_np, residual_np = self._unpack_hidden_residual(
            packed_hidden_residual_np,
            rows,
        )
        attention_output_padded = np.zeros(
            (padded_rows, self.K), dtype=packed_hidden_residual_np.dtype
        )
        residual_padded = np.zeros(
            (padded_rows, self.K), dtype=packed_hidden_residual_np.dtype
        )
        attention_output_padded[:rows, :] = attention_output_np
        residual_padded[:rows, :] = residual_np
        return (
            self._pack_hidden_residual(attention_output_padded, residual_padded),
            padded_rows,
        )

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
        packed_hidden_residual_np: np.ndarray,
        B_Up_np: np.ndarray | None = None,
        B_Down_np: np.ndarray | None = None,
    ) -> np.ndarray:
        M = packed_hidden_residual_np.size // (2 * self.K)
        K = self.K
        K2, N = B_Up_np.shape if B_Up_np is not None else self.weight_up_proj.T.shape
        N2, K3 = (
            B_Down_np.shape if B_Down_np is not None else self.weight_down_proj.T.shape
        )

        assert M == self.M
        assert K == K2 and K == K3 and K == self.K
        assert N == N2 and N == self.N

        self.write_buffer(
            "packed_hidden_residual",
            packed_hidden_residual_np,
        )
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
