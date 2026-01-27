# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
import logging
from pathlib import Path

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

from iron.common.utils import torch_to_numpy, numpy_to_torch


class AIEGEMM(AIEOperatorBase):
    """AIE-accelerated General Matrix Multiplication (GEMM) layer"""

    def __init__(
        self,
        M,
        K,
        N,
        use_static_weight=False,
        tile_m=64,
        tile_k=64,
        tile_n=64,
        num_aie_columns=8,
        batch_A=(1, 0),  # (batch size, batch stride dim)
        batch_B=(1, 0),  # (batch size, batch stride dim)
        batch_C=(1, 0),  # (batch size, batch stride dim)
        context=None,
        skip_add_to_list=False,
        **gemm_kwargs,
    ):

        self.tile_m = tile_m
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.num_aie_columns = num_aie_columns
        self.batch_A = batch_A
        self.batch_B = batch_B
        self.batch_C = batch_C
        self.gemm_args = gemm_kwargs

        # Set frequently accessed gemm_args
        self.b_col_maj = gemm_kwargs.get("b_col_maj", False)
        self.c_col_maj = gemm_kwargs.get("c_col_maj", False)
        self.weight = (
            None
            if not use_static_weight
            else torch.zeros((K, N), dtype=torch.bfloat16).T
        )
        self.static_weight_shape = (K, N)

        # The operator's M, K, N represent what the NPU operator supports.
        # Calls to forward() may supply matrices of different sizes, and the
        # Python code will perform necessary padding/repeated application of
        # the NPU operator.
        M_padded, K_padded, N_padded = self._get_padded_dims(M, K, N)
        self.M = M_padded
        self.K = K_padded
        self.N = N_padded
        self.b_col_maj = self.gemm_args.get("b_col_maj", False)
        self.c_col_maj = self.gemm_args.get("c_col_maj", False)

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="gemm_"):
        # Extract parameters from self
        operator_dir = Path(__file__).parent
        tile_m = self.tile_m
        tile_k = self.tile_k
        tile_n = self.tile_n
        M = self.M
        K = self.K
        N = self.N
        num_aie_columns = self.num_aie_columns
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        dtype_in = self.gemm_args.get("dtype_in", "bf16")
        dtype_out = self.gemm_args.get("dtype_out", "bf16")
        emulate_bf16_mmul_with_bfp16 = self.gemm_args.get(
            "emulate_bf16_mmul_with_bfp16", True
        )
        prio_accuracy = self.gemm_args.get("prio_accuracy", False)
        use_scalar = self.gemm_args.get("use_scalar", False)
        round_conv_even = self.gemm_args.get("round_conv_even", True)

        if emulate_bf16_mmul_with_bfp16:
            min_tile_m, min_tile_k, min_tile_n = 8, 8, 8
        else:
            min_tile_m, min_tile_k, min_tile_n = 4, 8, 8
        assert tile_m >= min_tile_m, f"tile_m ({tile_m}) must be >= {min_tile_m}"
        assert tile_k >= min_tile_k, f"tile_k ({tile_k}) must be >= {min_tile_k}"
        assert tile_n >= min_tile_n, f"tile_n ({tile_n}) must be >= {min_tile_n}"

        file_name_tile_base = f"{prefix}{tile_m}x{tile_k}x{tile_n}"
        file_name_total_base = f"{prefix}{M}x{K}x{N}_{num_aie_columns}_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
        file_name_total_base += f"_batchA{self.batch_A[0]}d{self.batch_A[1]}_batchB{self.batch_B[0]}d{self.batch_B[1]}_batchC{self.batch_C[0]}d{self.batch_C[1]}"
        xclbin_kernel_name = f"gemm_{file_name_tile_base}"
        kernel_flags = [
            f"-DDIM_M={tile_m}",
            f"-DDIM_K={tile_k}",
            f"-DDIM_N={tile_n}",
        ]
        if prio_accuracy:
            kernel_flags.append("-Dbf16_f32_ONLY")
        else:
            kernel_flags.append("-Dbf16_bf16_ONLY")
        if round_conv_even:
            kernel_flags.append("-DROUND_CONV_EVEN")
        if emulate_bf16_mmul_with_bfp16:
            kernel_flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
        if self.b_col_maj:
            kernel_flags.append("-DB_COL_MAJ")
        if self.c_col_maj:
            kernel_flags.append("-DC_COL_MAJ")

        kernel_archive = f"gemm_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_total_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_matmul",
            callback_kwargs={
                "dev": device_str,
                "M": M,
                "K": K,
                "N": N,
                "m": tile_m,
                "k": tile_k,
                "n": tile_n,
                "n_aie_cols": num_aie_columns,
                "dtype_in_str": dtype_in,
                "dtype_out_str": dtype_out,
                "b_col_maj": int(self.b_col_maj),
                "c_col_maj": int(self.c_col_maj),
                "use_scalar": use_scalar,
                "emulate_bf16_mmul_with_bfp16": emulate_bf16_mmul_with_bfp16,
                "prio_accuracy": prio_accuracy,
                "trace_size": 0,
                "archive": kernel_archive,
                "generate_taps": False,
                "batch_A": self.batch_A,
                "batch_B": self.batch_B,
                "batch_C": self.batch_C,
            },
            requires_context=False,
        )

        # FIXME: We should be able to reuse the same xclbin for same tile
        # sizes, only swapping out the instruction sequence for different
        # problem sizes. However, there seem to be cases where this does
        # not work and the GEMM appears to be misconfigured for the wrong
        # size (resulting in a timeout when trying to run it). Perhaps
        # XRT is caching something, or something is wrong with the run-
        # time parameter (synchronization)? For now, create separate
        # xclbins for each problem size.
        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_total_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            f"gemm_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}_acc{int(prio_accuracy)}_embf16{int(emulate_bf16_mmul_with_bfp16)}_round{int(round_conv_even)}.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "mm.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "convert_copy.o",
                            [
                                SourceArtifact.new(
                                    base_dir
                                    / "aie_kernels"
                                    / "generic"
                                    / "convert_copy.cc"
                                )
                            ],
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

        return (xclbin_artifact, insts_artifact)

    def set_up_artifacts(self):
        # Describe required artifacts (xclbin, insts.bin)
        device_str = self.context.device_manager.device_str()
        xclbin_artifact, insts_artifact = self.get_artifacts()

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact

        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        static_weights = None
        if self.weight is not None:
            static_weights = self.weight.T
            if isinstance(static_weights, torch.Tensor):
                static_weights = torch_to_numpy(static_weights)
        self.add_kernel(
            "gemm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("A", self.M * self.K * self.batch_A[0])
        self.add_buffer(
            "B", self.K * self.N * self.batch_B[0], static_data=static_weights
        )
        self.add_buffer("C", self.M * self.N * self.batch_C[0])
        self.add_to_runlist("gemm", "A", "B", "C")

    def forward(self, A, B=None):
        """Forward pass through GEMM operation: C = A @ B
        A, B, C are expected to be 2D matrices unless the output has a
        batch size > 1, i.e batch_C[0] > 1
        """
        if self.batch_C[0] == 1:
            A_inp = A
            B_inp = B
            if len(A.shape) > 2 and A.shape[0] == 1:
                A_inp = A.squeeze(0)
            if B is not None and len(B.shape) > 2 and B.shape[0] == 1:
                B_inp = B.squeeze(0)
            result = self._do_unbatched_gemm(A_inp, B_inp)
            if len(result.shape) < len(A.shape):
                result = result.unsqueeze(0)
            return result
        else:
            return self._do_batched_gemm(A, B)

    def _do_unbatched_gemm(self, A, B=None):
        """Forward pass through GEMM operation: C = A @ B
        A, B, C are expected to be 2D matrices
        """
        B_shape = B.shape if B is not None else self.weight.T.shape
        K2, N = B_shape
        M, K = A.shape
        expected_output_shape = (M, N)
        applicable = K == K2 and M <= self.M and K <= self.K and N <= self.N
        if not applicable:
            raise AIEOperatorConstraintError("AIEGEMM: incompatible tensor shape(s)")

        A_padded = self._pad_A(torch_to_numpy(A))
        if B is not None:
            B_padded = self._pad_B(torch_to_numpy(B))
        else:
            B_padded = None

        logging.debug(
            f"Executing GEMM for dimensions M={M}, K={K}, N={N} using NPU operator with M={self.M}, K={self.K}, N={self.N}"
        )

        if self.c_col_maj:
            result_padded = np.zeros((N, M), dtype=A_padded.dtype)
        else:
            result_padded = np.zeros((M, N), dtype=A_padded.dtype)
        for M_lo in range(0, M, self.M):
            A_part = A_padded[M_lo : M_lo + self.M, :]
            result_parts = self._execute_aie_operation(A_part, B_padded)
            max_M = min(M_lo + self.M, M)
            if self.c_col_maj:
                result_padded[:, M_lo:max_M] = result_parts[:N, :max_M]
            else:
                result_padded[M_lo:max_M, :] = result_parts[:max_M, :N]

        # GEMM produces 2D result, reshape to expected output shape
        if self.c_col_maj:
            result = numpy_to_torch(result_padded[:N, :M])
        else:
            result = numpy_to_torch(result_padded[:M, :N])
        result = result.view(expected_output_shape)

        return result

    def _get_gemm_shapes(self, mtx_shape, batch_params):
        """Determine the 2D GEMM shapes from the 3D matrix shape based on the batch params"""
        batch_size, batch_stride_dim = batch_params
        # Assume the view of the tensor corresponds with the batch stride dim
        if batch_params[0] == 1:
            return mtx_shape
        if batch_stride_dim == 0:  # corresponds to (batch size, mtx_shape)
            if batch_size == mtx_shape[0]:
                return mtx_shape[1:]
            else:
                raise AIEOperatorConstraintError(
                    "AIEGEMM: unexpected batched tensor shape"
                )
        else:  # corresponds to (mtx_shape, batch size)
            if batch_size == mtx_shape[1]:
                return mtx_shape[:-1]
            else:
                raise AIEOperatorConstraintError(
                    "AIEGEMM: unexpected batched tensor shape"
                )

    def _do_batched_gemm(self, A, B=None):
        """Forward pass through GEMM operation: C = A @ B
        A, B, C are expected to be 3D matrices
        """
        B_shape = B.shape if B is not None else self.weight.T.shape
        K2, N = self._get_gemm_shapes(B_shape, self.batch_B)
        M, K = self._get_gemm_shapes(A.shape, self.batch_B)
        batch_size_C, batch_stride_dim_C = self.batch_C
        if batch_stride_dim_C == 0:
            expected_output_shape = (batch_size_C, M, N)
        else:
            expected_output_shape = (M, N, batch_size_C)
        applicable = K == K2 and M <= self.M and K <= self.K and N <= self.N
        if not applicable:
            raise AIEOperatorConstraintError("AIEGEMM: incompatible tensor shape(s)")

        A_padded = self._pad_A_batched(torch_to_numpy(A))
        if B is not None:
            B_padded = self._pad_B_batched(torch_to_numpy(B))
        else:
            B_padded = None

        logging.debug(
            f"Executing batched GEMM for dimensions M={M}, K={K}, N={N} using NPU operator with M={self.M}, K={self.N}, N={self.N}"
        )

        if batch_stride_dim_C == 0:
            result_padded = np.zeros((batch_size_C, M, self.N), dtype=A_padded.dtype)
            for M_lo in range(0, M, self.M):
                if self.batch_A[1] == 0:
                    A_part = A_padded[:, M_lo : M_lo + self.M, :]
                else:
                    A_part = A_padded[M_lo : M_lo + self.M, :, :]
                result_part = self._execute_aie_operation(A_part, B_padded)
                max_M = min(M_lo + self.M, M)
                result_padded[:, M_lo:max_M, :] = result_part[:, :max_M, :]
            result = numpy_to_torch(result_padded[:, :M, :N])
        else:
            result_padded = np.zeros((M, self.N, batch_size_C), dtype=A_padded.dtype)
            for M_lo in range(0, M, self.M):
                if self.batch_A[1] == 0:
                    A_part = A_padded[:, M_lo : M_lo + self.M, :]
                else:
                    A_part = A_padded[M_lo : M_lo + self.M, :, :]
                A_part = A_padded[M_lo : M_lo + self.M, :]
                result_part = self._execute_aie_operation(A_part, B_padded)
                max_M = min(M_lo + self.M, M)
                result_padded[M_lo:max_M, :, :] = result_part[:max_M, :, :]
            result = numpy_to_torch(result_padded[:M, :N, :])

        # Reshape to expected output shape
        result = result.view(expected_output_shape)

        return result

    def _get_padded_dims(self, M, K, N):
        tile_m, tile_k, tile_n = self.tile_m, self.tile_k, self.tile_n
        num_aie_columns = self.num_aie_columns
        num_aie_rows = 4
        logging.info(f"Calculating padded dimensions for requested M={M}, K={K}, N={N}")
        logging.info(
            f"Using tile sizes tile_m={tile_m}, tile_k={tile_k}, tile_n={tile_n}, num_aie_columns={num_aie_columns}"
        )

        min_M = tile_m * num_aie_rows
        min_K = tile_k
        min_N = tile_n * num_aie_columns

        # Calculate padded dimensions
        M_padded = ((M + min_M - 1) // min_M) * min_M
        K_padded = ((K + min_K - 1) // min_K) * min_K
        N_padded = ((N + min_N - 1) // min_N) * min_N

        return M_padded, K_padded, N_padded

    def _pad_A(self, A_np):
        """Pad A matrix to match operator dimensions (M, K)"""
        M, K = A_np.shape
        if M % self.M == 0 and K == self.K:
            return A_np

        M_padded = ((M + self.M - 1) // self.M) * self.M
        A_padded = np.zeros((M_padded, self.K), dtype=A_np.dtype)
        A_padded[:M, :K] = A_np
        return A_padded

    def _pad_A_batched(self, A_np):
        batch_size, batch_stride_dim = self.batch_A
        A_padded = A_np
        if batch_stride_dim == 0:
            M, K = A_np.shape[1:]
            if M % self.M != 0 or K != self.K:
                M_multiple = (M + self.M - 1) // self.M * self.M
                A_padded = np.zeros((batch_size, M_multiple, self.K), dtype=A_np.dtype)
                A_padded[:, :M, :K] = A_np
        else:
            M, K = A_np.shape[:-1]
            if M % self.M != 0 or K != self.K:
                M_multiple = (M + self.M - 1) // self.M * self.M
                A_padded = np.zeros((M_multiple, self.K, batch_size), dtype=A_np.dtype)
                A_padded[:M, :K, :] = A_np
        return A_padded

    def _pad_B(self, B_np):
        """Pad B matrix to match operator dimensions based on layout"""
        if self.b_col_maj:
            N, K = B_np.shape
            if N == self.N and K == self.K:
                return B_np
            B_padded = np.zeros((self.N, self.K), dtype=B_np.dtype)
            B_padded[:N, :K] = B_np
        else:
            K, N = B_np.shape
            if K == self.K and N == self.N:
                return B_np
            B_padded = np.zeros((self.K, self.N), dtype=B_np.dtype)
            B_padded[:K, :N] = B_np
        return B_padded

    def _pad_B_batched(self, B_np):
        batch_size, batch_stride_dim = self.batch_B
        B_padded = B_np
        if batch_stride_dim == 0:
            K, N = B_np.shape[1:]
            if K != self.K or N != self.N:
                B_padded = np.zeros((batch_size, self.K, self.N), dtype=B_np.dtype)
                B_padded[:, :K, :N] = B_np
        else:
            K, N = B_np.shape[:-1]
            if K != self.K or N != self.N:
                B_padded = np.zeros((self.K, self.N, batch_size), dtype=B_np.dtype)
                B_padded[:, :K, :N] = B_np
        return B_padded

    def _execute_aie_operation(self, A_np, B_np=None):
        """Execute GEMM operation on AIE hardware"""
        M, K = self._get_gemm_shapes(A_np.shape, self.batch_A)
        K2, N = (
            self._get_gemm_shapes(B_np.shape, self.batch_B)
            if B_np is not None
            else self._get_gemm_shapes(self.weight.T.shape, self.batch_B)
        )

        # Validate dimensions match operator configuration
        assert M == self.M
        assert K == K2 and K == self.K
        assert N == self.N

        self.write_buffer("A", A_np)
        if B_np is not None:
            self.write_buffer("B", B_np)
        self.run_runlist()
        if self.batch_C[0] == 1:
            result_np = self.read_buffer("C", shape=(M, N), dtype=bfloat16)
        else:
            if self.batch_C[1] == 0:
                result_np = self.read_buffer(
                    "C", shape=(self.batch_C[0], M, N), dtype=bfloat16
                )
            else:
                result_np = self.read_buffer(
                    "C", shape=(M, N, self.batch_C[0]), dtype=bfloat16
                )

        # Check for NaN and fail hard
        # if np.isnan(result_np).any():
        #     nan_count = np.isnan(result_np).sum()
        #     total_count = result_np.size
        #     raise RuntimeError(
        #         f"AIE execution returned {nan_count}/{total_count} NaN values. "
        #     )

        # Convert back to torch tensor
        return result_np
