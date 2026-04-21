# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


class AIEGEMM(AIEOperatorBase):
    """AIE-accelerated General Matrix Multiplication (GEMM) layer."""

    def __init__(
        self,
        M,
        K,
        N,
        use_static_weight=False,
        tile_m=64,
        tile_k=64,
        tile_n=64,
        partition_N=1,
        num_aie_columns=8,
        batch_A=(1, 0),
        batch_B=(1, 0),
        batch_C=(1, 0),
        context=None,
        skip_add_to_list=False,
        **gemm_kwargs,
    ):
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.num_aie_columns = num_aie_columns
        self.partition_N = partition_N
        self.batch_A = batch_A
        self.batch_B = batch_B
        self.batch_C = batch_C
        self.input_a_buffer_shape = gemm_kwargs.pop("input_a_buffer_shape", None)
        self.input_b_buffer_shape = gemm_kwargs.pop("input_b_buffer_shape", None)
        self.output_c_buffer_shape = gemm_kwargs.pop("output_c_buffer_shape", None)
        self.input_a_offset = gemm_kwargs.pop("input_a_offset", 0)
        self.input_b_offset = gemm_kwargs.pop("input_b_offset", 0)
        self.output_c_offset = gemm_kwargs.pop("output_c_offset", 0)
        self.gemm_args = gemm_kwargs

        self.b_col_maj = gemm_kwargs.get("b_col_maj", False)
        self.c_col_maj = gemm_kwargs.get("c_col_maj", False)
        self.weight = (
            None
            if not use_static_weight
            else torch.zeros((K, N), dtype=torch.bfloat16).T
        )
        self.static_weight_shape = (K, N)

        if len(batch_A) != 2 or len(batch_B) != 2 or len(batch_C) != 2:
            raise AIEOperatorConstraintError(
                "batch_A, batch_B, and batch_C must be 2-tuples of (batch_size, batch_stride_dim)"
            )
        if self._uses_batched_layout() and partition_N != 1:
            raise AIEOperatorConstraintError(
                "partition_N > 1 is not supported together with batched GEMM"
            )
        if batch_C[0] == 1 and (batch_A[0] > 1 or batch_B[0] > 1):
            raise AIEOperatorConstraintError(
                "batch_A/batch_B > 1 requires batch_C > 1 in AIEGEMM"
            )
        if self._uses_batched_layout() and use_static_weight and batch_B[0] > 1:
            raise AIEOperatorConstraintError(
                "Static weights are only supported for batch_B=(1, 0) in batched GEMM"
            )
        if not self._uses_batched_layout() and (
            self.input_a_buffer_shape is not None
            or self.input_b_buffer_shape is not None
            or self.output_c_buffer_shape is not None
            or self.input_a_offset != 0
            or self.input_b_offset != 0
            or self.output_c_offset != 0
        ):
            raise AIEOperatorConstraintError(
                "Buffer window overrides are only supported for batched GEMM"
            )

        if self._uses_batched_layout():
            M_padded, K_padded, N_padded = self._get_padded_dims(M, K, N)
        else:
            if N % partition_N != 0:
                raise AIEOperatorConstraintError(
                    f"N ({N}) must be divisible by partition_N ({partition_N})"
                )
            M_padded, K_padded, N_padded = self._get_padded_dims(M, K, N // partition_N)
        self.M = M_padded
        self.K = K_padded
        self.N = N_padded

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def _uses_batched_layout(self):
        return self.batch_C[0] > 1

    @staticmethod
    def _buffer_shape_element_count(shape):
        return int(np.prod(shape))

    def _buffer_window_name_suffix(self):
        if not self._uses_batched_layout():
            return ""

        parts = []
        for label, shape in (
            ("bufA", self.input_a_buffer_shape),
            ("bufB", self.input_b_buffer_shape),
            ("bufC", self.output_c_buffer_shape),
        ):
            if shape is not None:
                parts.append(f"{label}{shape[0]}x{shape[1]}")
        for label, offset in (
            ("offA", self.input_a_offset),
            ("offB", self.input_b_offset),
            ("offC", self.output_c_offset),
        ):
            if offset:
                parts.append(f"{label}{offset}")
        if not parts:
            return ""
        return "_" + "_".join(parts)

    def _get_runtime_dims(self):
        num_aie_rows = 4
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns
        return min_M, min_K, min_N

    def _partitioned_b_buffer_name(self, partition_index: int) -> str:
        if self.partition_N == 1:
            return "B"
        return f"B_{partition_index}"

    def _partitioned_c_buffer_name(self, partition_index: int) -> str:
        if self.partition_N == 1:
            return "C"
        return f"C_{partition_index}"

    def _get_artifact_name_base(self, prefix, M, K, N, include_partition_suffix=True):
        dtype_in = self.gemm_args.get("dtype_in", "bf16")
        dtype_out = self.gemm_args.get("dtype_out", "bf16")
        emulate_bf16_mmul_with_bfp16 = self.gemm_args.get(
            "emulate_bf16_mmul_with_bfp16", True
        )
        prio_accuracy = self.gemm_args.get("prio_accuracy", False)
        use_scalar = self.gemm_args.get("use_scalar", False)
        round_conv_even = self.gemm_args.get("round_conv_even", True)
        file_name_total_base = (
            f"{prefix}{M}x{K}x{N}_{self.num_aie_columns}_{self.tile_m}x{self.tile_k}x{self.tile_n}"
            f"_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
            f"_{dtype_in}_{dtype_out}"
            f"_sc{int(use_scalar)}"
            f"_acc{int(prio_accuracy)}"
            f"_embf16{int(emulate_bf16_mmul_with_bfp16)}"
            f"_round{int(round_conv_even)}"
        )
        if self._uses_batched_layout():
            file_name_total_base += (
                f"_batchA{self.batch_A[0]}d{self.batch_A[1]}"
                f"_batchB{self.batch_B[0]}d{self.batch_B[1]}"
                f"_batchC{self.batch_C[0]}d{self.batch_C[1]}"
            )
            file_name_total_base += self._buffer_window_name_suffix()
        else:
            file_name_total_base += "_ctiles1"
        if (
            not self._uses_batched_layout()
            and include_partition_suffix
            and self.partition_N > 1
        ):
            file_name_total_base += f"_partN{self.partition_N}"
        return file_name_total_base

    def _build_mlir_artifact(self, prefix, M, K, N, include_partition_suffix=True):
        operator_dir = Path(__file__).parent
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

        tile_m = self.tile_m
        tile_k = self.tile_k
        tile_n = self.tile_n
        num_aie_columns = self.num_aie_columns

        if emulate_bf16_mmul_with_bfp16:
            min_tile_m, min_tile_k, min_tile_n = 8, 8, 8
        else:
            min_tile_m, min_tile_k, min_tile_n = 4, 8, 8
        assert tile_m >= min_tile_m, f"tile_m ({tile_m}) must be >= {min_tile_m}"
        assert tile_k >= min_tile_k, f"tile_k ({tile_k}) must be >= {min_tile_k}"
        assert tile_n >= min_tile_n, f"tile_n ({tile_n}) must be >= {min_tile_n}"

        file_name_total_base = self._get_artifact_name_base(
            prefix,
            M,
            K,
            N,
            include_partition_suffix=include_partition_suffix,
        )

        kernel_archive = (
            f"gemm_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
            f"_{dtype_in}_{dtype_out}"
            f"_sc{int(use_scalar)}"
            f"_acc{int(prio_accuracy)}"
            f"_embf16{int(emulate_bf16_mmul_with_bfp16)}"
            f"_round{int(round_conv_even)}.a"
        )
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

        if self._uses_batched_layout():
            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_total_base}.mlir",
                import_path=operator_dir / "design_batched.py",
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
                    "input_a_buffer_shape": self.input_a_buffer_shape,
                    "input_b_buffer_shape": self.input_b_buffer_shape,
                    "output_c_buffer_shape": self.output_c_buffer_shape,
                    "input_a_offset": self.input_a_offset,
                    "input_b_offset": self.input_b_offset,
                    "output_c_offset": self.output_c_offset,
                },
                requires_context=False,
            )
        else:
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
                    # Keep one general non-batched topology so partition_N only
                    # affects workload insts/runtime sequencing, not xclbin identity.
                    "separate_c_tiles": 1,
                    "trace_size": 0,
                    "archive": kernel_archive,
                    "generate_taps": False,
                },
                requires_context=False,
            )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_total_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            (
                                f"gemm_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
                                f"_acc{int(prio_accuracy)}_embf16{int(emulate_bf16_mmul_with_bfp16)}"
                                f"_round{int(round_conv_even)}.o"
                            ),
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "mm.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "zero_scalar.o",
                            [
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "zero.cc"
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
        return xclbin_artifact, insts_artifact

    def get_artifacts(self, prefix="gemm_"):
        xclbin_artifact, insts_artifact = self._build_mlir_artifact(
            prefix, self.M, self.K, self.N
        )
        insts_artifact.xclbin_input = None
        insts_artifact.kernel_name = None
        return xclbin_artifact, insts_artifact

    def get_insts_artifact(self, prefix="gemm_", xclbin_input=None, kernel_name=None):
        _, insts_artifact = self._build_mlir_artifact(prefix, self.M, self.K, self.N)
        insts_artifact.xclbin_input = xclbin_input
        insts_artifact.kernel_name = (
            kernel_name
            if kernel_name is not None
            else (
                xclbin_input.kernel_name
                if xclbin_input is not None and hasattr(xclbin_input, "kernel_name")
                else None
            )
        )
        return insts_artifact

    def get_runtime_xclbin_artifact(self, prefix="gemm_runtime_"):
        runtime_M, runtime_K, runtime_N = self._get_runtime_dims()
        xclbin_artifact, _ = self._build_mlir_artifact(
            prefix,
            runtime_M,
            runtime_K,
            runtime_N,
            include_partition_suffix=False,
        )
        return xclbin_artifact

    def set_up_artifacts(self):
        if self.xclbin_artifact is None or self.insts_artifact is None:
            xclbin_artifact, insts_artifact = self.get_artifacts()
            self.xclbin_artifact = xclbin_artifact
            self.insts_artifact = insts_artifact
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

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

        if self._uses_batched_layout():
            a_count = (
                self._buffer_shape_element_count(self.input_a_buffer_shape)
                if self.input_a_buffer_shape is not None
                else self.M * self.K * self.batch_A[0]
            )
            self.add_buffer("A", a_count)
            if static_weights is None:
                b_count = (
                    self._buffer_shape_element_count(self.input_b_buffer_shape)
                    if self.input_b_buffer_shape is not None
                    else self.K * self.N * self.batch_B[0]
                )
                self.add_buffer("B", b_count)
            else:
                self.add_buffer("B", self.K * self.N, static_data=static_weights)
            c_count = (
                self._buffer_shape_element_count(self.output_c_buffer_shape)
                if self.output_c_buffer_shape is not None
                else self.M * self.N * self.batch_C[0]
            )
            self.add_buffer("C", c_count)
            self.add_to_runlist("gemm", "A", "B", "C")
            return

        self.add_buffer("A", self.M * self.K)
        B_parts = self._partition_B(static_weights)
        for i, B_part in enumerate(B_parts):
            b_name = self._partitioned_b_buffer_name(i)
            c_name = self._partitioned_c_buffer_name(i)
            if B_part is None:
                self.add_buffer(b_name, self.K * self.N)
            else:
                self.add_buffer(b_name, self.K * self.N, static_data=B_part)
            self.add_buffer(c_name, self.M * self.N)
            self.add_to_runlist("gemm", "A", b_name, c_name)

    def _get_B_dims(self, B_shape):
        if self.b_col_maj:
            return B_shape[-1], B_shape[-2]
        return B_shape[-2], B_shape[-1]

    def forward(self, A, B=None):
        """Forward pass through GEMM operation: C = A @ B."""
        if self._uses_batched_layout():
            return self._do_batched_gemm(A, B)

        B_shape = B.shape if B is not None else self.static_weight_shape
        K2, N = self._get_B_dims(B_shape)
        N_part = N // self.partition_N
        expected_output_shape = (
            A.shape[:-2] + (N, A.shape[-1]) if self.c_col_maj else A.shape[:-1] + (N,)
        )

        if len(A.shape) > 2:
            A = A.view(-1, A.shape[-1])
        if B is not None and len(B.shape) > 2:
            B = B.view(-1, B_shape[-1])

        M, K = A.shape
        applicable = (
            K == K2
            and (M <= self.M or not self.c_col_maj)
            and K <= self.K
            and N <= self.N * self.partition_N
        )
        if not applicable:
            raise AIEOperatorConstraintError("AIEGEMM: incompatible tensor shape(s)")

        A_padded = self._pad_A(torch_to_numpy(A))
        B_parts = self._partition_B(torch_to_numpy(B)) if B is not None else None

        logging.debug(
            "Executing GEMM for dimensions M=%s, K=%s, N=%s using NPU operator with M=%s, K=%s, N=%s",
            M,
            K,
            N,
            self.M,
            self.K,
            self.N,
        )

        if self.c_col_maj:
            result_padded = np.zeros((N, M), dtype=A_padded.dtype)
        else:
            result_padded = np.zeros((M, N), dtype=A_padded.dtype)
        for M_lo in range(0, M, self.M):
            A_part = A_padded[M_lo : M_lo + self.M, :]
            result_parts = self._execute_partitioned_aie_operation(A_part, B_parts)
            max_M = min(M_lo + self.M, M)
            for part in range(self.partition_N):
                if self.c_col_maj:
                    result_padded[part * N_part : (part + 1) * N_part, M_lo:max_M] = (
                        result_parts[part][:N_part, :max_M]
                    )
                else:
                    result_padded[M_lo:max_M, part * N_part : (part + 1) * N_part] = (
                        result_parts[part][:max_M, :N_part]
                    )

        if self.c_col_maj:
            result = numpy_to_torch(result_padded[:N, :M])
        else:
            result = numpy_to_torch(result_padded[:M, :N])
        return result.view(expected_output_shape)

    def _get_gemm_shapes(self, mtx_shape, batch_params):
        batch_size, batch_stride_dim = batch_params
        if batch_size == 1:
            return mtx_shape
        if batch_stride_dim == 0:
            if batch_size == mtx_shape[0]:
                return mtx_shape[1:]
            raise AIEOperatorConstraintError("AIEGEMM: unexpected batched tensor shape")
        if batch_size == mtx_shape[1]:
            return mtx_shape[0], mtx_shape[2]
        raise AIEOperatorConstraintError("AIEGEMM: unexpected batched tensor shape")

    def _do_batched_gemm(self, A, B=None):
        B_shape = B.shape if B is not None else self.weight.T.shape
        K2, N = self._get_gemm_shapes(B_shape, self.batch_B)
        M, K = self._get_gemm_shapes(A.shape, self.batch_A)
        batch_size_C, batch_stride_dim_C = self.batch_C
        expected_output_shape = (
            (batch_size_C, M, N) if batch_stride_dim_C == 0 else (M, batch_size_C, N)
        )
        applicable = K == K2 and M <= self.M and K <= self.K and N <= self.N
        if not applicable:
            raise AIEOperatorConstraintError("AIEGEMM: incompatible tensor shape(s)")

        A_padded = self._pad_A_batched(torch_to_numpy(A))
        B_padded = self._pad_B_batched(torch_to_numpy(B)) if B is not None else None

        logging.debug(
            "Executing batched GEMM for dimensions M=%s, K=%s, N=%s using NPU operator with M=%s, K=%s, N=%s",
            M,
            K,
            N,
            self.M,
            self.K,
            self.N,
        )

        if batch_stride_dim_C == 0:
            result_padded = np.zeros((batch_size_C, M, self.N), dtype=A_padded.dtype)
            for M_lo in range(0, M, self.M):
                if self.batch_A[1] == 0:
                    A_part = A_padded[:, M_lo : M_lo + self.M, :]
                else:
                    A_part = A_padded[M_lo : M_lo + self.M, :, :]
                result_part = self._execute_batched_aie_operation(A_part, B_padded)
                max_M = min(M_lo + self.M, M)
                result_padded[:, M_lo:max_M, :N] = result_part[:, :max_M, :N]
            result = numpy_to_torch(result_padded[:, :M, :N])
        else:
            result_padded = np.zeros((M, batch_size_C, self.N), dtype=A_padded.dtype)
            for M_lo in range(0, M, self.M):
                if self.batch_A[1] == 0:
                    A_part = A_padded[:, M_lo : M_lo + self.M, :]
                else:
                    A_part = A_padded[M_lo : M_lo + self.M, :, :]
                result_part = self._execute_batched_aie_operation(A_part, B_padded)
                max_M = min(M_lo + self.M, M)
                result_padded[M_lo:max_M, :, :N] = result_part[:max_M, :, :N]
            result = numpy_to_torch(result_padded[:M, :, :N])

        return result.view(expected_output_shape)

    def _get_padded_dims(self, M, K, N):
        num_aie_rows = 4
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns
        M_padded = ((M + min_M - 1) // min_M) * min_M
        K_padded = ((K + min_K - 1) // min_K) * min_K
        N_padded = ((N + min_N - 1) // min_N) * min_N
        return M_padded, K_padded, N_padded

    def _pad_A(self, A_np):
        M, K = A_np.shape
        if M % self.M == 0 and K == self.K:
            return A_np
        M_padded = ((M + self.M - 1) // self.M) * self.M
        A_padded = np.zeros((M_padded, self.K), dtype=A_np.dtype)
        A_padded[:M, :K] = A_np
        return A_padded

    def _pad_A_batched(self, A_np):
        batch_size, batch_stride_dim = self.batch_A
        if batch_stride_dim == 0:
            M, K = A_np.shape[1:]
            if M % self.M == 0 and K == self.K:
                return A_np
            M_multiple = ((M + self.M - 1) // self.M) * self.M
            A_padded = np.zeros((batch_size, M_multiple, self.K), dtype=A_np.dtype)
            A_padded[:, :M, :K] = A_np
            return A_padded

        M, K = self._get_gemm_shapes(A_np.shape, self.batch_A)
        if M % self.M == 0 and K == self.K:
            return A_np
        M_multiple = ((M + self.M - 1) // self.M) * self.M
        A_padded = np.zeros((M_multiple, batch_size, self.K), dtype=A_np.dtype)
        A_padded[:M, :, :K] = A_np
        return A_padded

    def _pad_B(self, B_np):
        if self.b_col_maj:
            N, K = B_np.shape
            if N == self.N and K == self.K:
                return B_np
            B_padded = np.zeros((self.N, self.K), dtype=B_np.dtype)
            B_padded[:N, :K] = B_np
            return B_padded

        K, N = B_np.shape
        if K == self.K and N == self.N:
            return B_np
        B_padded = np.zeros((self.K, self.N), dtype=B_np.dtype)
        B_padded[:K, :N] = B_np
        return B_padded

    def _pad_B_batched(self, B_np):
        batch_size, batch_stride_dim = self.batch_B
        if batch_stride_dim == 0:
            K, N = B_np.shape[1:]
            if K == self.K and N == self.N:
                return B_np
            B_padded = np.zeros((batch_size, self.K, self.N), dtype=B_np.dtype)
            B_padded[:, :K, :N] = B_np
            return B_padded

        K, N = self._get_gemm_shapes(B_np.shape, self.batch_B)
        if K == self.K and N == self.N:
            return B_np
        B_padded = np.zeros((self.K, batch_size, self.N), dtype=B_np.dtype)
        B_padded[:K, :, :N] = B_np
        return B_padded

    def _partition_B(self, B_np):
        parts = [None] * self.partition_N
        if B_np is None:
            return parts
        for i in range(self.partition_N):
            col_start = i * self.N
            col_end = (i + 1) * self.N
            if self.b_col_maj:
                parts[i] = self._pad_B(B_np[col_start:col_end, :])
            else:
                parts[i] = self._pad_B(B_np[:, col_start:col_end])
        self.static_weight_shape = parts[0].shape
        return parts

    def _execute_partitioned_aie_operation(self, A_np, B_nps=None):
        M, K = A_np.shape
        B_shape = B_nps[0].shape if B_nps is not None else self.static_weight_shape
        K2, N = self._get_B_dims(B_shape)
        C_shape = (N, M) if self.c_col_maj else (M, N)

        assert M == self.M
        assert K == K2 and K == self.K
        assert N == self.N

        self.write_buffer("A", A_np)
        if B_nps is not None:
            for i, B_np in enumerate(B_nps):
                self.write_buffer(self._partitioned_b_buffer_name(i), B_np)
        self.run_runlist()
        return [
            self.read_buffer(
                self._partitioned_c_buffer_name(i), shape=C_shape, dtype=bfloat16
            )
            for i in range(self.partition_N)
        ]

    def _execute_batched_aie_operation(self, A_np, B_np=None):
        M, K = self._get_gemm_shapes(A_np.shape, self.batch_A)
        if B_np is not None:
            K2, N = self._get_gemm_shapes(B_np.shape, self.batch_B)
        else:
            K2, N = self._get_gemm_shapes(self.weight.T.shape, self.batch_B)

        assert M == self.M
        assert K == K2 and K == self.K
        assert N == self.N

        self.write_buffer("A", A_np)
        if B_np is not None:
            self.write_buffer("B", B_np)
        self.run_runlist()

        if self.batch_C[1] == 0:
            return self.read_buffer("C", shape=(self.batch_C[0], M, N), dtype=bfloat16)
        return self.read_buffer("C", shape=(M, self.batch_C[0], N), dtype=bfloat16)
