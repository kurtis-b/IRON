# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone dynamic GEMM operator with runtime-sequence tail handling.

Supported execution surface:
- non-batched GEMM only
- ``num_aie_columns`` in ``{4, 8}``
- ``M % tile_m == 0``
- ``K % tile_k == 0``
- ``N % tile_n == 0``
- exact, column-tail, row-tail, and corner-tail execution when there is at
  least one full row block available for the fixed C-join sink scheme
- ``b_col_maj`` supported
- ``c_col_maj=False`` only

Unsupported shape classes and API surface:
- batched GEMM
- ``c_col_maj=True``
- bootstrap row/corner tails where ``M < 4 * tile_m`` and ``M`` is not a full
  row block; the current fixed C join requires one preceding full row block as
  an in-bounds sink for invalid tail rows
- buffer-shape and offset overrides
"""

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


class AIEDynamicGEMM(AIEOperatorBase):
    """AIE GEMM with shared-topology artifacts and runtime-sequence M/N tails.

    The operator is standalone: it does not wrap ``AIEGEMM`` and it executes
    against the exact logical ``A``/``B``/``C`` buffers supplied by the caller.

    The one deliberately unsupported tail case is a bootstrap M-tail with no
    preceding full row block. The current C join topology emits a fixed
    4-row-by-1-column joined block per compute column, so row/corner tails need
    one earlier full row block as an in-bounds sink for invalid rows.
    """

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
        context=None,
        skip_add_to_list=False,
        **gemm_kwargs,
    ):
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.num_aie_columns = num_aie_columns
        self.gemm_args = gemm_kwargs
        self.b_col_maj = gemm_kwargs.get("b_col_maj", False)
        self.c_col_maj = gemm_kwargs.get("c_col_maj", False)
        if self.c_col_maj:
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM does not yet support c_col_maj=True"
            )

        batch_A = gemm_kwargs.pop("batch_A", (1, 0))
        batch_B = gemm_kwargs.pop("batch_B", (1, 0))
        batch_C = gemm_kwargs.pop("batch_C", (1, 0))
        if batch_A != (1, 0) or batch_B != (1, 0) or batch_C != (1, 0):
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM only supports non-batched GEMM in v1"
            )

        for key in (
            "input_a_buffer_shape",
            "input_b_buffer_shape",
            "output_c_buffer_shape",
            "input_a_offset",
            "input_b_offset",
            "output_c_offset",
        ):
            if key in gemm_kwargs:
                raise AIEOperatorConstraintError(
                    f"{key} is not supported by AIEDynamicGEMM"
                )

        if num_aie_columns not in (4, 8):
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM currently supports num_aie_columns of 4 or 8"
            )
        if M % tile_m != 0:
            raise AIEOperatorConstraintError(
                f"M ({M}) must be divisible by tile_m ({tile_m})"
            )
        if K % tile_k != 0:
            raise AIEOperatorConstraintError(
                f"K ({K}) must be divisible by tile_k ({tile_k})"
            )
        if N % tile_n != 0:
            raise AIEOperatorConstraintError(
                f"N ({N}) must be divisible by tile_n ({tile_n})"
            )
        self.full_M = tile_m * 4
        self.full_N = tile_n * num_aie_columns
        if M % self.full_M != 0 and M < self.full_M:
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM does not support bootstrap row/corner tails with "
                f"M < 4*tile_m (M={M}, tile_m={tile_m}). The fixed C join "
                "topology requires one preceding full row block as an in-bounds "
                "sink for invalid tail rows."
            )

        self.M = M
        self.K = K
        self.N = N

        if use_static_weight:
            weight_shape = (N, K) if self.b_col_maj else (K, N)
            self.weight = torch.zeros(weight_shape, dtype=torch.bfloat16)
            self.static_weight_shape = weight_shape
        else:
            self.weight = None
            self.static_weight_shape = (N, K) if self.b_col_maj else (K, N)

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def _get_runtime_dims(self):
        num_aie_rows = 4
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns
        return min_M, min_K, min_N

    def _get_artifact_name_base(
        self, prefix, *, include_workload_dims, M=None, K=None, N=None
    ):
        dtype_in = self.gemm_args.get("dtype_in", "bf16")
        dtype_out = self.gemm_args.get("dtype_out", "bf16")
        emulate_bf16_mmul_with_bfp16 = self.gemm_args.get(
            "emulate_bf16_mmul_with_bfp16", True
        )
        prio_accuracy = self.gemm_args.get("prio_accuracy", False)
        use_scalar = self.gemm_args.get("use_scalar", False)
        round_conv_even = self.gemm_args.get("round_conv_even", True)

        if include_workload_dims:
            assert M is not None and K is not None and N is not None
            workload_prefix = f"{M}x{K}x{N}_"
        else:
            workload_prefix = "runtime_"

        return (
            f"{prefix}{workload_prefix}"
            f"{self.num_aie_columns}_{self.tile_m}x{self.tile_k}x{self.tile_n}"
            f"_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
            f"_{dtype_in}_{dtype_out}"
            f"_sc{int(use_scalar)}"
            f"_acc{int(prio_accuracy)}"
            f"_embf16{int(emulate_bf16_mmul_with_bfp16)}"
            f"_round{int(round_conv_even)}"
            "_ctiles1"
        )

    def _build_mlir_artifact(self, prefix, M, K, N, *, include_workload_dims):
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

        min_tile_m, min_tile_k, min_tile_n = (
            (8, 8, 8) if emulate_bf16_mmul_with_bfp16 else (4, 8, 8)
        )
        assert self.tile_m >= min_tile_m
        assert self.tile_k >= min_tile_k
        assert self.tile_n >= min_tile_n

        file_name_total_base = self._get_artifact_name_base(
            prefix,
            include_workload_dims=include_workload_dims,
            M=M,
            K=K,
            N=N,
        )

        kernel_archive = (
            f"gemm_{self.tile_m}x{self.tile_k}x{self.tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
            f"_{dtype_in}_{dtype_out}"
            f"_sc{int(use_scalar)}"
            f"_acc{int(prio_accuracy)}"
            f"_embf16{int(emulate_bf16_mmul_with_bfp16)}"
            f"_round{int(round_conv_even)}.a"
        )
        kernel_flags = [
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
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

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_total_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_matmul",
            callback_kwargs={
                "dev": device_str,
                "M": M,
                "K": K,
                "N": N,
                "m": self.tile_m,
                "k": self.tile_k,
                "n": self.tile_n,
                "n_aie_cols": self.num_aie_columns,
                "dtype_in_str": dtype_in,
                "dtype_out_str": dtype_out,
                "b_col_maj": int(self.b_col_maj),
                "c_col_maj": int(self.c_col_maj),
                "use_scalar": use_scalar,
                "emulate_bf16_mmul_with_bfp16": emulate_bf16_mmul_with_bfp16,
                "prio_accuracy": prio_accuracy,
                "separate_c_tiles": 1,
                "trace_size": 0,
                "archive": kernel_archive,
                "generate_taps": False,
            },
            requires_context=False,
        )

        kernel_archive_artifact = KernelArchiveArtifact.new(
            kernel_archive,
            depends=[
                KernelObjectArtifact.new(
                    (
                        f"gemm_{self.tile_m}x{self.tile_k}x{self.tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
                        f"_acc{int(prio_accuracy)}_embf16{int(emulate_bf16_mmul_with_bfp16)}"
                        f"_round{int(round_conv_even)}.o"
                    ),
                    extra_flags=kernel_flags,
                    depends=[
                        SourceArtifact.new(base_dir / "aie_kernels" / "aie2p" / "mm.cc")
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
                            base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
                        )
                    ],
                ),
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_total_base}.xclbin",
            depends=[mlir_artifact, kernel_archive_artifact],
            extra_flags=["--dynamic-objFifos"],
        )
        insts_artifact = InstsBinArtifact.new(
            f"{file_name_total_base}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )
        return xclbin_artifact, insts_artifact

    def get_artifacts(self, prefix="dynamic_gemm_"):
        xclbin_artifact, insts_artifact = self._build_mlir_artifact(
            prefix,
            self.M,
            self.K,
            self.N,
            include_workload_dims=True,
        )
        insts_artifact.xclbin_input = None
        insts_artifact.kernel_name = None
        return xclbin_artifact, insts_artifact

    def get_insts_artifact(
        self,
        prefix="dynamic_gemm_",
        xclbin_input=None,
        kernel_name=None,
    ):
        _, insts_artifact = self._build_mlir_artifact(
            prefix,
            self.M,
            self.K,
            self.N,
            include_workload_dims=True,
        )
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

    def get_runtime_xclbin_artifact(self, prefix="dynamic_gemm_"):
        runtime_M, runtime_K, runtime_N = self._get_runtime_dims()
        xclbin_artifact, _ = self._build_mlir_artifact(
            prefix,
            runtime_M,
            runtime_K,
            runtime_N,
            include_workload_dims=False,
        )
        return xclbin_artifact

    def set_up_artifacts(self):
        if self.xclbin_artifact is None or self.insts_artifact is None:
            xclbin_artifact, insts_artifact = self.get_artifacts()
            self.xclbin_artifact = xclbin_artifact
            self.insts_artifact = insts_artifact
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        self.add_kernel(
            "dynamic_gemm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )

        self.add_buffer("A", self.M * self.K)
        if self.weight is None:
            self.add_buffer("B", self.K * self.N)
        else:
            static_weights = self.weight
            if isinstance(static_weights, torch.Tensor):
                static_weights = torch_to_numpy(static_weights)
            self.add_buffer(
                "B",
                self.K * self.N,
                static_data=static_weights.reshape(-1),
            )
        self.add_buffer("C", self.M * self.N)
        self.add_to_runlist("dynamic_gemm", "A", "B", "C")

    def _get_B_dims(self, B_shape):
        if self.b_col_maj:
            return B_shape[-1], B_shape[-2]
        return B_shape[-2], B_shape[-1]

    def forward(self, A, B=None):
        if len(A.shape) != 2:
            raise AIEOperatorConstraintError("AIEDynamicGEMM expects a 2D A tensor")
        if B is not None and len(B.shape) != 2:
            raise AIEOperatorConstraintError("AIEDynamicGEMM expects a 2D B tensor")
        if self.weight is None and B is None:
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM requires B or static weight"
            )

        B_shape = B.shape if B is not None else self.static_weight_shape
        K2, N = self._get_B_dims(B_shape)
        M, K = A.shape
        if M != self.M or K != self.K or K2 != self.K or N != self.N:
            raise AIEOperatorConstraintError(
                "AIEDynamicGEMM: incompatible tensor shape(s)"
            )

        logging.debug(
            "Executing Dynamic GEMM for logical dimensions M=%s, K=%s, N=%s",
            M,
            K,
            N,
        )

        self.write_buffer("A", torch_to_numpy(A))
        if B is not None:
            self.write_buffer("B", torch_to_numpy(B))
        self.run_runlist()

        C_shape = (self.N, self.M) if self.c_col_maj else (self.M, self.N)
        C = self.read_buffer("C", shape=C_shape, dtype=bfloat16)
        return numpy_to_torch(C)
