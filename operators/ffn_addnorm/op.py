# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
import logging
from pathlib import Path

from operators.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)

from operators.common.utils import torch_to_numpy, numpy_to_torch


class AIEFFNAN(AIEOperatorBase):
    """
    AIE-accelerated ANFFN block for BERT, which has an add & norm, up-projection and down-projection with a GeLU in between, then another add & norm.
    The ANFFN block computes: R2 = LN(A) + R, C = LN(GeLU(R2 @ B_Up) @ B_Down) + R2
    where A and R are of shape (M, K), B_Up is of shape (K, N), B_Down is of shape (N, K), and C is of shape (M, K).
    The operator supports static weights for B_Up, B_Down, and the two layer norm weights.
    The add & norm stage is pipelined to the up-projection, and the output of that is pipelined to down-projection fused with GeLU, which is then
    pipelined to the final add & norm stage.
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
        down_proj_depth=1,
        num_aie_columns=2,
        ln2_weight=None,
        debug_mode=-1,
        context=None,
        skip_add_to_list=False,
        **anffn_kwargs,
    ):

        self.tile_m = tile_m
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.down_proj_depth = down_proj_depth
        self.num_aie_columns = num_aie_columns
        self.n_aie_rows = 2
        self.anffn_args = anffn_kwargs
        self.weight_up_proj = (
            None
            if not use_static_weight
            else torch.zeros((K, N), dtype=torch.bfloat16).T
        )
        self.weight_down_proj = (
            None
            if not use_static_weight
            else torch.zeros((N, K), dtype=torch.bfloat16).T
        )

        # The operator's M, K, N represent what the NPU operator supports.
        # Calls to forward() may supply matrices of different sizes, and the
        # Python code will perform necessary padding/repeated application of
        # the NPU operator.
        # M_padded, K_padded, N_padded = self._get_padded_dims(M, K, N)
        # self.M = M_padded
        # self.K = K_padded
        # self.N = N_padded
        self.M = M
        self.K = K
        self.N = N

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        self.ln2_weight = ln2_weight
        self.debug_mode = debug_mode

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="anffn_"):
        # Get parameters from self
        operator_dir = Path(__file__).parent
        tile_m = self.tile_m
        tile_k = self.tile_k
        tile_n = self.tile_n
        M = self.M
        K = self.K
        N = self.N
        down_proj_depth = self.down_proj_depth
        num_aie_columns = self.num_aie_columns
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        dtype_in = self.anffn_args.get("dtype_in", "bf16")
        dtype_out = self.anffn_args.get("dtype_out", "bf16")
        emulate_bf16_mmul_with_bfp16 = self.anffn_args.get(
            "emulate_bf16_mmul_with_bfp16", True
        )
        round_conv_even = self.anffn_args.get("round_conv_even", True)
        nA_tiles_distributed = self.anffn_args.get("nA_tiles_distributed", 1)
        nB_tiles_distributed = self.anffn_args.get("nB_tiles_distributed", 1)
        stage_only = self.anffn_args.get(
            "stage_only", None
        )  # 0: up_proj only, 1: down_proj only, None: all
        gelu_stage = self.anffn_args.get(
            "gelu_stage", 1
        )  # 0: after up_proj, 1: after down_proj

        if emulate_bf16_mmul_with_bfp16:
            min_tile_m, min_tile_k, min_tile_n = 8, 8, 8
        else:
            min_tile_m, min_tile_k, min_tile_n = 4, 8, 8
        assert tile_m >= min_tile_m, f"tile_m ({tile_m}) must be >= {min_tile_m}"
        assert tile_k >= min_tile_k, f"tile_k ({tile_k}) must be >= {min_tile_k}"
        assert tile_n >= min_tile_n, f"tile_n ({tile_n}) must be >= {min_tile_n}"

        file_name_total_base = (
            f"{prefix}{M}x{K}x{N}_"
            f"{tile_m}x{tile_k}x{tile_n}_"
            f"{down_proj_depth}_"
            f"{nA_tiles_distributed}_"
            f"{nB_tiles_distributed}_"
            f"{stage_only}_"
            f"{gelu_stage}_"
        )

        # Save the weight weights to a npy file so that the design.py can load it at compile time
        ln2_weight_file_name = (
            self.context.build_dir / f"{file_name_total_base}_ln2_weight_{self.K}.npy"
        )
        np.save(ln2_weight_file_name, torch_to_numpy(self.ln2_weight))

        kernel_archive = f"anffn_{tile_m}x{tile_k}x{tile_n}.a"

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
                "down_proj_depth": down_proj_depth,
                "nA_tiles_distributed": nA_tiles_distributed,
                "nB_tiles_distributed": nB_tiles_distributed,
                "n_aie_cols": num_aie_columns,
                "dtype_in_str": dtype_in,
                "dtype_out_str": dtype_out,
                "emulate_bf16_mmul_with_bfp16": emulate_bf16_mmul_with_bfp16,
                "trace_size": 0,
                "stage_only": stage_only,
                "gelu_stage": gelu_stage,
                "archive": kernel_archive,
                "ln2_weight_file": ln2_weight_file_name,
                "generate_taps": False,
            },
            requires_context=False,
        )
        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            f"-DDIM_M={tile_m}",
            f"-DDIM_K={tile_k}",
            f"-DDIM_N={tile_n}",
        ]
        if self.debug_mode == 0 or self.debug_mode == 1:
            encoder_kernel_flags.append(f"-DDEBUG_AIE_KERNELS={self.debug_mode}")

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_total_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            f"fused_encoder_{tile_m}x{tile_k}x{tile_n}.o",
                            depends=[
                                SourceArtifact.new(
                                    self.context.base_dir
                                    / "aie_kernels"
                                    / "aie2p"
                                    / "encoder.cc"
                                )
                            ],
                            extra_flags=encoder_kernel_flags,
                        ),
                        KernelObjectArtifact.new(
                            f"ffn_passThrough_{tile_m}x{tile_k}x{tile_n}.o",
                            extra_flags=[
                                "-DBIT_WIDTH=16",
                            ],
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
                            f"ln_passThrough_{tile_m}x{tile_k}x{tile_n}.o",
                            extra_flags=[
                                "-DBIT_WIDTH=16",
                            ],
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
            extra_flags=[
                "--dynamic-objFifos",
                "--profile",
                "-v",
                "--progress",
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_total_base}.bin",
            depends=[mlir_artifact],
            extra_flags=[
                "--dynamic-objFifos",
                "--profile",
                "-v",
                "--progress",
            ],
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

        # Describe runtime components
        # The static weights might not yet be loaded upon initialization; therefore, the provided self.static_weights field is a callback that provides the weights at set-up time.
        static_weights_up_proj = None
        if self.weight_up_proj is not None:
            static_weights_up_proj = self.weight_up_proj.T
            if isinstance(static_weights_up_proj, torch.Tensor):
                static_weights_up_proj = torch_to_numpy(static_weights_up_proj)
        static_weights_down_proj = None
        if self.weight_down_proj is not None:
            static_weights_down_proj = self.weight_down_proj.T
            if isinstance(static_weights_down_proj, torch.Tensor):
                static_weights_down_proj = torch_to_numpy(static_weights_down_proj)
        self.add_kernel(
            "anffn",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("A", self.M * self.K)
        self.add_buffer("R", self.M * self.K)
        self.add_buffer("B_Up", self.K * self.N, static_data=static_weights_up_proj)
        self.add_buffer("B_Down", self.K * self.N, static_data=static_weights_down_proj)
        self.add_buffer("C", self.M * self.K)
        self.add_to_runlist("anffn", "A", "R", "B_Up", "B_Down", "C")

    def forward(self, A, R, B_Up=None, B_Down=None):
        """Forward pass through ANFFN block: R2 = LN(A) + R, C = LN(GeLU(R2 @ B_Up) @ B_Down) + R2"""
        B_Up_shape = B_Up.shape if B_Up is not None else self.weight_up_proj.T.shape
        B_Down_shape = (
            B_Down.shape if B_Down is not None else self.weight_down_proj.T.shape
        )
        expected_output_shape = A.shape
        if expected_output_shape != R.shape:
            raise AIEOperatorConstraintError(
                "AIEFFNAN: input A and residual R must have the same shape"
            )

        # Remove batch dimension, if any
        if len(A.shape) > 2:
            A = A.view(-1, A.shape[-1])
        if len(R.shape) > 2:
            R = R.view(-1, R.shape[-1])
        if B_Up is not None and len(B_Up.shape) > 2:
            B_Up = B_Up.view(-1, B_Up_shape[-1])
        if B_Down is not None and len(B_Down.shape) > 2:
            B_Down = B_Down.view(-1, B_Down_shape[-1])

        M, K = A.shape
        K2, N = B_Up_shape
        N2, K3 = B_Down_shape

        applicable = (
            K == K2
            and K == K3
            and N == N2
            and (M <= self.M)
            and K <= self.K
            and N <= self.N
        )
        if not applicable:
            raise AIEOperatorConstraintError("AIEFFNAN: incompatible tensor shape(s)")

        A_padded = self._pad_A(torch_to_numpy(A))
        R_padded = self._pad_A(torch_to_numpy(R))
        if B_Up is not None:
            B_Up_padded = self._pad_B(torch_to_numpy(B_Up), b_col_maj=False)
        else:
            B_Up_padded = None
        if B_Down is not None:
            B_Down_padded = self._pad_B(
                torch_to_numpy(B_Down), b_col_maj=True
            )  # B_Down is shaped like it's column major, but isn't laid out like column major
        else:
            B_Down_padded = None

        logging.debug(
            f"Executing BERT ANFFN for dimensions M={M}, K={K}, N={N} using NPU operator with M={self.M}, K={self.N}, N={self.N}"
        )

        result_padded = np.zeros((M, self.K), dtype=A_padded.dtype)
        for M_lo in range(0, M, self.M):
            A_part = A_padded[M_lo : M_lo + self.M, :]
            R_part = R_padded[M_lo : M_lo + self.M, :]
            result_part = self._execute_aie_operation(
                A_part, R_part, B_Up_padded, B_Down_padded
            )
            max_M = min(M_lo + self.M, M)
            result_padded[M_lo:max_M, :] = result_part[:max_M, :]

        # ANFFN produces 2D result, reshape to expected output shape
        result = numpy_to_torch(result_padded[:M, :K])
        result = result.view(expected_output_shape)

        return result

    # def _get_padded_dims(self, M, K, N):
    #     tile_m, tile_k, tile_n = self.tile_m, self.tile_n, self.tile_k
    #     num_aie_columns = self.num_aie_columns

    #     min_M = tile_m * self.n_aie_rows
    #     min_K = tile_k
    #     min_N = tile_n * num_aie_columns

    #     # Calculate padded dimensions
    #     M_padded = ((M + min_M - 1) // min_M) * min_M
    #     K_padded = ((K + min_K - 1) // min_K) * min_K
    #     N_padded = ((N + min_N - 1) // min_N) * min_N

    #     return M_padded, K_padded, N_padded

    def _pad_A(self, A_np):
        M, K = A_np.shape
        A_padded = A_np
        if M % self.M != 0 or K != self.K:
            M_multiple = (M + self.M - 1) // self.M * self.M
            A_padded = np.zeros((M_multiple, self.K), dtype=A_np.dtype)
            A_padded[:M, :K] = A_np
        return A_padded

    def _pad_B(self, B_np, b_col_maj):
        if b_col_maj:
            K, N = B_np.shape
            B_padded_shape = (self.N, self.K)
            dim1_end = N
            dim2_end = K
        else:
            N, K = B_np.shape
            B_padded_shape = (self.K, self.N)
            dim1_end = K
            dim2_end = N
        if K == self.K and N == self.N:
            return B_np
        else:
            B_padded = np.zeros((dim1_end, dim2_end), dtype=B_np.dtype)
            B_padded[:dim1_end, :dim2_end] = B_np
        return B_padded

    def _execute_aie_operation(self, A_np, R_np, B_Up_np=None, B_Down_np=None):
        """Execute ANFFN operation on AIE hardware"""
        M, K = A_np.shape
        K2, N = B_Up_np.shape if B_Up_np is not None else self.weight_up_proj.T.shape
        N2, K3 = (
            B_Down_np.shape if B_Down_np is not None else self.weight_down_proj.T.shape
        )

        # If M is larger than kernel supports, split large GEMMs with many rows
        # into multiple invocations of the kernel. This is only supported for
        # row-wise concatenation of output in row-major order.
        assert M == self.M
        assert K == K2 and K == K3 and K == self.K
        assert N == N2 and N == self.N

        self.write_buffer("A", A_np)
        self.write_buffer("R", R_np)
        if B_Up_np is not None:
            self.write_buffer("B_Up", B_Up_np)
        if B_Down_np is not None:
            self.write_buffer("B_Down", B_Down_np)
        self.run_runlist()
        result_np = self.read_buffer("C", shape=(M, K), dtype=bfloat16)

        # Check for NaN and fail hard
        if np.isnan(result_np).any():
            nan_count = np.isnan(result_np).sum()
            total_count = result_np.size
            raise RuntimeError(
                f"AIE execution returned {nan_count}/{total_count} NaN values. "
            )

        # Convert back to torch tensor
        return result_np
