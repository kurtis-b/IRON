# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
from iron.common.utils import torch_to_numpy


class AIEQKVProj(AIEOperatorBase):
    """AIE-accelerated fused Q/K/V projection."""

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        use_static_weight=False,
        tile_m=32,
        tile_k=64,
        tile_n=16,
        parallel_seq=1,
        parallel_emb=1,
        context=None,
        skip_add_to_list=False,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.combined_hidden_size = 3 * hidden_size
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.parallel_seq = parallel_seq
        self.parallel_emb = parallel_emb

        self.M = seq_len
        self.K = hidden_size
        self.N = self.combined_hidden_size

        self.weight = (
            None
            if not use_static_weight
            else torch.zeros((self.K, self.N), dtype=torch.bfloat16)
        )

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="qkv_proj_"):
        operator_dir = Path(__file__).parent
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        file_name_base = (
            f"{prefix}{self.M}x{self.K}x{self.N}_"
            f"{self.tile_m}x{self.tile_k}x{self.tile_n}_"
            f"ps{self.parallel_seq}_pe{self.parallel_emb}"
            "_0_0_bf16_bf16_sc0_acc0_embf161_round1_batchA1d0_batchB1d0_batchC1d0"
        )

        kernel_archive = (
            f"qkv_proj_{self.tile_m}x{self.tile_k}x{self.tile_n}_0_0_bf16_bf16"
            "_sc0_acc0_embf161_round1.a"
        )
        mm_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_qkv_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_qkv_proj",
            "zero_bf16": "zero_bf16_qkv_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_qkv_proj",
        }
        kernel_flags = [
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
            "-Dbf16_bf16_ONLY",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_qkv_proj",
            callback_kwargs={
                "dev": device_str,
                "seq_len": self.M,
                "hidden_size": self.K,
                "combined_hidden_size": self.N,
                "tile_m": self.tile_m,
                "tile_k": self.tile_k,
                "tile_n": self.tile_n,
                "parallel_seq": self.parallel_seq,
                "parallel_emb": self.parallel_emb,
                "dtype_in_str": "bf16",
                "dtype_out_str": "bf16",
                "use_scalar": False,
                "emulate_bf16_mmul_with_bfp16": True,
                "prio_accuracy": False,
                "trace_size": 0,
                "archive": kernel_archive,
            },
            requires_context=False,
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            (
                                f"qkv_proj_{self.tile_m}x{self.tile_k}x{self.tile_n}_0_0"
                                "_acc0_embf161_round1.o"
                            ),
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "mm.cc"
                                )
                            ],
                            rename_symbols=mm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            "zero_scalar.o",
                            depends=[
                                SourceArtifact.new(
                                    base_dir / "aie_kernels" / "aie2p" / "zero.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "convert_copy.o",
                            depends=[
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
            f"{file_name_base}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        static_weight = None
        if self.weight is not None:
            static_weight = torch_to_numpy(self.weight)

        self.add_kernel(
            "qkv_proj",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("A", self.M * self.K)
        self.add_buffer("B", self.K * self.N, static_data=static_weight)
        self.add_buffer("Q", self.M * self.K)
        self.add_buffer("K", self.M * self.K)
        self.add_buffer("V", self.M * self.K)
        self.add_to_runlist("qkv_proj", "A", "B", "Q", "K", "V")

    def forward(self, hidden_states, B=None):
        if tuple(hidden_states.shape) != (self.seq_len, self.hidden_size):
            raise AIEOperatorConstraintError(
                "AIEQKVProj: expected hidden_states shape "
                f"{(self.seq_len, self.hidden_size)}"
            )
        if B is None and self.weight is None:
            raise AIEOperatorConstraintError("AIEQKVProj: missing stacked QKV weight")
        if B is not None and tuple(B.shape) != (
            self.hidden_size,
            self.combined_hidden_size,
        ):
            raise AIEOperatorConstraintError(
                "AIEQKVProj: expected stacked weight shape "
                f"{(self.hidden_size, self.combined_hidden_size)}"
            )

        return self._execute_aie_operation(hidden_states, B)

    def _execute_aie_operation(self, hidden_states, B=None):
        self.write_buffer("A", hidden_states)
        if B is not None:
            self.write_buffer("B", B)
        self.run_runlist()

        q = self.read_buffer_as_torch(
            "Q", shape=(self.seq_len, self.hidden_size), dtype=bfloat16
        )
        k = self.read_buffer_as_torch(
            "K", shape=(self.seq_len, self.hidden_size), dtype=bfloat16
        )
        v = self.read_buffer_as_torch(
            "V", shape=(self.seq_len, self.hidden_size), dtype=bfloat16
        )
        return q, k, v
