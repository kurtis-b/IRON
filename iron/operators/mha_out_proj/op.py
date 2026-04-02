# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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


class AIEMHAOutProj(AIEOperatorBase):
    """AIE-accelerated multi-head attention with output projection."""

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        parallel_seq: int = 1,
        q_seq_tile: int = 32,
        kv_seq_tile: int = 64,
        emb_tile: int = 64,
        parallel_heads: int = 1,
        o_proj_acc_depth: int = 1,
        static_weights: bool = False,
        context=None,
        skip_add_to_list=False,
    ):
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.parallel_seq = parallel_seq
        self.q_seq_tile = q_seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.parallel_heads = parallel_heads
        self.o_proj_acc_depth = o_proj_acc_depth
        self.embed_sz = num_heads * d

        self.weight_o_proj = (
            None
            if not static_weights
            else torch.zeros((self.embed_sz, self.embed_sz), dtype=torch.bfloat16)
        )

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent

        file_name_base = (
            f"mha_out_proj_{self.num_heads}h_{self.seq_len}s_{self.d}d_"
            f"{self.q_seq_tile}q_{self.kv_seq_tile}kv_{self.emb_tile}e_"
            f"{self.parallel_seq}ps_{self.parallel_heads}ph_{self.o_proj_acc_depth}acc"
        )

        mm_source = self.context.base_dir / "aie_kernels" / "aie2p" / "mm.cc"
        softmax_source = self.context.base_dir / "aie_kernels" / "aie2p" / "softmax.cc"
        mha_source = self.context.base_dir / "aie_kernels" / "aie2p" / "mha.cc"
        passthrough_source = (
            self.context.base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        add_source = self.context.base_dir / "aie_kernels" / "generic" / "add.cc"

        mm_defines_rowmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.q_seq_tile}",
            f"-DDIM_K={self.kv_seq_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        mm_defines_colmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.q_seq_tile}",
            f"-DDIM_K={self.d}",
            f"-DDIM_N={self.kv_seq_tile}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DB_COL_MAJ",
        ]
        mm_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_rowmaj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_rowmaj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_rowmaj",
            "zero_bf16": "zero_bf16_rowmaj",
            "zero_scalar_bf16": "zero_scalar_bf16_rowmaj",
        }
        mm_o_proj_defines = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.q_seq_tile}",
            f"-DDIM_K={self.d}",
            f"-DDIM_N={self.emb_tile}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
        ]
        mm_o_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_o_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_o_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_o_proj",
            "zero_bf16": "zero_bf16_o_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_o_proj",
        }

        kernel_archive = f"{file_name_base}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_mha",
            callback_kwargs={
                "heads": self.num_heads,
                "seq_len": self.seq_len,
                "d": self.d,
                "parallel_seq": self.parallel_seq,
                "q_seq_tile": self.q_seq_tile,
                "kv_seq_tile": self.kv_seq_tile,
                "emb_tile": self.emb_tile,
                "o_proj_acc_depth": self.o_proj_acc_depth,
                "parallel_heads": self.parallel_heads,
                "emulate_bf16_mmul_with_bfp16": True,
                "kernel_archive": kernel_archive,
                "trace_size": 0,
            },
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            f"mha_mm_{self.q_seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                            extra_flags=mm_defines_colmaj,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_mm_rowmaj_{self.q_seq_tile}m_{self.kv_seq_tile}k_{self.d}n.o",
                            extra_flags=mm_defines_rowmaj,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_mm_o_proj_{self.q_seq_tile}m_{self.d}k_{self.emb_tile}n.o",
                            extra_flags=mm_o_proj_defines,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_o_proj_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_softmax_{self.q_seq_tile}m_{self.kv_seq_tile}n.o",
                            depends=[SourceArtifact.new(softmax_source)],
                            extra_flags=[f"-DSM_VEC_LEN={self.kv_seq_tile}"],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_{self.q_seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                            depends=[SourceArtifact.new(mha_source)],
                            extra_flags=[
                                "-DIS_CAUSAL=0",
                                "-DDEBUG=0",
                                f"-DVECTOR_LENGTH={min(self.q_seq_tile, self.kv_seq_tile)}",
                                f"-DSCALE_VECTOR_LENGTH={self.q_seq_tile}",
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_pass_through_{self.q_seq_tile}m_{self.emb_tile}n.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_pass_through_o_proj_{self.q_seq_tile}m_{self.emb_tile}n.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                            rename_symbols={
                                "passThroughLine": "passThroughLine_o_proj",
                                "passThroughTile": "passThroughTile_o_proj",
                            },
                        ),
                        KernelObjectArtifact.new(
                            f"mha_add_{self.q_seq_tile}m_{self.emb_tile}n.o",
                            depends=[SourceArtifact.new(add_source)],
                            rename_symbols={
                                "eltwise_add_bf16_scalar": "eltwise_add_bf16_scalar_o_proj",
                                "eltwise_add_bf16_vector": "eltwise_add_bf16_vector_o_proj",
                                "eltwise_add_f32_vector": "eltwise_add_f32_vector_o_proj",
                            },
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

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        static_weight = None
        if self.weight_o_proj is not None:
            static_weight = torch_to_numpy(self.weight_o_proj)

        self.add_kernel(
            "mha_out_proj",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("W_O", self.embed_sz * self.embed_sz, static_data=static_weight)
        self.add_buffer("Q", self.embed_sz * self.seq_len)
        self.add_buffer("K", self.embed_sz * self.seq_len)
        self.add_buffer("V", self.embed_sz * self.seq_len)
        self.add_buffer("O", self.embed_sz * self.seq_len)
        self.add_to_runlist("mha_out_proj", "W_O", "Q", "K", "V", "O")

    def forward(self, q, k, v, W_O=None):
        expected_qkv_shape = (self.seq_len, self.embed_sz)
        if tuple(q.shape) != expected_qkv_shape:
            raise AIEOperatorConstraintError(
                f"AIEMHAOutProj: expected q shape {expected_qkv_shape}"
            )
        if tuple(k.shape) != expected_qkv_shape:
            raise AIEOperatorConstraintError(
                f"AIEMHAOutProj: expected k shape {expected_qkv_shape}"
            )
        if tuple(v.shape) != expected_qkv_shape:
            raise AIEOperatorConstraintError(
                f"AIEMHAOutProj: expected v shape {expected_qkv_shape}"
            )
        if W_O is None and self.weight_o_proj is None:
            raise AIEOperatorConstraintError(
                "AIEMHAOutProj: missing output projection weight"
            )
        if W_O is not None and tuple(W_O.shape) != (self.embed_sz, self.embed_sz):
            raise AIEOperatorConstraintError(
                "AIEMHAOutProj: expected W_O shape " f"{(self.embed_sz, self.embed_sz)}"
            )

        return self._execute_aie_operation(q, k, v, W_O)

    def _execute_aie_operation(self, q, k, v, W_O=None):
        self.write_buffer("Q", q)
        self.write_buffer("K", k)
        self.write_buffer("V", v)
        if W_O is not None:
            self.write_buffer("W_O", W_O)

        self.run_runlist()
        return self.read_buffer_as_torch(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
