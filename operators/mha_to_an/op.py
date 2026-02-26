# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path
from typing import Dict, List

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


class AIEMHAOutProj(AIEOperatorBase):

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        seq_tile: int = 64,
        kv_seq_tile: int = 64,
        emb_tile: int = 96,
        parallel_heads: int = 1,
        o_proj_acc_depth: int = 1,
        static_weights: bool = False,
        debug: int = 0,
        ln_weight=None,
        context=None,
        skip_add_to_list=False,
    ):
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.seq_tile = seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.parallel_heads = parallel_heads
        self.o_proj_acc_depth = o_proj_acc_depth
        self.debug = debug
        self.embed_sz = d * num_heads
        assert d == 64, "Only d=64 is supported in this version"

        # Allocate static weights before inference
        self.w_o_proj = None
        if static_weights:
            self.w_o_proj = torch.zeros(
                (self.embed_sz, self.embed_sz), dtype=torch.bfloat16
            ).T

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        self.ln_weight = ln_weight

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="mha_o_proj"):
        # Set up compilation artifacts
        # ---
        operator_dir = Path(__file__).parent

        file_name_base = f"mha_to_an_{self.num_heads}h_{self.seq_len}s_{self.d}d_{self.seq_tile}qt_{self.kv_seq_tile}kvt_{self.emb_tile}e_{self.parallel_heads}ph_{self.o_proj_acc_depth}acc_lnstage"

        # Save layer norm weights to .npy for use at compile time
        ln_weight_file_name = (
            self.context.build_dir / f"{file_name_base}_ln_weight_{self.embed_sz}.npy"
        )
        np.save(ln_weight_file_name, torch_to_numpy(self.ln_weight))

        # Define source files
        mm_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(
            self.context.base_dir / "aie_kernels" / "aie2p" / "softmax.cc"
        )
        mha_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        passthrough_source = str(
            self.context.base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        add_source = str(self.context.base_dir / "aie_kernels" / "generic" / "add.cc")

        # Compile mm.cc (col-major)
        mm_defines_rowmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.kv_seq_tile}",
            f"-DDIM_N={self.d}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        mm_defines_colmaj = [
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={self.seq_tile}",
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
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.d}",
            f"-DDIM_N={self.emb_tile}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",  # NOTE: Matmul with accumulation won't be included in compilation unless this is passed in
        ]
        mm_o_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_o_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_o_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_o_proj",
            "zero_bf16": "zero_bf16_o_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_o_proj",
        }

        kernel_archive = f"mha_to_an_kernels_{self.num_heads}h_{self.seq_len}s_{self.d}d_{self.debug}debug_lnstage.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_mha",
            callback_kwargs={
                "heads": self.num_heads,
                "seq_len": self.seq_len,
                "d": self.d,
                "seq_tile": self.seq_tile,
                "kv_seq_tile": self.kv_seq_tile,
                "emb_tile": self.emb_tile,
                "o_proj_acc_depth": self.o_proj_acc_depth,
                "parallel_heads": self.parallel_heads,
                "emulate_bf16_mmul_with_bfp16": True,
                "kernel_archive": kernel_archive,
                "trace_size": 0,
                "ln_weight_file": ln_weight_file_name,
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
                            f"mha_o_proj_mm_qk_{self.seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                            extra_flags=mm_defines_colmaj,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mm_pv_rowmaj_{self.seq_tile}m_{self.kv_seq_tile}k_{self.d}n.o",
                            extra_flags=mm_defines_rowmaj,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mm_o_{self.seq_tile}m_{self.emb_tile}n_{self.d}k.o",
                            extra_flags=mm_o_proj_defines,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_o_proj_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_softmax_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k.o",
                            depends=[SourceArtifact.new(softmax_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mha_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k_causal0_{self.debug}.o",
                            depends=[SourceArtifact.new(mha_source)],
                            extra_flags=[
                                "-DIS_CAUSAL=0",
                                f"-DDEBUG={self.debug}",
                                f"-DVECTOR_LENGTH={self.seq_tile}",
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_passThrough_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_passThrough_o_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                            rename_symbols={
                                "passThroughLine": "passThroughLine_o_proj",
                                "passThroughTile": "passThroughTile_o_proj",
                            },
                        ),
                        KernelObjectArtifact.new(
                            f"mha_add_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            depends=[SourceArtifact.new(add_source)],
                            rename_symbols={
                                "eltwise_add_bf16_scalar": "eltwise_add_bf16_scalar_o_proj",
                                "eltwise_add_bf16_vector": "eltwise_add_bf16_vector_o_proj",
                                "eltwise_add_f32_vector": "eltwise_add_f32_vector_o_proj",
                            },
                        ),
                        KernelObjectArtifact.new(
                            f"mha_to_an_encoder_{self.seq_tile}x{self.emb_tile}.o",
                            depends=[
                                SourceArtifact.new(
                                    self.context.base_dir
                                    / "aie_kernels"
                                    / "aie2p"
                                    / "encoder.cc"
                                )
                            ],
                            extra_flags=[
                                "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                                "-DBUILD_ADDNORM",
                                f"-DDIM_M={self.seq_tile}",
                                f"-DDIM_K={self.emb_tile}",
                                f"-DDIM_N={self.emb_tile}",  # Unused for layer norm but required for compilation, set to emb_tile for simplicity
                                # TODO: Need to think about how debug mode should work here
                                # f"-DDEBUG_AIE_KERNELS={self.debug_mode}",
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

        return (xclbin_artifact, insts_artifact)

    def set_up_artifacts(self):
        # Describe required artifacts (xclbin, insts.bin)
        device_str = self.context.device_manager.device_str()
        xclbin_artifact, insts_artifact = self.get_artifacts()

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact

        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        # Set up runtime
        # ---
        self.add_kernel(
            "mha",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        static_w_o_proj = None
        if self.w_o_proj is not None:
            static_w_o_proj = self.w_o_proj.T
            if isinstance(static_w_o_proj, torch.Tensor):
                static_w_o_proj = torch_to_numpy(static_w_o_proj)
        self.add_buffer(
            "W_O",
            self.embed_sz * self.embed_sz,
            static_data=static_w_o_proj,
        )
        self.add_buffer(
            "QKV",
            3 * self.embed_sz * self.seq_len,
        )
        self.add_buffer(
            "OR",
            2 * self.embed_sz * self.seq_len,
        )
        self.add_buffer(
            "O",
            self.embed_sz * self.seq_len,
        )
        # Output view aliases the first half of OR.
        self.buffer_aliases["O"] = "OR"
        self.add_to_runlist("mha", "W_O", "QKV", "OR")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor = None,
        w_o: torch.Tensor = None,
    ):
        applicable = (
            q.shape[-1] == self.d
            and k.shape[-1] == self.d
            and v.shape[-1] == self.d
            and q.shape[-2] == self.seq_len
            and k.shape[-2] == self.seq_len
            and v.shape[-2] == self.seq_len
            and self.seq_len % 64 == 0,  # Sequence length must be multiple of 64
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEElementwiseAdd: incompatible tensor shape(s)"
            )

        ret = self._execute_aie_operation(q, k, v, r, w_o)
        return ret

    def _execute_aie_operation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor = None,
        w_o: torch.Tensor = None,
    ):
        # Convert to numpy
        q_np = torch_to_numpy(q)
        k_np = torch_to_numpy(k)
        v_np = torch_to_numpy(v)
        qkv_np = np.concatenate((q_np, k_np, v_np), axis=0)
        if r is not None:
            r_np = torch_to_numpy(r)
        else:
            r_np = np.zeros((self.seq_len, self.embed_sz), dtype=bfloat16)
        or_np = np.concatenate((np.zeros_like(r_np), r_np), axis=0)

        # Write padded buffers
        self.write_buffer("QKV", qkv_np)
        self.write_buffer("OR", or_np)
        if w_o is not None:
            w_o_np = torch_to_numpy(w_o)
            self.write_buffer("W_O", w_o_np)

        # Execute
        self.run_runlist()

        # Read padded output
        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )

        # Convert back to torch with correct shape
        result = numpy_to_torch(o_np)
        return result
