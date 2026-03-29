# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
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
from iron.operators.mha_out_proj.design import mha_out_proj_design


def _canonicalize_head_major_qkv(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    num_heads: int,
    d: int,
    embed_sz: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim == 2:
        if tensor.shape != (seq_len, embed_sz):
            raise AIEOperatorConstraintError(
                f"AIEMHAOutProj: expected {name} shape {(seq_len, embed_sz)}"
            )
        return tensor.view(seq_len, num_heads, d).permute(1, 0, 2).contiguous()
    if tensor.ndim == 3 and tensor.shape == (num_heads, seq_len, d):
        return tensor.contiguous()
    raise AIEOperatorConstraintError("AIEMHAOutProj: incompatible tensor shape(s)")


def _flatten_head_major_qkv(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    embed_sz: int,
) -> np.ndarray:
    return torch_to_numpy(tensor.permute(1, 0, 2).contiguous().view(seq_len, embed_sz))


def _pack_qkv_head_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_len: int,
    embed_sz: int,
) -> np.ndarray:
    q_np = _flatten_head_major_qkv(q, seq_len=seq_len, embed_sz=embed_sz)
    k_np = _flatten_head_major_qkv(k, seq_len=seq_len, embed_sz=embed_sz)
    v_np = _flatten_head_major_qkv(v, seq_len=seq_len, embed_sz=embed_sz)
    return np.concatenate((q_np, k_np, v_np), axis=0)


class AIEMHAOutProj(AIEOperatorBase):
    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        q_seq_tile: int = 32,
        kv_seq_tile: int = 64,
        emb_tile: int = 96,
        parallel_heads: int = 1,
        o_proj_acc_depth: int = 1,
        topology_id: str | None = None,
        static_weights: bool = False,
        debug: int = 0,
        context=None,
        skip_add_to_list=False,
    ):
        config = None
        if topology_id is not None:
            config = mha_out_proj_design(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=d,
                topology_id=topology_id,
            )
            q_seq_tile = int(config["q_seq_tile"])
            kv_seq_tile = int(config["kv_seq_tile"])
            emb_tile = int(config["emb_tile"])
            parallel_heads = int(config["parallel_heads"])
            o_proj_acc_depth = int(config["o_proj_acc_depth"])
        else:
            try:
                candidate = mha_out_proj_design(
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=d,
                )
            except ValueError:
                candidate = None
            if candidate is not None and (
                int(candidate["q_seq_tile"]) == q_seq_tile
                and int(candidate["kv_seq_tile"]) == kv_seq_tile
                and int(candidate["emb_tile"]) == emb_tile
                and int(candidate["parallel_heads"]) == parallel_heads
                and int(candidate["o_proj_acc_depth"]) == o_proj_acc_depth
            ):
                config = candidate

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.q_seq_tile = q_seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.parallel_heads = parallel_heads
        self.o_proj_acc_depth = o_proj_acc_depth
        self.topology_id = (
            str(config["topology_id"])
            if config is not None
            else (
                f"q{q_seq_tile}_kv{kv_seq_tile}_e{emb_tile}"
                f"_ph{parallel_heads}_acc{o_proj_acc_depth}"
            )
        )
        self.topology_family = (
            str(config["topology_family"])
            if config is not None
            else "custom_fused_mha_out_proj"
        )
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

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def set_up_artifacts(self):
        # Set up compilation artifacts
        # ---
        operator_dir = Path(__file__).parent

        file_name_base = f"mha_o_proj_{self.num_heads}h_{self.seq_len}s_{self.d}d_{self.q_seq_tile}qseqtile_{self.kv_seq_tile}kvseqtile_{self.emb_tile}e_{self.parallel_heads}ph_{self.o_proj_acc_depth}acc"

        mm_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(
            self.context.base_dir / "aie_kernels" / "aie2p" / "softmax.cc"
        )
        mha_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        passthrough_source = str(
            self.context.base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        add_source = str(self.context.base_dir / "aie_kernels" / "generic" / "add.cc")

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

        kernel_archive = f"mha_o_proj_kernels_{self.num_heads}h_{self.seq_len}s_{self.d}d_{self.debug}debug.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_mha",
            callback_kwargs={
                "heads": self.num_heads,
                "seq_len": self.seq_len,
                "d": self.d,
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
                            f"mha_o_proj_mm_{self.q_seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                            extra_flags=mm_defines_colmaj,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mm_rowmaj_{self.q_seq_tile}m_{self.kv_seq_tile}k_{self.d}n.o",
                            extra_flags=mm_defines_rowmaj,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mm_o_{self.q_seq_tile}m_{self.d}k_{self.emb_tile}n.o",
                            extra_flags=mm_o_proj_defines,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_o_proj_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_softmax_{self.q_seq_tile}m_{self.kv_seq_tile}n.o",
                            depends=[SourceArtifact.new(softmax_source)],
                            extra_flags=[f"-DSM_VEC_LEN={self.kv_seq_tile}"],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_mha_{self.q_seq_tile}m_{self.d}k_{self.kv_seq_tile}n_causal0_{self.debug}.o",
                            depends=[SourceArtifact.new(mha_source)],
                            extra_flags=[
                                "-DIS_CAUSAL=0",
                                f"-DDEBUG={self.debug}",
                                f"-DVECTOR_LENGTH={min(self.q_seq_tile, self.kv_seq_tile)}",
                                f"-DSCALE_VECTOR_LENGTH={self.q_seq_tile}",
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_passThrough_{self.q_seq_tile}m_{self.emb_tile}n.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"mha_o_proj_passThrough_o_{self.q_seq_tile}m_{self.emb_tile}n.o",
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
            "Q",
            self.embed_sz * self.seq_len,
        )
        self.add_buffer(
            "K",
            self.embed_sz * self.seq_len,
        )
        self.add_buffer(
            "V",
            self.embed_sz * self.seq_len,
        )
        self.add_buffer(
            "O",
            self.embed_sz * self.seq_len,
        )
        self.add_to_runlist("mha", "W_O", "Q", "K", "V", "O")

    # TODO: Update forward and execute functions
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_o: torch.Tensor = None,
    ):
        q = _canonicalize_head_major_qkv(
            q,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            d=self.d,
            embed_sz=self.embed_sz,
            name="q",
        )
        k = _canonicalize_head_major_qkv(
            k,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            d=self.d,
            embed_sz=self.embed_sz,
            name="k",
        )
        v = _canonicalize_head_major_qkv(
            v,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            d=self.d,
            embed_sz=self.embed_sz,
            name="v",
        )
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

        ret = self._execute_aie_operation(q, k, v, w_o)
        return ret

    def _execute_aie_operation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_o: torch.Tensor = None,
    ):
        q_np = _flatten_head_major_qkv(
            q,
            seq_len=self.seq_len,
            embed_sz=self.embed_sz,
        )
        k_np = _flatten_head_major_qkv(
            k,
            seq_len=self.seq_len,
            embed_sz=self.embed_sz,
        )
        v_np = _flatten_head_major_qkv(
            v,
            seq_len=self.seq_len,
            embed_sz=self.embed_sz,
        )

        # Write padded buffers
        self.write_buffer("Q", q_np)
        self.write_buffer("K", k_np)
        self.write_buffer("V", v_np)
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
