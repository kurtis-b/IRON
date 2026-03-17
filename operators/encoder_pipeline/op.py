# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

from operators.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from operators.common.aie_device_manager import pyxrt
from operators.common.utils import numpy_to_torch, torch_to_numpy
from operators.encoder_pipeline.placements import SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES


class AIEEncoderPipeline(AIEOperatorBase):
    """Minimal full encoder pipeline operator."""

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        seq_tile: int = 32,
        kv_seq_tile: int = 64,
        emb_tile: int = 96,
        ffn_tile: int | None = None,
        parallel_seq: int = 1,
        parallel_heads: int = 1,
        proj_acc_depth: int = 1,
        o_proj_acc_group_size: int = 1,
        nB_tiles_distributed: int = 1,
        ffn_intermediate_size: int | None = None,
        static_weights: bool = False,
        ln1_weight=None,
        ln2_weight=None,
        ln_weight=None,
        context=None,
        skip_add_to_list: bool = False,
    ):
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.seq_tile = seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.ffn_tile = emb_tile if ffn_tile is None else ffn_tile
        self.parallel_seq = parallel_seq
        self.parallel_heads = parallel_heads
        self.proj_acc_depth = proj_acc_depth
        self.o_proj_acc_group_size = o_proj_acc_group_size
        self.nB_tiles_distributed = nB_tiles_distributed
        self.embed_sz = d * num_heads
        self.ffn_intermediate_size = (
            4 * self.embed_sz
            if ffn_intermediate_size is None
            else ffn_intermediate_size
        )

        self._validate_configuration()

        self.w_o_proj = None
        self.weight_up_proj = None
        self.weight_down_proj = None
        if static_weights:
            self.w_o_proj = torch.zeros(
                (self.embed_sz, self.embed_sz), dtype=torch.bfloat16
            ).T
            self.weight_up_proj = torch.zeros(
                (self.embed_sz, self.ffn_intermediate_size), dtype=torch.bfloat16
            ).T
            self.weight_down_proj = torch.zeros(
                (self.ffn_intermediate_size, self.embed_sz), dtype=torch.bfloat16
            ).T

        if ln_weight is not None:
            if ln1_weight is None:
                ln1_weight = ln_weight
            if ln2_weight is None:
                ln2_weight = ln_weight
        self.ln1_weight = (
            ln1_weight
            if ln1_weight is not None
            else torch.ones(self.embed_sz, dtype=torch.bfloat16)
        )
        self.ln2_weight = (
            ln2_weight
            if ln2_weight is not None
            else torch.ones(self.embed_sz, dtype=torch.bfloat16)
        )

        self.xclbin_artifact = None
        self.insts_artifact = None
        self.front_xclbin_artifact = None
        self.front_insts_artifact = None
        self.tail_xclbin_artifact = None
        self.tail_insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def _validate_configuration(self):
        topology_key = (
            self.num_heads,
            self.seq_len,
            self.d,
            self.seq_tile,
            self.kv_seq_tile,
            self.emb_tile,
            self.ffn_tile,
            self.parallel_seq,
            self.parallel_heads,
            self.proj_acc_depth,
            self.o_proj_acc_group_size,
            self.nB_tiles_distributed,
            self.ffn_intermediate_size,
        )
        if self.d != 64:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline only supports d=64 today (got {self.d})"
            )
        if self.parallel_seq <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires parallel_seq > 0 (got {self.parallel_seq})"
            )
        if self.parallel_heads <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires parallel_heads > 0 (got {self.parallel_heads})"
            )
        if self.nB_tiles_distributed <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires nB_tiles_distributed > 0 "
                f"(got {self.nB_tiles_distributed})"
            )
        if self.seq_len % self.seq_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires seq_len divisible by seq_tile "
                f"({self.seq_len} % {self.seq_tile} != 0)"
            )
        if (self.seq_len // self.seq_tile) % self.parallel_seq != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires num_q_seq_blocks divisible by parallel_seq "
                f"({self.seq_len // self.seq_tile} % {self.parallel_seq} != 0)"
            )
        if self.embed_sz % self.emb_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile to divide embed_sz "
                f"({self.embed_sz} % {self.emb_tile} != 0)"
            )
        if self.proj_acc_depth != self.embed_sz // self.emb_tile:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile * proj_acc_depth == embed_sz "
                f"({self.emb_tile} * {self.proj_acc_depth} != {self.embed_sz})"
            )
        if self.ffn_tile % 16 != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_tile divisible by 16 "
                f"({self.ffn_tile} % 16 != 0)"
            )
        if self.ffn_intermediate_size % self.ffn_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_intermediate_size divisible by ffn_tile "
                f"({self.ffn_intermediate_size} % {self.ffn_tile} != 0)"
            )
        if self.o_proj_acc_group_size <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires o_proj_acc_group_size > 0 "
                f"(got {self.o_proj_acc_group_size})"
            )
        if topology_key not in SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES:
            raise AIEOperatorConstraintError(
                "encoder_pipeline currently supports only hardcoded placement "
                f"topologies {sorted(SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES)} "
                f"(got {topology_key})"
            )

    def _artifact_stem(self, prefix: str) -> str:
        identity = "|".join(
            map(
                str,
                [
                    prefix,
                    self.num_heads,
                    self.seq_len,
                    self.d,
                    self.seq_tile,
                    self.kv_seq_tile,
                    self.emb_tile,
                    self.ffn_tile,
                    self.parallel_seq,
                    self.parallel_heads,
                    self.proj_acc_depth,
                    self.o_proj_acc_group_size,
                    self.nB_tiles_distributed,
                    self.ffn_intermediate_size,
                ],
            )
        )
        digest = hashlib.blake2s(identity.encode(), digest_size=6).hexdigest()
        return (
            f"{prefix}_{self.num_heads}h_{self.seq_len}s_{self.emb_tile}e_"
            f"{self.ffn_tile}f_"
            f"{self.parallel_seq}ps_{self.parallel_heads}ph_{self.proj_acc_depth}pa_"
            f"{self.o_proj_acc_group_size}g_{self.nB_tiles_distributed}pf_{digest}"
        )

    def _use_seqpar_phase_split(self) -> bool:
        return self.parallel_seq > 4

    def get_artifacts(self, prefix: str = "encoder_pipeline", phase: str = "all"):
        operator_dir = Path(__file__).parent
        file_name_base = self._artifact_stem(prefix)

        ln1_weight_file_name = (
            self.context.build_dir / f"{file_name_base}_ln1_weight_{self.embed_sz}.npy"
        )
        ln2_weight_file_name = (
            self.context.build_dir / f"{file_name_base}_ln2_weight_{self.embed_sz}.npy"
        )
        np.save(ln1_weight_file_name, torch_to_numpy(self.ln1_weight))
        np.save(ln2_weight_file_name, torch_to_numpy(self.ln2_weight))

        base_dir = self.context.base_dir
        mm_source = str(base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(base_dir / "aie_kernels" / "aie2p" / "softmax.cc")
        mha_source = str(base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        passthrough_source = str(
            base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        convert_copy_source = str(
            base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
        )
        add_source = str(base_dir / "aie_kernels" / "generic" / "add.cc")
        encoder_source = str(base_dir / "aie_kernels" / "aie2p" / "encoder.cc")

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
            "-DGENERATE_MATMUL_WITH_ACC_KERNELS",
            "-DGENERATE_MATMUL_INIT_KERNELS",
        ]
        mm_o_proj_rename_symbols = {
            "matmul_bf16_bf16": "matmul_bf16_bf16_o_proj",
            "matmul_init_bf16_bf16": "matmul_init_bf16_bf16_o_proj",
            "matmul_scalar_bf16_bf16": "matmul_scalar_bf16_bf16_o_proj",
            "matmul_with_acc_bf16_bf16": "matmul_with_acc_bf16_bf16_o_proj",
            "zero_bf16": "zero_bf16_o_proj",
            "zero_scalar_bf16": "zero_scalar_bf16_o_proj",
        }
        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.ffn_tile}",
        ]

        kernel_archive = f"{file_name_base}_kernels.a"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="encoder_pipeline",
            tracked_paths=[
                operator_dir / "design.py",
                operator_dir / "op.py",
                operator_dir / "placements.py",
            ],
            callback_kwargs={
                "heads": self.num_heads,
                "seq_len": self.seq_len,
                "d": self.d,
                "seq_tile": self.seq_tile,
                "kv_seq_tile": self.kv_seq_tile,
                "emb_tile": self.emb_tile,
                "ffn_tile": self.ffn_tile,
                "parallel_seq": self.parallel_seq,
                "parallel_heads": self.parallel_heads,
                "proj_acc_depth": self.proj_acc_depth,
                "o_proj_acc_group_size": self.o_proj_acc_group_size,
                "nB_tiles_distributed": self.nB_tiles_distributed,
                "ffn_intermediate_size": self.ffn_intermediate_size,
                "emulate_bf16_mmul_with_bfp16": True,
                "kernel_archive": kernel_archive,
                "ln1_weight_file": ln1_weight_file_name,
                "ln2_weight_file": ln2_weight_file_name,
                "trace_size": 0,
                "phase": phase,
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
                            f"{prefix}_mm_qk_{self.seq_tile}m_{self.d}k_{self.kv_seq_tile}n.o",
                            extra_flags=mm_defines_colmaj,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_mm_pv_rowmaj_{self.seq_tile}m_{self.kv_seq_tile}k_{self.d}n.o",
                            extra_flags=mm_defines_rowmaj,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_mm_o_{self.seq_tile}m_{self.emb_tile}n_{self.d}k.o",
                            extra_flags=mm_o_proj_defines,
                            depends=[SourceArtifact.new(mm_source)],
                            rename_symbols=mm_o_proj_rename_symbols,
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_softmax_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k.o",
                            depends=[SourceArtifact.new(softmax_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_mha_{self.seq_tile}m_{self.kv_seq_tile}n_{self.d}k_causal0_0.o",
                            depends=[SourceArtifact.new(mha_source)],
                            extra_flags=[
                                "-DIS_CAUSAL=0",
                                f"-DVECTOR_LENGTH={self.seq_tile}",
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_passThrough_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_passThrough_o_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            extra_flags=["-DBIT_WIDTH=16"],
                            depends=[SourceArtifact.new(passthrough_source)],
                            rename_symbols={
                                "passThroughLine": "passThroughLine_o_proj",
                                "passThroughTile": "passThroughTile_o_proj",
                            },
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_convert_copy.o",
                            depends=[SourceArtifact.new(convert_copy_source)],
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_add_{self.seq_tile}m_{self.seq_tile}n_{self.d}k.o",
                            depends=[SourceArtifact.new(add_source)],
                            rename_symbols={
                                "eltwise_add_bf16_scalar": "eltwise_add_bf16_scalar_o_proj",
                                "eltwise_add_bf16_vector": "eltwise_add_bf16_vector_o_proj",
                                "eltwise_add_f32_vector": "eltwise_add_f32_vector_o_proj",
                            },
                        ),
                        KernelObjectArtifact.new(
                            f"{prefix}_encoder_{self.seq_tile}x{self.emb_tile}x{self.ffn_tile}.o",
                            depends=[SourceArtifact.new(encoder_source)],
                            extra_flags=encoder_kernel_flags,
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
        if self._use_seqpar_phase_split():
            (
                self.front_xclbin_artifact,
                self.front_insts_artifact,
            ) = self.get_artifacts(prefix="encoder_pipeline_front", phase="front")
            (
                self.tail_xclbin_artifact,
                self.tail_insts_artifact,
            ) = self.get_artifacts(prefix="encoder_pipeline_tail", phase="tail")
            self.add_artifacts(
                [
                    self.front_xclbin_artifact,
                    self.front_insts_artifact,
                    self.tail_xclbin_artifact,
                    self.tail_insts_artifact,
                ]
            )
            return

        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def _or_buffer_shape(self):
        if self.parallel_seq > 1:
            ln1_stage_rows = self.seq_len
        else:
            ln1_stage_rows = self.proj_acc_depth * self.seq_tile
        return (2 * self.seq_len + ln1_stage_rows, self.embed_sz)

    def set_up_runtime(self):
        if self._use_seqpar_phase_split():
            self.add_kernel(
                "encoder_pipeline_front",
                self.front_xclbin_artifact,
                self.front_xclbin_artifact.kernel_name,
                self.front_insts_artifact,
            )
            self.add_kernel(
                "encoder_pipeline_tail",
                self.tail_xclbin_artifact,
                self.tail_xclbin_artifact.kernel_name,
                self.tail_insts_artifact,
            )
        else:
            self.add_kernel(
                "encoder_pipeline",
                self.xclbin_artifact,
                self.xclbin_artifact.kernel_name,
                self.insts_artifact,
            )

        static_w_o_proj = None
        if self.w_o_proj is not None:
            static_w_o_proj = self.w_o_proj.T
            if isinstance(static_w_o_proj, torch.Tensor):
                static_w_o_proj = torch_to_numpy(static_w_o_proj)
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

        self.add_buffer(
            "W_O", self.embed_sz * self.embed_sz, static_data=static_w_o_proj
        )
        self.add_buffer("QKV", 3 * self.embed_sz * self.seq_len)
        self.add_buffer("OR", int(np.prod(self._or_buffer_shape())))
        self.add_buffer("O", self.embed_sz * self.seq_len)
        self.add_buffer(
            "B_Up",
            self.embed_sz * self.ffn_intermediate_size,
            static_data=static_weights_up_proj,
        )
        self.add_buffer(
            "B_Down",
            self.ffn_intermediate_size * self.embed_sz,
            static_data=static_weights_down_proj,
        )
        self.buffer_aliases["O"] = "OR"
        if self._use_seqpar_phase_split():
            self.add_to_runlist("encoder_pipeline_front", "W_O", "QKV", "OR")
            self.add_to_runlist("encoder_pipeline_tail", "OR", "B_Up", "B_Down")
        else:
            self.add_to_runlist(
                "encoder_pipeline", "W_O", "QKV", "OR", "B_Up", "B_Down"
            )

    def run_runlist(self):
        if not self._use_seqpar_phase_split():
            return super().run_runlist()

        elapsed = 0.0
        for kernel_name, *buffer_args in self.runlist:
            context, xrt_kernel, insts_bo, insts_len = self.xrt_kernels[kernel_name]
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bos = [self.buffer_bos[buffer_arg] for buffer_arg in buffer_args]
            for bo in bos:
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            opcode = 3
            start = time.perf_counter()
            run = xrt_kernel(opcode, insts_bo, insts_len, *bos)
            result = run.wait()
            stop = time.perf_counter()
            elapsed += stop - start
            if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(
                    f"Kernel {kernel_name} did not complete correctly: {result}"
                )
            for bo in bos:
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return elapsed

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor | None = None,
        w_o: torch.Tensor | None = None,
        b_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ):
        b_up_shape = (
            b_up.shape
            if b_up is not None
            else (
                self.weight_up_proj.T.shape if self.weight_up_proj is not None else None
            )
        )
        b_down_shape = (
            b_down.shape
            if b_down is not None
            else (
                self.weight_down_proj.T.shape
                if self.weight_down_proj is not None
                else None
            )
        )
        if b_up_shape is None or b_down_shape is None:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: B_Up/B_Down must be provided unless static_weights=True"
            )

        expected_b_up = (self.embed_sz, self.ffn_intermediate_size)
        expected_b_down = (self.ffn_intermediate_size, self.embed_sz)
        if tuple(b_up_shape) != expected_b_up or tuple(b_down_shape) != expected_b_down:
            raise AIEOperatorConstraintError(
                f"AIEEncoderPipeline: expected B_Up shape {expected_b_up} and B_Down shape {expected_b_down}, "
                f"got {tuple(b_up_shape)} and {tuple(b_down_shape)}"
            )

        applicable = (
            q.shape[-1] == self.d
            and k.shape[-1] == self.d
            and v.shape[-1] == self.d
            and q.shape[-2] == self.seq_len
            and k.shape[-2] == self.seq_len
            and v.shape[-2] == self.seq_len
            and self.seq_len % 64 == 0
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEEncoderPipeline: incompatible tensor shape(s)"
            )
        return self._execute_aie_operation(q, k, v, r, w_o, b_up, b_down)

    def _execute_aie_operation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor | None,
        w_o: torch.Tensor | None,
        b_up: torch.Tensor | None,
        b_down: torch.Tensor | None,
    ):
        q_np = torch_to_numpy(q)
        k_np = torch_to_numpy(k)
        v_np = torch_to_numpy(v)
        qkv_np = np.concatenate((q_np, k_np, v_np), axis=0)

        if r is not None:
            r_np = torch_to_numpy(r)
        else:
            r_np = np.zeros((self.seq_len, self.embed_sz), dtype=bfloat16)
        or_np = np.concatenate((np.zeros_like(r_np), r_np), axis=0)

        ln1_stage_rows = self._or_buffer_shape()[0] - 2 * self.seq_len
        if ln1_stage_rows > 0:
            or_np = np.concatenate(
                (
                    or_np,
                    np.zeros((ln1_stage_rows, self.embed_sz), dtype=or_np.dtype),
                ),
                axis=0,
            )

        self.write_buffer("QKV", qkv_np)
        self.write_buffer("OR", or_np)
        if w_o is not None:
            self.write_buffer("W_O", torch_to_numpy(w_o))
        if b_up is not None:
            self.write_buffer("B_Up", torch_to_numpy(b_up))
        if b_down is not None:
            self.write_buffer("B_Down", torch_to_numpy(b_down))

        self.run_runlist()
        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        return numpy_to_torch(np.asarray(o_np))
