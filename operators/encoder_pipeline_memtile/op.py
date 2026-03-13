# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

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
from operators.encoder_pipeline_memtile.debug_modes import resolve_pipeline_debug_modes


class AIEEncoderPipeline(AIEOperatorBase):
    """Encoder pipeline operator entrypoint.

    Implements a fused MHA + AddNorm1 + FFN + AddNorm2 pipeline and enforces
    shape/depth constraints required by the current hardware graph.
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        seq_tile: int = 32,
        kv_seq_tile: int = 64,
        emb_tile: int = 96,
        parallel_heads: int = 1,
        proj_acc_depth: int = 1,
        o_proj_acc_group_size: int = 1,
        nB_tiles_distributed: int = 1,
        ffn_intermediate_size: int | None = None,
        static_weights: bool = False,
        debug: int = -1,
        ln1_weight=None,
        ln2_weight=None,
        ln1_staging_design: str | None = None,
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
        self.proj_acc_depth = proj_acc_depth
        self.o_proj_acc_group_size = o_proj_acc_group_size
        self.nB_tiles_distributed = nB_tiles_distributed
        self.debug = debug
        del ln1_staging_design
        self.ln1_staging_design = "memtile"
        try:
            (
                self.mha_debug,
                self.ffn_stage_only,
                self.addnorm1_debug_mode,
                self.addnorm2_debug_mode,
            ) = resolve_pipeline_debug_modes(debug)
        except ValueError as exc:
            raise AIEOperatorConstraintError(str(exc)) from exc
        self.embed_sz = d * num_heads
        self.ffn_intermediate_size = (
            ffn_intermediate_size
            if ffn_intermediate_size is not None
            else 4 * self.embed_sz
        )
        assert d == 64, "Only d=64 is supported in this version"
        if self.nB_tiles_distributed <= 0:
            raise AIEOperatorConstraintError(
                f"encoder_pipeline requires nB_tiles_distributed > 0 (got {self.nB_tiles_distributed})"
            )
        if self.o_proj_acc_group_size <= 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires o_proj_acc_group_size > 0 "
                f"(got {self.o_proj_acc_group_size})"
            )
        if self.o_proj_acc_group_size > self.parallel_heads:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires o_proj_acc_group_size <= parallel_heads "
                f"({self.o_proj_acc_group_size} > {self.parallel_heads})"
            )
        if self.o_proj_acc_group_size not in (1, 2, 4):
            raise AIEOperatorConstraintError(
                "encoder_pipeline currently supports o_proj_acc_group_size in {1, 2, 4} "
                f"(got {self.o_proj_acc_group_size})"
            )
        if self.parallel_heads % self.o_proj_acc_group_size != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires parallel_heads divisible by "
                "o_proj_acc_group_size "
                f"({self.parallel_heads} % {self.o_proj_acc_group_size} != 0)"
            )

        expected_depth = self.embed_sz // self.emb_tile
        if self.embed_sz % self.emb_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile to divide embed_sz "
                f"({self.embed_sz} % {self.emb_tile} != 0)"
            )
        if self.proj_acc_depth != expected_depth:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires emb_tile * proj_acc_depth == embed_sz "
                f"({self.emb_tile} * {self.proj_acc_depth} != {self.embed_sz})"
            )
        if self.ffn_intermediate_size % self.emb_tile != 0:
            raise AIEOperatorConstraintError(
                "encoder_pipeline requires ffn_intermediate_size divisible by emb_tile "
                f"({self.ffn_intermediate_size} % {self.emb_tile} != 0)"
            )

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

        self.xclbin_artifact = None
        self.insts_artifact = None

        # Backward-compatible fallback: ln_weight applies to both stages.
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

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    @staticmethod
    def _resolve_ln1_staging_design(ln1_staging_design: str | None) -> str:
        raw = ln1_staging_design if ln1_staging_design is not None else "ddr"
        mode = raw.strip().lower()
        if mode in ("ddr", "dram", "host"):
            return "ddr"
        if mode in ("memtile", "mt", "onchip"):
            return "memtile"
        raise AIEOperatorConstraintError(
            "ln1_staging_design must be one of "
            "{ddr, dram, host, memtile, mt, onchip} "
            f"(got '{raw}')"
        )

    def _design_import_path(self, operator_dir: Path) -> Path:
        del operator_dir
        return Path(__file__).with_name("design.py")

    @staticmethod
    def _short_prefix(prefix: str) -> str:
        parts = [part[:1] for part in prefix.split("_") if part]
        if parts:
            return "".join(parts)
        return prefix[:4]

    def _artifact_stem(self, prefix: str) -> str:
        ffn_stage = self.ffn_stage_only if self.ffn_stage_only is not None else "all"
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
                    self.parallel_heads,
                    self.proj_acc_depth,
                    self.o_proj_acc_group_size,
                    self.nB_tiles_distributed,
                    self.ffn_intermediate_size,
                    self.debug,
                    self.mha_debug,
                    ffn_stage,
                    self.addnorm1_debug_mode,
                    self.addnorm2_debug_mode,
                    self.ln1_staging_design,
                ],
            )
        )
        digest = hashlib.blake2s(identity.encode(), digest_size=6).hexdigest()
        prefix_tag = self._short_prefix(prefix)
        stage_tag = "a" if self.ffn_stage_only is None else ffn_stage
        return (
            f"{prefix_tag}_{self.num_heads}h_{self.seq_len}s_{self.emb_tile}e_"
            f"{self.parallel_heads}ph_{self.proj_acc_depth}pa_"
            f"{self.o_proj_acc_group_size}g_{self.nB_tiles_distributed}pf_"
            f"d{self.debug}_m{self.mha_debug}_f{stage_tag}_"
            f"n1{self.addnorm1_debug_mode}_n2{self.addnorm2_debug_mode}_"
            f"{self.ln1_staging_design[:2]}_{digest}"
        )

    def get_artifacts(self, prefix="encoder_pipeline"):
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

        mm_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(
            self.context.base_dir / "aie_kernels" / "aie2p" / "softmax.cc"
        )
        mha_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        passthrough_source = str(
            self.context.base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )
        convert_copy_source = str(
            self.context.base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
        )
        add_source = str(self.context.base_dir / "aie_kernels" / "generic" / "add.cc")

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

        kernel_archive = f"{file_name_base}_kernels.a"
        encoder_kernel_flags = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DBUILD_FFN",
            "-DBUILD_ADDNORM",
            f"-DDIM_M={self.seq_tile}",
            f"-DDIM_K={self.emb_tile}",
            f"-DDIM_N={self.emb_tile}",
        ]

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=self._design_import_path(operator_dir),
            callback_fn="fused_mha",
            tracked_paths=[
                operator_dir / "design.py",
                operator_dir / "hooks.py",
                operator_dir / "debug_modes.py",
                operator_dir / "mapping_validation.py",
                operator_dir / "row_store.py",
            ],
            callback_kwargs={
                "heads": self.num_heads,
                "seq_len": self.seq_len,
                "d": self.d,
                "seq_tile": self.seq_tile,
                "kv_seq_tile": self.kv_seq_tile,
                "emb_tile": self.emb_tile,
                "proj_acc_depth": self.proj_acc_depth,
                "parallel_heads": self.parallel_heads,
                "emulate_bf16_mmul_with_bfp16": True,
                "kernel_archive": kernel_archive,
                "trace_size": 0,
                "o_proj_acc_group_size": self.o_proj_acc_group_size,
                "ln1_weight_file": ln1_weight_file_name,
                "ln2_weight_file": ln2_weight_file_name,
                "nB_tiles_distributed": self.nB_tiles_distributed,
                "ffn_intermediate_size": self.ffn_intermediate_size,
                "ffn_stage_only": self.ffn_stage_only,
                "addnorm1_debug_mode": self.addnorm1_debug_mode,
                "addnorm2_debug_mode": self.addnorm2_debug_mode,
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
                            f"{prefix}_mha_{self.seq_tile}m_{self.kv_seq_tile}n_"
                            f"{self.d}k_causal0_{self.mha_debug}.o",
                            depends=[SourceArtifact.new(mha_source)],
                            extra_flags=[
                                "-DIS_CAUSAL=0",
                                f"-DDEBUG={self.mha_debug}",
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
                            f"{prefix}_encoder_{self.seq_tile}x{self.emb_tile}.o",
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
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
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
            "W_O",
            self.embed_sz * self.embed_sz,
            static_data=static_w_o_proj,
        )
        self.add_buffer(
            "QKV",
            3 * self.embed_sz * self.seq_len,
        )
        ln1_stage_rows = 0
        if self.ln1_staging_design == "ddr":
            ln1_stage_rows = (
                self.ffn_intermediate_size // self.emb_tile
            ) * self.seq_tile
        self.add_buffer(
            "OR",
            (2 * self.seq_len + ln1_stage_rows) * self.embed_sz,
        )
        self.add_buffer(
            "O",
            self.embed_sz * self.seq_len,
        )
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
        self.add_to_runlist("encoder_pipeline", "W_O", "QKV", "OR", "B_Up", "B_Down")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor = None,
        w_o: torch.Tensor = None,
        b_up: torch.Tensor = None,
        b_down: torch.Tensor = None,
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
                f"AIEEncoderPipeline: expected B_Up shape {expected_b_up} and B_Down "
                f"shape {expected_b_down}, got {tuple(b_up_shape)} and {tuple(b_down_shape)}"
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
        r: torch.Tensor = None,
        w_o: torch.Tensor = None,
        b_up: torch.Tensor = None,
        b_down: torch.Tensor = None,
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
        if self.ln1_staging_design == "ddr":
            ln1_stage_rows = (
                self.ffn_intermediate_size // self.emb_tile
            ) * self.seq_tile
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
            w_o_np = torch_to_numpy(w_o)
            self.write_buffer("W_O", w_o_np)
        if b_up is not None:
            b_up_np = torch_to_numpy(b_up)
            self.write_buffer("B_Up", b_up_np)
        if b_down is not None:
            b_down_np = torch_to_numpy(b_down)
            self.write_buffer("B_Down", b_down_np)

        self.run_runlist()

        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        result = numpy_to_torch(o_np)
        return result


AIEEncoderPipelineMemtile = AIEEncoderPipeline
