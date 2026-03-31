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
from iron.operators.mha_out_proj.topology import mha_out_proj_design


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
        return tensor.contiguous()
    if tensor.ndim == 3 and tensor.shape == (num_heads, seq_len, d):
        return tensor.contiguous()
    raise AIEOperatorConstraintError("AIEMHAOutProj: incompatible tensor shape(s)")


def _flatten_head_major_qkv(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    embed_sz: int,
) -> np.ndarray:
    if tensor.ndim == 2:
        return torch_to_numpy(tensor)
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


def _is_canonical_qkv_tensor(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    num_heads: int,
    d: int,
    embed_sz: int,
) -> bool:
    return tensor.shape in (
        (seq_len, embed_sz),
        (num_heads, seq_len, d),
    )


def _canonicalize_residual_output(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    embed_sz: int,
) -> torch.Tensor:
    if tensor.ndim == 2 and tensor.shape == (seq_len, embed_sz):
        return tensor.contiguous()
    if tensor.ndim == 3 and tensor.shape == (1, seq_len, embed_sz):
        return tensor.view(seq_len, embed_sz).contiguous()
    raise AIEOperatorConstraintError(
        f"AIEMHAOutProj: expected residual shape {(seq_len, embed_sz)}"
    )


def _pack_block3_residual_output(
    residual_np: np.ndarray,
    *,
    seq_len: int,
    packed_rows: int,
    embed_sz: int,
    q_seq_tile: int,
    emb_tile: int,
    parallel_seq: int,
) -> np.ndarray:
    if seq_len % q_seq_tile != 0:
        raise AIEOperatorConstraintError(
            "AIEMHAOutProj: packed Block 3 handoff requires seq_len divisible by q_seq_tile"
        )
    if embed_sz % emb_tile != 0:
        raise AIEOperatorConstraintError(
            "AIEMHAOutProj: packed Block 3 handoff requires embed_sz divisible by emb_tile"
        )
    num_q_seq_blocks = seq_len // q_seq_tile
    num_packed_q_seq_blocks = packed_rows // q_seq_tile
    if packed_rows < seq_len or packed_rows % q_seq_tile != 0:
        raise AIEOperatorConstraintError(
            "AIEMHAOutProj: packed Block 3 handoff requires packed_rows to be a "
            "q_seq_tile-aligned row capacity that covers seq_len"
        )
    if num_packed_q_seq_blocks % parallel_seq != 0:
        raise AIEOperatorConstraintError(
            "AIEMHAOutProj: packed Block 3 handoff requires "
            "(packed_rows / q_seq_tile) divisible by Block 3 parallel_seq"
        )

    num_col_groups = embed_sz // emb_tile
    tile_elems = q_seq_tile * emb_tile
    packed_np = np.zeros((2 * packed_rows * embed_sz,), dtype=residual_np.dtype)

    for q_block_idx in range(num_q_seq_blocks):
        row_start = q_block_idx * q_seq_tile
        for col_group in range(num_col_groups):
            col_start = col_group * emb_tile
            tile_index = q_block_idx * num_col_groups + col_group
            tile_offset = tile_index * (2 * tile_elems)
            packed_np[tile_offset + tile_elems : tile_offset + (2 * tile_elems)] = (
                residual_np[
                    row_start : row_start + q_seq_tile,
                    col_start : col_start + emb_tile,
                ].reshape(tile_elems)
            )

    return packed_np


def _unpack_block3_attention_output(
    packed_np: np.ndarray,
    *,
    seq_len: int,
    packed_rows: int,
    embed_sz: int,
    q_seq_tile: int,
    emb_tile: int,
    parallel_seq: int,
) -> np.ndarray:
    num_q_seq_blocks = seq_len // q_seq_tile
    num_col_groups = embed_sz // emb_tile
    tile_elems = q_seq_tile * emb_tile
    output_np = np.zeros((seq_len, embed_sz), dtype=packed_np.dtype)

    for q_block_idx in range(num_q_seq_blocks):
        row_start = q_block_idx * q_seq_tile
        for col_group in range(num_col_groups):
            col_start = col_group * emb_tile
            tile_index = q_block_idx * num_col_groups + col_group
            tile_offset = tile_index * (2 * tile_elems)
            output_np[
                row_start : row_start + q_seq_tile,
                col_start : col_start + emb_tile,
            ] = packed_np[tile_offset : tile_offset + tile_elems].reshape(
                q_seq_tile, emb_tile
            )

    return output_np


class AIEMHAOutProj(AIEOperatorBase):
    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d: int,
        parallel_seq: int = 1,
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
        packed_output_parallel_seq: int | None = None,
        packed_output_rows: int | None = None,
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
            parallel_seq = int(config["parallel_seq"])
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
                int(candidate["parallel_seq"]) == parallel_seq
                and int(candidate["q_seq_tile"]) == q_seq_tile
                and int(candidate["kv_seq_tile"]) == kv_seq_tile
                and int(candidate["emb_tile"]) == emb_tile
                and int(candidate["parallel_heads"]) == parallel_heads
                and int(candidate["o_proj_acc_depth"]) == o_proj_acc_depth
            ):
                config = candidate

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d = d
        self.parallel_seq = parallel_seq
        self.q_seq_tile = q_seq_tile
        self.kv_seq_tile = kv_seq_tile
        self.emb_tile = emb_tile
        self.parallel_heads = parallel_heads
        self.o_proj_acc_depth = o_proj_acc_depth
        self.packed_output_parallel_seq = packed_output_parallel_seq
        self.packed_output_rows = packed_output_rows
        self.topology_id = (
            str(config["topology_id"])
            if config is not None
            else (
                f"q{q_seq_tile}_kv{kv_seq_tile}_e{emb_tile}"
                f"_ps{parallel_seq}_ph{parallel_heads}_acc{o_proj_acc_depth}"
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
        if self.packed_output_parallel_seq is not None:
            if self.seq_len % self.q_seq_tile != 0:
                raise AIEOperatorConstraintError(
                    "AIEMHAOutProj: packed Block 3 handoff requires seq_len divisible by q_seq_tile"
                )
            if self.packed_output_rows is None:
                raise AIEOperatorConstraintError(
                    "AIEMHAOutProj: packed output mode requires packed_output_rows"
                )
            if (
                self.packed_output_rows < self.seq_len
                or self.packed_output_rows % self.q_seq_tile != 0
            ):
                raise AIEOperatorConstraintError(
                    "AIEMHAOutProj: packed Block 3 handoff requires packed_output_rows "
                    "to be q_seq_tile-aligned and cover seq_len"
                )
            if (
                self.packed_output_rows // self.q_seq_tile
            ) % self.packed_output_parallel_seq != 0:
                raise AIEOperatorConstraintError(
                    "AIEMHAOutProj: packed Block 3 handoff requires "
                    "(packed_output_rows / q_seq_tile) divisible by "
                    "packed_output_parallel_seq"
                )

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

        file_name_base = (
            f"mha_o_proj_{self.num_heads}h_{self.seq_len}s_{self.d}d_"
            f"{self.q_seq_tile}qseqtile_{self.kv_seq_tile}kvseqtile_{self.emb_tile}e_"
            f"{self.parallel_seq}ps_{self.parallel_heads}ph_{self.o_proj_acc_depth}acc"
        )
        if self.packed_output_parallel_seq is not None:
            file_name_base += (
                f"_packps{self.packed_output_parallel_seq}"
                f"_packrows{self.packed_output_rows}"
            )

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
                "parallel_seq": self.parallel_seq,
                "q_seq_tile": self.q_seq_tile,
                "kv_seq_tile": self.kv_seq_tile,
                "emb_tile": self.emb_tile,
                "o_proj_acc_depth": self.o_proj_acc_depth,
                "parallel_heads": self.parallel_heads,
                "packed_output_parallel_seq": self.packed_output_parallel_seq,
                "packed_output_rows": self.packed_output_rows,
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
            (
                2 * self.embed_sz * self.packed_output_rows
                if self.packed_output_parallel_seq is not None
                else self.embed_sz * self.seq_len
            ),
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
            _is_canonical_qkv_tensor(
                q,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                d=self.d,
                embed_sz=self.embed_sz,
            )
            and _is_canonical_qkv_tensor(
                k,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                d=self.d,
                embed_sz=self.embed_sz,
            )
            and _is_canonical_qkv_tensor(
                v,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                d=self.d,
                embed_sz=self.embed_sz,
            )
            and self.seq_len % 64 == 0
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEMHAOutProj: incompatible tensor shape(s)"
            )

        ret = self._execute_aie_operation(q, k, v, w_o)
        return ret

    def forward_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        w_o: torch.Tensor = None,
    ) -> torch.Tensor:
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
        residual = _canonicalize_residual_output(
            residual,
            seq_len=self.seq_len,
            embed_sz=self.embed_sz,
        )
        if self.packed_output_parallel_seq is None:
            raise AIEOperatorConstraintError(
                "AIEMHAOutProj: packed output mode requires packed_output_parallel_seq"
            )
        return self._execute_aie_operation(q, k, v, w_o, residual=residual)

    def _execute_aie_operation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_o: torch.Tensor = None,
        residual: torch.Tensor | None = None,
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
        if residual is not None:
            residual_np = torch_to_numpy(residual)
            packed_output_np = _pack_block3_residual_output(
                residual_np,
                seq_len=self.seq_len,
                packed_rows=self.packed_output_rows,
                embed_sz=self.embed_sz,
                q_seq_tile=self.q_seq_tile,
                emb_tile=self.emb_tile,
                parallel_seq=self.packed_output_parallel_seq,
            )
            self.write_buffer("O", packed_output_np)

        # Execute
        self.run_runlist()

        if residual is not None:
            packed_o_np = self.read_buffer(
                "O",
                shape=(2 * self.packed_output_rows * self.embed_sz,),
                dtype=bfloat16,
            )
            return numpy_to_torch(packed_o_np)

        if self.packed_output_parallel_seq is not None:
            packed_o_np = self.read_buffer(
                "O",
                shape=(2 * self.packed_output_rows * self.embed_sz,),
                dtype=bfloat16,
            )
            return numpy_to_torch(
                _unpack_block3_attention_output(
                    packed_o_np,
                    seq_len=self.seq_len,
                    packed_rows=self.packed_output_rows,
                    embed_sz=self.embed_sz,
                    q_seq_tile=self.q_seq_tile,
                    emb_tile=self.emb_tile,
                    parallel_seq=self.packed_output_parallel_seq,
                )
            )

        o_np = self.read_buffer(
            "O", shape=(self.seq_len, self.embed_sz), dtype=bfloat16
        )
        return numpy_to_torch(o_np)
