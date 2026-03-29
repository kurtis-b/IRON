# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch

from iron.common import (
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from iron.operators.qkv_proj.design import qkv_proj_design
from iron.operators.gemm.op import AIEGEMM


class _ProjectionWeightView:
    def __init__(self, block: "AIEQKVProj", projection_index: int) -> None:
        self._block = block
        self._projection_index = projection_index

    @property
    def weight(self) -> torch.Tensor:
        return self._block._projection_weight(self._projection_index)

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self._block._set_projection_weight(self._projection_index, value)

    @property
    def insts_artifact(self):
        return self._block.qkv_proj.insts_artifact

    @property
    def runtime_xclbin_artifact(self):
        return self._block.qkv_proj.runtime_xclbin_artifact

    @property
    def xclbin_artifact(self):
        return self._block.qkv_proj.xclbin_artifact


class _AIEFusedQKVGemm(AIEGEMM):
    """Local fused-QKV GEMM that lowers through qkv_proj/design.py."""

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
            f"qkv_proj_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
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

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_total_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_qkv_proj",
            callback_kwargs={
                "dev": device_str,
                "seq_len": M,
                "hidden_size": K,
                "combined_hidden_size": N,
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "num_aie_columns": num_aie_columns,
                "dtype_in_str": dtype_in,
                "dtype_out_str": dtype_out,
                "use_scalar": use_scalar,
                "emulate_bf16_mmul_with_bfp16": emulate_bf16_mmul_with_bfp16,
                "prio_accuracy": prio_accuracy,
                "trace_size": 0,
                "archive": kernel_archive,
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
                                f"qkv_proj_{tile_m}x{tile_k}x{tile_n}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
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

    def get_artifacts(self, prefix="qkv_proj_"):
        return self._build_mlir_artifact(prefix, self.M, self.K, self.N)

    def get_insts_artifact(
        self, prefix="qkv_proj_", xclbin_input=None, kernel_name=None
    ):
        return super().get_insts_artifact(
            prefix=prefix,
            xclbin_input=xclbin_input,
            kernel_name=kernel_name,
        )

    def get_runtime_xclbin_artifact(self, prefix="qkv_proj_runtime_"):
        return super().get_runtime_xclbin_artifact(prefix=prefix)


class AIEQKVProj:
    """Block 1 Q/K/V projection backed by a local fused GEMM design."""

    @staticmethod
    def _to_head_major(
        tensor: torch.Tensor,
        *,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        head_dim = hidden_size // num_heads
        return tensor.view(seq_len, num_heads, head_dim).permute(1, 0, 2).contiguous()

    @staticmethod
    def _combined_hidden_size(hidden_size: int) -> int:
        return hidden_size * 3

    def __init__(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        context,
        num_heads: int,
        topology_id: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context = context
        topology = qkv_proj_design(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            topology_id=topology_id,
        )
        self.topology_id = str(topology["topology_id"])
        self.topology_family = str(topology["topology_family"])
        self.parallel_seq = int(topology["parallel_seq"])
        self.parallel_heads = int(topology["parallel_heads"])
        self.parallel_head_dim = int(topology["parallel_head_dim"])
        self.tile_m = int(topology["tile_m"])
        self.tile_k = int(topology["tile_k"])
        self.tile_n = int(topology["tile_n"])
        gemm_common = {
            "tile_m": self.tile_m,
            "tile_k": self.tile_k,
            "tile_n": self.tile_n,
            "num_aie_columns": int(topology["num_aie_columns"]),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
            "use_static_weight": True,
            "force_batched_design": True,
            "context": context,
        }
        self.qkv_proj = _AIEFusedQKVGemm(
            M=seq_len,
            K=hidden_size,
            N=self._combined_hidden_size(hidden_size),
            **gemm_common,
        )
        self.q_proj = _ProjectionWeightView(self, 0)
        self.k_proj = _ProjectionWeightView(self, 1)
        self.v_proj = _ProjectionWeightView(self, 2)

    def _projection_slice(self, projection_index: int) -> slice:
        start = projection_index * self.hidden_size
        end = start + self.hidden_size
        return slice(start, end)

    def _projection_weight(self, projection_index: int) -> torch.Tensor:
        return self.qkv_proj.weight[self._projection_slice(projection_index)]

    def _set_projection_weight(
        self, projection_index: int, value: torch.Tensor
    ) -> None:
        expected_shape = (self.hidden_size, self.hidden_size)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"AIEQKVProj: expected projection weight shape {expected_shape}, got {tuple(value.shape)}"
            )
        self.qkv_proj.weight[self._projection_slice(projection_index)] = (
            value.contiguous()
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(hidden_states).contiguous()
        q_matrix, k_matrix, v_matrix = torch.split(qkv, self.hidden_size, dim=-1)

        q = self._to_head_major(
            q_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        k = self._to_head_major(
            k_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        v = self._to_head_major(
            v_matrix,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        return q, k, v
