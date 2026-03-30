# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from iron.operators.qkv_proj.topology import qkv_proj_design


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
        return self._block.insts_artifact

    @property
    def runtime_xclbin_artifact(self):
        return self._block.runtime_xclbin_artifact

    @property
    def xclbin_artifact(self):
        return self._block.xclbin_artifact


class AIEQKVProj(AIEOperatorBase):
    """Block 1 fused Q/K/V projection."""

    @staticmethod
    def _pack_parallel_head_dim_layout(
        tensor: torch.Tensor,
        *,
        hidden_size: int,
        num_heads: int,
        parallel_heads: int,
        parallel_head_dim: int,
    ) -> torch.Tensor:
        if parallel_head_dim == 1:
            return tensor.contiguous()
        head_dim = hidden_size // num_heads
        heads_per_group = num_heads // parallel_heads
        head_dim_chunk = head_dim // parallel_head_dim
        view = tensor.view(
            tensor.shape[0],
            3,
            parallel_heads,
            heads_per_group,
            parallel_head_dim,
            head_dim_chunk,
        )
        return (
            view.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(tensor.shape[0], 3 * hidden_size)
        )

    @staticmethod
    def _unpack_parallel_head_dim_layout(
        tensor: torch.Tensor,
        *,
        hidden_size: int,
        num_heads: int,
        parallel_heads: int,
        parallel_head_dim: int,
    ) -> torch.Tensor:
        if parallel_head_dim == 1:
            return tensor.contiguous()
        head_dim = hidden_size // num_heads
        heads_per_group = num_heads // parallel_heads
        head_dim_chunk = head_dim // parallel_head_dim
        view = tensor.view(
            tensor.shape[0],
            3,
            parallel_heads,
            parallel_head_dim,
            heads_per_group,
            head_dim_chunk,
        )
        return (
            view.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(tensor.shape[0], 3 * hidden_size)
        )

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
        self.num_aie_columns = int(topology["num_aie_columns"])

        self.M, self.K, self.N = self._get_padded_dims(
            seq_len,
            hidden_size,
            self._combined_hidden_size(hidden_size),
        )
        self.weight = torch.zeros(
            (self._combined_hidden_size(hidden_size), hidden_size),
            dtype=torch.bfloat16,
        )

        self.xclbin_artifact = None
        self.insts_artifact = None
        self.runtime_xclbin_artifact = None
        self.runtime_kernel_name = None

        self.q_proj = _ProjectionWeightView(self, 0)
        self.k_proj = _ProjectionWeightView(self, 1)
        self.v_proj = _ProjectionWeightView(self, 2)

        AIEOperatorBase.__init__(self, context=context)

    def _get_padded_dims(self, M: int, K: int, N: int) -> tuple[int, int, int]:
        num_aie_rows = self.parallel_seq
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns
        M_padded = ((M + min_M - 1) // min_M) * min_M
        K_padded = ((K + min_K - 1) // min_K) * min_K
        N_padded = ((N + min_N - 1) // min_N) * min_N
        return M_padded, K_padded, N_padded

    def _get_artifact_name_base(self, prefix: str, M: int, K: int, N: int) -> str:
        return (
            f"{prefix}{M}x{K}x{N}_{self.num_aie_columns}_{self.tile_m}x{self.tile_k}x{self.tile_n}"
            f"_ps{self.parallel_seq}_ph{self.parallel_heads}_pd{self.parallel_head_dim}"
            "_0_0_bf16_bf16_sc0_acc0_embf161_round1_batchA1d0_batchB1d0_batchC1d0"
        )

    def _build_mlir_artifact(self, prefix: str, M: int, K: int, N: int):
        operator_dir = Path(__file__).parent
        base_dir = self.context.base_dir
        device_str = self.context.device_manager.device_str()

        file_name_total_base = self._get_artifact_name_base(prefix, M, K, N)
        kernel_archive = (
            f"qkv_proj_{self.tile_m}x{self.tile_k}x{self.tile_n}_0_0_bf16_bf16"
            "_sc0_acc0_embf161_round1.a"
        )
        kernel_flags = [
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
            "-Dbf16_bf16_ONLY",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_total_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="fused_qkv_proj",
            callback_kwargs={
                "dev": device_str,
                "seq_len": M,
                "hidden_size": K,
                "combined_hidden_size": N,
                "tile_m": self.tile_m,
                "tile_k": self.tile_k,
                "tile_n": self.tile_n,
                "num_aie_columns": self.num_aie_columns,
                "parallel_seq": self.parallel_seq,
                "parallel_heads": self.parallel_heads,
                "parallel_head_dim": self.parallel_head_dim,
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
            f"{file_name_total_base}.xclbin",
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

    def get_artifacts(self, prefix: str = "qkv_proj_"):
        return self._build_mlir_artifact(prefix, self.M, self.K, self.N)

    def set_up_artifacts(self):
        if self.xclbin_artifact is None or self.insts_artifact is None:
            self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        artifacts = [self.xclbin_artifact, self.insts_artifact]
        if (
            self.runtime_xclbin_artifact is not None
            and self.runtime_xclbin_artifact is not self.xclbin_artifact
        ):
            artifacts.append(self.runtime_xclbin_artifact)
        self.add_artifacts(artifacts)

    def set_up_runtime(self):
        runtime_xclbin_artifact = self.runtime_xclbin_artifact or self.xclbin_artifact
        runtime_kernel_name = (
            self.runtime_kernel_name or runtime_xclbin_artifact.kernel_name
        )
        self.add_kernel(
            "qkv_proj",
            runtime_xclbin_artifact,
            runtime_kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("A", self.M * self.K)
        self.add_buffer(
            "B",
            self.K * self.N,
            static_data=torch_to_numpy(self._packed_weight_matrix()),
        )
        self.add_buffer("C", self.M * self.N)
        self.add_to_runlist("qkv_proj", "A", "B", "C")

    def _projection_slice(self, projection_index: int) -> slice:
        start = projection_index * self.hidden_size
        end = start + self.hidden_size
        return slice(start, end)

    def _projection_weight(self, projection_index: int) -> torch.Tensor:
        return self.weight[self._projection_slice(projection_index)]

    def _set_projection_weight(
        self, projection_index: int, value: torch.Tensor
    ) -> None:
        expected_shape = (self.hidden_size, self.hidden_size)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"AIEQKVProj: expected projection weight shape {expected_shape}, got {tuple(value.shape)}"
            )
        self.weight[self._projection_slice(projection_index)] = value.contiguous()

    def _pad_hidden_states(self, hidden_states: np.ndarray) -> np.ndarray:
        seq_len, hidden_size = hidden_states.shape
        padded = np.zeros((1, self.M, self.K), dtype=hidden_states.dtype)
        padded[0, :seq_len, :hidden_size] = hidden_states
        return padded

    def _packed_weight_matrix(self) -> torch.Tensor:
        packed = self.weight.T.contiguous()
        if self.parallel_head_dim > 1:
            packed = self._pack_parallel_head_dim_layout(
                packed,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                parallel_heads=self.parallel_heads,
                parallel_head_dim=self.parallel_head_dim,
            )
        return packed

    def _execute_fused_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_np = torch_to_numpy(hidden_states)
        seq_len, hidden_size = hidden_states_np.shape

        if hidden_size != self.hidden_size:
            raise AIEOperatorConstraintError(
                "AIEQKVProj: incompatible hidden dimension"
            )
        if seq_len > self.M:
            raise AIEOperatorConstraintError(
                "AIEQKVProj: sequence length exceeds compiled runtime shape"
            )

        self.write_buffer("A", self._pad_hidden_states(hidden_states_np))
        self.run_runlist()

        qkv = self.read_buffer(
            "C",
            shape=(1, self.M, self.N),
            dtype=bfloat16,
        )[0, :seq_len, : self._combined_hidden_size(self.hidden_size)]
        qkv_torch = numpy_to_torch(np.array(qkv, copy=True))
        if self.parallel_head_dim > 1:
            qkv_torch = self._unpack_parallel_head_dim_layout(
                qkv_torch,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                parallel_heads=self.parallel_heads,
                parallel_head_dim=self.parallel_head_dim,
            )
        return qkv_torch

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2:
            raise AIEOperatorConstraintError(
                "AIEQKVProj: expected a 2D hidden_states tensor"
            )
        if hidden_states.shape[0] != self.seq_len:
            raise AIEOperatorConstraintError(
                "AIEQKVProj: hidden_states sequence length must match the compiled operator"
            )

        qkv = self._execute_fused_qkv(hidden_states).contiguous()
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
