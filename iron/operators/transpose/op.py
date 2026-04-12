# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIETranspose(AIEOperatorBase):
    """AIE-accelerated transpose operator"""

    def __init__(
        self,
        M,
        N,
        num_aie_columns,
        num_channels,
        m,
        n,
        s,
        context=None,
        skip_add_to_list=False,
    ):
        self.M = M
        self.N = N
        self.m = m
        self.n = n
        self.s = s
        self.size = M * N
        self.tile_size = m * n

        self.num_columns = num_aie_columns
        self.num_channels = num_channels

        if self.M % self.num_channels != 0:
            raise AIEOperatorConstraintError(
                "AIETranspose: M must be divisible by num_channels "
                f"(got M={self.M}, num_channels={self.num_channels})"
            )
        if self.N % self.num_columns != 0:
            raise AIEOperatorConstraintError(
                "AIETranspose: N must be divisible by num_aie_columns "
                f"(got N={self.N}, num_aie_columns={self.num_columns})"
            )

        channel_rows = self.M // self.num_channels
        column_cols = self.N // self.num_columns
        if channel_rows % self.m != 0:
            raise AIEOperatorConstraintError(
                "AIETranspose: per-channel row partition must be divisible by m "
                f"(got M/num_channels={channel_rows}, m={self.m})"
            )
        if column_cols % self.n != 0:
            raise AIEOperatorConstraintError(
                "AIETranspose: per-column column partition must be divisible by n "
                f"(got N/num_aie_columns={column_cols}, n={self.n})"
            )

        total_shimdma_channels = self.num_columns * self.num_channels
        if 1 > 1:
            total_shimdma_channels *= 1
        assert total_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="transpose_"):
        operator_dir = Path(__file__).parent
        file_name_base = f"{prefix}{self.num_columns}c_{self.num_channels}ch_{self.M}x{self.N}_{self.m}x{self.n}_{self.s}s"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="shuffle_transpose",
            callback_args=[
                self.context.device_manager.device_type,
                self.M,
                self.N,
                self.num_columns,
                self.num_channels,
                0,
                self.m,
                self.n,
                self.s,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    f"transpose_{self.m}x{self.n}.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "transpose.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_m={self.m}",
                        f"-DDIM_n={self.n}",
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("input", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "transpose",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("transpose", "input", "output")

    def forward(self, x):
        if x.numel() > self.size:
            raise AIEOperatorConstraintError(
                "AIETranspose: input too large for configured size"
            )

        original_shape = x.shape
        x_flat = x.reshape(-1)

        pad_len = self.size - x_flat.numel()
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

        self.write_buffer("input", x_flat)
        self.write_buffer("output", np.zeros(self.size, dtype=bfloat16))
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=(self.size,), dtype=bfloat16)

        if pad_len > 0:
            result = result[: x_flat.numel() - pad_len]

        return result.reshape(*original_shape)
