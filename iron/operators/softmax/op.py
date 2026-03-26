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
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIESoftmax(AIEOperatorBase):
    LONG_ROW_CHUNK_SIZE = 8192

    @staticmethod
    def _resolve_kernel_vector_width(cols: int) -> int:
        for width in (128, 64, 32, 16):
            if cols >= width and cols % width == 0:
                return width
        raise ValueError(
            f"AIESoftmax requires cols to be divisible by one of "
            f"(16, 32, 64, 128); got cols={cols}"
        )

    def __init__(
        self,
        rows: int,
        cols: int,
        num_aie_columns=1,
        num_channels=1,
        context=None,
        skip_add_to_list: bool = False,
    ):
        self.size = rows * cols
        self.rows = rows
        self.cols = cols
        self.kernel_vec_len = self._resolve_kernel_vector_width(cols)
        self.row_chunk_size = self._resolve_row_chunk_size(cols)

        self.num_channels = num_channels
        self.num_columns = num_aie_columns

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    @classmethod
    def _resolve_row_chunk_size(cls, cols: int) -> int:
        if cols <= cls.LONG_ROW_CHUNK_SIZE:
            return cols
        if cols % cls.LONG_ROW_CHUNK_SIZE != 0:
            raise ValueError(
                "AIESoftmax long-row path requires cols divisible by "
                f"{cls.LONG_ROW_CHUNK_SIZE}; got cols={cols}"
            )
        return cls.LONG_ROW_CHUNK_SIZE

    def get_artifacts(self, prefix="softmax_"):
        # Compilation artifacts
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.num_columns}c_{self.num_channels}ch_"
            f"{self.size}_{self.cols}t_rs{self.row_chunk_size}_sm{self.kernel_vec_len}"
        )
        kernel_object_name = f"softmax_sm{self.kernel_vec_len}.o"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="softmax",
            callback_args=[
                self.context.device_manager.device_type,
                self.rows * self.cols,
                self.num_columns,
                self.num_channels,
                0,
                self.cols,
                self.row_chunk_size,
                kernel_object_name,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    kernel_object_name,
                    extra_flags=[f"-DSM_VEC_LEN={self.kernel_vec_len}"],
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "aie2p"
                            / "softmax.cc"
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"gemm_{file_name_base}.bin", depends=[mlir_artifact]
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        # Runlist setup
        self.add_buffer("in", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "softmax",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("softmax", "in", "output")

    def forward(self, x):
        x_2d = x.reshape(-1, self.cols)
        applicable = (
            x_2d.shape[0] * x_2d.shape[1] == self.size
            and x_2d.shape[1] == self.cols
            and x_2d.shape[1] % 16 == 0
            and x_2d.shape[0] % 16 == 0
        )
        if not applicable:
            raise AIEOperatorConstraintError("AIESoftmax: incompatible tensor shape(s)")

        return self._execute_aie_operation(x)

    def _execute_aie_operation(self, x):
        original_shape = x.shape
        x_2d = x.reshape(-1, self.cols)
        self.write_buffer("in", x_2d)
        test_pattern = np.zeros(self.size, dtype=bfloat16)
        self.write_buffer("output", test_pattern)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output",
            shape=x_2d.shape,
            dtype=bfloat16,
        )
        return result.reshape(original_shape)
