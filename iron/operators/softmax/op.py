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

        self.num_channels = num_channels
        self.num_columns = num_aie_columns

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="softmax_"):
        # Compilation artifacts
        operator_dir = Path(__file__).parent
        file_name_base = f"{prefix}{self.num_columns}c_{self.num_channels}ch_{self.size}_{self.cols}t"

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
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    f"softmax.o",
                    extra_flags=[f"-DSM_VEC_LEN={self.cols}"],
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
        applicable = (
            x.shape[-1] * x.shape[-2] == self.size
            and x.shape[-1] == self.cols
            and x.shape[-1] % 16 == 0
            and x.shape[-2] % 16 == 0
        )
        if not applicable:
            raise AIEOperatorConstraintError("AIESoftmax: incompatible tensor shape(s)")

        return self._execute_aie_operation(x)

    def _execute_aie_operation(self, x):
        original_shape = x.shape

        # Reshape for processing
        # Split x into a list of H tensors of size [S_q, S_kv]
        heads = x.shape[1]
        x_list = [x[0, h, :, :] for h in range(heads)]
        results = []
        for i in range(heads):
            x_iter = x_list[i]
            input_size = x_iter.nbytes
            self.write_buffer("in", x_iter)
            test_pattern = np.zeros(len(x_iter), dtype=bfloat16)
            self.write_buffer("output", test_pattern)
            self.run_runlist()
            result = self.read_buffer_as_torch(
                "output", shape=x_list[i].shape, dtype=bfloat16
            )
            results.append(result)

        result = torch.stack(results, dim=0).unsqueeze(
            0
        )  # Shape: (1, heads, S_q, S_kv)

        return result
