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


class AIEElementwiseMul(AIEOperatorBase):
    """AIE-accelerated element-wise multiplication"""

    def __init__(
        self,
        size,
        num_aie_columns,
        num_channels,
        tile_size,
        scalar_broadcast=None,
        trace_size=0,
        context=None,
    ):
        max_multiple = num_aie_columns * tile_size
        padded_size = ((size + max_multiple - 1) // max_multiple) * max_multiple
        self.orig_size = size
        self.size = padded_size
        self.tile_size = tile_size
        self.num_aie_columns = num_aie_columns
        self.num_channels = num_channels
        self.scalar_broadcast = scalar_broadcast
        self.trace_size = trace_size

        total_shimdma_channels = self.num_aie_columns * self.num_channels
        assert total_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="eltwise_mul_"):
        operator_dir = Path(__file__).parent
        if self.scalar_broadcast is None:
            file_name_base = f"{prefix}{self.num_aie_columns}c_{self.num_channels}ch_{self.size}_{self.tile_size}t"
            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_base}.mlir",
                import_path=operator_dir / "design.py",
                callback_fn="my_eltwise_mul",
                callback_args=[
                    self.context.device_manager.device_type,
                    self.size,
                    self.num_aie_columns,
                    self.tile_size,
                    self.trace_size,
                ],
            )
        else:
            file_name_base = f"{prefix}{self.num_aie_columns}c_{self.num_channels}ch_{self.size}_{self.scalar_broadcast}sb_{self.tile_size}t"
            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_base}.mlir",
                import_path=operator_dir / "design.py",
                callback_fn="my_eltwise_mul_broadcast_scalar",
                callback_args=[
                    self.context.device_manager.device_type,
                    self.size,
                    self.num_aie_columns,
                    self.num_channels,
                    self.tile_size,
                    self.scalar_broadcast,
                    self.trace_size,
                ],
            )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    f"mul.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                        )
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

        mlir_artifact = xclbin_artifact.depends[0]
        mlir_artifact.callback_args[0] = self.context.device_manager.device_type

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact

        artifacts = [xclbin_artifact, insts_artifact]
        self.add_artifacts(artifacts)

    def set_up_runtime(self):
        self.add_buffer("input1", self.size)
        if self.scalar_broadcast is None:
            self.add_buffer("input2", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "eltwise_mul",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        if self.scalar_broadcast is None:
            self.add_to_runlist("eltwise_mul", "input1", "input2", "output")
        else:
            self.add_to_runlist("eltwise_mul", "input1", "output")

    def forward(self, x, y=None):
        """Forward pass for element-wise multiplication"""
        applicable = (
            len(x.shape) >= 1
            and len(y.shape) >= 1
            and x.shape[-1] <= self.size
            and y.shape[-1] <= self.size
            and x.numel() <= self.size
            and y.numel() <= self.size
            and x.numel() == y.numel()
            and x.shape == y.shape
        )

        # Always flatten to [batch, orig_size]
        original_shape = x.shape
        batch = x.shape[0] if x.dim() > 1 else 1
        x_flat = x.reshape(batch, -1)
        if y is not None:
            y_flat = y.reshape(batch, -1)
        else:
            y_flat = None

        pad_len = self.size - x_flat.shape[1]
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))
            if y_flat is not None:
                y_flat = torch.nn.functional.pad(y_flat, (0, pad_len))

        out = self._execute_aie_operation(x_flat, y_flat)

        # Remove padding if added
        numel = np.prod(original_shape)
        if pad_len > 0:
            out = out.reshape(-1)[..., :numel]
        # Restore original shape
        out = out.reshape(*original_shape)

        return out

    def _execute_aie_operation(self, x, y=None):
        """Execute element-wise multiplication operation on AIE hardware"""
        # x, y are [batch, size]
        batch = x.shape[0] if x.dim() > 1 else 1

        # Flatten inputs for AIE processing
        x_flat = x.view(-1)
        # Verify size matches expected
        if len(x_flat) != self.size:
            raise AIEOperatorConstraintError(
                f"Input size x={len(x_flat)} doesn't match configured size {self.size}"
            )
        if y is not None:
            y_flat = y.view(-1)
            if len(y_flat) != self.size:
                raise AIEOperatorConstraintError(
                    f"Input size y={len(y_flat)} doesn't match configured size {self.size}"
                )
            self.write_buffer("input2", y_flat)

        self.write_buffer("input1", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=x_flat.shape, dtype=bfloat16)

        return result
