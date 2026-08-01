# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16

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
from iron.common.utils import torch_to_numpy


class AIEAddAndNorm(AIEOperatorBase):
    """AIE-accelerated ADD & LAYER NORM operator"""

    @staticmethod
    def _weight_signature(weights):
        if isinstance(weights, np.ndarray):
            weight_np = np.ascontiguousarray(weights)
        else:
            weight_np = np.ascontiguousarray(torch_to_numpy(weights))
        return hashlib.sha1(weight_np.view(np.uint8)).hexdigest()[:12]

    def __init__(
        self,
        size,
        num_aie_columns=None,
        tile_size=None,
        weights=None,
        trace_size=0,
        context=None,
        skip_add_to_list=False,
    ):
        max_multiple = num_aie_columns * tile_size
        padded_size = ((size + max_multiple - 1) // max_multiple) * max_multiple
        self.orig_size = size
        self.size = padded_size
        self.tile_size = tile_size
        self.trace_size = trace_size
        self.num_aie_columns = num_aie_columns

        total_shimdma_channels = self.num_aie_columns
        assert total_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        self.weight = weights

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="weighted_layer_norm_"):
        # Compilation artifacts
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.num_aie_columns}c_{self.size}_{self.tile_size}t"
        )
        weight_signature = self._weight_signature(self.weight)
        file_name_base = f"{file_name_base}_{weight_signature}"

        # Save the weight weights to a npy file so that the design.py can load it at compile time
        weight_file_name = self.context.build_dir / f"{file_name_base}_weights.npy"
        np.save(weight_file_name, torch_to_numpy(self.weight))

        kernel_archive = f"{file_name_base}_layer_norm_archive.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_weighted_layer_norm",
            callback_args=[
                self.context.device_manager.device_type,
                self.size,
                self.num_aie_columns,
                self.tile_size,
                weight_file_name,
                kernel_archive,
                0,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    kernel_archive,
                    depends=[
                        KernelObjectArtifact.new(
                            f"{file_name_base}_layer_norm.o",
                            depends=[
                                SourceArtifact.new(
                                    self.context.base_dir
                                    / "aie_kernels"
                                    / "aie2p"
                                    / "layer_norm.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"{file_name_base}_mul.o",
                            depends=[
                                SourceArtifact.new(
                                    self.context.base_dir
                                    / "aie_kernels"
                                    / "generic"
                                    / "mul.cc"
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        return (xclbin_artifact, insts_artifact)

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("input1", self.size)
        self.add_buffer("input2", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "addnorm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("addnorm", "input1", "input2", "output")

    def forward(self, x, y):
        """Forward pass for element-wise addition"""
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
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEAddAndNorm: incompatible tensor shape(s)"
            )

        # Always flatten to [batch, orig_size]
        original_shape = x.shape
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        pad_len = self.size - x_flat.numel()
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))
            y_flat = torch.nn.functional.pad(y_flat, (0, pad_len))

        out = self._execute_aie_operation(x_flat, y_flat)

        # Remove padding if added
        numel = np.prod(original_shape)
        if pad_len > 0:
            out = out[:numel]
        # Restore original shape
        out = out.reshape(*original_shape)

        return out

    def _execute_aie_operation(self, x, y):
        """Execute element-wise addition operation on AIE hardware"""
        # x, y are [batch, size]
        batch = x.shape[0] if x.dim() > 1 else 1

        # Flatten inputs for AIE processing
        x_flat = x.view(-1)
        y_flat = y.view(-1)

        # Verify size matches expected
        if len(x_flat) != self.size or len(y_flat) != self.size:
            raise AIEOperatorConstraintError(
                f"Input size x={len(x_flat)}, y={len(y_flat)} doesn't match configured size {self.size}"
            )

        self.write_buffer("input1", x_flat)
        self.write_buffer("input2", y_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=x_flat.shape, dtype=bfloat16)

        return result
