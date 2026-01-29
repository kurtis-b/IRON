# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

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
from operators.common.utils import torch_to_numpy


class AIEAddAndNorm(AIEOperatorBase):
    """AIE-accelerated ADD & LAYER NORM operator"""

    def __init__(
        self,
        M=512,
        K=768,
        m=4,
        k=192,
        s=8,
        num_aie_columns=None,
        weights=None,
        trace_size=0,
        context=None,
        skip_add_to_list=False,
    ):
        self.M = M
        self.K = K
        self.m = m
        self.k = k
        self.s = s
        self.trace_size = trace_size
        self.num_aie_columns = num_aie_columns

        used_shimdma_channels = self.num_aie_columns
        assert used_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        self.weight = weights

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="weighted_layer_norm_alt_"):
        # Compilation artifacts
        operator_dir = Path(__file__).parent
        file_name_base = f"{prefix}{self.num_aie_columns}c_{self.M}x{self.K}_{self.m}x{self.k}x{self.s}"

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
                self.M,
                self.K,
                self.m,
                self.k,
                self.s,
                self.num_aie_columns,
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
                            f"{file_name_base}_fused_layer_norm.o",
                            depends=[
                                SourceArtifact.new(
                                    self.context.base_dir
                                    / "aie_kernels"
                                    / "aie2p"
                                    / "encoder.cc"
                                )
                            ],
                            extra_flags=[
                                "-DADD_NORM_LAYER",
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
        self.add_buffer("input1", self.M * self.K)
        self.add_buffer("input2", self.M * self.K)
        self.add_buffer("output", self.M * self.K)
        self.add_kernel(
            "add_and_norm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("add_and_norm", "input1", "input2", "output")

    def forward(self, x, y):
        """Forward pass for element-wise addition"""
        applicable = (
            len(x.shape) >= 1
            and len(y.shape) >= 1
            and x.shape[-1] <= self.K
            and y.shape[-1] <= self.K
            and x.numel() <= self.M * self.K
            and y.numel() <= self.M * self.K
            and x.numel() == y.numel()
            and x.shape == y.shape
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEAddAndNorm: incompatible tensor shape(s)"
            )

        # Always flatten to [batch, orig_size]
        original_shape = x.shape
        batch = x.shape[0] if x.dim() > 1 else 1
        x_flat = x.reshape(batch, -1)
        y_flat = y.reshape(batch, -1)

        pad_len = self.M * self.K - x_flat.shape[1]
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))
            y_flat = torch.nn.functional.pad(y_flat, (0, pad_len))

        out = self._execute_aie_operation(x_flat, y_flat)

        # Remove padding if added
        numel = np.prod(original_shape)
        if pad_len > 0:
            out = out.reshape(-1)[..., :numel]
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
        if len(x_flat) != self.M * self.K or len(y_flat) != self.M * self.K:
            raise AIEOperatorConstraintError(
                f"Input size x={len(x_flat)}, y={len(y_flat)} doesn't match configured size {self.M * self.K}"
            )

        self.write_buffer("input1", x_flat)
        self.write_buffer("input2", y_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=x_flat.shape, dtype=bfloat16)

        return result
