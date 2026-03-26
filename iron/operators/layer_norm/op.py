# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
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
from iron.common.utils import torch_to_numpy


class AIELayerNorm(AIEOperatorBase):
    """AIE-accelerated LAYER NORM operator."""

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
        num_channels=None,
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
        self.num_channels = num_channels
        self.weight = weights

        total_shimdma_channels = self.num_aie_columns * self.num_channels
        assert total_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(
            self, context=context, skip_add_to_list=skip_add_to_list
        )

    def get_artifacts(self, prefix="weighted_layer_norm_"):
        operator_dir = Path(__file__).parent
        file_name_base = f"{prefix}{self.num_aie_columns}c_{self.num_channels}ch_{self.size}_{self.tile_size}t"

        if self.weight is not None:
            weight_signature = self._weight_signature(self.weight)
            file_name_base = f"{file_name_base}_{weight_signature}"
            weight_file_name = self.context.build_dir / f"{file_name_base}_weights.npy"
            np.save(weight_file_name, torch_to_numpy(self.weight))
            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_base}.mlir",
                import_path=operator_dir / "design_weighted.py",
                callback_fn="my_weighted_layer_norm",
                callback_args=[
                    self.context.device_manager.device_type,
                    self.size,
                    self.num_aie_columns,
                    self.num_channels,
                    self.tile_size,
                    weight_file_name,
                    self.trace_size,
                ],
            )
            xclbin_artifact = XclbinArtifact.new(
                f"{file_name_base}.xclbin",
                depends=[
                    mlir_artifact,
                    KernelArchiveArtifact.new(
                        "layer_norm_archive.a",
                        depends=[
                            KernelObjectArtifact.new(
                                "layer_norm.o",
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
                                "mul.o",
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
        else:
            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_base}.mlir",
                import_path=operator_dir / "design.py",
                callback_fn="my_layer_norm",
                callback_args=[
                    self.context.device_manager.device_type,
                    self.size,
                    self.num_aie_columns,
                    self.num_channels,
                    self.trace_size,
                    self.tile_size,
                ],
            )
            xclbin_artifact = XclbinArtifact.new(
                f"{file_name_base}.xclbin",
                depends=[
                    mlir_artifact,
                    KernelObjectArtifact.new(
                        "layer_norm.o",
                        depends=[
                            SourceArtifact.new(
                                self.context.base_dir
                                / "aie_kernels"
                                / "aie2p"
                                / "layer_norm.cc"
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
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("input", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "layer_norm",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("layer_norm", "input", "output")

    def forward(self, x, y=None):
        del y
        if x.numel() > self.size:
            raise AIEOperatorConstraintError(
                "AIELayerNorm: input too large for configured size"
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
