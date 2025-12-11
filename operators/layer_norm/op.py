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


class AIELayerNorm(AIEOperatorBase):
    """AIE-accelerated LAYER NORM operator"""

    def __init__(
        
        self,
        size,
        num_aie_columns,
        num_channels,
        tile_size,
        trace_size=0,
        weighted=False,
        context=None
    ):
        max_multiple = num_aie_columns * tile_size
        padded_size = ((size + max_multiple - 1) // max_multiple) * max_multiple
        self.orig_size = size
        self.size = padded_size
        self.tile_size = tile_size
        self.trace_size = trace_size
        self.num_aie_columns = num_aie_columns
        self.num_channels = num_channels
        self.weighted = weighted

        # Initializes weights to 1. Weights have size embedding dim, which is assumed to be tile size
        self.weight = nn.Parameter(torch.ones(tile_size, dtype=torch.bfloat16))

        total_shimdma_channels = self.num_aie_columns * self.num_channels
        assert total_shimdma_channels <= 16, "Conservative ShimDMA limit"

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent

        if self.weighted:
            file_name_base = f"weighted_layer_norm_{self.num_aie_columns}c_{self.num_channels}ch_{self.size}_{self.tile_size}t"

            mlir_artifact = PythonGeneratedMLIRArtifact.new(
                f"{file_name_base}.mlir",
                import_path=operator_dir / "design_weighted.py",
                callback_fn="my_weighted_layer_norm",
                callback_args=[
                    self.device_manager.device_type,
                    self.size,
                    self.num_aie_columns,
                    self.num_channels,
                    self.tile_size,
                    0,
                ],
            )

            xclbin_artifact = XclbinArtifact.new(
                f"{file_name_base}.xclbin",
                depends=[
                    mlir_artifact,
                    KernelArchiveArtifact.new(
                        f"layer_norm_archive.a",
                        depends=[
                            KernelObjectArtifact.new(
                                f"layer_norm.o",
                                depends=[
                                    SourceArtifact.new(
                                        self.base_dir
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
                                        self.base_dir
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
            file_name_base = f"layer_norm_{self.num_aie_columns}c_{self.num_channels}ch_{self.size}_{self.tile_size}t"

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
                    f"layer_norm.o",
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

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        if self.weighted:
            static_weights = None
            if self.weight is not None:
                static_weights = torch_to_numpy(self.weight)

            self.add_buffer("input", self.size)
            self.add_buffer("weight", self.tile_size, static_data=static_weights)
            self.add_buffer("output", self.size)
            self.add_kernel(
                "eltwise_mul",
                self.xclbin_artifact,
                self.xclbin_artifact.kernel_name,
                self.insts_artifact,
            )
            self.add_to_runlist("eltwise_mul", "input", "weight", "output")
        else:
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
        if x.numel() > self.size:
            raise AIEOperatorConstraintError(
                "AIELayerNorm: input too large for configured size"
            )

        original_shape = x.shape
        x_flat = x.reshape(-1)

        pad_len = self.size - x_flat.numel()
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

        if self.weighted:
            self.write_buffer("input", x_flat)
            if y is not None:
                self.write_buffer("weight", y)
            else:
                assert (
                    self.weight is not None
                ), "Weights must be provided either as input or during initialization."
            self.write_buffer("output", np.zeros(self.size, dtype=bfloat16))
            self.run_runlist()
            result = self.read_buffer_as_torch(
                "output", shape=(self.size,), dtype=bfloat16
            )
        else:
            self.write_buffer("input", x_flat)
            self.write_buffer("output", np.zeros(self.size, dtype=bfloat16))
            self.run_runlist()
            result = self.read_buffer_as_torch(
                "output", shape=(self.size,), dtype=bfloat16
            )

        if pad_len > 0:
            result = result[: x_flat.numel() - pad_len]

        return result.reshape(*original_shape)
