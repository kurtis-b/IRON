# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.applications.transformer_layer.gpu_inference import (
    SUPPORTED_POWER_BACKENDS,
    benchmark_gpu_layer,
    resolve_amd_gpu_device,
)
from iron.applications.transformer_layer.src.bench import (
    SUPPORTED_POWER_BACKENDS as structured_supported_power_backends,
)
from iron.applications.transformer_layer.src.bench import (
    benchmark_gpu_layer as structured_benchmark_gpu_layer,
)
from iron.applications.transformer_layer.src.bench import (
    resolve_amd_gpu_device as structured_resolve_amd_gpu_device,
)


def test_gpu_inference_restructure_preserves_legacy_imports_and_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    assert SUPPORTED_POWER_BACKENDS is structured_supported_power_backends
    assert benchmark_gpu_layer is structured_benchmark_gpu_layer
    assert resolve_amd_gpu_device is structured_resolve_amd_gpu_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        RuntimeError,
        match="AMD GPU comparison requires a ROCm-enabled torch build with a visible GPU",
    ):
        resolve_amd_gpu_device("cuda:0")
