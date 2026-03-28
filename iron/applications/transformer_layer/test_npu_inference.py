# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.npu_inference import (
    SUPPORTED_EXECUTION_MODES,
    benchmark_pattern,
    build_pattern,
)
from iron.applications.transformer_layer.src.bench import (
    SUPPORTED_EXECUTION_MODES as structured_supported_execution_modes,
)
from iron.applications.transformer_layer.src.bench import (
    benchmark_pattern as structured_benchmark_pattern,
)
from iron.applications.transformer_layer.src.bench import (
    build_pattern as structured_build_pattern,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_npu_inference_restructure_preserves_legacy_imports_and_modes():
    assert SUPPORTED_EXECUTION_MODES is structured_supported_execution_modes
    assert benchmark_pattern is structured_benchmark_pattern
    assert build_pattern is structured_build_pattern

    spec = TransformerLayerSpec(seq_len=64)
    for execution_mode in SUPPORTED_EXECUTION_MODES:
        pattern = build_pattern(execution_mode, spec)
        assert getattr(pattern, "pattern_label")
