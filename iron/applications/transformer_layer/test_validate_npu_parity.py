# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pipeline import (
    error_stats as structured_error_stats,
)
from iron.applications.transformer_layer.src.pipeline import (
    run_parity_suite as structured_run_parity_suite,
)
from iron.applications.transformer_layer.src.pipeline import (
    validate_pattern_parity as structured_validate_pattern_parity,
)
from iron.applications.transformer_layer.validate_npu_parity import (
    error_stats,
    run_parity_suite,
    validate_pattern_parity,
)


def test_validate_npu_parity_restructure_preserves_legacy_imports_and_helpers():
    assert error_stats is structured_error_stats
    assert validate_pattern_parity is structured_validate_pattern_parity
    assert run_parity_suite is structured_run_parity_suite

    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.5, 1.0], dtype=torch.float32)
    stats = error_stats(reference, candidate)
    assert stats == {"max_abs_diff": 1.0, "mean_abs_diff": 0.75}

    spec = TransformerLayerSpec(seq_len=64)
    rows = run_parity_suite(
        execution_modes=[],
        seq_lens=[spec.seq_len],
        base_spec=spec,
        seed=0,
    )
    assert rows == []
