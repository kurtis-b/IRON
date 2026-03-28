# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_layer_spec_is_hidden_states_only():
    spec = TransformerLayerSpec()
    assert spec.hidden_size == 768
    assert "input_boundary" not in TransformerLayerSpec.__dataclass_fields__

    with pytest.raises(TypeError):
        TransformerLayerSpec(unsupported_boundary="post_projection")
