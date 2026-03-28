# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_synthetic_inputs_are_hidden_states_only():
    spec = TransformerLayerSpec(seq_len=64)
    inputs = make_synthetic_layer_inputs(spec, seed=3)
    assert inputs.hidden_states.shape == (1, 64, 768)


def test_synthetic_weights_include_qkv_projections():
    spec = TransformerLayerSpec(seq_len=64)
    weights = make_synthetic_layer_weights(spec, seed=4)
    assert "q_proj_weight" in weights
    assert "k_proj_weight" in weights
    assert "v_proj_weight" in weights


def test_reference_layer_runs_from_hidden_states():
    spec = TransformerLayerSpec(seq_len=64)
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=5)
    inputs = make_synthetic_layer_inputs(spec, seed=6)
    layer.assign_weights(weights)
    output = layer(inputs)
    assert output.shape == (1, 64, 768)
