import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)


def test_reference_layer_runs_single_layer_contract():
    spec = TransformerLayerSpec(seq_len=64)
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=0)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=1)
    layer.assign_weights(weights)

    output = layer(layer_inputs)

    assert output.shape == layer_inputs.r.shape
    assert output.dtype == layer_inputs.r.dtype


def test_reference_layer_is_deterministic_for_same_inputs():
    spec = TransformerLayerSpec(seq_len=64)
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=0)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=1)
    layer.assign_weights(weights)

    first = layer(layer_inputs)
    second = layer(layer_inputs)

    assert torch.equal(first, second)


def test_reference_layer_runs_hidden_state_projection_boundary():
    spec = TransformerLayerSpec(seq_len=64, input_boundary="hidden_states")
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=0)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=1)
    layer.assign_weights(weights)

    output = layer(layer_inputs)

    assert output.shape == layer_inputs.r.shape
    assert output.dtype == layer_inputs.r.dtype
