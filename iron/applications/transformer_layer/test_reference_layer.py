import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_hidden_states,
    make_synthetic_layer_weights,
)


def test_reference_layer_runs_single_layer_contract():
    spec = TransformerLayerSpec(seq_len=64)
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=0)
    hidden_states = make_synthetic_hidden_states(spec, seed=1)
    layer.assign_weights(weights)

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_reference_layer_is_deterministic_for_same_inputs():
    spec = TransformerLayerSpec(seq_len=64)
    layer = ReferenceTransformerLayer(spec)
    weights = make_synthetic_layer_weights(spec, seed=0)
    hidden_states = make_synthetic_hidden_states(spec, seed=1)
    layer.assign_weights(weights)

    first = layer(hidden_states)
    second = layer(hidden_states)

    assert torch.equal(first, second)
