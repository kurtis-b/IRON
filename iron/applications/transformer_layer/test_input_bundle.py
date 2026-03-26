import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)


def test_make_synthetic_layer_inputs_matches_spec_shapes():
    spec = TransformerLayerSpec(seq_len=64, hidden_size=768, num_attention_heads=12)
    inputs = make_synthetic_layer_inputs(spec, seed=0)

    assert inputs.q.shape == (1, 12, 64, 64)
    assert inputs.k.shape == (1, 12, 64, 64)
    assert inputs.v.shape == (1, 12, 64, 64)
    assert inputs.r.shape == (1, 64, 768)
    inputs.validate(spec)


def test_make_synthetic_layer_inputs_is_deterministic():
    spec = TransformerLayerSpec(seq_len=64)
    first = make_synthetic_layer_inputs(spec, seed=7)
    second = make_synthetic_layer_inputs(spec, seed=7)

    assert torch.equal(first.q, second.q)
    assert torch.equal(first.k, second.k)
    assert torch.equal(first.v, second.v)
    assert torch.equal(first.r, second.r)


def test_make_synthetic_layer_weights_are_post_projection_only():
    spec = TransformerLayerSpec(seq_len=64)
    weights = make_synthetic_layer_weights(spec, seed=0)

    assert "q_proj_weight" not in weights
    assert "k_proj_weight" not in weights
    assert "v_proj_weight" not in weights
    assert "out_proj_weight" in weights
    assert "ffn_up_weight" in weights
    assert "ffn_down_weight" in weights
    assert "ln1_weight" in weights
    assert "ln2_weight" in weights
