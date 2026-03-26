import pytest

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_imported_weights_require_source_metadata():
    with pytest.raises(ValueError):
        TransformerLayerSpec(weights_source="imported")


def test_attention_head_size_is_derived():
    spec = TransformerLayerSpec(hidden_size=768, num_attention_heads=12)
    assert spec.attention_head_size == 64


def test_spec_round_trips_through_dict():
    spec = TransformerLayerSpec(seq_len=256, dtype="bfloat16")
    restored = TransformerLayerSpec.from_dict(spec.to_dict())
    assert restored == spec
