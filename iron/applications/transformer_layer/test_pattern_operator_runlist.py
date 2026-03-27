import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pattern_operator_runlist import (
    OperatorRunlistPattern,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)
from iron.operators.encoder_runlist.op import AIEEncoderRunlist


def test_operator_runlist_wraps_composite_encoder_runlist_operator():
    spec = TransformerLayerSpec(seq_len=64)
    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(make_synthetic_layer_weights(spec, seed=0))

    assert isinstance(pattern.encoder_runlist, AIEEncoderRunlist)
    assert pattern.encoder_runlist.component_order[0] == "k_transpose"
    assert pattern.encoder_runlist.component_order[-1] == "ln2"


def test_operator_runlist_expected_outputs_cover_added_layer_norm_nodes():
    spec = TransformerLayerSpec(seq_len=64)
    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(make_synthetic_layer_weights(spec, seed=0))
    layer_inputs = make_synthetic_layer_inputs(spec, seed=1)

    expected = pattern.encoder_runlist.expected_component_outputs(
        layer_inputs.q.squeeze(0),
        layer_inputs.k.squeeze(0),
        layer_inputs.v.squeeze(0),
        layer_inputs.r.squeeze(0),
        eps=spec.layer_norm_eps,
        names=("attn_scores", "ln1", "ln2"),
    )

    assert set(expected) == {"attn_scores", "ln1", "ln2"}
    assert expected["attn_scores"].shape == (
        spec.num_attention_heads,
        spec.seq_len,
        spec.seq_len,
    )
    assert expected["ln1"].shape == (spec.seq_len, spec.hidden_size)
    assert expected["ln2"].shape == (spec.seq_len, spec.hidden_size)
    assert torch.equal(
        expected["ln2"],
        pattern.encoder_runlist.expected_component_outputs(
            layer_inputs.q.squeeze(0),
            layer_inputs.k.squeeze(0),
            layer_inputs.v.squeeze(0),
            layer_inputs.r.squeeze(0),
            eps=spec.layer_norm_eps,
            names=("ln2",),
        )["ln2"],
    )


def test_operator_runlist_validation_component_case_uses_component_boundary_shapes():
    spec = TransformerLayerSpec(seq_len=64)
    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(make_synthetic_layer_weights(spec, seed=0))
    layer_inputs = make_synthetic_layer_inputs(spec, seed=1)

    case = pattern.encoder_runlist.validation_component_case(
        "attn_softmax",
        layer_inputs.q.squeeze(0),
        layer_inputs.k.squeeze(0),
        layer_inputs.v.squeeze(0),
        layer_inputs.r.squeeze(0),
        eps=spec.layer_norm_eps,
    )

    assert len(case["args"]) == 1
    assert case["args"][0].shape == (
        spec.num_attention_heads * spec.seq_len,
        spec.seq_len,
    )
    assert case["expected"].shape == (
        spec.num_attention_heads * spec.seq_len,
        spec.seq_len,
    )


def test_operator_runlist_hidden_state_boundary_adds_projection_gemms():
    spec = TransformerLayerSpec(seq_len=64, input_boundary="hidden_states")
    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(make_synthetic_layer_weights(spec, seed=0))

    assert pattern.q_proj is not None
    assert pattern.k_proj is not None
    assert pattern.v_proj is not None
    metadata = pattern.get_benchmark_metadata()
    assert metadata["npu_dispatch_count"] == len(pattern.encoder_runlist.runlist) + 3
    assert metadata["npu_unique_xclbin_count"] >= 1
