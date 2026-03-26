import pytest
import torch

from iron.applications.transformer_layer.validate_operator_runlist_stability import (
    _parse_target_names,
    _run_single_check,
)


def test_run_single_check_reports_stable_repeated_outputs():
    expected = torch.ones((2, 2), dtype=torch.float32)

    row = _run_single_check(
        name="dummy",
        repeats=3,
        expected=expected,
        fn=lambda: expected.clone(),
    )

    assert row["target"] == "dummy"
    assert row["stable"] is True
    assert row["max_abs_diff"] == 0.0
    assert row["mean_abs_diff"] == 0.0


def test_parse_target_names_defaults_to_all_component_targets():
    assert _parse_target_names(None)[0] == "k_transpose"
    assert _parse_target_names(None)[-1] == "ln2"


def test_parse_target_names_normalizes_requested_subset():
    assert _parse_target_names("attn_scores, ln2") == ("attn_scores", "ln2")


def test_parse_target_names_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unknown encoder_runlist component"):
        _parse_target_names("attn_scores,unknown_stage")
