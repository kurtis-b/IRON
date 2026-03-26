import json
from pathlib import Path

import torch

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.validate_npu_parity import (
    _validate_operator_runlist_parity_isolated,
    error_stats,
    validate_pattern_parity,
)


def test_error_stats_reports_zero_for_identical_tensors():
    tensor = torch.ones((1, 2, 3), dtype=torch.float32)
    stats = error_stats(tensor, tensor.clone())
    assert stats["max_abs_diff"] == 0.0
    assert stats["mean_abs_diff"] == 0.0


def test_validate_pattern_parity_routes_operator_runlist_to_child(monkeypatch):
    expected = {
        "study_id": "synthetic_transformer_layer",
        "execution_mode": "operator_runlist",
        "seq_len": 64,
        "batch_size": 1,
        "dtype": "bfloat16",
        "weights_source": "synthetic",
        "seed": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
    }
    monkeypatch.setattr(
        "iron.applications.transformer_layer.validate_npu_parity._validate_operator_runlist_parity_isolated",
        lambda **kwargs: expected,
    )

    row = validate_pattern_parity(
        execution_mode="operator_runlist",
        spec=TransformerLayerSpec(seq_len=64),
        seed=0,
    )

    assert row == expected


def test_operator_runlist_parity_child_request(monkeypatch, tmp_path):
    spec = TransformerLayerSpec(seq_len=64)

    def _fake_subprocess_run(command, cwd, check):
        request_path = Path(command[command.index("--request-json") + 1])
        response_path = Path(command[command.index("--response-json") + 1])
        request = json.loads(request_path.read_text())
        assert request["mode"] == "parity"
        assert request["spec"]["seq_len"] == 64
        response_path.write_text(
            json.dumps(
                {
                    "study_id": "synthetic_transformer_layer",
                    "execution_mode": "operator_runlist",
                    "seq_len": 64,
                    "batch_size": 1,
                    "dtype": "bfloat16",
                    "weights_source": "synthetic",
                    "seed": 0,
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                }
            )
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "iron.applications.transformer_layer.validate_npu_parity.subprocess.run",
        _fake_subprocess_run,
    )

    row = _validate_operator_runlist_parity_isolated(
        spec=spec,
        seed=0,
        study_id="synthetic_transformer_layer",
    )

    assert row["execution_mode"] == "operator_runlist"
    assert row["max_abs_diff"] == 0.0
