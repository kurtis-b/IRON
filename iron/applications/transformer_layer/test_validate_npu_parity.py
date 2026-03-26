import torch

from iron.applications.transformer_layer.validate_npu_parity import error_stats


def test_error_stats_reports_zero_for_identical_tensors():
    tensor = torch.ones((1, 2, 3), dtype=torch.float32)
    stats = error_stats(tensor, tensor.clone())
    assert stats["max_abs_diff"] == 0.0
    assert stats["mean_abs_diff"] == 0.0
