#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_npu_parity import (
    compute_hidden_state_metrics,
    find_budget_failures,
    parse_args,
)


def test_validate_npu_parity_parse_args_defaults():
    args = parse_args([])

    assert args.seq_lens == "64,128,512"
    assert args.topology_policy == "fixed"
    assert args.cpu_dtype == "float32"
    assert args.execution_mode == "encoder_pipeline"
    assert args.disable_all_biases is False
    assert args.rel_tol == pytest.approx(4.0e-2)
    assert args.abs_tol == pytest.approx(1.5e-1)
    assert args.max_error_fraction == pytest.approx(5.0e-2)


def test_validate_npu_parity_parse_args_accepts_gemm_only_no_bias():
    args = parse_args(["--execution-mode", "gemm_only", "--disable-all-biases"])

    assert args.execution_mode == "gemm_only"
    assert args.disable_all_biases is True


def test_compute_hidden_state_metrics_identical_tensors():
    import torch

    hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    metrics = compute_hidden_state_metrics(
        hidden_state,
        hidden_state.clone(),
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
    )

    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert metrics["mean_abs_error"] == pytest.approx(0.0)
    assert metrics["error_count"] == 0
    assert metrics["error_fraction"] == pytest.approx(0.0)


def test_compute_hidden_state_metrics_detects_error():
    import torch

    reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    candidate = torch.tensor([[[0.9, 0.1], [0.2, 0.7]]], dtype=torch.float32)

    metrics = compute_hidden_state_metrics(
        reference,
        candidate,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert metrics["cosine_similarity"] < 1.0
    assert metrics["max_abs_error"] == pytest.approx(0.3)
    assert metrics["mean_abs_error"] == pytest.approx(0.175)
    assert metrics["error_count"] == 4
    assert metrics["error_fraction"] == pytest.approx(1.0)


def test_find_budget_failures_filters_passing_rows():
    rows = [
        {
            "error_count": "2",
            "max_acceptable_errors": "4",
        },
        {
            "error_count": "5",
            "max_acceptable_errors": "4",
        },
    ]

    failures = find_budget_failures(rows)

    assert failures == [rows[1]]
