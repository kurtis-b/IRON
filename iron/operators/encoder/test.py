# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.encoder.op import AIEBERTEncoder
from iron.operators.encoder.reference import generate_golden_reference

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
WARMUP_ITERS = 10
TIMED_ITERS = 20


BENCHMARK_CASES = (
    pytest.param(
        512,
        768,
        3072,
        12,
        1,
        "bert",
        id="bert_encoder_1layer_512seq_768hidden_3072ffn_12heads",
    ),
    pytest.param(
        512,
        768,
        3072,
        12,
        2,
        "bert",
        id="bert_encoder_2layer_512seq_768hidden_3072ffn_12heads",
    ),
    pytest.param(
        512,
        768,
        3072,
        12,
        1,
        "roberta",
        id="roberta_encoder_1layer_512seq_768hidden_3072ffn_12heads",
    ),
    pytest.param(
        512,
        768,
        3072,
        12,
        1,
        "distilbert",
        id="distilbert_encoder_1layer_512seq_768hidden_3072ffn_12heads",
    ),
)


def _count_errors(actual: torch.Tensor, expected: torch.Tensor) -> int:
    actual_f32 = actual.to(torch.float32).reshape(-1)
    expected_f32 = expected.to(torch.float32).reshape(-1)
    matches = torch.isclose(actual_f32, expected_f32, rtol=REL_TOL, atol=ABS_TOL)
    return int((~matches).sum().item())


def _run_encoder_stack(operator: AIEBERTEncoder, input_tensor: torch.Tensor):
    operator.compile_all()
    operator.prepare_runtime()

    try:
        for _ in range(WARMUP_ITERS):
            operator(input_tensor)

        elapsed_total = 0.0
        output = None
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            output = operator(input_tensor).clone()
            elapsed_total += time.perf_counter() - start
        assert output is not None
        return output, (elapsed_total / TIMED_ITERS) * 1e6
    finally:
        operator.cleanup()


@pytest.mark.metrics(Latency=r"Latency \(us\): (?P<value>[\d\.]+)")
@pytest.mark.parametrize(
    "seq_len,hidden_size,intermediate_size,num_heads,num_hidden_layers,model_type",
    BENCHMARK_CASES,
)
def test_bert_encoder(
    seq_len,
    hidden_size,
    intermediate_size,
    num_heads,
    num_hidden_layers,
    model_type,
):
    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        model_type=model_type,
    )

    operator = AIEBERTEncoder(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        model_type=model_type,
    )
    operator.assign_weights(golden_ref["weights"])

    output, latency_us = _run_encoder_stack(operator, golden_ref["input"])
    errors = _count_errors(output, golden_ref["output"])
    max_acceptable_errors = int(seq_len * hidden_size * ERROR_THRESHOLD)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"({errors} errors out of {max_acceptable_errors} max allowable)\n")

    assert errors <= max_acceptable_errors
