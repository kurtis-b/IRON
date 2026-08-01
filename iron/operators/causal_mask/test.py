#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from ml_dtypes import bfloat16

from iron.operators.causal_mask.op import AIECausalMask, _build_causal_mask
from iron.operators.causal_mask.reference import generate_golden_reference


def test_build_causal_mask_matches_expected_future_token_pattern():
    query_block_size = 4
    seq_len = 6
    num_heads = 2
    q_start = 2
    masked_fill_value = -7.0

    actual = _build_causal_mask(
        query_block_size=query_block_size,
        seq_len=seq_len,
        num_heads=num_heads,
        q_start=q_start,
        masked_fill_value=masked_fill_value,
    )

    base_mask = torch.tensor(
        [
            [0.0, 0.0, 0.0, -7.0, -7.0, -7.0],
            [0.0, 0.0, 0.0, 0.0, -7.0, -7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    expected = base_mask.unsqueeze(0).repeat(num_heads, 1, 1).reshape(-1)

    assert torch.equal(actual, expected)


def generate_test_params():
    params = [
        (64, 64, 12, 0, 8, 2, 1024),
        (32, 64, 12, 32, 8, 2, 512),
    ]
    names = [
        "causal_mask_prefill_64q_64kv_12h_qstart0",
        "causal_mask_block_32q_64kv_12h_qstart32",
    ]
    return params, names


regular_params, regular_names = generate_test_params()
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "query_block_size,seq_len,num_heads,q_start,num_aie_columns,num_channels,tile_size",
    all_params,
)
def test_causal_mask(
    query_block_size,
    seq_len,
    num_heads,
    q_start,
    num_aie_columns,
    num_channels,
    tile_size,
    aie_context,
):
    golden_ref = generate_golden_reference(
        query_block_size=query_block_size,
        seq_len=seq_len,
        num_heads=num_heads,
        q_start=q_start,
    )

    operator = AIECausalMask(
        query_block_size=query_block_size,
        seq_len=seq_len,
        num_heads=num_heads,
        q_start=q_start,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["input"]}
    operator.context.compile_all()
    operator.context.prepare_runtime()
    operator.write_buffer("input1", input_buffers["input1"])
    operator.write_buffer("output", torch.zeros_like(golden_ref["output"]))
    latency_us = operator.run_runlist() * 1e6
    output = operator.read_buffer_as_torch(
        "output",
        shape=golden_ref["output"].shape,
        dtype=bfloat16,
    )
    total_bytes = operator.buffers["input1"] + operator.buffers["output"]
    bandwidth_gbps = total_bytes / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    mask = golden_ref["mask"].reshape(-1).float()
    output_f = output.float()
    input_f = golden_ref["input"].reshape(-1).float()
    unmasked = mask == 0
    masked = ~unmasked

    assert torch.equal(output_f[unmasked], input_f[unmasked])

    exact_masked_sum = input_f[masked] + mask[masked]
    masked_abs_diff = (output_f[masked] - exact_masked_sum).abs()
    assert float(masked_abs_diff.max()) <= 64.0
