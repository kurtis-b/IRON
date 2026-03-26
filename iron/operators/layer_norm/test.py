#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from iron.common.test_utils import run_test
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.layer_norm.reference import generate_golden_reference


def generate_test_params(extensive=False):
    max_aie_columns = 8
    num_channels = 2
    input_lengths = [2048] if not extensive else [1024, 4096, 8192]

    params = []
    names = []
    for input_length in input_lengths:
        for num_aie_columns in range(1, max_aie_columns + 1):
            for num_channels_layer in range(1, 3):
                total_cores = num_aie_columns * num_channels_layer
                tile_size = input_length // total_cores
                if tile_size > 8192:
                    tile_size = 8192
                check_length = tile_size * total_cores
                if check_length != input_length:
                    continue
                names.append(
                    f"layer_norm_{num_aie_columns}_cols_{num_channels_layer}_channels_{input_length}_tile_{tile_size}"
                )
                params.append(
                    (
                        input_length,
                        num_aie_columns,
                        num_channels_layer,
                        tile_size,
                        False,
                    )
                )

    # The weighted design has a different tiling contract. Keep the
    # old BERT-sized surfaces that are known-valid for hardware tests.
    weighted_params = [
        (393216, 8, 2, 768, True),
        (393216, 4, 2, 768, True),
        (393216, 2, 2, 768, True),
    ]
    weighted_names = [
        "weighted_layer_norm_8_cols_2_channels_393216_weights_768",
        "weighted_layer_norm_4_cols_2_channels_393216_weights_768",
        "weighted_layer_norm_2_cols_2_channels_393216_weights_768",
    ]
    params.extend(weighted_params)
    names.extend(weighted_names)
    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
]


def test_weighted_artifact_names_track_weight_values():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.build_dir = self.base_dir / "build"
            self.device_manager = SimpleNamespace(device_type="npu2")

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    context = DummyContext()
    weight_a = torch.ones(768, dtype=torch.bfloat16)
    weight_b = torch.linspace(1, 2, 768, dtype=torch.float32).to(torch.bfloat16)

    op_a = AIELayerNorm(
        size=64 * 768,
        num_aie_columns=8,
        num_channels=2,
        tile_size=768,
        weights=weight_a,
        context=context,
    )
    op_b = AIELayerNorm(
        size=64 * 768,
        num_aie_columns=8,
        num_channels=2,
        tile_size=768,
        weights=weight_b,
        context=context,
    )
    op_a_same = AIELayerNorm(
        size=64 * 768,
        num_aie_columns=8,
        num_channels=2,
        tile_size=768,
        weights=weight_a.clone(),
        context=context,
    )

    xclbin_a, _ = op_a.get_artifacts()
    xclbin_b, _ = op_b.get_artifacts()
    xclbin_a_same, _ = op_a_same.get_artifacts()

    assert xclbin_a.path.name != xclbin_b.path.name
    assert xclbin_a.path.name == xclbin_a_same.path.name


def test_weighted_layer_norm_short_surface(aie_context):
    rows, cols = 64, 768
    weight = torch.linspace(1, 2, cols, dtype=torch.float32).to(torch.bfloat16)
    x = torch.randn(rows, cols, dtype=torch.bfloat16)
    mean = x.to(torch.float32).mean(dim=-1, keepdim=True)
    var = x.to(torch.float32).var(dim=-1, unbiased=False, keepdim=True)
    y = (
        (x.to(torch.float32) - mean) / torch.sqrt(var + 1e-5) * weight.to(torch.float32)
    ).to(torch.bfloat16)

    operator = AIELayerNorm(
        size=rows * cols,
        num_aie_columns=8,
        num_channels=2,
        tile_size=cols,
        weights=weight,
        context=aie_context,
    )

    errors, _, _ = run_test(
        operator,
        {"input": x},
        {"output": y},
        rel_tol=0.1,
        abs_tol=0.1,
        warmup_iters=1,
        timed_iters=1,
    )

    assert not errors, f"Short weighted surface failed with errors: {errors}"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,num_aie_columns,num_channels,tile_size,weighted",
    all_params,
)
def test_layer_norm(
    input_length, num_aie_columns, num_channels, tile_size, weighted, aie_context
):
    rows = input_length // tile_size
    cols = tile_size
    golden_ref = generate_golden_reference(rows=rows, cols=cols, weighted=weighted)

    operator = AIELayerNorm(
        size=input_length,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        weights=golden_ref["weight"] if weighted else None,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.1, abs_tol=0.1
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
