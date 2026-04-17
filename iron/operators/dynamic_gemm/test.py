#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
from aie.helpers.taplib import TensorAccessPattern

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.common import AIEOperatorConstraintError
from iron.common.test_utils import run_test
from iron.operators.dynamic_gemm.design import (
    my_matmul as dynamic_gemm_design,
    plan_dynamic_gemm_fill_drain_tasks,
)
from iron.operators.dynamic_gemm.op import AIEDynamicGEMM
from iron.operators.dynamic_gemm.reference import generate_golden_reference


def _tap_signature(tap: TensorAccessPattern) -> tuple[object, ...]:
    return (
        tuple(tap.tensor_dims),
        int(tap.offset),
        tuple(int(value) for value in tap.sizes),
        tuple(int(value) for value in tap.strides),
    )


def test_plan_dynamic_gemm_fill_drain_tasks_matches_runtime_taps():
    plan = plan_dynamic_gemm_fill_drain_tasks(
        M=448,
        K=768,
        N=720,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        b_col_maj=False,
        c_col_maj=False,
        separate_c_tiles=True,
    )

    runtime_A_taps, runtime_B_taps, runtime_C_taps = dynamic_gemm_design(
        dev="npu2",
        M=448,
        K=768,
        N=720,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        dtype_in_str="bf16",
        dtype_out_str="bf16",
        b_col_maj=False,
        c_col_maj=False,
        use_scalar=False,
        emulate_bf16_mmul_with_bfp16=True,
        prio_accuracy=False,
        separate_c_tiles=True,
        trace_size=0,
        archive=None,
        generate_taps=True,
    )

    assert [_tap_signature(tap) for tap in runtime_A_taps] == [
        _tap_signature(tap) for tap in plan.A_taps
    ]
    assert [_tap_signature(tap) for tap in runtime_B_taps] == [
        _tap_signature(tap) for tap in plan.B_taps
    ]
    assert [_tap_signature(tap) for tap in runtime_C_taps] == [
        _tap_signature(tap) for tap in plan.C_taps
    ]


def test_dynamic_gemm_plan_builds_all_tail_phases():
    plan = plan_dynamic_gemm_fill_drain_tasks(
        M=448,
        K=768,
        N=720,
        m=64,
        k=96,
        n=48,
        n_aie_cols=8,
        b_col_maj=False,
        c_col_maj=False,
        separate_c_tiles=True,
    )

    assert [phase.name for phase in plan.phases] == [
        "whole",
        "row_tail",
        "column_tail",
        "corner_tail",
    ]
    assert plan.phases[0].worker_tile_counts[0][0] == 1
    assert plan.phases[1].worker_tile_counts[3][7] == 1
    assert plan.phases[2].worker_tile_counts[0][6] == 1
    assert plan.phases[2].worker_tile_counts[0][7] == 1
    assert plan.phases[3].worker_tile_counts[0][6] == 1
    assert plan.phases[3].worker_tile_counts[0][7] == 1

    column_tail_b_fills = [
        task
        for task in plan.phases[2].tasks
        if task.tensor_name == "B" and task.action == "fill"
    ]
    assert len(column_tail_b_fills) == 8
    assert _tap_signature(column_tail_b_fills[7].tap) == _tap_signature(
        column_tail_b_fills[6].tap
    )

    corner_tail_c_drains = [
        task
        for task in plan.phases[3].tasks
        if task.tensor_name == "C" and task.action == "drain"
    ]
    assert len(corner_tail_c_drains) == 8


def test_dynamic_gemm_artifact_identity_shares_runtime_xclbin(aie_context):
    op_a = AIEDynamicGEMM(
        M=256,
        K=768,
        N=384,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
    )
    op_b = AIEDynamicGEMM(
        M=512,
        K=768,
        N=768,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
    )

    runtime_xclbin_a = op_a.get_runtime_xclbin_artifact()
    runtime_xclbin_b = op_b.get_runtime_xclbin_artifact()
    insts_a = op_a.get_insts_artifact(xclbin_input=runtime_xclbin_a)
    insts_b = op_b.get_insts_artifact(xclbin_input=runtime_xclbin_b)

    assert runtime_xclbin_a.path == runtime_xclbin_b.path
    assert insts_a.path != insts_b.path


def test_dynamic_gemm_uses_single_runtime_buffers(aie_context):
    op = AIEDynamicGEMM(
        M=256,
        K=768,
        N=720,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
    )

    op.set_up_artifacts()
    op.set_up_runtime()

    assert set(op.buffers.keys()) == {"A", "B", "C"}
    assert op.buffer_aliases == {}
    assert op.buffers["A"] == 256 * 768 * 2
    assert op.buffers["B"] == 768 * 720 * 2
    assert op.buffers["C"] == 256 * 720 * 2
    assert op.runlist == [("dynamic_gemm", "A", "B", "C")]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N,use_static_weight",
    [
        (256, 384, False),
        (256, 384, True),
        (256, 720, False),
        (256, 720, True),
        (448, 384, False),
        (448, 384, True),
        (448, 720, False),
        (448, 720, True),
    ],
    ids=[
        "exact_dynamic_b",
        "exact_static_weight",
        "n_tail_dynamic_b",
        "n_tail_static_weight",
        "row_tail_dynamic_b",
        "row_tail_static_weight",
        "corner_tail_dynamic_b",
        "corner_tail_static_weight",
    ],
)
def test_dynamic_gemm_runtime_smoke(M, N, use_static_weight, aie_context):
    K = 768
    golden_ref = generate_golden_reference(M=M, K=K, N=N)

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        use_static_weight=use_static_weight,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
    )
    if use_static_weight:
        operator.weight = golden_ref["input_b"]

    input_buffers = {"A": golden_ref["input"].flatten()}
    if not use_static_weight:
        input_buffers["B"] = golden_ref["input_b"].flatten()
    output_buffers = {"C": golden_ref["output"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N",
    [
        (256, 768),
        (256, 1104),
        (448, 768),
        (448, 1104),
        (512, 720),
        (512, 768),
    ],
    ids=[
        "multi_col_exact",
        "multi_col_n_tail",
        "multi_col_row_tail",
        "multi_col_corner_tail",
        "multi_row_n_tail",
        "multi_row_exact",
    ],
)
def test_dynamic_gemm_runtime_smoke_multi_block_regression(M, N, aie_context):
    K = 768
    golden_ref = generate_golden_reference(M=M, K=K, N=N)

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
    )

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"A": golden_ref["input"].flatten(), "B": golden_ref["input_b"].flatten()},
        {"C": golden_ref["output"].flatten()},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N",
    [
        (256, 192),
        (256, 336),
        (448, 192),
        (448, 336),
    ],
    ids=[
        "cols4_exact",
        "cols4_n_tail",
        "cols4_row_tail",
        "cols4_corner_tail",
    ],
)
def test_dynamic_gemm_runtime_smoke_four_columns(M, N, aie_context):
    K = 768
    golden_ref = generate_golden_reference(M=M, K=K, N=N)

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=4,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
    )

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"A": golden_ref["input"].flatten(), "B": golden_ref["input_b"].flatten()},
        {"C": golden_ref["output"].flatten()},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N",
    [
        (256, 720),
        (448, 720),
    ],
    ids=[
        "layout_exact",
        "layout_corner_tail",
    ],
)
@pytest.mark.parametrize(
    "b_col_maj,c_col_maj",
    [
        (True, False),
    ],
    ids=[
        "b_col_major",
    ],
)
def test_dynamic_gemm_runtime_smoke_layouts(M, N, b_col_maj, c_col_maj, aie_context):
    K = 768
    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"A": golden_ref["input"].flatten(), "B": golden_ref["input_b"].flatten()},
        {"C": golden_ref["output"].flatten()},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c_col_maj": True},
        {"b_col_maj": True, "c_col_maj": True},
    ],
    ids=["c_col_major", "bc_col_major"],
)
def test_dynamic_gemm_rejects_c_col_major(kwargs, aie_context):
    base_kwargs = {
        "M": 256,
        "K": 768,
        "N": 384,
        "tile_m": 64,
        "tile_k": 96,
        "tile_n": 48,
        "num_aie_columns": 8,
        "context": aie_context,
    }
    base_kwargs.update(kwargs)
    with pytest.raises(AIEOperatorConstraintError, match="c_col_maj=True"):
        AIEDynamicGEMM(**base_kwargs)


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N",
    [
        (256, 768),
        (256, 1104),
        (448, 768),
        (448, 1104),
        (512, 720),
        (512, 768),
    ],
    ids=[
        "static_multi_col_exact",
        "static_multi_col_n_tail",
        "static_multi_col_row_tail",
        "static_multi_col_corner_tail",
        "static_multi_row_n_tail",
        "static_multi_row_exact",
    ],
)
def test_dynamic_gemm_runtime_smoke_multi_block_static_weight(M, N, aie_context):
    K = 768
    golden_ref = generate_golden_reference(M=M, K=K, N=N)

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        use_static_weight=True,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
    )
    operator.weight = golden_ref["input_b"]

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"A": golden_ref["input"].flatten()},
        {"C": golden_ref["output"].flatten()},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "M,N",
    [
        (256, 1152),
        (256, 1296),
        (768, 384),
        (960, 720),
    ],
    ids=[
        "stress_three_col_exact",
        "stress_three_col_n_tail",
        "stress_three_row_exact",
        "stress_three_row_corner_tail",
    ],
)
def test_dynamic_gemm_runtime_smoke_large_block_stress(M, N, aie_context):
    K = 768
    golden_ref = generate_golden_reference(M=M, K=K, N=N)

    operator = AIEDynamicGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        num_aie_columns=8,
        context=aie_context,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
    )

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"A": golden_ref["input"].flatten(), "B": golden_ref["input_b"].flatten()},
        {"C": golden_ref["output"].flatten()},
        rel_tol=0.1,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")

    max_acceptable_errors = int(M * N * 0.05)
    if errors:
        assert (
            len(errors["C"]) <= max_acceptable_errors
        ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M": 400, "K": 768, "N": 384},
        {"M": 192, "K": 768, "N": 384},
        {"M": 256, "K": 768, "N": 730},
        {"M": 448, "K": 770, "N": 384},
        {"M": 448, "K": 768, "N": 384, "num_aie_columns": 2},
        {"M": 192, "K": 768, "N": 720},
    ],
)
def test_dynamic_gemm_rejects_invalid_shapes(kwargs, aie_context):
    base_kwargs = {
        "M": 448,
        "K": 768,
        "N": 384,
        "tile_m": 64,
        "tile_k": 96,
        "tile_n": 48,
        "num_aie_columns": 8,
        "context": aie_context,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(AIEOperatorConstraintError):
        AIEDynamicGEMM(**base_kwargs)


@pytest.mark.parametrize("M", [64, 128, 192], ids=["m1_tile", "m2_tiles", "m3_tiles"])
def test_dynamic_gemm_rejects_bootstrap_row_corner_tails(M, aie_context):
    with pytest.raises(
        AIEOperatorConstraintError,
        match="does not support bootstrap row/corner tails with M < 4\\*tile_m",
    ):
        AIEDynamicGEMM(
            M=M,
            K=64,
            N=64,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            num_aie_columns=8,
            context=aie_context,
        )
