#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference


class _DummyContext:
    def __init__(self):
        self.operators = []
        self.static_data_pool = {}
        self.base_dir = Path(__file__).resolve().parents[3]
        self.device_manager = type(
            "_DummyDeviceManager",
            (),
            {"device_str": staticmethod(lambda: "npu1_4col")},
        )()

    def register_operator(self, operator, skip_add_to_list=False):
        operator.context = self
        if not skip_add_to_list:
            self.operators.append(operator)


def test_artifact_names_track_accuracy_flags():
    context = _DummyContext()
    common = dict(
        M=512,
        K=768,
        N=768,
        num_aie_columns=8,
        tile_m=64,
        tile_k=96,
        tile_n=48,
        context=context,
    )
    acc0 = AIEGEMM(
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
        **common,
    )
    acc1 = AIEGEMM(
        prio_accuracy=True,
        emulate_bf16_mmul_with_bfp16=True,
        **common,
    )
    emb0 = AIEGEMM(
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=False,
        **common,
    )

    acc0_xclbin, acc0_insts = acc0.get_artifacts()
    acc1_xclbin, acc1_insts = acc1.get_artifacts()
    emb0_xclbin, emb0_insts = emb0.get_artifacts()

    assert acc0_xclbin.path.name != acc1_xclbin.path.name
    assert acc0_insts.path.name != acc1_insts.path.name
    assert acc0_xclbin.path.name != emb0_xclbin.path.name
    assert acc0_insts.path.name != emb0_insts.path.name


def generate_test_params(extensive=False):
    params = [
        # M, K, N, cols, b_col_maj, c_col_maj, m, k, n, prio_accuracy, emulate_bf16, trace_size, partition_N, batch_A, batch_B, batch_C
        (
            2048,
            2048,
            2048,
            1,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            2048,
            2,
            True,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            2048,
            8,
            True,
            True,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            384,
            1536,
            1792,
            4,
            True,
            False,
            32,
            48,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            1792,
            896,
            1152,
            8,
            False,
            True,
            64,
            32,
            48,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            896,
            1792,
            640,
            8,
            False,
            True,
            32,
            64,
            80,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            192,
            384,
            64,
            4,
            False,
            False,
            48,
            96,
            16,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            192,
            384,
            64,
            4,
            True,
            True,
            48,
            96,
            16,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            512,
            768,
            768,
            8,
            False,
            False,
            64,
            96,
            48,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            256,
            512,
            2048,
            4,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            2,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        # Restored batched BERT coverage.
        (
            512,
            64,
            512,
            8,
            False,
            False,
            64,
            64,
            64,
            False,
            True,
            0,
            1,
            (12, 1),
            (12, 1),
            (12, 0),
        ),
        (
            512,
            512,
            64,
            4,
            False,
            False,
            64,
            64,
            16,
            False,
            True,
            0,
            1,
            (12, 0),
            (12, 1),
            (12, 1),
        ),
    ]
    extensive_params = [
        (
            2048,
            2048,
            2048,
            8,
            False,
            False,
            32,
            32,
            128,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            8192,
            2,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            8192,
            2048,
            2,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            2048,
            2,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            8192,
            2,
            False,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            2048,
            8,
            True,
            False,
            128,
            32,
            32,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            8192,
            2,
            True,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            8192,
            2048,
            2,
            True,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            2048,
            2,
            True,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            8192,
            2,
            True,
            False,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            2048,
            2,
            False,
            True,
            8,
            16,
            32,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            2048,
            8192,
            2,
            False,
            True,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            8192,
            2048,
            2,
            False,
            True,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            2048,
            2,
            False,
            True,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            2048,
            64,
            8192,
            2,
            False,
            True,
            64,
            64,
            64,
            True,
            False,
            0,
            1,
            (1, 0),
            (1, 0),
            (1, 0),
        ),
        (
            512,
            64,
            512,
            8,
            False,
            False,
            64,
            64,
            64,
            False,
            True,
            0,
            1,
            (12, 1),
            (12, 1),
            (12, 0),
        ),
        (
            512,
            512,
            64,
            4,
            False,
            False,
            64,
            64,
            16,
            False,
            True,
            0,
            1,
            (12, 0),
            (12, 1),
            (12, 1),
        ),
    ]

    if extensive:
        params = extensive_params

    names = []
    for (
        M,
        K,
        N,
        num_aie_columns,
        b_col_maj,
        c_col_maj,
        m,
        k,
        n,
        prio_accuracy,
        emulate_bf16,
        trace_size,
        partition_N,
        batch_A,
        batch_B,
        batch_C,
    ) in params:
        name = f"gemm_{M}x{K}x{N}_{m}x{k}x{n}_{num_aie_columns}cols"
        if b_col_maj:
            name += "_bcolmaj"
        if c_col_maj:
            name += "_ccolmaj"
        if partition_N > 1:
            name += f"_{partition_N}npart"
        if batch_C[0] > 1:
            name += (
                f"_batchA{batch_A[0]}d{batch_A[1]}"
                f"_batchB{batch_B[0]}d{batch_B[1]}"
                f"_batchC{batch_C[0]}d{batch_C[1]}"
            )
        if trace_size > 0:
            name += f"_{trace_size}trace"
        names.append(name)

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


def test_batched_stride_dim_one_shape_inference_uses_middle_axis():
    operator = AIEGEMM(
        M=512,
        K=64,
        N=512,
        tile_m=64,
        tile_k=64,
        tile_n=64,
        num_aie_columns=8,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
        batch_A=(12, 1),
        batch_B=(12, 1),
        batch_C=(12, 0),
    )

    assert operator._get_gemm_shapes((512, 12, 64), operator.batch_A) == (512, 64)
    assert operator._get_gemm_shapes((64, 12, 512), operator.batch_B) == (64, 512)


def test_batched_gemm_trims_padded_n_for_batch_stride_dim_1():
    operator = AIEGEMM(
        M=64,
        K=64,
        N=64,
        tile_m=64,
        tile_k=64,
        tile_n=16,
        num_aie_columns=8,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
        batch_A=(12, 0),
        batch_B=(12, 1),
        batch_C=(12, 1),
        context=_DummyContext(),
    )

    operator._execute_batched_aie_operation = lambda _A_np, _B_np=None: np.ones(
        (64, 12, 128),
        dtype=np.dtype("bfloat16"),
    )

    output = operator(
        torch.zeros((12, 64, 64), dtype=torch.bfloat16),
        torch.zeros((64, 12, 64), dtype=torch.bfloat16),
    )

    assert tuple(output.shape) == (64, 12, 64)


def test_runtime_xclbin_and_instruction_artifacts_can_be_bound_independently():
    common_kwargs = dict(
        tile_m=64,
        tile_k=64,
        tile_n=16,
        num_aie_columns=8,
        emulate_bf16_mmul_with_bfp16=True,
    )

    shared_builder = AIEGEMM(
        M=256,
        K=64,
        N=128,
        context=_DummyContext(),
        skip_add_to_list=True,
        **common_kwargs,
    )
    shared_xclbin = shared_builder.get_runtime_xclbin_artifact(
        prefix="gemm_shared_runtime_"
    )
    shared_xclbin.kernel_name = "gemm_shared"

    def make_bound_op(prefix, M, K, N, use_static_weight=False):
        inst_builder = AIEGEMM(
            M=M,
            K=K,
            N=N,
            use_static_weight=use_static_weight,
            context=_DummyContext(),
            skip_add_to_list=True,
            **common_kwargs,
        )
        insts_artifact = inst_builder.get_insts_artifact(prefix=f"{prefix}_")
        op = AIEGEMM(
            M=M,
            K=K,
            N=N,
            use_static_weight=use_static_weight,
            context=_DummyContext(),
            **common_kwargs,
        )
        op.bind_artifacts(
            shared_xclbin,
            insts_artifact,
            runtime_xclbin_artifact=shared_xclbin,
            runtime_kernel_name=shared_xclbin.kernel_name,
        )
        return op

    qkv_op = make_bound_op("gemm_shared_qkv", 64, 768, 2304, use_static_weight=True)
    scores_op = make_bound_op("gemm_shared_scores", 64, 64, 64)

    assert qkv_op.runtime_xclbin_artifact.path == scores_op.runtime_xclbin_artifact.path
    assert qkv_op.runtime_kernel_name == scores_op.runtime_kernel_name
    assert qkv_op.insts_artifact.path != scores_op.insts_artifact.path


def test_partition_n_changes_insts_but_not_runtime_xclbin():
    common_kwargs = dict(
        M=16384,
        K=64,
        N=16384,
        tile_m=64,
        tile_k=64,
        tile_n=16,
        num_aie_columns=8,
        emulate_bf16_mmul_with_bfp16=True,
        context=_DummyContext(),
        skip_add_to_list=True,
    )

    unpartitioned = AIEGEMM(partition_N=1, **common_kwargs)
    partitioned = AIEGEMM(partition_N=4, **common_kwargs)

    unpartitioned_runtime = unpartitioned.get_runtime_xclbin_artifact(
        prefix="gemm_partition_runtime_"
    )
    partitioned_runtime = partitioned.get_runtime_xclbin_artifact(
        prefix="gemm_partition_runtime_"
    )
    unpartitioned_insts = unpartitioned.get_insts_artifact(prefix="gemm_partition_1_")
    partitioned_insts = partitioned.get_insts_artifact(prefix="gemm_partition_4_")

    assert unpartitioned_runtime.path == partitioned_runtime.path
    assert unpartitioned_insts.path != partitioned_insts.path


@pytest.mark.extensive
def test_batched_attn_scores_16384_compiles():
    context = AIEContext(use_runlist=False)
    AIEGEMM(
        M=256,
        K=64,
        N=16384,
        tile_m=64,
        tile_k=64,
        tile_n=16,
        num_aie_columns=8,
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=True,
        batch_A=(12, 1),
        batch_B=(12, 1),
        batch_C=(12, 0),
        context=context,
    )

    context.compile_all()


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,prio_accuracy,emulate_bf16,trace_size,partition_N,batch_A,batch_B,batch_C",
    all_params,
)
def test_gemm(
    M,
    K,
    N,
    num_aie_columns,
    b_col_maj,
    c_col_maj,
    m,
    k,
    n,
    prio_accuracy,
    emulate_bf16,
    trace_size,
    partition_N,
    batch_A,
    batch_B,
    batch_C,
    aie_context,
):
    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=N,
        partition_N=partition_N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        batch_A=batch_A,
        batch_B=batch_B,
        batch_C=batch_C,
    )

    operator = AIEGEMM(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        prio_accuracy=prio_accuracy,
        emulate_bf16_mmul_with_bfp16=emulate_bf16,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        partition_N=partition_N,
        batch_A=batch_A,
        batch_B=batch_B,
        batch_C=batch_C,
        context=aie_context,
    )

    if batch_C[0] > 1:
        input_buffers = {
            "A": golden_ref["input"].flatten(),
            "B": golden_ref["input_b"].flatten(),
        }
        output_buffers = {"C": golden_ref["output"].flatten()}
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=0.1,
            abs_tol=0.5,
        )
        gflops = (2.0 * M * K * N * batch_C[0]) / (latency_us * 1e-6) / 1e9
        error_threshold = 0.05
        max_acceptable_errors = int(M * N * batch_C[0] * error_threshold)
        print(f"\nLatency (us): {latency_us:.1f}")
        print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
        print(f"Throughput: {gflops:.6e} GFLOP/s\n")
        if errors:
            assert (
                len(errors["C"]) <= max_acceptable_errors
            ), f"Test failed with {len(errors['C'])} errors (max allowable: {max_acceptable_errors})"
        return

    input_buffers = {"A": golden_ref["input"].flatten()}
    output_buffers = {}
    if partition_N > 1:
        for i in range(partition_N):
            input_buffers[f"B_{i}"] = golden_ref["input_b"][i].flatten()
            output_buffers[f"C_{i}"] = golden_ref["output"][i].flatten()
    else:
        input_buffers["B_0"] = golden_ref["input_b"].flatten()
        output_buffers["C_0"] = golden_ref["output"].flatten()

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )
    gflops = (2.0 * M * K * N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, "Test failed"
