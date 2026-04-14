#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.aie_context import AIEContext
from operators.qkv_proj.op import AIEQKVProj
from operators.qkv_proj.reference import generate_golden_reference
from iron.common.test_utils import run_test

TEST_BERT = True


def generate_test_params(extensive=False):
    if TEST_BERT:
        params = [
            # seq_len, hidden_size, tile_m, tile_k, tile_n, parallel_seq, parallel_emb
            (64, 768, 32, 64, 16, 1, 1),
            (512, 768, 32, 64, 16, 1, 1),
            (64, 768, 32, 64, 16, 1, 4),
            (128, 768, 32, 64, 16, 4, 1),
            (64, 768, 32, 64, 16, 1, 1),
            (64, 1024, 32, 64, 16, 1, 1),
            (512, 768, 64, 64, 64, 4, 6),
        ]
        extensive_params = []
    else:
        params = []
        extensive_params = []

    if extensive:
        params = extensive_params

    names = []
    for (
        seq_len,
        hidden_size,
        tile_m,
        tile_k,
        tile_n,
        parallel_seq,
        parallel_emb,
    ) in params:
        names.append(
            f"qkv_proj_{seq_len}x{hidden_size}_{tile_m}x{tile_k}x{tile_n}_"
            f"ps{parallel_seq}_pe{parallel_emb}"
        )

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


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,hidden_size,tile_m,tile_k,tile_n,parallel_seq,parallel_emb",
    all_params,
)
def test_qkv_proj(
    seq_len,
    hidden_size,
    tile_m,
    tile_k,
    tile_n,
    parallel_seq,
    parallel_emb,
    aie_context,
):
    logging.debug(
        "Testing QKV projection with seq_len=%s hidden_size=%s tile=%sx%sx%s "
        "parallel_seq=%s parallel_emb=%s",
        seq_len,
        hidden_size,
        tile_m,
        tile_k,
        tile_n,
        parallel_seq,
        parallel_emb,
    )

    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        hidden_size=hidden_size,
    )

    operator = AIEQKVProj(
        seq_len=seq_len,
        hidden_size=hidden_size,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        parallel_seq=parallel_seq,
        parallel_emb=parallel_emb,
        context=aie_context,
    )

    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "B": golden_ref["input_b"].flatten(),
    }
    output_buffers = {
        "Q": golden_ref["output_q"].flatten(),
        "K": golden_ref["output_k"].flatten(),
        "V": golden_ref["output_v"].flatten(),
    }

    if TEST_BERT:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=4.0e-2,
            abs_tol=1.5e-1,
            warmup_iters=10,
            timed_iters=100,
        )
    else:
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=4.0e-2,
            abs_tol=1.5e-1,
        )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    max_acceptable_errors = int(seq_len * hidden_size * 0.005)
    for buf_name in ("Q", "K", "V"):
        buf_errors = len(errors.get(buf_name, []))
        if buf_errors:
            logging.info(
                "(%s errors out of %s max allowable) for %s",
                buf_errors,
                max_acceptable_errors,
                buf_name,
            )
        assert (
            buf_errors <= max_acceptable_errors
        ), f"Test failed for {buf_name} with {buf_errors} errors (max allowable: {max_acceptable_errors})"


def test_qkv_proj_artifacts_namespace_matmul_symbols(tmp_path):
    context = AIEContext(use_runlist=False)
    context.build_dir = tmp_path
    context.build_dir.mkdir(parents=True, exist_ok=True)

    operator = AIEQKVProj(
        seq_len=64,
        hidden_size=768,
        tile_m=64,
        tile_k=64,
        tile_n=48,
        parallel_seq=4,
        parallel_emb=8,
        context=context,
    )
    xclbin_artifact, _ = operator.get_artifacts()
    archive = next(
        dep
        for dep in xclbin_artifact.depends
        if dep.__class__.__name__ == "KernelArchiveArtifact"
    )
    mm_object = next(
        dep for dep in archive.depends if dep.path.name.startswith("qkv_proj_")
    )
    assert mm_object.rename_symbols["matmul_bf16_bf16"] == "matmul_bf16_bf16_qkv_proj"
    assert mm_object.rename_symbols["zero_bf16"] == "zero_bf16_qkv_proj"
