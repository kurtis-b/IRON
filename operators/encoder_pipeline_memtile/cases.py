#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.common.test_utils import run_test
from operators.encoder_pipeline_memtile.reference import generate_golden_reference
from operators.encoder_pipeline_memtile.op import AIEEncoderPipeline

DEBUG_MODE = -1
REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
TEST_WARMUP_ITERS = 3
TEST_TIMED_ITERS = 20
_INPUT_KEYS = ("QKV", "W_O", "OR", "B_Up", "B_Down")

_REGULAR_BASE_CASES = [
    (64, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
    (64, 64, 12, 3072, 32, 64, 96, 1, 2, 8),
    (64, 64, 12, 3072, 32, 64, 96, 2, 2, 8),
    (64, 64, 12, 3072, 32, 64, 96, 6, 1, 8),
    (64, 64, 12, 3072, 32, 64, 96, 1, 4, 8),
    (128, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
    (512, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
    (2048, 64, 12, 3072, 32, 64, 96, 1, 1, 8),
    (64, 64, 12, 3072, 32, 64, 96, 6, 2, 8),
    (512, 64, 12, 3072, 32, 64, 96, 6, 2, 8),
    (2048, 64, 12, 3072, 32, 64, 96, 6, 2, 8),
    (1024, 64, 12, 3072, 32, 64, 96, 6, 2, 8),
    (64, 64, 12, 3072, 32, 64, 96, 4, 4, 8),
    (512, 64, 12, 3072, 32, 64, 96, 4, 4, 8),
    (2048, 64, 12, 3072, 32, 64, 96, 4, 4, 8),
    (1024, 64, 12, 3072, 32, 64, 96, 4, 4, 8),
    (64, 64, 12, 3072, 32, 64, 128, 4, 6, 6),
    (512, 64, 12, 3072, 32, 64, 128, 4, 6, 6),
    (2048, 64, 12, 3072, 32, 64, 128, 4, 6, 6),
    (1024, 64, 12, 3072, 32, 64, 128, 4, 6, 6),
    (512, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
    (1024, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
    (2048, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
]


def _default_case_o_proj_acc_group_size(parallel_heads, parallel_ffn, proj_acc_depth):
    if parallel_heads == 4 and parallel_ffn == 4 and proj_acc_depth >= 8:
        return 4
    return (
        2
        if parallel_heads % 2 == 0 and (parallel_heads >= 4 or parallel_ffn >= 4)
        else 1
    )


def _is_evenly_partitioned(case):
    _, _, _, intermediate_size, _, _, emb_tile, _, parallel_ffn, _ = case
    return intermediate_size % (emb_tile * parallel_ffn) == 0


def _high_pacc_seq64_variant(case):
    seq_len, d, heads, intermediate_size, _, kv_seq_tile, _, parallel_heads, parallel_ffn, _ = case
    return (
        seq_len,
        d,
        heads,
        intermediate_size,
        64,
        kv_seq_tile,
        (d * heads) // 16,
        parallel_heads,
        parallel_ffn,
        16,
    )


def _case_with_default_opg(case):
    return (*case, _default_case_o_proj_acc_group_size(case[7], case[8], case[9]))


def _validate_case(case):
    _, d, heads, intermediate_size, _, _, emb_tile, parallel_heads, parallel_ffn, proj_acc_depth, opg = case
    emb_size = d * heads
    if emb_tile * proj_acc_depth != emb_size:
        raise ValueError(case)
    if intermediate_size % (emb_tile * parallel_ffn) != 0:
        raise ValueError(case)
    if opg > parallel_heads or parallel_heads % opg != 0:
        raise ValueError(case)


def _case_name(case):
    seq_len, d, heads, intermediate_size, q_seq_tile, kv_seq_tile, emb_tile, parallel_heads, parallel_ffn, proj_acc_depth, o_proj_acc_group_size = case
    return (
        f"encoder_{seq_len}seq_{d}hdim_{heads}heads_{intermediate_size}ffn_"
        f"{q_seq_tile}qseqtile_{kv_seq_tile}kvtile_{emb_tile}embtile_"
        f"{parallel_heads}pheads_{parallel_ffn}pffn_{proj_acc_depth}pacc_"
        f"{o_proj_acc_group_size}opg"
    )


@lru_cache(maxsize=64)
def _cached_golden_reference(seq_len, d, heads, intermediate_size, debug_mode):
    return generate_golden_reference(
        seq_len=seq_len,
        d=d,
        heads=heads,
        intermediate_size=intermediate_size,
        seed=42,
        debug=debug_mode,
    )


def _assert_error_budget(errors, case):
    if not errors:
        return
    max_errors = int(case[0] * case[1] * case[2] * ERROR_THRESHOLD)
    num_errors = len(errors.get("O", ()))
    print(f"({num_errors} errors out of {max_errors} max allowable)")
    assert num_errors <= max_errors


def _run_case(case, aie_context):
    _validate_case(case)
    seq_len, d, heads, intermediate_size, q_seq_tile, kv_seq_tile, emb_tile, parallel_heads, parallel_ffn, proj_acc_depth, o_proj_acc_group_size = case
    ref = _cached_golden_reference(seq_len, d, heads, intermediate_size, DEBUG_MODE)
    op = AIEEncoderPipeline(
        seq_len=seq_len,
        d=d,
        num_heads=heads,
        seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=o_proj_acc_group_size,
        ffn_intermediate_size=intermediate_size,
        nB_tiles_distributed=parallel_ffn,
        debug=DEBUG_MODE,
        ln1_weight=ref["ln1_weight"].clone(),
        ln2_weight=ref["ln2_weight"].clone(),
        context=aie_context,
    )
    return run_test(
        op,
        {k: ref[k].flatten() for k in _INPUT_KEYS},
        {"O": ref["O"].flatten()},
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=TEST_WARMUP_ITERS,
        timed_iters=TEST_TIMED_ITERS,
    )


_COMPARISON_BASE_CASES = list(
    dict.fromkeys(
        _high_pacc_seq64_variant(case)
        for case in _REGULAR_BASE_CASES
        if _is_evenly_partitioned(_high_pacc_seq64_variant(case))
    )
)
_DDR_ONLY_REGULAR_CASES = {
    _case_with_default_opg(case) for case in _REGULAR_BASE_CASES[-3:]
} | {
    _case_with_default_opg(_high_pacc_seq64_variant(case))
    for case in _REGULAR_BASE_CASES[-3:]
}
MEMTILE_CASES = [
    pytest.param(case, id=_case_name(case))
    for case in (_case_with_default_opg(c) for c in (*_REGULAR_BASE_CASES, *_COMPARISON_BASE_CASES))
    if case not in _DDR_ONLY_REGULAR_CASES
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("case", MEMTILE_CASES)
def test_encoder_pipeline_memtile(case, aie_context):
    errors, latency_us, bandwidth_gbps = _run_case(case, aie_context)
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    _assert_error_budget(errors, case)
