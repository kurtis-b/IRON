#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.common.test_utils import run_test
from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.reference import generate_golden_reference

DEBUG_MODE = -1
REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005

TEST_WARMUP_ITERS = 3
TEST_TIMED_ITERS = 20
TEST_LN1_STAGING_DESIGNS = ("memtile", "ddr")

ENABLE_STAGE_PROFILE_TESTS = False
STAGE_PROFILE_BASE_CASE = (1024, 64, 12, 3072, 32, 64, 96, 6, 3, 8)
STAGE_PROFILE_MODES = (
    "full",
    "mha",
    "an1",
    "an1_stats",
    "an1_post",
    "up",
    "down",
    "an2",
)
STAGE_PROFILE_WARMUP_ITERS = 3
STAGE_PROFILE_TIMED_ITERS = 20

_INPUT_KEYS = ("QKV", "W_O", "OR", "B_Up", "B_Down")
_STAGE_PROFILE_SPECS = {
    "full": (-1, "full"),
    "mha": (6, "mha_only"),
    "an1": (7, "addnorm1_only"),
    "an1_stats": (8, "addnorm1_stats_only"),
    "an1_post": (9, "addnorm1_post_only"),
    "up": (3, "up_only"),
    "down": (4, "down_only"),
    "an2": (5, "addnorm2_only"),
}
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
    # NOTE: The bottom 3 in the list becomes DDR-only tests
    (512, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
    (1024, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
    (2048, 64, 16, 4096, 32, 64, 128, 4, 4, 8),
]


def _default_case_o_proj_acc_group_size(
    parallel_heads: int, parallel_ffn: int, proj_acc_depth: int
) -> int:
    if parallel_heads == 4 and parallel_ffn == 4 and proj_acc_depth >= 8:
        return 4
    return (
        2
        if parallel_heads % 2 == 0 and (parallel_heads >= 4 or parallel_ffn >= 4)
        else 1
    )


def _is_evenly_partitioned(case: tuple[int, ...]) -> bool:
    _, _, _, intermediate_size, _, _, emb_tile, _, parallel_ffn, _ = case
    return intermediate_size % (emb_tile * parallel_ffn) == 0


def _high_pacc_seq64_variant(case: tuple[int, ...]) -> tuple[int, ...]:
    (
        seq_len,
        d,
        heads,
        intermediate_size,
        _,
        kv_seq_tile,
        _,
        parallel_heads,
        parallel_ffn,
        _,
    ) = case
    embed_sz = d * heads
    return (
        seq_len,
        d,
        heads,
        intermediate_size,
        64,
        kv_seq_tile,
        embed_sz // 16,
        parallel_heads,
        parallel_ffn,
        16,
    )


def _case_with_default_opg(case: tuple[int, ...]) -> tuple[int, ...]:
    return (*case, _default_case_o_proj_acc_group_size(case[7], case[8], case[9]))


def _validate_case(case: tuple[int, ...]) -> None:
    _, d, heads, intermediate_size, _, _, emb_tile, parallel_heads, parallel_ffn, proj_acc_depth, opg = case
    emb_size = d * heads
    if emb_size % emb_tile:
        raise ValueError(
            f"emb_tile must divide emb_size ({emb_size} % {emb_tile} != 0) for {case}"
        )
    if emb_tile * proj_acc_depth != emb_size:
        raise ValueError(
            f"proj_acc_depth * emb_tile must equal emb_size "
            f"({proj_acc_depth} * {emb_tile} != {emb_size}) for {case}"
        )
    if not _is_evenly_partitioned(case[:-1]):
        raise ValueError(
            "FFN branches must be evenly partitioned "
            f"({intermediate_size} % ({emb_tile} * {parallel_ffn}) != 0) for {case}"
        )
    invalid_opg_checks = [
        opg > parallel_heads,
        parallel_heads % opg != 0,
    ]
    if any(invalid_opg_checks):
        raise ValueError(f"o_proj_acc_group_size must divide parallel_heads for {case}")


STAGE_PROFILE_CASE = _case_with_default_opg(STAGE_PROFILE_BASE_CASE)

_COMPARISON_BASE_CASES = list(
    dict.fromkeys(
        _high_pacc_seq64_variant(case)
        for case in _REGULAR_BASE_CASES
        if _is_evenly_partitioned(_high_pacc_seq64_variant(case))
    )
)
_REGULAR_CASES = [
    _case_with_default_opg(case)
    for case in (*_REGULAR_BASE_CASES, *_COMPARISON_BASE_CASES)
]
for _case in _REGULAR_CASES:
    _validate_case(_case)
_DDR_ONLY_REGULAR_CASES = {
    _case_with_default_opg(case) for case in _REGULAR_BASE_CASES[-3:]
} | {
    _case_with_default_opg(_high_pacc_seq64_variant(case))
    for case in _REGULAR_BASE_CASES[-3:]
}
@lru_cache(maxsize=64)
def _cached_golden_reference(
    seq_len: int, d: int, heads: int, intermediate_size: int, debug_mode: int
):
    return generate_golden_reference(
        seq_len=seq_len,
        d=d,
        heads=heads,
        intermediate_size=intermediate_size,
        seed=42,
        debug=debug_mode,
    )


def _case_name(
    seq_len,
    d,
    heads,
    intermediate_size,
    q_seq_tile,
    kv_seq_tile,
    emb_tile,
    parallel_heads,
    parallel_ffn,
    proj_acc_depth,
    o_proj_acc_group_size,
):
    return (
        f"encoder_{seq_len}seq_{d}hdim_{heads}heads_{intermediate_size}ffn_"
        f"{q_seq_tile}qseqtile_{kv_seq_tile}kvtile_{emb_tile}embtile_"
        f"{parallel_heads}pheads_{parallel_ffn}pffn_{proj_acc_depth}pacc_"
        f"{o_proj_acc_group_size}opg"
    )

def _run_encoder_pipeline_case(
    case,
    *,
    debug_mode,
    aie_context,
    warmup_iters,
    timed_iters,
    ln1_staging_design,
):
    _validate_case(case)
    (
        seq_len,
        d,
        heads,
        intermediate_size,
        q_seq_tile,
        kv_seq_tile,
        emb_tile,
        parallel_heads,
        parallel_ffn,
        proj_acc_depth,
        o_proj_acc_group_size,
    ) = case

    ref = _cached_golden_reference(seq_len, d, heads, intermediate_size, debug_mode)
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
        debug=debug_mode,
        ln1_weight=ref["ln1_weight"].clone(),
        ln2_weight=ref["ln2_weight"].clone(),
        ln1_staging_design=ln1_staging_design,
        context=aie_context,
    )
    return run_test(
        op,
        {k: ref[k].flatten() for k in _INPUT_KEYS},
        {"O": ref["O"].flatten()},
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )


def _assert_error_budget(errors, case):
    if not errors:
        return
    max_errors = int(case[0] * case[1] * case[2] * ERROR_THRESHOLD)
    num_errors = len(errors.get("O", ()))
    print(f"({num_errors} errors out of {max_errors} max allowable)")
    assert (
        num_errors <= max_errors
    ), f"Test failed with {num_errors} errors (max allowable: {max_errors})"


def _build_encoder_params():
    params = []
    for case in _REGULAR_CASES:
        designs = (
            ("ddr",) if case in _DDR_ONLY_REGULAR_CASES else TEST_LN1_STAGING_DESIGNS
        )
        for design in designs:
            params.append(
                pytest.param(
                    case,
                    design,
                    id=f"lnstage_{design}-{_case_name(*case)}",
                )
            )
    return params


def _build_stage_profile_params():
    params = []
    for stage_mode in STAGE_PROFILE_MODES:
        debug_mode, label = _STAGE_PROFILE_SPECS[stage_mode]
        for design in TEST_LN1_STAGING_DESIGNS:
            params.append(
                pytest.param(
                    stage_mode,
                    debug_mode,
                    label,
                    STAGE_PROFILE_CASE,
                    design,
                    id=(
                        f"lnstage_{design}-"
                        f"{STAGE_PROFILE_CASE[0]}seq_{STAGE_PROFILE_CASE[2]}heads_"
                        f"{STAGE_PROFILE_CASE[7]}pheads_{STAGE_PROFILE_CASE[8]}pffn_"
                        f"{STAGE_PROFILE_CASE[9]}pacc-{label}"
                    ),
                )
            )
    return params


ENCODER_PARAMS = _build_encoder_params()
STAGE_PROFILE_PARAMS = _build_stage_profile_params()


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("case,ln1_staging_design", ENCODER_PARAMS)
def test_encoder_pipeline(case, ln1_staging_design, aie_context):
    errors, latency_us, bandwidth_gbps = _run_encoder_pipeline_case(
        case,
        debug_mode=DEBUG_MODE,
        aie_context=aie_context,
        warmup_iters=TEST_WARMUP_ITERS,
        timed_iters=TEST_TIMED_ITERS,
        ln1_staging_design=ln1_staging_design,
    )
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    _assert_error_budget(errors, case)


@pytest.mark.parametrize(
    "stage_mode,debug_mode,label,stage_case,ln1_staging_design", STAGE_PROFILE_PARAMS
)
def test_encoder_pipeline_stage_profile(
    stage_mode,
    debug_mode,
    label,
    stage_case,
    ln1_staging_design,
    aie_context,
):
    if not ENABLE_STAGE_PROFILE_TESTS:
        pytest.skip("Set ENABLE_STAGE_PROFILE_TESTS=True in test.py to run")
    errors, latency_us, bandwidth_gbps = _run_encoder_pipeline_case(
        stage_case,
        debug_mode=debug_mode,
        aie_context=aie_context,
        warmup_iters=STAGE_PROFILE_WARMUP_ITERS,
        timed_iters=STAGE_PROFILE_TIMED_ITERS,
        ln1_staging_design=ln1_staging_design,
    )
    print(
        f"\nStage Profile [{label}] (debug={debug_mode}, "
        f"warmup={STAGE_PROFILE_WARMUP_ITERS}, timed={STAGE_PROFILE_TIMED_ITERS})"
    )
    print(f"Latency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    if stage_mode == "full":
        _assert_error_budget(errors, stage_case)
    elif errors:
        print(
            f"Stage profile mode '{label}' observed {len(errors.get('O', []))} output mismatches; "
            "numeric threshold check is intentionally skipped for stage-only profiling."
        )
