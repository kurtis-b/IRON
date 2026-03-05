#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.reference import generate_golden_reference
from operators.common.test_utils import run_test

DEBUG_MODE = int(os.getenv("ENCODER_PIPELINE_DEBUG_MODE", "-1"))

STAGE_PROFILE_ENABLED = os.getenv("ENCODER_PIPELINE_STAGE_PROFILE", "0") == "1"
STAGE_PROFILE_CASE_RAW = os.getenv(
    "ENCODER_PIPELINE_STAGE_PROFILE_CASE",
    # seq_len,d,heads,intermediate_size,q_seq_tile,kv_seq_tile,emb_tile,parallel_heads,parallel_ffn,proj_acc_depth
    "1024,64,12,3072,32,64,128,6,3,6",
)
STAGE_PROFILE_MODES_RAW = os.getenv(
    "ENCODER_PIPELINE_STAGE_PROFILE_MODES",
    "none,mha,an1,0,1,2",
)
STAGE_PROFILE_WARMUP_ITERS = int(
    os.getenv("ENCODER_PIPELINE_STAGE_PROFILE_WARMUP_ITERS", "3")
)
STAGE_PROFILE_TIMED_ITERS = int(
    os.getenv("ENCODER_PIPELINE_STAGE_PROFILE_TIMED_ITERS", "20")
)

_STAGE_PROFILE_MODE_TO_DEBUG = {
    "full": -1,  # full pipeline
    "up": 3,  # FFN up-proj-focused
    "down": 4,  # FFN down-proj-focused
    "an2": 5,  # FFN AddNorm2-focused
    "mha": 6,  # MHA-focused (FFN up/down disabled, AddNorm bypassed)
    "an1": 7,  # AddNorm1-focused (FFN up/down disabled, AddNorm2 bypassed)
}

_STAGE_PROFILE_MODE_ALIASES = {
    "full": ("none", "full"),
    "up": ("0", "up", "up_only", "ffn_up"),
    "down": ("1", "down", "down_only", "ffn_down"),
    "an2": ("2", "an2", "addnorm2", "addnorm2_only", "ln2"),
    "mha": ("3", "mha", "mha_only"),
    "an1": ("4", "an1", "addnorm1", "addnorm1_only", "ln1"),
}
_STAGE_PROFILE_MODE_BY_ALIAS = {
    alias: mode
    for mode, aliases in _STAGE_PROFILE_MODE_ALIASES.items()
    for alias in aliases
}
_STAGE_PROFILE_MODE_LABELS = {
    "full": "full",
    "up": "up_only",
    "down": "down_only",
    "an2": "addnorm2_only",
    "mha": "mha_only",
    "an1": "addnorm1_only",
}

ERROR_THRESHOLD = 0.005
REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1


def _parse_stage_profile_case(raw: str):
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 10:
        raise ValueError(
            "ENCODER_PIPELINE_STAGE_PROFILE_CASE must contain 10 comma-separated "
            "ints: seq_len,d,heads,intermediate_size,q_seq_tile,kv_seq_tile,"
            "emb_tile,parallel_heads,parallel_ffn,proj_acc_depth "
            f"(got {len(values)} from '{raw}')"
        )
    return tuple(values)


def _parse_stage_profile_modes(raw: str):
    modes = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if t == "":
            continue
        mode = _STAGE_PROFILE_MODE_BY_ALIAS.get(t)
        if mode is not None:
            modes.append(mode)
            continue
        raise ValueError(
            "ENCODER_PIPELINE_STAGE_PROFILE_MODES entries must be one of "
            "{none,full,0/up,1/down,2/an2,3/mha,4/an1} (got '{tok}')"
        )
    deduped = tuple(dict.fromkeys(modes))
    if not deduped:
        raise ValueError(
            "ENCODER_PIPELINE_STAGE_PROFILE_MODES must include at least one mode"
        )
    return deduped


STAGE_PROFILE_CASE = _parse_stage_profile_case(STAGE_PROFILE_CASE_RAW)
STAGE_PROFILE_MODES = _parse_stage_profile_modes(STAGE_PROFILE_MODES_RAW)


def _stage_profile_mode_label(stage_only):
    try:
        return _STAGE_PROFILE_MODE_LABELS[stage_only]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage profile mode: {stage_only}") from exc


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
):
    return (
        f"encoder_{seq_len}seq_{d}hdim_{heads}heads_{intermediate_size}ffn_{q_seq_tile}qseqtile_{kv_seq_tile}kvtile_"
        f"{emb_tile}embtile_{parallel_heads}pheads_{parallel_ffn}pffn_{proj_acc_depth}pacc"
    )


def generate_test_params(extensive=False):
    # Unified tuple:
    # seq_len, d, heads, intermediate_size, q_seq_tile, kv_seq_tile,
    # emb_tile, parallel_heads, parallel_ffn, proj_acc_depth
    regular_params = [
        (64, 64, 12, 3072, 32, 64, 128, 1, 1, 6),
        (128, 64, 12, 3072, 32, 64, 128, 1, 1, 6),
        (512, 64, 12, 3072, 32, 64, 128, 1, 1, 6),
        (2048, 64, 12, 3072, 32, 64, 128, 1, 1, 6),
        (64, 64, 12, 3072, 32, 64, 128, 6, 3, 6),
        (512, 64, 12, 3072, 32, 64, 128, 6, 3, 6),
        (2048, 64, 12, 3072, 32, 64, 128, 6, 3, 6),
        (1024, 64, 12, 3072, 32, 64, 128, 6, 3, 6),
        (512, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
        (1024, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
        (2048, 64, 16, 4096, 32, 64, 128, 4, 3, 8),
    ]
    extensive_params = []

    params = extensive_params if extensive else regular_params
    names = [_case_name(*case) for case in params]
    return params, names


def _run_encoder_pipeline_case(
    case,
    *,
    debug_mode,
    aie_context,
    warmup_iters,
    timed_iters,
):
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
    ) = case

    golden_ref = generate_golden_reference(
        seq_len=seq_len,
        d=d,
        heads=heads,
        intermediate_size=intermediate_size,
        seed=42,
        debug=debug_mode,
    )

    operator = AIEEncoderPipeline(
        seq_len=seq_len,
        d=d,
        num_heads=heads,
        seq_tile=q_seq_tile,
        kv_seq_tile=kv_seq_tile,
        emb_tile=emb_tile,
        parallel_heads=parallel_heads,
        proj_acc_depth=proj_acc_depth,
        ffn_intermediate_size=intermediate_size,
        nB_tiles_distributed=parallel_ffn,
        debug=debug_mode,
        ln1_weight=golden_ref["ln1_weight"],
        ln2_weight=golden_ref["ln2_weight"],
        context=aie_context,
    )

    input_buffers = {
        "QKV": golden_ref["QKV"].flatten(),
        "W_O": golden_ref["W_O"].flatten(),
        "OR": golden_ref["OR"].flatten(),
        "B_Up": golden_ref["B_Up"].flatten(),
        "B_Down": golden_ref["B_Down"].flatten(),
    }
    output_buffers = {"O": golden_ref["O"].flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )
    return errors, latency_us, bandwidth_gbps


def _assert_error_budget(errors, seq_len, d, heads):
    if not errors:
        return
    max_acceptable_errors = int(seq_len * d * heads * ERROR_THRESHOLD)
    num_errors = len(errors.get("O", ()))
    print(f"({num_errors} errors out of {max_acceptable_errors} max allowable)")
    assert (
        num_errors <= max_acceptable_errors
    ), f"Test failed with {num_errors} errors (max allowable: {max_acceptable_errors})"


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
    "seq_len,d,heads,intermediate_size,q_seq_tile,kv_seq_tile,emb_tile,parallel_heads,parallel_ffn,proj_acc_depth",
    all_params,
)
def test_encoder_pipeline(
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
    aie_context,
):
    case = (
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
    )
    errors, latency_us, bandwidth_gbps = _run_encoder_pipeline_case(
        case,
        debug_mode=DEBUG_MODE,
        aie_context=aie_context,
        warmup_iters=10,
        timed_iters=100,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    _assert_error_budget(errors, seq_len, d, heads)


if STAGE_PROFILE_ENABLED:

    @pytest.mark.parametrize(
        "stage_profile_mode",
        STAGE_PROFILE_MODES,
        ids=lambda stage_profile_mode: _stage_profile_mode_label(stage_profile_mode),
    )
    def test_encoder_pipeline_stage_profile(stage_profile_mode, aie_context):
        seq_len, d, heads, *_ = STAGE_PROFILE_CASE
        debug_mode = _STAGE_PROFILE_MODE_TO_DEBUG[stage_profile_mode]
        errors, latency_us, bandwidth_gbps = _run_encoder_pipeline_case(
            STAGE_PROFILE_CASE,
            debug_mode=debug_mode,
            aie_context=aie_context,
            warmup_iters=STAGE_PROFILE_WARMUP_ITERS,
            timed_iters=STAGE_PROFILE_TIMED_ITERS,
        )

        stage_label = _stage_profile_mode_label(stage_profile_mode)
        print(
            f"\nStage Profile [{stage_label}] "
            f"(debug={debug_mode}, warmup={STAGE_PROFILE_WARMUP_ITERS}, timed={STAGE_PROFILE_TIMED_ITERS})"
        )
        print(f"Latency (us): {latency_us:.1f}")
        print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

        if stage_profile_mode == "full":
            _assert_error_budget(errors, seq_len, d, heads)
        elif errors:
            print(
                f"Stage profile mode '{stage_label}' observed {len(errors['O'])} output mismatches; "
                "numeric threshold check is intentionally skipped for stage-only profiling."
            )
