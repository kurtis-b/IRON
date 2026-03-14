#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

DEBUG_DISABLED = -1

STAGE_QK = 0
STAGE_SOFTMAX = 1
STAGE_PV = 2
STAGE_O_PROJ = 3
STAGE_LN1 = 4
STAGE_UP_PROJ = 5
STAGE_DOWN_PROJ = 6
STAGE_LN2 = 7

_STAGE_NAME_BY_ID = {
    STAGE_QK: "qk",
    STAGE_SOFTMAX: "softmax",
    STAGE_PV: "pv",
    STAGE_O_PROJ: "o_proj",
    STAGE_LN1: "ln1",
    STAGE_UP_PROJ: "up_proj",
    STAGE_DOWN_PROJ: "down_proj",
    STAGE_LN2: "ln2",
}

VERIFY_STAGE_OFFSET = 100

DEBUG_PROFILE_QK = STAGE_QK
DEBUG_PROFILE_SOFTMAX = STAGE_SOFTMAX
DEBUG_PROFILE_PV = STAGE_PV
DEBUG_PROFILE_O_PROJ = STAGE_O_PROJ
DEBUG_PROFILE_LN1 = STAGE_LN1
DEBUG_PROFILE_UP_PROJ = STAGE_UP_PROJ
DEBUG_PROFILE_DOWN_PROJ = STAGE_DOWN_PROJ
DEBUG_PROFILE_LN2 = STAGE_LN2

DEBUG_VERIFY_O_PROJ = VERIFY_STAGE_OFFSET + STAGE_O_PROJ
DEBUG_VERIFY_DOWN_PROJ = VERIFY_STAGE_OFFSET + STAGE_DOWN_PROJ
DEBUG_VERIFY_LN2 = VERIFY_STAGE_OFFSET + STAGE_LN2

PROFILE_DEBUG_CHOICES = tuple(sorted(_STAGE_NAME_BY_ID))
VERIFY_STAGE_IDS = (
    STAGE_O_PROJ,
    STAGE_DOWN_PROJ,
    STAGE_LN2,
)
VERIFY_DEBUG_CHOICES = tuple(
    VERIFY_STAGE_OFFSET + stage_id for stage_id in VERIFY_STAGE_IDS
)
DEBUG_CHOICES = (DEBUG_DISABLED, *PROFILE_DEBUG_CHOICES, *VERIFY_DEBUG_CHOICES)


@dataclass(frozen=True)
class PipelineDebugConfig:
    profile_stage: int | None
    verify_stage: int | None


def stage_name(stage_id: int) -> str:
    try:
        return _STAGE_NAME_BY_ID[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown stage id {stage_id}") from exc


def is_direct_verify_stage(stage_id: int | None) -> bool:
    return stage_id in {STAGE_O_PROJ}


def resolve_pipeline_debug_modes(debug: int) -> PipelineDebugConfig:
    if debug == DEBUG_DISABLED:
        return PipelineDebugConfig(profile_stage=None, verify_stage=None)
    if debug in PROFILE_DEBUG_CHOICES:
        return PipelineDebugConfig(profile_stage=debug, verify_stage=None)
    if debug in VERIFY_DEBUG_CHOICES:
        return PipelineDebugConfig(
            profile_stage=None,
            verify_stage=debug - VERIFY_STAGE_OFFSET,
        )
    raise ValueError(f"debug must be one of {DEBUG_CHOICES} (got {debug})")


def stage_compute_enabled(
    *,
    stage_id: int,
    profile_stage: int | None,
    verify_stage: int | None,
) -> bool:
    if verify_stage is not None:
        return stage_id <= verify_stage
    if profile_stage is not None:
        return stage_id == profile_stage
    return True
