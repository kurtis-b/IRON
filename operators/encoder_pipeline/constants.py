# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum


class DebugMode(IntEnum):
    FULL = 0
    SELF_ATTN = 1
    O_PROJ = 2
    FFN_UP_ONLY = 3
    FFN_DOWN_ONLY = 4
    FFN_ADDNORM_ONLY = 5


class AddNormDebugMode(IntEnum):
    DISABLED = -1
    INPUT = 0
    RESIDUAL = 1


# Backward-compatible scalar aliases used across operator/reference/tests.
DEBUG_FULL = int(DebugMode.FULL)
DEBUG_SELF_ATTN = int(DebugMode.SELF_ATTN)
DEBUG_O_PROJ = int(DebugMode.O_PROJ)
DEBUG_FFN_UP_ONLY = int(DebugMode.FFN_UP_ONLY)
DEBUG_FFN_DOWN_ONLY = int(DebugMode.FFN_DOWN_ONLY)
DEBUG_FFN_ADDNORM_ONLY = int(DebugMode.FFN_ADDNORM_ONLY)

ADDNORM_DEBUG_DISABLED = int(AddNormDebugMode.DISABLED)
ADDNORM_DEBUG_INPUT = int(AddNormDebugMode.INPUT)
ADDNORM_DEBUG_RESIDUAL = int(AddNormDebugMode.RESIDUAL)

MHA_DEBUG_MODES = frozenset((DEBUG_FULL, DEBUG_SELF_ATTN, DEBUG_O_PROJ))
VALID_DEBUG_MODES = tuple(int(mode) for mode in DebugMode)
VALID_ADDNORM_DEBUG_MODES = tuple(int(mode) for mode in AddNormDebugMode)
VALID_DEBUG_MODES_STR = ", ".join(str(mode) for mode in VALID_DEBUG_MODES)
VALID_ADDNORM_DEBUG_MODES_STR = ", ".join(
    str(mode) for mode in VALID_ADDNORM_DEBUG_MODES
)

FFN_STAGE_ONLY_BY_DEBUG = {
    DEBUG_FFN_UP_ONLY: 0,
    DEBUG_FFN_DOWN_ONLY: 1,
    DEBUG_FFN_ADDNORM_ONLY: 2,
}

MHA_DEBUG_BY_DEBUG = {
    DEBUG_FULL: DEBUG_FULL,
    DEBUG_SELF_ATTN: DEBUG_SELF_ATTN,
    DEBUG_O_PROJ: DEBUG_O_PROJ,
    # FFN stage-isolation modes reuse deterministic O-proj MHA inputs/weights
    # so FFN/AddNorm validation is not dominated by MHA numeric variance.
    DEBUG_FFN_UP_ONLY: DEBUG_O_PROJ,
    DEBUG_FFN_DOWN_ONLY: DEBUG_O_PROJ,
    DEBUG_FFN_ADDNORM_ONLY: DEBUG_O_PROJ,
}


def resolve_mha_debug_mode(debug: int) -> int:
    if debug not in VALID_DEBUG_MODES:
        raise ValueError(
            f"debug must be one of {{{VALID_DEBUG_MODES_STR}}} (got {debug})"
        )
    return MHA_DEBUG_BY_DEBUG[debug]


def resolve_ffn_stage_only(debug: int) -> int | None:
    if debug not in VALID_DEBUG_MODES:
        raise ValueError(
            f"debug must be one of {{{VALID_DEBUG_MODES_STR}}} (got {debug})"
        )
    return FFN_STAGE_ONLY_BY_DEBUG.get(debug, None)


def resolve_addnorm_modes(
    addnorm_debug_mode: int,
    addnorm1_debug_mode: int | None,
    addnorm2_debug_mode: int | None,
) -> tuple[int, int, int]:
    addnorm1_mode = (
        addnorm_debug_mode if addnorm1_debug_mode is None else addnorm1_debug_mode
    )
    addnorm2_mode = (
        addnorm_debug_mode if addnorm2_debug_mode is None else addnorm2_debug_mode
    )
    for name, value in (
        ("addnorm_debug_mode", addnorm_debug_mode),
        ("addnorm1_debug_mode", addnorm1_mode),
        ("addnorm2_debug_mode", addnorm2_mode),
    ):
        if value not in VALID_ADDNORM_DEBUG_MODES:
            raise ValueError(
                f"{name} must be one of {{{VALID_ADDNORM_DEBUG_MODES_STR}}} (got {value})"
            )
    return addnorm_debug_mode, addnorm1_mode, addnorm2_mode
