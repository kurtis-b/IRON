# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

DEBUG_DISABLED = -1

DEBUG_SELF_ATTN = 0
DEBUG_MHA_INPUT_PATH = 1
DEBUG_RESIDUAL_PATH = 2
DEBUG_FFN_UP_ONLY = 3
DEBUG_FFN_DOWN_ONLY = 4
DEBUG_FFN_ADDNORM_ONLY = 5

ADDNORM_DEBUG_DISABLED = -1
ADDNORM_DEBUG_INPUT = 0
ADDNORM_DEBUG_RESIDUAL = 1

_PIPELINE_DEBUG_TO_INTERNAL = {
    DEBUG_DISABLED: (0, None, ADDNORM_DEBUG_DISABLED, ADDNORM_DEBUG_DISABLED),
    DEBUG_SELF_ATTN: (1, None, ADDNORM_DEBUG_INPUT, ADDNORM_DEBUG_INPUT),
    DEBUG_MHA_INPUT_PATH: (-1, None, ADDNORM_DEBUG_INPUT, ADDNORM_DEBUG_INPUT),
    DEBUG_RESIDUAL_PATH: (-1, None, ADDNORM_DEBUG_RESIDUAL, ADDNORM_DEBUG_RESIDUAL),
    DEBUG_FFN_UP_ONLY: (-1, 0, ADDNORM_DEBUG_INPUT, ADDNORM_DEBUG_INPUT),
    DEBUG_FFN_DOWN_ONLY: (-1, 1, ADDNORM_DEBUG_INPUT, ADDNORM_DEBUG_INPUT),
    DEBUG_FFN_ADDNORM_ONLY: (-1, 2, ADDNORM_DEBUG_INPUT, ADDNORM_DEBUG_INPUT),
}

_DEBUG_CHOICES = tuple(_PIPELINE_DEBUG_TO_INTERNAL)


def resolve_pipeline_debug_modes(debug: int) -> tuple[int, int | None, int, int]:
    """Map top-level debug mode to (mha_debug, ffn_stage_only, an1_mode, an2_mode)."""
    try:
        return _PIPELINE_DEBUG_TO_INTERNAL[debug]
    except KeyError as exc:
        raise ValueError(
            f"debug must be one of {_DEBUG_CHOICES} (got {debug})"
        ) from exc
