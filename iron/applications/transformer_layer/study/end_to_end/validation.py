#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

REFERENCE_TOLERANCE_VALIDATION_MODE = "reference_tolerance_validation"
FINAL_REL_TOL = 0.1
FINAL_ABS_TOL = 0.5
FINAL_ERROR_THRESHOLD = 0.05


def validation_policy_manifest() -> dict[str, float | str]:
    return {
        "validation_mode": REFERENCE_TOLERANCE_VALIDATION_MODE,
        "rtol": FINAL_REL_TOL,
        "atol": FINAL_ABS_TOL,
        "max_error_fraction": FINAL_ERROR_THRESHOLD,
    }
