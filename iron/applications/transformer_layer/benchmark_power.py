# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext


def empty_power_stats() -> dict[str, float | None]:
    return {
        "avg_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "power_sample_count": None,
    }


def create_power_monitor(*_args, **_kwargs):
    return nullcontext(empty_power_stats())
