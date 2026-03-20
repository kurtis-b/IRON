# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from iron.operators.encoder_pipeline.reference import (
    generate_golden_reference as _generate_encoder_pipeline_reference,
)


def generate_golden_reference(**kwargs):
    return _generate_encoder_pipeline_reference(
        ln1_staging_design="memtile",
        **kwargs,
    )
