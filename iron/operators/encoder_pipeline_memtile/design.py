# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from iron.operators.encoder_pipeline.design import encoder_pipeline as _encoder_pipeline


def encoder_pipeline_memtile(*args, ln1_staging_design: str | None = None, **kwargs):
    if ln1_staging_design is not None and ln1_staging_design.strip().lower() not in {
        "memtile",
        "mt",
        "onchip",
    }:
        raise ValueError(
            "encoder_pipeline_memtile requires ln1_staging_design='memtile'"
        )
    parallel_seq = kwargs.get("parallel_seq", 1)
    if parallel_seq >= 4:
        raise ValueError(
            "encoder_pipeline_memtile does not support parallel_seq >= 4. "
            "The pure on-chip LN1->FFN/LN2 duplication for 4ps exceeds the "
            "current channel budget without an extra worker or a DDR fallback."
        )
    return _encoder_pipeline(*args, ln1_staging_design="memtile", **kwargs)


# Backward-compatible entrypoint name used by the stale fork.
def fused_mha(*args, ln1_stage_mode=None, **kwargs):
    del ln1_stage_mode
    return encoder_pipeline_memtile(*args, **kwargs)
