# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from iron.common import AIEOperatorConstraintError
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline as _AIEEncoderPipeline


class AIEEncoderPipelineMemtile(_AIEEncoderPipeline):
    """Encoder pipeline variant with on-chip LN1 -> FFN transport."""

    def __init__(
        self,
        *args,
        debug: int = -1,
        ln1_staging_design: str | None = None,
        artifact_prefix: str | None = None,
        **kwargs,
    ):
        if debug != -1:
            raise AIEOperatorConstraintError(
                "encoder_pipeline_memtile no longer supports legacy debug modes; "
                f"expected debug=-1 (got {debug})"
            )
        if ln1_staging_design is not None:
            mode = self._resolve_ln1_staging_design(ln1_staging_design)
            if mode != "memtile":
                raise AIEOperatorConstraintError(
                    "encoder_pipeline_memtile requires ln1_staging_design='memtile'"
                )
        parallel_seq = kwargs.get("parallel_seq", 1)
        if parallel_seq >= 4:
            raise AIEOperatorConstraintError(
                "encoder_pipeline_memtile does not support parallel_seq >= 4. "
                "The pure on-chip LN1->FFN/LN2 duplication for 4ps exceeds the "
                "current channel budget without an extra worker or a DDR fallback."
            )
        super().__init__(
            *args,
            ln1_staging_design="memtile",
            artifact_prefix=artifact_prefix or "encoder_pipeline_memtile",
            **kwargs,
        )

    def _design_import_path(self) -> Path:
        return Path(__file__).with_name("design.py")

    def _design_callback_fn(self) -> str:
        return "encoder_pipeline_memtile"

    def _tracked_design_paths(self) -> list[Path]:
        local_dir = Path(__file__).parent
        shared_dir = Path(__file__).parents[1] / "encoder_pipeline"
        return [
            local_dir / "design.py",
            local_dir / "op.py",
            shared_dir / "design.py",
            shared_dir / "op.py",
            shared_dir / "placements.py",
        ]


AIEEncoderPipeline = AIEEncoderPipelineMemtile
