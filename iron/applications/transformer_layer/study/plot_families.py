#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PLOT_FAMILY_GRID_SHAPE: tuple[int, int] = (2, 3)
PLOT_ROW_LABELS: tuple[str, str] = ("Encoder", "Decoder")


@dataclass(frozen=True)
class PlotFamilySpec:
    family_id: str
    workload_variant: str
    plot_label: str
    row_index: int
    col_index: int


PLOT_FAMILY_SPECS: tuple[PlotFamilySpec, ...] = (
    PlotFamilySpec(
        family_id="tinybert_512",
        workload_variant="encoder_bert",
        plot_label="B-S",
        row_index=0,
        col_index=0,
    ),
    PlotFamilySpec(
        family_id="baseline_768",
        workload_variant="encoder_bert",
        plot_label="B-M",
        row_index=0,
        col_index=1,
    ),
    PlotFamilySpec(
        family_id="baseline_1024",
        workload_variant="encoder_bert",
        plot_label="B-L",
        row_index=0,
        col_index=2,
    ),
    PlotFamilySpec(
        family_id="gpt2_512",
        workload_variant="decoder_gpt2",
        plot_label="G-S",
        row_index=1,
        col_index=0,
    ),
    PlotFamilySpec(
        family_id="gpt2_small_768",
        workload_variant="decoder_gpt2",
        plot_label="G-M",
        row_index=1,
        col_index=1,
    ),
    PlotFamilySpec(
        family_id="gpt2_medium_1024",
        workload_variant="decoder_gpt2",
        plot_label="G-L",
        row_index=1,
        col_index=2,
    ),
)
PLOT_FAMILY_ORDER: tuple[str, ...] = tuple(spec.family_id for spec in PLOT_FAMILY_SPECS)
PLOT_FAMILY_IDS_BY_VARIANT: dict[str, tuple[str, ...]] = {
    workload_variant: tuple(
        spec.family_id
        for spec in PLOT_FAMILY_SPECS
        if spec.workload_variant == workload_variant
    )
    for workload_variant in ("encoder_bert", "decoder_gpt2")
}
_PLOT_FAMILY_SPECS_BY_ID = {spec.family_id: spec for spec in PLOT_FAMILY_SPECS}


def plot_family_spec(family_id: str) -> PlotFamilySpec:
    return _PLOT_FAMILY_SPECS_BY_ID[family_id]


def plot_family_label(family_id: str) -> str:
    return plot_family_spec(family_id).plot_label


def plot_family_grid_position(family_id: str) -> tuple[int, int]:
    spec = plot_family_spec(family_id)
    return spec.row_index, spec.col_index


def ordered_plot_family_ids(
    present_family_ids: Iterable[str] | None = None,
    *,
    workload_variant: str | None = None,
) -> tuple[str, ...]:
    present = None if present_family_ids is None else set(present_family_ids)
    ordered: list[str] = []
    for spec in PLOT_FAMILY_SPECS:
        if workload_variant is not None and spec.workload_variant != workload_variant:
            continue
        if present is not None and spec.family_id not in present:
            continue
        ordered.append(spec.family_id)
    return tuple(ordered)


def ordered_plot_families(
    present_family_ids: Iterable[str] | None = None,
    *,
    workload_variant: str | None = None,
) -> tuple[PlotFamilySpec, ...]:
    family_ids = ordered_plot_family_ids(
        present_family_ids,
        workload_variant=workload_variant,
    )
    return tuple(plot_family_spec(family_id) for family_id in family_ids)


__all__ = [
    "PLOT_FAMILY_GRID_SHAPE",
    "PLOT_FAMILY_IDS_BY_VARIANT",
    "PLOT_FAMILY_ORDER",
    "PLOT_FAMILY_SPECS",
    "PLOT_ROW_LABELS",
    "PlotFamilySpec",
    "ordered_plot_families",
    "ordered_plot_family_ids",
    "plot_family_grid_position",
    "plot_family_label",
    "plot_family_spec",
]
