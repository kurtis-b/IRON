#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

STUDY_ID = "memcpy_bandwidth"
SIZE_LADDER: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)
NUM_CORES: tuple[int, ...] = tuple(range(1, 17))
NUM_CHANNELS: tuple[int, ...] = (1, 2)
BYPASS_VALUES: tuple[bool, ...] = (False, True)
ELEMENT_BYTES = 2
MAX_TILE_SIZE = 8192


@dataclass(frozen=True)
class MemcpyBandwidthCase:
    size_elements: int
    num_cores: int
    num_channels: int
    bypass: bool
    tile_size: int

    @property
    def size_bytes(self) -> int:
        return self.size_elements * ELEMENT_BYTES

    @property
    def total_moved_bytes(self) -> int:
        return self.size_bytes * 2

    @property
    def case_id(self) -> str:
        bypass_label = "bypass" if self.bypass else "kernel"
        return (
            f"memcpy_{self.size_elements}_cores{self.num_cores}"
            f"_ch{self.num_channels}_{bypass_label}"
        )


def _resolve_tile_size(
    *,
    size_elements: int,
    num_cores: int,
    num_channels: int,
) -> int | None:
    if num_cores < num_channels:
        return None
    if num_cores > (8 * num_channels):
        return None
    if size_elements % num_cores != 0:
        return None

    tile_size = size_elements // num_cores
    if tile_size > MAX_TILE_SIZE:
        tile_size = MAX_TILE_SIZE
    if tile_size * num_cores != size_elements:
        return None
    return tile_size


def iter_cases(
    *,
    size_filter: str = "all",
    num_cores_filter: str = "all",
    num_channels_filter: str = "all",
    bypass_filter: str = "all",
) -> tuple[MemcpyBandwidthCase, ...]:
    size_values = SIZE_LADDER if size_filter == "all" else (int(size_filter),)
    core_values = NUM_CORES if num_cores_filter == "all" else (int(num_cores_filter),)
    channel_values = (
        NUM_CHANNELS if num_channels_filter == "all" else (int(num_channels_filter),)
    )
    bypass_values = (
        BYPASS_VALUES
        if bypass_filter == "all"
        else (str(bypass_filter).lower() == "true",)
    )

    cases: list[MemcpyBandwidthCase] = []
    for size_elements in size_values:
        for num_channels in channel_values:
            for num_cores in core_values:
                tile_size = _resolve_tile_size(
                    size_elements=size_elements,
                    num_cores=num_cores,
                    num_channels=num_channels,
                )
                if tile_size is None:
                    continue
                for bypass in bypass_values:
                    cases.append(
                        MemcpyBandwidthCase(
                            size_elements=size_elements,
                            num_cores=num_cores,
                            num_channels=num_channels,
                            bypass=bypass,
                            tile_size=tile_size,
                        )
                    )

    return tuple(
        sorted(
            cases,
            key=lambda case: (
                case.size_elements,
                case.num_channels,
                case.num_cores,
                int(case.bypass),
            ),
        )
    )


__all__ = [
    "BYPASS_VALUES",
    "MAX_TILE_SIZE",
    "MemcpyBandwidthCase",
    "NUM_CHANNELS",
    "NUM_CORES",
    "SIZE_LADDER",
    "STUDY_ID",
    "iter_cases",
]
