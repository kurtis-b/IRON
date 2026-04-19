#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import statistics

AIE_TILE_LOCAL_MEMORY_BYTES = 65536
MEM_TILE_LOCAL_MEMORY_BYTES = 524288
SHIM_DMA_CHANNELS_PER_DIRECTION = 2
SHIM_DMA_CHANNELS_TOTAL = SHIM_DMA_CHANNELS_PER_DIRECTION * 2
MEM_TILE_DMA_CHANNELS_TOTAL = 6
COMPUTE_TILE_DMA_CHANNELS_TOTAL = 2

_CORE_RE = re.compile(r"aie\.core\(\s*(%tile_\d+_\d+)\s*\)")
_BUFFER_RE = re.compile(
    r"aie\.buffer\(\s*(%(?:tile|mem_tile)_\d+_\d+)\s*\).*?:\s*memref<([^>]+)>",
    re.DOTALL,
)
_DMA_ALLOCATION_RE = re.compile(
    r"aie\.(?P<kind>\w*dma_allocation)\s+@\w+\(\s*(?P<tile>%(?:shim_noc_tile|mem_tile|tile)_\d+_\d+)\s*,\s*(?P<direction>S2MM|MM2S)\s*,\s*(?P<channel>\d+)\s*\)"
)

_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "ui8": 1,
    "i16": 2,
    "ui16": 2,
    "i32": 4,
    "ui32": 4,
    "i64": 8,
    "ui64": 8,
    "index": 8,
}


@dataclass(frozen=True)
class ResourceUsage:
    compute_tiles_used: int
    aie_tiles_with_buffers: int
    aie_tile_allocated_bytes: int
    aie_tile_memory_utilization: float
    aie_tile_memory_utilization_min: float | None
    aie_tile_memory_utilization_max: float | None
    aie_tile_memory_utilization_mean: float | None
    aie_tile_memory_utilization_median: float | None
    mem_tiles_with_buffers: int
    mem_tile_allocated_bytes: int
    mem_tile_memory_utilization: float
    mem_tile_memory_utilization_min: float | None
    mem_tile_memory_utilization_max: float | None
    mem_tile_memory_utilization_mean: float | None
    mem_tile_memory_utilization_median: float | None
    shim_tiles_with_s2mm: int
    shim_s2mm_channels_used: int
    shim_s2mm_channel_utilization: float
    shim_tiles_with_mm2s: int
    shim_mm2s_channels_used: int
    shim_mm2s_channel_utilization: float
    shim_tiles_with_dma: int
    shim_dma_channels_used: int
    shim_dma_channel_utilization: float
    mem_tiles_with_dma: int | None
    mem_dma_channels_used: int | None
    mem_dma_channel_utilization: float | None
    mem_dma_channel_note: str | None
    compute_tiles_with_dma: int | None
    compute_dma_channels_used: int | None
    compute_dma_channel_utilization: float | None
    compute_dma_channel_note: str | None

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _memref_allocation_bytes(memref_spec: str) -> int | None:
    spec = memref_spec.split(",", 1)[0].strip()
    parts = [part.strip() for part in spec.split("x") if part.strip()]
    if len(parts) < 2:
        return None
    dtype = parts[-1]
    itemsize = _DTYPE_BYTES.get(dtype)
    if itemsize is None:
        return None
    count = 1
    for dim in parts[:-1]:
        if not dim.isdigit():
            return None
        count *= int(dim)
    return count * itemsize


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _tile_utilization_stats(
    allocated_bytes_by_tile: dict[str, int],
    *,
    tile_capacity_bytes: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not allocated_bytes_by_tile:
        return (None, None, None, None)
    utilizations = [
        allocated_bytes / float(tile_capacity_bytes)
        for allocated_bytes in allocated_bytes_by_tile.values()
    ]
    return (
        min(utilizations),
        max(utilizations),
        statistics.fmean(utilizations),
        statistics.median(utilizations),
    )


def _dma_channel_summary(
    channels_by_tile: dict[str, set[int]],
    *,
    total_channels_per_tile: int,
    missing_note: str,
) -> tuple[int | None, int | None, float | None, str | None]:
    if not channels_by_tile:
        return (None, None, None, missing_note)
    tile_count = len(channels_by_tile)
    channel_count = sum(len(channels) for channels in channels_by_tile.values())
    return (
        tile_count,
        channel_count,
        _safe_fraction(channel_count, tile_count * total_channels_per_tile),
        None,
    )


def parse_input_physical_mlir(path: str | Path) -> ResourceUsage:
    content = Path(path).read_text(encoding="utf-8")

    compute_tiles = {match.group(1) for match in _CORE_RE.finditer(content)}
    aie_tile_bytes: dict[str, int] = {}
    mem_tile_bytes: dict[str, int] = {}
    shim_s2mm_channels: dict[str, set[int]] = {}
    shim_mm2s_channels: dict[str, set[int]] = {}
    shim_dma_channels: dict[str, set[int]] = {}
    mem_dma_channels: dict[str, set[int]] = {}
    compute_dma_channels: dict[str, set[int]] = {}

    for match in _BUFFER_RE.finditer(content):
        tile_ref = match.group(1)
        allocated_bytes = _memref_allocation_bytes(match.group(2))
        if allocated_bytes is None:
            continue
        if tile_ref.startswith("%mem_tile_"):
            mem_tile_bytes[tile_ref] = mem_tile_bytes.get(tile_ref, 0) + allocated_bytes
        else:
            aie_tile_bytes[tile_ref] = aie_tile_bytes.get(tile_ref, 0) + allocated_bytes

    for match in _DMA_ALLOCATION_RE.finditer(content):
        tile_ref = match.group("tile")
        direction = match.group("direction")
        channel = int(match.group("channel"))
        if tile_ref.startswith("%shim_noc_tile_"):
            shim_dma_channels.setdefault(tile_ref, set()).add(channel)
            if direction == "S2MM":
                shim_s2mm_channels.setdefault(tile_ref, set()).add(channel)
            else:
                shim_mm2s_channels.setdefault(tile_ref, set()).add(channel)
        elif tile_ref.startswith("%mem_tile_"):
            mem_dma_channels.setdefault(tile_ref, set()).add(channel)
        else:
            compute_dma_channels.setdefault(tile_ref, set()).add(channel)

    aie_allocated_bytes = sum(aie_tile_bytes.values())
    mem_allocated_bytes = sum(mem_tile_bytes.values())
    aie_tiles_with_buffers = len(aie_tile_bytes)
    mem_tiles_with_buffers = len(mem_tile_bytes)
    shim_tiles_with_s2mm = len(shim_s2mm_channels)
    shim_tiles_with_mm2s = len(shim_mm2s_channels)
    shim_s2mm_channels_used = sum(
        len(channels) for channels in shim_s2mm_channels.values()
    )
    shim_mm2s_channels_used = sum(
        len(channels) for channels in shim_mm2s_channels.values()
    )
    shim_tiles_with_dma = len(shim_dma_channels)
    shim_dma_channels_used = sum(
        len(channels) for channels in shim_dma_channels.values()
    )

    (
        aie_util_min,
        aie_util_max,
        aie_util_mean,
        aie_util_median,
    ) = _tile_utilization_stats(
        aie_tile_bytes,
        tile_capacity_bytes=AIE_TILE_LOCAL_MEMORY_BYTES,
    )
    (
        mem_util_min,
        mem_util_max,
        mem_util_mean,
        mem_util_median,
    ) = _tile_utilization_stats(
        mem_tile_bytes,
        tile_capacity_bytes=MEM_TILE_LOCAL_MEMORY_BYTES,
    )

    (
        mem_tiles_with_dma,
        mem_dma_channels_used,
        mem_dma_channel_utilization,
        mem_dma_channel_note,
    ) = _dma_channel_summary(
        mem_dma_channels,
        total_channels_per_tile=MEM_TILE_DMA_CHANNELS_TOTAL,
        missing_note="no explicit memory-tile DMA allocations found in input_physical.mlir",
    )
    (
        compute_tiles_with_dma,
        compute_dma_channels_used,
        compute_dma_channel_utilization,
        compute_dma_channel_note,
    ) = _dma_channel_summary(
        compute_dma_channels,
        total_channels_per_tile=COMPUTE_TILE_DMA_CHANNELS_TOTAL,
        missing_note="no explicit compute-tile DMA allocations found in input_physical.mlir",
    )

    return ResourceUsage(
        compute_tiles_used=len(compute_tiles),
        aie_tiles_with_buffers=aie_tiles_with_buffers,
        aie_tile_allocated_bytes=aie_allocated_bytes,
        aie_tile_memory_utilization=_safe_fraction(
            aie_allocated_bytes,
            aie_tiles_with_buffers * AIE_TILE_LOCAL_MEMORY_BYTES,
        ),
        aie_tile_memory_utilization_min=aie_util_min,
        aie_tile_memory_utilization_max=aie_util_max,
        aie_tile_memory_utilization_mean=aie_util_mean,
        aie_tile_memory_utilization_median=aie_util_median,
        mem_tiles_with_buffers=mem_tiles_with_buffers,
        mem_tile_allocated_bytes=mem_allocated_bytes,
        mem_tile_memory_utilization=_safe_fraction(
            mem_allocated_bytes,
            mem_tiles_with_buffers * MEM_TILE_LOCAL_MEMORY_BYTES,
        ),
        mem_tile_memory_utilization_min=mem_util_min,
        mem_tile_memory_utilization_max=mem_util_max,
        mem_tile_memory_utilization_mean=mem_util_mean,
        mem_tile_memory_utilization_median=mem_util_median,
        shim_tiles_with_s2mm=shim_tiles_with_s2mm,
        shim_s2mm_channels_used=shim_s2mm_channels_used,
        shim_s2mm_channel_utilization=_safe_fraction(
            shim_s2mm_channels_used,
            shim_tiles_with_s2mm * SHIM_DMA_CHANNELS_PER_DIRECTION,
        ),
        shim_tiles_with_mm2s=shim_tiles_with_mm2s,
        shim_mm2s_channels_used=shim_mm2s_channels_used,
        shim_mm2s_channel_utilization=_safe_fraction(
            shim_mm2s_channels_used,
            shim_tiles_with_mm2s * SHIM_DMA_CHANNELS_PER_DIRECTION,
        ),
        shim_tiles_with_dma=shim_tiles_with_dma,
        shim_dma_channels_used=shim_dma_channels_used,
        shim_dma_channel_utilization=_safe_fraction(
            shim_dma_channels_used,
            shim_tiles_with_dma * SHIM_DMA_CHANNELS_TOTAL,
        ),
        mem_tiles_with_dma=mem_tiles_with_dma,
        mem_dma_channels_used=mem_dma_channels_used,
        mem_dma_channel_utilization=mem_dma_channel_utilization,
        mem_dma_channel_note=mem_dma_channel_note,
        compute_tiles_with_dma=compute_tiles_with_dma,
        compute_dma_channels_used=compute_dma_channels_used,
        compute_dma_channel_utilization=compute_dma_channel_utilization,
        compute_dma_channel_note=compute_dma_channel_note,
    )


__all__ = [
    "AIE_TILE_LOCAL_MEMORY_BYTES",
    "COMPUTE_TILE_DMA_CHANNELS_TOTAL",
    "MEM_TILE_DMA_CHANNELS_TOTAL",
    "MEM_TILE_LOCAL_MEMORY_BYTES",
    "ResourceUsage",
    "SHIM_DMA_CHANNELS_PER_DIRECTION",
    "SHIM_DMA_CHANNELS_TOTAL",
    "parse_input_physical_mlir",
]
