#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

AIE_TILE_LOCAL_MEMORY_BYTES = 65536
MEM_TILE_LOCAL_MEMORY_BYTES = 524288
SHIM_DMA_CHANNELS_PER_DIRECTION = 2

_CORE_RE = re.compile(r"aie\.core\(\s*(%tile_\d+_\d+)\s*\)")
_BUFFER_RE = re.compile(
    r"aie\.buffer\(\s*(%(?:tile|mem_tile)_\d+_\d+)\s*\).*?:\s*memref<([^>]+)>",
    re.DOTALL,
)
_SHIM_DMA_RE = re.compile(
    r"aie\.shim_dma_allocation\s+@\w+\(\s*(%shim_noc_tile_\d+_\d+)\s*,\s*(S2MM|MM2S)\s*,\s*(\d+)\s*\)"
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
    mem_tiles_with_buffers: int
    mem_tile_allocated_bytes: int
    mem_tile_memory_utilization: float
    shim_tiles_with_s2mm: int
    shim_s2mm_channels_used: int
    shim_s2mm_channel_utilization: float
    shim_tiles_with_mm2s: int
    shim_mm2s_channels_used: int
    shim_mm2s_channel_utilization: float

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


def parse_input_physical_mlir(path: str | Path) -> ResourceUsage:
    content = Path(path).read_text(encoding="utf-8")

    compute_tiles = {match.group(1) for match in _CORE_RE.finditer(content)}
    aie_tile_bytes: dict[str, int] = {}
    mem_tile_bytes: dict[str, int] = {}
    shim_s2mm_channels: dict[str, set[int]] = {}
    shim_mm2s_channels: dict[str, set[int]] = {}

    for match in _BUFFER_RE.finditer(content):
        tile_ref = match.group(1)
        allocated_bytes = _memref_allocation_bytes(match.group(2))
        if allocated_bytes is None:
            continue
        if tile_ref.startswith("%mem_tile_"):
            mem_tile_bytes[tile_ref] = mem_tile_bytes.get(tile_ref, 0) + allocated_bytes
        else:
            aie_tile_bytes[tile_ref] = aie_tile_bytes.get(tile_ref, 0) + allocated_bytes

    for match in _SHIM_DMA_RE.finditer(content):
        tile_ref = match.group(1)
        direction = match.group(2)
        channel = int(match.group(3))
        if direction == "S2MM":
            shim_s2mm_channels.setdefault(tile_ref, set()).add(channel)
        else:
            shim_mm2s_channels.setdefault(tile_ref, set()).add(channel)

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

    return ResourceUsage(
        compute_tiles_used=len(compute_tiles),
        aie_tiles_with_buffers=aie_tiles_with_buffers,
        aie_tile_allocated_bytes=aie_allocated_bytes,
        aie_tile_memory_utilization=_safe_fraction(
            aie_allocated_bytes,
            aie_tiles_with_buffers * AIE_TILE_LOCAL_MEMORY_BYTES,
        ),
        mem_tiles_with_buffers=mem_tiles_with_buffers,
        mem_tile_allocated_bytes=mem_allocated_bytes,
        mem_tile_memory_utilization=_safe_fraction(
            mem_allocated_bytes,
            mem_tiles_with_buffers * MEM_TILE_LOCAL_MEMORY_BYTES,
        ),
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
    )


__all__ = [
    "AIE_TILE_LOCAL_MEMORY_BYTES",
    "MEM_TILE_LOCAL_MEMORY_BYTES",
    "ResourceUsage",
    "SHIM_DMA_CHANNELS_PER_DIRECTION",
    "parse_input_physical_mlir",
]
