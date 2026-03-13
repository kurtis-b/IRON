# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aie import ir  # type: ignore
from aie.dialects._aie_enum_gen import ObjectFifoPort  # type: ignore
from aie.dialects.aie import (
    memtile_row_store,
    memtile_row_store_acquire,
    memtile_row_store_release,
    tile as aie_tile,
)
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron.device import Tile
from aie.iron.resolvable import Resolvable


class MemTileRowStore(Resolvable):
    def __init__(
        self,
        obj_type,
        compute_tile: Tile,
        mem_tile: Tile,
        part_count: int,
        name: str,
        buffer_count: int = 1,
        compute_buffer_count: int = 1,
        compute_produce_buffer_count: int = 0,
        compute_consume_buffer_count: int = 0,
        compute_mm2s_channel: int = 0,
        compute_s2mm_channel: int = 0,
        memtile_ingress_channel: int = 0,
        memtile_egress_channel: int = 1,
    ):
        if part_count < 1:
            raise ValueError(f"part_count must be >= 1, got {part_count}")
        if buffer_count < 1:
            raise ValueError(f"buffer_count must be >= 1, got {buffer_count}")
        if compute_buffer_count < 1:
            raise ValueError(
                "compute_buffer_count must be >= 1, "
                f"got {compute_buffer_count}"
            )
        if compute_produce_buffer_count < 0:
            raise ValueError(
                "compute_produce_buffer_count must be >= 0, "
                f"got {compute_produce_buffer_count}"
            )
        if compute_consume_buffer_count < 0:
            raise ValueError(
                "compute_consume_buffer_count must be >= 0, "
                f"got {compute_consume_buffer_count}"
            )
        if buffer_count not in (1, 2):
            raise ValueError(f"buffer_count must be 1 or 2, got {buffer_count}")
        self.obj_type = obj_type
        self.compute_tile = compute_tile
        self.mem_tile = mem_tile
        self.part_count = part_count
        self.buffer_count = buffer_count
        self.compute_buffer_count = compute_buffer_count
        self.compute_produce_buffer_count = compute_produce_buffer_count
        self.compute_consume_buffer_count = compute_consume_buffer_count
        self.name = name
        self.compute_mm2s_channel = compute_mm2s_channel
        self.compute_s2mm_channel = compute_s2mm_channel
        self.memtile_ingress_channel = memtile_ingress_channel
        self.memtile_egress_channel = memtile_egress_channel
        self._resolved = False
        self._prod = _MemTileRowStoreHandle(self, ObjectFifoPort.Produce)
        self._cons = _MemTileRowStoreHandle(self, ObjectFifoPort.Consume)

    def prod(self):
        return self._prod

    def cons(self, depth=None):
        del depth
        return self._cons

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._resolved:
            return
        compute_tile_op = self._ensure_tile_op(self.compute_tile, loc=loc, ip=ip)
        mem_tile_op = self._ensure_tile_op(self.mem_tile, loc=loc, ip=ip)
        row_store_op = memtile_row_store(
            self.name,
            compute_tile_op,
            mem_tile_op,
            self.part_count,
            ir.TypeAttr.get(np_ndarray_type_to_memref_type(self.obj_type)),
            buffer_count=self.buffer_count,
            compute_mm2s_channel=self.compute_mm2s_channel,
            compute_s2mm_channel=self.compute_s2mm_channel,
            memtile_ingress_channel=self.memtile_ingress_channel,
            memtile_egress_channel=self.memtile_egress_channel,
            loc=loc,
            ip=ip,
        )
        row_store_op.operation.attributes["compute_buffer_count"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32),
            self.compute_buffer_count,
        )
        row_store_op.operation.attributes["compute_produce_buffer_count"] = (
            ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32),
                self.compute_produce_buffer_count,
            )
        )
        row_store_op.operation.attributes["compute_consume_buffer_count"] = (
            ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32),
                self.compute_consume_buffer_count,
            )
        )
        self._resolved = True

    @staticmethod
    def _ensure_tile_op(
        placement_tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        try:
            return placement_tile.op
        except ValueError:
            tile_op = aie_tile(placement_tile.col, placement_tile.row, loc=loc, ip=ip)
            placement_tile.op = tile_op
            return tile_op


class _MemTileRowStoreHandle(Resolvable):
    def __init__(self, row_store: MemTileRowStore, port: ObjectFifoPort):
        self._row_store = row_store
        self._port = port

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._row_store.resolve(loc=loc, ip=ip)

    def acquire(self, num_elem: int):
        if num_elem != 1:
            raise ValueError(
                f"MemTileRowStore only supports acquire(1), got acquire({num_elem})"
            )
        return memtile_row_store_acquire(
            np_ndarray_type_to_memref_type(self._row_store.obj_type),
            self._port,
            self._row_store.name,
        )

    def release(self, num_elem: int):
        if num_elem != 1:
            raise ValueError(
                f"MemTileRowStore only supports release(1), got release({num_elem})"
            )
        memtile_row_store_release(
            self._port,
            self._row_store.name,
        )
