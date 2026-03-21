#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_BASE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_placements_module():
    placements_path = _BASE_DIR / "placements.py"
    spec = importlib.util.spec_from_file_location(
        "_iron_encoder_pipeline_placements",
        placements_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load encoder_pipeline placements from {placements_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_encoder_pipeline_topology_placements():
    return _load_placements_module().TOPOLOGY_PLACEMENTS


def load_supported_encoder_pipeline_topology_keys():
    return _load_placements_module().SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES


@dataclass(frozen=True)
class EncoderTopology:
    num_heads: int
    seq_len: int
    d: int
    seq_tile: int
    kv_seq_tile: int
    emb_tile: int
    ffn_tile: int
    parallel_seq: int
    parallel_heads: int
    proj_acc_depth: int
    o_proj_acc_group_size: int
    parallel_ffn: int
    ffn_intermediate_size: int
    placement: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def key(self) -> tuple[int, ...]:
        return topology_key_from_fields(
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            d=self.d,
            seq_tile=self.seq_tile,
            kv_seq_tile=self.kv_seq_tile,
            emb_tile=self.emb_tile,
            ffn_tile=self.ffn_tile,
            parallel_seq=self.parallel_seq,
            parallel_heads=self.parallel_heads,
            proj_acc_depth=self.proj_acc_depth,
            o_proj_acc_group_size=self.o_proj_acc_group_size,
            parallel_ffn=self.parallel_ffn,
            ffn_intermediate_size=self.ffn_intermediate_size,
        )

    @property
    def family_id(self) -> str:
        return f"seq{self.seq_tile}_kv{self.kv_seq_tile}"

    @property
    def legacy_alias(self) -> str:
        parts = [f"{self.parallel_seq}ps"]
        if self.parallel_heads > 1:
            parts.append(f"{self.parallel_heads}ph")
        if self.parallel_ffn > 1:
            parts.append(f"{self.parallel_ffn}pffn")
        return "_".join(parts)

    @property
    def topology_id(self) -> str:
        return (
            f"{self.family_id}__ps{self.parallel_seq}"
            f"_ph{self.parallel_heads}_pffn{self.parallel_ffn}"
        )

    @property
    def cache_signature(self) -> tuple[int, ...]:
        return (
            self.parallel_seq,
            self.parallel_heads,
            self.parallel_ffn,
            self.seq_tile,
            self.kv_seq_tile,
            self.emb_tile,
            self.ffn_tile,
            self.proj_acc_depth,
            self.o_proj_acc_group_size,
            self.ffn_intermediate_size,
        )

    @property
    def compute_tile_count(self) -> int:
        return compute_tile_count(self.placement)

    @property
    def utilization_fraction(self) -> float:
        return compute_utilization(self.compute_tile_count)

    def to_runtime_dict(self) -> dict[str, int]:
        return {
            "parallel_seq": self.parallel_seq,
            "parallel_heads": self.parallel_heads,
            "parallel_ffn": self.parallel_ffn,
            "seq_tile": self.seq_tile,
            "kv_seq_tile": self.kv_seq_tile,
            "emb_tile": self.emb_tile,
            "ffn_tile": self.ffn_tile,
            "proj_acc_depth": self.proj_acc_depth,
            "o_proj_acc_group_size": self.o_proj_acc_group_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
        }


def topology_key_from_fields(
    *,
    num_heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    ffn_tile: int,
    parallel_seq: int,
    parallel_heads: int,
    proj_acc_depth: int,
    o_proj_acc_group_size: int,
    parallel_ffn: int,
    ffn_intermediate_size: int,
) -> tuple[int, ...]:
    return (
        int(num_heads),
        int(seq_len),
        int(d),
        int(seq_tile),
        int(kv_seq_tile),
        int(emb_tile),
        int(ffn_tile),
        int(parallel_seq),
        int(parallel_heads),
        int(proj_acc_depth),
        int(o_proj_acc_group_size),
        int(parallel_ffn),
        int(ffn_intermediate_size),
    )


def topology_from_key(
    key: tuple[int, ...], placement: Mapping[str, Any] | None = None
) -> EncoderTopology:
    return EncoderTopology(
        num_heads=int(key[0]),
        seq_len=int(key[1]),
        d=int(key[2]),
        seq_tile=int(key[3]),
        kv_seq_tile=int(key[4]),
        emb_tile=int(key[5]),
        ffn_tile=int(key[6]),
        parallel_seq=int(key[7]),
        parallel_heads=int(key[8]),
        proj_acc_depth=int(key[9]),
        o_proj_acc_group_size=int(key[10]),
        parallel_ffn=int(key[11]),
        ffn_intermediate_size=int(key[12]),
        placement=placement,
    )


def topology_from_fields(
    *,
    num_heads: int,
    seq_len: int,
    d: int,
    seq_tile: int,
    kv_seq_tile: int,
    emb_tile: int,
    ffn_tile: int,
    parallel_seq: int,
    parallel_heads: int,
    proj_acc_depth: int,
    o_proj_acc_group_size: int,
    parallel_ffn: int,
    ffn_intermediate_size: int,
    placement: Mapping[str, Any] | None = None,
) -> EncoderTopology:
    return topology_from_key(
        topology_key_from_fields(
            num_heads=num_heads,
            seq_len=seq_len,
            d=d,
            seq_tile=seq_tile,
            kv_seq_tile=kv_seq_tile,
            emb_tile=emb_tile,
            ffn_tile=ffn_tile,
            parallel_seq=parallel_seq,
            parallel_heads=parallel_heads,
            proj_acc_depth=proj_acc_depth,
            o_proj_acc_group_size=o_proj_acc_group_size,
            parallel_ffn=parallel_ffn,
            ffn_intermediate_size=ffn_intermediate_size,
        ),
        placement=placement,
    )


def topology_from_placement_entry(
    key: tuple[int, ...], placement: Mapping[str, Any]
) -> EncoderTopology:
    return topology_from_key(key, placement=placement)


def topology_from_config(config, seq_len: int) -> EncoderTopology:
    aie_cfg = config.aie_config
    emb_tile = int(getattr(aie_cfg, "encoder_pipeline_emb_tile", 96))
    ffn_tile = int(getattr(aie_cfg, "encoder_pipeline_ffn_tile", 64))
    num_heads = int(config.model_config.num_attention_heads)
    d = int(config.model_config.hidden_size // num_heads)
    proj_acc_depth = int(
        getattr(
            aie_cfg,
            "encoder_pipeline_proj_acc_depth",
            config.model_config.hidden_size // emb_tile,
        )
    )
    return topology_from_fields(
        num_heads=num_heads,
        seq_len=seq_len,
        d=d,
        seq_tile=int(getattr(aie_cfg, "encoder_pipeline_seq_tile", 32)),
        kv_seq_tile=int(getattr(aie_cfg, "encoder_pipeline_kv_seq_tile", 64)),
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=int(getattr(aie_cfg, "encoder_pipeline_parallel_seq", 1)),
        parallel_heads=int(getattr(aie_cfg, "encoder_pipeline_parallel_heads", 1)),
        proj_acc_depth=proj_acc_depth,
        o_proj_acc_group_size=int(
            getattr(aie_cfg, "encoder_pipeline_o_proj_acc_group_size", 1)
        ),
        parallel_ffn=int(getattr(aie_cfg, "encoder_pipeline_parallel_ffn", 1)),
        ffn_intermediate_size=int(
            getattr(
                aie_cfg,
                "encoder_pipeline_ffn_intermediate_size",
                config.model_config.intermediate_size,
            )
        ),
    )


def all_supported_topologies() -> list[EncoderTopology]:
    placements = load_encoder_pipeline_topology_placements()
    return [
        topology_from_placement_entry(key, placements[key])
        for key in sorted(load_supported_encoder_pipeline_topology_keys())
    ]


def supported_topologies_for_config_family(
    config, seq_len: int
) -> list[EncoderTopology]:
    current = topology_from_config(config, seq_len)
    return _matching_topologies(current, include_all_families=False)


def parse_candidate_topology_filters(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    parsed = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            parsed.append(token)
    return set(parsed) if parsed else None


def filter_topologies_by_alias_or_id(
    topologies: list[EncoderTopology],
    filters: set[str] | None,
) -> list[EncoderTopology]:
    if filters is None:
        return list(topologies)
    return [
        topology
        for topology in topologies
        if topology.topology_id in filters or topology.legacy_alias in filters
    ]


def supported_topologies_for_seq_len(
    config, seq_len: int, candidate_ids: set[str] | None = None
) -> list[EncoderTopology]:
    current = topology_from_config(config, seq_len)
    matches = filter_topologies_by_alias_or_id(
        _matching_topologies(current, include_all_families=True),
        candidate_ids,
    )
    if not matches:
        requested = (
            sorted(candidate_ids) if candidate_ids is not None else "all supported"
        )
        raise RuntimeError(
            f"No supported encoder_pipeline topologies found for seq_len={seq_len} "
            f"with candidate filter {requested}"
        )
    return matches


def apply_topology_to_config(config, topology: EncoderTopology | Mapping[str, Any]):
    config.aie_config.encoder_pipeline_seq_tile = _topology_int(topology, "seq_tile")
    config.aie_config.encoder_pipeline_kv_seq_tile = _topology_int(
        topology, "kv_seq_tile"
    )
    config.aie_config.encoder_pipeline_emb_tile = _topology_int(topology, "emb_tile")
    config.aie_config.encoder_pipeline_ffn_tile = _topology_int(topology, "ffn_tile")
    config.aie_config.encoder_pipeline_parallel_seq = _topology_int(
        topology, "parallel_seq"
    )
    config.aie_config.encoder_pipeline_parallel_heads = _topology_int(
        topology, "parallel_heads"
    )
    config.aie_config.encoder_pipeline_proj_acc_depth = _topology_int(
        topology, "proj_acc_depth"
    )
    config.aie_config.encoder_pipeline_o_proj_acc_group_size = _topology_int(
        topology, "o_proj_acc_group_size"
    )
    config.aie_config.encoder_pipeline_parallel_ffn = _topology_int(
        topology, "parallel_ffn"
    )
    config.aie_config.encoder_pipeline_ffn_intermediate_size = _topology_int(
        topology, "ffn_intermediate_size"
    )
    return config


def topology_id(topology: EncoderTopology | Mapping[str, Any]) -> str:
    if isinstance(topology, EncoderTopology):
        return topology.topology_id
    return (
        f"seq{_topology_int(topology, 'seq_tile')}"
        f"_kv{_topology_int(topology, 'kv_seq_tile')}__"
        f"ps{_topology_int(topology, 'parallel_seq')}"
        f"_ph{_topology_int(topology, 'parallel_heads')}"
        f"_pffn{_topology_int(topology, 'parallel_ffn')}"
    )


def topology_signature(
    topology: EncoderTopology | Mapping[str, Any] | None,
) -> tuple[int, ...] | None:
    if topology is None:
        return None
    if isinstance(topology, EncoderTopology):
        return topology.cache_signature
    try:
        return (
            _topology_int(topology, "parallel_seq"),
            _topology_int(topology, "parallel_heads"),
            _topology_int(topology, "parallel_ffn"),
            _topology_int(topology, "seq_tile"),
            _topology_int(topology, "kv_seq_tile"),
            _topology_int(topology, "emb_tile"),
            _topology_int(topology, "ffn_tile"),
            _topology_int(topology, "proj_acc_depth"),
            _topology_int(topology, "o_proj_acc_group_size"),
            _topology_int(topology, "ffn_intermediate_size"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def topology_matches_signature(
    topology: EncoderTopology | Mapping[str, Any], signature: tuple[int, ...] | None
) -> bool:
    return signature is not None and topology_signature(topology) == signature


def topology_cache_key(config, seq_len: int) -> str:
    current = topology_from_config(config, seq_len)
    return (
        "shape:"
        f"h{current.num_heads}"
        f"_s{current.seq_len}"
        f"_d{current.d}"
        f"_st{current.seq_tile}"
        f"_kt{current.kv_seq_tile}"
        f"_et{current.emb_tile}"
        f"_ft{current.ffn_tile}"
        f"_pa{current.proj_acc_depth}"
        f"_og{current.o_proj_acc_group_size}"
        f"_is{current.ffn_intermediate_size}"
    )


def find_cached_topology(
    cache_data: Mapping[str, Any],
    config,
    seq_len: int,
    candidate_ids: set[str] | None = None,
) -> EncoderTopology | None:
    supported = supported_topologies_for_seq_len(
        config, seq_len, candidate_ids=candidate_ids
    )
    supported_by_signature = {
        topology.cache_signature: topology for topology in supported
    }
    for cache_key in (topology_cache_key(config, seq_len), str(seq_len)):
        cached = cache_data.get(cache_key)
        signature = topology_signature(cached)
        if signature in supported_by_signature:
            return supported_by_signature[signature]
    return None


def compute_tile_count(placement: Mapping[str, Any] | None) -> int:
    if not placement:
        return 0

    coords: set[tuple[int, int]] = set()
    sequence_parallel = placement.get("sequence_parallel")
    if sequence_parallel is None:
        for col in placement["mha_cols"]:
            for row in (2, 3, 4, 5):
                coords.add((int(col), row))
        _collect_tile_coords(placement["tail_tiles"], coords)
    else:
        _collect_tile_coords(sequence_parallel.get("lane_tiles"), coords)
    return len(coords)


def compute_utilization(
    compute_tile_count: int, total_compute_tiles: int = 32
) -> float:
    if total_compute_tiles <= 0:
        raise ValueError(f"total_compute_tiles must be > 0 (got {total_compute_tiles})")
    return float(compute_tile_count) / float(total_compute_tiles)


def _collect_tile_coords(value: Any, coords: set[tuple[int, int]]):
    if isinstance(value, Mapping):
        for nested_value in value.values():
            _collect_tile_coords(nested_value, coords)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(dim, int) for dim in value):
            coords.add((int(value[0]), int(value[1])))
            return
        for item in value:
            _collect_tile_coords(item, coords)


def _topology_int(
    topology: EncoderTopology | Mapping[str, Any], field_name: str
) -> int:
    if isinstance(topology, EncoderTopology):
        return int(getattr(topology, field_name))
    return int(topology[field_name])


def _matching_topologies(
    current: EncoderTopology, *, include_all_families: bool
) -> list[EncoderTopology]:
    matches = []
    for topology in all_supported_topologies():
        if (
            topology.num_heads == current.num_heads
            and topology.seq_len == current.seq_len
            and topology.d == current.d
            and topology.emb_tile == current.emb_tile
            and topology.ffn_tile == current.ffn_tile
            and topology.proj_acc_depth == current.proj_acc_depth
            and topology.o_proj_acc_group_size == current.o_proj_acc_group_size
            and topology.ffn_intermediate_size == current.ffn_intermediate_size
        ):
            if not include_all_families and (
                topology.seq_tile != current.seq_tile
                or topology.kv_seq_tile != current.kv_seq_tile
            ):
                continue
            matches.append(topology)
    matches.sort(
        key=lambda topology: (
            (
                0
                if (
                    topology.seq_tile == current.seq_tile
                    and topology.kv_seq_tile == current.kv_seq_tile
                )
                else 1
            ),
            topology.seq_tile,
            topology.kv_seq_tile,
            topology.parallel_seq,
            topology.parallel_heads,
            topology.parallel_ffn,
        )
    )
    return matches
