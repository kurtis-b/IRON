# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import islice, product

from iron.operators.addnorm_ffn_addnorm.design import (
    addnorm_ffn_addnorm_topologies,
    addnorm_ffn_addnorm_practical_topologies,
)
from iron.operators.mha_out_proj.design import (
    mha_out_proj_practical_topologies,
    mha_out_proj_topologies,
)
from iron.operators.qkv_proj.design import (
    qkv_proj_practical_topologies,
    qkv_proj_topologies,
)

from .layer_spec import TransformerLayerSpec

_RUNTIME_BLOCK_TOPOLOGY_FAMILIES = {
    "block1": "shared_runtime_qkv_proj",
    "block2": "fused_mha_out_proj",
    "block3": "pipelined_addnorm_ffn_addnorm",
}


def practical_block_topology_catalog(
    spec: TransformerLayerSpec,
    *,
    max_block1_candidates: int | None = None,
    max_block2_candidates: int | None = None,
    max_block3_candidates: int | None = None,
    runtime_only: bool = False,
) -> dict[str, tuple[dict[str, object], ...]]:
    head_dim = spec.attention_head_size

    block1_supported_ids = {
        str(candidate["topology_id"])
        for candidate in qkv_proj_topologies(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            num_heads=spec.num_attention_heads,
        )
    }
    block1_runtime_signature_map = {
        (
            int(candidate["tile_m"]),
            int(candidate["tile_k"]),
            int(candidate["tile_n"]),
            int(candidate["num_aie_columns"]),
            int(candidate["parallel_seq"]),
            int(candidate["parallel_heads"]),
            int(candidate["parallel_head_dim"]),
        ): (
            str(candidate["topology_id"]),
            str(candidate["topology_family"]),
        )
        for candidate in qkv_proj_topologies(
            seq_len=spec.seq_len,
            hidden_size=spec.hidden_size,
            num_heads=spec.num_attention_heads,
        )
    }
    block1 = _block_catalog(
        requested_topology_id=spec.block1_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block1"],
        supported_ids=block1_supported_ids,
        candidates=tuple(
            _block1_catalog_candidate(
                candidate,
                runtime_signature_map=block1_runtime_signature_map,
            )
            for candidate in qkv_proj_practical_topologies(
                seq_len=spec.seq_len,
                hidden_size=spec.hidden_size,
                num_heads=spec.num_attention_heads,
            )
        ),
        limit=max_block1_candidates,
        runtime_only=runtime_only,
    )
    block2_supported_ids = {
        str(candidate["topology_id"])
        for candidate in mha_out_proj_topologies(
            num_heads=spec.num_attention_heads,
            head_dim=head_dim,
        )
    }
    block2 = _block_catalog(
        requested_topology_id=spec.block2_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block2"],
        supported_ids=block2_supported_ids,
        candidates=tuple(
            {
                "topology_id": str(candidate["topology_id"]),
                "topology_family": str(candidate["topology_family"]),
                "runtime_supported": str(candidate["topology_id"])
                in block2_supported_ids,
                "runtime_topology_id": (
                    str(candidate["topology_id"])
                    if str(candidate["topology_id"]) in block2_supported_ids
                    else None
                ),
                "runtime_topology_family": (
                    _RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block2"]
                    if str(candidate["topology_id"]) in block2_supported_ids
                    else None
                ),
            }
            for candidate in mha_out_proj_practical_topologies(
                seq_len=spec.seq_len,
                num_heads=spec.num_attention_heads,
                head_dim=head_dim,
            )
        ),
        limit=max_block2_candidates,
        runtime_only=runtime_only,
    )
    block3_supported_ids = {
        str(candidate["topology_id"])
        for candidate in addnorm_ffn_addnorm_topologies(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
        )
    }
    block3_runtime_signature_map = {
        (
            int(candidate["tile_k"]),
            int(candidate["tile_n"]),
            int(candidate["down_proj_depth"]),
            int(candidate["num_aie_columns"]),
            int(candidate["parallel_seq"]),
            int(candidate["parallel_int_dim"]),
            int(candidate["gelu_stage"]),
        ): (
            str(candidate["topology_id"]),
            str(candidate["topology_family"]),
        )
        for candidate in addnorm_ffn_addnorm_topologies(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
        )
    }
    block3 = _block_catalog(
        requested_topology_id=spec.block3_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block3"],
        supported_ids=block3_supported_ids,
        candidates=tuple(
            _block3_catalog_candidate(
                candidate,
                runtime_signature_map=block3_runtime_signature_map,
            )
            for candidate in addnorm_ffn_addnorm_practical_topologies(
                seq_len=spec.seq_len,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
            )
        ),
        limit=max_block3_candidates,
        runtime_only=runtime_only,
    )
    return {
        "block1": block1,
        "block2": block2,
        "block3": block3,
    }


def practical_layer_topology_combinations(
    spec: TransformerLayerSpec,
    *,
    max_block1_candidates: int | None = None,
    max_block2_candidates: int | None = None,
    max_block3_candidates: int | None = None,
    max_combinations: int | None = None,
    runtime_only: bool = False,
) -> tuple[dict[str, object], ...]:
    catalog = practical_block_topology_catalog(
        spec,
        max_block1_candidates=max_block1_candidates,
        max_block2_candidates=max_block2_candidates,
        max_block3_candidates=max_block3_candidates,
        runtime_only=runtime_only,
    )
    combinations = (
        {
            "block1_topology_id": block1["topology_id"],
            "block1_topology_family": block1["topology_family"],
            "block1_runtime_supported": bool(block1["runtime_supported"]),
            "block1_runtime_topology_id": block1["runtime_topology_id"],
            "block1_runtime_topology_family": block1["runtime_topology_family"],
            "block2_topology_id": block2["topology_id"],
            "block2_topology_family": block2["topology_family"],
            "block2_runtime_supported": bool(block2["runtime_supported"]),
            "block2_runtime_topology_id": block2["runtime_topology_id"],
            "block2_runtime_topology_family": block2["runtime_topology_family"],
            "block3_topology_id": block3["topology_id"],
            "block3_topology_family": block3["topology_family"],
            "block3_runtime_supported": bool(block3["runtime_supported"]),
            "block3_runtime_topology_id": block3["runtime_topology_id"],
            "block3_runtime_topology_family": block3["runtime_topology_family"],
            "runtime_supported": bool(
                block1["runtime_supported"]
                and block2["runtime_supported"]
                and block3["runtime_supported"]
            ),
        }
        for block1, block2, block3 in product(
            catalog["block1"],
            catalog["block2"],
            catalog["block3"],
        )
    )
    if max_combinations is not None:
        combinations = islice(combinations, max_combinations)
    return tuple(combinations)


def _block_catalog(
    *,
    requested_topology_id: str | None,
    runtime_family: str,
    supported_ids: set[str],
    candidates: tuple[dict[str, object], ...],
    limit: int | None,
    runtime_only: bool,
) -> tuple[dict[str, object], ...]:
    if requested_topology_id is not None:
        return (
            {
                "topology_id": requested_topology_id,
                "topology_family": runtime_family,
                "runtime_supported": requested_topology_id in supported_ids,
                "runtime_topology_id": requested_topology_id,
                "runtime_topology_family": runtime_family,
            },
        )
    filtered = candidates
    if runtime_only:
        filtered = tuple(
            candidate for candidate in filtered if bool(candidate["runtime_supported"])
        )
    return _limit_candidates(filtered, limit=limit)


def _limit_candidates(
    candidates: tuple[dict[str, object], ...],
    *,
    limit: int | None,
) -> tuple[dict[str, object], ...]:
    if limit is None:
        return candidates
    return candidates[:limit]


def _block3_catalog_candidate(
    candidate: dict[str, int | str],
    *,
    runtime_signature_map: dict[tuple[int, ...], tuple[str, str]],
) -> dict[str, object]:
    runtime_signature = (
        int(candidate["tile_k"]),
        int(candidate["tile_n"]),
        int(candidate["down_proj_depth"]),
        int(candidate["num_aie_columns"]),
        int(candidate["parallel_seq"]),
        int(candidate["parallel_int_dim"]),
        int(candidate["gelu_stage"]),
    )
    runtime_match = runtime_signature_map.get(runtime_signature)
    return {
        "topology_id": str(candidate["topology_id"]),
        "topology_family": str(candidate["topology_family"]),
        "runtime_supported": runtime_match is not None,
        "runtime_topology_id": runtime_match[0] if runtime_match is not None else None,
        "runtime_topology_family": (
            runtime_match[1] if runtime_match is not None else None
        ),
    }


def _block1_catalog_candidate(
    candidate: dict[str, int | str],
    *,
    runtime_signature_map: dict[tuple[int, ...], tuple[str, str]],
) -> dict[str, object]:
    runtime_signature = (
        int(candidate["tile_m"]),
        int(candidate["tile_k"]),
        int(candidate["tile_n"]),
        int(candidate["num_aie_columns"]),
        int(candidate["parallel_seq"]),
        int(candidate["parallel_heads"]),
        int(candidate["parallel_head_dim"]),
    )
    runtime_match = runtime_signature_map.get(runtime_signature)
    return {
        "topology_id": str(candidate["topology_id"]),
        "topology_family": str(candidate["topology_family"]),
        "runtime_supported": runtime_match is not None,
        "runtime_topology_id": runtime_match[0] if runtime_match is not None else None,
        "runtime_topology_family": (
            runtime_match[1] if runtime_match is not None else None
        ),
    }
