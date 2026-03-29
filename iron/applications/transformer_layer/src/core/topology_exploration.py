# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import islice, product

from iron.operators.addnorm_ffn_addnorm.design import (
    addnorm_ffn_addnorm_practical_topologies,
)
from iron.operators.mha_out_proj.design import mha_out_proj_practical_topologies
from iron.operators.qkv_proj.design import qkv_proj_practical_topologies

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
) -> dict[str, tuple[dict[str, str], ...]]:
    head_dim = spec.attention_head_size

    block1 = _block_catalog(
        requested_topology_id=spec.block1_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block1"],
        candidates=_limit_candidates(
            tuple(
                {
                    "topology_id": str(candidate["topology_id"]),
                    "topology_family": str(candidate["topology_family"]),
                }
                for candidate in qkv_proj_practical_topologies(
                    seq_len=spec.seq_len,
                    hidden_size=spec.hidden_size,
                    num_heads=spec.num_attention_heads,
                )
            ),
            limit=max_block1_candidates,
        ),
    )
    block2 = _block_catalog(
        requested_topology_id=spec.block2_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block2"],
        candidates=_limit_candidates(
            tuple(
                {
                    "topology_id": str(candidate["topology_id"]),
                    "topology_family": str(candidate["topology_family"]),
                }
                for candidate in mha_out_proj_practical_topologies(
                    seq_len=spec.seq_len,
                    num_heads=spec.num_attention_heads,
                    head_dim=head_dim,
                )
            ),
            limit=max_block2_candidates,
        ),
    )
    block3 = _block_catalog(
        requested_topology_id=spec.block3_topology_id,
        runtime_family=_RUNTIME_BLOCK_TOPOLOGY_FAMILIES["block3"],
        candidates=_limit_candidates(
            tuple(
                {
                    "topology_id": str(candidate["topology_id"]),
                    "topology_family": str(candidate["topology_family"]),
                }
                for candidate in addnorm_ffn_addnorm_practical_topologies(
                    seq_len=spec.seq_len,
                    hidden_size=spec.hidden_size,
                    intermediate_size=spec.intermediate_size,
                )
            ),
            limit=max_block3_candidates,
        ),
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
) -> tuple[dict[str, str], ...]:
    catalog = practical_block_topology_catalog(
        spec,
        max_block1_candidates=max_block1_candidates,
        max_block2_candidates=max_block2_candidates,
        max_block3_candidates=max_block3_candidates,
    )
    combinations = (
        {
            "block1_topology_id": block1["topology_id"],
            "block1_topology_family": block1["topology_family"],
            "block2_topology_id": block2["topology_id"],
            "block2_topology_family": block2["topology_family"],
            "block3_topology_id": block3["topology_id"],
            "block3_topology_family": block3["topology_family"],
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
    candidates: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    if requested_topology_id is not None:
        return (
            {
                "topology_id": requested_topology_id,
                "topology_family": runtime_family,
            },
        )
    return candidates


def _limit_candidates(
    candidates: tuple[dict[str, str], ...],
    *,
    limit: int | None,
) -> tuple[dict[str, str], ...]:
    if limit is None:
        return candidates
    return candidates[:limit]
