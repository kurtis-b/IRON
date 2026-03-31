# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.src.core import (
    TransformerLayerSpec,
    practical_block_topology_catalog,
    practical_layer_topology_combinations,
)


def test_practical_block_topology_catalog_returns_ranked_block_candidates():
    spec = TransformerLayerSpec(seq_len=64)

    catalog = practical_block_topology_catalog(
        spec,
        max_block1_candidates=2,
        max_block2_candidates=2,
        max_block3_candidates=2,
    )

    assert tuple(catalog) == ("block1", "block2", "block3")
    assert len(catalog["block1"]) == 2
    assert len(catalog["block2"]) == 2
    assert len(catalog["block3"]) == 2
    assert catalog["block1"][0]["topology_family"].endswith("_practical")
    assert catalog["block2"][0]["topology_family"].endswith("_practical")
    assert catalog["block3"][0]["topology_family"].endswith("_practical")
    assert "runtime_supported" in catalog["block1"][0]
    assert "runtime_supported" in catalog["block2"][0]
    assert "runtime_supported" in catalog["block3"][0]


def test_practical_layer_topology_combinations_cross_product_block_catalogs():
    spec = TransformerLayerSpec(seq_len=64)

    combinations = practical_layer_topology_combinations(
        spec,
        max_block1_candidates=2,
        max_block2_candidates=2,
        max_block3_candidates=2,
    )

    assert len(combinations) == 8
    first = combinations[0]
    assert "block1_topology_id" in first
    assert "block2_topology_id" in first
    assert "block3_topology_id" in first
    assert "runtime_supported" in first
    assert first["block1_topology_family"].endswith("_practical")
    assert first["block2_topology_family"].endswith("_practical")
    assert first["block3_topology_family"].endswith("_practical")


def test_practical_layer_topology_combinations_respect_requested_overrides():
    spec = TransformerLayerSpec(
        seq_len=64,
        block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
    )

    combinations = practical_layer_topology_combinations(
        spec,
        max_block1_candidates=2,
        max_block2_candidates=2,
        max_block3_candidates=2,
    )

    assert len(combinations) == 4
    assert {row["block2_topology_id"] for row in combinations} == {
        "q32_kv64_e96_ps1_ph1_acc1"
    }
    assert {row["block2_topology_family"] for row in combinations} == {
        "fused_mha_out_proj"
    }


def test_practical_layer_topology_combinations_support_global_truncation():
    spec = TransformerLayerSpec(seq_len=64)

    combinations = practical_layer_topology_combinations(
        spec,
        max_block1_candidates=2,
        max_block2_candidates=2,
        max_block3_candidates=2,
        max_combinations=3,
    )

    assert len(combinations) == 3


def test_practical_catalog_runtime_only_filters_before_truncation():
    spec = TransformerLayerSpec(seq_len=512)
    full_runtime_catalog = practical_block_topology_catalog(
        spec,
        runtime_only=True,
    )

    catalog = practical_block_topology_catalog(
        spec,
        max_block1_candidates=2,
        max_block2_candidates=2,
        max_block3_candidates=2,
        runtime_only=True,
    )

    assert len(catalog["block1"]) == min(2, len(full_runtime_catalog["block1"]))
    assert len(catalog["block2"]) == min(2, len(full_runtime_catalog["block2"]))
    assert len(catalog["block3"]) == min(2, len(full_runtime_catalog["block3"]))
    assert all(bool(row["runtime_supported"]) for row in catalog["block1"])
    assert all(bool(row["runtime_supported"]) for row in catalog["block2"])
    assert all(bool(row["runtime_supported"]) for row in catalog["block3"])


def test_practical_layer_topology_combinations_runtime_only_exposes_runtime_ids():
    spec = TransformerLayerSpec(seq_len=512)

    combinations = practical_layer_topology_combinations(
        spec,
        max_block1_candidates=1,
        max_block2_candidates=1,
        max_block3_candidates=2,
        runtime_only=True,
    )

    assert combinations
    assert all(bool(row["runtime_supported"]) for row in combinations)
    assert all(row["block1_runtime_topology_id"] is not None for row in combinations)
    assert all(
        row["block1_runtime_topology_id"] == row["block1_topology_id"]
        for row in combinations
    )
    assert all(
        row["block2_runtime_topology_id"] == row["block2_topology_id"]
        for row in combinations
    )
    assert all(row["block3_runtime_topology_id"] is not None for row in combinations)
    assert any(
        row["block3_runtime_topology_id"] != row["block3_topology_id"]
        for row in combinations
    )


def test_practical_block2_catalog_prefers_real_lowered_axes():
    spec = TransformerLayerSpec(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        seq_len=512,
    )

    catalog = practical_block_topology_catalog(
        spec,
        max_block1_candidates=1,
        max_block2_candidates=2,
        max_block3_candidates=1,
    )

    block2_ids = [row["topology_id"] for row in catalog["block2"]]

    assert len(block2_ids) == 2
    assert all("_acc8" in topology_id for topology_id in block2_ids)
    assert "q32_kv128_e96_ps8_ph1_acc8" in block2_ids
    assert "q32_kv128_e96_ps4_ph2_acc8" in block2_ids
