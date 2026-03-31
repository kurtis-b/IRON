# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_layer_spec_is_hidden_states_only():
    spec = TransformerLayerSpec()
    assert spec.hidden_size == 768
    assert "input_boundary" not in TransformerLayerSpec.__dataclass_fields__
    assert spec.block1_topology_id is None
    assert spec.block2_topology_id is None
    assert spec.block3_topology_id is None

    with pytest.raises(TypeError):
        TransformerLayerSpec(unsupported_boundary="post_projection")


def test_layer_spec_round_trips_block_topology_overrides():
    spec = TransformerLayerSpec(
        block1_topology_id="m64_k64_n16_c8_ps1_pe1",
        block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
        block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
    )
    rebuilt = TransformerLayerSpec.from_dict(spec.to_dict())

    assert rebuilt.block1_topology_id == spec.block1_topology_id
    assert rebuilt.block2_topology_id == spec.block2_topology_id
    assert rebuilt.block3_topology_id == spec.block3_topology_id
