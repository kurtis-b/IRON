# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_roofline_estimates_block_flops_ordering():
    spec = TransformerLayerSpec(seq_len=64)
    block1 = estimate_layer_flops(spec, execution_mode="block1_qkv_proj")
    block2 = estimate_layer_flops(spec, execution_mode="block2_mha_out_proj")
    block3 = estimate_layer_flops(spec, execution_mode="block3_addnorm_ffn_addnorm")
    full = estimate_layer_flops(spec, execution_mode="dataflow")
    assert block1 > 0
    assert block2 > 0
    assert block3 > 0
    assert full == block1 + block2 + block3


def test_roofline_estimates_bytes_for_reconfig_modes_match_full_gemm_surface():
    spec = TransformerLayerSpec(seq_len=128)
    assert estimate_layer_bytes(
        spec, execution_mode="gemm_offload_gemm_sequence"
    ) == estimate_layer_bytes(spec, execution_mode="gemm_offload")
