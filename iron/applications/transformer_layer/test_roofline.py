# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
)
from iron.applications.transformer_layer.peak_reference import BackendPeakReference
from iron.applications.transformer_layer.src.analysis import (
    BackendPeakReference as StructuredBackendPeakReference,
)
from iron.applications.transformer_layer.src.analysis import (
    estimate_layer_bytes as structured_estimate_layer_bytes,
)
from iron.applications.transformer_layer.src.analysis import (
    estimate_layer_flops as structured_estimate_layer_flops,
)
from iron.applications.transformer_layer.src.analysis import (
    SUPPORT_MATRIX_FIELD_ORDER as structured_support_matrix_field_order,
)
from iron.applications.transformer_layer.src.analysis import (
    summarize_support_rows as structured_summarize_support_rows,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.support_matrix import (
    summarize_support_rows,
)


def test_restructured_analysis_package_preserves_legacy_imports():
    assert estimate_layer_flops is structured_estimate_layer_flops
    assert estimate_layer_bytes is structured_estimate_layer_bytes
    assert BackendPeakReference is StructuredBackendPeakReference
    assert summarize_support_rows is structured_summarize_support_rows


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


def test_summarize_support_rows_preserves_block_topology_metadata():
    rows = summarize_support_rows(
        [
            {
                "study_id": "study",
                "study_case_id": "case",
                "study_case_label": "case",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "seq_len": 64,
                "execution_mode": "dataflow",
                "block1_topology_id": "m64_k64_n16_c8_ps1_pe1",
                "block1_topology_family": "shared_runtime_qkv_proj",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block2_topology_family": "fused_mha_out_proj",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "block3_topology_family": "pipelined_addnorm_ffn_addnorm",
                "exploration_block1_topology_id": "m32_k256_n24_c8_ps2_pe4",
                "exploration_block1_topology_family": "shared_runtime_qkv_proj_practical",
                "exploration_block2_topology_id": "q32_kv64_e96_ps1_ph6_acc1",
                "exploration_block2_topology_family": "fused_mha_out_proj_practical",
                "exploration_block3_topology_id": "m16_k384_n16_c8_ps4_pi3_d2_g1",
                "exploration_block3_topology_family": "pipelined_addnorm_ffn_addnorm_practical",
                "run_status": "unsupported",
                "failure_category": "unsupported_topology_or_placement",
            }
        ]
    )

    row = rows[0]
    assert row["block1_topology_id"] == "m64_k64_n16_c8_ps1_pe1"
    assert row["block2_topology_family"] == "fused_mha_out_proj"
    assert row["block3_topology_id"] == "m32_k96_n64_ps4_pi3_d8_g1"
    assert row["exploration_block1_topology_id"] == "m32_k256_n24_c8_ps2_pe4"
    assert row["exploration_block2_topology_family"] == "fused_mha_out_proj_practical"
    assert row["exploration_block3_topology_id"] == "m16_k384_n16_c8_ps4_pi3_d2_g1"
    assert "block1_topology_id" in structured_support_matrix_field_order
    assert "block3_topology_family" in structured_support_matrix_field_order
    assert "exploration_block1_topology_id" in structured_support_matrix_field_order
    assert "exploration_block3_topology_family" in structured_support_matrix_field_order
