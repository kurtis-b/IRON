# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from iron.applications.transformer_layer.npu_inference import build_pattern
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.utils import (
    make_in_process_npu_metadata,
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)
from iron.operators.mha_out_proj.op import _pack_qkv_head_major
from iron.operators.qkv_proj.op import AIEQKVProj


def test_build_pattern_supports_thesis_modes():
    spec = TransformerLayerSpec(seq_len=64)
    for execution_mode in (
        "dataflow",
        "runlist",
        "gemm_offload",
        "block1_qkv_proj",
        "block2_mha_out_proj",
        "block3_addnorm_ffn_addnorm",
        "gemm_offload_gemm_sequence",
        "runlist_gemm_sequence",
    ):
        pattern = build_pattern(execution_mode, spec)
        assert getattr(pattern, "pattern_label")


def test_block_patterns_prepare_hidden_state_inputs():
    spec = TransformerLayerSpec(seq_len=64)
    weights = make_synthetic_layer_weights(spec, seed=1)
    inputs = make_synthetic_layer_inputs(spec, seed=2)
    for execution_mode in (
        "block1_qkv_proj",
        "block2_mha_out_proj",
        "block3_addnorm_ffn_addnorm",
    ):
        pattern = build_pattern(execution_mode, spec)
        pattern.assign_weights(weights)
        pattern.prepare_benchmark_inputs(inputs)


def test_make_in_process_npu_metadata_formats_common_fields():
    metadata = make_in_process_npu_metadata(
        compile_setup_time_sec=0.25,
        dispatch_count=7,
        unique_instruction_binary_count=3,
        unique_xclbin_count=2,
    )
    assert metadata == {
        "compile_setup_time_ms": 250.0,
        "npu_dispatch_count": 7,
        "npu_unique_instruction_binary_count": 3,
        "npu_unique_xclbin_count": 2,
        "process_model": "in_process",
    }


def test_block1_contract_reshapes_projection_outputs_to_head_major():
    matrix = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    head_major = AIEQKVProj._to_head_major(
        matrix,
        seq_len=3,
        hidden_size=8,
        num_heads=2,
    )

    expected = matrix.view(3, 2, 4).permute(1, 0, 2).contiguous()
    assert head_major.shape == (2, 3, 4)
    assert torch.equal(head_major, expected)


def test_block2_contract_packs_head_major_qkv_into_runtime_qkv_layout():
    q = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    k = torch.arange(24, 48, dtype=torch.bfloat16).reshape(2, 3, 4)
    v = torch.arange(48, 72, dtype=torch.bfloat16).reshape(2, 3, 4)

    packed = _pack_qkv_head_major(q, k, v, seq_len=3, embed_sz=8)

    expected = torch.cat(
        (
            q.permute(1, 0, 2).contiguous().view(3, 8),
            k.permute(1, 0, 2).contiguous().view(3, 8),
            v.permute(1, 0, 2).contiguous().view(3, 8),
        ),
        dim=0,
    )
    assert packed.shape == (9, 8)
    assert np.array_equal(packed.astype(np.float32), expected.float().numpy())
