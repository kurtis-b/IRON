#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.common import AIEOperatorConstraintError
from iron.common.test_utils import run_test
from iron.operators.encoder_pipeline.design import (
    _normalize_base_placement,
    _normalize_slot_coords,
    _normalize_slot_mem_cols,
    _validate_sequence_parallel_placement,
    build_q_block_schedule,
    compute_ffn_col_group_count,
    compute_num_kv_seq_blocks,
    compute_q_blocks_per_lane,
    compute_qkv_head_blocks_per_parallel_head,
    encoder_pipeline,
)
from iron.operators.encoder_pipeline.op import AIEEncoderPipeline
from iron.operators.encoder_pipeline.reference import generate_golden_reference
from iron.operators.encoder_pipeline.topology import (
    all_supported_topologies,
    filter_topologies_by_alias_or_id,
    load_encoder_pipeline_topology_placements,
    topology_from_key,
)

REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.05
DEFAULT_SCALED_SEQ_LENS = tuple(1 << exp for exp in range(6, 14))
OPTIONAL_SCALED_SEQ_LENS = (16384,)
INCLUDE_SEQ16384 = os.environ.get("IRON_ENCODER_PIPELINE_INCLUDE_SEQ16384") == "1"
SCALED_SEQ_LENS = (
    DEFAULT_SCALED_SEQ_LENS + OPTIONAL_SCALED_SEQ_LENS
    if INCLUDE_SEQ16384
    else DEFAULT_SCALED_SEQ_LENS
)
BASE_TOPOLOGY = (64, 12, 3072, 32, 64, 96, 64)
MIRRORED_BASE_TOPOLOGY = (64, 12, 3072, 64, 32, 96, 64)
MIRRORED_LARGE_BASE_TOPOLOGY = (64, 16, 4096, 64, 32, 128, 64)
TOPOLOGY_CASES = (
    ("", (1, 1, 8, 1, 1, 1)),
    ("_2ps", (2, 1, 8, 1, 1, 1)),
    ("_2ps_2ph", (2, 2, 8, 1, 1, 1)),
    ("_2ps_2pffn", (2, 1, 8, 1, 1, 2)),
    ("_2ps_2ph_2pffn", (2, 2, 8, 1, 1, 2)),
    ("_2ps_4pffn", (2, 1, 8, 1, 1, 4)),
    ("_4ps", (4, 1, 8, 1, 1, 1)),
    ("_2ph", (1, 2, 8, 1, 1, 1)),
    ("_4ph", (1, 4, 8, 1, 1, 1)),
    ("_2pffn", (1, 1, 8, 1, 1, 2)),
    ("_2ph_2pffn", (1, 2, 8, 1, 1, 2)),
    ("_4pffn", (1, 1, 8, 1, 1, 4)),
    ("_2ph_4pffn", (1, 2, 8, 1, 1, 4)),
)


def topology_name(base_topology: tuple[int, ...], suffix: str) -> str:
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        base_topology
    )
    return (
        f"{d}d_{num_heads}h_{ffn_intermediate_size}ffn_{seq_tile}q_"
        f"{kv_seq_tile}kv_{emb_tile}emb_{ffn_tile}ffnt{suffix}"
    )


def execution_matrix_supported(base_topology: tuple[int, ...]) -> bool:
    return base_topology != MIRRORED_LARGE_BASE_TOPOLOGY


class DummyContext:
    def __init__(self, tmp_path):
        self.operators = []
        self.static_data_pool = {}
        self.base_dir = Path(__file__).resolve().parents[3]
        self.build_dir = tmp_path
        self.device_manager = SimpleNamespace(
            device_str=lambda: "npu2",
            device_type="npu2",
        )

    def register_operator(self, operator, skip_add_to_list=False):
        operator.context = self
        if not skip_add_to_list:
            self.operators.append(operator)


def encoder_pipeline_common_kwargs(context, **overrides):
    kwargs = dict(
        num_heads=12,
        seq_len=64,
        d=64,
        seq_tile=32,
        kv_seq_tile=64,
        emb_tile=96,
        ffn_tile=64,
        parallel_seq=2,
        parallel_heads=2,
        proj_acc_depth=8,
        o_proj_acc_group_size=1,
        ffn_down_acc_group_size=1,
        nB_tiles_distributed=2,
        ffn_intermediate_size=3072,
        context=context,
    )
    kwargs.update(overrides)
    return kwargs


def render_encoder_pipeline_mlir(**overrides):
    kwargs = dict(
        heads=12,
        seq_len=64,
        d=64,
        seq_tile=32,
        kv_seq_tile=64,
        emb_tile=96,
        ffn_tile=64,
        proj_acc_depth=8,
        parallel_heads=1,
        parallel_seq=1,
        emulate_bf16_mmul_with_bfp16=True,
        kernel_archive="dummy.a",
        o_proj_acc_group_size=1,
        ffn_down_acc_group_size=1,
        nB_tiles_distributed=1,
        ffn_intermediate_size=3072,
    )
    kwargs.update(overrides)
    return str(encoder_pipeline(**kwargs))


def test_artifacts_share_xclbin_across_ln_weight_variants(tmp_path):
    common_kwargs = encoder_pipeline_common_kwargs(DummyContext(tmp_path))
    op_a = AIEEncoderPipeline(
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        **common_kwargs,
    )
    op_b = AIEEncoderPipeline(
        ln1_weight=torch.full((768,), 2, dtype=torch.bfloat16),
        ln2_weight=torch.full((768,), 3, dtype=torch.bfloat16),
        **common_kwargs,
    )

    xclbin_a, insts_a = op_a.get_artifacts(prefix="encoder_pipeline")
    xclbin_b, insts_b = op_b.get_artifacts(prefix="encoder_pipeline")

    assert xclbin_a.path == xclbin_b.path
    assert insts_a.path != insts_b.path
    assert insts_a.xclbin_input is xclbin_a
    assert insts_b.xclbin_input is xclbin_b


def test_layout_overrides_expand_xclbin_design_space(tmp_path):
    common_kwargs = encoder_pipeline_common_kwargs(DummyContext(tmp_path))
    default_op = AIEEncoderPipeline(**common_kwargs)
    override_op = AIEEncoderPipeline(o_proj_fifo_depth=1, **common_kwargs)

    default_xclbin, _ = default_op.get_artifacts(prefix="encoder_pipeline")
    override_xclbin, _ = override_op.get_artifacts(prefix="encoder_pipeline")

    assert default_xclbin.path != override_xclbin.path
    assert default_xclbin.depends[0].callback_kwargs["o_proj_fifo_depth"] == 2
    assert override_xclbin.depends[0].callback_kwargs["o_proj_fifo_depth"] == 1


def test_explicit_addnorm_toggle_expands_xclbin_design_space(tmp_path):
    common_kwargs = encoder_pipeline_common_kwargs(DummyContext(tmp_path))
    default_op = AIEEncoderPipeline(**common_kwargs)
    disabled_op = AIEEncoderPipeline(
        use_fused_replayed_addnorm=False,
        **common_kwargs,
    )

    default_xclbin, _ = default_op.get_artifacts(prefix="encoder_pipeline")
    disabled_xclbin, _ = disabled_op.get_artifacts(prefix="encoder_pipeline")

    assert default_xclbin.path != disabled_xclbin.path
    assert default_xclbin.depends[0].callback_kwargs["use_fused_replayed_addnorm"]
    assert (
        disabled_xclbin.depends[0].callback_kwargs["use_fused_replayed_addnorm"]
        is False
    )


def test_data_movement_overrides_expand_xclbin_design_space(tmp_path):
    common_kwargs = encoder_pipeline_common_kwargs(
        DummyContext(tmp_path),
        seq_len=128,
        parallel_seq=4,
        parallel_heads=1,
        nB_tiles_distributed=1,
    )
    default_op = AIEEncoderPipeline(**common_kwargs)
    qr_split_disabled = AIEEncoderPipeline(use_unified_qr_split=False, **common_kwargs)
    transport_groups_disabled = AIEEncoderPipeline(
        use_transport_groups=False,
        **common_kwargs,
    )

    default_xclbin, _ = default_op.get_artifacts(prefix="encoder_pipeline")
    qr_split_disabled_xclbin, _ = qr_split_disabled.get_artifacts(
        prefix="encoder_pipeline"
    )
    transport_groups_disabled_xclbin, _ = transport_groups_disabled.get_artifacts(
        prefix="encoder_pipeline"
    )

    assert default_xclbin.path != qr_split_disabled_xclbin.path
    assert default_xclbin.path != transport_groups_disabled_xclbin.path
    assert (
        qr_split_disabled_xclbin.depends[0].callback_kwargs["use_unified_qr_split"]
        is False
    )
    assert (
        transport_groups_disabled_xclbin.depends[0].callback_kwargs[
            "use_transport_groups"
        ]
        is False
    )


def test_combine_qkv_projection_parameters_returns_matmul_ready_weight_and_bias():
    query_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    key_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    value_weight = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    query_bias = torch.tensor([0.1, 0.2])
    key_bias = torch.tensor([0.3, 0.4])
    value_bias = torch.tensor([0.5, 0.6])

    combined_weight, combined_bias = (
        AIEEncoderPipeline.combine_qkv_projection_parameters(
            query_weight,
            key_weight,
            value_weight,
            query_bias,
            key_bias,
            value_bias,
        )
    )

    assert tuple(combined_weight.shape) == (2, 6)
    assert tuple(combined_bias.shape) == (6,)
    assert torch.equal(
        combined_weight,
        torch.tensor(
            [
                [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ]
        ),
    )
    assert torch.equal(
        combined_bias,
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )


def test_pack_projected_qkv_rows_matches_runtime_layout():
    projected_qkv = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]
    )

    packed_qkv = AIEEncoderPipeline.pack_projected_qkv_rows(
        projected_qkv,
        seq_len=2,
        embed_sz=2,
    )

    assert torch.equal(
        packed_qkv,
        torch.tensor(
            [
                [1.0, 2.0],
                [7.0, 8.0],
                [3.0, 4.0],
                [9.0, 10.0],
                [5.0, 6.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_pack_qkv_tensors_matches_runtime_layout():
    q = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    k = q + 10.0
    v = q + 20.0

    packed_qkv = AIEEncoderPipeline.pack_qkv_tensors(
        q,
        k,
        v,
        seq_len=2,
        num_heads=2,
        d=2,
    )

    assert torch.equal(
        packed_qkv,
        torch.tensor(
            [
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 7.0, 8.0],
                [11.0, 12.0, 15.0, 16.0],
                [13.0, 14.0, 17.0, 18.0],
                [21.0, 22.0, 25.0, 26.0],
                [23.0, 24.0, 27.0, 28.0],
            ]
        ),
    )


def test_split_combined_qkv_projection_parameters_recovers_split_weights_and_biases():
    combined_weight = torch.arange(12, dtype=torch.bfloat16).view(2, 6)
    combined_bias = torch.arange(6, dtype=torch.bfloat16)

    q_weight, k_weight, v_weight, q_bias, k_bias, v_bias = (
        AIEEncoderPipeline.split_combined_qkv_projection_parameters(
            combined_weight,
            combined_bias,
        )
    )

    assert torch.equal(q_weight, combined_weight[:, :2])
    assert torch.equal(k_weight, combined_weight[:, 2:4])
    assert torch.equal(v_weight, combined_weight[:, 4:6])
    assert torch.equal(q_bias, combined_bias[:2])
    assert torch.equal(k_bias, combined_bias[2:4])
    assert torch.equal(v_bias, combined_bias[4:6])


def test_pack_projection_bias_tiles_match_runtime_shapes():
    q_bias = torch.arange(4, dtype=torch.bfloat16)
    k_bias = torch.arange(4, dtype=torch.bfloat16)
    v_bias = torch.arange(4, dtype=torch.bfloat16)

    q_tiles = AIEEncoderPipeline.pack_q_projection_bias_tiles(
        q_bias,
        seq_tile=2,
        num_heads=2,
        d=2,
    )
    k_tiles = AIEEncoderPipeline.pack_k_projection_bias_tiles(
        k_bias,
        kv_seq_tile=4,
        num_heads=2,
        d=2,
    )
    v_tiles = AIEEncoderPipeline.pack_v_projection_bias_tiles(
        v_bias,
        kv_seq_tile=4,
        num_heads=2,
        d=2,
    )

    assert tuple(q_tiles.shape) == (4, 2)
    assert tuple(k_tiles.shape) == (4, 4)
    assert tuple(v_tiles.shape) == (8, 2)
    assert torch.equal(
        q_tiles[:2], torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.bfloat16)
    )
    assert torch.equal(
        k_tiles[:2],
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16
        ),
    )
    assert torch.equal(
        v_tiles[:4], torch.tensor([[0.0, 1.0]] * 4, dtype=torch.bfloat16)
    )


def test_project_hidden_states_to_packed_qkv_uses_combined_parameters(tmp_path):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(DummyContext(tmp_path)),
        skip_add_to_list=True,
    )
    operator.qkv_proj_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    operator.qkv_proj_bias = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        dtype=torch.bfloat16,
    )
    operator.embed_sz = 2
    operator.seq_len = 2
    hidden_states = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.bfloat16,
    )

    packed_qkv = operator.project_hidden_states_to_packed_qkv(hidden_states)

    expected_projected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            [3.0, 4.0, 5.0, 4.0, 8.0, 10.0],
        ],
        dtype=torch.bfloat16,
    )
    expected_packed = AIEEncoderPipeline.pack_projected_qkv_rows(
        expected_projected,
        seq_len=2,
        embed_sz=2,
    )
    assert torch.equal(packed_qkv, expected_packed)


def test_staged_hidden_states_mode_supports_sequence_parallel_runtime_contract(
    tmp_path,
):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=2,
            parallel_heads=1,
            nB_tiles_distributed=1,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    operator.xclbin_artifact = SimpleNamespace(kernel_name="encoder_pipeline_kernel")
    operator.insts_artifact = object()

    operator.set_up_runtime()

    assert "QKV" not in operator.buffers
    assert "X" in operator.buffers
    assert operator.runlist == [
        ("encoder_pipeline", "W_ATTN", "X", "OR", "B_Up", "B_Down")
    ]
    assert operator._or_buffer_shape() == (5 * operator.seq_len, operator.embed_sz)


def test_staged_hidden_states_mode_supports_sequence_parallel_parallel_heads_two_runtime_contract(
    tmp_path,
):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=2,
            parallel_heads=2,
            nB_tiles_distributed=1,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    operator.xclbin_artifact = SimpleNamespace(kernel_name="encoder_pipeline_kernel")
    operator.insts_artifact = object()

    operator.set_up_runtime()

    assert "QKV" not in operator.buffers
    assert "X" in operator.buffers
    assert operator.runlist == [
        ("encoder_pipeline", "W_ATTN", "X", "OR", "B_Up", "B_Down")
    ]
    assert operator._or_buffer_shape() == (5 * operator.seq_len, operator.embed_sz)


def test_staged_hidden_states_mode_supports_parallel_heads_two_runtime_contract(
    tmp_path,
):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=1,
            parallel_heads=2,
            nB_tiles_distributed=1,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    operator.xclbin_artifact = SimpleNamespace(kernel_name="encoder_pipeline_kernel")
    operator.insts_artifact = object()

    operator.set_up_runtime()

    assert "QKV" not in operator.buffers
    assert "X" in operator.buffers
    assert operator.runlist == [
        ("encoder_pipeline", "W_ATTN", "X", "OR", "B_Up", "B_Down")
    ]


def test_staged_hidden_states_mode_supports_two_ffn_branches_runtime_contract(
    tmp_path,
):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=1,
            parallel_heads=1,
            nB_tiles_distributed=2,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    operator.xclbin_artifact = SimpleNamespace(kernel_name="encoder_pipeline_kernel")
    operator.insts_artifact = object()

    operator.set_up_runtime()

    assert "QKV" not in operator.buffers
    assert "X" in operator.buffers
    assert operator.runlist == [
        ("encoder_pipeline", "W_ATTN", "X", "OR", "B_Up", "B_Down")
    ]


def test_staged_hidden_states_mode_rejects_parallel_heads_over_two(tmp_path):
    with pytest.raises(
        AIEOperatorConstraintError,
        match="staged_hidden_states QKV projection currently supports parallel_heads <= 2",
    ):
        AIEEncoderPipeline(
            **encoder_pipeline_common_kwargs(
                DummyContext(tmp_path),
                parallel_seq=1,
                parallel_heads=4,
                nB_tiles_distributed=1,
                qkv_projection_mode="staged_hidden_states",
            ),
            skip_add_to_list=True,
        )


def test_staged_hidden_states_runtime_contract_omits_qkv_bo(tmp_path):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=1,
            parallel_heads=1,
            nB_tiles_distributed=1,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    operator.xclbin_artifact = SimpleNamespace(kernel_name="encoder_pipeline_kernel")
    operator.insts_artifact = object()

    operator.set_up_runtime()

    assert "QKV" not in operator.buffers
    assert "X" in operator.buffers
    expected_or_rows_before_ln1_stage = (
        3 * operator.seq_len
        if operator._uses_staged_hidden_state_kv_cache()
        else operator.seq_len
    )
    assert operator._or_buffer_shape() == (
        expected_or_rows_before_ln1_stage + operator.proj_acc_depth * operator.seq_tile,
        operator.embed_sz,
    )
    assert operator.runlist == [
        ("encoder_pipeline", "W_ATTN", "X", "OR", "B_Up", "B_Down")
    ]


def test_staged_hidden_states_mlir_generation_supports_seq512():
    mlir = render_encoder_pipeline_mlir(
        seq_len=512,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<512x768xbf16>" in mlir


def test_staged_hidden_states_seq8192_splits_long_kv_dma_fills():
    mlir = render_encoder_pipeline_mlir(
        seq_len=8192,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "<size = 128, stride = 49152>" not in mlir
    assert "<size = 64, stride = 49152>" in mlir


def test_staged_hidden_states_seq8192_supports_parallel_heads_two_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=8192,
        parallel_heads=2,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<8192x768xbf16>" in mlir


def test_staged_hidden_states_seq128_supports_two_ffn_branches_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=128,
        parallel_seq=1,
        parallel_heads=1,
        nB_tiles_distributed=2,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<128x768xbf16>" in mlir


def test_staged_hidden_states_seq128_supports_sequence_parallel_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=128,
        parallel_seq=2,
        parallel_heads=1,
        nB_tiles_distributed=1,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<128x768xbf16>" in mlir


def test_staged_hidden_states_seq128_supports_sequence_parallel_parallel_heads_two_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=128,
        parallel_seq=2,
        parallel_heads=2,
        nB_tiles_distributed=1,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<128x768xbf16>" in mlir


def test_staged_hidden_states_seq8192_supports_sequence_parallel_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=8192,
        parallel_seq=2,
        parallel_heads=1,
        nB_tiles_distributed=1,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<8192x768xbf16>" in mlir
    assert "<size = 64, stride = 49152>" in mlir


def test_staged_hidden_states_seq8192_supports_sequence_parallel_parallel_heads_two_mlir_generation():
    mlir = render_encoder_pipeline_mlir(
        seq_len=8192,
        parallel_seq=2,
        parallel_heads=2,
        nB_tiles_distributed=1,
        qkv_projection_mode="staged_hidden_states",
    )

    assert "memref<8192x768xbf16>" in mlir


def test_staged_hidden_states_rejects_external_residual(tmp_path):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(
            DummyContext(tmp_path),
            parallel_seq=1,
            parallel_heads=1,
            nB_tiles_distributed=1,
            qkv_projection_mode="staged_hidden_states",
            static_weights=True,
        ),
        skip_add_to_list=True,
    )
    hidden_states = torch.zeros(
        (operator.seq_len, operator.embed_sz), dtype=torch.bfloat16
    )

    with pytest.raises(
        AIEOperatorConstraintError,
        match="derives the initial residual from hidden_states internally",
    ):
        operator.forward_hidden_states(hidden_states, r=hidden_states)


def test_forward_defaults_to_hidden_states_path(tmp_path, monkeypatch):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(DummyContext(tmp_path)),
        skip_add_to_list=True,
    )
    hidden_states = torch.zeros(
        (operator.seq_len, operator.embed_sz), dtype=torch.bfloat16
    )
    seen = {}

    def fake_forward_hidden_states(tensor, **kwargs):
        seen["hidden_states"] = tensor
        seen["kwargs"] = kwargs
        return "hidden-states-path"

    monkeypatch.setattr(operator, "forward_hidden_states", fake_forward_hidden_states)

    result = operator.forward(hidden_states, r=hidden_states)

    assert result == "hidden-states-path"
    assert torch.equal(seen["hidden_states"], hidden_states)
    assert torch.equal(seen["kwargs"]["r"], hidden_states)


def test_forward_uses_split_qkv_compatibility_path(tmp_path, monkeypatch):
    operator = AIEEncoderPipeline(
        **encoder_pipeline_common_kwargs(DummyContext(tmp_path)),
        skip_add_to_list=True,
    )
    q = torch.zeros(
        (operator.num_heads, operator.seq_len, operator.d), dtype=torch.bfloat16
    )
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    residual = torch.zeros((operator.seq_len, operator.embed_sz), dtype=torch.bfloat16)
    seen = {}

    def fake_forward_split_qkv(q_tensor, k_tensor, v_tensor, **kwargs):
        seen["q"] = q_tensor
        seen["k"] = k_tensor
        seen["v"] = v_tensor
        seen["kwargs"] = kwargs
        return "split-qkv-path"

    monkeypatch.setattr(operator, "forward_split_qkv", fake_forward_split_qkv)

    result = operator.forward(q, k, v, r=residual)

    assert result == "split-qkv-path"
    assert torch.equal(seen["q"], q)
    assert torch.equal(seen["k"], k)
    assert torch.equal(seen["v"], v)
    assert torch.equal(seen["kwargs"]["r"], residual)


def test_invalid_layout_override_is_rejected():
    with pytest.raises(
        AIEOperatorConstraintError,
        match="encoder_pipeline requires o_proj_fifo_depth > 0",
    ):
        AIEEncoderPipeline(
            num_heads=12,
            seq_len=64,
            d=64,
            seq_tile=32,
            kv_seq_tile=64,
            emb_tile=96,
            ffn_tile=64,
            parallel_seq=2,
            parallel_heads=2,
            proj_acc_depth=8,
            o_proj_acc_group_size=1,
            ffn_down_acc_group_size=1,
            nB_tiles_distributed=2,
            ffn_intermediate_size=3072,
            o_proj_fifo_depth=0,
            skip_add_to_list=True,
        )


def test_unsupported_explicit_addnorm_replay_is_rejected():
    with pytest.raises(
        AIEOperatorConstraintError,
        match="encoder_pipeline only supports use_fused_replayed_addnorm=True",
    ):
        AIEEncoderPipeline(
            num_heads=12,
            seq_len=64,
            d=64,
            seq_tile=32,
            kv_seq_tile=64,
            emb_tile=96,
            ffn_tile=64,
            parallel_seq=1,
            parallel_heads=1,
            proj_acc_depth=8,
            o_proj_acc_group_size=1,
            ffn_down_acc_group_size=1,
            nB_tiles_distributed=1,
            ffn_intermediate_size=3072,
            use_fused_replayed_addnorm=True,
            skip_add_to_list=True,
        )


def test_unsupported_data_movement_override_is_rejected():
    with pytest.raises(
        AIEOperatorConstraintError,
        match="encoder_pipeline only supports use_unified_qr_split",
    ):
        AIEEncoderPipeline(
            num_heads=12,
            seq_len=64,
            d=64,
            seq_tile=32,
            kv_seq_tile=64,
            emb_tile=96,
            ffn_tile=64,
            parallel_seq=2,
            parallel_heads=2,
            proj_acc_depth=8,
            o_proj_acc_group_size=1,
            ffn_down_acc_group_size=1,
            nB_tiles_distributed=2,
            ffn_intermediate_size=3072,
            use_unified_qr_split=False,
            skip_add_to_list=True,
        )


def generate_test_params(base_topology: tuple[int, ...]):
    if not execution_matrix_supported(base_topology):
        return []
    params = []
    supported_topologies = load_encoder_pipeline_topology_placements()
    d, num_heads, ffn_intermediate_size, seq_tile, kv_seq_tile, emb_tile, ffn_tile = (
        base_topology
    )
    for topology_suffix, runtime_topology in TOPOLOGY_CASES:
        (
            parallel_seq,
            parallel_heads,
            proj_acc_depth,
            o_proj_acc_group_size,
            ffn_down_acc_group_size,
            nB_tiles_distributed,
        ) = runtime_topology
        for seq_len in SCALED_SEQ_LENS:
            if seq_len % seq_tile != 0 or (seq_len // seq_tile) % parallel_seq != 0:
                continue
            topology_key = (
                num_heads,
                seq_len,
                d,
                seq_tile,
                kv_seq_tile,
                emb_tile,
                ffn_tile,
                parallel_seq,
                parallel_heads,
                proj_acc_depth,
                o_proj_acc_group_size,
                nB_tiles_distributed,
                ffn_intermediate_size,
            )
            if topology_key not in supported_topologies:
                continue
            params.append(
                pytest.param(
                    seq_len,
                    d,
                    num_heads,
                    ffn_intermediate_size,
                    seq_tile,
                    kv_seq_tile,
                    emb_tile,
                    ffn_tile,
                    parallel_seq,
                    parallel_heads,
                    proj_acc_depth,
                    o_proj_acc_group_size,
                    ffn_down_acc_group_size,
                    nB_tiles_distributed,
                    id=f"encoder_pipeline_{seq_len}seq_{topology_name(base_topology, topology_suffix)}",
                )
            )
    return params


all_params = generate_test_params(BASE_TOPOLOGY)
mirrored_params = generate_test_params(MIRRORED_BASE_TOPOLOGY)
mirrored_large_params = generate_test_params(MIRRORED_LARGE_BASE_TOPOLOGY)


def test_topology_key_round_trip_exposes_canonical_and_legacy_ids():
    key = (12, 64, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072)

    topology = topology_from_key(key)

    assert topology.key == key
    assert topology.family_id == "seq32_kv64"
    assert topology.legacy_alias == "2ps_4pffn"
    assert topology.topology_id == "seq32_kv64__ps2_ph1_pffn4"
    assert topology.cache_signature == (2, 1, 4, 32, 64, 96, 64, 8, 1, 3072)


def test_mirrored_large_family_remains_registered_but_not_in_execution_matrix():
    supported_topologies = load_encoder_pipeline_topology_placements()

    assert (
        16,
        64,
        64,
        64,
        32,
        128,
        64,
        1,
        1,
        8,
        1,
        1,
        4096,
    ) in supported_topologies
    assert mirrored_large_params == []


def test_seq_len_matrix_excludes_16384_by_default():
    assert 16384 not in DEFAULT_SCALED_SEQ_LENS
    assert OPTIONAL_SCALED_SEQ_LENS == (16384,)
    assert max(SCALED_SEQ_LENS) in {8192, 16384}


def test_filter_topologies_by_alias_or_id_accepts_legacy_and_canonical_tokens():
    target = topology_from_key((12, 64, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072))
    alternatives = [
        target,
        topology_from_key((12, 64, 64, 32, 64, 96, 64, 1, 1, 8, 1, 1, 3072)),
    ]

    assert filter_topologies_by_alias_or_id(alternatives, {target.legacy_alias}) == [
        target
    ]
    assert filter_topologies_by_alias_or_id(alternatives, {target.topology_id}) == [
        target
    ]


def test_mirrored_64_32_family_is_registered_for_supported_shapes():
    mirrored = topology_from_key((12, 128, 64, 64, 32, 96, 64, 2, 1, 8, 1, 4, 3072))
    large = topology_from_key((16, 64, 64, 64, 32, 128, 64, 1, 1, 8, 1, 1, 4096))
    supported = {topology.key for topology in all_supported_topologies()}

    assert mirrored.key in supported
    assert large.key in supported


def test_topology_ids_are_unique_across_32_64_and_64_32_families():
    dual_family = [
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 2
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 4
    ]

    assert {topology.family_id for topology in dual_family} == {
        "seq32_kv64",
        "seq64_kv32",
    }
    assert {topology.topology_id for topology in dual_family} == {
        "seq32_kv64__ps2_ph1_pffn4",
        "seq64_kv32__ps2_ph1_pffn4",
    }


def test_topology_compute_tile_count_tracks_full_and_underfilled_layouts():
    one_ps = next(
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 1
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
    )
    four_ps = next(
        topology
        for topology in all_supported_topologies()
        if topology.num_heads == 12
        and topology.seq_len == 128
        and topology.parallel_seq == 4
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
    )

    assert one_ps.compute_tile_count == 8
    assert four_ps.compute_tile_count == 32
    assert one_ps.utilization_fraction == pytest.approx(0.25)
    assert four_ps.utilization_fraction == pytest.approx(1.0)


def test_compute_q_blocks_per_lane_requires_exact_lane_partition():
    assert compute_q_blocks_per_lane(256, 32, 4) == 2

    with pytest.raises(ValueError, match="num_q_seq_blocks divisible by parallel_seq"):
        compute_q_blocks_per_lane(192, 64, 2)


def test_compute_num_kv_seq_blocks_requires_exact_kv_tiling():
    assert compute_num_kv_seq_blocks(256, 64) == 4

    with pytest.raises(ValueError, match="seq_len must be divisible by kv_seq_tile"):
        compute_num_kv_seq_blocks(160, 64)


def test_compute_qkv_head_blocks_requires_parallel_head_partition():
    assert compute_qkv_head_blocks_per_parallel_head(16, 4) == 4

    with pytest.raises(ValueError, match="heads divisible by parallel_heads"):
        compute_qkv_head_blocks_per_parallel_head(12, 5)


def test_compute_ffn_col_group_count_requires_exact_branch_partition():
    assert compute_ffn_col_group_count(3072, 64, 4) == 12

    with pytest.raises(ValueError, match="divide ln1_broadcast_groups"):
        compute_ffn_col_group_count(3072, 64, 5)


def test_build_q_block_schedule_visits_each_q_block_once():
    q_blocks_per_lane = compute_q_blocks_per_lane(256, 32, 4)

    schedule = build_q_block_schedule(4, q_blocks_per_lane)

    assert schedule == [0, 2, 4, 6, 1, 3, 5, 7]
    assert sorted(schedule) == list(range(8))


def test_normalize_slot_coords_supports_single_and_multi_slot_layouts():
    assert _normalize_slot_coords((5, 6), 1, "slot") == [(5, 6)]
    assert _normalize_slot_coords([(1, 2), (3, 4)], 2, "slot") == [
        (1, 2),
        (3, 4),
    ]

    with pytest.raises(ValueError, match="must provide 2 placements"):
        _normalize_slot_coords((1, 2), 2, "slot")


def test_normalize_slot_mem_cols_supports_single_and_multi_slot_layouts():
    assert _normalize_slot_mem_cols(7, 1, "slot") == [7]
    assert _normalize_slot_mem_cols([8, 9], 2, "slot") == [8, 9]

    with pytest.raises(ValueError, match="must provide 2 memtile columns"):
        _normalize_slot_mem_cols(7, 2, "slot")


def test_validate_sequence_parallel_placement_checks_lane_coverage_and_group_size():
    valid = {
        "lane_tiles": [{}, {}, {}, {}],
        "lane_o_proj_acc_mem_cols": [1, 2, 3, 4],
        "lane_tail_mem_cols": [5, 6, 7, 8],
        "transport_groups": [{"lanes": (0, 1)}, {"lanes": (2, 3)}],
    }

    _validate_sequence_parallel_placement(valid, 4)

    with pytest.raises(ValueError, match="cover every lane exactly once"):
        _validate_sequence_parallel_placement(
            {
                **valid,
                "transport_groups": [{"lanes": (0, 1)}, {"lanes": (1, 2)}],
            },
            4,
        )

    with pytest.raises(ValueError, match="uniform size"):
        _validate_sequence_parallel_placement(
            {
                **valid,
                "transport_groups": [{"lanes": (0,)}, {"lanes": (1, 2, 3)}],
            },
            4,
        )


def test_validate_sequence_parallel_placement_checks_o_proj_replay_lane_count():
    with pytest.raises(ValueError, match="O-proj replay memtile per sequence lane"):
        _validate_sequence_parallel_placement(
            {
                "lane_tiles": [{}, {}],
                "lane_o_proj_acc_mem_cols": [1, 2],
                "lane_o_proj_stage_mem_cols": [3],
                "lane_tail_mem_cols": [4, 5],
            },
            2,
        )


def test_normalize_base_placement_preserves_known_tile_and_mem_columns():
    topology = topology_from_key((12, 128, 64, 32, 64, 96, 64, 2, 1, 8, 1, 4, 3072))
    placement = load_encoder_pipeline_topology_placements()[topology.key]

    normalized = _normalize_base_placement(placement)

    assert [tile.col for tile in normalized["qk_tiles"]] == list(placement["mha_cols"])
    assert [tile.row for tile in normalized["qk_tiles"]] == [2]
    assert [tile.col for tile in normalized["o_proj_tiles"]] == list(
        placement["mha_cols"]
    )
    assert len(normalized["ffn_up_tiles"]) == topology.parallel_ffn
    assert len(normalized["ffn_down_tiles"]) == topology.parallel_ffn
    assert normalized["q_mem_col"] == placement["mem_tiles"]["q"]
    assert normalized["output_shim_col"] == placement["shim_tiles"]["output"]


def _run_encoder_pipeline_case(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    golden_ref = generate_golden_reference(
        heads=num_heads,
        seq_len=seq_len,
        d=d,
        intermediate_size=ffn_intermediate_size,
        seq_tile=seq_tile,
        emb_tile=emb_tile,
        ffn_tile=ffn_tile,
        parallel_seq=parallel_seq,
    )

    operator = AIEEncoderPipeline(
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
        ffn_down_acc_group_size=ffn_down_acc_group_size,
        nB_tiles_distributed=nB_tiles_distributed,
        ffn_intermediate_size=ffn_intermediate_size,
        ln1_weight=golden_ref["ln1_weight"],
        ln2_weight=golden_ref["ln2_weight"],
        context=aie_context,
    )

    input_buffers = {
        "QKV": golden_ref["QKV"],
        "OR": golden_ref["OR"],
        "W_O": golden_ref["W_O"],
        "B_Up": golden_ref["B_Up"],
        "B_Down": golden_ref["B_Down"],
    }
    output_buffers = {"O": golden_ref["O"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        warmup_iters=10,
        timed_iters=100,
    )

    max_acceptable_errors = int(seq_len * d * num_heads * ERROR_THRESHOLD)

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    print(
        f"({len(errors.get('O', []))} errors out of {max_acceptable_errors} max allowable)"
    )

    assert len(errors.get("O", [])) <= max_acceptable_errors


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,ffn_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,ffn_down_acc_group_size,nB_tiles_distributed",
    all_params,
)
def test_encoder_pipeline(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    _run_encoder_pipeline_case(
        seq_len,
        d,
        num_heads,
        ffn_intermediate_size,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        ffn_tile,
        parallel_seq,
        parallel_heads,
        proj_acc_depth,
        o_proj_acc_group_size,
        ffn_down_acc_group_size,
        nB_tiles_distributed,
        aie_context,
    )


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,ffn_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,ffn_down_acc_group_size,nB_tiles_distributed",
    mirrored_params,
)
def test_encoder_pipeline_mirrored_family(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    _run_encoder_pipeline_case(
        seq_len,
        d,
        num_heads,
        ffn_intermediate_size,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        ffn_tile,
        parallel_seq,
        parallel_heads,
        proj_acc_depth,
        o_proj_acc_group_size,
        ffn_down_acc_group_size,
        nB_tiles_distributed,
        aie_context,
    )


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,d,num_heads,ffn_intermediate_size,seq_tile,kv_seq_tile,emb_tile,ffn_tile,parallel_seq,parallel_heads,proj_acc_depth,o_proj_acc_group_size,ffn_down_acc_group_size,nB_tiles_distributed",
    mirrored_large_params,
)
def test_encoder_pipeline_mirrored_large_family(
    seq_len,
    d,
    num_heads,
    ffn_intermediate_size,
    seq_tile,
    kv_seq_tile,
    emb_tile,
    ffn_tile,
    parallel_seq,
    parallel_heads,
    proj_acc_depth,
    o_proj_acc_group_size,
    ffn_down_acc_group_size,
    nB_tiles_distributed,
    aie_context,
):
    _run_encoder_pipeline_case(
        seq_len,
        d,
        num_heads,
        ffn_intermediate_size,
        seq_tile,
        kv_seq_tile,
        emb_tile,
        ffn_tile,
        parallel_seq,
        parallel_heads,
        proj_acc_depth,
        o_proj_acc_group_size,
        ffn_down_acc_group_size,
        nB_tiles_distributed,
        aie_context,
    )
