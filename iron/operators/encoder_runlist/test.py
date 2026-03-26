# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import subprocess
import sys
import pytest
import logging
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.encoder_runlist.op import AIEEncoderRunlist
from iron.operators.encoder_runlist.reference import generate_golden_reference
from iron.common.test_utils import run_test, verify_buffer


def _named_params(params, *, prefix: str):
    names = []
    for seq_len, embedding_dim, ffn_dim, num_heads in params:
        names.append(f"{prefix}_{seq_len}x{embedding_dim}x{ffn_dim}x{num_heads}")
    return [pytest.param(*param_set, id=name) for param_set, name in zip(params, names)]


SMOKE_PARAMS = [
    (64, 768, 3072, 12),
    (128, 768, 3072, 12),
    (512, 768, 3072, 12),
    (64, 1024, 4096, 16),
    (128, 1024, 4096, 16),
    (512, 1024, 4096, 16),
]
LONG_SEQ_EXEC_PARAMS = [
    (1024, 768, 3072, 12),
    (2048, 768, 3072, 12),
    (4096, 768, 3072, 12),
    (1024, 1024, 4096, 16),
    (2048, 1024, 4096, 16),
    (4096, 1024, 4096, 16),
]
BENCHMARK_PARAMS = [
    (512, 768, 3072, 12),
    (512, 1024, 4096, 16),
]

smoke_params = _named_params(SMOKE_PARAMS, prefix="encoder_runlist_smoke")
long_seq_exec_params = [
    pytest.param(
        *param_set,
        marks=pytest.mark.extensive,
        id=name,
    )
    for param_set, name in zip(
        LONG_SEQ_EXEC_PARAMS,
        [
            f"encoder_runlist_long_seq_{seq_len}x{embedding_dim}x{ffn_dim}x{num_heads}"
            for seq_len, embedding_dim, ffn_dim, num_heads in LONG_SEQ_EXEC_PARAMS
        ],
    )
]
benchmark_params = _named_params(
    BENCHMARK_PARAMS, prefix="encoder_runlist_non_pipelined"
)


def test_resolve_k_transpose_num_channels_tracks_seq_len():
    assert AIEEncoderRunlist._resolve_k_transpose_num_channels(64, 64) == 1
    assert AIEEncoderRunlist._resolve_k_transpose_num_channels(128, 64) == 2
    with pytest.raises(ValueError):
        AIEEncoderRunlist._resolve_k_transpose_num_channels(96, 64)


def test_resolve_short_seq_tiles_track_seq_len():
    assert AIEEncoderRunlist._resolve_short_seq_gemm_tile_m(64) == 8
    assert AIEEncoderRunlist._resolve_short_seq_gemm_tile_m(128) == 16
    assert AIEEncoderRunlist._resolve_short_seq_gemm_tile_m(256) == 32
    assert AIEEncoderRunlist._resolve_short_seq_gemm_tile_m(512) == 64
    assert AIEEncoderRunlist._resolve_attn_scores_tile_n(64, 8) == 8
    assert AIEEncoderRunlist._resolve_attn_scores_tile_n(128, 8) == 16
    assert AIEEncoderRunlist._resolve_attn_scores_tile_n(256, 8) == 32
    assert AIEEncoderRunlist._resolve_attn_scores_tile_n(512, 8) == 64


def test_forward_accepts_2d_qkvr_inputs(monkeypatch):
    operator = object.__new__(AIEEncoderRunlist)
    operator.seq_len = 64
    operator.hidden_size = 768
    operator.num_heads = 12
    operator.head_dim = 64
    operator.use_static_runtime_weights = True
    captured = {}

    def fake_write_buffer(name, value):
        captured[name] = value

    monkeypatch.setattr(operator, "write_buffer", fake_write_buffer)
    monkeypatch.setattr(operator, "run_runlist", lambda: None)
    monkeypatch.setattr(
        operator,
        "read_buffer_as_torch",
        lambda name, shape, dtype: torch.zeros(shape, dtype=torch.bfloat16),
    )

    q = torch.zeros((64, 768), dtype=torch.bfloat16)
    k = torch.zeros((64, 768), dtype=torch.bfloat16)
    v = torch.zeros((64, 768), dtype=torch.bfloat16)
    r = torch.zeros((64, 768), dtype=torch.bfloat16)
    result = AIEEncoderRunlist.forward(operator, q, k, v, r)

    assert result.shape == q.shape
    assert captured["Q"].shape[0] == 64 * 768
    assert captured["K"].shape[0] == 64 * 768
    assert captured["V"].shape[0] == 64 * 768
    assert captured["R"].shape[0] == 64 * 768


def test_forward_rejects_non_2d_input():
    operator = object.__new__(AIEEncoderRunlist)
    operator.seq_len = 64
    operator.hidden_size = 768
    operator.num_heads = 12
    operator.head_dim = 64
    operator.use_static_runtime_weights = True

    q = torch.zeros((1, 64, 768))
    k = torch.zeros((64, 768))
    v = torch.zeros((64, 768))
    r = torch.zeros((64, 768))
    with pytest.raises(ValueError, match="expects Q to have shape"):
        AIEEncoderRunlist.forward(operator, q, k, v, r)


def test_write_runtime_weights_preserves_downstream_weight_layout(monkeypatch):
    operator = object.__new__(AIEEncoderRunlist)
    operator.use_pip_addnorm = False
    operator.use_pip_an_ffn = False
    operator.attn_output_weight = torch.arange(12, 24, dtype=torch.bfloat16).reshape(
        3, 4
    )
    operator.ffn_up_weight = torch.arange(24, 36, dtype=torch.bfloat16).reshape(3, 4)
    operator.ffn_down_weight = torch.arange(36, 48, dtype=torch.bfloat16).reshape(3, 4)
    operator.ln1_weight = torch.ones(4, dtype=torch.bfloat16)
    operator.ln2_weight = torch.ones(4, dtype=torch.bfloat16)
    captured = {}

    def fake_write_buffer(name, value):
        captured[name] = value

    monkeypatch.setattr(operator, "write_buffer", fake_write_buffer)

    AIEEncoderRunlist.write_runtime_weights(operator)

    np.testing.assert_array_equal(
        captured["out_proj_weight"].astype(np.float32),
        operator.attn_output_weight.float().numpy(),
    )
    np.testing.assert_array_equal(
        captured["ffn_up_weight"].astype(np.float32),
        operator.ffn_up_weight.float().numpy(),
    )
    np.testing.assert_array_equal(
        captured["ffn_down_weight"].astype(np.float32),
        operator.ffn_down_weight.float().numpy(),
    )


def test_non_pipelined_runlist_uses_add_before_layer_norm():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = SimpleNamespace(
                device_str=lambda: "npu1_4col",
                device_type="npu1_4col",
            )

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    operator = AIEEncoderRunlist(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()
    operator.set_up_runtime()

    assert (
        "encoder_add",
        "output_proj_output",
        "R",
        "ln1_norm_output",
    ) in operator.runlist
    assert (
        "encoder_ln1",
        "ln1_norm_output",
        "ln1_output",
    ) in operator.runlist
    assert (
        "encoder_add",
        "down_proj_output",
        "ln1_output",
        "ln2_norm_output",
    ) in operator.runlist
    assert (
        "encoder_ln2",
        "ln2_norm_output",
        "output",
    ) in operator.runlist

    add1_index = operator.runlist.index(
        ("encoder_add", "output_proj_output", "R", "ln1_norm_output")
    )
    ln1_index = operator.runlist.index(("encoder_ln1", "ln1_norm_output", "ln1_output"))
    add2_index = operator.runlist.index(
        ("encoder_add", "down_proj_output", "ln1_output", "ln2_norm_output")
    )
    ln2_index = operator.runlist.index(("encoder_ln2", "ln2_norm_output", "output"))

    assert add1_index < ln1_index
    assert add2_index < ln2_index


def test_stage_insts_keep_stage_specific_targets():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = SimpleNamespace(
                device_str=lambda: "npu1_4col",
                device_type="npu1_4col",
            )

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    operator = AIEEncoderRunlist(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()

    stage_insts = [
        operator.out_proj_insts,
        operator.k_transpose_insts,
        operator.attn_scores_insts,
        operator.attn_scale_insts,
        operator.attn_softmax_insts,
        operator.attn_output_insts,
        operator.ln1_insts,
        operator.add_insts,
        operator.up_proj_insts,
        operator.gelu_insts,
        operator.down_proj_insts,
        operator.ln2_insts,
    ]
    assert operator.combined_xclbin is not None
    assert all(insts is not None for insts in stage_insts)
    assert all(insts.xclbin_input is None for insts in stage_insts)
    assert (
        operator.k_transpose_insts.kernel_name
        == operator.k_transpose_xclbin.kernel_name
    )
    assert (
        operator.attn_scores_insts.kernel_name
        == operator.attn_scores_xclbin.kernel_name
    )
    assert "_8x96x48_" in operator.out_proj_xclbin.path.name
    assert "_8x64x8_" in operator.attn_scores_xclbin.path.name
    assert "_8x64x8_" in operator.attn_output_xclbin.path.name
    assert operator.ln2_insts.kernel_name == operator.ln2_xclbin.kernel_name


def test_runtime_kernels_bind_stage_specific_xclbins():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = SimpleNamespace(
                device_str=lambda: "npu1_4col",
                device_type="npu1_4col",
            )

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    operator = AIEEncoderRunlist(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()
    operator.set_up_runtime()

    assert operator.kernels["encoder_out_proj"][0] is operator.out_proj_xclbin
    assert operator.kernels["encoder_k_transpose"][0] is operator.k_transpose_xclbin
    assert operator.kernels["encoder_attn_scores"][0] is operator.attn_scores_xclbin
    assert operator.kernels["encoder_attn_scale"][0] is operator.attn_scale_xclbin
    assert operator.kernels["encoder_attn_softmax"][0] is operator.attn_softmax_xclbin
    assert operator.kernels["encoder_attn_output"][0] is operator.attn_output_xclbin
    assert operator.kernels["encoder_ln1"][0] is operator.ln1_xclbin
    assert operator.kernels["encoder_add"][0] is operator.add_xclbin
    assert operator.kernels["encoder_up_proj"][0] is operator.up_proj_xclbin
    assert operator.kernels["encoder_gelu"][0] is operator.gelu_xclbin
    assert operator.kernels["encoder_down_proj"][0] is operator.down_proj_xclbin
    assert operator.kernels["encoder_ln2"][0] is operator.ln2_xclbin


def test_runtime_reuses_attention_score_buffer_for_softmax_output():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = SimpleNamespace(
                device_str=lambda: "npu1_4col",
                device_type="npu1_4col",
            )

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    operator = AIEEncoderRunlist(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()
    operator.set_up_runtime()

    assert operator.buffer_aliases["attn_weights_output"] == "attn_scores_output"
    assert (
        operator.buffers["attn_weights_output"]
        == operator.buffers["attn_scores_output"]
    )


def test_encoder_runlist_defaults_to_eager_kernel_loading():
    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = SimpleNamespace(
                device_str=lambda: "npu1_4col",
                device_type="npu1_4col",
            )

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    operator = AIEEncoderRunlist(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        context=DummyContext(),
    )

    assert operator.lazy_kernel_loading is False


@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads",
    smoke_params,
)
def test_encoder_runlist_smoke(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
    aie_context,
):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIEEncoderRunlist(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        context=aie_context,
    )

    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    input_buffers = {
        "Q": golden_ref["Q"],
        "K": golden_ref["K"],
        "V": golden_ref["V"],
        "R": golden_ref["R"],
    }
    output_buffers = {
        "output": golden_ref["output"],
    }
    intermediate_buffers = {}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        intermediate_buffers,
        rel_tol=0.05,
        abs_tol=0.5,
        warmup_iters=1,
        timed_iters=1,
    )

    print(f"\nSmoke latency (us): {latency_us:.1f}\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * embedding_dim * error_threshold)

    output_errors = errors.get("output", [])
    assert len(output_errors) <= max_acceptable_errors, (
        f"Test failed with {len(output_errors)} errors "
        f"(max allowable: {max_acceptable_errors})"
    )


@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads",
    long_seq_exec_params,
)
def test_encoder_runlist_long_seq_executes_via_npu_inference(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
    tmp_path,
):
    output_csv = tmp_path / f"encoder_runlist_{seq_len}_{embedding_dim}.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "iron.applications.transformer_layer.npu_inference",
            "--execution-mode",
            "operator_runlist",
            "--seq-len",
            str(seq_len),
            "--hidden-size",
            str(embedding_dim),
            "--intermediate-size",
            str(ffn_dim),
            "--num-attention-heads",
            str(num_heads),
            "--warmup-runs",
            "0",
            "--runs-per-sample",
            "1",
            "--output-csv",
            str(output_csv),
        ],
        check=True,
    )

    with output_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["execution_mode"] == "operator_runlist"
    assert row["process_model"] == "child_process"
    assert int(row["seq_len"]) == seq_len
    assert float(row["avg_latency_ms"]) > 0.0


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads",
    benchmark_params,
)
def test_encoder_runlist_benchmark(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
    aie_context,
):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIEEncoderRunlist(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        context=aie_context,
    )

    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    input_buffers = {
        "Q": golden_ref["Q"],
        "K": golden_ref["K"],
        "V": golden_ref["V"],
        "R": golden_ref["R"],
    }
    output_buffers = {
        "output": golden_ref["output"],
    }
    intermediate_buffers = {}

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        intermediate_buffers,
        rel_tol=0.05,
        abs_tol=0.5,
        warmup_iters=10,
        timed_iters=100,
    )

    # Use batch_size of C for total operations
    total_macs = seq_len * embedding_dim * embedding_dim
    total_macs += seq_len * (embedding_dim // num_heads) * seq_len * num_heads * 2
    total_macs += seq_len * embedding_dim * ffn_dim * 2
    total_ops = total_macs * 2  # 2 operations per MAC
    gflops = total_ops / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    # print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * embedding_dim * error_threshold)

    output_errors = errors.get("output", [])
    assert len(output_errors) <= max_acceptable_errors, (
        f"Test failed with {len(output_errors)} errors "
        f"(max allowable: {max_acceptable_errors})"
    )
