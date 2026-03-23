# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import logging
import torch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iron.operators.encoder.op import AIEBERTEncoder
from iron.operators.encoder.reference import generate_golden_reference
from iron.common.test_utils import run_test, verify_buffer


def generate_test_params(extensive=False):
    params = [
        # seq_len,embedding_dim,ffn_dim,num_heads,use_pip_ffn,use_pip_addnorm,use_pip_mha,use_pip_anffn
        (512, 768, 3072, 12, False, False, False, False),
        (512, 768, 3072, 12, True, False, False, False),
        (512, 768, 3072, 12, False, True, False, False),
        (512, 768, 3072, 12, False, False, True, False),
        (512, 768, 3072, 12, True, True, False, False),
        (512, 768, 3072, 12, True, False, True, False),
        (512, 768, 3072, 12, False, True, True, False),
        (512, 768, 3072, 12, True, True, True, False),
        (512, 768, 3072, 12, False, False, False, True),
        (512, 768, 3072, 12, False, False, True, True),
    ]
    extensive_params = []

    if extensive:
        params = extensive_params

    names = []
    for (
        seq_len,
        embedding_dim,
        ffn_dim,
        num_heads,
        use_pip_ffn,
        use_pip_addnorm,
        use_pip_mha,
        use_pip_an_ffn,
    ) in params:
        name = f"bert_encoder_{seq_len}x{embedding_dim}x{ffn_dim}x{num_heads}xpipffn_{use_pip_ffn}_pipaddnorm_{use_pip_addnorm}_pipmha_{use_pip_mha}_pipanffn_{use_pip_an_ffn}"
        names.append(name)

    return params, names


regular_params, regular_names = generate_test_params()

# Combine params with marks - extensive params get pytest.mark.extensive
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
]


def test_resolve_k_transpose_num_channels_tracks_seq_len():
    assert AIEBERTEncoder._resolve_k_transpose_num_channels(64, 64) == 1
    assert AIEBERTEncoder._resolve_k_transpose_num_channels(128, 64) == 2
    with pytest.raises(ValueError):
        AIEBERTEncoder._resolve_k_transpose_num_channels(96, 64)


def test_resolve_mha_num_pipelines_tracks_seq_len_blocks():
    assert AIEBERTEncoder._resolve_mha_num_pipelines(64) == 1
    assert AIEBERTEncoder._resolve_mha_num_pipelines(128) == 2
    assert AIEBERTEncoder._resolve_mha_num_pipelines(256) == 4
    assert AIEBERTEncoder._resolve_mha_num_pipelines(512) == 8
    assert AIEBERTEncoder._resolve_mha_num_pipelines(1024) == 8


def test_forward_accepts_2d_seq_hidden_input(monkeypatch):
    operator = object.__new__(AIEBERTEncoder)
    operator.seq_len = 64
    operator.hidden_size = 768
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

    x = torch.zeros((64, 768), dtype=torch.bfloat16)
    result = AIEBERTEncoder.forward(operator, x)

    assert result.shape == x.shape
    assert captured["input"].shape[0] == 64 * 768


def test_forward_rejects_non_2d_input():
    operator = object.__new__(AIEBERTEncoder)
    operator.seq_len = 64
    operator.hidden_size = 768

    with pytest.raises(ValueError, match="expects a 2D tensor"):
        AIEBERTEncoder.forward(operator, torch.zeros((1, 64, 768)))


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

    operator = AIEBERTEncoder(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        use_pip_ffn=False,
        use_pip_addnorm=False,
        use_pip_mha=False,
        use_pip_an_ffn=False,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()

    stage_insts = [
        operator.qkvo_proj_insts,
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

    operator = AIEBERTEncoder(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        use_pip_ffn=False,
        use_pip_addnorm=False,
        use_pip_mha=False,
        use_pip_an_ffn=False,
        ln1_weight=torch.ones(768, dtype=torch.bfloat16),
        ln2_weight=torch.ones(768, dtype=torch.bfloat16),
        context=DummyContext(),
    )
    operator.set_up_artifacts()
    operator.set_up_runtime()

    assert operator.kernels["encoder_qkvo_proj"][0] is operator.qkvo_proj_xclbin
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


def test_encoder_defaults_to_eager_kernel_loading():
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

    operator = AIEBERTEncoder(
        seq_len=64,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        use_pip_ffn=False,
        use_pip_addnorm=False,
        use_pip_mha=False,
        use_pip_an_ffn=False,
        context=DummyContext(),
    )

    assert operator.lazy_kernel_loading is False


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "seq_len,embedding_dim,ffn_dim,num_heads,use_pip_ffn,use_pip_addnorm,use_pip_mha,use_pip_an_ffn",
    all_params,
)
def test_bert_encoder(
    seq_len,
    embedding_dim,
    ffn_dim,
    num_heads,
    use_pip_ffn,
    use_pip_addnorm,
    use_pip_mha,
    use_pip_an_ffn,
    aie_context,
):
    golden_ref = generate_golden_reference(seq_len, embedding_dim, ffn_dim, num_heads)

    operator = AIEBERTEncoder(
        seq_len=seq_len,
        hidden_size=embedding_dim,
        intermediate_size=ffn_dim,
        num_heads=num_heads,
        use_pip_ffn=use_pip_ffn,
        use_pip_addnorm=use_pip_addnorm,
        use_pip_mha=use_pip_mha,
        use_pip_an_ffn=use_pip_an_ffn,
        ln1_weight=golden_ref["weights"]["ln1_weight"],
        ln2_weight=golden_ref["weights"]["ln2_weight"],
        context=aie_context,
    )

    operator.q_weight = golden_ref["weights"]["q_weight"]
    operator.k_weight = golden_ref["weights"]["k_weight"]
    operator.v_weight = golden_ref["weights"]["v_weight"]
    operator.attn_output_weight = golden_ref["weights"]["attn_output_weight"]
    operator.ffn_up_weight = golden_ref["weights"]["ffn_up_weight"]
    operator.ffn_down_weight = golden_ref["weights"]["ffn_down_weight"]

    input_buffers = {
        "input": golden_ref["input"],
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
    total_macs = seq_len * embedding_dim * embedding_dim * 4
    total_macs += seq_len * (embedding_dim // num_heads) * seq_len * num_heads * 2
    total_macs += seq_len * embedding_dim * ffn_dim * 2
    total_ops = total_macs * 2  # 2 operations per MAC
    gflops = total_ops / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    # print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    error_threshold = 0.05
    max_acceptable_errors = int(seq_len * embedding_dim * error_threshold)

    assert (
        len(errors["output"]) <= max_acceptable_errors
    ), f"Test failed with {len(errors['output'])} errors (max allowable: {max_acceptable_errors})"
