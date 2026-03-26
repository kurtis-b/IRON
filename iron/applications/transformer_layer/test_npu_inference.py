import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from iron.applications.transformer_layer.npu_inference import (
    SUPPORTED_EXECUTION_MODES,
    benchmark_pattern,
    build_pattern,
)
from iron.applications.transformer_layer.src.input_bundle import TransformerLayerInputs
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


class _FakePattern:
    def __init__(self, spec):
        self.spec = spec
        self.weights = None

    def assign_weights(self, weights):
        self.weights = weights

    def forward_with_stage_timings(self, layer_inputs):
        return layer_inputs.r, {
            "encoder_pipeline_sec": 0.002,
        }

    def get_benchmark_metadata(self):
        return {
            "compile_setup_time_ms": 12.5,
            "npu_dispatch_count": 1,
            "npu_unique_instruction_binary_count": 1,
            "process_model": "in_process",
        }

    def __call__(self, layer_inputs):
        return layer_inputs.r


LONG_SEQ_EXEC_PARAMS = [
    pytest.param(
        "encoder_pipeline",
        16384,
        768,
        3072,
        12,
        marks=pytest.mark.extensive,
        id="encoder_pipeline_16384x768x3072x12",
    ),
    pytest.param(
        "encoder_pipeline",
        16384,
        1024,
        4096,
        16,
        marks=pytest.mark.extensive,
        id="encoder_pipeline_16384x1024x4096x16",
    ),
    pytest.param(
        "gemm_only",
        16384,
        768,
        3072,
        12,
        marks=pytest.mark.extensive,
        id="gemm_only_16384x768x3072x12",
    ),
    pytest.param(
        "gemm_only",
        16384,
        1024,
        4096,
        16,
        marks=pytest.mark.extensive,
        id="gemm_only_16384x1024x4096x16",
    ),
]


@pytest.mark.parametrize("execution_mode", SUPPORTED_EXECUTION_MODES)
def test_build_pattern_dispatches(monkeypatch, execution_mode):
    spec = TransformerLayerSpec(seq_len=64)
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.EncoderPipelinePattern",
        _FakePattern,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.GemmOnlyPattern",
        _FakePattern,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.OperatorRunlistPattern",
        _FakePattern,
    )

    pattern = build_pattern(execution_mode, spec)

    assert isinstance(pattern, _FakePattern)
    assert pattern.spec == spec


def test_benchmark_pattern_emits_expected_schema(monkeypatch, tmp_path):
    spec = TransformerLayerSpec(seq_len=64)
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.build_pattern",
        lambda execution_mode, spec: _FakePattern(spec),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.make_synthetic_layer_weights",
        lambda spec, seed: {"dummy": torch.ones(1)},
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.make_synthetic_layer_inputs",
        lambda spec, seed: TransformerLayerInputs(
            q=torch.ones(
                (
                    spec.batch_size,
                    spec.num_attention_heads,
                    spec.seq_len,
                    spec.attention_head_size,
                ),
                dtype=spec.torch_dtype,
            ),
            k=torch.ones(
                (
                    spec.batch_size,
                    spec.num_attention_heads,
                    spec.seq_len,
                    spec.attention_head_size,
                ),
                dtype=spec.torch_dtype,
            ),
            v=torch.ones(
                (
                    spec.batch_size,
                    spec.num_attention_heads,
                    spec.seq_len,
                    spec.attention_head_size,
                ),
                dtype=spec.torch_dtype,
            ),
            r=torch.ones(
                (spec.batch_size, spec.seq_len, spec.hidden_size),
                dtype=spec.torch_dtype,
            ),
        ),
    )
    output_csv = tmp_path / "results.csv"

    rows = benchmark_pattern(
        execution_mode="encoder_pipeline",
        spec=spec,
        warmup_runs=1,
        runs_per_sample=2,
        output_csv=str(output_csv),
        seed=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["execution_mode"] == "encoder_pipeline"
    assert row["backend"] == "npu"
    assert row["throughput_flops_per_sec"] is not None
    assert row["measured_inference_count"] == 2
    assert row["avg_encoder_pipeline_latency_ms"] == pytest.approx(2.0)
    assert row["compile_setup_time_ms"] == 12.5
    assert row["npu_dispatch_count"] == 1
    assert output_csv.exists()


def test_benchmark_pattern_runs_operator_runlist_in_child_process(
    monkeypatch, tmp_path
):
    spec = TransformerLayerSpec(seq_len=64)

    def _fake_subprocess_run(command, cwd, check):
        request_path = Path(command[command.index("--request-json") + 1])
        response_path = Path(command[command.index("--response-json") + 1])
        request = json.loads(request_path.read_text())
        assert request["spec"]["seq_len"] == 64
        response_path.write_text(
            json.dumps(
                {
                    "study_id": "synthetic_transformer_layer",
                    "backend": "npu",
                    "execution_mode": "operator_runlist",
                    "pattern_label": "operator_runlist",
                    "seq_len": 64,
                    "batch_size": 1,
                    "dtype": "bfloat16",
                    "use_bias": False,
                    "weights_source": "synthetic",
                    "source_model_name": None,
                    "source_layer_index": None,
                    "warmup_runs": 1,
                    "runs_per_sample": 2,
                    "measured_inference_count": 2,
                    "timed_total_sec": 0.02,
                    "avg_latency_ms": 10.0,
                    "compile_setup_time_ms": 42.0,
                    "avg_operator_runlist_latency_ms": 9.5,
                    "npu_dispatch_count": 12,
                    "npu_unique_instruction_binary_count": 13,
                    "process_model": "child_process",
                    "throughput_flops_per_sec": 1.0,
                    "estimated_flops_per_inference": 2.0,
                    "estimated_bytes_per_inference": 3.0,
                    "operational_intensity_flops_per_byte": 4.0,
                    "backend_peak_ops_per_sec": None,
                    "roofline_bound_ops_per_sec": None,
                    "backend_pct_of_peak": None,
                    "roofline_pct": None,
                    "avg_power_w": None,
                    "max_power_w": None,
                    "energy_j": None,
                    "power_sample_count": 0,
                }
            )
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "iron.applications.transformer_layer.npu_inference.subprocess.run",
        _fake_subprocess_run,
    )

    output_csv = tmp_path / "operator_runlist.csv"
    rows = benchmark_pattern(
        execution_mode="operator_runlist",
        spec=spec,
        warmup_runs=1,
        runs_per_sample=2,
        output_csv=str(output_csv),
        seed=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["execution_mode"] == "operator_runlist"
    assert row["process_model"] == "child_process"
    assert row["npu_dispatch_count"] == 12
    assert output_csv.exists()


@pytest.mark.parametrize(
    "execution_mode,seq_len,hidden_size,intermediate_size,num_attention_heads",
    LONG_SEQ_EXEC_PARAMS,
)
def test_npu_inference_long_seq_executes(
    execution_mode,
    seq_len,
    hidden_size,
    intermediate_size,
    num_attention_heads,
    tmp_path,
):
    output_csv = tmp_path / f"{execution_mode}_{seq_len}_{hidden_size}.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "iron.applications.transformer_layer.npu_inference",
            "--execution-mode",
            execution_mode,
            "--seq-len",
            str(seq_len),
            "--hidden-size",
            str(hidden_size),
            "--intermediate-size",
            str(intermediate_size),
            "--num-attention-heads",
            str(num_attention_heads),
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
    assert row["execution_mode"] == execution_mode
    assert int(row["seq_len"]) == seq_len
    assert float(row["avg_latency_ms"]) > 0.0
