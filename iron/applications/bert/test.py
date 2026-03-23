#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))

import benchmark_common
import automated_benchmark
import cpu_inference
import igpu_inference
import npu_inference
import run_automated_benchmark_job
from automated_benchmark import (
    build_suite_row,
    case_command,
    enumerate_cases,
    is_transient_npu_startup_failure,
    npu_case_skip_reason,
    parse_seq_len_int_overrides,
    run_case,
)
from benchmark_common import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_BENCHMARK_SEQ_LENS,
    acceptable_cooldown_temp,
    cooldown_before_benchmark,
    encode_model_valid_texts,
    parse_npu_execution_modes,
    parse_seq_lens,
    validate_benchmark_mode_request,
)
from benchmark_power import (
    discover_powercap_rapl_zones,
    empty_power_stats,
    parse_power_log,
    resolve_power_backend,
    summarize_powercap_rapl,
)
from model_support import (
    canonicalize_app_config_dict,
    canonicalize_local_backbone_weights,
    estimate_encoder_forward_flops,
    zero_bias_tensors_in_state_dict,
    zero_all_model_biases,
)
from npu_inference import (
    current_topology_from_config,
    find_cached_topology,
    load_encoder_pipeline_topology_placements,
    load_encoder_pipeline_config,
    parse_candidate_topology_ids,
    resolve_topology,
    select_autotune_topology,
    supported_topologies_for_config_family,
    supported_topologies_for_seq_len,
    topology_cache_key,
    topology_id,
)

WEIGHTS_FILE = TEST_DIR / "model.safetensors"
CONFIG_FILE = TEST_DIR / "config" / "config.json"
ROBERTA_CONFIG_FILE = TEST_DIR / "config" / "config_roberta_base.json"
DISTILBERT_CONFIG_FILE = TEST_DIR / "config" / "config_distilbert_base.json"
STUDY_MANIFEST = TEST_DIR / "study" / "encoder_only_models.json"


def npu_device_accessible():
    device_path = Path("/dev/accel/accel0")
    return device_path.exists() and os.access(device_path, os.R_OK | os.W_OK)


@pytest.mark.parametrize(
    "script_name,output_csv",
    [
        ("cpu_inference.py", "cpu_benchmark_latest.csv"),
        ("npu_inference.py", "npu_benchmark_latest.csv"),
    ],
    ids=["cpu_encoder_benchmark", "npu_encoder_benchmark"],
)
def test_bert_encoder_benchmarks(script_name, output_csv):
    if not WEIGHTS_FILE.exists():
        pytest.skip(f"Missing benchmark weights at {WEIGHTS_FILE}")
    if script_name == "npu_inference.py" and not npu_device_accessible():
        pytest.skip("NPU device is not accessible in this environment")

    command = [
        "python3",
        str(TEST_DIR / script_name),
        str(WEIGHTS_FILE),
        str(CONFIG_FILE),
        "--seq-lens",
        "64",
        "--num-samples",
        "1",
        "--warmup-runs",
        "0",
        "--runs-per-sample",
        "1",
        "--output-csv",
        output_csv,
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if (
        script_name == "npu_inference.py"
        and result.returncode != 0
        and (
            "Open /dev/accel/accel0 failed" in result.stdout
            or "Open /dev/accel/accel0 failed" in result.stderr
            or "Permission denied" in result.stdout
            or "Permission denied" in result.stderr
        )
    ):
        pytest.skip("NPU device access is denied in this environment")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (TEST_DIR / output_csv).exists(), f"Expected {output_csv} to be written"


def test_bert_automated_benchmark_cpu_smoke(tmp_path):
    if not WEIGHTS_FILE.exists():
        pytest.skip(f"Missing benchmark weights at {WEIGHTS_FILE}")

    result_csv = tmp_path / "automated_benchmark_latest.csv"
    state_json = tmp_path / "automated_benchmark_state.json"
    logs_dir = tmp_path / "logs"
    command = [
        "python3",
        str(TEST_DIR / "automated_benchmark.py"),
        str(WEIGHTS_FILE),
        str(CONFIG_FILE),
        "--modes",
        "cpu",
        "--seq-lens",
        "64",
        "--num-samples",
        "1",
        "--warmup-runs",
        "0",
        "--runs-per-sample",
        "1",
        "--cpu-thread-counts",
        "8",
        "--power-backend",
        "none",
        "--state-json",
        str(state_json),
        "--output-csv",
        str(result_csv),
        "--logs-dir",
        str(logs_dir),
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert result_csv.exists(), f"Expected {result_csv} to be written"
    assert state_json.exists(), f"Expected {state_json} to be written"


def test_bert_automated_benchmark_job_wrapper_smoke(tmp_path):
    if not WEIGHTS_FILE.exists():
        pytest.skip(f"Missing benchmark weights at {WEIGHTS_FILE}")

    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            f'  "weights_file_path": "{WEIGHTS_FILE}",\n'
            f'  "config_file_path": "{CONFIG_FILE}",\n'
            '  "modes": "cpu",\n'
            '  "seq_lens": "64",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = [
        "python3",
        str(TEST_DIR / "run_automated_benchmark_job.py"),
        str(job_config),
        "--print-command",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "automated_benchmark.py" in result.stdout


def test_bert_automated_benchmark_job_wrapper_prints_runs_per_sample_overrides(
    tmp_path,
):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "npu",\n'
            '  "seq_lens": "2048,4096,8192",\n'
            '  "runs_per_sample": 100,\n'
            '  "runs_per_sample_overrides": "2048=50,4096=15,8192=5",\n'
            '  "npu_autotune_runs": 5,\n'
            '  "npu_autotune_runs_overrides": "2048=3,4096=2,8192=1",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = [
        "python3",
        str(TEST_DIR / "run_automated_benchmark_job.py"),
        str(job_config),
        "--print-command",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "--runs-per-sample 100" in result.stdout
    assert "--runs-per-sample-overrides 2048=50,4096=15,8192=5" in result.stdout
    assert "--npu-autotune-runs 5" in result.stdout
    assert "--npu-autotune-runs-overrides 2048=3,4096=2,8192=1" in result.stdout


def test_job_wrapper_print_command_includes_peak_reference_flags(tmp_path):
    peak_reference = tmp_path / "peak_references.json"
    peak_reference.write_text('{"references": []}\n', encoding="utf-8")
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "npu",\n'
            '  "seq_lens": "64",\n'
            '  "benchmark_mode": "model_valid",\n'
            '  "peak_reference": "peak_references.json",\n'
            '  "bytes_model_version": "v1",\n'
            '  "bytes_model_weights_policy": "streamed",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = run_automated_benchmark_job.build_command(
        str(job_config), run_automated_benchmark_job.load_job_config(str(job_config))
    )

    assert "--peak-reference" in command
    assert command[command.index("--peak-reference") + 1] == str(
        peak_reference.resolve()
    )
    assert "--benchmark-mode" in command
    assert command[command.index("--benchmark-mode") + 1] == "model_valid"
    assert "--bytes-model-version" in command
    assert command[command.index("--bytes-model-version") + 1] == "v1"
    assert "--bytes-model-weights-policy" in command
    assert command[command.index("--bytes-model-weights-policy") + 1] == "streamed"


def test_cpu_parse_args_defaults_to_bfloat16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cpu_inference.py"])

    args = cpu_inference.parse_args()

    assert args.dtype == "bfloat16"
    assert args.benchmark_mode == DEFAULT_BENCHMARK_MODE
    assert args.disable_all_biases is False


def test_cpu_parse_args_preserves_explicit_float32_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cpu_inference.py", "--dtype", "float32"])

    args = cpu_inference.parse_args()

    assert args.dtype == "float32"


def test_cpu_parse_args_accepts_disable_all_biases(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cpu_inference.py", "--disable-all-biases"])

    args = cpu_inference.parse_args()

    assert args.disable_all_biases is True


def test_igpu_parse_args_defaults_to_bfloat16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["igpu_inference.py"])

    args = igpu_inference.parse_args()

    assert args.dtype == "bfloat16"
    assert args.benchmark_mode == DEFAULT_BENCHMARK_MODE
    assert args.disable_all_biases is False


def test_npu_parse_args_defaults_benchmark_mode_to_synthetic_dense(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["npu_inference.py"])

    args = npu_inference.parse_args()

    assert args.benchmark_mode == DEFAULT_BENCHMARK_MODE
    assert args.execution_mode == "encoder_pipeline"
    assert args.disable_all_biases is False


def test_npu_parse_args_accepts_gemm_only_execution_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["npu_inference.py", "--execution-mode", "gemm_only"],
    )

    args = npu_inference.parse_args()

    assert args.execution_mode == "gemm_only"


def test_npu_parse_args_accepts_operator_runlist_execution_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["npu_inference.py", "--execution-mode", "operator_runlist"],
    )

    args = npu_inference.parse_args()

    assert args.execution_mode == "operator_runlist"


def test_npu_parse_args_accepts_disable_all_biases(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["npu_inference.py", "--disable-all-biases"])

    args = npu_inference.parse_args()

    assert args.disable_all_biases is True


def test_igpu_parse_args_preserves_explicit_float16_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["igpu_inference.py", "--dtype", "float16"])

    args = igpu_inference.parse_args()

    assert args.dtype == "float16"


def test_igpu_parse_args_accepts_disable_all_biases(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["igpu_inference.py", "--disable-all-biases"])

    args = igpu_inference.parse_args()

    assert args.disable_all_biases is True


def test_automated_benchmark_parse_args_defaults_igpu_dtype_to_bfloat16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["automated_benchmark.py"])

    args = automated_benchmark.parse_args()

    assert args.igpu_dtype == "bfloat16"
    assert args.benchmark_mode == DEFAULT_BENCHMARK_MODE
    assert args.disable_all_biases is False


def test_zero_all_model_biases_zeroes_linear_and_layernorm_biases_only():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )
    original_linear_weight = model[0].weight.detach().clone()
    original_ln_weight = model[1].weight.detach().clone()

    zeroed = zero_all_model_biases(model)

    assert sorted(zeroed) == ["0.bias", "1.bias"]
    assert torch.count_nonzero(model[0].bias).item() == 0
    assert torch.count_nonzero(model[1].bias).item() == 0
    assert torch.equal(model[0].weight, original_linear_weight)
    assert torch.equal(model[1].weight, original_ln_weight)


def test_zero_bias_tensors_in_state_dict_zeroes_bias_entries_only():
    state_dict = {
        "encoder.layer.0.attention.self.query.bias": torch.tensor([1.0, -2.0]),
        "encoder.layer.0.attention.self.query.weight": torch.tensor([[3.0, 4.0]]),
        "encoder.layer.0.output.LayerNorm.bias": torch.tensor([5.0, 6.0]),
    }

    zeroed = zero_bias_tensors_in_state_dict(state_dict)

    assert sorted(zeroed) == [
        "encoder.layer.0.attention.self.query.bias",
        "encoder.layer.0.output.LayerNorm.bias",
    ]
    assert (
        torch.count_nonzero(
            state_dict["encoder.layer.0.attention.self.query.bias"]
        ).item()
        == 0
    )
    assert (
        torch.count_nonzero(state_dict["encoder.layer.0.output.LayerNorm.bias"]).item()
        == 0
    )
    assert torch.equal(
        state_dict["encoder.layer.0.attention.self.query.weight"],
        torch.tensor([[3.0, 4.0]]),
    )


def test_validate_benchmark_mode_request_rejects_npu_model_valid():
    with pytest.raises(ValueError, match="attention_mask=None"):
        validate_benchmark_mode_request(
            "model_valid",
            backend_mode="npu",
            seq_lens=[64],
        )


def test_parse_npu_execution_modes_accepts_multiple_modes():
    assert parse_npu_execution_modes("encoder_pipeline,gemm_only,operator_runlist") == [
        "encoder_pipeline",
        "gemm_only",
        "operator_runlist",
    ]


def test_validate_benchmark_mode_request_rejects_model_valid_seq_len_above_512():
    with pytest.raises(ValueError, match="seq_len <= 512"):
        validate_benchmark_mode_request(
            "model_valid",
            backend_mode="cpu",
            seq_lens=[1024],
        )


def test_encode_model_valid_texts_fills_missing_token_type_ids():
    class FakeTokenizer:
        def __call__(
            self, text, truncation, padding, max_length, return_attention_mask
        ):
            assert truncation is True
            assert padding == "max_length"
            assert return_attention_mask is True
            assert text == "sample text"
            assert max_length == 4
            return {
                "input_ids": [101, 2003, 102, 0],
                "attention_mask": [1, 1, 1, 0],
            }

    samples = encode_model_valid_texts(
        ["sample text"],
        seq_len=4,
        tokenizer=FakeTokenizer(),
    )

    assert len(samples) == 1
    assert samples[0]["input_ids"].tolist() == [[101, 2003, 102, 0]]
    assert samples[0]["attention_mask"].tolist() == [[1, 1, 1, 0]]
    assert samples[0]["token_type_ids"].tolist() == [[0, 0, 0, 0]]


def test_prepare_benchmark_samples_uses_arange_inputs_for_operator_runlist():
    samples = benchmark_common.prepare_benchmark_samples(
        benchmark_mode="synthetic_dense",
        execution_mode="operator_runlist",
        texts=["sample"],
        seq_len=8,
        vocab_size=32,
        pad_token_id=0,
        weights_file_path="weights",
        config_file_path="config",
    )

    assert len(samples) == 1
    assert samples[0]["input_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert samples[0]["token_type_ids"].tolist() == [[0, 0, 0, 0, 0, 0, 0, 0]]


def test_job_wrapper_print_command_reflects_bfloat16_igpu_dtype(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "igpu",\n'
            '  "seq_lens": "64",\n'
            '  "igpu_dtype": "bfloat16",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = run_automated_benchmark_job.build_command(
        str(job_config), run_automated_benchmark_job.load_job_config(str(job_config))
    )

    assert "--igpu-dtype" in command
    assert command[command.index("--igpu-dtype") + 1] == "bfloat16"


def test_job_wrapper_build_command_emits_disable_all_biases_as_bare_flag(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "cpu,igpu",\n'
            '  "seq_lens": "64",\n'
            '  "disable_all_biases": true,\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = run_automated_benchmark_job.build_command(
        str(job_config), run_automated_benchmark_job.load_job_config(str(job_config))
    )

    assert "--disable-all-biases" in command
    assert "True" not in command


def test_job_wrapper_build_command_includes_npu_execution_modes(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "npu",\n'
            '  "seq_lens": "64",\n'
            '  "npu_execution_modes": "encoder_pipeline,gemm_only,operator_runlist",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = run_automated_benchmark_job.build_command(
        str(job_config), run_automated_benchmark_job.load_job_config(str(job_config))
    )

    assert "--npu-execution-modes" in command
    assert command[command.index("--npu-execution-modes") + 1] == (
        "encoder_pipeline,gemm_only,operator_runlist"
    )


def test_bert_automated_benchmark_job_wrapper_study_id_smoke(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    models_root = tmp_path / "models"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            f'  "models_root": "{models_root}",\n'
            '  "modes": "cpu",\n'
            '  "seq_lens": "64",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = [
        "python3",
        str(TEST_DIR / "run_automated_benchmark_job.py"),
        str(job_config),
        "--print-command",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "--study-id bert-base-uncased" in result.stdout


def test_cpu_encoder_benchmark_study_id_smoke(tmp_path):
    if not WEIGHTS_FILE.exists():
        pytest.skip(f"Missing benchmark weights at {WEIGHTS_FILE}")

    models_root = tmp_path / "models"
    model_dir = models_root / "bert-base-uncased"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEIGHTS_FILE, model_dir / "model.safetensors")

    output_csv = tmp_path / "cpu_study_id_latest.csv"
    command = [
        "python3",
        str(TEST_DIR / "cpu_inference.py"),
        "--study-id",
        "bert-base-uncased",
        "--models-root",
        str(models_root),
        "--seq-lens",
        "64",
        "--num-samples",
        "1",
        "--warmup-runs",
        "0",
        "--runs-per-sample",
        "1",
        "--num-threads",
        "1",
        "--output-csv",
        str(output_csv),
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert output_csv.exists(), f"Expected {output_csv} to be written"


def test_download_model_print_plan_smoke(tmp_path):
    models_root = tmp_path / "models"
    command = [
        "python3",
        str(TEST_DIR / "download_model.py"),
        "--study-id",
        "bert-base-uncased",
        "--models-root",
        str(models_root),
        "--print-plan",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "study_id=bert-base-uncased" in result.stdout
    assert str(models_root / "bert-base-uncased" / "model.safetensors") in result.stdout


def test_bringup_checklist_dry_run_smoke(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_id": "bert-base-uncased",\n'
            '  "modes": "cpu",\n'
            '  "seq_lens": "64",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = [
        "bash",
        str(TEST_DIR / "bringup_checklist.sh"),
        "--dry-run",
        "--skip-download",
        "--skip-autotune",
        "--skip-power-check",
        "--skip-suite",
        "--job-config",
        str(job_config),
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Phase: env" in result.stdout
    assert "cpu_inference.py" in result.stdout
    assert "npu_inference.py" in result.stdout
    assert "run_automated_benchmark_job.py" in result.stdout


@pytest.mark.parametrize(
    "config_path,expected_family",
    [
        (CONFIG_FILE, "bert"),
        (ROBERTA_CONFIG_FILE, "roberta"),
        (DISTILBERT_CONFIG_FILE, "distilbert"),
    ],
    ids=["bert_config", "roberta_config", "distilbert_config"],
)
def test_supported_encoder_model_configs_parse(config_path, expected_family):
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    canonical = canonicalize_app_config_dict(config_json)
    assert canonical["model_config"]["model_type"] == expected_family


def test_distilbert_weights_map_to_canonical_local_schema():
    raw_weights = {
        "distilbert.embeddings.word_embeddings.weight": torch.zeros(8, 4),
        "distilbert.embeddings.position_embeddings.weight": torch.zeros(16, 4),
        "distilbert.embeddings.LayerNorm.weight": torch.ones(4),
        "distilbert.embeddings.LayerNorm.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.attention.q_lin.weight": torch.zeros(4, 4),
        "distilbert.transformer.layer.0.attention.q_lin.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.attention.k_lin.weight": torch.zeros(4, 4),
        "distilbert.transformer.layer.0.attention.k_lin.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.attention.v_lin.weight": torch.zeros(4, 4),
        "distilbert.transformer.layer.0.attention.v_lin.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.attention.out_lin.weight": torch.zeros(4, 4),
        "distilbert.transformer.layer.0.attention.out_lin.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.sa_layer_norm.weight": torch.ones(4),
        "distilbert.transformer.layer.0.sa_layer_norm.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.ffn.lin1.weight": torch.zeros(8, 4),
        "distilbert.transformer.layer.0.ffn.lin1.bias": torch.zeros(8),
        "distilbert.transformer.layer.0.ffn.lin2.weight": torch.zeros(4, 8),
        "distilbert.transformer.layer.0.ffn.lin2.bias": torch.zeros(4),
        "distilbert.transformer.layer.0.output_layer_norm.weight": torch.ones(4),
        "distilbert.transformer.layer.0.output_layer_norm.bias": torch.zeros(4),
    }
    config = {
        "model_type": "distilbert",
        "vocab_size": 8,
        "hidden_size": 4,
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "max_position_embeddings": 16,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-12,
    }

    canonical = canonicalize_local_backbone_weights(raw_weights, config)
    assert "embeddings.word_embeddings.weight" in canonical
    assert "encoder.layer.0.attention.self.query.weight" in canonical
    assert "encoder.layer.0.output.LayerNorm.weight" in canonical


def test_encoder_study_manifest_references_existing_configs():
    import json

    with open(STUDY_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["study_models"], "Expected at least one study model entry"
    for entry in manifest["study_models"]:
        config_path = TEST_DIR / entry["config_file"]
        assert (
            config_path.exists()
        ), f"Missing config referenced by manifest: {config_path}"


def test_encoder_forward_flops_estimate_grows_with_seq_len():
    import json

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    model_config = canonicalize_app_config_dict(config_json)["model_config"]

    seq64 = estimate_encoder_forward_flops(model_config, 64)
    seq512 = estimate_encoder_forward_flops(model_config, 512)

    assert seq64 > 0
    assert seq512 > seq64


def test_default_benchmark_seq_lens_stop_at_8192():
    assert parse_seq_lens(DEFAULT_BENCHMARK_SEQ_LENS) == [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]


def test_acceptable_cooldown_temp_applies_fractional_slack():
    assert acceptable_cooldown_temp(50.0, 0.05) == pytest.approx(52.5)
    assert acceptable_cooldown_temp(50.0, 0.0) == pytest.approx(50.0)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))


def test_cooldown_before_benchmark_accepts_temperature_within_tolerance(monkeypatch):
    clock = _FakeClock()
    readings = iter(
        [
            {"temperature_c": 55.0, "source": "fake:cpu"},
            {"temperature_c": 52.4, "source": "fake:cpu"},
        ]
    )

    monkeypatch.setattr(benchmark_common.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(benchmark_common.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        benchmark_common,
        "read_temperature_reading",
        lambda mode, requested_source: next(readings),
    )

    args = SimpleNamespace(
        cooldown_sec=0.0,
        cooldown_until_temp_c=50.0,
        cooldown_temp_source="cpu",
        cooldown_temp_tolerance_frac=0.05,
        cooldown_poll_sec=2.0,
        cooldown_timeout_sec=300.0,
    )
    stats = cooldown_before_benchmark(args, mode="cpu", label="toy-case")

    assert stats["cooldown_start_temp_c"] == pytest.approx(55.0)
    assert stats["cooldown_end_temp_c"] == pytest.approx(52.4)
    assert stats["cooldown_temp_source"] == "fake:cpu"
    assert stats["cooldown_wait_sec"] == pytest.approx(0.0)


def test_cooldown_before_benchmark_continues_after_timeout(monkeypatch):
    clock = _FakeClock()
    readings = iter(
        [
            {"temperature_c": 60.0, "source": "fake:cpu"},
            {"temperature_c": 55.0, "source": "fake:cpu"},
            {"temperature_c": 55.0, "source": "fake:cpu"},
            {"temperature_c": 55.0, "source": "fake:cpu"},
            {"temperature_c": 55.0, "source": "fake:cpu"},
        ]
    )

    monkeypatch.setattr(benchmark_common.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(benchmark_common.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        benchmark_common,
        "read_temperature_reading",
        lambda mode, requested_source: next(readings),
    )

    args = SimpleNamespace(
        cooldown_sec=0.0,
        cooldown_until_temp_c=50.0,
        cooldown_temp_source="cpu",
        cooldown_temp_tolerance_frac=0.05,
        cooldown_poll_sec=2.0,
        cooldown_timeout_sec=5.0,
    )
    stats = cooldown_before_benchmark(args, mode="cpu", label="toy-case")

    assert stats["cooldown_start_temp_c"] == pytest.approx(60.0)
    assert stats["cooldown_end_temp_c"] == pytest.approx(55.0)
    assert stats["cooldown_wait_sec"] == pytest.approx(6.0)
    assert stats["cooldown_temp_source"] == "fake:cpu"


def test_parse_power_log_accepts_periodic_samples(tmp_path):
    log_path = tmp_path / "turbostat.log"
    log_path.write_text(
        "2.000000 sec\n" "CorWatt\tPkgWatt\n" "1.00\t10.00\n" "2.00\t20.00\n" "0.00\n",
        encoding="utf-8",
    )

    stats = parse_power_log(log_path)

    assert stats["power_sample_count"] == 2
    assert stats["power_window_sec"] == "2.000000"
    assert stats["avg_pkg_watt"] == "15.000000"
    assert stats["max_pkg_watt"] == "20.000000"
    assert stats["avg_cor_watt"] == "1.500000"


def test_parse_power_log_accepts_explicit_duration_without_header(tmp_path):
    log_path = tmp_path / "turbostat_no_duration.log"
    log_path.write_text(
        "CorWatt\tPkgWatt\n" "1.00\t10.00\n" "2.00\t20.00\n",
        encoding="utf-8",
    )

    stats = parse_power_log(log_path, duration_sec=3.5)

    assert stats["power_sample_count"] == 2
    assert stats["power_window_sec"] == "3.500000"
    assert stats["avg_pkg_watt"] == "15.000000"
    assert stats["avg_cor_watt"] == "1.500000"


def test_parse_power_log_accepts_rocm_smi_style_samples(tmp_path):
    log_path = tmp_path / "rocm_smi.log"
    log_path.write_text(
        "2.500000 sec\n" "GFXWatt\n" "3.00\n" "5.00\n",
        encoding="utf-8",
    )

    stats = parse_power_log(log_path)

    assert stats["power_sample_count"] == 2
    assert stats["power_window_sec"] == "2.500000"
    assert stats["avg_gfx_watt"] == "4.000000"
    assert stats["avg_pkg_watt"] == ""


def test_parse_power_log_accepts_summary_headers_without_pkg_gfx_aliasing(tmp_path):
    log_path = tmp_path / "summary.log"
    log_path.write_text(
        "1.500000 sec\n"
        "CorWatt\tPkgWatt\n"
        "0.90\t6.00\n"
        "CorWatt\tPkgWatt\n"
        "1.10\t8.00\n",
        encoding="utf-8",
    )

    stats = parse_power_log(log_path)

    assert stats["power_sample_count"] == 2
    assert stats["avg_pkg_watt"] == "7.000000"
    assert stats["avg_cor_watt"] == "1.000000"


def test_discover_powercap_rapl_zones_reads_package_and_core(tmp_path):
    package_dir = tmp_path / "intel-rapl:0"
    core_dir = tmp_path / "intel-rapl:0:0"
    package_dir.mkdir(parents=True)
    core_dir.mkdir(parents=True)
    (package_dir / "name").write_text("package-0\n", encoding="utf-8")
    (package_dir / "energy_uj").write_text("1000\n", encoding="utf-8")
    (package_dir / "max_energy_range_uj").write_text("999999\n", encoding="utf-8")
    (core_dir / "name").write_text("core\n", encoding="utf-8")
    (core_dir / "energy_uj").write_text("200\n", encoding="utf-8")
    (core_dir / "max_energy_range_uj").write_text("999999\n", encoding="utf-8")

    zones = discover_powercap_rapl_zones(tmp_path)

    assert [zone["name"] for zone in zones["package"]] == ["package-0"]
    assert [zone["name"] for zone in zones["core"]] == ["core"]
    assert zones["package"][0]["energy_path"].endswith("intel-rapl:0/energy_uj")
    assert zones["core"][0]["energy_path"].endswith("intel-rapl:0:0/energy_uj")


def test_summarize_powercap_rapl_aggregates_package_and_core_energy():
    zones = {
        "package": [
            {
                "name": "package-0",
                "energy_path": "/tmp/package0",
                "max_energy_range_uj": 1_000_000,
            },
            {
                "name": "package-1",
                "energy_path": "/tmp/package1",
                "max_energy_range_uj": 1_000_000,
            },
        ],
        "core": [
            {
                "name": "core",
                "energy_path": "/tmp/core0",
                "max_energy_range_uj": 1_000_000,
            }
        ],
    }
    start_snapshot = {
        "/tmp/package0": 10_000,
        "/tmp/package1": 20_000,
        "/tmp/core0": 5_000,
    }
    end_snapshot = {
        "/tmp/package0": 210_000,
        "/tmp/package1": 320_000,
        "/tmp/core0": 55_000,
    }

    stats, zone_rows = summarize_powercap_rapl(
        zones,
        start_snapshot,
        end_snapshot,
        duration_sec=2.0,
    )

    assert stats["power_sample_count"] == 2
    assert stats["power_window_sec"] == "2.000000"
    assert stats["avg_pkg_watt"] == "0.250000"
    assert stats["avg_cor_watt"] == "0.025000"
    assert stats["max_pkg_watt"] == ""
    assert [row["group"] for row in zone_rows] == ["package", "package", "core"]


def test_summarize_powercap_rapl_handles_counter_wrap():
    zones = {
        "package": [
            {
                "name": "package-0",
                "energy_path": "/tmp/package0",
                "max_energy_range_uj": 1000,
            }
        ],
        "core": [],
    }
    start_snapshot = {"/tmp/package0": 900}
    end_snapshot = {"/tmp/package0": 100}

    stats, _ = summarize_powercap_rapl(
        zones,
        start_snapshot,
        end_snapshot,
        duration_sec=0.1,
    )

    assert stats["avg_pkg_watt"] == "0.002000"


def test_supported_topologies_for_seq_len_avoids_operator_package_import(
    monkeypatch,
):
    for module_name in list(sys.modules):
        if module_name == "iron.operators" or module_name.startswith("iron.operators."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    config = load_encoder_pipeline_config(str(CONFIG_FILE), 64)
    supported = supported_topologies_for_seq_len(config, 64)

    assert supported
    assert "iron.operators" not in sys.modules
    assert "iron.operators.encoder_pipeline.placements" not in sys.modules
    assert load_encoder_pipeline_topology_placements()


def test_supported_topologies_for_seq_len_accepts_legacy_alias_and_canonical_filters():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 64)

    legacy_filtered = supported_topologies_for_seq_len(
        config,
        64,
        candidate_ids=parse_candidate_topology_ids("2ps_4pffn"),
    )
    canonical_filtered = supported_topologies_for_seq_len(
        config,
        64,
        candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps2_ph1_pffn4"),
    )

    assert {topology_id(topology) for topology in legacy_filtered} == {
        "seq32_kv64__ps2_ph1_pffn4"
    }
    assert {topology_id(topology) for topology in canonical_filtered} == {
        "seq32_kv64__ps2_ph1_pffn4"
    }


def test_supported_topologies_for_seq_len_discovers_both_families_when_legal():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)

    families = {
        topology.family_id for topology in supported_topologies_for_seq_len(config, 128)
    }

    assert families == {"seq32_kv64", "seq64_kv32"}


@pytest.mark.parametrize(
    "config_path,seq_len",
    [
        (CONFIG_FILE, 128),
        (TEST_DIR / "config" / "config_bert_large.json", 64),
    ],
    ids=["bert_base_12h", "bert_large_16h"],
)
def test_supported_topologies_for_seq_len_discovers_both_families_for_12_and_16_heads(
    config_path, seq_len
):
    config = load_encoder_pipeline_config(str(config_path), seq_len)

    families = {
        topology.family_id
        for topology in supported_topologies_for_seq_len(config, seq_len)
    }

    assert families == {"seq32_kv64", "seq64_kv32"}


def test_supported_topologies_for_config_family_remains_fixed_to_config_family():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)

    families = {
        topology.family_id
        for topology in supported_topologies_for_config_family(config, 128)
    }

    assert families == {"seq32_kv64"}


def test_supported_topologies_for_seq_len_alias_filter_matches_both_families():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)

    filtered = supported_topologies_for_seq_len(
        config,
        128,
        candidate_ids=parse_candidate_topology_ids("2ps_4pffn"),
    )

    assert {topology_id(topology) for topology in filtered} == {
        "seq32_kv64__ps2_ph1_pffn4",
        "seq64_kv32__ps2_ph1_pffn4",
    }


def test_supported_topologies_for_seq_len_exposes_unique_ids_across_tile_families():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)

    filtered = supported_topologies_for_seq_len(
        config,
        128,
        candidate_ids=parse_candidate_topology_ids("2ps_4pffn"),
    )

    assert {topology.family_id for topology in filtered} == {
        "seq32_kv64",
        "seq64_kv32",
    }
    assert {topology_id(topology) for topology in filtered} == {
        "seq32_kv64__ps2_ph1_pffn4",
        "seq64_kv32__ps2_ph1_pffn4",
    }


def test_supported_topologies_expose_compute_tile_utilization_counts():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)
    supported = supported_topologies_for_seq_len(config, 128)

    one_ps = next(
        topology
        for topology in supported
        if topology.parallel_seq == 1
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
        and topology.family_id == "seq32_kv64"
    )
    four_ps = next(
        topology
        for topology in supported
        if topology.parallel_seq == 4
        and topology.parallel_heads == 1
        and topology.parallel_ffn == 1
        and topology.family_id == "seq32_kv64"
    )

    assert one_ps.compute_tile_count == 8
    assert four_ps.compute_tile_count == 32
    assert one_ps.utilization_fraction == pytest.approx(0.25)
    assert four_ps.utilization_fraction == pytest.approx(1.0)


def test_supported_topologies_for_seq_len_omits_illegal_mirrored_shapes():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 64)

    ids = {
        topology_id(topology)
        for topology in supported_topologies_for_seq_len(config, 64)
    }

    assert "seq64_kv32__ps1_ph1_pffn1" in ids
    assert "seq64_kv32__ps4_ph1_pffn1" not in ids


def test_is_transient_npu_startup_failure_matches_known_xrt_open_error():
    case = {"mode": "npu"}
    result = subprocess.CompletedProcess(
        args=["python3", "npu_inference.py"],
        returncode=1,
        stdout="Selected topology: seq_len=64 topology=seq32_kv64__ps2_ph1_pffn4",
        stderr=(
            "RuntimeError: mmap(addr=0x7fa44c000000, len=67108864, prot=3, "
            "flags=8209, offset=4294967296) failed (err=-11): "
            "Resource temporarily unavailable\n"
            "self._device = pyxrt.device(0)\n"
            "_DefaultNPURuntime = CachedXRTRuntime()\n"
            "from aie.utils import DefaultNPURuntime\n"
        ),
    )

    assert is_transient_npu_startup_failure(case, result) is True


def test_run_case_retries_transient_npu_startup_failure_and_remeasures_idle_power(
    tmp_path, monkeypatch
):
    sleep_calls = []
    idle_calls = []
    build_calls = []
    child_csv = tmp_path / "logs" / "toy_npu_seq64.csv"
    power_log = tmp_path / "logs" / "toy_npu_seq64_power.log"
    idle_log = tmp_path / "logs" / "toy_npu_seq64_idle_power.log"

    monkeypatch.setattr(
        automated_benchmark,
        "case_command",
        lambda *args, **kwargs: ["python3", "npu_inference.py", "--seq-lens", "64"],
    )
    monkeypatch.setattr(
        automated_benchmark,
        "resolve_power_backend",
        lambda requested_backend, mode: "turbostat",
    )
    monkeypatch.setattr(
        automated_benchmark.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    def fake_measure_idle_power(args, case, logs_dir, power_backend):
        idle_calls.append((case["case_id"], power_backend))
        idle_log.parent.mkdir(parents=True, exist_ok=True)
        idle_log.write_text(f"idle attempt {len(idle_calls)}\n", encoding="utf-8")
        stats = empty_power_stats(
            sample_count_key="idle_power_sample_count",
            window_key="idle_power_window_sec",
        )
        stats["idle_power_sample_count"] = 1
        stats["idle_power_window_sec"] = "5.000000"
        stats["avg_pkg_watt"] = f"{8 + len(idle_calls):.6f}"
        return stats, idle_log

    monkeypatch.setattr(
        automated_benchmark,
        "measure_idle_power",
        fake_measure_idle_power,
    )

    results = iter(
        [
            subprocess.CompletedProcess(
                args=["python3", "npu_inference.py"],
                returncode=1,
                stdout="Selected topology: seq_len=64 topology=seq32_kv64__ps2_ph1_pffn4",
                stderr=(
                    "RuntimeError: mmap(addr=0x7fa44c000000, len=67108864, prot=3, "
                    "flags=8209, offset=4294967296) failed (err=-11): "
                    "Resource temporarily unavailable\n"
                    "self._device = pyxrt.device(0)\n"
                    "_DefaultNPURuntime = CachedXRTRuntime()\n"
                    "from aie.utils import DefaultNPURuntime\n"
                ),
            ),
            subprocess.CompletedProcess(
                args=["python3", "npu_inference.py"],
                returncode=0,
                stdout="ok",
                stderr="",
            ),
        ]
    )

    def fake_subprocess_run(command, cwd, text, capture_output, check):
        result = next(results)
        if result.returncode == 0:
            child_csv.parent.mkdir(parents=True, exist_ok=True)
            child_csv.write_text("case_id,avg_latency_ms\n", encoding="utf-8")
            power_log.write_text("2.000000 sec\nPkgWatt\n20.0\n", encoding="utf-8")
        return result

    monkeypatch.setattr(automated_benchmark.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        automated_benchmark,
        "parse_child_row",
        lambda path: {
            "study_id": "bert-base-uncased",
            "seq_len": "64",
            "num_threads": "12",
            "dtype": "bfloat16",
            "num_samples": "1",
            "runs_per_sample": "100",
            "warmup_runs": "10",
            "measured_inference_count": "100",
            "timed_total_sec": "2.000000",
            "throughput_inferences_per_sec": "50.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "5.000000e+10",
            "topology_id": "seq32_kv64__ps2_ph1_pffn4",
            "parallel_seq": "2",
            "parallel_heads": "1",
            "parallel_ffn": "4",
            "min_latency_ms": "20.000000",
            "avg_latency_ms": "20.000000",
            "max_latency_ms": "20.000000",
            "power_backend": "turbostat",
            "power_sample_count": "2",
            "power_window_sec": "2.000000",
            "avg_pkg_watt": "20.000000",
            "max_pkg_watt": "21.000000",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "power_log": str(power_log),
        },
    )

    def fake_build_suite_row(
        child_row, case, cooldown_stats, power_backend, idle_stats, idle_log_path
    ):
        build_calls.append(
            {
                "power_backend": power_backend,
                "idle_pkg_watt": idle_stats["avg_pkg_watt"],
                "idle_log_path": str(idle_log_path),
            }
        )
        return {
            "case_id": case["case_id"],
            "avg_latency_ms": child_row["avg_latency_ms"],
        }

    monkeypatch.setattr(automated_benchmark, "build_suite_row", fake_build_suite_row)

    args = SimpleNamespace(
        power_backend="auto",
        power_interval_sec=0.5,
        npu_idle_baseline_sec=5.0,
    )
    case = {
        "case_id": "toy_npu_seq64",
        "mode": "npu",
        "seq_len": 64,
        "num_threads": 12,
        "study_id": "bert-base-uncased",
    }
    cooldown_stats = {
        "cooldown_wait_sec": 0.0,
        "cooldown_start_temp_c": None,
        "cooldown_end_temp_c": None,
        "cooldown_temp_source": "",
    }

    suite_row = run_case(args, case, tmp_path / "logs", cooldown_stats)

    assert suite_row["case_id"] == "toy_npu_seq64"
    assert idle_calls == [
        ("toy_npu_seq64", "turbostat"),
        ("toy_npu_seq64", "turbostat"),
    ]
    assert sleep_calls == [15.0]
    assert build_calls == [
        {
            "power_backend": "turbostat",
            "idle_pkg_watt": "10.000000",
            "idle_log_path": str(idle_log),
        }
    ]


def test_resolve_power_backend_uses_mode_specific_defaults():
    assert resolve_power_backend("auto", "cpu") == "powercap-rapl"
    assert resolve_power_backend("auto", "npu") == "turbostat"
    assert resolve_power_backend("auto", "igpu") == "rocm-smi"


def test_run_case_operator_runlist_uses_pipe_capture_and_writes_process_log(
    tmp_path, monkeypatch
):
    child_csv = tmp_path / "logs" / "toy_npu_operator_runlist_seq64.csv"
    process_log = tmp_path / "logs" / "toy_npu_operator_runlist_seq64_process.log"
    subprocess_calls = []

    monkeypatch.setattr(
        automated_benchmark,
        "case_command",
        lambda *args, **kwargs: [
            "python3",
            "npu_inference_import_main.py",
            "--execution-mode",
            "operator_runlist",
        ],
    )
    monkeypatch.setattr(
        automated_benchmark,
        "resolve_power_backend",
        lambda requested_backend, mode: "none",
    )

    def fake_subprocess_run(command, cwd, text, capture_output, check):
        subprocess_calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "check": check,
            }
        )
        child_csv.parent.mkdir(parents=True, exist_ok=True)
        child_csv.write_text("case_id,avg_latency_ms\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="operator stdout\n",
            stderr="operator stderr\n",
        )

    monkeypatch.setattr(automated_benchmark.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        automated_benchmark,
        "parse_child_row",
        lambda path: {
            "study_id": "bert-base-uncased",
            "seq_len": "64",
            "num_threads": "12",
            "dtype": "bfloat16",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.100000",
            "throughput_inferences_per_sec": "10.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "1.000000e+10",
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "min_latency_ms": "100.000000",
            "avg_latency_ms": "100.000000",
            "max_latency_ms": "100.000000",
            "power_backend": "none",
            "power_sample_count": "",
            "power_window_sec": "",
            "avg_pkg_watt": "",
            "max_pkg_watt": "",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "power_log": "",
        },
    )
    monkeypatch.setattr(
        automated_benchmark,
        "build_suite_row",
        lambda child_row, case, cooldown_stats, power_backend, idle_stats, idle_log_path: {
            "case_id": case["case_id"],
            "avg_latency_ms": child_row["avg_latency_ms"],
        },
    )

    args = SimpleNamespace(
        power_backend="none",
        power_interval_sec=0.5,
        npu_idle_baseline_sec=5.0,
    )
    case = {
        "case_id": "toy_npu_operator_runlist_seq64",
        "mode": "npu",
        "execution_mode": "operator_runlist",
        "seq_len": 64,
        "num_threads": 12,
        "study_id": "bert-base-uncased",
    }
    cooldown_stats = {
        "cooldown_wait_sec": 0.0,
        "cooldown_start_temp_c": None,
        "cooldown_end_temp_c": None,
        "cooldown_temp_source": "",
    }

    suite_row = run_case(args, case, tmp_path / "logs", cooldown_stats)

    assert suite_row["case_id"] == "toy_npu_operator_runlist_seq64"
    assert subprocess_calls == [
        {
            "command": [
                "python3",
                "npu_inference_import_main.py",
                "--execution-mode",
                "operator_runlist",
            ],
            "capture_output": True,
            "check": False,
        }
    ]
    assert (
        process_log.read_text(encoding="utf-8") == "operator stdout\noperator stderr\n"
    )


def test_run_case_retries_transient_operator_runlist_failure(tmp_path, monkeypatch):
    child_csv = tmp_path / "logs" / "toy_npu_operator_runlist_seq64.csv"
    sleep_calls = []
    subprocess_calls = []
    results = iter(
        [
            subprocess.CompletedProcess(
                args=["python3", "npu_inference_import_main.py"],
                returncode=1,
                stdout=(
                    "Using host CPU threads: 12\n"
                    "Benchmarking local BertModel(add_pooling_layer=False) encoder "
                    "with execution mode operator_runlist\n"
                    "Starting NPU benchmark: seq_len=64\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["python3", "npu_inference_import_main.py"],
                returncode=0,
                stdout="operator stdout\n",
                stderr="",
            ),
        ]
    )

    monkeypatch.setattr(
        automated_benchmark,
        "case_command",
        lambda *args, **kwargs: [
            "python3",
            "npu_inference_import_main.py",
            "--execution-mode",
            "operator_runlist",
        ],
    )
    monkeypatch.setattr(
        automated_benchmark,
        "resolve_power_backend",
        lambda requested_backend, mode: "none",
    )
    monkeypatch.setattr(
        automated_benchmark.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    def fake_subprocess_run(command, cwd, text, capture_output, check):
        subprocess_calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "check": check,
            }
        )
        result = next(results)
        if result.returncode == 0:
            child_csv.parent.mkdir(parents=True, exist_ok=True)
            child_csv.write_text("case_id,avg_latency_ms\n", encoding="utf-8")
        return result

    monkeypatch.setattr(automated_benchmark.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        automated_benchmark,
        "parse_child_row",
        lambda path: {
            "study_id": "bert-base-uncased",
            "seq_len": "64",
            "num_threads": "12",
            "dtype": "bfloat16",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.100000",
            "throughput_inferences_per_sec": "10.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "1.000000e+10",
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "min_latency_ms": "100.000000",
            "avg_latency_ms": "100.000000",
            "max_latency_ms": "100.000000",
            "power_backend": "none",
            "power_sample_count": "",
            "power_window_sec": "",
            "avg_pkg_watt": "",
            "max_pkg_watt": "",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "power_log": "",
        },
    )
    monkeypatch.setattr(
        automated_benchmark,
        "build_suite_row",
        lambda child_row, case, cooldown_stats, power_backend, idle_stats, idle_log_path: {
            "case_id": case["case_id"],
            "avg_latency_ms": child_row["avg_latency_ms"],
        },
    )

    args = SimpleNamespace(
        power_backend="none",
        power_interval_sec=0.5,
        npu_idle_baseline_sec=5.0,
    )
    case = {
        "case_id": "toy_npu_operator_runlist_seq64",
        "mode": "npu",
        "execution_mode": "operator_runlist",
        "seq_len": 64,
        "num_threads": 12,
        "study_id": "bert-base-uncased",
    }
    cooldown_stats = {
        "cooldown_wait_sec": 0.0,
        "cooldown_start_temp_c": None,
        "cooldown_end_temp_c": None,
        "cooldown_temp_source": "",
    }

    suite_row = run_case(args, case, tmp_path / "logs", cooldown_stats)

    assert suite_row["case_id"] == "toy_npu_operator_runlist_seq64"
    assert len(subprocess_calls) == 2
    assert sleep_calls == [5.0]


@pytest.mark.parametrize("invalid_backend", ["rocm-smi", "powercap-rapl", "turbostat"])
def test_resolve_power_backend_rejects_explicit_tool_selection(invalid_backend):
    with pytest.raises(ValueError, match="Unsupported power_backend"):
        resolve_power_backend(invalid_backend, "cpu")


def test_build_suite_row_uses_gfx_power_for_igpu_energy():
    child_row = {
        "study_id": "toy-model",
        "seq_len": "64",
        "num_threads": "8",
        "dtype": "float16",
        "num_samples": "1",
        "runs_per_sample": "1",
        "warmup_runs": "0",
        "measured_inference_count": "1",
        "timed_total_sec": "0.010000",
        "throughput_inferences_per_sec": "100.000000",
        "model_type": "bert",
        "shape": "(1, 64, 768)",
        "estimated_flops_per_inference": "1.000000e+09",
        "throughput_flops_per_sec": "1.000000e+11",
        "topology_id": "",
        "parallel_seq": "",
        "parallel_heads": "",
        "parallel_ffn": "",
        "min_latency_ms": "10.000000",
        "avg_latency_ms": "10.000000",
        "max_latency_ms": "10.000000",
        "power_backend": "rocm-smi",
        "power_sample_count": "2",
        "power_window_sec": "0.020000",
        "avg_pkg_watt": "",
        "max_pkg_watt": "",
        "avg_cor_watt": "",
        "max_cor_watt": "",
        "avg_gfx_watt": "20.000000",
        "max_gfx_watt": "24.000000",
        "avg_ram_watt": "",
        "max_ram_watt": "",
        "power_log": "toy_igpu_power.log",
        "benchmark_csv": "toy_igpu.csv",
    }
    case = {"case_id": "toy_igpu_seq64", "mode": "igpu"}
    cooldown_stats = {
        "cooldown_wait_sec": 0.0,
        "cooldown_start_temp_c": None,
        "cooldown_end_temp_c": None,
        "cooldown_temp_source": "",
    }
    idle_stats = empty_power_stats(
        sample_count_key="idle_power_sample_count",
        window_key="idle_power_window_sec",
    )
    idle_stats["idle_power_sample_count"] = 1
    idle_stats["idle_power_window_sec"] = "0.010000"
    idle_stats["avg_gfx_watt"] = "8.000000"

    suite_row = build_suite_row(
        child_row,
        case,
        cooldown_stats,
        "rocm-smi",
        idle_stats,
        None,
    )

    assert suite_row["avg_pkg_watt"] == ""
    assert suite_row["avg_gfx_watt"] == "20.000000"
    assert suite_row["pseudo_device_avg_pkg_watt"] == "12.000000"
    assert suite_row["estimated_gflops_per_watt_sec"] == "2.500000"
    assert suite_row["pseudo_device_estimated_gflops_per_watt_sec"] == "4.166667"
    assert suite_row["benchmark_mode"] == DEFAULT_BENCHMARK_MODE
    assert suite_row["execution_mode"] == "host_hf"


def test_build_suite_row_uses_pkg_power_for_npu_energy():
    child_row = {
        "study_id": "toy-model",
        "benchmark_mode": "synthetic_dense",
        "execution_mode": "encoder_pipeline",
        "seq_len": "64",
        "num_threads": "8",
        "dtype": "bfloat16",
        "num_samples": "1",
        "runs_per_sample": "1",
        "warmup_runs": "0",
        "measured_inference_count": "1",
        "timed_total_sec": "0.020000",
        "throughput_inferences_per_sec": "50.000000",
        "model_type": "bert",
        "shape": "(1, 64, 768)",
        "estimated_flops_per_inference": "1.000000e+09",
        "throughput_flops_per_sec": "5.000000e+10",
        "topology_id": "seq32_kv64__ps2_ph1_pffn4",
        "parallel_seq": "2",
        "parallel_heads": "1",
        "parallel_ffn": "4",
        "compute_tile_count": "16",
        "compute_tile_utilization_fraction": "0.500000",
        "compile_setup_time_ms": "123.000000",
        "topology_selection_time_ms": "45.000000",
        "topology_cache_status": "cache_hit",
        "cached_steady_state_avg_latency_ms": "20.000000",
        "avg_embedding_latency_ms": "1.500000",
        "avg_qkv_projection_latency_ms": "2.500000",
        "avg_encoder_pipeline_latency_ms": "12.500000",
        "avg_operator_runlist_latency_ms": "",
        "avg_host_preprocess_latency_ms": "",
        "avg_npu_gemm_latency_ms": "",
        "avg_host_postprocess_latency_ms": "",
        "avg_device_sync_latency_ms": "",
        "npu_dispatch_count": "",
        "npu_unique_instruction_binary_count": "",
        "min_latency_ms": "20.000000",
        "avg_latency_ms": "20.000000",
        "max_latency_ms": "20.000000",
        "power_backend": "turbostat",
        "power_sample_count": "2",
        "power_window_sec": "0.020000",
        "avg_pkg_watt": "18.000000",
        "max_pkg_watt": "20.000000",
        "avg_cor_watt": "",
        "max_cor_watt": "",
        "avg_gfx_watt": "",
        "max_gfx_watt": "",
        "avg_ram_watt": "",
        "max_ram_watt": "",
        "power_log": "toy_npu_power.log",
        "benchmark_csv": "toy_npu.csv",
    }
    case = {"case_id": "toy_npu_seq64", "mode": "npu"}
    cooldown_stats = {
        "cooldown_wait_sec": 0.0,
        "cooldown_start_temp_c": None,
        "cooldown_end_temp_c": None,
        "cooldown_temp_source": "",
    }
    idle_stats = empty_power_stats(
        sample_count_key="idle_power_sample_count",
        window_key="idle_power_window_sec",
    )
    idle_stats["idle_power_sample_count"] = 1
    idle_stats["idle_power_window_sec"] = "0.010000"
    idle_stats["avg_pkg_watt"] = "8.000000"

    suite_row = build_suite_row(
        child_row,
        case,
        cooldown_stats,
        "turbostat",
        idle_stats,
        None,
    )

    assert suite_row["avg_pkg_watt"] == "18.000000"
    assert suite_row["pseudo_device_avg_pkg_watt"] == "10.000000"
    assert suite_row["pseudo_npu_avg_pkg_watt"] == "10.000000"
    assert suite_row["estimated_gflops_per_watt_sec"] == "2.777778"
    assert suite_row["pseudo_npu_estimated_gflops_per_watt_sec"] == "5.000000"
    assert suite_row["compute_tile_count"] == "16"
    assert suite_row["compute_tile_utilization_fraction"] == "0.500000"
    assert suite_row["compile_setup_time_ms"] == "123.000000"
    assert suite_row["topology_selection_time_ms"] == "45.000000"
    assert suite_row["topology_cache_status"] == "cache_hit"
    assert suite_row["cached_steady_state_avg_latency_ms"] == "20.000000"
    assert suite_row["avg_embedding_latency_ms"] == "1.500000"
    assert suite_row["avg_qkv_projection_latency_ms"] == "2.500000"
    assert suite_row["avg_encoder_pipeline_latency_ms"] == "12.500000"
    assert suite_row["avg_operator_runlist_latency_ms"] == ""
    assert suite_row["avg_npu_gemm_latency_ms"] == ""
    assert suite_row["benchmark_mode"] == DEFAULT_BENCHMARK_MODE
    assert suite_row["execution_mode"] == "encoder_pipeline"


def test_npu_benchmark_with_config_reports_stage_latency_breakdown(monkeypatch):
    class FakeTopology(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

    class FakeContext:
        def __init__(self):
            self.reset_runtime_calls = 0

        def reset_runtime(self):
            self.reset_runtime_calls += 1

    class FakeModel:
        def __init__(self):
            self.dtype = torch.bfloat16
            self.warmup_calls = 0
            self.timed_calls = 0

        def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
            self.warmup_calls += 1
            return torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16)

        def forward_with_stage_timings(
            self, input_ids, token_type_ids=None, attention_mask=None
        ):
            self.timed_calls += 1
            return (
                torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16),
                {
                    "embedding_sec": 0.001,
                    "qkv_projection_sec": 0.002,
                    "encoder_pipeline_sec": 0.003,
                },
            )

    fake_model = FakeModel()
    fake_context = FakeContext()

    monkeypatch.setattr(
        npu_inference,
        "build_npu_encoder_model",
        lambda weights_file_path, config, seq_len, execution_mode="encoder_pipeline", disable_all_biases=False: (
            fake_model,
            fake_context,
            7.5,
        ),
    )
    monkeypatch.setattr(
        npu_inference,
        "prepare_benchmark_samples",
        lambda **kwargs: [
            {
                "input_ids": torch.ones((1, 64), dtype=torch.long),
                "token_type_ids": torch.zeros((1, 64), dtype=torch.long),
            }
        ],
    )
    monkeypatch.setattr(
        npu_inference,
        "estimate_encoder_forward_flops",
        lambda model_config, seq_len: 1.0e9,
    )

    perf_values = iter([0.0, 0.010, 0.020, 0.030])
    monkeypatch.setattr(npu_inference.time, "perf_counter", lambda: next(perf_values))

    @contextmanager
    def fake_power_monitor(*args, **kwargs):
        yield SimpleNamespace(
            stats=empty_power_stats(),
        )

    monkeypatch.setattr(npu_inference, "create_power_monitor", fake_power_monitor)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model_type="bert",
            vocab_size=30522,
            pad_token_id=0,
        ),
    )
    topology = FakeTopology(
        topology_id="seq32_kv64__ps2_ph1_pffn4",
        family_id="seq32_kv64",
        seq_tile=32,
        kv_seq_tile=64,
        parallel_seq=2,
        parallel_heads=1,
        parallel_ffn=4,
        compute_tile_count=28,
        utilization_fraction=0.875,
    )

    result = npu_inference.benchmark_with_config(
        weights_file_path="model.safetensors",
        config_file_path="config.json",
        config=config,
        seq_len=64,
        texts=["sample"],
        warmup_runs=1,
        runs_per_sample=2,
        benchmark_mode="synthetic_dense",
        execution_mode="encoder_pipeline",
        topology=topology,
        power_backend="none",
        power_interval_sec=0.5,
        power_log_path=None,
    )

    assert fake_model.warmup_calls == 1
    assert fake_model.timed_calls == 2
    assert result["benchmark_mode"] == "synthetic_dense"
    assert result["execution_mode"] == "encoder_pipeline"
    assert result["compile_setup_time_ms"] == pytest.approx(7.5)
    assert result["cached_steady_state_avg_latency_ms"] == pytest.approx(10.0)
    assert result["topology_selection_time_ms"] == ""
    assert result["topology_cache_status"] == ""
    assert result["avg_embedding_latency_ms"] == pytest.approx(1.0)
    assert result["avg_qkv_projection_latency_ms"] == pytest.approx(2.0)
    assert result["avg_encoder_pipeline_latency_ms"] == pytest.approx(3.0)
    assert result["avg_operator_runlist_latency_ms"] == ""
    assert result["avg_host_preprocess_latency_ms"] == ""
    assert result["avg_npu_gemm_latency_ms"] == ""
    assert result["avg_host_postprocess_latency_ms"] == ""
    assert result["npu_dispatch_count"] == ""
    assert result["avg_latency_ms"] == pytest.approx(10.0)
    assert fake_context.reset_runtime_calls == 1


def test_npu_benchmark_with_config_reports_gemm_only_breakdown(monkeypatch):
    class FakeContext:
        def __init__(self):
            self.reset_runtime_calls = 0

        def reset_runtime(self):
            self.reset_runtime_calls += 1

    class FakeModel:
        def __init__(self):
            self.dtype = torch.bfloat16

        def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
            return torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16)

        def forward_with_stage_timings(
            self, input_ids, token_type_ids=None, attention_mask=None
        ):
            return (
                torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16),
                {
                    "embedding_sec": 0.001,
                    "qkv_projection_sec": 0.002,
                    "host_preprocess_sec": 0.0005,
                    "npu_gemm_sec": 0.006,
                    "host_postprocess_sec": 0.0015,
                    "device_sync_sec": 0.00025,
                    "npu_dispatch_count": 28,
                    "npu_unique_instruction_binary_count": 6,
                },
            )

    monkeypatch.setattr(
        npu_inference,
        "build_npu_encoder_model",
        lambda weights_file_path, config, seq_len, execution_mode="encoder_pipeline", disable_all_biases=False: (
            FakeModel(),
            FakeContext(),
            12.5,
        ),
    )
    monkeypatch.setattr(
        npu_inference,
        "prepare_benchmark_samples",
        lambda **kwargs: [
            {
                "input_ids": torch.ones((1, 64), dtype=torch.long),
                "token_type_ids": torch.zeros((1, 64), dtype=torch.long),
            }
        ],
    )
    monkeypatch.setattr(
        npu_inference,
        "estimate_encoder_forward_flops",
        lambda model_config, seq_len: 1.0e9,
    )

    perf_values = iter([0.0, 0.010, 0.020, 0.030])
    monkeypatch.setattr(npu_inference.time, "perf_counter", lambda: next(perf_values))

    @contextmanager
    def fake_power_monitor(*args, **kwargs):
        yield SimpleNamespace(
            stats=empty_power_stats(),
        )

    monkeypatch.setattr(npu_inference, "create_power_monitor", fake_power_monitor)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model_type="bert",
            vocab_size=30522,
            pad_token_id=0,
        ),
    )

    result = npu_inference.benchmark_with_config(
        weights_file_path="model.safetensors",
        config_file_path="config.json",
        config=config,
        seq_len=64,
        texts=["sample"],
        warmup_runs=1,
        runs_per_sample=2,
        benchmark_mode="synthetic_dense",
        execution_mode="gemm_only",
        topology=None,
        power_backend="none",
        power_interval_sec=0.5,
        power_log_path=None,
    )

    assert result["execution_mode"] == "gemm_only"
    assert result["topology_id"] == ""
    assert result["avg_qkv_projection_latency_ms"] == pytest.approx(2.0)
    assert result["avg_encoder_pipeline_latency_ms"] == ""
    assert result["avg_operator_runlist_latency_ms"] == ""
    assert result["avg_host_preprocess_latency_ms"] == pytest.approx(0.5)
    assert result["avg_npu_gemm_latency_ms"] == pytest.approx(6.0)
    assert result["avg_host_postprocess_latency_ms"] == pytest.approx(1.5)
    assert result["avg_device_sync_latency_ms"] == pytest.approx(0.25)
    assert result["npu_dispatch_count"] == 28
    assert result["npu_unique_instruction_binary_count"] == 6


def test_gemm_only_layer_uses_one_runtime_xclbin():
    from src.block.transformer_gemm_only import BertEncoderGemmOnlyLayer

    class DummyContext:
        def __init__(self):
            self.operators = []
            self.static_data_pool = {}
            self.base_dir = Path(__file__).resolve().parents[3]
            self.device_manager = type(
                "_DummyDeviceManager",
                (),
                {"device_str": staticmethod(lambda: "npu1_4col")},
            )()

        def register_operator(self, operator, skip_add_to_list=False):
            operator.context = self
            if not skip_add_to_list:
                self.operators.append(operator)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            layer_norm_eps=1.0e-12,
        ),
        aie_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    layer = BertEncoderGemmOnlyLayer(config, seq_len=64, context=DummyContext())
    ops = [
        layer.qkv_proj,
        layer.attn_scores,
        layer.attn_output,
        layer.out_proj,
        layer.ffn_up,
        layer.ffn_down,
    ]

    runtime_xclbins = {id(op.runtime_xclbin_artifact) for op in ops}
    declared_xclbins = {id(op.xclbin_artifact) for op in ops}
    insts_xclbin_inputs = {id(op.insts_artifact.xclbin_input) for op in ops}
    runtime_paths = {op.runtime_xclbin_artifact.path for op in ops}

    assert layer.UNIQUE_XCLBIN_COUNT == 1
    assert len(runtime_xclbins) == 1
    assert len(declared_xclbins) == 1
    assert len(insts_xclbin_inputs) == 1
    assert len(runtime_paths) == 1


def test_npu_benchmark_with_config_reports_operator_runlist_breakdown(monkeypatch):
    class FakeContext:
        def __init__(self):
            self.reset_runtime_calls = 0

        def reset_runtime(self):
            self.reset_runtime_calls += 1

    class FakeModel:
        def __init__(self):
            self.dtype = torch.bfloat16

        def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
            return torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16)

        def forward_with_stage_timings(
            self, input_ids, token_type_ids=None, attention_mask=None
        ):
            return (
                torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16),
                {
                    "embedding_sec": 0.001,
                    "operator_runlist_sec": 0.007,
                    "npu_dispatch_count": 13,
                    "npu_unique_instruction_binary_count": 10,
                },
            )

    fake_context = FakeContext()
    monkeypatch.setattr(
        npu_inference,
        "build_npu_encoder_model",
        lambda weights_file_path, config, seq_len, execution_mode="encoder_pipeline", disable_all_biases=False: (
            FakeModel(),
            fake_context,
            15.0,
        ),
    )
    monkeypatch.setattr(
        npu_inference,
        "prepare_benchmark_samples",
        lambda **kwargs: [
            {
                "input_ids": torch.ones((1, 64), dtype=torch.long),
                "token_type_ids": torch.zeros((1, 64), dtype=torch.long),
            }
        ],
    )
    monkeypatch.setattr(
        npu_inference,
        "estimate_encoder_forward_flops",
        lambda model_config, seq_len: 1.0e9,
    )

    perf_values = iter([0.0, 0.010, 0.020, 0.030])
    monkeypatch.setattr(npu_inference.time, "perf_counter", lambda: next(perf_values))

    @contextmanager
    def fake_power_monitor(*args, **kwargs):
        yield SimpleNamespace(
            stats=empty_power_stats(),
        )

    monkeypatch.setattr(npu_inference, "create_power_monitor", fake_power_monitor)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model_type="bert",
            vocab_size=30522,
            pad_token_id=0,
        ),
    )

    result = npu_inference.benchmark_with_config(
        weights_file_path="model.safetensors",
        config_file_path="config.json",
        config=config,
        seq_len=64,
        texts=["sample"],
        warmup_runs=1,
        runs_per_sample=2,
        benchmark_mode="synthetic_dense",
        execution_mode="operator_runlist",
        topology=None,
        power_backend="none",
        power_interval_sec=0.5,
        power_log_path=None,
    )

    assert result["execution_mode"] == "operator_runlist"
    assert result["topology_id"] == ""
    assert result["avg_encoder_pipeline_latency_ms"] == ""
    assert result["avg_operator_runlist_latency_ms"] == pytest.approx(7.0)
    assert result["avg_npu_gemm_latency_ms"] == ""
    assert result["npu_dispatch_count"] == 13
    assert result["npu_unique_instruction_binary_count"] == 10


def test_npu_benchmark_with_config_resets_runtime_after_failure(monkeypatch):
    class FakeContext:
        def __init__(self):
            self.reset_runtime_calls = 0

        def reset_runtime(self):
            self.reset_runtime_calls += 1

    class FakeModel:
        def __init__(self):
            self.dtype = torch.bfloat16

        def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
            return torch.zeros((1, input_ids.shape[1], 8), dtype=torch.bfloat16)

        def forward_with_stage_timings(
            self, input_ids, token_type_ids=None, attention_mask=None
        ):
            raise RuntimeError("boom")

    fake_context = FakeContext()

    monkeypatch.setattr(
        npu_inference,
        "build_npu_encoder_model",
        lambda weights_file_path, config, seq_len, execution_mode="encoder_pipeline", disable_all_biases=False: (
            FakeModel(),
            fake_context,
            9.0,
        ),
    )
    monkeypatch.setattr(
        npu_inference,
        "prepare_benchmark_samples",
        lambda **kwargs: [
            {
                "input_ids": torch.ones((1, 64), dtype=torch.long),
                "token_type_ids": torch.zeros((1, 64), dtype=torch.long),
            }
        ],
    )

    @contextmanager
    def fake_power_monitor(*args, **kwargs):
        yield SimpleNamespace(stats=empty_power_stats())

    monkeypatch.setattr(npu_inference, "create_power_monitor", fake_power_monitor)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model_type="bert",
            vocab_size=30522,
            pad_token_id=0,
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        npu_inference.benchmark_with_config(
            weights_file_path="model.safetensors",
            config_file_path="config.json",
            config=config,
            seq_len=64,
            texts=["sample"],
            warmup_runs=1,
            runs_per_sample=2,
            benchmark_mode="synthetic_dense",
            execution_mode="encoder_pipeline",
            topology=None,
            power_backend="none",
            power_interval_sec=0.5,
            power_log_path=None,
        )

    assert fake_context.reset_runtime_calls == 1


def test_build_npu_encoder_model_uses_backend_specific_context(monkeypatch):
    class FakeContext:
        def __init__(self, name):
            self.name = name
            self.compiled = False
            self.prepared = False

        def compile_all(self):
            self.compiled = True

        def prepare_runtime(self):
            self.prepared = True

    backend_context = FakeContext("backend")
    default_context = FakeContext("default")

    class FakeModel:
        def __init__(self):
            self.encoder_context = backend_context
            self.eval_called = False

        def assign_backbone_weights(self, weights):
            self.weights = weights

        def eval(self):
            self.eval_called = True

    monkeypatch.setattr(
        npu_inference,
        "reset_default_context",
        lambda: default_context,
    )
    monkeypatch.setattr(
        npu_inference,
        "canonicalize_local_backbone_weights",
        lambda raw_weights, model_config: dict(raw_weights),
    )
    monkeypatch.setattr(
        npu_inference,
        "extend_or_trim_position_embeddings",
        lambda weights, max_pos: None,
    )
    monkeypatch.setattr(
        npu_inference.time,
        "perf_counter",
        iter([0.0, 0.005]).__next__,
    )

    import sys
    import types

    fake_module = types.ModuleType("src.model_operator_runlist")
    fake_module.EncoderBackboneOperatorRunlist = lambda config, seq_len=512: FakeModel()
    monkeypatch.setitem(sys.modules, "src.model_operator_runlist", fake_module)
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        types.SimpleNamespace(load_file=lambda path: {"w": torch.tensor([1.0])}),
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_position_embeddings=512),
    )

    model, context, compile_setup_time_ms = npu_inference.build_npu_encoder_model(
        weights_file_path="model.safetensors",
        config=config,
        seq_len=64,
        execution_mode="operator_runlist",
    )

    assert model.eval_called is True
    assert context is backend_context
    assert backend_context.compiled is True
    assert backend_context.prepared is True
    assert default_context.compiled is False
    assert default_context.prepared is False
    assert compile_setup_time_ms == pytest.approx(5.0)


def test_operator_runlist_backbone_reuses_single_encoder_operator(monkeypatch):
    import importlib

    module = importlib.import_module("src.model_operator_runlist")

    class FakeContext:
        def __init__(self, use_runlist=True):
            self.use_runlist = use_runlist

    class FakeEmbeddings:
        def __init__(self, config):
            self.config = config
            self.word_embeddings = SimpleNamespace(weight=None)
            self.position_embeddings = SimpleNamespace(weight=None)
            self.token_type_embeddings = SimpleNamespace(weight=None)
            self.LayerNorm = SimpleNamespace(weight=None, bias=None)

        def eval(self):
            return self

    class FakeEncoder:
        def __init__(self, **kwargs):
            self.context = kwargs["context"]
            self.use_static_runtime_weights = kwargs["use_static_runtime_weights"]
            self.lazy_kernel_loading = True
            self.runlist = [1, 2, 3]
            self.kernels = {"k0": object(), "k1": object()}

    monkeypatch.setattr(module, "AIEContext", FakeContext)
    monkeypatch.setattr(module, "EncoderEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(module, "AIEBERTEncoder", FakeEncoder)

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=12,
        ),
        aie_config=SimpleNamespace(dtype=torch.bfloat16),
    )

    model = module.EncoderBackboneOperatorRunlist(config, seq_len=64)

    assert isinstance(model.encoder_context, FakeContext)
    assert isinstance(model.encoder_layer, FakeEncoder)
    assert model.encoder_context is model.encoder_layer.context
    assert model.encoder_layer.use_static_runtime_weights is False
    assert model.encoder_layer.lazy_kernel_loading is False
    assert model.num_hidden_layers == 12


def test_finalize_process_for_execution_mode_is_noop_for_non_operator_runlist(
    monkeypatch,
):
    exit_calls = []
    monkeypatch.setattr(npu_inference.os, "_exit", lambda code: exit_calls.append(code))
    monkeypatch.setattr(npu_inference.sys.stdout, "flush", lambda: None)
    monkeypatch.setattr(npu_inference.sys.stderr, "flush", lambda: None)

    npu_inference.finalize_process_for_execution_mode("encoder_pipeline")

    assert exit_calls == []


def test_finalize_process_for_execution_mode_exits_for_operator_runlist(monkeypatch):
    exit_calls = []
    flush_calls = []
    monkeypatch.setattr(npu_inference.os, "_exit", lambda code: exit_calls.append(code))
    monkeypatch.setattr(
        npu_inference.sys.stdout,
        "flush",
        lambda: flush_calls.append("stdout"),
    )
    monkeypatch.setattr(
        npu_inference.sys.stderr,
        "flush",
        lambda: flush_calls.append("stderr"),
    )

    npu_inference.finalize_process_for_execution_mode("operator_runlist")

    assert flush_calls == ["stdout", "stderr"]
    assert exit_calls == [0]


def test_validate_execution_mode_request_allows_multi_seq_for_encoder_pipeline():
    npu_inference.validate_execution_mode_request("encoder_pipeline", [64, 128])


def test_validate_execution_mode_request_rejects_multi_seq_operator_runlist():
    with pytest.raises(
        ValueError,
        match="operator_runlist currently requires one seq_len per process",
    ):
        npu_inference.validate_execution_mode_request("operator_runlist", [64, 128])


@pytest.mark.parametrize(
    ("execution_mode", "disable_all_biases", "expected"),
    [
        ("encoder_pipeline", False, False),
        ("encoder_pipeline", True, True),
        ("gemm_only", False, False),
        ("operator_runlist", False, True),
        ("operator_runlist", True, True),
    ],
)
def test_effective_disable_all_biases(execution_mode, disable_all_biases, expected):
    assert (
        npu_inference.effective_disable_all_biases(
            execution_mode,
            disable_all_biases,
        )
        is expected
    )


def test_execution_autograd_context_uses_no_grad_for_operator_runlist(monkeypatch):
    calls = []

    class DummyContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    monkeypatch.setattr(
        torch,
        "no_grad",
        lambda: calls.append("no_grad") or DummyContext(),
    )
    monkeypatch.setattr(
        torch,
        "inference_mode",
        lambda: calls.append("inference_mode") or DummyContext(),
    )

    with npu_inference.execution_autograd_context("operator_runlist"):
        pass
    with npu_inference.execution_autograd_context("encoder_pipeline"):
        pass

    assert calls == [
        "no_grad",
        "enter",
        "exit",
        "inference_mode",
        "enter",
        "exit",
    ]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--execution-mode", "operator_runlist"], True),
        (["--execution-mode=operator_runlist"], True),
        (["--execution-mode", "encoder_pipeline"], False),
        (["--study-id", "bert-base-uncased"], False),
    ],
)
def test_argv_requests_operator_runlist(argv, expected):
    assert npu_inference.argv_requests_operator_runlist(argv) is expected


def test_maybe_reexec_operator_runlist_import_main_noops_without_request(monkeypatch):
    exec_calls = []
    monkeypatch.delenv("IRON_NPU_IMPORTED_MAIN", raising=False)
    monkeypatch.setattr(
        npu_inference.os, "execve", lambda *args: exec_calls.append(args)
    )

    npu_inference.maybe_reexec_operator_runlist_import_main(
        ["npu_inference.py", "--execution-mode", "encoder_pipeline"]
    )

    assert exec_calls == []


def test_maybe_reexec_operator_runlist_import_main_reexecs(monkeypatch):
    exec_calls = []
    monkeypatch.delenv("IRON_NPU_IMPORTED_MAIN", raising=False)
    monkeypatch.setattr(
        npu_inference.sys,
        "executable",
        "/usr/bin/python3",
    )
    monkeypatch.setattr(
        npu_inference.os,
        "execvpe",
        lambda executable, argv, env: exec_calls.append((executable, argv, env)),
    )

    npu_inference.maybe_reexec_operator_runlist_import_main(
        ["npu_inference.py", "--execution-mode", "operator_runlist", "--seq-lens", "64"]
    )

    assert len(exec_calls) == 1
    executable, argv, env = exec_calls[0]
    assert executable == "/usr/bin/python3"
    assert argv[:2] == [
        "/usr/bin/python3",
        str(TEST_DIR / "npu_inference_import_main.py"),
    ]
    assert argv[2:] == ["--execution-mode", "operator_runlist", "--seq-lens", "64"]
    assert env["IRON_NPU_IMPORTED_MAIN"] == "1"


def test_build_npu_csv_row_formats_operator_runlist_fields():
    row = npu_inference._build_npu_csv_row(
        study_id="bert-base-uncased",
        seq_len=64,
        num_threads=12,
        num_samples=1,
        runs_per_sample=1,
        warmup_runs=0,
        power_backend="none",
        power_log_path=None,
        result={
            "benchmark_mode": "synthetic_dense",
            "execution_mode": "operator_runlist",
            "dtype": "bfloat16",
            "measured_inference_count": 1,
            "timed_total_sec": 0.25,
            "throughput_inferences_per_sec": 4.0,
            "model_type": "bert",
            "shape": (1, 64, 768),
            "estimated_flops_per_inference": 1.0e9,
            "throughput_flops_per_sec": 4.0e9,
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "compute_tile_count": "",
            "compute_tile_utilization_fraction": "",
            "compile_setup_time_ms": 100.0,
            "topology_selection_time_ms": "",
            "topology_cache_status": "",
            "cached_steady_state_avg_latency_ms": 250.0,
            "avg_embedding_latency_ms": 1.0,
            "avg_qkv_projection_latency_ms": "",
            "avg_encoder_pipeline_latency_ms": "",
            "avg_operator_runlist_latency_ms": 249.0,
            "avg_host_preprocess_latency_ms": 0.0,
            "avg_npu_gemm_latency_ms": "",
            "avg_host_postprocess_latency_ms": "",
            "avg_device_sync_latency_ms": 0.0,
            "npu_dispatch_count": 192,
            "npu_unique_instruction_binary_count": 12,
            "min_latency_ms": 250.0,
            "avg_latency_ms": 250.0,
            "max_latency_ms": 250.0,
            "power_stats": empty_power_stats(),
        },
    )

    assert row["execution_mode"] == "operator_runlist"
    assert row["avg_operator_runlist_latency_ms"] == "249.000000"
    assert row["npu_dispatch_count"] == "192"


def test_resolve_topology_reports_cache_hit_metadata(monkeypatch):
    fake_topology = SimpleNamespace(topology_id="seq32_kv64__ps2_ph1_pffn4")
    args = SimpleNamespace(
        config_file_path="config.json",
        candidate_topologies=None,
        topology_cache="cache.json",
        topology_policy="cache",
        weights_file_path="model.safetensors",
        benchmark_mode="synthetic_dense",
    )

    monkeypatch.setattr(
        npu_inference,
        "load_encoder_pipeline_config",
        lambda config_file_path, seq_len: "config",
    )
    monkeypatch.setattr(
        npu_inference,
        "current_topology_from_config",
        lambda config, seq_len: fake_topology,
    )
    monkeypatch.setattr(
        npu_inference,
        "parse_candidate_topology_ids",
        lambda raw_ids: None,
    )
    monkeypatch.setattr(npu_inference, "load_topology_cache", lambda path: {})
    monkeypatch.setattr(
        npu_inference,
        "topology_cache_key",
        lambda config, seq_len: "cache-key",
    )
    monkeypatch.setattr(
        npu_inference,
        "find_cached_topology",
        lambda cache_data, config, seq_len, candidate_ids=None: fake_topology,
    )

    perf_values = iter([1.0, 1.025])
    monkeypatch.setattr(npu_inference.time, "perf_counter", lambda: next(perf_values))

    topology, metadata = resolve_topology(args, 64, ["sample"])

    assert topology is fake_topology
    assert metadata["topology_cache_status"] == "cache_hit"
    assert metadata["topology_selection_time_ms"] == pytest.approx(25.0)


def test_resolve_topology_reports_cache_miss_metadata(monkeypatch):
    fake_topology = SimpleNamespace(
        topology_id="seq32_kv64__ps2_ph1_pffn4",
        to_runtime_dict=lambda: {"topology_id": "seq32_kv64__ps2_ph1_pffn4"},
    )
    saved = {}
    args = SimpleNamespace(
        config_file_path="config.json",
        candidate_topologies=None,
        topology_cache="cache.json",
        topology_policy="cache",
        weights_file_path="model.safetensors",
        benchmark_mode="synthetic_dense",
        autotune_warmup_runs=2,
        autotune_runs=5,
    )

    monkeypatch.setattr(
        npu_inference,
        "load_encoder_pipeline_config",
        lambda config_file_path, seq_len: "config",
    )
    monkeypatch.setattr(
        npu_inference,
        "current_topology_from_config",
        lambda config, seq_len: fake_topology,
    )
    monkeypatch.setattr(
        npu_inference,
        "parse_candidate_topology_ids",
        lambda raw_ids: None,
    )
    monkeypatch.setattr(npu_inference, "load_topology_cache", lambda path: {})
    monkeypatch.setattr(
        npu_inference,
        "topology_cache_key",
        lambda config, seq_len: "cache-key",
    )
    monkeypatch.setattr(
        npu_inference,
        "find_cached_topology",
        lambda cache_data, config, seq_len, candidate_ids=None: None,
    )
    monkeypatch.setattr(
        npu_inference,
        "autotune_topology",
        lambda **kwargs: fake_topology,
    )
    monkeypatch.setattr(
        npu_inference,
        "save_topology_cache",
        lambda path, cache_data: saved.update(cache_data),
    )

    perf_values = iter([2.0, 2.125])
    monkeypatch.setattr(npu_inference.time, "perf_counter", lambda: next(perf_values))

    topology, metadata = resolve_topology(args, 64, ["sample"])

    assert topology is fake_topology
    assert metadata["topology_cache_status"] == "cache_miss"
    assert metadata["topology_selection_time_ms"] == pytest.approx(125.0)
    assert saved["cache-key"] == {"topology_id": "seq32_kv64__ps2_ph1_pffn4"}


def test_select_autotune_topology_prefers_higher_utilization_within_one_percent():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)
    low_util = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps1_ph1_pffn1"),
        )
        if topology.topology_id == "seq32_kv64__ps1_ph1_pffn1"
    )
    high_util = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps4_ph1_pffn1"),
        )
        if topology.topology_id == "seq32_kv64__ps4_ph1_pffn1"
    )

    selected = select_autotune_topology(
        [
            {"topology": low_util, "avg_latency_ms": 100.0},
            {"topology": high_util, "avg_latency_ms": 100.8},
        ],
        preferred_family_id="seq32_kv64",
    )

    assert selected["topology"].topology_id == "seq32_kv64__ps4_ph1_pffn1"


def test_select_autotune_topology_prefers_latency_outside_one_percent_band():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)
    low_util = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps1_ph1_pffn1"),
        )
        if topology.topology_id == "seq32_kv64__ps1_ph1_pffn1"
    )
    high_util = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps4_ph1_pffn1"),
        )
        if topology.topology_id == "seq32_kv64__ps4_ph1_pffn1"
    )

    selected = select_autotune_topology(
        [
            {"topology": low_util, "avg_latency_ms": 100.0},
            {"topology": high_util, "avg_latency_ms": 102.0},
        ],
        preferred_family_id="seq32_kv64",
    )

    assert selected["topology"].topology_id == "seq32_kv64__ps1_ph1_pffn1"


def test_select_autotune_topology_prefers_config_family_when_latency_and_utilization_tie():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)
    default_family_topology = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq32_kv64__ps2_ph1_pffn4"),
        )
        if topology.family_id == "seq32_kv64"
    )
    mirrored_family_topology = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq64_kv32__ps2_ph1_pffn4"),
        )
        if topology.family_id == "seq64_kv32"
    )

    selected = select_autotune_topology(
        [
            {"topology": mirrored_family_topology, "avg_latency_ms": 100.0},
            {"topology": default_family_topology, "avg_latency_ms": 100.0},
        ],
        preferred_family_id="seq32_kv64",
    )

    assert selected["topology"].topology_id == "seq32_kv64__ps2_ph1_pffn4"


def test_automated_benchmark_enumerate_cases_multi_study_skips_npu_unready():
    args = SimpleNamespace(
        weights_file_path=None,
        config_file_path=None,
        study_id=None,
        study_ids="all",
        study_manifest=str(STUDY_MANIFEST),
        models_root=None,
        modes="cpu,npu,igpu",
        seq_lens="64",
        benchmark_mode=DEFAULT_BENCHMARK_MODE,
        npu_execution_modes="encoder_pipeline",
        cpu_thread_counts="1",
        npu_num_threads=1,
        igpu_num_threads=1,
    )

    cases, skipped = enumerate_cases(args)
    case_ids = {case["case_id"] for case in cases}
    skipped_ids = {study_id for study_id, mode, _ in skipped if mode == "npu"}

    assert "bert-base-uncased_cpu_seq64_1t" in case_ids
    assert "bert-base-uncased_npu_seq64" in case_ids
    assert "bert-base-uncased_igpu_seq64" in case_ids
    assert "bert-large-uncased_npu_seq64" in case_ids
    assert "roberta-large_npu_seq64" in case_ids
    assert skipped_ids == set()


def test_automated_benchmark_enumerate_cases_adds_nondefault_benchmark_mode_suffix():
    args = SimpleNamespace(
        weights_file_path=None,
        config_file_path=None,
        study_id="bert-base-uncased",
        study_ids=None,
        study_manifest=str(STUDY_MANIFEST),
        models_root=None,
        modes="cpu,igpu",
        seq_lens="64",
        benchmark_mode="model_valid",
        npu_execution_modes="encoder_pipeline",
        cpu_thread_counts="1",
        npu_num_threads=1,
        igpu_num_threads=1,
    )

    cases, skipped = enumerate_cases(args)
    case_ids = {case["case_id"] for case in cases}

    assert "bert-base-uncased_cpu_model_valid_seq64_1t" in case_ids
    assert "bert-base-uncased_igpu_model_valid_seq64" in case_ids
    assert skipped == []


def test_automated_benchmark_enumerate_cases_adds_no_bias_suffix_for_cpu_and_igpu():
    args = SimpleNamespace(
        weights_file_path=None,
        config_file_path=None,
        study_id="bert-base-uncased",
        study_ids=None,
        study_manifest=str(STUDY_MANIFEST),
        models_root=None,
        modes="cpu,igpu,npu",
        seq_lens="64",
        benchmark_mode=DEFAULT_BENCHMARK_MODE,
        npu_execution_modes="encoder_pipeline",
        disable_all_biases=True,
        cpu_thread_counts="1",
        npu_num_threads=1,
        igpu_num_threads=1,
        npu_topology_policy="cache",
        npu_candidate_topologies=None,
    )

    cases, skipped = enumerate_cases(args)
    case_ids = {case["case_id"] for case in cases}

    assert "bert-base-uncased_cpu_nobias_seq64_1t" in case_ids
    assert "bert-base-uncased_igpu_nobias_seq64" in case_ids
    assert "bert-base-uncased_npu_seq64" in case_ids
    assert skipped == []


def test_automated_benchmark_enumerate_cases_include_supported_npu_seq_len_16384():
    args = SimpleNamespace(
        weights_file_path=str(WEIGHTS_FILE),
        config_file_path=str(CONFIG_FILE),
        study_id=None,
        study_ids=None,
        study_manifest=None,
        models_root=None,
        modes="npu",
        seq_lens="64,16384",
        benchmark_mode=DEFAULT_BENCHMARK_MODE,
        npu_execution_modes="encoder_pipeline",
        cpu_thread_counts="1",
        npu_num_threads=1,
        igpu_num_threads=1,
        npu_topology_policy="cache",
        npu_candidate_topologies=None,
    )

    cases, skipped = enumerate_cases(args)
    case_ids = {case["case_id"] for case in cases}
    skipped_reasons = [reason for _, mode, reason in skipped if mode == "npu"]

    assert "npu_seq64" in case_ids
    assert "npu_seq16384" in case_ids
    assert not any("seq_len=16384" in reason for reason in skipped_reasons)


def test_automated_benchmark_enumerate_cases_fans_out_npu_execution_modes():
    args = SimpleNamespace(
        weights_file_path=None,
        config_file_path=None,
        study_id="bert-base-uncased",
        study_ids=None,
        study_manifest=str(STUDY_MANIFEST),
        models_root=None,
        modes="npu",
        seq_lens="64",
        benchmark_mode=DEFAULT_BENCHMARK_MODE,
        npu_execution_modes="encoder_pipeline,gemm_only,operator_runlist",
        cpu_thread_counts="1",
        npu_num_threads=1,
        igpu_num_threads=1,
        npu_topology_policy="cache",
        npu_candidate_topologies=None,
    )

    cases, skipped = enumerate_cases(args)
    case_ids = {case["case_id"] for case in cases}

    assert "bert-base-uncased_npu_encoder_pipeline_seq64" in case_ids
    assert "bert-base-uncased_npu_gemm_only_seq64" in case_ids
    assert "bert-base-uncased_npu_operator_runlist_seq64" in case_ids
    assert skipped == []


def test_parse_seq_len_int_overrides_accepts_long_seq_len_map():
    overrides = parse_seq_len_int_overrides("2048=50,4096=15,8192=5")

    assert overrides == {2048: 50, 4096: 15, 8192: 5}


def test_case_command_applies_seq_len_run_overrides_by_seq_len(tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        warmup_runs=10,
        runs_per_sample=100,
        runs_per_sample_overrides={2048: 50, 4096: 15, 8192: 5},
        study_manifest=None,
        models_root=None,
        npu_topology_policy="cache",
        npu_topology_cache=str(tmp_path / "topology_cache.json"),
        npu_autotune_warmup_runs=2,
        npu_autotune_runs=5,
        npu_autotune_runs_overrides={2048: 3, 4096: 2, 8192: 1},
        npu_candidate_topologies=None,
        igpu_dtype="float16",
        igpu_device_index=0,
        power_interval_sec=0.5,
    )
    case_1024 = {
        "case_id": "npu_seq1024",
        "study_id": "",
        "mode": "npu",
        "execution_mode": "encoder_pipeline",
        "seq_len": 1024,
        "num_threads": 12,
        "weights_file_path": str(WEIGHTS_FILE),
        "config_file_path": str(CONFIG_FILE),
    }
    case_2048 = {
        **case_1024,
        "case_id": "npu_seq2048",
        "seq_len": 2048,
    }
    case_4096 = {
        **case_1024,
        "case_id": "npu_seq4096",
        "seq_len": 4096,
    }
    case_8192 = {
        **case_1024,
        "case_id": "npu_seq8192",
        "seq_len": 8192,
    }

    command_1024 = case_command(
        args,
        case_1024,
        tmp_path / "1024.csv",
        "auto",
        tmp_path / "1024_power.log",
    )
    command_2048 = case_command(
        args,
        case_2048,
        tmp_path / "2048.csv",
        "auto",
        tmp_path / "2048_power.log",
    )
    command_4096 = case_command(
        args,
        case_4096,
        tmp_path / "4096.csv",
        "auto",
        tmp_path / "4096_power.log",
    )
    command_8192 = case_command(
        args,
        case_8192,
        tmp_path / "8192.csv",
        "auto",
        tmp_path / "8192_power.log",
    )

    def value_for_flag(command, flag):
        return command[command.index(flag) + 1]

    assert value_for_flag(command_1024, "--runs-per-sample") == "100"
    assert value_for_flag(command_1024, "--execution-mode") == "encoder_pipeline"
    assert value_for_flag(command_2048, "--runs-per-sample") == "50"
    assert value_for_flag(command_4096, "--runs-per-sample") == "15"
    assert value_for_flag(command_8192, "--runs-per-sample") == "5"
    assert value_for_flag(command_1024, "--autotune-runs") == "5"
    assert value_for_flag(command_2048, "--autotune-runs") == "3"
    assert value_for_flag(command_4096, "--autotune-runs") == "2"
    assert value_for_flag(command_8192, "--autotune-runs") == "1"
    assert value_for_flag(command_1024, "--power-backend") == "auto"
    assert value_for_flag(command_1024, "--power-log-path").endswith("1024_power.log")


def test_case_command_preserves_auto_power_backend_for_cpu(tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        warmup_runs=10,
        runs_per_sample=100,
        runs_per_sample_overrides={},
        study_manifest=None,
        models_root=None,
        npu_topology_policy="cache",
        npu_topology_cache=str(tmp_path / "topology_cache.json"),
        npu_autotune_warmup_runs=2,
        npu_autotune_runs=5,
        npu_autotune_runs_overrides={},
        npu_candidate_topologies=None,
        igpu_dtype="float16",
        igpu_device_index=0,
        power_interval_sec=0.5,
    )
    case = {
        "case_id": "cpu_seq1024_12t",
        "study_id": "",
        "mode": "cpu",
        "seq_len": 1024,
        "num_threads": 12,
        "weights_file_path": str(WEIGHTS_FILE),
        "config_file_path": str(CONFIG_FILE),
    }

    command = case_command(
        args,
        case,
        tmp_path / "cpu.csv",
        "auto",
        tmp_path / "cpu_power.log",
    )

    def value_for_flag(flag):
        return command[command.index(flag) + 1]

    assert value_for_flag("--power-backend") == "auto"
    assert value_for_flag("--power-log-path").endswith("cpu_power.log")


def test_case_command_passes_disable_all_biases_only_to_cpu_and_igpu(tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        warmup_runs=10,
        runs_per_sample=100,
        runs_per_sample_overrides={},
        study_manifest=None,
        models_root=None,
        npu_topology_policy="cache",
        npu_topology_cache=str(tmp_path / "topology_cache.json"),
        npu_autotune_warmup_runs=2,
        npu_autotune_runs=5,
        npu_autotune_runs_overrides={},
        npu_candidate_topologies=None,
        igpu_dtype="bfloat16",
        igpu_device_index=0,
        disable_all_biases=True,
        power_interval_sec=0.5,
    )
    cpu_case = {
        "case_id": "cpu_seq64_12t",
        "study_id": "",
        "mode": "cpu",
        "seq_len": 64,
        "num_threads": 12,
        "weights_file_path": str(WEIGHTS_FILE),
        "config_file_path": str(CONFIG_FILE),
    }
    igpu_case = {
        **cpu_case,
        "case_id": "igpu_seq64",
        "mode": "igpu",
    }
    npu_case = {
        **cpu_case,
        "case_id": "npu_seq64",
        "mode": "npu",
        "execution_mode": "gemm_only",
    }

    cpu_command = case_command(
        args,
        cpu_case,
        tmp_path / "cpu.csv",
        "none",
        None,
    )
    igpu_command = case_command(
        args,
        igpu_case,
        tmp_path / "igpu.csv",
        "none",
        None,
    )
    npu_command = case_command(
        args,
        npu_case,
        tmp_path / "npu.csv",
        "none",
        None,
    )

    assert "--disable-all-biases" in cpu_command
    assert "--disable-all-biases" in igpu_command
    assert "--disable-all-biases" not in npu_command
    assert npu_command[npu_command.index("--execution-mode") + 1] == "gemm_only"


def test_case_command_uses_import_wrapper_for_operator_runlist(tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        warmup_runs=0,
        runs_per_sample=1,
        runs_per_sample_overrides={},
        study_manifest=None,
        models_root=None,
        npu_topology_policy="cache",
        npu_topology_cache=str(tmp_path / "topology_cache.json"),
        npu_autotune_warmup_runs=2,
        npu_autotune_runs=5,
        npu_autotune_runs_overrides={},
        npu_candidate_topologies=None,
        igpu_dtype="bfloat16",
        igpu_device_index=0,
        disable_all_biases=False,
        power_interval_sec=0.5,
    )
    case = {
        "case_id": "npu_operator_runlist_seq64",
        "study_id": "",
        "mode": "npu",
        "seq_len": 64,
        "num_threads": 12,
        "weights_file_path": str(WEIGHTS_FILE),
        "config_file_path": str(CONFIG_FILE),
        "execution_mode": "operator_runlist",
    }

    command = case_command(
        args,
        case,
        tmp_path / "npu.csv",
        "none",
        None,
    )

    assert command[1] == str(TEST_DIR / "npu_inference_import_main.py")
    assert command[command.index("--execution-mode") + 1] == "operator_runlist"
    assert "--disable-all-biases" in command


@pytest.mark.parametrize("seq_len", [64, 16384], ids=["64", "16384"])
def test_npu_case_skip_reason_accepts_supported_seq_len(seq_len):
    args = SimpleNamespace(
        study_manifest=None,
        models_root=None,
        npu_topology_policy="cache",
        npu_candidate_topologies=None,
    )
    target = {
        "study_id": "",
        "config_file_path": str(CONFIG_FILE),
    }

    assert npu_case_skip_reason(args, target, seq_len) is None


@pytest.mark.parametrize(
    "config_path",
    [
        TEST_DIR / "config" / "config_bert_large.json",
        TEST_DIR / "config" / "config_roberta_large.json",
    ],
    ids=["bert_large", "roberta_large"],
)
def test_large_model_npu_topologies_are_discoverable(config_path):
    seq64_config = load_encoder_pipeline_config(str(config_path), 64)
    seq128_config = load_encoder_pipeline_config(str(config_path), 128)
    seq16384_config = load_encoder_pipeline_config(str(config_path), 16384)

    topo_ids_64 = {
        topology_id(topology)
        for topology in supported_topologies_for_seq_len(seq64_config, 64)
    }
    topo_ids_128 = {
        topology_id(topology)
        for topology in supported_topologies_for_seq_len(seq128_config, 128)
    }
    topo_ids_16384 = {
        topology_id(topology)
        for topology in supported_topologies_for_seq_len(seq16384_config, 16384)
    }

    assert "seq32_kv64__ps1_ph1_pffn1" in topo_ids_64
    assert "seq64_kv32__ps1_ph1_pffn1" in topo_ids_64
    assert "seq32_kv64__ps1_ph4_pffn1" in topo_ids_64
    assert "seq32_kv64__ps4_ph1_pffn1" in topo_ids_128
    assert "seq64_kv32__ps2_ph1_pffn1" in topo_ids_128
    assert "seq32_kv64__ps4_ph1_pffn1" in topo_ids_16384
    assert "seq64_kv32__ps4_ph1_pffn1" in topo_ids_16384


def test_topology_cache_is_shape_aware_for_large_models():
    base_config = load_encoder_pipeline_config(str(CONFIG_FILE), 64)
    large_config = load_encoder_pipeline_config(
        str(TEST_DIR / "config" / "config_bert_large.json"), 64
    )

    legacy_cache = {
        "64": current_topology_from_config(base_config, 64).to_runtime_dict()
    }
    assert find_cached_topology(legacy_cache, large_config, 64) is None

    large_topology = current_topology_from_config(large_config, 64)
    shape_aware_cache = {
        topology_cache_key(large_config, 64): large_topology.to_runtime_dict()
    }
    cached = find_cached_topology(shape_aware_cache, large_config, 64)

    assert cached is not None
    assert topology_id(cached) == topology_id(large_topology)


def test_legacy_seq_len_cache_compatibility_matches_same_shape_signature():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 64)
    topology = current_topology_from_config(config, 64)
    legacy_cache = {"64": topology.to_runtime_dict()}

    cached = find_cached_topology(legacy_cache, config, 64)

    assert cached is not None
    assert topology_id(cached) == topology.topology_id


def test_topology_cache_matches_mirrored_family_entries_and_legacy_seq_len_keys():
    config = load_encoder_pipeline_config(str(CONFIG_FILE), 128)
    mirrored_topology = next(
        topology
        for topology in supported_topologies_for_seq_len(
            config,
            128,
            candidate_ids=parse_candidate_topology_ids("seq64_kv32__ps2_ph1_pffn4"),
        )
        if topology.family_id == "seq64_kv32"
    )

    family_aware_cache = {
        topology_cache_key(config, 128): mirrored_topology.to_runtime_dict()
    }
    legacy_cache = {"128": mirrored_topology.to_runtime_dict()}

    assert topology_id(find_cached_topology(family_aware_cache, config, 128)) == (
        mirrored_topology.topology_id
    )
    assert topology_id(find_cached_topology(legacy_cache, config, 128)) == (
        mirrored_topology.topology_id
    )


def test_bert_automated_benchmark_job_wrapper_study_ids_smoke(tmp_path):
    job_config = tmp_path / "benchmark_job.json"
    job_config.write_text(
        (
            "{\n"
            '  "study_ids": "all",\n'
            '  "modes": "cpu",\n'
            '  "seq_lens": "64",\n'
            '  "power_backend": "none"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    command = [
        "python3",
        str(TEST_DIR / "run_automated_benchmark_job.py"),
        str(job_config),
        "--print-command",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "--study-ids all" in result.stdout


def test_plot_benchmark_results_smoke(tmp_path):
    try:
        import matplotlib  # noqa: F401
    except Exception:
        pytest.skip("matplotlib is not installed in this environment")

    result_csv = tmp_path / "suite.csv"
    rows = [
        {
            "case_id": "toy_cpu_seq64_8t",
            "study_id": "toy-model",
            "mode": "cpu",
            "seq_len": "64",
            "num_threads": "8",
            "dtype": "float32",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.050000",
            "throughput_inferences_per_sec": "20.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "2.000000e+10",
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "min_latency_ms": "50.000000",
            "avg_latency_ms": "50.000000",
            "max_latency_ms": "50.000000",
            "cooldown_wait_sec": "0.000000",
            "cooldown_start_temp_c": "",
            "cooldown_end_temp_c": "",
            "cooldown_temp_source": "",
            "power_backend": "none",
            "power_sample_count": "0",
            "power_window_sec": "0.100000",
            "idle_power_sample_count": "0",
            "idle_power_window_sec": "",
            "idle_pkg_watt": "",
            "idle_cor_watt": "",
            "idle_gfx_watt": "",
            "idle_ram_watt": "",
            "avg_pkg_watt": "40.000000",
            "max_pkg_watt": "40.000000",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "pseudo_device_avg_pkg_watt": "",
            "pseudo_device_max_pkg_watt": "",
            "pseudo_npu_avg_pkg_watt": "",
            "pseudo_npu_max_pkg_watt": "",
            "estimated_total_timed_flops": "1.000000e+09",
            "estimated_gflops_per_watt_sec": "25.000000",
            "pseudo_device_estimated_gflops_per_watt_sec": "",
            "pseudo_npu_estimated_gflops_per_watt_sec": "",
            "idle_power_log": "",
            "power_log": "",
            "benchmark_csv": "toy_cpu.csv",
        },
        {
            "case_id": "toy_cpu_seq64_16t",
            "study_id": "toy-model",
            "mode": "cpu",
            "seq_len": "64",
            "num_threads": "16",
            "dtype": "float32",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.045000",
            "throughput_inferences_per_sec": "22.222222",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "2.222222e+10",
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "min_latency_ms": "45.000000",
            "avg_latency_ms": "45.000000",
            "max_latency_ms": "45.000000",
            "cooldown_wait_sec": "0.000000",
            "cooldown_start_temp_c": "",
            "cooldown_end_temp_c": "",
            "cooldown_temp_source": "",
            "power_backend": "none",
            "power_sample_count": "0",
            "power_window_sec": "0.090000",
            "idle_power_sample_count": "0",
            "idle_power_window_sec": "",
            "idle_pkg_watt": "",
            "idle_cor_watt": "",
            "idle_gfx_watt": "",
            "idle_ram_watt": "",
            "avg_pkg_watt": "42.000000",
            "max_pkg_watt": "42.000000",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "pseudo_device_avg_pkg_watt": "",
            "pseudo_device_max_pkg_watt": "",
            "pseudo_npu_avg_pkg_watt": "",
            "pseudo_npu_max_pkg_watt": "",
            "estimated_total_timed_flops": "1.000000e+09",
            "estimated_gflops_per_watt_sec": "26.455026",
            "pseudo_device_estimated_gflops_per_watt_sec": "",
            "pseudo_npu_estimated_gflops_per_watt_sec": "",
            "idle_power_log": "",
            "power_log": "",
            "benchmark_csv": "toy_cpu_16.csv",
        },
        {
            "case_id": "toy_igpu_seq64",
            "study_id": "toy-model",
            "mode": "igpu",
            "seq_len": "64",
            "num_threads": "8",
            "dtype": "float16",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.010000",
            "throughput_inferences_per_sec": "100.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "1.000000e+11",
            "topology_id": "",
            "parallel_seq": "",
            "parallel_heads": "",
            "parallel_ffn": "",
            "min_latency_ms": "10.000000",
            "avg_latency_ms": "10.000000",
            "max_latency_ms": "10.000000",
            "cooldown_wait_sec": "0.000000",
            "cooldown_start_temp_c": "",
            "cooldown_end_temp_c": "",
            "cooldown_temp_source": "",
            "power_backend": "rocm-smi",
            "power_sample_count": "1",
            "power_window_sec": "0.030000",
            "idle_power_sample_count": "1",
            "idle_power_window_sec": "0.020000",
            "idle_pkg_watt": "8.000000",
            "idle_cor_watt": "",
            "idle_gfx_watt": "",
            "idle_ram_watt": "",
            "avg_pkg_watt": "20.000000",
            "max_pkg_watt": "20.000000",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "pseudo_device_avg_pkg_watt": "12.000000",
            "pseudo_device_max_pkg_watt": "12.000000",
            "pseudo_npu_avg_pkg_watt": "",
            "pseudo_npu_max_pkg_watt": "",
            "estimated_total_timed_flops": "1.000000e+09",
            "estimated_gflops_per_watt_sec": "50.000000",
            "pseudo_device_estimated_gflops_per_watt_sec": "83.333333",
            "pseudo_npu_estimated_gflops_per_watt_sec": "",
            "idle_power_log": "toy_igpu_idle.log",
            "power_log": "toy_igpu_power.log",
            "benchmark_csv": "toy_igpu.csv",
        },
        {
            "case_id": "toy_npu_seq64",
            "study_id": "toy-model",
            "mode": "npu",
            "seq_len": "64",
            "num_threads": "8",
            "dtype": "bfloat16",
            "num_samples": "1",
            "runs_per_sample": "1",
            "warmup_runs": "0",
            "measured_inference_count": "1",
            "timed_total_sec": "0.020000",
            "throughput_inferences_per_sec": "50.000000",
            "model_type": "bert",
            "shape": "(1, 64, 768)",
            "estimated_flops_per_inference": "1.000000e+09",
            "throughput_flops_per_sec": "5.000000e+10",
            "topology_id": "seq32_kv64__ps2_ph1_pffn4",
            "parallel_seq": "2",
            "parallel_heads": "1",
            "parallel_ffn": "4",
            "min_latency_ms": "20.000000",
            "avg_latency_ms": "20.000000",
            "max_latency_ms": "20.000000",
            "cooldown_wait_sec": "0.000000",
            "cooldown_start_temp_c": "",
            "cooldown_end_temp_c": "",
            "cooldown_temp_source": "",
            "power_backend": "turbostat",
            "power_sample_count": "1",
            "power_window_sec": "0.040000",
            "idle_power_sample_count": "1",
            "idle_power_window_sec": "0.020000",
            "idle_pkg_watt": "8.000000",
            "idle_cor_watt": "",
            "idle_gfx_watt": "",
            "idle_ram_watt": "",
            "avg_pkg_watt": "18.000000",
            "max_pkg_watt": "18.000000",
            "avg_cor_watt": "",
            "max_cor_watt": "",
            "avg_gfx_watt": "",
            "max_gfx_watt": "",
            "avg_ram_watt": "",
            "max_ram_watt": "",
            "pseudo_device_avg_pkg_watt": "10.000000",
            "pseudo_device_max_pkg_watt": "10.000000",
            "pseudo_npu_avg_pkg_watt": "10.000000",
            "pseudo_npu_max_pkg_watt": "10.000000",
            "estimated_total_timed_flops": "1.000000e+09",
            "estimated_gflops_per_watt_sec": "55.555556",
            "pseudo_device_estimated_gflops_per_watt_sec": "100.000000",
            "pseudo_npu_estimated_gflops_per_watt_sec": "100.000000",
            "idle_power_log": "toy_npu_idle.log",
            "power_log": "toy_npu_power.log",
            "benchmark_csv": "toy_npu.csv",
        },
    ]
    peak_fields = {
        "chip_sku": "AMD Ryzen AI 9 HX 370",
        "chip_family": "Ryzen AI 9 HX",
        "chip_codename": "Strix Point",
        "bytes_model_version": "v1",
        "bytes_model_weights_policy": "resident",
        "dtype_bytes": "",
        "estimated_bytes_per_inference": "1.953125e+07",
        "backend_peak_ops_per_sec": "",
        "primary_peak_source_kind": "measured",
        "backend_peak_source_note": "synthetic calibration",
        "backend_pct_of_peak": "",
        "ddr_peak_bytes_per_sec": "1.280000e+11",
        "ddr_peak_source_kind": "measured",
        "operational_intensity_flops_per_byte": "51.200000",
        "roofline_bound_ops_per_sec": "4.000000e+11",
        "roofline_pct": "",
        "official_peak_ops_per_sec": "",
        "derived_theoretical_peak_ops_per_sec": "",
        "measured_peak_ops_per_sec": "",
    }
    per_mode_peak_fields = {
        "cpu": {
            "dtype_bytes": "4",
            "backend_peak_ops_per_sec": "8.000000e+10",
            "backend_pct_of_peak": "0.277778",
            "roofline_pct": "0.250000",
            "measured_peak_ops_per_sec": "8.000000e+10",
        },
        "igpu": {
            "dtype_bytes": "2",
            "backend_peak_ops_per_sec": "2.500000e+11",
            "backend_pct_of_peak": "0.400000",
            "roofline_pct": "0.781250",
            "measured_peak_ops_per_sec": "2.500000e+11",
        },
        "npu": {
            "dtype_bytes": "2",
            "backend_peak_ops_per_sec": "4.000000e+11",
            "backend_pct_of_peak": "0.125000",
            "roofline_pct": "0.125000",
            "measured_peak_ops_per_sec": "4.000000e+11",
        },
    }
    for row in rows:
        row.update(peak_fields)
        row.update(per_mode_peak_fields[row["mode"]])

    with result_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "plots"
    command = [
        "python3",
        str(TEST_DIR / "plot_benchmark_results.py"),
        str(result_csv),
        "--output-dir",
        str(output_dir),
        "--dpi",
        "120",
    ]
    result = subprocess.run(
        command,
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (output_dir / "summary_overview.png").exists()
    assert (output_dir / "grouped_latency_by_backend.png").exists()
    assert (output_dir / "grouped_throughput_by_backend.png").exists()
    assert (output_dir / "grouped_power_by_backend.png").exists()
    assert (output_dir / "grouped_efficiency_by_backend.png").exists()
    assert (output_dir / "grouped_pct_of_peak_by_backend.png").exists()
    assert (output_dir / "grouped_roofline_pct_by_backend.png").exists()
    assert (output_dir / "roofline_overview.png").exists()
    assert (output_dir / "toy-model_overview.png").exists()
    assert (output_dir / "toy-model_cpu_threads.png").exists()
    assert (output_dir / "index.html").exists()
