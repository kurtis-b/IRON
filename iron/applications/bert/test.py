#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))

from model_support import (
    canonicalize_app_config_dict,
    canonicalize_local_backbone_weights,
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
