#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))

from automated_benchmark import enumerate_cases, parse_turbostat_log
from model_support import (
    canonicalize_app_config_dict,
    canonicalize_local_backbone_weights,
    estimate_encoder_forward_flops,
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


def test_encoder_forward_flops_estimate_grows_with_seq_len():
    import json

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    model_config = canonicalize_app_config_dict(config_json)["model_config"]

    seq64 = estimate_encoder_forward_flops(model_config, 64)
    seq512 = estimate_encoder_forward_flops(model_config, 512)

    assert seq64 > 0
    assert seq512 > seq64


def test_parse_turbostat_log_accepts_periodic_samples(tmp_path):
    log_path = tmp_path / "turbostat.log"
    log_path.write_text(
        "2.000000 sec\n" "CorWatt\tPkgWatt\n" "1.00\t10.00\n" "2.00\t20.00\n" "0.00\n",
        encoding="utf-8",
    )

    stats = parse_turbostat_log(log_path)

    assert stats["power_sample_count"] == 2
    assert stats["power_window_sec"] == "2.000000"
    assert stats["avg_pkg_watt"] == "15.000000"
    assert stats["max_pkg_watt"] == "20.000000"
    assert stats["avg_cor_watt"] == "1.500000"


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
    assert "bert-large-uncased_npu_seq64" not in case_ids
    assert "roberta-large_npu_seq64" not in case_ids
    assert skipped_ids == {"bert-large-uncased", "roberta-large"}


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
            "power_backend": "turbostat",
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
            "topology_id": "2ps_4pffn",
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
    assert (output_dir / "toy-model_overview.png").exists()
    assert (output_dir / "toy-model_cpu_threads.png").exists()
    assert (output_dir / "index.html").exists()
