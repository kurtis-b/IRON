#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
WEIGHTS_FILE = TEST_DIR / "model.safetensors"
CONFIG_FILE = TEST_DIR / "config" / "config.json"


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
