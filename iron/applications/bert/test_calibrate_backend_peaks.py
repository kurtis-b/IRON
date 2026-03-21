#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import calibrate_backend_peaks
from calibrate_backend_peaks import (
    collect_measured_calibration,
    collect_npu_bf16_reference,
    merge_calibration_data,
    parse_cpu_affinity,
    parse_npu_benchmark_csv_text,
    parse_shape_list,
    parse_xrt_smi_validate_gemm_json,
    select_best_npu_bf16_peak_row,
)


def test_parse_shape_list_accepts_triplets():
    assert parse_shape_list("2048x2048x2048,4096x3072x4096") == (
        (2048, 2048, 2048),
        (4096, 3072, 4096),
    )


def test_parse_shape_list_rejects_invalid_item():
    with pytest.raises(ValueError, match="Invalid GEMM shape"):
        parse_shape_list("2048x2048,4096x4096x4096")


def test_parse_cpu_affinity_accepts_ranges_and_singletons():
    assert parse_cpu_affinity("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)


def test_parse_xrt_smi_validate_gemm_json_extracts_tops_from_success_payload():
    payload = {
        "logical_devices": [
            {
                "tests": [
                    {
                        "name": "gemm",
                        "status": "PASSED",
                        "log": [
                            {
                                "Performance": "47.5 TOPS",
                            }
                        ],
                    }
                ]
            }
        ]
    }

    parsed = parse_xrt_smi_validate_gemm_json(json.dumps(payload))

    assert parsed["measured_peak_ops_per_sec"] == pytest.approx(47.5e12)
    assert "status=PASSED" in parsed["note"]


def test_parse_xrt_smi_validate_gemm_json_accepts_label_before_value():
    payload = {
        "logical_devices": [
            {
                "tests": [
                    {
                        "name": "gemm",
                        "status": "PASSED",
                        "log": [
                            {
                                "Details": "TOPS: 51.0",
                            }
                        ],
                    }
                ]
            }
        ]
    }

    parsed = parse_xrt_smi_validate_gemm_json(json.dumps(payload))

    assert parsed["measured_peak_ops_per_sec"] == pytest.approx(51.0e12)


def test_parse_xrt_smi_validate_gemm_json_preserves_failure_reason():
    payload = {
        "logical_devices": [
            {
                "tests": [
                    {
                        "name": "gemm",
                        "status": "FAILED",
                        "log": [{"Error": "No archive provided, skipping test"}],
                    }
                ]
            }
        ]
    }

    parsed = parse_xrt_smi_validate_gemm_json(json.dumps(payload))

    assert "measured_peak_ops_per_sec" not in parsed
    assert "No archive provided, skipping test" in parsed["note"]


def test_merge_calibration_data_deep_merges_nested_peaks():
    merged = merge_calibration_data(
        {
            "tool_versions": {"torch": "2.9.1"},
            "peaks": {"cpu": {"bfloat16": {"measured_peak_ops_per_sec": 1.0}}},
        },
        {
            "tool_versions": {"numpy": "1.26.4"},
            "peaks": {"igpu": {"float32": {"measured_peak_ops_per_sec": 2.0}}},
        },
    )

    assert merged["tool_versions"] == {"torch": "2.9.1", "numpy": "1.26.4"}
    assert merged["peaks"]["cpu"]["bfloat16"]["measured_peak_ops_per_sec"] == 1.0
    assert merged["peaks"]["igpu"]["float32"]["measured_peak_ops_per_sec"] == 2.0


def test_collect_measured_calibration_validates_backend_list():
    host_profile = {"cpu": {"threads": 24}}
    args = type(
        "Args",
        (),
        {
            "collect_backends": "cpu,unknown",
            "gemm_shapes": "2048x2048x2048",
            "cpu_affinity": None,
            "warmup_iters": 1,
            "measure_iters": 1,
            "repetitions": 1,
            "cpu_threads": None,
            "igpu_device_index": 0,
            "stream_num_elements": 1024,
        },
    )()

    with pytest.raises(ValueError, match="Unsupported backends"):
        collect_measured_calibration(host_profile, args)


def test_parse_npu_benchmark_csv_text_reads_rows():
    csv_text = """study_id,seq_len,throughput_flops_per_sec,topology_id,avg_latency_ms
bert-base-uncased,256,1.0e12,4ps,100.0
bert-base-uncased,512,1.5e12,4ps,200.0
"""

    rows = parse_npu_benchmark_csv_text(csv_text)

    assert len(rows) == 2
    assert rows[1]["seq_len"] == "512"


def test_select_best_npu_bf16_peak_row_prefers_highest_throughput():
    rows = [
        {
            "study_id": "bert-base-uncased",
            "seq_len": "256",
            "throughput_flops_per_sec": "1.0e12",
            "topology_id": "4ps",
            "avg_latency_ms": "100.0",
        },
        {
            "study_id": "bert-base-uncased",
            "seq_len": "512",
            "throughput_flops_per_sec": "1.6e12",
            "topology_id": "4ps",
            "avg_latency_ms": "180.0",
        },
    ]

    best_row = select_best_npu_bf16_peak_row(rows)

    assert best_row["seq_len"] == "512"
    assert best_row["throughput_flops_per_sec"] == pytest.approx(1.6e12)


def test_collect_npu_bf16_reference_parses_best_row(tmp_path, monkeypatch):
    csv_path = tmp_path / "npu_bf16_calibration.csv"

    class _FakeTempDir:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(command, capture_output, text, check):
        assert "--study-id" in command
        assert "--power-backend" in command
        assert command[command.index("--power-backend") + 1] == "none"
        csv_path.write_text(
            "study_id,seq_len,throughput_flops_per_sec,topology_id,avg_latency_ms\n"
            "bert-base-uncased,256,1.1e12,4ps,110.0\n"
            "bert-base-uncased,512,1.7e12,4ps,190.0\n",
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(
        calibrate_backend_peaks.tempfile, "TemporaryDirectory", lambda: _FakeTempDir()
    )
    monkeypatch.setattr(calibrate_backend_peaks.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "npu_bf16_study_id": "bert-base-uncased",
            "npu_bf16_seq_lens": "256,512",
            "npu_bf16_num_samples": 1,
            "npu_bf16_warmup_runs": 2,
            "npu_bf16_runs": 5,
            "npu_bf16_num_threads": 12,
            "cpu_threads": 12,
            "npu_bf16_topology_policy": "cache",
            "npu_bf16_topology_cache": str(tmp_path / "cache.json"),
            "npu_bf16_study_manifest": None,
            "npu_bf16_models_root": None,
        },
    )()

    measured = collect_npu_bf16_reference(args)
    peak = measured["peaks"]["npu"]["bfloat16"]

    assert peak["measured_peak_ops_per_sec"] == pytest.approx(1.7e12)
    assert "seq_len=512" in peak["note"]


def test_collect_npu_bf16_reference_reports_failure_note(tmp_path, monkeypatch):
    class _FakeTempDir:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(command, capture_output, text, check):
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "", "stderr": "device not ready"},
        )()

    monkeypatch.setattr(
        calibrate_backend_peaks.tempfile, "TemporaryDirectory", lambda: _FakeTempDir()
    )
    monkeypatch.setattr(calibrate_backend_peaks.subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "npu_bf16_study_id": "bert-base-uncased",
            "npu_bf16_seq_lens": "256,512",
            "npu_bf16_num_samples": 1,
            "npu_bf16_warmup_runs": 2,
            "npu_bf16_runs": 5,
            "npu_bf16_num_threads": 12,
            "cpu_threads": 12,
            "npu_bf16_topology_policy": "cache",
            "npu_bf16_topology_cache": str(tmp_path / "cache.json"),
            "npu_bf16_study_manifest": None,
            "npu_bf16_models_root": None,
        },
    )()

    measured = collect_npu_bf16_reference(args)
    assert "device not ready" in measured["peaks"]["npu"]["bfloat16"]["note"]
