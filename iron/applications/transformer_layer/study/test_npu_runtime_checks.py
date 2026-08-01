#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import subprocess

from iron.applications.transformer_layer.study import npu_runtime_checks


def test_extract_power_mode_from_text_parses_turbo() -> None:
    text = """
Platform
  Name                   : NPU Strix
  Power Mode             : Turbo
  Total Columns          : 8
"""

    assert npu_runtime_checks._extract_power_mode_from_text(text) == "turbo"


def test_find_power_mode_in_json_walks_nested_payload() -> None:
    payload = {
        "devices": [
            {
                "platform": {
                    "name": "NPU Strix",
                    "power_mode": "balanced",
                }
            }
        ]
    }

    assert npu_runtime_checks._find_power_mode_in_json(payload) == "balanced"


def test_detect_npu_power_mode_prefers_json(monkeypatch) -> None:
    monkeypatch.setattr(npu_runtime_checks, "_xrt_smi_path", lambda: "/usr/bin/xrt-smi")

    def fake_run(args, capture_output, text, check):  # noqa: ARG001
        output_path = args[-1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"platform": {"power_mode": "turbo"}}, handle)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(npu_runtime_checks.subprocess, "run", fake_run)

    status = npu_runtime_checks.detect_npu_power_mode()

    assert status.mode == "turbo"
    assert status.source == "xrt-smi-json"


def test_detect_npu_power_mode_falls_back_to_text(monkeypatch) -> None:
    monkeypatch.setattr(npu_runtime_checks, "_xrt_smi_path", lambda: "/usr/bin/xrt-smi")
    monkeypatch.setattr(
        npu_runtime_checks,
        "_detect_power_mode_from_json",
        lambda path: None,
    )

    def fake_run(args, capture_output, text, check):  # noqa: ARG001
        stdout = """
Platform
  Name                   : NPU Strix
  Power Mode             : Default
"""
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(npu_runtime_checks.subprocess, "run", fake_run)

    status = npu_runtime_checks.detect_npu_power_mode()

    assert status.mode == "default"
    assert status.source == "xrt-smi-text"


def test_warn_if_npu_power_mode_not_turbo_logs_warning(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        npu_runtime_checks,
        "detect_npu_power_mode",
        lambda: npu_runtime_checks.NPUPowerModeStatus(
            mode="balanced",
            source="xrt-smi-text",
            detail="parsed from test",
        ),
    )

    with caplog.at_level(logging.WARNING):
        npu_runtime_checks.warn_if_npu_power_mode_not_turbo(
            logging.getLogger("test"),
            study_name="block study",
        )

    assert "configure --pmode turbo" in caplog.text
    assert "balanced" in caplog.text


def test_require_npu_power_mode_turbo_raises_for_non_turbo(monkeypatch) -> None:
    monkeypatch.setattr(
        npu_runtime_checks,
        "detect_npu_power_mode",
        lambda: npu_runtime_checks.NPUPowerModeStatus(
            mode="default",
            source="xrt-smi-text",
            detail="parsed from test",
        ),
    )

    try:
        npu_runtime_checks.require_npu_power_mode_turbo(
            study_name="end-to-end benchmark"
        )
    except RuntimeError as exc:
        assert "default" in str(exc)
        assert "configure --pmode turbo" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the NPU mode is not turbo")


def test_require_npu_power_mode_turbo_passes_for_turbo(monkeypatch) -> None:
    monkeypatch.setattr(
        npu_runtime_checks,
        "detect_npu_power_mode",
        lambda: npu_runtime_checks.NPUPowerModeStatus(
            mode="turbo",
            source="xrt-smi-text",
            detail="parsed from test",
        ),
    )

    status = npu_runtime_checks.require_npu_power_mode_turbo(study_name="block study")

    assert status.mode == "turbo"
