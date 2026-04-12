#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

_POWER_MODE_RE = re.compile(r"Power Mode\s*:\s*([A-Za-z][A-Za-z0-9_-]*)", re.IGNORECASE)
_TURBO_MODE = "turbo"
_RECOMMENDED_COMMAND = "sudo xrt-smi configure --pmode turbo"
_VERIFY_COMMAND = "xrt-smi examine -r all"


@dataclass(frozen=True)
class NPUPowerModeStatus:
    mode: str | None
    source: str
    detail: str

    @property
    def is_turbo(self) -> bool:
        return self.mode == _TURBO_MODE


def _normalize_mode(value: object) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _extract_power_mode_from_text(text: str) -> str | None:
    match = _POWER_MODE_RE.search(text)
    if match is None:
        return None
    return _normalize_mode(match.group(1))


def _find_power_mode_in_json(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in {
                "power_mode",
                "performance_mode",
                "pmode",
            } or ("power" in normalized_key and "mode" in normalized_key):
                mode = _normalize_mode(value)
                if mode is not None:
                    return mode
            nested = _find_power_mode_in_json(value)
            if nested is not None:
                return nested
        return None

    if isinstance(payload, list):
        for item in payload:
            nested = _find_power_mode_in_json(item)
            if nested is not None:
                return nested
    return None


def _xrt_smi_path() -> str | None:
    return shutil.which("xrt-smi")


def _detect_power_mode_from_json(xrt_smi_path: str) -> NPUPowerModeStatus | None:
    temp_fd, temp_path = tempfile.mkstemp(prefix="xrt-smi-", suffix=".json")
    os.close(temp_fd)
    try:
        result = subprocess.run(
            [
                xrt_smi_path,
                "--batch",
                "examine",
                "-f",
                "JSON",
                "-r",
                "all",
                "-o",
                temp_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        output_path = Path(temp_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            return None
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        mode = _find_power_mode_in_json(payload)
        if mode is None:
            return None
        return NPUPowerModeStatus(
            mode=mode,
            source="xrt-smi-json",
            detail="power mode parsed from xrt-smi JSON examine report",
        )
    except Exception:
        return None
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _detect_power_mode_from_text(xrt_smi_path: str) -> NPUPowerModeStatus:
    try:
        result = subprocess.run(
            [xrt_smi_path, "--batch", "examine", "-r", "all"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return NPUPowerModeStatus(
            mode=None,
            source="xrt-smi-text",
            detail=f"failed to run xrt-smi examine: {exc}",
        )

    combined_output = "\n".join(
        part for part in (result.stdout, result.stderr) if part and part.strip()
    )
    if result.returncode != 0:
        return NPUPowerModeStatus(
            mode=None,
            source="xrt-smi-text",
            detail=(
                f"xrt-smi examine exited with code {result.returncode}: "
                f"{combined_output.strip()}"
            ).strip(),
        )

    mode = _extract_power_mode_from_text(combined_output)
    if mode is None:
        return NPUPowerModeStatus(
            mode=None,
            source="xrt-smi-text",
            detail="Power Mode field was not found in xrt-smi text examine output",
        )

    return NPUPowerModeStatus(
        mode=mode,
        source="xrt-smi-text",
        detail="power mode parsed from xrt-smi text examine report",
    )


def detect_npu_power_mode() -> NPUPowerModeStatus:
    xrt_smi_path = _xrt_smi_path()
    if xrt_smi_path is None:
        return NPUPowerModeStatus(
            mode=None,
            source="unavailable",
            detail="xrt-smi was not found on PATH",
        )

    json_status = _detect_power_mode_from_json(xrt_smi_path)
    if json_status is not None and json_status.mode is not None:
        return json_status

    return _detect_power_mode_from_text(xrt_smi_path)


def warn_if_npu_power_mode_not_turbo(
    logger: logging.Logger,
    *,
    study_name: str,
) -> NPUPowerModeStatus:
    status = detect_npu_power_mode()
    if status.is_turbo:
        logger.info(
            "%s: confirmed NPU power mode is turbo via %s",
            study_name,
            status.source,
        )
        return status

    if status.mode is None:
        logger.warning(
            "%s: could not determine the NPU power mode via xrt-smi (%s). "
            "Recommended benchmark setting: `%s`. Verify with `%s`.",
            study_name,
            status.detail,
            _RECOMMENDED_COMMAND,
            _VERIFY_COMMAND,
        )
        return status

    logger.warning(
        "%s: detected NPU power mode `%s`, not turbo. "
        "Recommended benchmark setting: `%s`. Verify with `%s`.",
        study_name,
        status.mode,
        _RECOMMENDED_COMMAND,
        _VERIFY_COMMAND,
    )
    return status


def require_npu_power_mode_turbo(
    *,
    study_name: str,
) -> NPUPowerModeStatus:
    status = detect_npu_power_mode()
    if status.is_turbo:
        return status

    if status.mode is None:
        raise RuntimeError(
            f"{study_name}: could not determine the NPU power mode via xrt-smi "
            f"({status.detail}). Set `{_RECOMMENDED_COMMAND}` and verify with "
            f"`{_VERIFY_COMMAND}` before benchmarking."
        )

    raise RuntimeError(
        f"{study_name}: detected NPU power mode `{status.mode}`, not turbo. "
        f"Set `{_RECOMMENDED_COMMAND}` and verify with `{_VERIFY_COMMAND}` "
        "before benchmarking."
    )


__all__ = [
    "NPUPowerModeStatus",
    "detect_npu_power_mode",
    "require_npu_power_mode_turbo",
    "warn_if_npu_power_mode_not_turbo",
    "_extract_power_mode_from_text",
    "_find_power_mode_in_json",
]
