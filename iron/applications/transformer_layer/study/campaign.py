#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_CAMPAIGN_ID = "canonical"
DEFAULT_REPEAT_COUNT = 5
DEFAULT_REPEAT_INDEX = 0


def optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def normalize_campaign_id(value: object) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_CAMPAIGN_ID


def normalize_repeat_index(value: object) -> int:
    resolved = optional_int(value)
    if resolved is None:
        return DEFAULT_REPEAT_INDEX
    return int(resolved)


def normalize_matched_run_id(
    value: object,
    *,
    campaign_id: str,
    study_case_id: str,
    execution_mode: str,
    seq_len: int,
    repeat_index: int,
) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return matched_run_id(
        campaign_id=campaign_id,
        study_case_id=study_case_id,
        execution_mode=execution_mode,
        seq_len=seq_len,
        repeat_index=repeat_index,
    )


def matched_run_id(
    *,
    campaign_id: str,
    study_case_id: str,
    execution_mode: str,
    seq_len: int,
    repeat_index: int,
) -> str:
    return (
        f"{normalize_campaign_id(campaign_id)}:"
        f"{study_case_id}:{execution_mode}:seq{int(seq_len)}:repeat{int(repeat_index)}"
    )


def current_git_sha() -> str | None:
    return _command_output(["git", "rev-parse", "HEAD"])


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(fraction)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def summarize_repeat_values(
    values: Iterable[float],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    resolved_values = [float(value) for value in values]
    count = len(resolved_values)
    if not resolved_values:
        return {
            "repeat_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_stddev": None,
            f"{prefix}_iqr": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_ci95": None,
        }

    sorted_values = sorted(resolved_values)
    q1 = _percentile(sorted_values, 0.25)
    q3 = _percentile(sorted_values, 0.75)
    iqr = None
    if q1 is not None and q3 is not None:
        iqr = q3 - q1
    stddev = statistics.stdev(resolved_values) if count >= 2 else 0.0
    ci95 = 0.0 if count == 1 else 1.96 * stddev / math.sqrt(float(count))
    return {
        "repeat_count": count,
        f"{prefix}_mean": statistics.fmean(resolved_values),
        f"{prefix}_median": statistics.median(resolved_values),
        f"{prefix}_stddev": stddev,
        f"{prefix}_iqr": iqr,
        f"{prefix}_min": min(resolved_values),
        f"{prefix}_max": max(resolved_values),
        f"{prefix}_ci95": ci95,
    }


def manifest_path_for_output(output_path: Path) -> Path:
    return output_path.with_name("campaign_manifest.json")


def load_campaign_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_output_campaign_manifest(output_path: Path) -> dict[str, object] | None:
    return load_campaign_manifest(manifest_path_for_output(output_path))


def manifest_matches_checkout(
    manifest: dict[str, object] | None,
    *,
    campaign_id: str,
    git_sha: str | None,
    study_name: str | None = None,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    if normalize_campaign_id(manifest.get("campaign_id")) != normalize_campaign_id(
        campaign_id
    ):
        return False
    if study_name is not None and str(manifest.get("study_name") or "") != study_name:
        return False
    if git_sha is None:
        return True
    manifest_git_sha = str(manifest.get("git_sha") or "").strip()
    return bool(manifest_git_sha) and manifest_git_sha == git_sha


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def collect_campaign_manifest(
    *,
    campaign_id: str,
    study_name: str,
    command_line: list[str],
    output_files: Iterable[Path],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "campaign_id": normalize_campaign_id(campaign_id),
        "study_name": study_name,
        "command_line": list(command_line),
        "git_sha": current_git_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "tool_paths": {
            tool_name: _command_output(["bash", "-lc", f"command -v {tool_name}"])
            for tool_name in ("python3", "turbostat", "rocm-smi", "xrt-smi")
        },
        "output_files": [str(path) for path in output_files],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_campaign_manifest(
    path: Path,
    *,
    campaign_id: str,
    study_name: str,
    command_line: list[str],
    output_files: Iterable[Path],
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect_campaign_manifest(
        campaign_id=campaign_id,
        study_name=study_name,
        command_line=command_line,
        output_files=output_files,
        extra=extra,
    )
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
