#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path

from .campaign import load_campaign_manifest

RESULTS_ROOT_MANIFEST_NAME = "artifact_manifest.json"
RESULTS_ROOT_MANIFEST_VERSION = 1
STUDY_MANIFEST_RELATIVE_PATHS: tuple[tuple[str, str], ...] = (
    ("end_to_end", "end_to_end/campaign_manifest.json"),
    ("host_comparison", "host_comparison/campaign_manifest.json"),
)


def results_root_manifest_path(results_root: str | Path) -> Path:
    return Path(results_root) / RESULTS_ROOT_MANIFEST_NAME


def load_results_root_manifest(results_root: str | Path) -> dict[str, object] | None:
    path = results_root_manifest_path(results_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _nonempty_text(manifest: dict[str, object], field: str) -> str | None:
    value = str(manifest.get(field) or "").strip()
    if not value:
        return None
    return value


def _shared_text_value(
    manifests: dict[str, dict[str, object]],
    field: str,
) -> str:
    values = {
        value
        for manifest in manifests.values()
        if (value := _nonempty_text(manifest, field)) is not None
    }
    if len(values) != 1:
        raise ValueError(
            f"Results root study manifests must agree on {field!r}, saw {sorted(values)}"
        )
    return next(iter(values))


def _study_command_lines(
    manifests: dict[str, dict[str, object]],
) -> dict[str, list[str]]:
    command_lines: dict[str, list[str]] = {}
    for study_name, manifest in manifests.items():
        raw = manifest.get("command_line")
        if not isinstance(raw, list):
            continue
        command_lines[study_name] = [str(value) for value in raw]
    return command_lines


def _study_tool_paths(
    manifests: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    tool_paths: dict[str, dict[str, object]] = {}
    for study_name, manifest in manifests.items():
        raw = manifest.get("tool_paths")
        if not isinstance(raw, dict):
            continue
        tool_paths[study_name] = {str(key): value for key, value in raw.items()}
    return tool_paths


def _study_tool_versions(
    manifests: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    tool_versions: dict[str, dict[str, object]] = {}
    for study_name, manifest in manifests.items():
        raw = manifest.get("tool_versions")
        if not isinstance(raw, dict):
            continue
        tool_versions[study_name] = {str(key): value for key, value in raw.items()}
    return tool_versions


def _study_system_snapshots(
    manifests: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for study_name, manifest in manifests.items():
        raw = manifest.get("system_snapshot")
        if not isinstance(raw, dict):
            continue
        snapshots[study_name] = {str(key): value for key, value in raw.items()}
    return snapshots


def _load_automation_state(results_root: Path) -> dict[str, object] | None:
    state_path = results_root / "automation" / "state.json"
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    jobs_summary: list[dict[str, object]] = []
    raw_jobs = payload.get("jobs")
    if isinstance(raw_jobs, list):
        for raw_job in raw_jobs:
            if not isinstance(raw_job, dict):
                continue
            jobs_summary.append(
                {
                    "id": str(raw_job.get("id") or ""),
                    "description": str(raw_job.get("description") or ""),
                    "module": str(raw_job.get("module") or ""),
                    "argv": [str(value) for value in raw_job.get("argv") or []],
                }
            )
    return {
        "run_id": str(payload.get("run_id") or ""),
        "artifact_profile": str(payload.get("artifact_profile") or ""),
        "plan_kind": str(payload.get("plan_kind") or ""),
        "plan_layout": str(payload.get("plan_layout") or ""),
        "run_user": str(payload.get("run_user") or ""),
        "job_count": len(jobs_summary),
        "jobs": jobs_summary,
    }


def evaluation_profile_for_plan_kind(plan_kind: str) -> str:
    normalized = str(plan_kind or "").strip()
    if normalized == "paper_p0":
        return "paper"
    return normalized


def collect_results_root_manifest(
    results_root: str | Path,
    *,
    source_results_root: str | Path | None = None,
    evaluation_profile: str | None = None,
) -> dict[str, object]:
    root = Path(results_root)
    manifests: dict[str, dict[str, object]] = {}
    study_manifest_files: dict[str, str] = {}
    for study_name, relative_path_text in STUDY_MANIFEST_RELATIVE_PATHS:
        relative_path = Path(relative_path_text)
        manifest = load_campaign_manifest(root / relative_path)
        if manifest is None:
            raise FileNotFoundError(
                f"Missing or invalid study campaign manifest: {root / relative_path}"
            )
        manifests[study_name] = manifest
        study_manifest_files[study_name] = str(relative_path)

    campaign_id = _shared_text_value(manifests, "campaign_id")
    git_sha = _shared_text_value(manifests, "git_sha")
    python_version = _shared_text_value(manifests, "python_version")
    platform = _shared_text_value(manifests, "platform")
    hostname = _shared_text_value(manifests, "hostname")

    source_root = root if source_results_root is None else Path(source_results_root)
    manifest: dict[str, object] = {
        "manifest_version": RESULTS_ROOT_MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "campaign_id": campaign_id,
        "git_sha": git_sha,
        "python_version": python_version,
        "platform": platform,
        "hostname": hostname,
        "study_manifest_files": study_manifest_files,
        "study_command_lines": _study_command_lines(manifests),
        "tool_paths_by_study": _study_tool_paths(manifests),
        "tool_versions_by_study": _study_tool_versions(manifests),
        "system_snapshot_by_study": _study_system_snapshots(manifests),
    }
    automation_state = _load_automation_state(source_root)
    if automation_state is not None:
        manifest["automation"] = automation_state
    plan_kind = ""
    if automation_state is not None:
        plan_kind = str(automation_state.get("plan_kind") or "").strip()
    manifest["plan_kind"] = plan_kind
    manifest["evaluation_profile"] = (
        str(evaluation_profile).strip()
        if evaluation_profile is not None
        else evaluation_profile_for_plan_kind(plan_kind)
    )
    if source_results_root is not None:
        manifest["source_results_root"] = str(source_root.resolve())
    return manifest


def write_results_root_manifest(
    results_root: str | Path,
    *,
    source_results_root: str | Path | None = None,
    evaluation_profile: str | None = None,
) -> Path:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    path = results_root_manifest_path(root)
    payload = collect_results_root_manifest(
        root,
        source_results_root=source_results_root,
        evaluation_profile=evaluation_profile,
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
