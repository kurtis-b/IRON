#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from .end_to_end.validation import REFERENCE_TOLERANCE_VALIDATION_MODE
from .results_manifest import RESULTS_ROOT_MANIFEST_NAME

ARTIFACT_PROFILES = ("canonical", "p0", "paper")

CANONICAL_EXECUTION_MODES = ("hybrid", "runlist", "offload")
SELECTION_PROVENANCE_VALUES = {
    "best_found_under_declared_topk_joint_search",
    "exhaustive_joint_search_best",
    "tuning_failed",
}
REQUIRED_BASE_FILES = (
    (RESULTS_ROOT_MANIFEST_NAME,),
    ("end_to_end", "results_all_power.csv"),
    ("end_to_end", "tuning_all_power.csv"),
    ("end_to_end", "campaign_manifest.json"),
    ("host_comparison", "results.csv"),
    ("host_comparison", "campaign_manifest.json"),
)
REQUIRED_CANONICAL_ONLY_FILES = (
    ("end_to_end", "correctness_spot_checks.csv"),
    ("end_to_end", "latency_variation.csv"),
    ("end_to_end", "fairness_repeatability.csv"),
    ("host_comparison", "fairness_repeatability.csv"),
)
REQUIRED_CANONICAL_FILES = REQUIRED_BASE_FILES + REQUIRED_CANONICAL_ONLY_FILES
REQUIRED_P0_ONLY_FILES = (
    ("end_to_end", "offload_breakdown.csv"),
    ("end_to_end", "runlist_launch_ablation.csv"),
    ("end_to_end", "staging_ablation.csv"),
    ("end_to_end", "search_validation.csv"),
    ("end_to_end", "search_validation_failure_taxonomy.csv"),
    ("end_to_end", "error_distribution.csv"),
    ("analysis", "crossover_report.csv"),
)
REQUIRED_P0_FILES = REQUIRED_CANONICAL_FILES + REQUIRED_P0_ONLY_FILES
REQUIRED_PAPER_FILES = REQUIRED_BASE_FILES + (
    ("block", "results.csv"),
    ("memory_tile_staging", "results.csv"),
    ("end_to_end", "offload_breakdown.csv"),
    ("end_to_end", "runlist_launch_ablation.csv"),
    ("end_to_end", "staging_ablation.csv"),
    ("end_to_end", "search_validation.csv"),
    ("end_to_end", "search_validation_failure_taxonomy.csv"),
    ("end_to_end", "error_distribution.csv"),
    ("memcpy_bandwidth", "results.csv"),
    ("resource_usage", "dataflow_block_best_configs.csv"),
    ("resource_usage", "hybrid_selected_ops.csv"),
    ("resource_usage", "runlist_selected_ops.csv"),
    ("resource_usage", "offload_selected_ops.csv"),
    ("roofline", "kernel_points.csv"),
    ("roofline", "implementation_points.csv"),
    ("analysis", "crossover_report.csv"),
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _require_files(
    results_root: Path,
    *,
    required_files: tuple[tuple[str, ...], ...] = REQUIRED_CANONICAL_FILES,
) -> None:
    missing = [
        str(results_root.joinpath(*parts))
        for parts in required_files
        if not results_root.joinpath(*parts).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Artifact root is missing required files:\n" + "\n".join(missing)
        )


def _nonempty_values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {
        str(row.get(field) or "").strip()
        for row in rows
        if str(row.get(field) or "").strip()
    }


def _require_columns(
    rows: list[dict[str, str]], required: tuple[str, ...], *, path: Path
) -> None:
    if not rows:
        raise ValueError(f"Expected at least one row in {path}")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _validate_end_to_end_results(
    *,
    results_root: Path,
    manifest: dict[str, object],
) -> str:
    results_path = results_root / "end_to_end" / "results_all_power.csv"
    rows = _load_csv_rows(results_path)
    _require_columns(
        rows,
        (
            "campaign_id",
            "matched_run_id",
            "execution_mode",
            "validation_mode",
            "power_boundary",
            "power_estimation_method",
            "baseline_policy",
            "joint_search_policy",
            "joint_candidate_count_total",
            "joint_candidates_evaluated",
            "selected_joint_rank",
            "selection_provenance",
            "candidate_inventory_json",
            "candidate_count_by_operator_json",
        ),
        path=results_path,
    )
    campaign_ids = _nonempty_values(rows, "campaign_id")
    if len(campaign_ids) != 1:
        raise ValueError(
            f"{results_path} must declare exactly one non-empty campaign_id, saw {sorted(campaign_ids)}"
        )
    campaign_id = next(iter(campaign_ids))
    if str(manifest.get("campaign_id") or "") != campaign_id:
        raise ValueError(
            f"{results_path} campaign_id {campaign_id!r} does not match manifest campaign_id "
            f"{manifest.get('campaign_id')!r}"
        )
    if not isinstance(manifest.get("candidate_policy_by_mode"), dict):
        raise ValueError(
            "end_to_end campaign manifest must publish candidate_policy_by_mode"
        )
    if not isinstance(manifest.get("effective_candidate_inventory_by_case"), dict):
        raise ValueError(
            "end_to_end campaign manifest must publish effective_candidate_inventory_by_case"
        )
    if not str(manifest.get("joint_search_policy") or "").strip():
        raise ValueError(
            "end_to_end campaign manifest must publish joint_search_policy"
        )
    if not isinstance(manifest.get("validation_policy"), dict):
        raise ValueError("end_to_end campaign manifest must publish validation_policy")
    execution_modes = _nonempty_values(rows, "execution_mode")
    if not execution_modes.issubset(set(CANONICAL_EXECUTION_MODES)):
        raise ValueError(
            f"{results_path} contains unsupported execution modes: {sorted(execution_modes)}"
        )
    for row in rows:
        if str(row.get("run_status") or "") != "passed":
            continue
        if str(row.get("validation_mode") or "") != REFERENCE_TOLERANCE_VALIDATION_MODE:
            raise ValueError(
                f"{results_path} contains a paper-facing row without the canonical "
                "reference-tolerance validation mode"
            )
        provenance = str(row.get("selection_provenance") or "").strip()
        if provenance not in SELECTION_PROVENANCE_VALUES:
            raise ValueError(
                f"{results_path} contains unsupported selection_provenance {provenance!r}"
            )
        total = int(float(str(row.get("joint_candidate_count_total") or 0)))
        evaluated = int(float(str(row.get("joint_candidates_evaluated") or 0)))
        if evaluated <= 0 or total <= 0 or evaluated > total:
            raise ValueError(
                f"{results_path} contains invalid joint search accounting: "
                f"total={total} evaluated={evaluated}"
            )
        if provenance == "best_found_under_declared_topk_joint_search" and not (
            evaluated < total
        ):
            raise ValueError(
                f"{results_path} marks a row heuristic-best despite exhaustive evaluation"
            )
        if provenance == "exhaustive_joint_search_best" and not (evaluated == total):
            raise ValueError(
                f"{results_path} marks a row exhaustive-best without evaluating the full joint space"
            )
        inventory = json.loads(str(row.get("candidate_inventory_json") or "{}"))
        counts = json.loads(str(row.get("candidate_count_by_operator_json") or "{}"))
        if not isinstance(inventory, dict) or not isinstance(counts, dict):
            raise ValueError(
                f"{results_path} contains malformed candidate inventory JSON"
            )
        for operator_name, candidate_ids in inventory.items():
            if not isinstance(candidate_ids, list):
                raise ValueError(
                    f"{results_path} candidate_inventory_json must map operators to lists"
                )
            expected_count = int(counts.get(operator_name, -1))
            if expected_count != len(candidate_ids):
                raise ValueError(
                    f"{results_path} candidate count mismatch for {operator_name}: "
                    f"expected {expected_count}, saw {len(candidate_ids)}"
                )
    declared_study_case_ids = manifest.get("study_case_ids")
    if isinstance(declared_study_case_ids, list):
        actual_study_case_ids = sorted(_nonempty_values(rows, "study_case_id"))
        if (
            sorted(str(value) for value in declared_study_case_ids)
            != actual_study_case_ids
        ):
            raise ValueError(
                f"{results_path} study_case_id coverage does not match the manifest"
            )
    return campaign_id


def _validate_host_comparison_results(
    *,
    results_root: Path,
    manifest: dict[str, object],
    expected_campaign_id: str,
) -> None:
    results_path = results_root / "host_comparison" / "results.csv"
    rows = _load_csv_rows(results_path)
    _require_columns(
        rows,
        (
            "campaign_id",
            "matched_run_id",
            "metric",
            "power_boundary_policy_family",
            "igpu_power_boundary",
            "npu_power_boundary",
            "igpu_power_estimation_method",
            "npu_power_estimation_method",
            "igpu_baseline_policy",
            "npu_baseline_policy",
            "cpu",
            "cpu_turbostat",
            *CANONICAL_EXECUTION_MODES,
        ),
        path=results_path,
    )
    campaign_ids = _nonempty_values(rows, "campaign_id")
    if campaign_ids != {expected_campaign_id}:
        raise ValueError(
            f"{results_path} campaign IDs {sorted(campaign_ids)} do not match end-to-end campaign "
            f"{expected_campaign_id!r}"
        )
    if str(manifest.get("campaign_id") or "") != expected_campaign_id:
        raise ValueError(
            f"{results_path} manifest campaign_id {manifest.get('campaign_id')!r} does not match "
            f"end-to-end campaign_id {expected_campaign_id!r}"
        )
    if not isinstance(manifest.get("validation_policy"), dict):
        raise ValueError(
            "host_comparison campaign manifest must publish validation_policy"
        )
    for row in rows:
        for execution_mode in CANONICAL_EXECUTION_MODES:
            if str(row.get(execution_mode) or "").strip() == "":
                raise ValueError(
                    f"{results_path} contains an incomplete comparison row missing "
                    f"{execution_mode} data"
                )
        if str(row.get("metric") or "") != "effective_gflops_per_sec_per_watt":
            continue
        if str(row.get("igpu_power_boundary") or "") != str(
            row.get("npu_power_boundary") or ""
        ):
            raise ValueError(f"{results_path} mixes power boundaries in a per-watt row")
        if str(row.get("igpu_power_estimation_method") or "") != str(
            row.get("npu_power_estimation_method") or ""
        ):
            raise ValueError(
                f"{results_path} mixes power estimation methods in a per-watt row"
            )
        if str(row.get("igpu_baseline_policy") or "") != str(
            row.get("npu_baseline_policy") or ""
        ):
            raise ValueError(
                f"{results_path} mixes baseline policies in a per-watt row"
            )


def _scan_legacy_dataflow_labels(results_root: Path) -> None:
    stale_paths: list[str] = []
    for subdir in ("end_to_end", "host_comparison"):
        root = results_root / subdir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            rel_path = path.relative_to(results_root)
            if "dataflow" in str(rel_path).lower():
                stale_paths.append(str(rel_path))
                continue
            if path.suffix.lower() not in {".csv", ".svg", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "dataflow" in text.lower():
                stale_paths.append(str(rel_path))
    if stale_paths:
        raise ValueError(
            "Canonical artifact root still contains legacy 'dataflow' labels:\n"
            + "\n".join(sorted(set(stale_paths)))
        )


def _validate_results_root_manifest(
    *,
    manifest: dict[str, object],
    expected_campaign_id: str,
    end_to_end_manifest: dict[str, object],
    host_manifest: dict[str, object],
) -> None:
    if int(manifest.get("manifest_version") or 0) <= 0:
        raise ValueError(
            "artifact root manifest must publish a positive manifest_version"
        )
    if str(manifest.get("campaign_id") or "") != expected_campaign_id:
        raise ValueError(
            "artifact root manifest campaign_id does not match the canonical campaign"
        )
    git_sha = str(manifest.get("git_sha") or "").strip()
    if not git_sha:
        raise ValueError("artifact root manifest must publish git_sha")
    expected_git_sha = str(end_to_end_manifest.get("git_sha") or "").strip()
    host_git_sha = str(host_manifest.get("git_sha") or "").strip()
    if (
        not expected_git_sha
        or not host_git_sha
        or git_sha
        not in {
            expected_git_sha,
            host_git_sha,
        }
    ):
        raise ValueError(
            "artifact root manifest git_sha must match study campaign manifests"
        )
    study_manifest_files = manifest.get("study_manifest_files")
    if not isinstance(study_manifest_files, dict):
        raise ValueError("artifact root manifest must publish study_manifest_files")
    expected_manifest_files = {
        "end_to_end": "end_to_end/campaign_manifest.json",
        "host_comparison": "host_comparison/campaign_manifest.json",
    }
    for study_name, relative_path in expected_manifest_files.items():
        if str(study_manifest_files.get(study_name) or "") != relative_path:
            raise ValueError(
                f"artifact root manifest must map {study_name} to {relative_path}"
            )
    command_lines = manifest.get("study_command_lines")
    if not isinstance(command_lines, dict):
        raise ValueError("artifact root manifest must publish study_command_lines")
    for study_name in expected_manifest_files:
        if not isinstance(command_lines.get(study_name), list):
            raise ValueError(
                f"artifact root manifest must include command_line entries for {study_name}"
            )
    tool_versions_by_study = manifest.get("tool_versions_by_study")
    if not isinstance(tool_versions_by_study, dict):
        raise ValueError("artifact root manifest must publish tool_versions_by_study")
    system_snapshot_by_study = manifest.get("system_snapshot_by_study")
    if not isinstance(system_snapshot_by_study, dict):
        raise ValueError(
            "artifact root manifest must publish system_snapshot_by_study"
        )
    plan_kind = manifest.get("plan_kind")
    if not isinstance(plan_kind, str):
        raise ValueError("artifact root manifest must publish plan_kind")
    evaluation_profile = manifest.get("evaluation_profile")
    if not isinstance(evaluation_profile, str):
        raise ValueError("artifact root manifest must publish evaluation_profile")
    automation = manifest.get("automation")
    if isinstance(automation, dict):
        automation_plan_kind = str(automation.get("plan_kind") or "")
        if automation_plan_kind and plan_kind != automation_plan_kind:
            raise ValueError(
                "artifact root manifest plan_kind does not match automation.plan_kind"
            )


def _validate_results_root(
    results_root: str | Path,
    *,
    required_files: tuple[tuple[str, ...], ...],
) -> None:
    root = Path(results_root)
    _require_files(root, required_files=required_files)
    results_root_manifest = _load_manifest(root / RESULTS_ROOT_MANIFEST_NAME)
    end_to_end_manifest = _load_manifest(root / "end_to_end" / "campaign_manifest.json")
    host_manifest = _load_manifest(root / "host_comparison" / "campaign_manifest.json")
    expected_campaign_id = _validate_end_to_end_results(
        results_root=root,
        manifest=end_to_end_manifest,
    )
    _validate_host_comparison_results(
        results_root=root,
        manifest=host_manifest,
        expected_campaign_id=expected_campaign_id,
    )
    _validate_results_root_manifest(
        manifest=results_root_manifest,
        expected_campaign_id=expected_campaign_id,
        end_to_end_manifest=end_to_end_manifest,
        host_manifest=host_manifest,
    )
    _scan_legacy_dataflow_labels(root)


def validate_canonical_results_root(results_root: str | Path) -> None:
    _validate_results_root(
        results_root,
        required_files=REQUIRED_CANONICAL_FILES,
    )


def validate_p0_results_root(results_root: str | Path) -> None:
    _validate_results_root(
        results_root,
        required_files=REQUIRED_P0_FILES,
    )


def validate_paper_results_root(results_root: str | Path) -> None:
    _validate_results_root(
        results_root,
        required_files=REQUIRED_PAPER_FILES,
    )


def validate_results_root(
    results_root: str | Path,
    *,
    artifact_profile: str,
) -> None:
    if artifact_profile == "canonical":
        validate_canonical_results_root(results_root)
        return
    if artifact_profile == "p0":
        validate_p0_results_root(results_root)
        return
    if artifact_profile == "paper":
        validate_paper_results_root(results_root)
        return
    raise ValueError(f"Unsupported artifact profile: {artifact_profile}")
