#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import heapq
import json
import logging
import sys
from pathlib import Path

from ..campaign import (
    DEFAULT_CAMPAIGN_ID,
    current_git_sha,
    load_output_campaign_manifest,
    manifest_matches_checkout,
    normalize_campaign_id,
    normalize_matched_run_id,
    normalize_repeat_index,
    write_campaign_manifest,
)
from ..npu_runtime_checks import warn_if_npu_power_mode_not_turbo
from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    SEQUENCE_LADDER,
    WORKLOAD_VARIANTS,
    EndToEndCase,
    candidate_count_by_operator,
    candidate_inventory_for_case,
    candidate_policy_manifest,
    candidate_table_for_case,
    iter_cases,
    mode_operators,
)
from .modes import (
    benchmark_mode,
    benchmark_mode_power_only,
    benchmark_operator_candidate,
    resolve_mode_operator_config,
)
from .power import PERSISTED_POWER_RESULT_FIELDS, SUPPORTED_POWER_BACKENDS
from .validation import (
    REFERENCE_TOLERANCE_VALIDATION_MODE,
    validation_policy_manifest,
)

LOGGER = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)
JOINT_SEARCH_TOP_K = 3
HEURISTIC_SELECTION_PROVENANCE = "best_found_under_declared_topk_joint_search"
EXHAUSTIVE_SELECTION_PROVENANCE = "exhaustive_joint_search_best"
FAILED_SELECTION_PROVENANCE = "tuning_failed"
END_TO_END_STUDY_NAME = "transformer_layer end-to-end study"


def _pattern_label(execution_mode: str) -> str:
    if execution_mode == "hybrid":
        return "Hybrid"
    if execution_mode == "runlist":
        return "Runlist"
    if execution_mode == "offload":
        return "GEMM Offload"
    return execution_mode


TUNING_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "internal_operator",
    "candidate_id",
    "selection_stage",
    "joint_rank",
    "joint_candidate_ids_json",
    "estimated_joint_latency_ms",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "warmup_runs",
    "runs_per_sample",
    "latency_sample_count",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "bandwidth_gbps",
    "validation_error_count",
    "run_status",
    "failure_message",
    "operator_config_json",
    "is_operator_best",
)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "matched_run_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "backend",
    "execution_mode",
    "pattern_label",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "batch_size",
    "dtype",
    "use_bias",
    "weights_source",
    "joint_search_policy",
    "joint_candidate_count_total",
    "joint_candidates_evaluated",
    "selected_joint_rank",
    "selection_provenance",
    "candidate_inventory_json",
    "candidate_count_by_operator_json",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "latency_sample_count",
    "timed_total_sec",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "compile_setup_time_ms",
    "host_qkv_precompute_ms",
    "effective_gflops_per_sec",
    "power_backend",
    *PERSISTED_POWER_RESULT_FIELDS,
    "effective_gflops_per_sec_per_watt",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "process_model",
    "validation_mode",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
    "is_best",
)


def _case_descriptor(case: EndToEndCase) -> str:
    return (
        f"{case.study_case_id} seq_len={case.seq_len} "
        f"hidden={case.hidden_size} inter={case.intermediate_size} "
        f"heads={case.num_attention_heads}"
    )


def _result_summary(result: dict[str, object]) -> str:
    status = str(result.get("run_status", ""))
    avg_latency_ms = result.get("avg_latency_ms")
    if status == "passed" and avg_latency_ms not in ("", None):
        return (
            f"passed avg_latency_ms={float(avg_latency_ms):.4f} "
            f"validation_error_count={int(result.get('validation_error_count', 0))}"
        )
    failure_message = str(result.get("failure_message", ""))
    return f"{status}: {failure_message}" if failure_message else status


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "results_all_power.csv"
    )


def default_tuning_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "tuning_all_power.csv"
    )


def default_resume_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2]
        / "results_final"
        / "end_to_end"
        / output_path.name
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def default_resume_tuning_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2]
        / "results_final"
        / "end_to_end"
        / output_path.name
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def _compatible_resume_paths(
    paths: tuple[Path, ...],
    *,
    campaign_id: str,
) -> tuple[Path, ...]:
    git_sha = current_git_sha()
    compatible_paths: list[Path] = []
    for path in paths:
        manifest = load_output_campaign_manifest(path)
        if manifest_matches_checkout(
            manifest,
            campaign_id=campaign_id,
            git_sha=git_sha,
            study_name=END_TO_END_STUDY_NAME,
        ):
            compatible_paths.append(path)
            continue
        LOGGER.warning(
            "Skipping resume path %s because its campaign manifest is missing or "
            "does not match campaign_id=%s and git_sha=%s",
            path,
            normalize_campaign_id(campaign_id),
            git_sha or "unknown",
        )
    return tuple(compatible_paths)


def _resume_execution_mode(value: object) -> str:
    return str(value or "")


def _resume_workload_variant(row: dict[str, str]) -> str:
    workload_variant = str(row.get("workload_variant") or "")
    if workload_variant:
        return workload_variant
    study_case_id = str(row.get("study_case_id") or "")
    if study_case_id in FAMILY_SPECS:
        return FAMILY_SPECS[study_case_id].workload_variant
    return ""


def _resume_campaign_id(row: dict[str, str]) -> str:
    return normalize_campaign_id(row.get("campaign_id"))


def _resume_repeat_index(row: dict[str, str]) -> int:
    return normalize_repeat_index(row.get("repeat_index"))


def _tuning_row_key(
    row: dict[str, str],
) -> tuple[str, int, str, str, str, str, int, str]:
    return (
        _resume_campaign_id(row),
        _resume_repeat_index(row),
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        str(row.get("internal_operator") or ""),
        int(float(str(row.get("seq_len") or 0))),
        str(row.get("candidate_id") or ""),
    )


def load_existing_tuning_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, int, str, str, str, str, int, str], dict[str, object]]:
    rows: dict[tuple[str, int, str, str, str, str, int, str], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("candidate_id") or ""):
                    continue
                rows[_tuning_row_key(row)] = dict(row)
    return rows


def reusable_tuning_row(
    existing_rows: dict[
        tuple[str, int, str, str, str, str, int, str], dict[str, object]
    ],
    *,
    campaign_id: str,
    repeat_index: int,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    operator_name: str,
    seq_len: int,
    candidate_id: str,
    resolved_config_json: str,
) -> dict[str, object] | None:
    row = existing_rows.get(
        (
            normalize_campaign_id(campaign_id),
            int(repeat_index),
            study_case_id,
            workload_variant,
            execution_mode,
            operator_name,
            seq_len,
            candidate_id,
        )
    )
    if row is None:
        return None
    if str(row.get("operator_config_json") or "") != resolved_config_json:
        return None
    if str(row.get("run_status") or "") != "passed":
        return None
    required_fields = (
        "latency_sample_count",
        "min_latency_ms",
        "max_latency_ms",
    )
    if any(row.get(field) in ("", None) for field in required_fields):
        return None
    return dict(row)


def _final_row_key(row: dict[str, str]) -> tuple[str, int, str, str, str, int]:
    return (
        _resume_campaign_id(row),
        _resume_repeat_index(row),
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        int(float(str(row.get("seq_len") or 0))),
    )


def load_existing_final_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, int, str, str, str, int], dict[str, object]]:
    rows: dict[tuple[str, int, str, str, str, int], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("execution_mode") or ""):
                    continue
                rows[_final_row_key(row)] = dict(row)
    return rows


def reusable_final_row(
    existing_rows: dict[tuple[str, int, str, str, str, int], dict[str, object]],
    *,
    campaign_id: str,
    repeat_index: int,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    seq_len: int,
    selected_candidate_ids_json: str,
    selected_config_json: str,
    power_backend: str,
) -> dict[str, object] | None:
    row = existing_rows.get(
        (
            normalize_campaign_id(campaign_id),
            int(repeat_index),
            study_case_id,
            workload_variant,
            execution_mode,
            seq_len,
        )
    )
    if row is None:
        return None
    if str(row.get("run_status") or "") != "passed":
        return None
    if str(row.get("selected_candidate_ids_json") or "") != selected_candidate_ids_json:
        return None
    if str(row.get("selected_config_json") or "") != selected_config_json:
        return None
    if str(row.get("validation_mode") or "") != REFERENCE_TOLERANCE_VALIDATION_MODE:
        return None
    required_selection_fields = (
        "joint_search_policy",
        "joint_candidate_count_total",
        "joint_candidates_evaluated",
        "selection_provenance",
        "candidate_inventory_json",
        "candidate_count_by_operator_json",
    )
    if any(
        str(row.get(field) or "").strip() == "" for field in required_selection_fields
    ):
        return None
    if power_backend != "auto" and str(row.get("power_backend") or "") != power_backend:
        return None
    required_fields = (
        "latency_sample_count",
        "min_latency_ms",
        "max_latency_ms",
    )
    if any(row.get(field) in ("", None) for field in required_fields):
        return None
    if str(row.get("power_backend") or "") != "none":
        power_fields = ("min_power_w", "max_power_w", "power_sample_count")
        if any(row.get(field) in ("", None) for field in power_fields):
            return None
        metadata_fields = ("power_boundary", "power_estimation_method", "sensor_source")
        if any(row.get(field) in ("", None) for field in metadata_fields):
            return None
    return dict(row)


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 256:
        return (1, 100)
    if seq_len <= 2048:
        return (1, 10)
    if seq_len <= 4096:
        return (1, 10)
    return (1, 5)


def json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _joint_search_policy(*, top_k: int) -> str:
    return (
        "isolated_operator_screen_then_top_"
        f"{int(top_k)}_estimated_joint_full_patterns"
    )


def _joint_candidate_count_total(
    *,
    successful_rows_by_operator: dict[str, list[dict[str, object]]],
    operator_order: tuple[str, ...],
) -> int:
    total = 1
    for operator_name in operator_order:
        total *= len(successful_rows_by_operator.get(operator_name, ()))
    return total


def _selection_provenance(
    *,
    joint_candidate_count_total: int,
    joint_candidates_evaluated: int,
) -> str:
    if joint_candidate_count_total <= 0 or joint_candidates_evaluated <= 0:
        return FAILED_SELECTION_PROVENANCE
    if joint_candidates_evaluated >= joint_candidate_count_total:
        return EXHAUSTIVE_SELECTION_PROVENANCE
    return HEURISTIC_SELECTION_PROVENANCE


def _candidate_inventory_metadata(case: EndToEndCase) -> dict[str, dict[str, str]]:
    inventory = candidate_inventory_for_case(case.study_case_id, case.seq_len)
    inventory_json = {
        execution_mode: json_dumps(
            {
                operator_name: list(candidate_ids)
                for operator_name, candidate_ids in operator_inventory.items()
            }
        )
        for execution_mode, operator_inventory in inventory.items()
    }
    count_json = {
        execution_mode: json_dumps(candidate_count_by_operator(operator_inventory))
        for execution_mode, operator_inventory in inventory.items()
    }
    return {
        execution_mode: {
            "candidate_inventory_json": inventory_json[execution_mode],
            "candidate_count_by_operator_json": count_json[execution_mode],
        }
        for execution_mode in EXECUTION_MODES
    }


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    best_index_by_group: dict[tuple[str, int, str, int], int] = {}
    best_latency_by_group: dict[tuple[str, int, str, int], float] = {}

    for index, row in enumerate(rows):
        row["is_best"] = False
        if row["run_status"] != "passed" or row["avg_latency_ms"] in ("", None):
            continue
        group = (
            normalize_campaign_id(row.get("campaign_id")),
            normalize_repeat_index(row.get("repeat_index")),
            str(row["study_case_id"]),
            int(row["seq_len"]),
        )
        latency = float(row["avg_latency_ms"])
        if group not in best_latency_by_group or latency < best_latency_by_group[group]:
            best_latency_by_group[group] = latency
            best_index_by_group[group] = index

    for index in best_index_by_group.values():
        rows[index]["is_best"] = True


def _failed_final_result(
    *,
    power_backend: str,
    failure_message: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "measured_inference_count": 0,
        "latency_sample_count": 0,
        "timed_total_sec": 0.0,
        "avg_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "compile_setup_time_ms": None,
        "host_qkv_precompute_ms": None,
        "effective_gflops_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "effective_gflops_per_sec_per_watt": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_mode": REFERENCE_TOLERANCE_VALIDATION_MODE,
        "validation_error_count": 0,
        "joint_search_policy": _joint_search_policy(top_k=JOINT_SEARCH_TOP_K),
        "joint_candidate_count_total": 0,
        "joint_candidates_evaluated": 0,
        "selected_joint_rank": None,
        "selection_provenance": FAILED_SELECTION_PROVENANCE,
        "candidate_inventory_json": json_dumps({}),
        "candidate_count_by_operator_json": json_dumps({}),
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }
    for field in PERSISTED_POWER_RESULT_FIELDS:
        result[field] = None
    return result


def _selected_default_row(
    *,
    campaign_id: str,
    repeat_index: int,
    case: EndToEndCase,
    execution_mode: str,
    operator_name: str,
    candidate_id: str,
    resolved_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
    run_status: str,
    failure_message: str,
) -> dict[str, object]:
    return {
        "study_id": "end_to_end_tuning",
        "campaign_id": normalize_campaign_id(campaign_id),
        "repeat_index": int(repeat_index),
        "study_case_id": case.study_case_id,
        "study_case_label": case.study_case_label,
        "workload_variant": case.workload_variant,
        "execution_mode": execution_mode,
        "internal_operator": operator_name,
        "candidate_id": candidate_id,
        "selection_stage": "isolated_operator",
        "joint_rank": "",
        "joint_candidate_ids_json": "",
        "estimated_joint_latency_ms": "",
        "seq_len": case.seq_len,
        "hidden_size": case.hidden_size,
        "intermediate_size": case.intermediate_size,
        "num_attention_heads": case.num_attention_heads,
        "attention_head_size": case.attention_head_size,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "latency_sample_count": "",
        "avg_latency_ms": "",
        "min_latency_ms": "",
        "max_latency_ms": "",
        "bandwidth_gbps": "",
        "validation_error_count": "",
        "run_status": run_status,
        "failure_message": failure_message,
        "operator_config_json": json_dumps(resolved_config),
        "is_operator_best": True,
        "_resolved_config": resolved_config,
    }


def _sorted_successful_operator_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] not in ("", None)
        ),
        key=lambda row: float(row["avg_latency_ms"]),
    )


def _joint_candidate_combinations(
    *,
    successful_rows_by_operator: dict[str, list[dict[str, object]]],
    operator_order: tuple[str, ...],
    limit: int,
) -> list[tuple[float, tuple[int, ...]]]:
    if not operator_order:
        return []
    for operator_name in operator_order:
        if not successful_rows_by_operator.get(operator_name):
            return []

    initial_indexes = tuple(0 for _ in operator_order)

    def estimated_total(indexes: tuple[int, ...]) -> float:
        return sum(
            float(successful_rows_by_operator[operator_name][index]["avg_latency_ms"])
            for operator_name, index in zip(operator_order, indexes, strict=True)
        )

    pending: list[tuple[float, tuple[int, ...]]] = [
        (estimated_total(initial_indexes), initial_indexes)
    ]
    seen = {initial_indexes}
    selected: list[tuple[float, tuple[int, ...]]] = []
    while pending and len(selected) < int(limit):
        estimated_latency_ms, indexes = heapq.heappop(pending)
        selected.append((estimated_latency_ms, indexes))
        for axis, operator_name in enumerate(operator_order):
            next_index = indexes[axis] + 1
            if next_index >= len(successful_rows_by_operator[operator_name]):
                continue
            next_indexes = list(indexes)
            next_indexes[axis] = next_index
            next_indexes_tuple = tuple(next_indexes)
            if next_indexes_tuple in seen:
                continue
            seen.add(next_indexes_tuple)
            heapq.heappush(
                pending,
                (estimated_total(next_indexes_tuple), next_indexes_tuple),
            )
    return selected


def _joint_tuning_row(
    *,
    campaign_id: str,
    repeat_index: int,
    case: EndToEndCase,
    execution_mode: str,
    joint_rank: int,
    estimated_joint_latency_ms: float,
    selected_candidate_ids: dict[str, str],
    selected_config: dict[str, dict[str, object]],
    warmup_runs: int,
    runs_per_sample: int,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "study_id": "end_to_end_tuning",
        "campaign_id": normalize_campaign_id(campaign_id),
        "repeat_index": int(repeat_index),
        "study_case_id": case.study_case_id,
        "study_case_label": case.study_case_label,
        "workload_variant": case.workload_variant,
        "execution_mode": execution_mode,
        "internal_operator": "__joint__",
        "candidate_id": f"joint_rank_{joint_rank}",
        "selection_stage": "joint_mode",
        "joint_rank": joint_rank,
        "joint_candidate_ids_json": json_dumps(selected_candidate_ids),
        "estimated_joint_latency_ms": estimated_joint_latency_ms,
        "seq_len": case.seq_len,
        "hidden_size": case.hidden_size,
        "intermediate_size": case.intermediate_size,
        "num_attention_heads": case.num_attention_heads,
        "attention_head_size": case.attention_head_size,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "operator_config_json": json_dumps(selected_config),
        "is_operator_best": False,
        **result,
    }


def _validate_joint_mode_candidates(
    case: EndToEndCase,
    *,
    campaign_id: str,
    repeat_index: int,
    execution_mode: str,
    operator_order: tuple[str, ...],
    successful_rows_by_operator: dict[str, list[dict[str, object]]],
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    top_k: int = JOINT_SEARCH_TOP_K,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, dict[str, object]],
    dict[str, object],
    str,
]:
    joint_candidate_count_total = _joint_candidate_count_total(
        successful_rows_by_operator=successful_rows_by_operator,
        operator_order=operator_order,
    )
    joint_candidates = _joint_candidate_combinations(
        successful_rows_by_operator=successful_rows_by_operator,
        operator_order=operator_order,
        limit=top_k,
    )
    joint_search_metadata = {
        "joint_search_policy": _joint_search_policy(top_k=top_k),
        "joint_candidate_count_total": int(joint_candidate_count_total),
        "joint_candidates_evaluated": len(joint_candidates),
        "selected_joint_rank": None,
        "selection_provenance": _selection_provenance(
            joint_candidate_count_total=joint_candidate_count_total,
            joint_candidates_evaluated=len(joint_candidates),
        ),
    }
    if not joint_candidates:
        return (
            [],
            {},
            {},
            joint_search_metadata,
            "tuning_failed: no passing joint candidate",
        )

    joint_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_candidate_ids: dict[str, str] = {}
    best_config: dict[str, dict[str, object]] = {}

    for joint_rank, (estimated_joint_latency_ms, indexes) in enumerate(
        joint_candidates, start=1
    ):
        selected_candidate_ids: dict[str, str] = {}
        selected_config: dict[str, dict[str, object]] = {}
        for operator_name, index in zip(operator_order, indexes, strict=True):
            selected_row = successful_rows_by_operator[operator_name][index]
            selected_candidate_ids[operator_name] = str(selected_row["candidate_id"])
            selected_config[operator_name] = dict(selected_row["_resolved_config"])
        result = benchmark_mode(
            execution_mode,
            case.workload,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
            power_backend="none",
            operator_config=selected_config,
            include_reference_output=True,
            scope_suffix=f"joint_rank_{joint_rank}_repeat_{repeat_index}",
        )
        joint_row = _joint_tuning_row(
            campaign_id=campaign_id,
            repeat_index=repeat_index,
            case=case,
            execution_mode=execution_mode,
            joint_rank=joint_rank,
            estimated_joint_latency_ms=estimated_joint_latency_ms,
            selected_candidate_ids=selected_candidate_ids,
            selected_config=selected_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            result=result,
        )
        joint_rows.append(joint_row)
        if joint_row["run_status"] != "passed" or joint_row["avg_latency_ms"] in (
            "",
            None,
        ):
            continue
        if best_row is None or float(joint_row["avg_latency_ms"]) < float(
            best_row["avg_latency_ms"]
        ):
            best_row = joint_row
            best_candidate_ids = selected_candidate_ids
            best_config = selected_config

    if best_row is None:
        return (
            joint_rows,
            {},
            {},
            joint_search_metadata,
            "tuning_failed: all joint full-pattern candidates failed",
        )
    best_row["is_operator_best"] = True
    joint_search_metadata["selected_joint_rank"] = int(best_row["joint_rank"])
    return joint_rows, best_candidate_ids, best_config, joint_search_metadata, ""


def _paired_mode_result(
    case: EndToEndCase,
    *,
    execution_mode: str,
    selected_config: dict[str, dict[str, object]],
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    power_backend: str,
    repeat_index: int,
) -> dict[str, object]:
    latency_result = benchmark_mode(
        execution_mode,
        case.workload,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
        power_backend="none",
        operator_config=selected_config,
        include_reference_output=True,
        scope_suffix=f"latency_repeat_{repeat_index}",
    )
    if latency_result.get("run_status") != "passed":
        return dict(latency_result)

    power_result = benchmark_mode_power_only(
        execution_mode,
        case.workload,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
        power_backend=power_backend,
        operator_config=selected_config,
        avg_latency_ms=latency_result.get("avg_latency_ms"),
        timed_total_sec=float(latency_result.get("timed_total_sec") or 0.0),
        effective_gflops_per_sec_value=latency_result.get("effective_gflops_per_sec"),
        scope_suffix=f"power_repeat_{repeat_index}",
    )
    if power_result.get("run_status") != "passed":
        failed = dict(latency_result)
        failed.update(
            {
                "power_backend": power_result.get(
                    "power_backend",
                    "none" if power_backend == "auto" else power_backend,
                ),
                "run_status": power_result.get("run_status"),
                "failure_message": power_result.get("failure_message"),
            }
        )
        for field in PERSISTED_POWER_RESULT_FIELDS:
            failed[field] = power_result.get(field)
        failed["effective_gflops_per_sec_per_watt"] = power_result.get(
            "effective_gflops_per_sec_per_watt"
        )
        return failed

    paired = dict(latency_result)
    paired["power_backend"] = power_result.get("power_backend", paired["power_backend"])
    for field in PERSISTED_POWER_RESULT_FIELDS:
        paired[field] = power_result.get(field)
    paired["effective_gflops_per_sec_per_watt"] = power_result.get(
        "effective_gflops_per_sec_per_watt"
    )
    paired["validation_mode"] = REFERENCE_TOLERANCE_VALIDATION_MODE
    return paired


def tune_mode(
    case: EndToEndCase,
    *,
    campaign_id: str,
    repeat_index: int,
    execution_mode: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    validate_long_seq_runlist: bool = True,
    existing_tuning_rows: (
        dict[tuple[str, int, str, str, str, str, int, str], dict[str, object]] | None
    ) = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, dict[str, object]],
    dict[str, object],
    str,
]:
    candidates_by_mode = candidate_table_for_case(case.study_case_id, case.seq_len)
    tuning_rows: list[dict[str, object]] = []
    successful_rows_by_operator: dict[str, list[dict[str, object]]] = {}
    operator_names = mode_operators(execution_mode, case.workload_variant)
    candidate_inventory_metadata = _candidate_inventory_metadata(case)[execution_mode]
    selection_metadata: dict[str, object] = {
        **candidate_inventory_metadata,
        "joint_search_policy": _joint_search_policy(top_k=JOINT_SEARCH_TOP_K),
        "joint_candidate_count_total": 0,
        "joint_candidates_evaluated": 0,
        "selected_joint_rank": None,
        "selection_provenance": FAILED_SELECTION_PROVENANCE,
    }

    LOGGER.info(
        "Tuning %s for %s (%d operator groups, warmup=%d, timed=%d)",
        execution_mode,
        _case_descriptor(case),
        len(operator_names),
        warmup_runs,
        runs_per_sample,
    )

    for operator_index, operator_name in enumerate(operator_names, start=1):
        operator_rows: list[dict[str, object]] = []
        candidates = candidates_by_mode[execution_mode][operator_name]
        LOGGER.info(
            "[%s] Operator %d/%d: %s (%d candidates)",
            execution_mode,
            operator_index,
            len(operator_names),
            operator_name,
            len(candidates),
        )
        for candidate_index, candidate in enumerate(candidates, start=1):
            LOGGER.info(
                "[%s] Candidate %d/%d for %s: %s",
                execution_mode,
                candidate_index,
                len(candidates),
                operator_name,
                candidate["candidate_id"],
            )
            resolved_config = resolve_mode_operator_config(
                execution_mode,
                case.workload,
                {operator_name: dict(candidate["config"])},
            )[operator_name]
            reused_row = reusable_tuning_row(
                {} if existing_tuning_rows is None else existing_tuning_rows,
                campaign_id=campaign_id,
                repeat_index=repeat_index,
                study_case_id=case.study_case_id,
                workload_variant=case.workload_variant,
                execution_mode=execution_mode,
                operator_name=operator_name,
                seq_len=case.seq_len,
                candidate_id=candidate["candidate_id"],
                resolved_config_json=json_dumps(resolved_config),
            )
            if reused_row is not None:
                LOGGER.info(
                    "[%s] Reusing %s candidate %s from existing tuning results",
                    execution_mode,
                    operator_name,
                    candidate["candidate_id"],
                )
                reused_row["_resolved_config"] = resolved_config
                operator_rows.append(reused_row)
                continue
            result = benchmark_operator_candidate(
                execution_mode,
                operator_name,
                case.workload,
                dict(candidate["config"]),
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
            )
            operator_rows.append(
                {
                    "study_id": "end_to_end_tuning",
                    "campaign_id": normalize_campaign_id(campaign_id),
                    "repeat_index": int(repeat_index),
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "workload_variant": case.workload_variant,
                    "execution_mode": execution_mode,
                    "internal_operator": operator_name,
                    "candidate_id": candidate["candidate_id"],
                    "selection_stage": "isolated_operator",
                    "joint_rank": "",
                    "joint_candidate_ids_json": "",
                    "estimated_joint_latency_ms": "",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "warmup_runs": warmup_runs,
                    "runs_per_sample": runs_per_sample,
                    "operator_config_json": json_dumps(resolved_config),
                    "is_operator_best": False,
                    "_resolved_config": resolved_config,
                    **result,
                }
            )
            LOGGER.info(
                "[%s] %s candidate %s -> %s",
                execution_mode,
                operator_name,
                candidate["candidate_id"],
                _result_summary(result),
            )

        successful_rows = _sorted_successful_operator_rows(operator_rows)
        tuning_rows.extend(operator_rows)
        if not successful_rows:
            LOGGER.warning(
                "[%s] No passing candidate for %s in %s",
                execution_mode,
                operator_name,
                _case_descriptor(case),
            )
            return (
                tuning_rows,
                {},
                {},
                selection_metadata,
                f"tuning_failed: no passing candidate for {operator_name}",
            )
        successful_rows[0]["is_operator_best"] = True
        successful_rows_by_operator[operator_name] = successful_rows
        LOGGER.info(
            "[%s] Best isolated %s candidate %s (avg_latency_ms=%.4f)",
            execution_mode,
            operator_name,
            successful_rows[0]["candidate_id"],
            float(successful_rows[0]["avg_latency_ms"]),
        )

    (
        joint_rows,
        selected_candidate_ids,
        selected_config,
        joint_search_metadata,
        joint_failure,
    ) = _validate_joint_mode_candidates(
        case,
        campaign_id=campaign_id,
        repeat_index=repeat_index,
        execution_mode=execution_mode,
        operator_order=operator_names,
        successful_rows_by_operator=successful_rows_by_operator,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
    )
    selection_metadata.update(joint_search_metadata)
    tuning_rows.extend(joint_rows)
    if joint_failure:
        return tuning_rows, {}, {}, selection_metadata, joint_failure

    if (
        validate_long_seq_runlist
        and execution_mode == "runlist"
        and case.seq_len >= 8192
    ):
        LOGGER.info(
            "[%s] Running long-sequence mode smoke for %s with selected operators %s",
            execution_mode,
            _case_descriptor(case),
            json_dumps(selected_candidate_ids),
        )
        smoke_result = benchmark_mode(
            execution_mode,
            case.workload,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
            power_backend="none",
            operator_config=selected_config,
            include_reference_output=True,
            scope_suffix=f"long_seq_smoke_repeat_{repeat_index}",
        )
        LOGGER.info(
            "[%s] Long-sequence mode smoke for %s -> %s",
            execution_mode,
            _case_descriptor(case),
            _result_summary(smoke_result),
        )
        if smoke_result["run_status"] != "passed":
            return (
                tuning_rows,
                selected_candidate_ids,
                selected_config,
                selection_metadata,
                "tuning_failed: long-sequence runlist mode smoke failed: "
                + str(smoke_result["failure_message"]),
            )

    return tuning_rows, selected_candidate_ids, selected_config, selection_metadata, ""


def build_rows(
    case: EndToEndCase,
    *,
    campaign_id: str,
    repeat_index: int,
    mode_filter: str,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    power_backend: str,
    existing_tuning_rows: (
        dict[tuple[str, int, str, str, str, str, int, str], dict[str, object]] | None
    ) = None,
    existing_final_rows: (
        dict[tuple[str, int, str, str, str, int], dict[str, object]] | None
    ) = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_modes = EXECUTION_MODES if mode_filter == "all" else (mode_filter,)
    resolved_warmup_runs, resolved_runs_per_sample = iteration_schedule(case.seq_len)
    if warmup_runs is not None:
        resolved_warmup_runs = warmup_runs
    if runs_per_sample is not None:
        resolved_runs_per_sample = runs_per_sample

    tuning_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for execution_mode in selected_modes:
        LOGGER.info(
            "Starting %s finalization for %s",
            execution_mode,
            _case_descriptor(case),
        )
        (
            mode_tuning_rows,
            selected_candidate_ids,
            selected_config,
            selection_metadata,
            tuning_failure,
        ) = tune_mode(
            case,
            campaign_id=campaign_id,
            repeat_index=repeat_index,
            execution_mode=execution_mode,
            warmup_runs=resolved_warmup_runs,
            runs_per_sample=resolved_runs_per_sample,
            seed=seed,
            validate_long_seq_runlist=True,
            existing_tuning_rows=existing_tuning_rows,
        )
        tuning_rows.extend(mode_tuning_rows)

        selected_candidate_ids_json = json_dumps(selected_candidate_ids)
        selected_config_json = json_dumps(selected_config)
        if tuning_failure:
            result = _failed_final_result(
                power_backend=power_backend,
                failure_message=tuning_failure,
            )
            LOGGER.warning(
                "Skipping final %s benchmark for %s: %s",
                execution_mode,
                _case_descriptor(case),
                tuning_failure,
            )
        else:
            reused_row = reusable_final_row(
                {} if existing_final_rows is None else existing_final_rows,
                campaign_id=campaign_id,
                repeat_index=repeat_index,
                study_case_id=case.study_case_id,
                workload_variant=case.workload_variant,
                execution_mode=execution_mode,
                seq_len=case.seq_len,
                selected_candidate_ids_json=selected_candidate_ids_json,
                selected_config_json=selected_config_json,
                power_backend=power_backend,
            )
            if reused_row is not None:
                LOGGER.info(
                    "Reusing final %s benchmark for %s with selected operators %s",
                    execution_mode,
                    _case_descriptor(case),
                    selected_candidate_ids_json,
                )
                result = {
                    key: value
                    for key, value in reused_row.items()
                    if key not in RESULTS_CSV_FIELDNAMES
                }
                result.update(
                    {
                        key: reused_row.get(key, "")
                        for key in RESULTS_CSV_FIELDNAMES
                        if key
                        not in {
                            "study_id",
                            "campaign_id",
                            "repeat_index",
                            "matched_run_id",
                            "study_case_id",
                            "study_case_label",
                            "workload_variant",
                            "backend",
                            "execution_mode",
                            "pattern_label",
                            "seq_len",
                            "hidden_size",
                            "intermediate_size",
                            "num_attention_heads",
                            "attention_head_size",
                            "batch_size",
                            "dtype",
                            "use_bias",
                            "weights_source",
                            "joint_search_policy",
                            "joint_candidate_count_total",
                            "joint_candidates_evaluated",
                            "selected_joint_rank",
                            "selection_provenance",
                            "candidate_inventory_json",
                            "candidate_count_by_operator_json",
                            "warmup_runs",
                            "runs_per_sample",
                            "selected_candidate_ids_json",
                            "selected_config_json",
                            "is_best",
                        }
                    }
                )
            else:
                LOGGER.info(
                    "Running final %s benchmark for %s with selected operators %s",
                    execution_mode,
                    _case_descriptor(case),
                    selected_candidate_ids_json,
                )
                result = _paired_mode_result(
                    case,
                    execution_mode=execution_mode,
                    selected_config=selected_config,
                    warmup_runs=resolved_warmup_runs,
                    runs_per_sample=resolved_runs_per_sample,
                    seed=seed,
                    power_backend=power_backend,
                    repeat_index=repeat_index,
                )
                LOGGER.info(
                    "Completed final %s benchmark for %s -> %s",
                    execution_mode,
                    _case_descriptor(case),
                    _result_summary(result),
                )

        final_rows.append(
            {
                "study_id": "end_to_end",
                "campaign_id": normalize_campaign_id(campaign_id),
                "repeat_index": int(repeat_index),
                "matched_run_id": normalize_matched_run_id(
                    None,
                    campaign_id=campaign_id,
                    study_case_id=case.study_case_id,
                    execution_mode=execution_mode,
                    seq_len=case.seq_len,
                    repeat_index=repeat_index,
                ),
                "study_case_id": case.study_case_id,
                "study_case_label": case.study_case_label,
                "workload_variant": case.workload_variant,
                "backend": "npu",
                "execution_mode": execution_mode,
                "pattern_label": _pattern_label(execution_mode),
                "seq_len": case.seq_len,
                "hidden_size": case.hidden_size,
                "intermediate_size": case.intermediate_size,
                "num_attention_heads": case.num_attention_heads,
                "attention_head_size": case.attention_head_size,
                "batch_size": 1,
                "dtype": "bf16",
                "use_bias": False,
                "weights_source": "synthetic",
                "joint_search_policy": selection_metadata["joint_search_policy"],
                "joint_candidate_count_total": selection_metadata[
                    "joint_candidate_count_total"
                ],
                "joint_candidates_evaluated": selection_metadata[
                    "joint_candidates_evaluated"
                ],
                "selected_joint_rank": selection_metadata["selected_joint_rank"],
                "selection_provenance": selection_metadata["selection_provenance"],
                "candidate_inventory_json": selection_metadata[
                    "candidate_inventory_json"
                ],
                "candidate_count_by_operator_json": selection_metadata[
                    "candidate_count_by_operator_json"
                ],
                "warmup_runs": resolved_warmup_runs,
                "runs_per_sample": resolved_runs_per_sample,
                "selected_candidate_ids_json": selected_candidate_ids_json,
                "selected_config_json": selected_config_json,
                **result,
            }
        )

    return tuning_rows, final_rows


def write_rows(
    output_path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transformer_layer end-to-end study"
    )
    parser.add_argument(
        "--workload-variant",
        choices=[*WORKLOAD_VARIANTS, "all"],
        default="all",
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
    )
    parser.add_argument(
        "--seq-len",
        choices=[*(str(value) for value in SEQUENCE_LADDER), "all"],
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=[*EXECUTION_MODES, "all"],
        default="all",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument(
        "--power-backend",
        choices=list(SUPPORTED_POWER_BACKENDS),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument(
        "--tuning-output", type=Path, default=default_tuning_output_path()
    )
    parser.add_argument("--resume-input", type=Path, default=None)
    parser.add_argument("--resume-tuning-input", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    warn_if_npu_power_mode_not_turbo(LOGGER, study_name="end-to-end study")

    output_path = args.output.expanduser()
    tuning_output_path = args.tuning_output.expanduser()
    manifest_output_path = (
        output_path.with_name("campaign_manifest.json")
        if args.manifest_output is None
        else args.manifest_output.expanduser()
    )
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="transformer_layer end-to-end study",
    ):
        resume_output_paths: tuple[Path, ...] = tuple()
        if args.no_resume:
            resume_output_paths = tuple()
        elif args.resume_input is not None:
            resume_output_paths = (args.resume_input.expanduser(),)
        else:
            resume_output_paths = default_resume_paths(output_path)
        resume_output_paths = _compatible_resume_paths(
            resume_output_paths,
            campaign_id=str(args.campaign_id),
        )

        resume_tuning_paths: tuple[Path, ...] = tuple()
        if args.no_resume:
            resume_tuning_paths = tuple()
        elif args.resume_tuning_input is not None:
            resume_tuning_paths = (args.resume_tuning_input.expanduser(),)
        else:
            resume_tuning_paths = default_resume_tuning_paths(tuning_output_path)
        resume_tuning_paths = _compatible_resume_paths(
            resume_tuning_paths,
            campaign_id=str(args.campaign_id),
        )
        cases = tuple(iter_cases(args.workload_variant, args.family, args.seq_len))

        LOGGER.info(
            "Starting end-to-end study with %d case(s), mode=%s, power_backend=%s",
            len(cases),
            args.mode,
            args.power_backend,
        )

        existing_tuning_rows = load_existing_tuning_rows(resume_tuning_paths)
        existing_final_rows = load_existing_final_rows(resume_output_paths)
        if resume_tuning_paths and existing_tuning_rows:
            LOGGER.info(
                "Loaded %d reusable tuning rows from %s",
                len(existing_tuning_rows),
                ", ".join(str(path) for path in resume_tuning_paths),
            )
        if resume_output_paths and existing_final_rows:
            LOGGER.info(
                "Loaded %d reusable final rows from %s",
                len(existing_final_rows),
                ", ".join(str(path) for path in resume_output_paths),
            )

        tuning_row_map: dict[
            tuple[str, int, str, str, str, str, int, str], dict[str, object]
        ] = dict(existing_tuning_rows)
        final_row_map: dict[tuple[str, int, str, str, str, int], dict[str, object]] = (
            dict(existing_final_rows)
        )
        tuning_rows = list(tuning_row_map.values())
        final_rows = list(final_row_map.values())
        for case_index, case in enumerate(cases, start=1):
            LOGGER.info(
                "Case %d/%d: %s", case_index, len(cases), _case_descriptor(case)
            )
            case_tuning_rows, case_final_rows = build_rows(
                case,
                campaign_id=str(args.campaign_id),
                repeat_index=int(args.repeat_index),
                mode_filter=args.mode,
                warmup_runs=args.warmup_iters,
                runs_per_sample=args.timed_iters,
                seed=args.seed,
                power_backend=args.power_backend,
                existing_tuning_rows=existing_tuning_rows,
                existing_final_rows=existing_final_rows,
            )
            for row in case_tuning_rows:
                tuning_row_map[
                    _tuning_row_key({key: str(value) for key, value in row.items()})
                ] = row
            for row in case_final_rows:
                final_row_map[
                    _final_row_key({key: str(value) for key, value in row.items()})
                ] = row
            tuning_rows = list(tuning_row_map.values())
            final_rows = list(final_row_map.values())
            mark_best_rows(final_rows)
            write_rows(output_path, fieldnames=RESULTS_CSV_FIELDNAMES, rows=final_rows)
            write_rows(
                tuning_output_path,
                fieldnames=TUNING_CSV_FIELDNAMES,
                rows=tuning_rows,
            )
            LOGGER.info(
                "Checkpointed %d end-to-end rows and %d tuning rows",
                len(final_rows),
                len(tuning_rows),
            )

        mark_best_rows(final_rows)
        write_rows(output_path, fieldnames=RESULTS_CSV_FIELDNAMES, rows=final_rows)
        write_rows(
            tuning_output_path,
            fieldnames=TUNING_CSV_FIELDNAMES,
            rows=tuning_rows,
        )
        write_campaign_manifest(
            manifest_output_path,
            campaign_id=str(args.campaign_id),
            study_name=END_TO_END_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.end_to_end.run",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path, tuning_output_path),
            extra={
                "mode_filter": str(args.mode),
                "power_backend": str(args.power_backend),
                "repeat_index": int(args.repeat_index),
                "joint_search_policy": _joint_search_policy(top_k=JOINT_SEARCH_TOP_K),
                "joint_search_top_k": JOINT_SEARCH_TOP_K,
                "selection_provenance_policy": (
                    "rows declare whether the selected joint configuration is "
                    "exhaustive-best or heuristic-best"
                ),
                "validation_policy": validation_policy_manifest(),
                "candidate_policy_by_mode": candidate_policy_manifest(),
                "effective_candidate_inventory_by_case": {
                    f"{case.study_case_id}:seq{case.seq_len}": {
                        execution_mode: {
                            "candidate_ids_by_operator": json.loads(
                                _candidate_inventory_metadata(case)[execution_mode][
                                    "candidate_inventory_json"
                                ]
                            ),
                            "candidate_count_by_operator": json.loads(
                                _candidate_inventory_metadata(case)[execution_mode][
                                    "candidate_count_by_operator_json"
                                ]
                            ),
                        }
                        for execution_mode in EXECUTION_MODES
                    }
                    for case in cases
                },
                "study_case_ids": [case.study_case_id for case in cases],
                "workload_variants": sorted({case.workload_variant for case in cases}),
                "sequence_lengths": sorted({case.seq_len for case in cases}),
            },
        )
        LOGGER.info("Wrote %d end-to-end rows to %s", len(final_rows), output_path)
        LOGGER.info("Wrote %d tuning rows to %s", len(tuning_rows), tuning_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
