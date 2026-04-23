#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from ..campaign import write_campaign_manifest
from ..run_lock import default_lock_path, hold_study_lock
from ..end_to_end.cases import get_case, mode_operators
from ..end_to_end.run import (
    _sorted_successful_operator_rows,
    _validate_joint_mode_candidates,
    default_output_path as default_end_to_end_results_path,
    default_tuning_output_path,
)
from ..end_to_end.select import load_result_rows, select_result_rows
from .failure_taxonomy import build_taxonomy, write_rows as write_taxonomy_rows

LOGGER = logging.getLogger(__name__)
SEARCH_VALIDATION_STUDY_NAME = "transformer_layer search validation"
DEFAULT_FAMILY_IDS = ("baseline_768", "gpt2_small_768")
DEFAULT_SEQ_LENS = (256, 2048, 8192)
DEFAULT_EXECUTION_MODES = ("hybrid", "runlist", "offload")
EXHAUSTIVE_MAX_COMBINATIONS = 128
HEURISTIC_TOP_K = 16
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "matched_run_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "validation_scope",
    "joint_search_policy",
    "joint_candidate_count_total",
    "joint_candidates_evaluated",
    "selected_joint_rank",
    "selection_provenance",
    "selected_matches_baseline",
    "avg_latency_ms",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
    "baseline_selected_candidate_ids_json",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "search_validation.csv"
    )


def default_manifest_output_path(output_path: Path) -> Path:
    return output_path.with_name("search_validation_manifest.json")


def default_taxonomy_output_path(output_path: Path) -> Path:
    return output_path.with_name("search_validation_failure_taxonomy.csv")


def _load_matching_tuning_rows(
    tuning_rows: list[dict[str, str]],
    *,
    selected_row,
) -> dict[str, list[dict[str, object]]]:
    operator_rows: dict[str, list[dict[str, object]]] = {
        operator_name: []
        for operator_name in mode_operators(
            selected_row.execution_mode,
            selected_row.workload_variant,
        )
    }
    for row in tuning_rows:
        if str(row.get("campaign_id") or "") != selected_row.campaign_id:
            continue
        if int(float(str(row.get("repeat_index") or 0))) != selected_row.repeat_index:
            continue
        if str(row.get("study_case_id") or "") != selected_row.study_case_id:
            continue
        if str(row.get("workload_variant") or "") != selected_row.workload_variant:
            continue
        if str(row.get("execution_mode") or "") != selected_row.execution_mode:
            continue
        if int(float(str(row.get("seq_len") or 0))) != selected_row.seq_len:
            continue
        operator_name = str(row.get("internal_operator") or "")
        if operator_name not in operator_rows:
            continue
        if str(row.get("selection_stage") or "") != "isolated_operator":
            continue
        if str(row.get("run_status") or "") != "passed":
            continue
        operator_rows[operator_name].append(
            {
                **row,
                "_resolved_config": json.loads(str(row.get("operator_config_json") or "{}")),
            }
        )
    return {
        operator_name: _sorted_successful_operator_rows(rows)
        for operator_name, rows in operator_rows.items()
    }


def build_rows(
    *,
    results_input: Path,
    tuning_input: Path,
    seed: int,
    warmup_runs_override: int | None = None,
    runs_per_sample_override: int | None = None,
) -> list[dict[str, object]]:
    selected_rows = [
        row
        for row in select_result_rows(load_result_rows(results_input))
        if row.study_case_id in DEFAULT_FAMILY_IDS
        and row.seq_len in DEFAULT_SEQ_LENS
        and row.execution_mode in DEFAULT_EXECUTION_MODES
    ]
    tuning_rows = load_result_rows(tuning_input)
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        successful_rows_by_operator = _load_matching_tuning_rows(
            tuning_rows,
            selected_row=selected_row,
        )
        joint_candidate_count_total = 1
        for operator_name in mode_operators(
            selected_row.execution_mode,
            selected_row.workload_variant,
        ):
            joint_candidate_count_total *= len(
                successful_rows_by_operator.get(operator_name, ())
            )
        exhaustive = joint_candidate_count_total <= EXHAUSTIVE_MAX_COMBINATIONS
        top_k = (
            joint_candidate_count_total
            if exhaustive
            else min(HEURISTIC_TOP_K, joint_candidate_count_total)
        )
        warmup_runs = max(
            1,
            int(
                (selected_row.warmup_runs or 1)
                if warmup_runs_override is None
                else warmup_runs_override
            ),
        )
        runs_per_sample = max(
            1,
            int(
                (selected_row.runs_per_sample or 3)
                if runs_per_sample_override is None
                else runs_per_sample_override
            ),
        )
        joint_rows, best_candidate_ids, best_config, search_metadata, failure_message = (
            _validate_joint_mode_candidates(
                case,
                campaign_id=selected_row.campaign_id,
                repeat_index=selected_row.repeat_index,
                execution_mode=selected_row.execution_mode,
                operator_order=mode_operators(
                    selected_row.execution_mode,
                    selected_row.workload_variant,
                ),
                successful_rows_by_operator=successful_rows_by_operator,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                top_k=max(1, top_k),
            )
        )
        best_joint_row = None
        for joint_row in joint_rows:
            if str(joint_row.get("run_status") or "") != "passed":
                continue
            if best_joint_row is None or float(joint_row["avg_latency_ms"]) < float(
                best_joint_row["avg_latency_ms"]
            ):
                best_joint_row = joint_row
        rows.append(
            {
                "study_id": "search_validation",
                "campaign_id": selected_row.campaign_id,
                "repeat_index": selected_row.repeat_index,
                "matched_run_id": selected_row.matched_run_id,
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "validation_scope": (
                    "exhaustive_joint_validation"
                    if exhaustive
                    else "widened_top16_joint_validation"
                ),
                "joint_search_policy": search_metadata.get("joint_search_policy"),
                "joint_candidate_count_total": search_metadata.get(
                    "joint_candidate_count_total"
                ),
                "joint_candidates_evaluated": search_metadata.get(
                    "joint_candidates_evaluated"
                ),
                "selected_joint_rank": search_metadata.get("selected_joint_rank"),
                "selection_provenance": search_metadata.get("selection_provenance"),
                "selected_matches_baseline": (
                    best_candidate_ids == selected_row.selected_candidate_ids
                ),
                "avg_latency_ms": (
                    None if best_joint_row is None else best_joint_row.get("avg_latency_ms")
                ),
                "validation_error_count": (
                    "" if best_joint_row is None else best_joint_row.get("validation_error_count", "")
                ),
                "run_status": (
                    "failed_exception"
                    if best_joint_row is None
                    else best_joint_row.get("run_status")
                ),
                "failure_message": (
                    failure_message
                    if best_joint_row is None
                    else best_joint_row.get("failure_message", "")
                ),
                "selected_candidate_ids_json": json.dumps(
                    best_candidate_ids,
                    sort_keys=True,
                ),
                "selected_config_json": json.dumps(best_config, sort_keys=True),
                "baseline_selected_candidate_ids_json": json.dumps(
                    selected_row.selected_candidate_ids,
                    sort_keys=True,
                ),
            }
        )
    return rows


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the published search policy against exhaustive or widened "
            "joint-search re-evaluation on a representative subset."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument("--results-input", type=Path, default=default_end_to_end_results_path())
    parser.add_argument("--tuning-input", type=Path, default=default_tuning_output_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--taxonomy-output", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    output_path = args.output.expanduser()
    taxonomy_output = (
        default_taxonomy_output_path(output_path)
        if args.taxonomy_output is None
        else args.taxonomy_output.expanduser()
    )
    manifest_output = (
        default_manifest_output_path(output_path)
        if args.manifest_output is None
        else args.manifest_output.expanduser()
    )
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="search validation",
    ):
        rows = build_rows(
            results_input=args.results_input.expanduser(),
            tuning_input=args.tuning_input.expanduser(),
            seed=int(args.seed),
            warmup_runs_override=args.warmup_runs,
            runs_per_sample_override=args.runs_per_sample,
        )
        write_rows(output_path, rows)
        write_taxonomy_rows(taxonomy_output, build_taxonomy(rows))
        campaign_id = str(rows[0]["campaign_id"]) if rows else "canonical"
        write_campaign_manifest(
            manifest_output,
            campaign_id=campaign_id,
            study_name=SEARCH_VALIDATION_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.search_validation.run",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path, taxonomy_output),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
