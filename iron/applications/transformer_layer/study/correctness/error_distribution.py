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
from ..end_to_end.cases import get_case
from ..end_to_end.modes import benchmark_mode
from ..end_to_end.select import default_results_path, load_result_rows, select_result_rows
from ..end_to_end.validation import FINAL_ABS_TOL, FINAL_REL_TOL
from .common import error_summary

LOGGER = logging.getLogger(__name__)
ERROR_DISTRIBUTION_STUDY_NAME = "transformer_layer correctness error distribution"
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
    "avg_latency_ms",
    "validation_error_count",
    "run_status",
    "failure_message",
    "abs_error_mean",
    "abs_error_p50",
    "abs_error_p90",
    "abs_error_p95",
    "abs_error_p99",
    "abs_error_max",
    "rel_error_mean",
    "rel_error_p95",
    "rel_error_max",
    "abs_threshold_fraction",
    "rel_threshold_fraction",
    "abs_histogram_edges_json",
    "abs_histogram_counts_json",
    "selected_candidate_ids_json",
    "selected_config_json",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "error_distribution.csv"
    )


def default_manifest_output_path(output_path: Path) -> Path:
    return output_path.with_name("error_distribution_manifest.json")


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
    mode_filter: str,
    seed: int,
    warmup_runs_override: int | None = None,
    runs_per_sample_override: int | None = None,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        load_result_rows(results_input),
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter=mode_filter,
    )
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
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
            (
                min(3, int(selected_row.runs_per_sample or 1))
                if runs_per_sample_override is None
                else int(runs_per_sample_override)
            ),
        )
        LOGGER.info(
            "Running correctness error distribution for %s seq_len=%s mode=%s",
            selected_row.study_case_id,
            selected_row.seq_len,
            selected_row.execution_mode,
        )
        result = benchmark_mode(
            selected_row.execution_mode,
            case.workload,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
            power_backend="none",
            operator_config=selected_row.selected_config,
            include_reference_output=True,
            capture_output_tensors=True,
            scope_key_override=(
                f"error_distribution_{selected_row.study_case_id}_"
                f"{selected_row.execution_mode}_{selected_row.seq_len}"
            ),
        )
        summary: dict[str, object] = {}
        output_tensor = result.get("output_tensor")
        reference_output_tensor = result.get("reference_output_tensor")
        if result.get("run_status") == "passed" and output_tensor is not None and reference_output_tensor is not None:
            summary = error_summary(
                output_tensor,
                reference_output_tensor,
                abs_tol=FINAL_ABS_TOL,
                rel_tol=FINAL_REL_TOL,
            )
        rows.append(
            {
                "study_id": "correctness_error_distribution",
                "campaign_id": selected_row.campaign_id,
                "repeat_index": selected_row.repeat_index,
                "matched_run_id": selected_row.matched_run_id,
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "avg_latency_ms": result.get("avg_latency_ms"),
                "validation_error_count": result.get("validation_error_count"),
                "run_status": result.get("run_status"),
                "failure_message": result.get("failure_message"),
                **summary,
                "selected_candidate_ids_json": json.dumps(
                    selected_row.selected_candidate_ids,
                    sort_keys=True,
                ),
                "selected_config_json": json.dumps(
                    selected_row.selected_config,
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
        description="Generate synthetic error-distribution summaries for selected end-to-end rows."
    )
    parser.add_argument("--workload-variant", default="all")
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument("--results-input", type=Path, default=default_results_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    output_path = args.output.expanduser()
    manifest_output = (
        default_manifest_output_path(output_path)
        if args.manifest_output is None
        else args.manifest_output.expanduser()
    )
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="correctness error distribution",
    ):
        rows = build_rows(
            results_input=args.results_input.expanduser(),
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            seq_len_filter=str(args.seq_len),
            mode_filter=str(args.mode),
            seed=int(args.seed),
            warmup_runs_override=args.warmup_runs,
            runs_per_sample_override=args.runs_per_sample,
        )
        write_rows(output_path, rows)
        campaign_id = str(rows[0]["campaign_id"]) if rows else "canonical"
        write_campaign_manifest(
            manifest_output,
            campaign_id=campaign_id,
            study_name=ERROR_DISTRIBUTION_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.correctness.error_distribution",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path,),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
