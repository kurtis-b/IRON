#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from ..npu_runtime_checks import warn_if_npu_power_mode_not_turbo
from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_SPECS,
    FAMILY_IDS,
    WORKLOAD_VARIANTS,
    EndToEndCase,
    get_case,
)
from .modes import benchmark_mode, benchmark_mode_subprocess
from .run import iteration_schedule
from .select import default_results_path, load_result_rows, select_result_rows

LOGGER = logging.getLogger(__name__)

SPOT_CHECK_SEQ_LENS: tuple[int, ...] = (512, 2048)
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "validation_mode",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "latency_sample_count",
    "min_latency_ms",
    "max_latency_ms",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "correctness_spot_checks.csv"
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


def _row_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        int(float(str(row.get("seq_len") or 0))),
    )


def load_existing_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, str, str, int], dict[str, object]]:
    rows: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("execution_mode") or ""):
                    continue
                rows[_row_key(row)] = dict(row)
    return rows


def merge_rows(
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]],
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged = {key: dict(value) for key, value in existing_rows.items()}
    for row in rows:
        merged[_row_key(row)] = dict(row)
    return [merged[key] for key in sorted(merged)]


def reusable_existing_row(
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]],
    *,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    seq_len: int,
    warmup_runs: int,
    runs_per_sample: int,
    selected_candidate_ids_json: str,
    selected_config_json: str,
) -> dict[str, object] | None:
    row = existing_rows.get((study_case_id, workload_variant, execution_mode, seq_len))
    if row is None:
        return None
    if str(row.get("run_status") or "") != "passed":
        return None
    if int(float(str(row.get("warmup_runs") or 0))) != int(warmup_runs):
        return None
    if int(float(str(row.get("runs_per_sample") or 0))) != int(runs_per_sample):
        return None
    if str(row.get("selected_candidate_ids_json") or "") != selected_candidate_ids_json:
        return None
    if str(row.get("selected_config_json") or "") != selected_config_json:
        return None
    required_fields = (
        "latency_sample_count",
        "min_latency_ms",
        "max_latency_ms",
    )
    if any(str(row.get(field) or "").strip() == "" for field in required_fields):
        return None
    return dict(row)


def _resolved_sampling(case: EndToEndCase, selected_row) -> tuple[int, int]:
    scheduled_warmup_runs, scheduled_runs_per_sample = iteration_schedule(case.seq_len)
    warmup_runs = (
        scheduled_warmup_runs
        if selected_row.warmup_runs in (None, 0)
        else int(selected_row.warmup_runs)
    )
    runs_per_sample = (
        scheduled_runs_per_sample
        if selected_row.runs_per_sample in (None, 0)
        else int(selected_row.runs_per_sample)
    )
    return warmup_runs, runs_per_sample


def validation_mode_for_seq_len(seq_len: int) -> str:
    if seq_len == 512:
        return "exact_reference"
    return "numerical_spot_check"


def _matches_seq_len_filter(seq_len: int, seq_len_filter: str) -> bool:
    return seq_len_filter == "all" or int(seq_len) == int(seq_len_filter)


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    mode_filter: str,
    seq_len_filter: str = "all",
    seed: int,
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]] | None = None,
    benchmark_fn=None,
) -> list[dict[str, object]]:
    if benchmark_fn is None:
        benchmark_fn = benchmark_mode
    selected_rows = [
        row
        for row in select_result_rows(
            load_result_rows(results_input),
            workload_variant_filter=workload_variant_filter,
            family_filter=family_filter,
            mode_filter=mode_filter,
        )
        if row.seq_len in SPOT_CHECK_SEQ_LENS
        and _matches_seq_len_filter(row.seq_len, seq_len_filter)
    ]

    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        warmup_runs, runs_per_sample = _resolved_sampling(case, selected_row)
        selected_candidate_ids_json = json.dumps(
            selected_row.selected_candidate_ids,
            sort_keys=True,
        )
        selected_config_json = json.dumps(
            selected_row.selected_config,
            sort_keys=True,
        )
        reused_row = reusable_existing_row(
            {} if existing_rows is None else existing_rows,
            study_case_id=selected_row.study_case_id,
            workload_variant=selected_row.workload_variant,
            execution_mode=selected_row.execution_mode,
            seq_len=selected_row.seq_len,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            selected_candidate_ids_json=selected_candidate_ids_json,
            selected_config_json=selected_config_json,
        )
        if reused_row is not None:
            rows.append(
                {field: reused_row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )
            continue
        LOGGER.info(
            "Running correctness spot check for %s seq_len=%s mode=%s",
            selected_row.study_case_id,
            selected_row.seq_len,
            selected_row.execution_mode,
        )
        result = benchmark_fn(
            selected_row.execution_mode,
            case.workload,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
            power_backend="none",
            operator_config=selected_row.selected_config,
            include_reference_output=True,
        )
        rows.append(
            {
                "study_id": "end_to_end_correctness_spot_checks",
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "validation_mode": validation_mode_for_seq_len(selected_row.seq_len),
                "seq_len": selected_row.seq_len,
                "hidden_size": selected_row.hidden_size,
                "intermediate_size": selected_row.intermediate_size,
                "num_attention_heads": selected_row.num_attention_heads,
                "attention_head_size": selected_row.attention_head_size,
                "warmup_runs": warmup_runs,
                "runs_per_sample": runs_per_sample,
                "avg_latency_ms": result.get("avg_latency_ms"),
                "latency_sample_count": result.get("latency_sample_count"),
                "min_latency_ms": result.get("min_latency_ms"),
                "max_latency_ms": result.get("max_latency_ms"),
                "validation_error_count": result.get("validation_error_count", ""),
                "run_status": result.get("run_status", ""),
                "failure_message": result.get("failure_message", ""),
                "selected_candidate_ids_json": selected_candidate_ids_json,
                "selected_config_json": selected_config_json,
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
        description="Run exact correctness spot checks for selected end-to-end NPU configs."
    )
    parser.add_argument(
        "--results-input",
        type=Path,
        default=default_results_path(),
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
        "--mode",
        choices=[*EXECUTION_MODES, "all"],
        default="all",
    )
    parser.add_argument(
        "--seq-len",
        choices=[*(str(value) for value in SPOT_CHECK_SEQ_LENS), "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--resume-input", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    warn_if_npu_power_mode_not_turbo(
        LOGGER,
        study_name="end-to-end correctness spot checks",
    )
    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="end-to-end correctness spot checks",
    ):
        resume_paths: tuple[Path, ...] = tuple()
        if not args.no_resume:
            if args.resume_input is not None:
                resume_paths = (args.resume_input.expanduser(),)
            else:
                resume_paths = default_resume_paths(output_path)
        existing_rows = load_existing_rows(resume_paths)
        if resume_paths and existing_rows:
            LOGGER.info(
                "Reusing %d correctness spot-check rows from %s",
                len(existing_rows),
                ", ".join(str(path) for path in resume_paths),
            )
        rows = build_rows(
            results_input=args.results_input.expanduser(),
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            mode_filter=str(args.mode),
            seq_len_filter=str(args.seq_len),
            seed=int(args.seed),
            existing_rows=existing_rows,
            benchmark_fn=benchmark_mode_subprocess,
        )
        merged_rows = merge_rows(existing_rows, rows)
        write_rows(output_path, merged_rows)
        LOGGER.info(
            "Wrote %d correctness spot-check rows to %s",
            len(merged_rows),
            output_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
