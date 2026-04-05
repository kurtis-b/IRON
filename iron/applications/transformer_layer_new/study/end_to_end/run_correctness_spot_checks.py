#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .cases import EXECUTION_MODES, FAMILY_IDS, EndToEndCase, get_case
from .modes import benchmark_mode
from .run import iteration_schedule
from .select import default_results_path, load_result_rows, select_result_rows

LOGGER = logging.getLogger(__name__)

SPOT_CHECK_SEQ_LENS: tuple[int, ...] = (512, 2048)
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
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


def build_rows(
    *,
    results_input: Path,
    family_filter: str,
    mode_filter: str,
    seed: int,
) -> list[dict[str, object]]:
    selected_rows = [
        row
        for row in select_result_rows(
            load_result_rows(results_input),
            family_filter=family_filter,
            mode_filter=mode_filter,
        )
        if row.seq_len in SPOT_CHECK_SEQ_LENS
    ]

    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        warmup_runs, runs_per_sample = _resolved_sampling(case, selected_row)
        LOGGER.info(
            "Running correctness spot check for %s seq_len=%s mode=%s",
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
        )
        rows.append(
            {
                "study_id": "end_to_end_correctness_spot_checks",
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
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
                "validation_error_count": result.get("validation_error_count", ""),
                "run_status": result.get("run_status", ""),
                "failure_message": result.get("failure_message", ""),
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
        description="Run exact correctness spot checks for selected end-to-end NPU configs."
    )
    parser.add_argument(
        "--results-input",
        type=Path,
        default=default_results_path(),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    rows = build_rows(
        results_input=args.results_input.expanduser(),
        family_filter=str(args.family),
        mode_filter=str(args.mode),
        seed=int(args.seed),
    )
    write_rows(args.output.expanduser(), rows)
    LOGGER.info("Wrote %d correctness spot-check rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
