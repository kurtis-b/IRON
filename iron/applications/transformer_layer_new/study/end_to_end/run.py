#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    MODE_OPERATORS,
    SEQUENCE_LADDER,
    EndToEndCase,
    candidate_table_for_case,
    iter_cases,
)
from .modes import (
    benchmark_mode,
    benchmark_operator_candidate,
    resolve_mode_operator_config,
)
from .power import SUPPORTED_POWER_BACKENDS

LOGGER = logging.getLogger(__name__)

TUNING_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "execution_mode",
    "internal_operator",
    "candidate_id",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "bandwidth_gbps",
    "validation_error_count",
    "run_status",
    "failure_message",
    "operator_config_json",
    "is_operator_best",
)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
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
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
    "compile_setup_time_ms",
    "host_qkv_precompute_ms",
    "tokens_per_sec",
    "power_backend",
    "avg_power_w",
    "tokens_per_sec_per_watt",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "process_model",
    "validation_error_count",
    "run_status",
    "failure_message",
    "selected_candidate_ids_json",
    "selected_config_json",
    "is_best",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "results" / "end_to_end" / "results.csv"
    )


def default_tuning_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "end_to_end" / "tuning.csv"


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 128:
        return (10, 100)
    if seq_len <= 512:
        return (5, 50)
    if seq_len <= 2048:
        return (3, 20)
    return (1, 10)


def json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    best_index_by_group: dict[tuple[str, int], int] = {}
    best_latency_by_group: dict[tuple[str, int], float] = {}

    for index, row in enumerate(rows):
        row["is_best"] = False
        if row["run_status"] != "passed" or row["avg_latency_ms"] in ("", None):
            continue
        group = (str(row["study_case_id"]), int(row["seq_len"]))
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
    return {
        "measured_inference_count": 0,
        "timed_total_sec": 0.0,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "host_qkv_precompute_ms": None,
        "tokens_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "tokens_per_sec_per_watt": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }


def tune_mode(
    case: EndToEndCase,
    *,
    execution_mode: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, dict[str, object]], str]:
    candidates_by_mode = candidate_table_for_case(case.study_case_id, case.seq_len)
    tuning_rows: list[dict[str, object]] = []
    selected_candidate_ids: dict[str, str] = {}
    selected_config: dict[str, dict[str, object]] = {}

    for operator_name in MODE_OPERATORS[execution_mode]:
        operator_rows: list[dict[str, object]] = []
        for candidate in candidates_by_mode[execution_mode][operator_name]:
            resolved_config = resolve_mode_operator_config(
                execution_mode,
                case.workload,
                {operator_name: dict(candidate["config"])},
            )[operator_name]
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
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "execution_mode": execution_mode,
                    "internal_operator": operator_name,
                    "candidate_id": candidate["candidate_id"],
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

        best_row = None
        successful_rows = [
            row
            for row in operator_rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] not in ("", None)
        ]
        if successful_rows:
            best_row = min(
                successful_rows, key=lambda row: float(row["avg_latency_ms"])
            )
            best_row["is_operator_best"] = True
            selected_candidate_ids[operator_name] = str(best_row["candidate_id"])
            selected_config[operator_name] = dict(best_row["_resolved_config"])

        tuning_rows.extend(operator_rows)
        if best_row is None:
            return (
                tuning_rows,
                selected_candidate_ids,
                selected_config,
                f"tuning_failed: no passing candidate for {operator_name}",
            )

    return tuning_rows, selected_candidate_ids, selected_config, ""


def build_rows(
    case: EndToEndCase,
    *,
    mode_filter: str,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    power_backend: str,
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
        mode_tuning_rows, selected_candidate_ids, selected_config, tuning_failure = (
            tune_mode(
                case,
                execution_mode=execution_mode,
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
                seed=seed,
            )
        )
        tuning_rows.extend(mode_tuning_rows)

        if tuning_failure:
            result = _failed_final_result(
                power_backend=power_backend,
                failure_message=tuning_failure,
            )
        else:
            result = benchmark_mode(
                execution_mode,
                case.workload,
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
                seed=seed,
                power_backend=power_backend,
                operator_config=selected_config,
            )

        final_rows.append(
            {
                "study_id": "end_to_end",
                "study_case_id": case.study_case_id,
                "study_case_label": case.study_case_label,
                "backend": "npu",
                "execution_mode": execution_mode,
                "pattern_label": execution_mode,
                "seq_len": case.seq_len,
                "hidden_size": case.hidden_size,
                "intermediate_size": case.intermediate_size,
                "num_attention_heads": case.num_attention_heads,
                "attention_head_size": case.attention_head_size,
                "batch_size": 1,
                "dtype": "bf16",
                "use_bias": False,
                "weights_source": "synthetic",
                "warmup_runs": resolved_warmup_runs,
                "runs_per_sample": resolved_runs_per_sample,
                "selected_candidate_ids_json": json_dumps(selected_candidate_ids),
                "selected_config_json": json_dumps(selected_config),
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
        description="Benchmark transformer_layer_new end-to-end study"
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
    parser.add_argument(
        "--power-backend",
        choices=list(SUPPORTED_POWER_BACKENDS),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument(
        "--tuning-output", type=Path, default=default_tuning_output_path()
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    tuning_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for case in iter_cases(args.family, args.seq_len):
        case_tuning_rows, case_final_rows = build_rows(
            case,
            mode_filter=args.mode,
            warmup_runs=args.warmup_iters,
            runs_per_sample=args.timed_iters,
            seed=args.seed,
            power_backend=args.power_backend,
        )
        tuning_rows.extend(case_tuning_rows)
        final_rows.extend(case_final_rows)

    mark_best_rows(final_rows)
    output_path = args.output.expanduser()
    tuning_output_path = args.tuning_output.expanduser()
    write_rows(output_path, fieldnames=RESULTS_CSV_FIELDNAMES, rows=final_rows)
    write_rows(
        tuning_output_path,
        fieldnames=TUNING_CSV_FIELDNAMES,
        rows=tuning_rows,
    )
    LOGGER.info("Wrote %d end-to-end rows to %s", len(final_rows), output_path)
    LOGGER.info("Wrote %d tuning rows to %s", len(tuning_rows), tuning_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
