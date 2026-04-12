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

from ..npu_runtime_checks import warn_if_npu_power_mode_not_turbo
from ..run_lock import default_lock_path, hold_study_lock
from .cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    SEQUENCE_LADDER,
    WORKLOAD_VARIANTS,
    EndToEndCase,
    candidate_table_for_case,
    iter_cases,
    mode_operators,
)
from .modes import (
    benchmark_mode,
    benchmark_operator_candidate,
    resolve_mode_operator_config,
)
from .power import SUPPORTED_POWER_BACKENDS

LOGGER = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


def _pattern_label(execution_mode: str) -> str:
    return "Hybrid Runlist+Dataflow" if execution_mode == "hybrid" else "Runlist"


TUNING_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
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
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
    "compile_setup_time_ms",
    "host_qkv_precompute_ms",
    "effective_gflops_per_sec",
    "power_backend",
    "avg_power_w",
    "effective_gflops_per_sec_per_watt",
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
        Path(__file__).resolve().parents[2] / "results" / "end_to_end" / "results.csv"
    )


def default_tuning_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "end_to_end" / "tuning.csv"


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


def _resume_execution_mode(value: object) -> str:
    execution_mode = str(value or "")
    return "hybrid" if execution_mode == "dataflow" else execution_mode


def _resume_workload_variant(row: dict[str, str]) -> str:
    workload_variant = str(row.get("workload_variant") or "")
    if workload_variant:
        return workload_variant
    study_case_id = str(row.get("study_case_id") or "")
    if study_case_id in FAMILY_SPECS:
        return FAMILY_SPECS[study_case_id].workload_variant
    return ""


def _tuning_row_key(row: dict[str, str]) -> tuple[str, str, str, str, int, str]:
    return (
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        str(row.get("internal_operator") or ""),
        int(float(str(row.get("seq_len") or 0))),
        str(row.get("candidate_id") or ""),
    )


def load_existing_tuning_rows(
    paths: tuple[Path, ...],
) -> dict[tuple[str, str, str, str, int, str], dict[str, object]]:
    rows: dict[tuple[str, str, str, str, int, str], dict[str, object]] = {}
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
    existing_rows: dict[tuple[str, str, str, str, int, str], dict[str, object]],
    *,
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
    if str(row.get("run_status") or "") not in (
        "passed",
        "skipped_singleton_default",
        "skipped_long_seq_default",
    ):
        return None
    return dict(row)


def _final_row_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (
        str(row.get("study_case_id") or ""),
        _resume_workload_variant(row),
        _resume_execution_mode(row.get("execution_mode")),
        int(float(str(row.get("seq_len") or 0))),
    )


def load_existing_final_rows(
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
                rows[_final_row_key(row)] = dict(row)
    return rows


def reusable_final_row(
    existing_rows: dict[tuple[str, str, str, int], dict[str, object]],
    *,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    seq_len: int,
    selected_candidate_ids_json: str,
    selected_config_json: str,
    power_backend: str,
) -> dict[str, object] | None:
    row = existing_rows.get((study_case_id, workload_variant, execution_mode, seq_len))
    if row is None:
        return None
    if str(row.get("run_status") or "") != "passed":
        return None
    if str(row.get("selected_candidate_ids_json") or "") != selected_candidate_ids_json:
        return None
    if str(row.get("selected_config_json") or "") != selected_config_json:
        return None
    if power_backend != "auto" and str(row.get("power_backend") or "") != power_backend:
        return None
    return dict(row)


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 256:
        return (1, 100)
    if seq_len <= 2048:
        return (1, 10)
    if seq_len <= 4096:
        return (1, 5)
    return (1, 2)


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
        "effective_gflops_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "effective_gflops_per_sec_per_watt": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }


def _skip_isolated_singleton_benchmark(
    case: EndToEndCase,
    *,
    execution_mode: str,
    operator_name: str,
    candidates: list[dict[str, object]],
) -> bool:
    return execution_mode != "hybrid" and len(candidates) == 1


def _selected_default_row(
    *,
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
        "study_case_id": case.study_case_id,
        "study_case_label": case.study_case_label,
        "workload_variant": case.workload_variant,
        "execution_mode": execution_mode,
        "internal_operator": operator_name,
        "candidate_id": candidate_id,
        "seq_len": case.seq_len,
        "hidden_size": case.hidden_size,
        "intermediate_size": case.intermediate_size,
        "num_attention_heads": case.num_attention_heads,
        "attention_head_size": case.attention_head_size,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "avg_latency_ms": "",
        "bandwidth_gbps": "",
        "validation_error_count": "",
        "run_status": run_status,
        "failure_message": failure_message,
        "operator_config_json": json_dumps(resolved_config),
        "is_operator_best": True,
        "_resolved_config": resolved_config,
    }


def tune_mode(
    case: EndToEndCase,
    *,
    execution_mode: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    validate_long_seq_runlist: bool = True,
    existing_tuning_rows: (
        dict[tuple[str, str, str, str, int, str], dict[str, object]] | None
    ) = None,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, dict[str, object]], str]:
    candidates_by_mode = candidate_table_for_case(case.study_case_id, case.seq_len)
    tuning_rows: list[dict[str, object]] = []
    selected_candidate_ids: dict[str, str] = {}
    selected_config: dict[str, dict[str, object]] = {}
    operator_names = mode_operators(execution_mode, case.workload_variant)

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
        if _skip_isolated_singleton_benchmark(
            case,
            execution_mode=execution_mode,
            operator_name=operator_name,
            candidates=candidates,
        ):
            candidate = candidates[0]
            resolved_config = resolve_mode_operator_config(
                execution_mode,
                case.workload,
                {operator_name: dict(candidate["config"])},
            )[operator_name]
            selected_candidate_ids[operator_name] = candidate["candidate_id"]
            selected_config[operator_name] = dict(resolved_config)
            run_status = "skipped_singleton_default"
            failure_message = (
                "Selected without isolated benchmark because only one candidate "
                "remained"
            )
            if execution_mode == "runlist" and case.seq_len >= 8192:
                run_status = "skipped_long_seq_default"
                failure_message = (
                    "Selected without isolated benchmark during long-sequence "
                    "runlist tuning because only one candidate remained"
                )
            tuning_rows.append(
                reusable_tuning_row(
                    {} if existing_tuning_rows is None else existing_tuning_rows,
                    study_case_id=case.study_case_id,
                    workload_variant=case.workload_variant,
                    execution_mode=execution_mode,
                    operator_name=operator_name,
                    seq_len=case.seq_len,
                    candidate_id=candidate["candidate_id"],
                    resolved_config_json=json_dumps(resolved_config),
                )
                or _selected_default_row(
                    case=case,
                    execution_mode=execution_mode,
                    operator_name=operator_name,
                    candidate_id=candidate["candidate_id"],
                    resolved_config=resolved_config,
                    warmup_runs=warmup_runs,
                    runs_per_sample=runs_per_sample,
                    run_status=run_status,
                    failure_message=failure_message,
                )
            )
            LOGGER.info(
                "[%s] Selected singleton %s candidate %s without isolated benchmarking",
                execution_mode,
                operator_name,
                candidate["candidate_id"],
            )
            continue
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
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "workload_variant": case.workload_variant,
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
            LOGGER.info(
                "[%s] %s candidate %s -> %s",
                execution_mode,
                operator_name,
                candidate["candidate_id"],
                _result_summary(result),
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
            LOGGER.info(
                "[%s] Selected %s candidate %s (avg_latency_ms=%.4f)",
                execution_mode,
                operator_name,
                best_row["candidate_id"],
                float(best_row["avg_latency_ms"]),
            )

        tuning_rows.extend(operator_rows)
        if best_row is None:
            LOGGER.warning(
                "[%s] No passing candidate for %s in %s",
                execution_mode,
                operator_name,
                _case_descriptor(case),
            )
            return (
                tuning_rows,
                selected_candidate_ids,
                selected_config,
                f"tuning_failed: no passing candidate for {operator_name}",
            )

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
                "tuning_failed: long-sequence runlist mode smoke failed: "
                + str(smoke_result["failure_message"]),
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
    existing_tuning_rows: (
        dict[tuple[str, str, str, str, int, str], dict[str, object]] | None
    ) = None,
    existing_final_rows: (
        dict[tuple[str, str, str, int], dict[str, object]] | None
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
        mode_tuning_rows, selected_candidate_ids, selected_config, tuning_failure = (
            tune_mode(
                case,
                execution_mode=execution_mode,
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
                seed=seed,
                validate_long_seq_runlist=False,
                existing_tuning_rows=existing_tuning_rows,
            )
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
                result = benchmark_mode(
                    execution_mode,
                    case.workload,
                    warmup_runs=resolved_warmup_runs,
                    runs_per_sample=resolved_runs_per_sample,
                    seed=seed,
                    power_backend=power_backend,
                    operator_config=selected_config,
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
        description="Benchmark transformer_layer_new end-to-end study"
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
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="transformer_layer_new end-to-end study",
    ):
        resume_output_paths: tuple[Path, ...] = tuple()
        if args.no_resume:
            resume_output_paths = tuple()
        elif args.resume_input is not None:
            resume_output_paths = (args.resume_input.expanduser(),)
        else:
            resume_output_paths = default_resume_paths(output_path)

        resume_tuning_paths: tuple[Path, ...] = tuple()
        if args.no_resume:
            resume_tuning_paths = tuple()
        elif args.resume_tuning_input is not None:
            resume_tuning_paths = (args.resume_tuning_input.expanduser(),)
        else:
            resume_tuning_paths = default_resume_tuning_paths(tuning_output_path)
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

        tuning_row_map: dict[tuple[str, str, str, str, int, str], dict[str, object]] = (
            dict(existing_tuning_rows)
        )
        final_row_map: dict[tuple[str, str, str, int], dict[str, object]] = dict(
            existing_final_rows
        )
        tuning_rows = list(tuning_row_map.values())
        final_rows = list(final_row_map.values())
        for case_index, case in enumerate(cases, start=1):
            LOGGER.info(
                "Case %d/%d: %s", case_index, len(cases), _case_descriptor(case)
            )
            case_tuning_rows, case_final_rows = build_rows(
                case,
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
        LOGGER.info("Wrote %d end-to-end rows to %s", len(final_rows), output_path)
        LOGGER.info("Wrote %d tuning rows to %s", len(tuning_rows), tuning_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
