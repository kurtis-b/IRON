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
    SEQUENCE_LADDER,
    ReconfigurationWorkload,
)
from .modes import benchmark_mode, resolve_mode_operator_config
from .power import SUPPORTED_POWER_BACKENDS
from .select import (
    default_reference_results_path,
    load_reference_rows,
    select_reference_rows,
)

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "backend",
    "execution_mode",
    "pattern_label",
    "source_end_to_end_execution_mode",
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
    "power_backend",
    "avg_power_w",
    "max_power_w",
    "energy_j",
    "power_sample_count",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "process_model",
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
        / "reconfiguration_overhead"
        / "results.csv"
    )


def iteration_schedule(seq_len: int) -> tuple[int, int]:
    if seq_len <= 2048:
        return (1, 10)
    if seq_len <= 4096:
        return (1, 5)
    return (1, 2)


def json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )


def _case_descriptor(row: dict[str, object]) -> str:
    return (
        f"{row['study_case_id']} seq_len={row['seq_len']} "
        f"mode={row['execution_mode']}"
    )


def _failed_result(
    *,
    power_backend: str,
    failure_message: str,
) -> dict[str, object]:
    return {
        "measured_inference_count": 0,
        "timed_total_sec": 0.0,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "power_sample_count": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }


def build_rows(
    *,
    family_filter: str,
    seq_len_filter: str,
    mode_filter: str,
    warmup_runs: int | None,
    runs_per_sample: int | None,
    seed: int,
    power_backend: str,
    power_sample_interval_sec: float,
    reference_input: Path,
) -> list[dict[str, object]]:
    selections = select_reference_rows(
        load_reference_rows(reference_input),
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter=mode_filter,
    )
    rows: list[dict[str, object]] = []
    for selection in selections:
        workload = ReconfigurationWorkload(
            seq_len=selection.seq_len,
            hidden_size=selection.hidden_size,
            intermediate_size=selection.intermediate_size,
            num_attention_heads=selection.num_attention_heads,
        )
        resolved_warmup_runs, resolved_runs_per_sample = iteration_schedule(
            selection.seq_len
        )
        if warmup_runs is not None:
            resolved_warmup_runs = warmup_runs
        if runs_per_sample is not None:
            resolved_runs_per_sample = runs_per_sample

        try:
            resolved_config = resolve_mode_operator_config(
                selection.execution_mode,
                workload,
                selection.selected_config,
            )
            result = benchmark_mode(
                selection.execution_mode,
                workload,
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
                seed=seed,
                power_backend=power_backend,
                operator_config=selection.selected_config,
                power_sample_interval_sec=power_sample_interval_sec,
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed %s benchmark for %s seq_len=%s: %s",
                selection.execution_mode,
                selection.study_case_id,
                selection.seq_len,
                exc,
            )
            resolved_config = resolve_mode_operator_config(
                selection.execution_mode,
                workload,
                selection.selected_config,
            )
            result = _failed_result(
                power_backend=power_backend,
                failure_message=str(exc),
            )

        row = {
            "study_id": "reconfiguration_overhead",
            "study_case_id": selection.study_case_id,
            "study_case_label": selection.study_case_label,
            "backend": "npu",
            "execution_mode": selection.execution_mode,
            "pattern_label": selection.execution_mode,
            "source_end_to_end_execution_mode": selection.source_execution_mode,
            "seq_len": selection.seq_len,
            "hidden_size": selection.hidden_size,
            "intermediate_size": selection.intermediate_size,
            "num_attention_heads": selection.num_attention_heads,
            "attention_head_size": selection.attention_head_size,
            "batch_size": selection.batch_size,
            "dtype": selection.dtype,
            "use_bias": selection.use_bias,
            "weights_source": selection.weights_source,
            "warmup_runs": resolved_warmup_runs,
            "runs_per_sample": resolved_runs_per_sample,
            "selected_candidate_ids_json": selection.selected_candidate_ids_json,
            "selected_config_json": json_dumps(resolved_config),
            **result,
        }
        rows.append(row)
        LOGGER.info("Completed %s -> %s", _case_descriptor(row), row["run_status"])

    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transformer_layer_new reconfiguration-overhead study"
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
    parser.add_argument("--power-sample-interval-sec", type=float, default=0.1)
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=default_reference_results_path(),
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    reference_input = args.reference_input.expanduser()
    output_path = args.output.expanduser()
    if not reference_input.exists():
        LOGGER.warning(
            "Reference end_to_end results not found at %s; writing empty CSV",
            reference_input,
        )
        write_rows(output_path, [])
        return 0

    rows = build_rows(
        family_filter=str(args.family),
        seq_len_filter=str(args.seq_len),
        mode_filter=str(args.mode),
        warmup_runs=args.warmup_iters,
        runs_per_sample=args.timed_iters,
        seed=args.seed,
        power_backend=str(args.power_backend),
        power_sample_interval_sec=float(args.power_sample_interval_sec),
        reference_input=reference_input,
    )
    if not rows:
        LOGGER.warning(
            "No matching end_to_end config rows found in %s for family=%s seq_len=%s "
            "mode=%s; writing empty CSV",
            reference_input,
            args.family,
            args.seq_len,
            args.mode,
        )
    write_rows(output_path, rows)
    LOGGER.info("Wrote %d reconfiguration-overhead rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
