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

LOGGER = logging.getLogger(__name__)
RUNLIST_LAUNCH_ABLATION_STUDY_NAME = "transformer_layer runlist launch ablation"
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
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "submission_model",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "latency_sample_count",
    "min_latency_ms",
    "max_latency_ms",
    "compile_setup_time_ms",
    "effective_gflops_per_sec",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "speedup_vs_separate_launch",
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
        / "runlist_launch_ablation.csv"
    )


def default_manifest_output_path(output_path: Path) -> Path:
    return output_path.with_name("runlist_launch_ablation_manifest.json")


def _resolved_sampling(selected_row) -> tuple[int, int]:
    return int(selected_row.warmup_runs or 1), int(selected_row.runs_per_sample or 3)


def _resolved_sampling_with_overrides(
    selected_row,
    *,
    warmup_runs_override: int | None,
    runs_per_sample_override: int | None,
) -> tuple[int, int]:
    warmup_runs, runs_per_sample = _resolved_sampling(selected_row)
    if warmup_runs_override is not None:
        warmup_runs = max(1, int(warmup_runs_override))
    if runs_per_sample_override is not None:
        runs_per_sample = max(1, int(runs_per_sample_override))
    return warmup_runs, runs_per_sample


def _row_key(row: dict[str, object]) -> tuple[str, int, str, int]:
    return (
        str(row.get("study_case_id") or ""),
        int(row.get("repeat_index") or 0),
        str(row.get("workload_variant") or ""),
        int(row.get("seq_len") or 0),
    )


def _annotate_speedups(rows: list[dict[str, object]]) -> None:
    separate_launch_latency: dict[tuple[str, int, str, int], float] = {}
    for row in rows:
        row["speedup_vs_separate_launch"] = ""
        if str(row.get("submission_model") or "") != "separate_launch":
            continue
        if str(row.get("run_status") or "") != "passed":
            continue
        latency = row.get("avg_latency_ms")
        if latency in ("", None):
            continue
        separate_launch_latency[_row_key(row)] = float(latency)
    for row in rows:
        if str(row.get("run_status") or "") != "passed":
            continue
        latency = row.get("avg_latency_ms")
        if latency in ("", None):
            continue
        baseline = separate_launch_latency.get(_row_key(row))
        if baseline is None or float(latency) <= 0:
            continue
        row["speedup_vs_separate_launch"] = baseline / float(latency)


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
    seed: int,
    warmup_runs_override: int | None = None,
    runs_per_sample_override: int | None = None,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        load_result_rows(results_input),
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter="runlist",
    )
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        warmup_runs, runs_per_sample = _resolved_sampling_with_overrides(
            selected_row,
            warmup_runs_override=warmup_runs_override,
            runs_per_sample_override=runs_per_sample_override,
        )
        for submission_model, use_runlist in (
            ("runlist", True),
            ("separate_launch", False),
        ):
            LOGGER.info(
                "Running runlist ablation for %s seq_len=%s mode=%s",
                selected_row.study_case_id,
                selected_row.seq_len,
                submission_model,
            )
            result = benchmark_mode(
                "runlist",
                case.workload,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                power_backend="none",
                operator_config=selected_row.selected_config,
                use_runlist=use_runlist,
                scope_key_override=(
                    f"runlist_launch_ablation_{selected_row.study_case_id}_"
                    f"{selected_row.seq_len}_{submission_model}"
                ),
            )
            rows.append(
                {
                    "study_id": "runlist_launch_ablation",
                    "campaign_id": selected_row.campaign_id,
                    "repeat_index": selected_row.repeat_index,
                    "matched_run_id": selected_row.matched_run_id,
                    "study_case_id": selected_row.study_case_id,
                    "study_case_label": selected_row.study_case_label,
                    "workload_variant": selected_row.workload_variant,
                    "execution_mode": "runlist",
                    "seq_len": selected_row.seq_len,
                    "hidden_size": selected_row.hidden_size,
                    "intermediate_size": selected_row.intermediate_size,
                    "num_attention_heads": selected_row.num_attention_heads,
                    "attention_head_size": selected_row.attention_head_size,
                    "submission_model": submission_model,
                    "warmup_runs": warmup_runs,
                    "runs_per_sample": runs_per_sample,
                    "avg_latency_ms": result.get("avg_latency_ms"),
                    "latency_sample_count": result.get("latency_sample_count"),
                    "min_latency_ms": result.get("min_latency_ms"),
                    "max_latency_ms": result.get("max_latency_ms"),
                    "compile_setup_time_ms": result.get("compile_setup_time_ms"),
                    "effective_gflops_per_sec": result.get("effective_gflops_per_sec"),
                    "npu_dispatch_count": result.get("npu_dispatch_count"),
                    "npu_unique_instruction_binary_count": result.get(
                        "npu_unique_instruction_binary_count"
                    ),
                    "npu_unique_xclbin_count": result.get("npu_unique_xclbin_count"),
                    "run_status": result.get("run_status"),
                    "failure_message": result.get("failure_message"),
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
    _annotate_speedups(rows)
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
            "Compare selected runlist configurations under fused runlist submission "
            "versus per-entry separate launches."
        )
    )
    parser.add_argument("--workload-variant", default="all")
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
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
    results_input = args.results_input.expanduser()
    output_path = args.output.expanduser()
    manifest_output = (
        default_manifest_output_path(output_path)
        if args.manifest_output is None
        else args.manifest_output.expanduser()
    )
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="runlist launch ablation",
    ):
        rows = build_rows(
            results_input=results_input,
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            seq_len_filter=str(args.seq_len),
            seed=int(args.seed),
            warmup_runs_override=args.warmup_runs,
            runs_per_sample_override=args.runs_per_sample,
        )
        write_rows(output_path, rows)
        campaign_id = str(rows[0]["campaign_id"]) if rows else "canonical"
        write_campaign_manifest(
            manifest_output,
            campaign_id=campaign_id,
            study_name=RUNLIST_LAUNCH_ABLATION_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.runlist_launch_ablation.run",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path,),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
