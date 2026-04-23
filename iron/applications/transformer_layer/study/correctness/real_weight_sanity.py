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

import torch

from ..campaign import write_campaign_manifest
from ..run_lock import default_lock_path, hold_study_lock
from ..end_to_end.cases import get_case
from ..end_to_end.modes import benchmark_mode
from ..end_to_end.select import default_results_path, load_result_rows, select_result_rows
from ..end_to_end.validation import FINAL_ABS_TOL, FINAL_REL_TOL
from ..host_comparison.run import _forward_reference
from .common import error_summary, load_reference_payload

LOGGER = logging.getLogger(__name__)
REAL_WEIGHT_SANITY_STUDY_NAME = "transformer_layer real-weight sanity"
RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "reference_source",
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
    "abs_error_p95",
    "abs_error_p99",
    "abs_error_max",
    "rel_error_mean",
    "rel_error_p95",
    "rel_error_max",
    "abs_threshold_fraction",
    "rel_threshold_fraction",
    "selected_candidate_ids_json",
    "selected_config_json",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "real_weight_sanity.csv"
    )


def default_manifest_output_path(output_path: Path) -> Path:
    return output_path.with_name("real_weight_sanity_manifest.json")


def _ensure_reference_output(payload: dict[str, object], *, workload_variant: str, num_attention_heads: int) -> None:
    if isinstance(payload.get("output"), torch.Tensor):
        return
    input_tensor = payload["input"]
    weights = payload["weights"]
    causal_mask = None
    if workload_variant == "decoder_gpt2":
        seq_len = int(input_tensor.shape[0])
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    payload["output"] = _forward_reference(
        input_tensor,
        weights,
        num_attention_heads=num_attention_heads,
        workload_variant=workload_variant,
        causal_mask=causal_mask,
    )


def build_rows(
    *,
    results_input: Path,
    reference_sources: tuple[Path, ...],
    mode_filter: str,
    seed: int,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(load_result_rows(results_input), mode_filter=mode_filter)
    rows: list[dict[str, object]] = []
    selected_index = {
        (row.study_case_id, row.seq_len, row.execution_mode): row for row in selected_rows
    }
    for reference_source in reference_sources:
        payload = load_reference_payload(reference_source)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Real-weight sanity payloads must provide metadata.json (or an in-file "
                "metadata dict) with study_case_id and seq_len"
            )
        study_case_id = str(metadata.get("study_case_id") or "")
        seq_len = int(metadata.get("seq_len") or 0)
        execution_modes = (
            ("hybrid", "runlist", "offload")
            if mode_filter == "all"
            else (str(mode_filter),)
        )
        for execution_mode in execution_modes:
            selected_row = selected_index.get((study_case_id, seq_len, execution_mode))
            if selected_row is None:
                continue
            case = get_case(study_case_id, seq_len)
            _ensure_reference_output(
                payload,
                workload_variant=selected_row.workload_variant,
                num_attention_heads=selected_row.num_attention_heads,
            )
            LOGGER.info(
                "Running real-weight sanity for %s seq_len=%s mode=%s from %s",
                study_case_id,
                seq_len,
                execution_mode,
                reference_source,
            )
            result = benchmark_mode(
                execution_mode,
                case.workload,
                warmup_runs=int(selected_row.warmup_runs or 1),
                runs_per_sample=max(1, min(3, int(selected_row.runs_per_sample or 1))),
                seed=seed,
                power_backend="none",
                operator_config=selected_row.selected_config,
                include_reference_output=True,
                capture_output_tensors=True,
                reference_override=payload,
                scope_key_override=(
                    f"real_weight_sanity_{study_case_id}_{execution_mode}_{seq_len}"
                ),
            )
            summary: dict[str, object] = {}
            if result.get("run_status") == "passed":
                output_tensor = result.get("output_tensor")
                reference_output_tensor = result.get("reference_output_tensor")
                if output_tensor is not None and reference_output_tensor is not None:
                    summary = error_summary(
                        output_tensor,
                        reference_output_tensor,
                        abs_tol=FINAL_ABS_TOL,
                        rel_tol=FINAL_REL_TOL,
                    )
            rows.append(
                {
                    "study_id": "real_weight_sanity",
                    "reference_source": str(reference_source),
                    "campaign_id": selected_row.campaign_id,
                    "repeat_index": selected_row.repeat_index,
                    "matched_run_id": selected_row.matched_run_id,
                    "study_case_id": selected_row.study_case_id,
                    "study_case_label": selected_row.study_case_label,
                    "workload_variant": selected_row.workload_variant,
                    "execution_mode": execution_mode,
                    "seq_len": seq_len,
                    "avg_latency_ms": result.get("avg_latency_ms"),
                    "validation_error_count": result.get("validation_error_count"),
                    "run_status": result.get("run_status"),
                    "failure_message": result.get("failure_message"),
                    "abs_error_mean": summary.get("abs_error_mean"),
                    "abs_error_p95": summary.get("abs_error_p95"),
                    "abs_error_p99": summary.get("abs_error_p99"),
                    "abs_error_max": summary.get("abs_error_max"),
                    "rel_error_mean": summary.get("rel_error_mean"),
                    "rel_error_p95": summary.get("rel_error_p95"),
                    "rel_error_max": summary.get("rel_error_max"),
                    "abs_threshold_fraction": summary.get("abs_threshold_fraction"),
                    "rel_threshold_fraction": summary.get("rel_threshold_fraction"),
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
        description="Run selected end-to-end configs against explicit local real-weight payload exports."
    )
    parser.add_argument("--results-input", type=Path, default=default_results_path())
    parser.add_argument("--reference-source", dest="reference_sources", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--seed", type=int, default=42)
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
        study_name="real-weight sanity",
    ):
        rows = build_rows(
            results_input=args.results_input.expanduser(),
            reference_sources=tuple(path.expanduser() for path in args.reference_sources),
            mode_filter=str(args.mode),
            seed=int(args.seed),
        )
        write_rows(output_path, rows)
        campaign_id = str(rows[0]["campaign_id"]) if rows else "canonical"
        write_campaign_manifest(
            manifest_output,
            campaign_id=campaign_id,
            study_name=REAL_WEIGHT_SANITY_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.correctness.real_weight_sanity",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path,),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
