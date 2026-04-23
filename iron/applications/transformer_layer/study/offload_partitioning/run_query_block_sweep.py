#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
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
from ...pattern.offload.op import resolve_offload_operator_config

LOGGER = logging.getLogger(__name__)
OFFLOAD_PARTITIONING_STUDY_NAME = "transformer_layer offload partitioning study"
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
    "query_block_size",
    "query_block_count",
    "source_query_block_size",
    "warmup_runs",
    "runs_per_sample",
    "avg_latency_ms",
    "latency_sample_count",
    "min_latency_ms",
    "max_latency_ms",
    "effective_gflops_per_sec",
    "npu_gemm_time_sec",
    "host_attention_time_sec",
    "host_elementwise_time_sec",
    "transfer_time_sec",
    "total_wall_time_sec",
    "launch_count",
    "bytes_written",
    "bytes_read",
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
        / "offload_breakdown.csv"
    )


def default_manifest_output_path(output_path: Path) -> Path:
    return output_path.with_name("offload_breakdown_manifest.json")


def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        return tuple()
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _resolved_sampling(selected_row) -> tuple[int, int]:
    warmup_runs = int(selected_row.warmup_runs or 1)
    runs_per_sample = int(selected_row.runs_per_sample or 3)
    return warmup_runs, runs_per_sample


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


def _config_with_query_block_size(selected_row, *, query_block_size: int) -> dict[str, dict[str, object]]:
    config = copy.deepcopy(selected_row.selected_config)
    config.setdefault("attn_scores", {})
    config.setdefault("attn_output", {})
    for operator_name in ("attn_scores", "attn_output"):
        config[operator_name]["query_block_size"] = int(query_block_size)
        config[operator_name]["M"] = int(query_block_size)
        config[operator_name]["num_heads"] = int(selected_row.num_attention_heads)
    return config


def _legal_query_block_sizes(selected_row) -> tuple[int, ...]:
    legal: list[int] = []
    for query_block_size in _divisors(selected_row.seq_len):
        try:
            resolve_offload_operator_config(
                selected_row.seq_len,
                selected_row.hidden_size,
                selected_row.intermediate_size,
                selected_row.num_attention_heads,
                workload_variant=selected_row.workload_variant,
                operator_config=_config_with_query_block_size(
                    selected_row,
                    query_block_size=query_block_size,
                ),
            )
        except Exception:
            continue
        legal.append(int(query_block_size))
    return tuple(sorted(set(legal)))


def _selected_query_block_sizes(
    selected_row,
    *,
    query_block_policy: str,
) -> tuple[int, ...]:
    legal_query_block_sizes = _legal_query_block_sizes(selected_row)
    if query_block_policy == "all":
        return legal_query_block_sizes
    if query_block_policy != "representative":
        raise ValueError(f"Unsupported query block policy: {query_block_policy}")
    if not legal_query_block_sizes:
        return tuple()
    selected_query_block_size = _optional_int(
        selected_row.selected_config.get("attn_scores", {}).get("query_block_size")
    )
    representative = {
        legal_query_block_sizes[0],
        legal_query_block_sizes[(len(legal_query_block_sizes) - 1) // 2],
        legal_query_block_sizes[-1],
    }
    if (
        selected_query_block_size is not None
        and selected_query_block_size in legal_query_block_sizes
    ):
        representative.add(int(selected_query_block_size))
    return tuple(sorted(representative))


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
    seed: int,
    warmup_runs_override: int | None = None,
    runs_per_sample_override: int | None = None,
    query_block_policy: str = "all",
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        load_result_rows(results_input),
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter="offload",
    )
    rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        warmup_runs, runs_per_sample = _resolved_sampling_with_overrides(
            selected_row,
            warmup_runs_override=warmup_runs_override,
            runs_per_sample_override=runs_per_sample_override,
        )
        case = get_case(selected_row.study_case_id, selected_row.seq_len)
        source_query_block_size = _optional_int(
            selected_row.selected_config.get("attn_scores", {}).get("query_block_size")
        ) or selected_row.seq_len
        for query_block_size in _selected_query_block_sizes(
            selected_row,
            query_block_policy=query_block_policy,
        ):
            operator_config = _config_with_query_block_size(
                selected_row,
                query_block_size=query_block_size,
            )
            LOGGER.info(
                "Running offload partition sweep for %s seq_len=%s qbs=%s",
                selected_row.study_case_id,
                selected_row.seq_len,
                query_block_size,
            )
            result = benchmark_mode(
                "offload",
                case.workload,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                seed=seed,
                power_backend="none",
                operator_config=operator_config,
                capture_runtime_breakdown=True,
                scope_key_override=(
                    f"offload_partition_{selected_row.study_case_id}_"
                    f"{selected_row.seq_len}_qbs{query_block_size}"
                ),
            )
            breakdown = result.get("runtime_breakdown")
            if not isinstance(breakdown, dict):
                breakdown = {}
            rows.append(
                {
                    "study_id": "offload_partitioning",
                    "campaign_id": selected_row.campaign_id,
                    "repeat_index": selected_row.repeat_index,
                    "matched_run_id": selected_row.matched_run_id,
                    "study_case_id": selected_row.study_case_id,
                    "study_case_label": selected_row.study_case_label,
                    "workload_variant": selected_row.workload_variant,
                    "execution_mode": "offload",
                    "seq_len": selected_row.seq_len,
                    "hidden_size": selected_row.hidden_size,
                    "intermediate_size": selected_row.intermediate_size,
                    "num_attention_heads": selected_row.num_attention_heads,
                    "attention_head_size": selected_row.attention_head_size,
                    "query_block_size": query_block_size,
                    "query_block_count": breakdown.get("query_block_count", ""),
                    "source_query_block_size": source_query_block_size,
                    "warmup_runs": warmup_runs,
                    "runs_per_sample": runs_per_sample,
                    "avg_latency_ms": result.get("avg_latency_ms"),
                    "latency_sample_count": result.get("latency_sample_count"),
                    "min_latency_ms": result.get("min_latency_ms"),
                    "max_latency_ms": result.get("max_latency_ms"),
                    "effective_gflops_per_sec": result.get("effective_gflops_per_sec"),
                    "npu_gemm_time_sec": breakdown.get("npu_gemm_time_sec"),
                    "host_attention_time_sec": breakdown.get("host_attention_time_sec"),
                    "host_elementwise_time_sec": breakdown.get(
                        "host_elementwise_time_sec"
                    ),
                    "transfer_time_sec": breakdown.get("transfer_time_sec"),
                    "total_wall_time_sec": breakdown.get("total_wall_time_sec"),
                    "launch_count": breakdown.get("launch_count"),
                    "bytes_written": breakdown.get("bytes_written"),
                    "bytes_read": breakdown.get("bytes_read"),
                    "run_status": result.get("run_status"),
                    "failure_message": result.get("failure_message"),
                    "selected_candidate_ids_json": json.dumps(
                        selected_row.selected_candidate_ids,
                        sort_keys=True,
                    ),
                    "selected_config_json": json.dumps(
                        operator_config,
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
            "Sweep legal offload query_block_size values and record the runtime "
            "breakdown for each selected offload configuration."
        )
    )
    parser.add_argument("--workload-variant", default="all")
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument(
        "--query-block-policy",
        choices=("all", "representative"),
        default="all",
    )
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
        study_name="offload partitioning",
    ):
        rows = build_rows(
            results_input=results_input,
            workload_variant_filter=str(args.workload_variant),
            family_filter=str(args.family),
            seq_len_filter=str(args.seq_len),
            seed=int(args.seed),
            warmup_runs_override=args.warmup_runs,
            runs_per_sample_override=args.runs_per_sample,
            query_block_policy=str(args.query_block_policy),
        )
        write_rows(output_path, rows)
        campaign_id = (
            str(rows[0]["campaign_id"]) if rows else "canonical"
        )
        write_campaign_manifest(
            manifest_output,
            campaign_id=campaign_id,
            study_name=OFFLOAD_PARTITIONING_STUDY_NAME,
            command_line=[
                "python3",
                "-m",
                "iron.applications.transformer_layer.study.offload_partitioning.run_query_block_sweep",
                *(sys.argv[1:] if argv is None else argv),
            ],
            output_files=(output_path,),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
