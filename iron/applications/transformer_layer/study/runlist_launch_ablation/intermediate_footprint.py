#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from iron.common import AIEContext

from ..end_to_end.select import default_results_path, load_result_rows, select_result_rows
from ..end_to_end.modes import _build_operator
from ...pattern.reference import generate_golden_reference

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "campaign_id",
    "repeat_index",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "intermediate_buffer_count",
    "intermediate_buffer_bytes_estimate",
    "largest_intermediate_buffer_bytes_estimate",
    "intermediate_buffer_names_json",
    "selected_candidate_ids_json",
    "selected_config_json",
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "runlist_intermediate_footprint.csv"
    )


def _buffer_bytes_estimate(buffer_elems: object) -> int:
    return int(buffer_elems or 0) * 2


def _summarize_buffers(selected_row) -> dict[str, object]:
    reference = generate_golden_reference(
        selected_row.seq_len,
        selected_row.hidden_size,
        selected_row.intermediate_size,
        selected_row.num_attention_heads,
        workload_variant=selected_row.workload_variant,
        include_output=False,
    )
    operator = _build_operator(
        "runlist",
        selected_row,
        reference["weights"],
        context=AIEContext(use_runlist=True),
        operator_config=selected_row.selected_config,
    )
    static_names = set(getattr(operator, "buffer_static_data", {}).keys())
    intermediate_names = [
        buffer_name
        for buffer_name in getattr(operator, "buffers", {})
        if buffer_name not in static_names and buffer_name not in {"input", "output"}
    ]
    byte_sizes = [
        _buffer_bytes_estimate(getattr(operator, "buffers", {}).get(buffer_name))
        for buffer_name in intermediate_names
    ]
    return {
        "intermediate_buffer_count": len(intermediate_names),
        "intermediate_buffer_bytes_estimate": sum(byte_sizes),
        "largest_intermediate_buffer_bytes_estimate": max(byte_sizes, default=0),
        "intermediate_buffer_names_json": json.dumps(sorted(intermediate_names)),
    }


def build_rows(
    *,
    results_input: Path,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
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
        summary = _summarize_buffers(selected_row)
        rows.append(
            {
                "study_id": "runlist_launch_ablation_intermediate_footprint",
                "campaign_id": selected_row.campaign_id,
                "repeat_index": selected_row.repeat_index,
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": "runlist",
                "seq_len": selected_row.seq_len,
                "hidden_size": selected_row.hidden_size,
                "intermediate_size": selected_row.intermediate_size,
                "num_attention_heads": selected_row.num_attention_heads,
                "attention_head_size": selected_row.attention_head_size,
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
        description="Estimate the intermediate runtime buffer footprint of selected runlist configs."
    )
    parser.add_argument("--workload-variant", default="all")
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--results-input", type=Path, default=default_results_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(
        results_input=args.results_input.expanduser(),
        workload_variant_filter=str(args.workload_variant),
        family_filter=str(args.family),
        seq_len_filter=str(args.seq_len),
    )
    write_rows(args.output.expanduser(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
