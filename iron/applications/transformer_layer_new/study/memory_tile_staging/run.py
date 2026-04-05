#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from iron.applications.transformer_layer_new.study.block.cases import (
    FAMILY_IDS,
    SEQUENCE_LADDER,
)
from iron.applications.transformer_layer_new.study.block.run import (
    benchmark_candidate,
    iteration_schedule,
)

from .plot_staging_depth import write_canonical_plots
from .select import (
    CONFIG_COLUMNS_BY_BLOCK_KIND,
    ReferenceSelection,
    STAGING_BLOCK_KINDS,
    default_reference_results_path,
    load_reference_rows,
    select_reference_rows,
)

LOGGER = logging.getLogger(__name__)

RESULTS_CSV_FIELDNAMES = (
    "study_id",
    "family_id",
    "family_label",
    "seq_len",
    "block_kind",
    "source_candidate_index",
    "source_staging_depth",
    "staging_depth",
    "head_dim",
    "num_heads",
    "hidden_size",
    "ffn_dim",
    "warmup_iters",
    "timed_iters",
    "avg_latency_ms",
    "bandwidth_gbps",
    "speedup_vs_depth1",
    "validation_error_count",
    "run_status",
    "is_best_depth",
    "error_message",
    *(
        column
        for block_kind in STAGING_BLOCK_KINDS
        for column in CONFIG_COLUMNS_BY_BLOCK_KIND[block_kind]
    ),
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "memory_tile_staging"
        / "results.csv"
    )


def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        return tuple()
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


def source_staging_depth(selection: ReferenceSelection) -> int:
    if selection.block_kind == "mha_out_proj":
        return int(selection.source_candidate[5])
    return int(selection.source_candidate[6])


def supported_staging_depths(selection: ReferenceSelection) -> tuple[int, ...]:
    if selection.block_kind == "mha_out_proj":
        emb_tile = int(selection.source_candidate[3])
        if emb_tile <= 0 or selection.hidden_size % emb_tile != 0:
            return tuple()
        return _divisors(selection.hidden_size // emb_tile)

    tile_k = int(selection.source_candidate[4])
    if tile_k <= 0 or selection.hidden_size % tile_k != 0:
        return tuple()
    return _divisors(selection.hidden_size // tile_k)


def candidate_with_staging_depth(
    selection: ReferenceSelection,
    staging_depth: int,
) -> tuple[object, ...]:
    candidate = list(selection.source_candidate)
    if selection.block_kind == "mha_out_proj":
        candidate[5] = int(staging_depth)
    else:
        candidate[6] = int(staging_depth)
    return tuple(candidate)


def config_row(
    block_kind: str,
    candidate: tuple[object, ...],
) -> dict[str, object]:
    row = {
        column: ""
        for supported_block_kind in STAGING_BLOCK_KINDS
        for column in CONFIG_COLUMNS_BY_BLOCK_KIND[supported_block_kind]
    }
    for column, value in zip(CONFIG_COLUMNS_BY_BLOCK_KIND[block_kind], candidate):
        row[column] = value
    return row


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        row["is_best_depth"] = False
        key = (str(row["family_id"]), int(row["seq_len"]), str(row["block_kind"]))
        grouped_rows.setdefault(key, []).append(row)

    for group_rows in grouped_rows.values():
        successful_rows = [
            row
            for row in group_rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] not in ("", None)
        ]
        if not successful_rows:
            continue
        best_row = min(successful_rows, key=lambda row: float(row["avg_latency_ms"]))
        best_row["is_best_depth"] = True


def annotate_speedup_vs_depth1(rows: list[dict[str, object]]) -> None:
    baseline_latency_by_group: dict[tuple[str, int, str], float] = {}
    for row in rows:
        row["speedup_vs_depth1"] = ""
        if (
            row["run_status"] != "passed"
            or row["avg_latency_ms"] in ("", None)
            or int(row["staging_depth"]) != 1
        ):
            continue
        baseline_latency_by_group[
            (str(row["family_id"]), int(row["seq_len"]), str(row["block_kind"]))
        ] = float(row["avg_latency_ms"])

    for row in rows:
        if row["run_status"] != "passed" or row["avg_latency_ms"] in ("", None):
            continue
        group = (str(row["family_id"]), int(row["seq_len"]), str(row["block_kind"]))
        baseline_latency = baseline_latency_by_group.get(group)
        current_latency = float(row["avg_latency_ms"])
        if baseline_latency is None or baseline_latency <= 0 or current_latency <= 0:
            continue
        row["speedup_vs_depth1"] = baseline_latency / current_latency


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in RESULTS_CSV_FIELDNAMES}
            )


def _selection_descriptor(selection: ReferenceSelection) -> str:
    return (
        f"{selection.family_id} seq_len={selection.seq_len} "
        f"block={selection.block_kind}"
    )


def build_selection_rows(
    selection: ReferenceSelection,
    *,
    warmup_iters: int | None,
    timed_iters: int | None,
    seed: int,
) -> list[dict[str, object]]:
    resolved_warmup_iters, resolved_timed_iters = iteration_schedule(
        selection.seq_len,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )
    depths = supported_staging_depths(selection)
    if not depths:
        LOGGER.warning(
            "Skipping %s because no supported staging depths were derived",
            _selection_descriptor(selection),
        )
        return []

    rows: list[dict[str, object]] = []
    source_depth = source_staging_depth(selection)
    for staging_depth in depths:
        candidate = candidate_with_staging_depth(selection, staging_depth)
        LOGGER.info(
            "Benchmarking %s staging_depth=%s warmup_iters=%s timed_iters=%s",
            _selection_descriptor(selection),
            staging_depth,
            resolved_warmup_iters,
            resolved_timed_iters,
        )
        result = benchmark_candidate(
            selection.block_kind,
            selection.workload,
            candidate,
            warmup_iters=resolved_warmup_iters,
            timed_iters=resolved_timed_iters,
            seed=seed,
        )
        rows.append(
            {
                "study_id": "memory_tile_staging",
                "family_id": selection.family_id,
                "family_label": selection.family_label,
                "seq_len": selection.seq_len,
                "block_kind": selection.block_kind,
                "source_candidate_index": selection.source_candidate_index,
                "source_staging_depth": source_depth,
                "staging_depth": staging_depth,
                "head_dim": selection.head_dim,
                "num_heads": selection.num_heads,
                "hidden_size": selection.hidden_size,
                "ffn_dim": selection.ffn_dim,
                "warmup_iters": resolved_warmup_iters,
                "timed_iters": resolved_timed_iters,
                **config_row(selection.block_kind, candidate),
                **result,
            }
        )

    annotate_speedup_vs_depth1(rows)
    mark_best_rows(rows)
    return rows


def build_rows(
    *,
    family_filter: str,
    seq_len_filter: str,
    block_filter: str,
    warmup_iters: int | None,
    timed_iters: int | None,
    seed: int,
    reference_input: Path,
) -> list[dict[str, object]]:
    selections = select_reference_rows(
        load_reference_rows(reference_input),
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        block_filter=block_filter,
    )
    rows: list[dict[str, object]] = []
    for selection in selections:
        rows.extend(
            build_selection_rows(
                selection,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                seed=seed,
            )
        )
    mark_best_rows(rows)
    annotate_speedup_vs_depth1(rows)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark memory-tile staging depth for selected block-study winners."
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
        "--block",
        choices=[*STAGING_BLOCK_KINDS, "all"],
        default="all",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=default_reference_results_path(),
    )
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)
    reference_input = args.reference_input.expanduser()
    output_path = args.output.expanduser()

    if not reference_input.exists():
        LOGGER.warning(
            "Reference block results not found at %s; writing empty outputs",
            reference_input,
        )
        write_rows(output_path, [])
        write_canonical_plots(output_path, output_path.parent)
        return 0

    rows = build_rows(
        family_filter=str(args.family),
        seq_len_filter=str(args.seq_len),
        block_filter=str(args.block),
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        seed=args.seed,
        reference_input=reference_input,
    )
    write_rows(output_path, rows)
    write_canonical_plots(output_path, output_path.parent)
    LOGGER.info("Wrote %d memory-tile staging rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
