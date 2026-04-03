#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.addnorm.reference import (
    generate_golden_reference as generate_addnorm_reference,
)
from iron.operators.ffn.op import AIEFFN
from iron.operators.ffn.reference import (
    generate_golden_reference as generate_ffn_reference,
)
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.mha_out_proj.reference import (
    generate_golden_reference as generate_mha_out_proj_reference,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import (
    generate_golden_reference as generate_qkv_proj_reference,
)

from .cases import (
    BLOCK_KINDS,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    BlockKind,
    BlockWorkload,
    iter_cases,
)

LOGGER = logging.getLogger(__name__)
REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005

BLOCK_CONFIG_COLUMNS: dict[BlockKind, tuple[str, ...]] = {
    "qkv_proj": (
        "qkv_proj_tile_m",
        "qkv_proj_tile_k",
        "qkv_proj_tile_n",
        "qkv_proj_parallel_seq",
        "qkv_proj_parallel_emb",
    ),
    "mha_out_proj": (
        "mha_out_proj_parallel_seq",
        "mha_out_proj_q_seq_tile",
        "mha_out_proj_kv_seq_tile",
        "mha_out_proj_emb_tile",
        "mha_out_proj_parallel_heads",
        "mha_out_proj_o_proj_acc_depth",
    ),
    "addnorm": (
        "addnorm_num_aie_columns",
        "addnorm_tile_size",
    ),
    "ffn": (
        "ffn_num_aie_columns",
        "ffn_b_col_maj",
        "ffn_c_col_maj",
        "ffn_tile_m",
        "ffn_tile_k",
        "ffn_tile_n",
        "ffn_down_proj_depth",
        "ffn_n_a_tiles_distributed",
        "ffn_n_b_tiles_distributed",
        "ffn_stage_only",
        "ffn_gelu_stage",
    ),
}

CSV_FIELDNAMES = (
    "study_id",
    "family_id",
    "family_label",
    "seq_len",
    "block_kind",
    "candidate_index",
    "head_dim",
    "num_heads",
    "hidden_size",
    "ffn_dim",
    "avg_latency_ms",
    "bandwidth_gbps",
    "warmup_iters",
    "timed_iters",
    "validation_error_count",
    "run_status",
    "is_best",
    "error_message",
    *(column for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns),
)


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "block" / "results.csv"


def operator_kwargs(
    workload: BlockWorkload,
    block_kind: BlockKind,
    candidate: tuple[object, ...],
) -> dict[str, object]:
    if block_kind == "qkv_proj":
        tile_m, tile_k, tile_n, parallel_seq, parallel_emb = candidate
        return {
            "seq_len": workload.seq_len,
            "hidden_size": workload.hidden_size,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "parallel_seq": parallel_seq,
            "parallel_emb": parallel_emb,
        }

    if block_kind == "mha_out_proj":
        (
            parallel_seq,
            q_seq_tile,
            kv_seq_tile,
            emb_tile,
            parallel_heads,
            o_proj_acc_depth,
        ) = candidate
        return {
            "num_heads": workload.num_heads,
            "seq_len": workload.seq_len,
            "d": workload.head_dim,
            "parallel_seq": parallel_seq,
            "q_seq_tile": q_seq_tile,
            "kv_seq_tile": kv_seq_tile,
            "emb_tile": emb_tile,
            "parallel_heads": parallel_heads,
            "o_proj_acc_depth": o_proj_acc_depth,
        }

    if block_kind == "addnorm":
        num_aie_columns, tile_size = candidate
        return {
            "size": workload.seq_len * workload.hidden_size,
            "num_aie_columns": num_aie_columns,
            "tile_size": tile_size,
        }

    if block_kind == "ffn":
        (
            num_aie_columns,
            b_col_maj,
            c_col_maj,
            tile_m,
            tile_k,
            tile_n,
            down_proj_depth,
            n_a_tiles_distributed,
            n_b_tiles_distributed,
            stage_only,
            gelu_stage,
        ) = candidate
        return {
            "M": workload.seq_len,
            "K": workload.hidden_size,
            "N": workload.ffn_dim,
            "num_aie_columns": num_aie_columns,
            "b_col_maj": b_col_maj,
            "c_col_maj": c_col_maj,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "down_proj_depth": down_proj_depth,
            "n_a_tiles_distributed": n_a_tiles_distributed,
            "n_b_tiles_distributed": n_b_tiles_distributed,
            "stage_only": stage_only,
            "gelu_stage": gelu_stage,
            "emulate_bf16_mmul_with_bfp16": True,
        }

    raise ValueError(f"Unsupported block kind: {block_kind!r}")


def config_row(
    block_kind: BlockKind, candidate: tuple[object, ...]
) -> dict[str, object]:
    row = {
        column: "" for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns
    }
    for column, value in zip(BLOCK_CONFIG_COLUMNS[block_kind], candidate):
        row[column] = value
    return row


def _threshold_validation_result(
    buffer_error_counts: dict[str, int],
    *,
    max_acceptable_errors: int,
    skip_validation: bool = False,
) -> dict[str, object]:
    validation_error_count = sum(buffer_error_counts.values())
    if skip_validation:
        return {
            "validation_error_count": validation_error_count,
            "run_status": "passed",
            "error_message": "",
        }

    passed = all(
        error_count <= max_acceptable_errors
        for error_count in buffer_error_counts.values()
    )
    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if passed else "failed_validation",
        "error_message": (
            ""
            if passed
            else (
                f"validation_error_count={validation_error_count} exceeds "
                f"max_acceptable_errors={max_acceptable_errors}"
            )
        ),
    }


def _benchmark_qkv_proj(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        reference = generate_qkv_proj_reference(
            seq_len=workload.seq_len,
            hidden_size=workload.hidden_size,
            seed=seed,
        )
        operator = AIEQKVProj(
            context=context, **operator_kwargs(workload, "qkv_proj", candidate)
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "A": reference["input"].flatten(),
                "B": reference["input_b"].flatten(),
            },
            {
                "Q": reference["output_q"].flatten(),
                "K": reference["output_k"].flatten(),
                "V": reference["output_v"].flatten(),
            },
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {
                "Q": len(errors.get("Q", [])),
                "K": len(errors.get("K", [])),
                "V": len(errors.get("V", [])),
            },
            max_acceptable_errors=int(
                workload.seq_len * workload.hidden_size * ERROR_THRESHOLD
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_mha_out_proj(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        reference = generate_mha_out_proj_reference(
            heads=workload.num_heads,
            seq_len=workload.seq_len,
            d=workload.head_dim,
            seed=seed,
        )
        operator = AIEMHAOutProj(
            context=context,
            **operator_kwargs(workload, "mha_out_proj", candidate),
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "Q": reference["input_q"].flatten(),
                "K": reference["input_k"].flatten(),
                "V": reference["input_v"].flatten(),
                "W_O": reference["input_w_o"].flatten(),
            },
            {"O": reference["output"].flatten()},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"O": len(errors.get("O", []))},
            max_acceptable_errors=int(
                workload.seq_len
                * workload.head_dim
                * workload.num_heads
                * ERROR_THRESHOLD
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_addnorm(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        addnorm_kwargs = operator_kwargs(workload, "addnorm", candidate)
        total_size = int(addnorm_kwargs["size"])
        tile_size = int(addnorm_kwargs["tile_size"])
        if total_size % tile_size != 0:
            raise ValueError(
                f"addnorm size {total_size} is not divisible by tile_size {tile_size}"
            )
        reference = generate_addnorm_reference(
            rows=total_size // tile_size,
            cols=tile_size,
            seed=seed,
        )
        operator = AIEAddAndNorm(
            weights=reference["weight"],
            context=context,
            **addnorm_kwargs,
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "input1": reference["input1"],
                "input2": reference["input2"],
            },
            {"output": reference["output"]},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=int(total_size * ERROR_THRESHOLD),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_ffn(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        ffn_kwargs = operator_kwargs(workload, "ffn", candidate)
        reference = generate_ffn_reference(
            M=workload.seq_len,
            K=workload.hidden_size,
            N=workload.ffn_dim,
            seed=seed,
            b_col_maj=bool(ffn_kwargs["b_col_maj"]),
            c_col_maj=bool(ffn_kwargs["c_col_maj"]),
        )
        operator = AIEFFN(context=context, **ffn_kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "A": reference["input"].flatten(),
                "B_Up": reference["input_b_up"].flatten(),
                "B_Down": reference["input_b_down"].flatten(),
            },
            {"C": reference["output"].flatten()},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        stage_only = ffn_kwargs["stage_only"]
        validation = _threshold_validation_result(
            {"C": len(errors.get("C", []))},
            max_acceptable_errors=int(
                workload.seq_len * workload.hidden_size * ERROR_THRESHOLD
            ),
            skip_validation=stage_only is not None,
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def benchmark_candidate(
    block_kind: BlockKind,
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    benchmarkers = {
        "qkv_proj": _benchmark_qkv_proj,
        "mha_out_proj": _benchmark_mha_out_proj,
        "addnorm": _benchmark_addnorm,
        "ffn": _benchmark_ffn,
    }
    try:
        return benchmarkers[block_kind](
            workload,
            candidate,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            seed=seed,
        )
    except Exception as exc:
        return {
            "avg_latency_ms": "",
            "bandwidth_gbps": "",
            "validation_error_count": "",
            "run_status": "failed_exception",
            "error_message": str(exc),
        }


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        row["is_best"] = False
        grouped_rows[
            (str(row["family_id"]), int(row["seq_len"]), str(row["block_kind"]))
        ].append(row)

    for group_rows in grouped_rows.values():
        successful_rows = [
            row
            for row in group_rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] != ""
        ]
        if not successful_rows:
            continue
        best_row = min(successful_rows, key=lambda row: float(row["avg_latency_ms"]))
        best_row["is_best"] = True


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark shared block-study candidates."
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
        help="Select one retained family or benchmark all families.",
    )
    parser.add_argument(
        "--seq-len",
        default="all",
        help="Select one sequence length from the retained ladder or use all.",
    )
    parser.add_argument(
        "--block",
        choices=[*BLOCK_KINDS, "all"],
        default="all",
        help="Select one block kind or benchmark all block kinds.",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args(argv)


def selected_block_kinds(block_argument: str) -> tuple[BlockKind, ...]:
    return BLOCK_KINDS if block_argument == "all" else (block_argument,)  # type: ignore[return-value]


def selected_seq_len(seq_len_argument: str) -> int | None:
    if seq_len_argument == "all":
        return None
    seq_len = int(seq_len_argument)
    if seq_len not in SEQUENCE_LADDER:
        raise ValueError(
            f"Unsupported seq_len {seq_len}; expected one of {SEQUENCE_LADDER}"
        )
    return seq_len


def iteration_schedule(
    seq_len: int,
    *,
    warmup_iters: int | None,
    timed_iters: int | None,
) -> tuple[int, int]:
    if seq_len <= 128:
        default_warmup_iters, default_timed_iters = 10, 100
    elif seq_len <= 512:
        default_warmup_iters, default_timed_iters = 5, 50
    elif seq_len <= 2048:
        default_warmup_iters, default_timed_iters = 3, 20
    else:
        default_warmup_iters, default_timed_iters = 1, 10

    return (
        default_warmup_iters if warmup_iters is None else warmup_iters,
        default_timed_iters if timed_iters is None else timed_iters,
    )


def build_rows(
    *,
    family_argument: str,
    seq_len_argument: str,
    block_argument: str,
    warmup_iters: int | None,
    timed_iters: int | None,
    seed: int,
) -> list[dict[str, object]]:
    family_id = None if family_argument == "all" else family_argument
    seq_len = selected_seq_len(seq_len_argument)
    block_kinds = selected_block_kinds(block_argument)

    rows: list[dict[str, object]] = []
    for case in iter_cases(family_id=family_id, seq_len=seq_len):
        case_warmup_iters, case_timed_iters = iteration_schedule(
            case.seq_len,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        for block_kind in block_kinds:
            for candidate_index, candidate in enumerate(case.candidates(block_kind)):
                LOGGER.info(
                    "Benchmarking family=%s seq_len=%s block=%s candidate=%s warmup_iters=%s timed_iters=%s",
                    case.family_id,
                    case.seq_len,
                    block_kind,
                    candidate_index,
                    case_warmup_iters,
                    case_timed_iters,
                )
                result = benchmark_candidate(
                    block_kind,
                    case.workload,
                    candidate,
                    warmup_iters=case_warmup_iters,
                    timed_iters=case_timed_iters,
                    seed=seed,
                )
                rows.append(
                    {
                        "study_id": "block",
                        "family_id": case.family_id,
                        "family_label": case.family_label,
                        "seq_len": case.seq_len,
                        "block_kind": block_kind,
                        "candidate_index": candidate_index,
                        "head_dim": case.workload.head_dim,
                        "num_heads": case.workload.num_heads,
                        "hidden_size": case.workload.hidden_size,
                        "ffn_dim": case.workload.ffn_dim,
                        "warmup_iters": case_warmup_iters,
                        "timed_iters": case_timed_iters,
                        **config_row(block_kind, candidate),
                        **result,
                    }
                )

    mark_best_rows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)
    rows = build_rows(
        family_argument=args.family,
        seq_len_argument=args.seq_len,
        block_argument=args.block,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        seed=args.seed,
    )
    write_rows(args.output, rows)
    LOGGER.info("Wrote %s rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
