#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..npu_runtime_checks import warn_if_npu_power_mode_not_turbo
from .cases import EXECUTION_MODES, FAMILY_IDS, SEQUENCE_LADDER, EndToEndCase, get_case
from .power import SUPPORTED_POWER_BACKENDS
from .run import (
    RESULTS_CSV_FIELDNAMES,
    TUNING_CSV_FIELDNAMES,
    build_rows,
    mark_best_rows,
    write_rows,
)

LOGGER = logging.getLogger(__name__)


def _case_descriptor(case: EndToEndCase) -> str:
    return (
        f"{case.study_case_id} seq_len={case.seq_len} hidden={case.hidden_size} "
        f"inter={case.intermediate_size} heads={case.num_attention_heads}"
    )


def selected_sequence_ladder(max_seq_len: int) -> tuple[int, ...]:
    return tuple(seq_len for seq_len in SEQUENCE_LADDER if seq_len <= max_seq_len)


def iter_selected_cases(
    family_filter: str = "all",
    *,
    max_seq_len: int,
) -> tuple[EndToEndCase, ...]:
    family_ids = FAMILY_IDS if family_filter == "all" else (family_filter,)
    seq_lengths = selected_sequence_ladder(max_seq_len)
    return tuple(
        get_case(family_id, seq_len)
        for family_id in family_ids
        for seq_len in seq_lengths
    )


def default_output_path(max_seq_len: int) -> Path:
    results_dir = Path(__file__).resolve().parents[2] / "results" / "end_to_end"
    if int(max_seq_len) == max(SEQUENCE_LADDER):
        return results_dir / "results_all_power.csv"
    return results_dir / f"results_upto{max_seq_len}_power.csv"


def default_tuning_output_path(max_seq_len: int) -> Path:
    results_dir = Path(__file__).resolve().parents[2] / "results" / "end_to_end"
    if int(max_seq_len) == max(SEQUENCE_LADDER):
        return results_dir / "tuning_all_power.csv"
    return results_dir / f"tuning_upto{max_seq_len}_power.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a powered transformer_layer end-to-end sweep "
        "across the retained sequence ladder up to a chosen maximum."
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        choices=SEQUENCE_LADDER,
        default=256,
    )
    parser.add_argument(
        "--mode",
        choices=[*EXECUTION_MODES, "all"],
        default="all",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--power-backend",
        choices=list(SUPPORTED_POWER_BACKENDS),
        default="turbostat_pkgwatt",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tuning-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    warn_if_npu_power_mode_not_turbo(LOGGER, study_name="powered end-to-end sweep")

    output_path = (
        default_output_path(int(args.max_seq_len))
        if args.output is None
        else args.output.expanduser()
    )
    tuning_output_path = (
        default_tuning_output_path(int(args.max_seq_len))
        if args.tuning_output is None
        else args.tuning_output.expanduser()
    )
    cases = iter_selected_cases(args.family, max_seq_len=int(args.max_seq_len))

    LOGGER.info(
        "Starting powered end-to-end sweep with %d case(s), mode=%s, power_backend=%s",
        len(cases),
        args.mode,
        args.power_backend,
    )

    tuning_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for case_index, case in enumerate(cases, start=1):
        LOGGER.info("Case %d/%d: %s", case_index, len(cases), _case_descriptor(case))
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
    LOGGER.info(
        "Wrote %d end-to-end rows to %s",
        len(final_rows),
        output_path,
    )
    LOGGER.info(
        "Wrote %d tuning rows to %s",
        len(tuning_rows),
        tuning_output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
