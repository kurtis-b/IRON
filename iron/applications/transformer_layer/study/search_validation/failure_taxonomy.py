#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

RESULTS_CSV_FIELDNAMES = (
    "execution_mode",
    "failure_kind",
    "count",
)


def build_taxonomy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        execution_mode = str(row.get("execution_mode") or "")
        run_status = str(row.get("run_status") or "")
        selected_matches_baseline = str(row.get("selected_matches_baseline") or "")
        if run_status == "passed":
            failure_kind = "passed" if selected_matches_baseline == "True" else "selection_delta"
        else:
            failure_kind = run_status or "failed"
        counter[(execution_mode, failure_kind)] += 1
    return [
        {
            "execution_mode": execution_mode,
            "failure_kind": failure_kind,
            "count": count,
        }
        for (execution_mode, failure_kind), count in sorted(counter.items())
    ]


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a failure taxonomy summary from search-validation results."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.input.expanduser().open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    write_rows(args.output.expanduser(), build_taxonomy(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
