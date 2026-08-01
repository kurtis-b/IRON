#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cases import EXECUTION_MODES, FAMILY_IDS, mode_operators
from .select import default_results_path, load_result_rows, select_result_rows

# The header written into the generated JSON, not this file's own license.
# REUSE-Ignore stops `reuse lint` reading the quoted tags as malformed SPDX
# expressions belonging to this module.
# REUSE-IgnoreStart
_GENERATED_JSON_HEADER = (
    "SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
    "SPDX-License-Identifier: Apache-2.0",
    "Generated fixed-winner candidates from end-to-end selected configs.",
)
# REUSE-IgnoreEnd


def _empty_payload() -> dict[str, object]:
    return {
        "_header": list(_GENERATED_JSON_HEADER),
        **{family_id: {} for family_id in FAMILY_IDS},
    }


def build_fixed_winner_payloads(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, object]]:
    payloads = {execution_mode: _empty_payload() for execution_mode in EXECUTION_MODES}
    seen: set[tuple[str, int, str, str]] = set()

    for selected_row in select_result_rows(rows):
        execution_mode = selected_row.execution_mode
        mode_payload = payloads[execution_mode]
        family_payload = mode_payload[selected_row.study_case_id]
        if not isinstance(family_payload, dict):
            raise TypeError("candidate payload family entry must be a dict")
        seq_payload = family_payload.setdefault(str(selected_row.seq_len), {})
        if not isinstance(seq_payload, dict):
            raise TypeError("candidate payload sequence entry must be a dict")
        for operator_name in mode_operators(
            execution_mode,
            selected_row.workload_variant,
        ):
            key = (
                selected_row.study_case_id,
                selected_row.seq_len,
                execution_mode,
                operator_name,
            )
            if key in seen:
                raise ValueError(
                    "Duplicate selected operator row for "
                    f"family={selected_row.study_case_id} seq_len={selected_row.seq_len} "
                    f"mode={execution_mode} operator={operator_name}"
                )
            seen.add(key)
            seq_payload[operator_name] = [
                {
                    "candidate_id": selected_row.selected_candidate_ids[operator_name],
                    "config": dict(selected_row.selected_config[operator_name]),
                }
            ]
    return payloads


def write_fixed_winner_payloads(
    output_dir: Path,
    payloads: dict[str, dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for execution_mode, payload in payloads.items():
        output_path = output_dir / f"{execution_mode}_candidates.json"
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed-winner candidate JSONs from completed end-to-end results."
        )
    )
    parser.add_argument("--results", type=Path, default=default_results_path())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = build_fixed_winner_payloads(load_result_rows(args.results.expanduser()))
    write_fixed_winner_payloads(args.output_dir.expanduser(), payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
