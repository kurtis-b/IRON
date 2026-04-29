#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .end_to_end.cases import EXECUTION_MODES, FAMILY_IDS, SEQUENCE_LADDER
from .end_to_end.run_correctness_spot_checks import SPOT_CHECK_SEQ_LENS
from .end_to_end.run_staging_ablation import (
    STAGING_ABLATION_SEQUENCE_LENGTHS,
    STAGING_BLOCK_KINDS,
)
from .memory_tile_staging.run import STAGING_SEQUENCE_LENGTHS
from .unattended_smoke_job import (
    REQUIRED_EXECUTION_OUTPUT_FILES,
    REQUIRED_PLOT_FILES,
)

PAPER_HELPER_SEQUENCE_LENGTHS: tuple[int, ...] = (512, 2048, 8192)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def default_build_root() -> Path:
    return repo_root() / "build" / "transformer_layer_end_to_end"


def _run_command(argv: list[str], *, cwd: Path | None = None) -> dict[str, object]:
    try:
        completed = subprocess.run(
            argv,
            cwd=None if cwd is None else str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "argv": argv,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": completed.returncode == 0,
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_metadata(root: Path) -> dict[str, object]:
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run_command(["git", "status", "--short"], cwd=root)
    diff_stat = _run_command(
        ["git", "diff", "--stat", "--", "iron/applications/transformer_layer"],
        cwd=root,
    )
    status_stdout = str(status.get("stdout") or "")
    return {
        "commit": str(commit.get("stdout") or ""),
        "dirty": bool(status_stdout.strip()),
        "status_short": status_stdout.splitlines(),
        "transformer_layer_diff_stat": str(diff_stat.get("stdout") or ""),
        "commands": {
            "commit": commit,
            "status": status,
            "diff_stat": diff_stat,
        },
    }


def _system_metadata() -> dict[str, object]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": platform.uname()._asdict(),
        "lscpu": _run_command(["lscpu"]),
        "xrt_smi_examine": _run_command(["xrt-smi", "examine", "-r", "all"]),
    }


def _file_record(results_root: Path, parts: tuple[str, ...]) -> dict[str, object]:
    path = results_root.joinpath(*parts)
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": "/".join(parts),
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _expected_files(results_root: Path) -> list[dict[str, object]]:
    seen: set[tuple[str, ...]] = set()
    records: list[dict[str, object]] = []
    for parts in (*REQUIRED_EXECUTION_OUTPUT_FILES, *REQUIRED_PLOT_FILES):
        if parts in seen:
            continue
        seen.add(parts)
        if parts == ("results_manifest.json",):
            records.append(
                {
                    "path": "results_manifest.json",
                    "exists": True,
                    "size_bytes": 0,
                    "self_record": True,
                }
            )
            continue
        records.append(_file_record(results_root, parts))
    return records


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _default_sequence_sets(suite_profile: str) -> dict[str, list[int]]:
    helper_seq_lens = (
        SEQUENCE_LADDER if suite_profile == "full" else PAPER_HELPER_SEQUENCE_LENGTHS
    )
    return {
        "block": list(SEQUENCE_LADDER),
        "memory_tile_staging": list(STAGING_SEQUENCE_LENGTHS),
        "end_to_end": list(SEQUENCE_LADDER),
        "selected_components": list(helper_seq_lens),
        "correctness": list(SPOT_CHECK_SEQ_LENS),
        "latency_variation": list(helper_seq_lens),
        "staging_ablation": list(STAGING_ABLATION_SEQUENCE_LENGTHS),
        "host_comparison": list(SEQUENCE_LADDER),
        "resource_usage": list(SEQUENCE_LADDER),
        "roofline": list(SEQUENCE_LADDER),
    }


def _state_suite_metadata(results_root: Path) -> tuple[str, dict[str, list[int]]]:
    state_path = results_root / "automation" / "state.json"
    if not state_path.exists():
        return "full", _default_sequence_sets("full")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "full", _default_sequence_sets("full")
    suite_profile = str(state.get("suite_profile") or "full")
    sequence_sets = state.get("suite_sequence_sets")
    if not isinstance(sequence_sets, dict):
        return suite_profile, _default_sequence_sets(suite_profile)
    normalized: dict[str, list[int]] = {}
    for key, values in sequence_sets.items():
        if not isinstance(values, list):
            continue
        normalized[str(key)] = [int(value) for value in values]
    merged = _default_sequence_sets(suite_profile)
    merged.update(normalized)
    return suite_profile, merged


def _coverage_record(
    *,
    name: str,
    seq_lens: list[int],
    expected_rows: int | None,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    passed_rows = sum(1 for row in rows if row.get("run_status") == "passed")
    return {
        "name": name,
        "seq_lens": seq_lens,
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "passed_rows": passed_rows,
        "complete": expected_rows is None or len(rows) >= expected_rows,
    }


def _suite_coverage(
    *,
    results_root: Path,
    suite_profile: str,
    sequence_sets: dict[str, list[int]],
) -> dict[str, object]:
    end_to_end_rows = _read_csv_rows(
        results_root / "end_to_end" / "results_all_power.csv"
    )
    selected_aggregate_rows = _read_csv_rows(
        results_root / "end_to_end" / "selected_component_aggregates.csv"
    )
    incomplete_selected_groups = [
        row
        for row in selected_aggregate_rows
        if row.get("row_kind") == "isolated_group"
        and str(row.get("is_complete")).lower() not in {"true", "1"}
    ]
    families = list(FAMILY_IDS)
    modes = list(EXECUTION_MODES)
    return {
        "profile": suite_profile,
        "families": families,
        "execution_modes": modes,
        "sequence_sets": sequence_sets,
        "coverage": {
            "end_to_end": _coverage_record(
                name="end_to_end",
                seq_lens=sequence_sets.get("end_to_end", list(SEQUENCE_LADDER)),
                expected_rows=len(families)
                * len(sequence_sets.get("end_to_end", []))
                * len(modes),
                rows=end_to_end_rows,
            ),
            "selected_components": {
                "seq_lens": sequence_sets.get("selected_components", []),
                "aggregate_rows": len(selected_aggregate_rows),
                "incomplete_group_count": len(incomplete_selected_groups),
                "incomplete_groups": incomplete_selected_groups,
            },
            "correctness": _coverage_record(
                name="correctness",
                seq_lens=sequence_sets.get("correctness", list(SPOT_CHECK_SEQ_LENS)),
                expected_rows=len(families)
                * len(sequence_sets.get("correctness", []))
                * len(modes),
                rows=_read_csv_rows(
                    results_root / "end_to_end" / "correctness_spot_checks.csv"
                ),
            ),
            "latency_variation": _coverage_record(
                name="latency_variation",
                seq_lens=sequence_sets.get("latency_variation", []),
                expected_rows=len(families)
                * len(sequence_sets.get("latency_variation", []))
                * len(modes),
                rows=_read_csv_rows(
                    results_root / "end_to_end" / "latency_variation.csv"
                ),
            ),
            "staging_ablation": _coverage_record(
                name="staging_ablation",
                seq_lens=sequence_sets.get("staging_ablation", []),
                expected_rows=len(families)
                * len(sequence_sets.get("staging_ablation", []))
                * len(STAGING_BLOCK_KINDS),
                rows=_read_csv_rows(
                    results_root / "end_to_end" / "staging_ablation.csv"
                ),
            ),
            "host_comparison": {
                "seq_lens": sequence_sets.get("host_comparison", list(SEQUENCE_LADDER)),
                "host_backends": ["igpu"],
                "cpu_host_jobs_included": False,
                "actual_rows": len(
                    _read_csv_rows(results_root / "host_comparison" / "results.csv")
                ),
            },
            "resource_usage": {
                "seq_lens": sequence_sets.get("resource_usage", list(SEQUENCE_LADDER)),
                "coverage_source": "existing artifacts and selected rows",
            },
            "roofline": {
                "seq_lens": sequence_sets.get("roofline", list(SEQUENCE_LADDER)),
                "coverage_source": "end-to-end, tuning, and memcpy rows",
            },
        },
    }


def build_manifest(
    *,
    results_root: Path,
    build_root: Path,
    repo: Path,
    suite_profile: str | None = None,
    sequence_sets: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    expected_files = _expected_files(results_root)
    missing_files = [
        str(record["path"])
        for record in expected_files
        if not bool(record.get("exists")) and not bool(record.get("self_record"))
    ]
    state_suite_profile, state_sequence_sets = _state_suite_metadata(results_root)
    resolved_suite_profile = suite_profile or state_suite_profile
    resolved_sequence_sets = (
        state_sequence_sets if sequence_sets is None else sequence_sets
    )
    return {
        "study_id": "transformer_layer_results_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "build_root": str(build_root),
        "build_root_exists": build_root.exists(),
        "repo_root": str(repo),
        "git": _git_metadata(repo),
        "system": _system_metadata(),
        "expected_files": expected_files,
        "missing_files": missing_files,
        "suite": _suite_coverage(
            results_root=results_root,
            suite_profile=resolved_suite_profile,
            sequence_sets=resolved_sequence_sets,
        ),
        "complete": not missing_files,
    }


def write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a frozen manifest for transformer-layer study results."
    )
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--build-root", type=Path, default=default_build_root())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--suite-profile", choices=("full", "paper"), default=None)
    parser.add_argument("--sequence-sets-json", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = args.results_root.expanduser()
    output = (
        results_root / "results_manifest.json"
        if args.output is None
        else args.output.expanduser()
    )
    manifest = build_manifest(
        results_root=results_root,
        build_root=args.build_root.expanduser(),
        repo=repo_root(),
        suite_profile=args.suite_profile,
        sequence_sets=(
            None
            if args.sequence_sets_json is None
            else {
                str(key): [int(value) for value in values]
                for key, values in json.loads(args.sequence_sets_json).items()
            }
        ),
    )
    write_manifest(output, manifest)
    if manifest["missing_files"] and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
