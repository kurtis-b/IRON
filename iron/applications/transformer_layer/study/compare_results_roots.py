#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two transformer-layer results roots.

Answers one question: are the numbers in a fresh suite run consistent with a
previous known-good run?

Equality is the wrong test. These are wall-clock measurements on a shared
machine, so every column drifts a little between runs and a few drift a lot.
The comparison is therefore tiered:

- Identifier columns must match exactly. A difference there is a structural
  change, not noise.
- Only mean-based columns gate the verdict. `min_*` and `max_*` are single
  samples from a noisy distribution, so one lucky or unlucky sample must not
  turn a healthy run red; they are reported for information.
- Gating is on the median and p90 of the per-mode drift distribution rather
  than the per-row maximum. With 162 rows a 15% per-row latency threshold fires
  on a handful of `offload` rows even when nothing is wrong.
- `offload` gets a wider tolerance than `hybrid` and `runlist` because it is
  roughly ten times noisier run to run.

Selection columns (`is_best`, `selected_config_json`) flip legitimately when
the autotuner picks between near-tied candidates, so they are counted rather
than flagged. The `pattern_label` rename `Hybrid` -> `Coarse runlist` is a
known intended difference and is counted the same way.

Before trusting any comparison, check that the two runs share a runtime stack.
`results_manifest.json` records the git commit, the dirty flag, the platform,
and the full `xrt-smi examine` output for exactly this purpose -- an XRT
version change alone can move `offload` latency by 25% at large sequence
lengths while leaving `hybrid` and `runlist` untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .end_to_end.select import load_result_rows

KEY_FIELDS = ("study_case_id", "execution_mode", "seq_len")

IDENTIFIER_FIELDS = (
    "study_id",
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "backend",
    "execution_mode",
    "pattern_label",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "batch_size",
    "dtype",
    "use_bias",
    "weights_source",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "power_backend",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "process_model",
    "validation_error_count",
    "run_status",
    "failure_message",
)

# Only these columns can fail the run. Everything else numeric is reported.
GATING_FIELDS = ("avg_latency_ms", "effective_gflops_per_sec", "avg_power_w")

LATENCY_FIELDS = (
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "effective_gflops_per_sec",
)
POWER_FIELDS = (
    "avg_power_w",
    "min_power_w",
    "max_power_w",
    "raw_avg_power_w",
    "raw_min_power_w",
    "raw_max_power_w",
    "power_std_w",
    "raw_power_std_w",
)

# (warn, fail) percentages applied to the median and p90 of the drift spread.
LATENCY_TOLERANCE = {
    "hybrid": (5.0, 15.0),
    "runlist": (5.0, 15.0),
    "offload": (20.0, 35.0),
}
DEFAULT_LATENCY_TOLERANCE = (10.0, 25.0)
POWER_TOLERANCE = (15.0, 50.0)

# Selection output flips when the autotuner breaks a near-tie differently.
INFORMATIONAL_FIELDS = (
    "selected_candidate_ids_json",
    "selected_config_json",
    "is_best",
)

# Differences introduced on purpose by the execution-strategy rename.
RENAMED_VALUES = {"pattern_label": {"Hybrid": "Coarse runlist"}}

RESULT_CSVS = (
    "end_to_end/results_all_power.csv",
    "end_to_end/latency_variation.csv",
    "end_to_end/correctness_spot_checks.csv",
)

# Re-derived from the measured CSVs. Byte-identical when the inputs match, so
# any difference is either the rename or a changed autotuner selection.
DERIVED_CSVS = (
    "roofline/kernel_points.csv",
    "resource_usage/hybrid_selected_ops.csv",
    "resource_usage/runlist_selected_ops.csv",
    "resource_usage/offload_selected_ops.csv",
    "resource_usage/dataflow_block_best_configs.csv",
    "end_to_end/fairness_repeatability.csv",
    "host_comparison/fairness_repeatability.csv",
)


def _numeric(value: object) -> float | None:
    text = str(value if value is not None else "").strip()
    if text in ("", "None", "nan", "NaN"):
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if result == result else None


def _relative_percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return None if candidate == 0.0 else float("inf")
    return abs(candidate - baseline) / abs(baseline) * 100.0


def _signed_percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in KEY_FIELDS)


def _is_intended_rename(field: str, baseline: str, candidate: str) -> bool:
    return RENAMED_VALUES.get(field, {}).get(baseline) == candidate


def percentile_90(values: list[float]) -> float:
    """p90 by nearest rank, so a short list still yields a defined value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return ordered[index]


class Report:
    """Collects report lines and counts warnings and failures."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings = 0
        self.failures = 0

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.lines.append(f"  WARN  {text}")

    def fail(self, text: str) -> None:
        self.failures += 1
        self.lines.append(f"  FAIL  {text}")

    def render(self) -> str:
        return "\n".join(self.lines)


def compare_manifests(
    report: Report, baseline_root: Path, candidate_root: Path
) -> None:
    report.say("\n=== results_manifest.json ===")
    baseline_path = baseline_root / "results_manifest.json"
    candidate_path = candidate_root / "results_manifest.json"
    if not baseline_path.exists() or not candidate_path.exists():
        report.say("  SKIP (missing on one side)")
        return

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    report.say(
        f"  complete: baseline={baseline.get('complete')} "
        f"candidate={candidate.get('complete')}"
    )
    if not candidate.get("complete"):
        report.fail("candidate manifest reports complete=false")

    baseline_git = baseline.get("git") or {}
    candidate_git = candidate.get("git") or {}
    report.say(
        f"  commit: {str(baseline_git.get('commit'))[:12]}"
        f"(dirty={baseline_git.get('dirty')}) -> "
        f"{str(candidate_git.get('commit'))[:12]}"
        f"(dirty={candidate_git.get('dirty')})"
    )
    if baseline_git.get("dirty") or candidate_git.get("dirty"):
        report.say("  NOTE  a run was measured from a dirty tree")

    baseline_system = baseline.get("system") or {}
    candidate_system = candidate.get("system") or {}
    baseline_platform = str(baseline_system.get("platform", ""))
    candidate_platform = str(candidate_system.get("platform", ""))
    if baseline_platform != candidate_platform:
        report.say(f"  NOTE  platform: {baseline_platform} -> {candidate_platform}")

    baseline_xrt = _xrt_version(baseline_system)
    candidate_xrt = _xrt_version(candidate_system)
    if baseline_xrt != candidate_xrt:
        report.say(
            f"  NOTE  XRT version: {baseline_xrt} -> {candidate_xrt}. "
            "Latency differences may be environmental rather than code."
        )

    baseline_coverage = (baseline.get("suite") or {}).get("coverage") or {}
    candidate_coverage = (candidate.get("suite") or {}).get("coverage") or {}
    for name in sorted(set(baseline_coverage) | set(candidate_coverage)):
        baseline_block = baseline_coverage.get(name) or {}
        candidate_block = candidate_coverage.get(name) or {}
        parts = []
        for key in ("actual_rows", "passed_rows", "aggregate_rows", "complete"):
            if key not in baseline_block and key not in candidate_block:
                continue
            before = baseline_block.get(key)
            after = candidate_block.get(key)
            marker = "" if before == after else "  <-- differs"
            parts.append(f"{key}: {before} -> {after}{marker}")
        if parts:
            report.say(f"  {name}: " + "; ".join(parts))


def _xrt_version(system_block: dict) -> str:
    """Pull the XRT version line out of the recorded xrt-smi examine output."""
    stdout = str((system_block.get("xrt_smi_examine") or {}).get("stdout", ""))
    seen_xrt_header = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "XRT":
            seen_xrt_header = True
            continue
        if seen_xrt_header and stripped.startswith("Version"):
            return stripped.split(":", 1)[-1].strip()
    return "unknown"


def compare_result_csv(
    report: Report, rel_path: str, baseline_root: Path, candidate_root: Path
) -> None:
    baseline_path = baseline_root / rel_path
    candidate_path = candidate_root / rel_path
    report.say(f"\n=== {rel_path} ===")
    if not baseline_path.exists():
        # A file the baseline never produced is not a candidate regression,
        # whether or not the candidate happens to have it.
        report.say("  SKIP (absent from baseline)")
        return
    if not candidate_path.exists():
        report.fail("missing in candidate")
        return

    baseline_rows = load_result_rows(baseline_path)
    candidate_rows = load_result_rows(candidate_path)
    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    candidate_by_key = {_row_key(row): row for row in candidate_rows}

    only_baseline = sorted(set(baseline_by_key) - set(candidate_by_key))
    only_candidate = sorted(set(candidate_by_key) - set(baseline_by_key))
    shared = sorted(set(baseline_by_key) & set(candidate_by_key))
    report.say(
        f"  rows: baseline={len(baseline_rows)} candidate={len(candidate_rows)} "
        f"matched={len(shared)}"
    )
    for key in only_baseline[:10]:
        report.fail(f"row only in baseline: {key}")
    for key in only_candidate[:10]:
        report.fail(f"row only in candidate: {key}")

    identifier_mismatches = 0
    renames: dict[str, int] = defaultdict(int)
    selection_flips: dict[str, int] = defaultdict(int)
    drift: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    signed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for key in shared:
        baseline_row = baseline_by_key[key]
        candidate_row = candidate_by_key[key]
        mode = str(baseline_row.get("execution_mode", "")).strip() or "unknown"

        for field in IDENTIFIER_FIELDS:
            if field not in baseline_row or field not in candidate_row:
                continue
            before = str(baseline_row.get(field, "")).strip()
            after = str(candidate_row.get(field, "")).strip()
            if before == after:
                continue
            if _is_intended_rename(field, before, after):
                renames[f"{field}: {before!r} -> {after!r}"] += 1
                continue
            identifier_mismatches += 1
            if identifier_mismatches <= 10:
                report.fail(f"{key} {field}: baseline={before!r} candidate={after!r}")

        for field in INFORMATIONAL_FIELDS:
            if field in baseline_row and str(baseline_row.get(field)) != str(
                candidate_row.get(field)
            ):
                selection_flips[field] += 1

        for field in LATENCY_FIELDS + POWER_FIELDS:
            if field not in baseline_row:
                continue
            before_value = _numeric(baseline_row.get(field))
            after_value = _numeric(candidate_row.get(field))
            if before_value is None or after_value is None:
                continue
            percent = _relative_percent(before_value, after_value)
            if percent is None:
                continue
            drift[mode][field].append(percent)
            signed_percent = _signed_percent(before_value, after_value)
            if signed_percent is not None:
                signed[mode][field].append(signed_percent)

    report.say(f"  identifier mismatches: {identifier_mismatches}")
    for label, count in sorted(renames.items()):
        report.say(f"  intended rename applied to {count} rows -- {label}")
    if selection_flips:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(selection_flips.items()))
        report.say(f"  selection flips (autotuner re-selection): {summary}")

    if drift:
        _report_drift(report, drift, signed)


def _report_drift(
    report: Report,
    drift: dict[str, dict[str, list[float]]],
    signed: dict[str, dict[str, list[float]]],
) -> None:
    report.say("  drift by execution_mode (median / p90 / max, % relative):")
    for mode in sorted(drift):
        latency_warn, latency_fail = LATENCY_TOLERANCE.get(
            mode, DEFAULT_LATENCY_TOLERANCE
        )
        for field in sorted(drift[mode]):
            values = drift[mode][field]
            if not values:
                continue
            median = statistics.median(values)
            p90 = percentile_90(values)
            largest = max(values)
            gating = field in GATING_FIELDS
            if field in POWER_FIELDS:
                warn_at, fail_at = POWER_TOLERANCE
            else:
                warn_at, fail_at = latency_warn, latency_fail
            signed_values = signed[mode].get(field) or []
            signed_median = statistics.median(signed_values) if signed_values else 0.0
            report.say(
                f"    [{'GATE' if gating else 'info'}] {mode:>8} {field:<32} "
                f"med={median:6.2f} p90={p90:6.2f} max={largest:7.2f} "
                f"signed_med={signed_median:+6.2f} n={len(values)}"
            )
            if not gating:
                continue
            if median > fail_at or p90 > fail_at:
                report.fail(
                    f"{mode}/{field}: median={median:.2f}% p90={p90:.2f}% "
                    f"exceeds fail threshold {fail_at}%"
                )
            elif median > warn_at or p90 > warn_at:
                report.warn(
                    f"{mode}/{field}: median={median:.2f}% p90={p90:.2f}% "
                    f"exceeds warn threshold {warn_at}%"
                )


def compare_derived_csv(
    report: Report, rel_path: str, baseline_root: Path, candidate_root: Path
) -> None:
    baseline_path = baseline_root / rel_path
    candidate_path = candidate_root / rel_path
    if not baseline_path.exists() or not candidate_path.exists():
        report.say(f"  {rel_path}: SKIP (missing on one side)")
        return

    baseline_text = baseline_path.read_text(encoding="utf-8")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    if baseline_text == candidate_text:
        report.say(f"  {rel_path}: byte-identical")
        return

    baseline_rows = list(csv.reader(baseline_text.splitlines()))
    candidate_rows = list(csv.reader(candidate_text.splitlines()))
    if len(baseline_rows) != len(candidate_rows):
        report.say(
            f"  {rel_path}: row count {len(baseline_rows)} -> "
            f"{len(candidate_rows)} (regrouped selections; check coverage)"
        )
        return

    rename_only = True
    differing_cells = 0
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        if baseline_row == candidate_row:
            continue
        if len(baseline_row) != len(candidate_row):
            rename_only = False
            continue
        for before, after in zip(baseline_row, candidate_row):
            if before == after:
                continue
            differing_cells += 1
            if before.replace("Hybrid", "Coarse runlist") != after:
                rename_only = False

    if rename_only:
        report.say(
            f"  {rel_path}: {differing_cells} cells differ, all explained by the "
            "Hybrid -> Coarse runlist rename"
        )
    else:
        report.say(
            f"  {rel_path}: {differing_cells} cells differ (autotuner re-selection)"
        )


def compare_roots(baseline_root: Path, candidate_root: Path) -> Report:
    report = Report()
    report.say(f"baseline : {baseline_root}")
    report.say(f"candidate: {candidate_root}")

    compare_manifests(report, baseline_root, candidate_root)
    for rel_path in RESULT_CSVS:
        compare_result_csv(report, rel_path, baseline_root, candidate_root)

    report.say("\n=== derived CSVs (re-derived from the measured rows) ===")
    for rel_path in DERIVED_CSVS:
        compare_derived_csv(report, rel_path, baseline_root, candidate_root)

    report.say("")
    report.say(f"warnings: {report.warnings}   failures: {report.failures}")
    report.say(f"VERDICT: {'OK' if report.failures == 0 else 'PROBLEM'}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two transformer-layer results roots with per-mode "
            "statistical tolerances."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Results root of the known-good reference run.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Results root of the run being checked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare_roots(
        args.baseline.expanduser().resolve(),
        args.candidate.expanduser().resolve(),
    )
    print(report.render())
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
