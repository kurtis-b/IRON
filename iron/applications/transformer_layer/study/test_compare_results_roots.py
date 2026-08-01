#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer.study.compare_results_roots import (
    _is_intended_rename,
    _numeric,
    _relative_percent,
    _signed_percent,
    _xrt_version,
    compare_roots,
    percentile_90,
)

RESULT_FIELDS = (
    "study_case_id",
    "workload_variant",
    "backend",
    "execution_mode",
    "pattern_label",
    "seq_len",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "effective_gflops_per_sec",
    "avg_power_w",
    "run_status",
    "selected_config_json",
    "is_best",
)


def _write_result_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _result_row(
    *,
    mode: str = "hybrid",
    seq_len: int = 512,
    latency: float = 10.0,
    pattern_label: str = "Hybrid",
    power: float = 20.0,
    selected_config_json: str = '{"a": 1}',
) -> dict[str, object]:
    return {
        "study_case_id": "baseline_768",
        "workload_variant": "encoder_bert",
        "backend": "npu",
        "execution_mode": mode,
        "pattern_label": pattern_label,
        "seq_len": seq_len,
        "avg_latency_ms": latency,
        "min_latency_ms": latency * 0.95,
        "max_latency_ms": latency * 1.05,
        "effective_gflops_per_sec": 1000.0 / latency,
        "avg_power_w": power,
        "run_status": "passed",
        "selected_config_json": selected_config_json,
        "is_best": "True",
    }


def _build_root(path: Path, rows: list[dict[str, object]]) -> Path:
    _write_result_csv(path / "end_to_end" / "results_all_power.csv", rows)
    (path / "results_manifest.json").write_text(
        json.dumps({"complete": True, "git": {"commit": "abc123", "dirty": False}}),
        encoding="utf-8",
    )
    return path


def test_identical_roots_compare_clean(tmp_path):
    rows = [_result_row(seq_len=seq) for seq in (512, 1024)]
    baseline = _build_root(tmp_path / "baseline", rows)
    candidate = _build_root(tmp_path / "candidate", rows)

    report = compare_roots(baseline, candidate)

    assert report.failures == 0
    assert report.warnings == 0
    assert "VERDICT: OK" in report.render()
    assert "matched=2" in report.render()


def test_pattern_label_rename_is_counted_not_flagged(tmp_path):
    baseline = _build_root(tmp_path / "baseline", [_result_row(pattern_label="Hybrid")])
    candidate = _build_root(
        tmp_path / "candidate", [_result_row(pattern_label="Coarse runlist")]
    )

    report = compare_roots(baseline, candidate)

    assert report.failures == 0
    assert "intended rename applied to 1 rows" in report.render()
    assert "identifier mismatches: 0" in report.render()


def test_unexpected_identifier_change_fails(tmp_path):
    baseline = _build_root(tmp_path / "baseline", [_result_row()])
    candidate_row = _result_row()
    candidate_row["workload_variant"] = "decoder_gpt2"
    candidate = _build_root(tmp_path / "candidate", [candidate_row])

    report = compare_roots(baseline, candidate)

    assert report.failures >= 1
    assert "VERDICT: PROBLEM" in report.render()
    assert "workload_variant" in report.render()


def test_offload_tolerates_more_latency_drift_than_hybrid(tmp_path):
    """Offload is roughly ten times noisier, so the same drift must not gate it."""
    drifted = 10.0 * 1.12  # 12% slower

    hybrid_baseline = _build_root(
        tmp_path / "hb", [_result_row(mode="hybrid", latency=10.0)]
    )
    hybrid_candidate = _build_root(
        tmp_path / "hc", [_result_row(mode="hybrid", latency=drifted)]
    )
    offload_baseline = _build_root(
        tmp_path / "ob", [_result_row(mode="offload", latency=10.0)]
    )
    offload_candidate = _build_root(
        tmp_path / "oc", [_result_row(mode="offload", latency=drifted)]
    )

    hybrid_report = compare_roots(hybrid_baseline, hybrid_candidate)
    offload_report = compare_roots(offload_baseline, offload_candidate)

    assert hybrid_report.warnings >= 1
    assert offload_report.warnings == 0
    assert hybrid_report.failures == 0
    assert offload_report.failures == 0


def test_large_latency_regression_fails(tmp_path):
    baseline = _build_root(tmp_path / "baseline", [_result_row(latency=10.0)])
    candidate = _build_root(tmp_path / "candidate", [_result_row(latency=30.0)])

    report = compare_roots(baseline, candidate)

    assert report.failures >= 1
    assert "VERDICT: PROBLEM" in report.render()


def test_min_and_max_latency_do_not_gate(tmp_path):
    """Sample extrema are reported but must not turn a healthy run red."""
    baseline_row = _result_row(latency=10.0)
    candidate_row = _result_row(latency=10.0)
    candidate_row["max_latency_ms"] = 100.0
    candidate_row["min_latency_ms"] = 0.5

    baseline = _build_root(tmp_path / "baseline", [baseline_row])
    candidate = _build_root(tmp_path / "candidate", [candidate_row])

    report = compare_roots(baseline, candidate)

    assert report.failures == 0
    assert report.warnings == 0
    assert "max_latency_ms" in report.render()


def test_selection_flip_is_counted_not_failed(tmp_path):
    baseline = _build_root(
        tmp_path / "baseline", [_result_row(selected_config_json='{"a": 1}')]
    )
    candidate = _build_root(
        tmp_path / "candidate", [_result_row(selected_config_json='{"a": 2}')]
    )

    report = compare_roots(baseline, candidate)

    assert report.failures == 0
    assert "selection flips" in report.render()


def test_missing_candidate_file_fails_but_missing_baseline_skips(tmp_path):
    rows = [_result_row()]
    baseline = _build_root(tmp_path / "baseline", rows)
    candidate = _build_root(tmp_path / "candidate", rows)
    # latency_variation.csv exists in neither root, so it must skip, not fail.
    report = compare_roots(baseline, candidate)
    assert report.failures == 0

    (candidate / "end_to_end" / "results_all_power.csv").unlink()
    broken = compare_roots(baseline, candidate)
    assert broken.failures >= 1
    assert "missing in candidate" in broken.render()


def test_incomplete_candidate_manifest_fails(tmp_path):
    rows = [_result_row()]
    baseline = _build_root(tmp_path / "baseline", rows)
    candidate = _build_root(tmp_path / "candidate", rows)
    (candidate / "results_manifest.json").write_text(
        json.dumps({"complete": False}), encoding="utf-8"
    )

    report = compare_roots(baseline, candidate)

    assert report.failures >= 1
    assert "complete=false" in report.render()


def test_xrt_version_change_is_reported(tmp_path):
    rows = [_result_row()]
    baseline = _build_root(tmp_path / "baseline", rows)
    candidate = _build_root(tmp_path / "candidate", rows)
    for root, version in ((baseline, "2.23.0"), (candidate, "2.21.0")):
        (root / "results_manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "system": {
                        "xrt_smi_examine": {
                            "stdout": f"System Configuration\nXRT\n  Version : {version}\n"
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    report = compare_roots(baseline, candidate)

    assert "XRT version: 2.23.0 -> 2.21.0" in report.render()


def test_xrt_version_parses_only_the_xrt_block():
    system = {
        "xrt_smi_examine": {
            "stdout": (
                "System Configuration\n"
                "  Version              : 99.9.9\n"
                "XRT\n"
                "  Version              : 2.21.0\n"
                "  Branch               : HEAD\n"
            )
        }
    }
    assert _xrt_version(system) == "2.21.0"
    assert _xrt_version({}) == "unknown"


def test_numeric_rejects_blank_and_nan():
    assert _numeric("1.5") == 1.5
    assert _numeric("") is None
    assert _numeric("None") is None
    assert _numeric("nan") is None
    assert _numeric("not-a-number") is None


def test_relative_and_signed_percent():
    assert _relative_percent(10.0, 11.0) == 10.0
    assert _relative_percent(10.0, 9.0) == 10.0
    assert _signed_percent(10.0, 9.0) == -10.0
    assert _signed_percent(0.0, 1.0) is None
    assert _relative_percent(0.0, 0.0) is None


def test_percentile_90_uses_nearest_rank():
    assert percentile_90([]) == 0.0
    assert percentile_90([5.0]) == 5.0
    assert percentile_90([1.0, 2.0, 3.0, 4.0, 5.0]) == 5.0


def test_is_intended_rename_only_matches_the_known_pair():
    assert _is_intended_rename("pattern_label", "Hybrid", "Coarse runlist")
    assert not _is_intended_rename("pattern_label", "Hybrid", "Something else")
    assert not _is_intended_rename("workload_variant", "Hybrid", "Coarse runlist")
