# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from iron.applications.transformer_layer.automated_benchmark import (
    _record_parity_results,
)


def test_record_parity_results_skips_empty_rows(tmp_path: Path):
    debug_log_csv = tmp_path / "debug.csv"
    parity_output = tmp_path / "parity.csv"

    _record_parity_results(
        debug_log_csv=str(debug_log_csv),
        study_id="design_patterns_end_to_end",
        parity={"output_csv": str(parity_output)},
        parity_rows=[],
    )

    assert not parity_output.exists()
    text = debug_log_csv.read_text(encoding="utf-8")
    assert "parity_skipped" in text
    assert "No parity rows matched" in text
