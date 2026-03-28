# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from iron.applications.transformer_layer.automated_benchmark import (
    _record_parity_results,
)
from iron.applications.transformer_layer.benchmark_common import parse_seq_lens
from iron.applications.transformer_layer.benchmark_power import empty_power_stats
from iron.applications.transformer_layer.debug_log import append_debug_event
from iron.applications.transformer_layer.src.bench import (
    append_debug_event as append_debug_event_structured,
)
from iron.applications.transformer_layer.src.bench import (
    empty_power_stats as empty_power_stats_structured,
)
from iron.applications.transformer_layer.src.bench import (
    parse_seq_lens as parse_seq_lens_structured,
)


def test_restructured_bench_package_preserves_legacy_imports():
    assert parse_seq_lens is parse_seq_lens_structured
    assert empty_power_stats is empty_power_stats_structured
    assert append_debug_event is append_debug_event_structured


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
