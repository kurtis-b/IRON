# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import csv

from iron.applications.transformer_layer.automated_benchmark import (
    _record_parity_results,
)
from iron.applications.transformer_layer.benchmark_common import (
    parse_seq_lens,
    write_results_csv,
)
from iron.applications.transformer_layer.benchmark_power import empty_power_stats
from iron.applications.transformer_layer.debug_log import append_debug_event
from iron.applications.transformer_layer.src.result_schema import RESULT_FIELD_ORDER
from iron.applications.transformer_layer.src.bench import (
    append_debug_event as append_debug_event_structured,
)
from iron.applications.transformer_layer.src.bench import (
    empty_power_stats as empty_power_stats_structured,
)
from iron.applications.transformer_layer.src.bench import (
    parse_seq_lens as parse_seq_lens_structured,
)
from iron.applications.transformer_layer.src.pipeline import (
    _record_parity_results as structured_record_parity_results,
)


def test_restructured_bench_package_preserves_legacy_imports():
    assert parse_seq_lens is parse_seq_lens_structured
    assert empty_power_stats is empty_power_stats_structured
    assert append_debug_event is append_debug_event_structured
    assert _record_parity_results is structured_record_parity_results


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


def test_write_results_csv_reserves_block_topology_columns(tmp_path: Path):
    output_csv = tmp_path / "results.csv"
    write_results_csv(
        output_csv,
        [
            {
                "study_id": "study",
                "backend": "npu",
                "execution_mode": "dataflow",
                "seq_len": 64,
                "batch_size": 1,
                "dtype": "bf16",
                "weights_source": "synthetic",
                "warmup_runs": 0,
                "runs_per_sample": 1,
                "measured_inference_count": 1,
                "timed_total_sec": 0.01,
                "avg_latency_ms": 10.0,
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block1_topology_family": "shared_runtime_qkv_proj",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block2_topology_family": "fused_mha_out_proj",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "block3_topology_family": "pipelined_addnorm_ffn_addnorm",
            }
        ],
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        header = csv.reader(handle).__next__()

    for field in (
        "block1_topology_id",
        "block1_topology_family",
        "block2_topology_id",
        "block2_topology_family",
        "block3_topology_id",
        "block3_topology_family",
    ):
        assert field in header

    assert header.index("process_model") < header.index("block1_topology_id")
    assert header.index("block3_topology_family") < header.index("run_status")
    assert RESULT_FIELD_ORDER.index("block1_topology_id") < RESULT_FIELD_ORDER.index(
        "run_status"
    )
