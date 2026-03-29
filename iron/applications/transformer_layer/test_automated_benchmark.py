# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import csv
import subprocess
from types import SimpleNamespace

from iron.applications.transformer_layer.automated_benchmark import (
    _record_parity_results,
)
from iron.applications.transformer_layer.benchmark_common import (
    parse_seq_lens,
    write_results_csv,
)
from iron.applications.transformer_layer.benchmark_power import empty_power_stats
from iron.applications.transformer_layer.debug_log import append_debug_event
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
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
from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    _decorate_row_for_case as structured_decorate_row_for_case,
)
from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    _failure_result_row as structured_failure_result_row,
)
from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    _resolve_spec as structured_resolve_spec,
)
from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    _run_parity_checks as structured_run_parity_checks,
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


def test_decorate_row_for_case_backfills_requested_block_topology_metadata():
    spec = TransformerLayerSpec(
        seq_len=64,
        block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
        block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
        block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
    )
    row = structured_decorate_row_for_case(
        {
            "study_id": "study",
            "backend": "npu",
            "execution_mode": "dataflow",
            "seq_len": 64,
            "batch_size": 1,
            "dtype": "bfloat16",
            "weights_source": "synthetic",
            "warmup_runs": 0,
            "runs_per_sample": 1,
            "measured_inference_count": 1,
            "timed_total_sec": 0.01,
            "avg_latency_ms": 10.0,
        },
        study_id="study",
        case={"case_id": "case", "case_label": "case"},
        spec=spec,
    )

    assert row["block1_topology_id"] == spec.block1_topology_id
    assert row["block1_topology_family"] == "shared_runtime_qkv_proj"
    assert row["block2_topology_id"] == spec.block2_topology_id
    assert row["block2_topology_family"] == "fused_mha_out_proj"
    assert row["block3_topology_id"] == spec.block3_topology_id
    assert row["block3_topology_family"] == "pipelined_addnorm_ffn_addnorm"


def test_failure_result_row_includes_requested_block_topology_metadata():
    spec = TransformerLayerSpec(
        seq_len=64,
        block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
        block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
        block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
    )

    row = structured_failure_result_row(
        study_id="study",
        case={"case_id": "case", "case_label": "case"},
        spec=spec,
        execution_mode="dataflow",
        warmup_runs=0,
        runs_per_sample=1,
        exc=RuntimeError("synthetic failure"),
    )

    assert row["block1_topology_id"] == spec.block1_topology_id
    assert row["block1_topology_family"] == "shared_runtime_qkv_proj"
    assert row["block2_topology_id"] == spec.block2_topology_id
    assert row["block2_topology_family"] == "fused_mha_out_proj"
    assert row["block3_topology_id"] == spec.block3_topology_id
    assert row["block3_topology_family"] == "pipelined_addnorm_ffn_addnorm"


def test_run_parity_checks_forwards_requested_block_topology_ids(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, check):
        captured["command"] = list(command)
        output_csv = Path(command[command.index("--output-csv") + 1])
        output_csv.write_text(
            "execution_mode,seq_len,max_abs_diff,mean_abs_diff\n"
            "dataflow,64,0.0,0.0\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rows = structured_run_parity_checks(
        parity={"seq_lens": [64], "execution_modes": ["dataflow"]},
        cases=[
            {
                "case_id": "case",
                "case_label": "case",
                "layer_spec": {
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                    "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                    "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                },
            }
        ],
        execution_modes=["dataflow"],
        seq_lens=[64],
        benchmark_rows=[
            {
                "study_case_id": "case",
                "execution_mode": "dataflow",
                "seq_len": 64,
                "run_status": "completed",
            }
        ],
        seed=0,
        study_id="study",
        args=SimpleNamespace(
            hidden_size=None,
            intermediate_size=None,
            num_attention_heads=None,
        ),
    )

    command = captured["command"]
    assert "--block1-topology-id" in command
    assert "--block2-topology-id" in command
    assert "--block3-topology-id" in command
    assert rows[0]["study_case_id"] == "case"


def test_resolve_spec_applies_block_topology_id_overrides():
    spec = structured_resolve_spec(
        SimpleNamespace(
            hidden_size=None,
            intermediate_size=None,
            num_attention_heads=None,
            block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
            block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
            block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
        ),
        {"hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12},
        64,
    )

    assert spec.block1_topology_id == "m64_k64_n16_ps1_ph1_pd1"
    assert spec.block2_topology_id == "q32_kv64_e96_ps1_ph1_acc1"
    assert spec.block3_topology_id == "m32_k96_n64_ps4_pi3_d8_g1"
