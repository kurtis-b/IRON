# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import csv
import subprocess
from types import SimpleNamespace

import iron.applications.transformer_layer.src.pipeline.automated_benchmark as structured_automated_benchmark_module
from iron.applications.transformer_layer.automated_benchmark import (
    _record_parity_results,
)
from iron.applications.transformer_layer.benchmark_common import (
    load_study_manifest,
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
from iron.applications.transformer_layer.src.pipeline.automated_benchmark import (
    run_manifest_benchmark as structured_run_manifest_benchmark,
)
from iron.applications.transformer_layer.src.pipeline import (
    _record_parity_results as structured_record_parity_results,
)


def test_restructured_bench_package_preserves_legacy_imports():
    assert parse_seq_lens is parse_seq_lens_structured
    assert empty_power_stats is empty_power_stats_structured
    assert append_debug_event is append_debug_event_structured
    assert _record_parity_results is structured_record_parity_results


def test_checked_in_dataflow_manifests_pin_retained_block_topologies():
    app_dir = Path(__file__).resolve().parent
    manifest_specs = {
        "design_patterns_end_to_end.json": {
            "baseline_768": {
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
            },
            "baseline_1024": {
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block2_topology_id": "q32_kv64_e128_ps1_ph1_acc1",
                "block3_topology_id": "m32_k128_n32_ps4_pi2_d8_g1",
            },
        },
        "dataflow_blocks.json": {
            "baseline_768": {
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
            },
            "baseline_1024": {
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block2_topology_id": "q32_kv64_e128_ps1_ph1_acc1",
                "block3_topology_id": "m32_k128_n32_ps4_pi2_d8_g1",
            },
        },
    }

    for manifest_name, expected_cases in manifest_specs.items():
        manifest = load_study_manifest(app_dir / "study" / manifest_name)
        by_case_id = {
            case["case_id"]: case["layer_spec"] for case in manifest["study_cases"]
        }
        for case_id, expected in expected_cases.items():
            for key, value in expected.items():
                assert by_case_id[case_id][key] == value


def test_load_study_manifest_normalizes_and_validates_layer_specs(tmp_path: Path):
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(
        """
        {
          "study_id": "study",
          "execution_modes": ["dataflow"],
          "seq_lens": [64],
          "layer_spec": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1"
          }
        }
        """,
        encoding="utf-8",
    )

    manifest = load_study_manifest(manifest_path)

    assert manifest["layer_spec"]["seq_len"] == 128
    assert manifest["layer_spec"]["batch_size"] == 1
    assert manifest["layer_spec"]["block1_topology_id"] == "m64_k64_n16_ps1_ph1_pd1"


def test_load_study_manifest_rejects_invalid_layer_specs(tmp_path: Path):
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(
        """
        {
          "study_id": "study",
          "execution_modes": ["dataflow"],
          "seq_lens": [64],
          "study_cases": [
            {
              "case_id": "invalid",
              "layer_spec": {
                "hidden_size": 769,
                "intermediate_size": 3072,
                "num_attention_heads": 12
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    try:
        load_study_manifest(manifest_path)
    except ValueError as exc:
        assert "hidden_size must be divisible by num_attention_heads" in str(exc)
    else:
        raise AssertionError("Expected invalid study-case layer_spec to be rejected")


def test_load_study_manifest_expands_practical_topology_exploration(tmp_path: Path):
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(
        """
        {
          "study_id": "study",
          "execution_modes": ["dataflow"],
          "seq_lens": [64],
          "layer_spec": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12
          },
          "topology_exploration": {
            "surface": "practical",
            "max_block1_candidates": 2,
            "max_block2_candidates": 1,
            "max_block3_candidates": 2
          }
        }
        """,
        encoding="utf-8",
    )

    manifest = load_study_manifest(manifest_path)

    assert manifest["topology_exploration"]["surface"] == "practical"
    assert len(manifest["study_cases"]) == 4
    assert manifest["study_cases"][0]["case_id"] == "practical_000"
    assert manifest["study_cases"][0]["layer_spec"]["seq_len"] == 64
    saw_distinct_exploration_provenance = False
    for case in manifest["study_cases"]:
        assert case["exploration_block1_topology_id"] is not None
        assert case["exploration_block2_topology_id"] is not None
        assert case["exploration_block3_topology_id"] is not None
        assert case["layer_spec"]["block1_topology_id"] is not None
        assert case["layer_spec"]["block2_topology_id"] is not None
        assert case["layer_spec"]["block3_topology_id"] is not None
        assert not str(case["layer_spec"]["block3_topology_id"]).startswith("cr")
        assert str(case["exploration_block3_topology_id"]).startswith("cr")
        if (
            case["exploration_block1_topology_id"]
            != case["layer_spec"]["block1_topology_id"]
            or case["exploration_block2_topology_id"]
            != case["layer_spec"]["block2_topology_id"]
            or case["exploration_block3_topology_id"]
            != case["layer_spec"]["block3_topology_id"]
        ):
            saw_distinct_exploration_provenance = True
    assert saw_distinct_exploration_provenance


def test_load_study_manifest_rejects_multiseq_topology_exploration(tmp_path: Path):
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(
        """
        {
          "study_id": "study",
          "execution_modes": ["dataflow"],
          "seq_lens": [64, 128],
          "layer_spec": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12
          },
          "topology_exploration": {
            "surface": "practical",
            "max_block1_candidates": 2
          }
        }
        """,
        encoding="utf-8",
    )

    try:
        load_study_manifest(manifest_path)
    except ValueError as exc:
        assert "exactly one seq_len" in str(exc)
    else:
        raise AssertionError("Expected multiseq topology_exploration to be rejected")


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


def test_write_results_csv_reserves_topology_columns(tmp_path: Path):
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
                "exploration_block1_topology_id": "m32_k256_n24_c8_ps2_ph1_pd4",
                "exploration_block1_topology_family": "shared_runtime_qkv_proj_practical",
                "exploration_block2_topology_id": "q32_kv64_e96_ps1_ph6_acc1",
                "exploration_block2_topology_family": "fused_mha_out_proj_practical",
                "exploration_block3_topology_id": "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1",
                "exploration_block3_topology_family": "pipelined_addnorm_ffn_addnorm_practical",
                "reference_npu_execution_mode": "dataflow",
                "reference_npu_avg_latency_ms": 9.5,
                "reference_npu_block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "reference_npu_block1_topology_family": "shared_runtime_qkv_proj",
                "reference_npu_block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "reference_npu_block2_topology_family": "fused_mha_out_proj",
                "reference_npu_block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "reference_npu_block3_topology_family": "pipelined_addnorm_ffn_addnorm",
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
        "exploration_block1_topology_id",
        "exploration_block1_topology_family",
        "exploration_block2_topology_id",
        "exploration_block2_topology_family",
        "exploration_block3_topology_id",
        "exploration_block3_topology_family",
        "reference_npu_execution_mode",
        "reference_npu_avg_latency_ms",
        "reference_npu_block1_topology_id",
        "reference_npu_block1_topology_family",
        "reference_npu_block2_topology_id",
        "reference_npu_block2_topology_family",
        "reference_npu_block3_topology_id",
        "reference_npu_block3_topology_family",
    ):
        assert field in header

    assert header.index("process_model") < header.index("block1_topology_id")
    assert header.index("block3_topology_family") < header.index(
        "exploration_block1_topology_id"
    )
    assert header.index("exploration_block3_topology_family") < header.index(
        "reference_npu_execution_mode"
    )
    assert header.index("reference_npu_block3_topology_family") < header.index(
        "run_status"
    )
    assert RESULT_FIELD_ORDER.index("block1_topology_id") < RESULT_FIELD_ORDER.index(
        "run_status"
    )
    assert RESULT_FIELD_ORDER.index(
        "exploration_block1_topology_id"
    ) < RESULT_FIELD_ORDER.index("run_status")
    assert RESULT_FIELD_ORDER.index(
        "reference_npu_execution_mode"
    ) < RESULT_FIELD_ORDER.index("run_status")


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
        case={
            "case_id": "case",
            "case_label": "case",
            "exploration_block1_topology_id": "m32_k256_n24_c8_ps2_ph1_pd4",
            "exploration_block1_topology_family": "shared_runtime_qkv_proj_practical",
            "exploration_block2_topology_id": "q32_kv64_e96_ps1_ph6_acc1",
            "exploration_block2_topology_family": "fused_mha_out_proj_practical",
            "exploration_block3_topology_id": "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1",
            "exploration_block3_topology_family": "pipelined_addnorm_ffn_addnorm_practical",
        },
        spec=spec,
    )

    assert row["block1_topology_id"] == spec.block1_topology_id
    assert row["block1_topology_family"] == "shared_runtime_qkv_proj"
    assert row["block2_topology_id"] == spec.block2_topology_id
    assert row["block2_topology_family"] == "fused_mha_out_proj"
    assert row["block3_topology_id"] == spec.block3_topology_id
    assert row["block3_topology_family"] == "pipelined_addnorm_ffn_addnorm"
    assert row["exploration_block1_topology_id"] == "m32_k256_n24_c8_ps2_ph1_pd4"
    assert row["exploration_block2_topology_id"] == "q32_kv64_e96_ps1_ph6_acc1"
    assert row["exploration_block3_topology_id"] == "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1"


def test_failure_result_row_includes_requested_block_topology_metadata():
    spec = TransformerLayerSpec(
        seq_len=64,
        block1_topology_id="m64_k64_n16_ps1_ph1_pd1",
        block2_topology_id="q32_kv64_e96_ps1_ph1_acc1",
        block3_topology_id="m32_k96_n64_ps4_pi3_d8_g1",
    )

    row = structured_failure_result_row(
        study_id="study",
        case={
            "case_id": "case",
            "case_label": "case",
            "exploration_block1_topology_id": "m32_k256_n24_c8_ps2_ph1_pd4",
            "exploration_block2_topology_id": "q32_kv64_e96_ps1_ph6_acc1",
            "exploration_block3_topology_id": "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1",
        },
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
    assert row["exploration_block1_topology_id"] == "m32_k256_n24_c8_ps2_ph1_pd4"
    assert row["exploration_block2_topology_id"] == "q32_kv64_e96_ps1_ph6_acc1"
    assert row["exploration_block3_topology_id"] == "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1"


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
                "exploration_block1_topology_id": "m32_k256_n24_c8_ps2_ph1_pd4",
                "exploration_block1_topology_family": "shared_runtime_qkv_proj_practical",
                "exploration_block2_topology_id": "q32_kv64_e96_ps1_ph6_acc1",
                "exploration_block2_topology_family": "fused_mha_out_proj_practical",
                "exploration_block3_topology_id": "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1",
                "exploration_block3_topology_family": "pipelined_addnorm_ffn_addnorm_practical",
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
    assert rows[0]["exploration_block1_topology_id"] == "m32_k256_n24_c8_ps2_ph1_pd4"
    assert rows[0]["exploration_block2_topology_id"] == "q32_kv64_e96_ps1_ph6_acc1"
    assert (
        rows[0]["exploration_block3_topology_id"]
        == "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1"
    )


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


def test_run_manifest_benchmark_records_timeout_rows_when_continue_on_error(
    tmp_path: Path, monkeypatch
):
    manifest_path = tmp_path / "study.json"
    output_csv = tmp_path / "results.csv"
    debug_log_csv = tmp_path / "debug.csv"
    manifest_path.write_text(
        """
        {
          "study_id": "study",
          "study_cases": [
            {
              "case_id": "baseline_768",
              "case_label": "baseline_768",
              "layer_spec": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12
              }
            }
          ],
          "execution_modes": ["dataflow", "runlist"],
          "seq_lens": [16384],
          "continue_on_error": true,
          "output_csv": "results.csv",
          "debug_log_csv": "debug.csv"
        }
        """,
        encoding="utf-8",
    )

    def fake_benchmark_pattern(**kwargs):
        if kwargs["execution_mode"] == "dataflow":
            raise RuntimeError("runlist failed execution (ERT_CMD_STATE_TIMEOUT)")
        return [
            {
                "backend": "npu",
                "execution_mode": kwargs["execution_mode"],
                "pattern_label": kwargs["execution_mode"],
                "seq_len": kwargs["spec"].seq_len,
                "batch_size": kwargs["spec"].batch_size,
                "dtype": kwargs["spec"].dtype,
                "weights_source": kwargs["spec"].weights_source,
                "warmup_runs": kwargs["warmup_runs"],
                "runs_per_sample": kwargs["runs_per_sample"],
                "measured_inference_count": 1,
                "timed_total_sec": 0.01,
                "avg_latency_ms": 10.0,
            }
        ]

    monkeypatch.setattr(
        structured_automated_benchmark_module,
        "benchmark_pattern",
        fake_benchmark_pattern,
    )

    structured_run_manifest_benchmark(
        SimpleNamespace(
            study_manifest=str(manifest_path),
            execution_modes="dataflow,runlist,gemm_offload",
            seq_lens="64,128,256,512",
            output_csv="transformer_layer_npu_suite.csv",
            debug_log_csv=None,
            peak_reference=None,
            annotated_output_csv=None,
            warmup_runs=None,
            runs_per_sample=None,
            seed=0,
            power_backend="none",
            power_sample_interval_sec=0.05,
            quiescent_baseline_duration_sec=0.5,
            run_parity_check=False,
            skip_parity_check=False,
            parity_output_csv=None,
            hidden_size=None,
            intermediate_size=None,
            num_attention_heads=None,
            block1_topology_id=None,
            block2_topology_id=None,
            block3_topology_id=None,
            enable_measurement_log=False,
            measurement_log_path=None,
        )
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    failed_row = next(row for row in rows if row["execution_mode"] == "dataflow")
    completed_row = next(row for row in rows if row["execution_mode"] == "runlist")

    assert failed_row["run_status"] == "failed"
    assert failed_row["failure_component"] == "runtime_execution"
    assert failed_row["failure_category"] == "runtime_timeout"
    assert completed_row["run_status"] == "completed"

    debug_text = debug_log_csv.read_text(encoding="utf-8")
    assert "benchmark_case_failed" in debug_text
    assert "study_completed" in debug_text
