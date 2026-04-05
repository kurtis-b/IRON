#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer_new.pattern.reference import (
    derive_offload_inputs,
    generate_golden_reference,
)
from iron.applications.transformer_layer_new.study.block.cases import BLOCK_CASES
from iron.applications.transformer_layer_new.study.end_to_end.cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    MODE_OPERATORS,
    SEQUENCE_LADDER,
    candidate_table_for_case,
    default_candidates_path,
    effective_dense_layer_flop_count,
    effective_gflops_per_sec,
    effective_gflops_per_sec_per_watt,
    get_case,
    iter_cases,
    load_candidate_payload,
    load_default_candidate_payloads,
)
from iron.applications.transformer_layer_new.study.end_to_end.modes import (
    DEFAULT_MIN_POWER_MEASUREMENT_DURATION_SEC,
    FINAL_ERROR_THRESHOLD,
    _benchmark_offload_shared_gemm,
    _elementwise_mul_buffers,
    _metadata_for_operator,
    _validate_output,
    benchmark_operator_candidate,
)
from iron.applications.transformer_layer_new.study.end_to_end.run import (
    build_rows,
    iteration_schedule,
    json_dumps,
    main,
    mark_best_rows,
    tune_mode,
)
from iron.applications.transformer_layer_new.study.end_to_end.power import (
    resolve_power_sample_interval_sec,
    resolve_power_probe_runs,
)
from iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep import (
    default_output_path as default_power_sweep_output_path,
    default_tuning_output_path as default_power_sweep_tuning_output_path,
    iter_selected_cases,
    main as power_sweep_main,
    selected_sequence_ladder,
)


def test_case_table_covers_retained_surface():
    assert tuple(FAMILY_SPECS) == FAMILY_IDS
    assert len(iter_cases()) == len(FAMILY_IDS) * len(SEQUENCE_LADDER)

    for family_id in FAMILY_IDS:
        for seq_len in SEQUENCE_LADDER:
            case = get_case(family_id, seq_len)
            assert case.study_case_id == family_id
            assert case.seq_len == seq_len
            assert case.attention_head_size == (
                case.hidden_size // case.num_attention_heads
            )


def test_power_sweep_selects_ladder_prefix():
    assert selected_sequence_ladder(64) == (64,)
    assert selected_sequence_ladder(256) == (64, 128, 256)
    assert selected_sequence_ladder(16384) == SEQUENCE_LADDER


def test_power_sweep_iter_selected_cases_respects_family_and_max_seq_len():
    cases = iter_selected_cases("baseline_768", max_seq_len=256)
    assert tuple(case.study_case_id for case in cases) == ("baseline_768",) * 3
    assert tuple(case.seq_len for case in cases) == (64, 128, 256)


def test_power_sweep_default_paths_live_in_results_directory():
    assert default_power_sweep_output_path(256).name == "results_upto256_power.csv"
    assert default_power_sweep_output_path(256).parent.name == "end_to_end"
    assert default_power_sweep_output_path(256).parent.parent.name == "results"
    assert (
        default_power_sweep_tuning_output_path(256).name == "tuning_upto256_power.csv"
    )
    assert default_power_sweep_tuning_output_path(256).parent.name == "end_to_end"
    assert default_power_sweep_tuning_output_path(256).parent.parent.name == "results"


def test_power_sweep_full_ladder_defaults_to_all_power_filenames():
    assert default_power_sweep_output_path(16384).name == "results_all_power.csv"
    assert default_power_sweep_tuning_output_path(16384).name == "tuning_all_power.csv"


def test_power_sampling_defaults_to_conservative_100ms_floor():
    assert (
        resolve_power_sample_interval_sec(
            requested_interval_sec=0.1,
            estimated_timed_window_sec=0.03,
        )
        == 0.1
    )


def test_power_probe_runs_target_full_second_window_for_short_cases():
    assert (
        resolve_power_probe_runs(
            avg_iteration_sec=0.042,
            baseline_runs=10,
            min_measurement_duration_sec=DEFAULT_MIN_POWER_MEASUREMENT_DURATION_SEC,
        )
        == 24
    )


def test_default_candidate_file_covers_all_modes_and_operators():
    payloads = load_default_candidate_payloads()
    for family_id in FAMILY_IDS:
        for seq_len in (64, 2048):
            table = candidate_table_for_case(family_id, seq_len, payloads=payloads)
            assert set(table) == set(EXECUTION_MODES)
            for mode in EXECUTION_MODES:
                assert set(table[mode]) == set(MODE_OPERATORS[mode])
                assert all(table[mode][operator] for operator in MODE_OPERATORS[mode])


def test_candidate_table_merges_family_defaults_with_seq_overrides(tmp_path):
    payloads = {
        "dataflow": {
            "baseline_768": {
                "all": {
                    "qkv_proj": [{"candidate_id": "default", "config": {"tile_m": 32}}],
                    "mha_out_proj": [
                        {"candidate_id": "default", "config": {"parallel_seq": 1}}
                    ],
                    "add_norm1": [
                        {"candidate_id": "default", "config": {"tile_size": 768}}
                    ],
                    "ffn": [{"candidate_id": "default", "config": {"tile_m": 64}}],
                    "add_norm2": [
                        {"candidate_id": "default", "config": {"tile_size": 768}}
                    ],
                },
                "64": {
                    "qkv_proj": [{"candidate_id": "override", "config": {"tile_m": 16}}]
                },
            },
            "baseline_1024": {
                "all": {
                    operator: [{"candidate_id": "default", "config": {}}]
                    for operator in MODE_OPERATORS["dataflow"]
                }
            },
        },
        "runlist": {
            family_id: {
                "all": {
                    operator: [{"candidate_id": "default", "config": {}}]
                    for operator in MODE_OPERATORS["runlist"]
                }
            }
            for family_id in FAMILY_IDS
        },
        "offload": {
            family_id: {
                "all": {"shared_gemm": [{"candidate_id": "default", "config": {}}]}
            }
            for family_id in FAMILY_IDS
        },
    }

    table = candidate_table_for_case("baseline_768", 64, payloads=payloads)
    assert table["dataflow"]["qkv_proj"][0]["candidate_id"] == "override"
    assert table["dataflow"]["ffn"][0]["candidate_id"] == "default"
    assert table["runlist"]["qkvo_proj"][0]["candidate_id"] == "default"


def test_split_candidate_files_define_low_sequence_overrides_and_fallbacks():
    payloads = load_default_candidate_payloads()
    for execution_mode in EXECUTION_MODES:
        payload = load_candidate_payload(
            execution_mode,
            default_candidates_path(execution_mode),
        )
        for family_id in FAMILY_IDS:
            assert "all" in payload[family_id]
            if execution_mode in {"dataflow", "runlist", "offload"}:
                for seq_key in ("64", "128"):
                    assert seq_key in payload[family_id]
            if execution_mode == "offload":
                for seq_key in ("256", "512"):
                    assert seq_key in payload[family_id]
            if execution_mode in {"dataflow", "runlist"}:
                assert "256" in payload[family_id]
            assert candidate_table_for_case(family_id, 256, payloads=payloads)[
                execution_mode
            ]
            assert candidate_table_for_case(family_id, 512, payloads=payloads)[
                execution_mode
            ]
            assert candidate_table_for_case(family_id, 16384, payloads=payloads)[
                execution_mode
            ]


def test_default_candidate_files_exclude_failed_canonical_candidates():
    tuning_csv = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "end_to_end"
        / "tuning_all_power.csv"
    )
    with tuning_csv.open(newline="", encoding="utf-8") as handle:
        failed_candidates = {
            (
                row["study_case_id"],
                row["seq_len"],
                row["execution_mode"],
                row["internal_operator"],
                row["candidate_id"],
            )
            for row in csv.DictReader(handle)
            if row["run_status"].startswith("failed")
        }

    payloads = load_default_candidate_payloads()
    for family_id in FAMILY_IDS:
        for seq_len in SEQUENCE_LADDER:
            table = candidate_table_for_case(family_id, seq_len, payloads=payloads)
            for execution_mode in EXECUTION_MODES:
                for operator_name in MODE_OPERATORS[execution_mode]:
                    for candidate in table[execution_mode][operator_name]:
                        assert (
                            family_id,
                            str(seq_len),
                            execution_mode,
                            operator_name,
                            candidate["candidate_id"],
                        ) not in failed_candidates


def test_derive_offload_inputs_matches_reference_shapes():
    reference = generate_golden_reference(64, 768, 3072, 12)
    inputs = derive_offload_inputs(reference, num_heads=12)
    assert tuple(inputs["q"].shape) == (12, 64, 64)
    assert tuple(inputs["k"].shape) == (12, 64, 64)
    assert tuple(inputs["v"].shape) == (12, 64, 64)
    assert tuple(inputs["residual"].shape) == (64, 768)


def _expected_dataflow_config_rows(
    block_case, operator_name: str
) -> list[dict[str, object]]:
    if operator_name == "qkv_proj":
        return [
            {
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "parallel_seq": parallel_seq,
                "parallel_emb": parallel_emb,
            }
            for tile_m, tile_k, tile_n, parallel_seq, parallel_emb in block_case.qkv_proj
        ]
    if operator_name == "mha_out_proj":
        return [
            {
                "parallel_seq": parallel_seq,
                "q_seq_tile": q_seq_tile,
                "kv_seq_tile": kv_seq_tile,
                "emb_tile": emb_tile,
                "parallel_heads": parallel_heads,
                "o_proj_acc_depth": o_proj_acc_depth,
            }
            for (
                parallel_seq,
                q_seq_tile,
                kv_seq_tile,
                emb_tile,
                parallel_heads,
                o_proj_acc_depth,
            ) in block_case.mha_out_proj
        ]
    if operator_name in {"add_norm1", "add_norm2"}:
        return [
            {
                "num_aie_columns": num_aie_columns,
                "tile_size": tile_size,
            }
            for num_aie_columns, tile_size in block_case.addnorm
        ]
    if operator_name == "ffn":
        return [
            {
                "num_aie_columns": num_aie_columns,
                "b_col_maj": b_col_maj,
                "c_col_maj": c_col_maj,
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "down_proj_depth": down_proj_depth,
                "n_a_tiles_distributed": n_a_tiles_distributed,
                "n_b_tiles_distributed": n_b_tiles_distributed,
                "stage_only": stage_only,
                "gelu_stage": gelu_stage,
            }
            for (
                num_aie_columns,
                b_col_maj,
                c_col_maj,
                tile_m,
                tile_k,
                tile_n,
                down_proj_depth,
                n_a_tiles_distributed,
                n_b_tiles_distributed,
                stage_only,
                gelu_stage,
            ) in block_case.ffn
        ]
    raise ValueError(operator_name)


def test_dataflow_candidate_file_matches_block_study_configs():
    payload = load_candidate_payload(
        "dataflow",
        default_candidates_path("dataflow"),
    )
    payloads = {
        "dataflow": payload,
        "runlist": load_candidate_payload(
            "runlist",
            default_candidates_path("runlist"),
        ),
        "offload": load_candidate_payload(
            "offload",
            default_candidates_path("offload"),
        ),
    }

    for family_id in FAMILY_IDS:
        family_cases = BLOCK_CASES[family_id]
        for seq_len in (64, 128, 256, 512):
            seq_payload = candidate_table_for_case(
                family_id,
                seq_len,
                payloads=payloads,
            )["dataflow"]
            block_case = family_cases[seq_len]
            for operator_name in MODE_OPERATORS["dataflow"]:
                actual = [
                    candidate["config"] for candidate in seq_payload[operator_name]
                ]
                expected = _expected_dataflow_config_rows(block_case, operator_name)
                if (
                    family_id == "baseline_768"
                    and seq_len == 64
                    and operator_name == "qkv_proj"
                ):
                    # Keep the smaller qkv surface for the powered end-to-end study,
                    # where the more aggressive qkv_3 candidate was unstable.
                    expected = expected[:-1]
                assert actual == expected

        all_payload = payload[family_id]["all"]
        block_case = family_cases[1024]
        for operator_name in MODE_OPERATORS["dataflow"]:
            actual = [candidate["config"] for candidate in all_payload[operator_name]]
            expected = _expected_dataflow_config_rows(block_case, operator_name)
            assert actual == expected


def test_iteration_schedule_matches_end_to_end_defaults():
    assert iteration_schedule(64) == (1, 10)
    assert iteration_schedule(128) == (1, 10)
    assert iteration_schedule(256) == (1, 10)
    assert iteration_schedule(512) == (1, 10)
    assert iteration_schedule(1024) == (1, 10)
    assert iteration_schedule(2048) == (1, 10)
    assert iteration_schedule(4096) == (1, 5)
    assert iteration_schedule(8192) == (1, 2)
    assert iteration_schedule(16384) == (1, 2)


def test_generate_golden_reference_can_skip_large_output():
    reference = generate_golden_reference(
        1024,
        768,
        3072,
        12,
        include_output=False,
        include_attention_mask=False,
    )
    assert reference["input"] is not None
    assert reference["weights"] is not None
    assert reference["output"] is None
    assert reference["attention_mask"] is None


def test_validate_output_uses_pattern_threshold():
    case = get_case("baseline_768", 64)
    reference_output = generate_golden_reference(
        case.seq_len,
        case.hidden_size,
        case.intermediate_size,
        case.num_attention_heads,
    )["output"]

    assert reference_output is not None

    passing = _validate_output(case.workload, reference_output, reference_output)
    assert passing["validation_error_count"] == 0
    assert passing["run_status"] == "passed"

    broken = reference_output.clone()
    max_acceptable_errors = int(case.seq_len * case.hidden_size * FINAL_ERROR_THRESHOLD)
    broken.reshape(-1)[: max_acceptable_errors + 1] = 10
    failing = _validate_output(case.workload, broken, reference_output)
    assert failing["run_status"] == "failed_validation"


def test_validate_output_uses_finite_check_without_reference():
    tensor = generate_golden_reference(
        64,
        1024,
        4096,
        16,
    )[
        "input"
    ][:1, :]
    case = get_case("baseline_1024", 2048)

    finite = _validate_output(case.workload, tensor, None)
    assert finite["run_status"] == "passed"

    non_finite_tensor = tensor.clone()
    non_finite_tensor.reshape(-1)[0] = float("nan")
    failing = _validate_output(case.workload, non_finite_tensor, None)
    assert failing["run_status"] == "failed_validation"
    assert failing["validation_error_count"] == 1


def test_metadata_helpers_count_runtime_artifacts():
    class FakeArtifact:
        pass

    shared_xclbin = FakeArtifact()

    class FakeOperator:
        def __init__(self):
            self.runlist = [1, 2, 3]
            self.q_xclbin = FakeArtifact()
            self.k_xclbin = FakeArtifact()
            self.combined_xclbin = self.k_xclbin
            self.q_insts = FakeArtifact()
            self.k_insts = FakeArtifact()
            self.gemm_ops = [
                ("a", type("Gemm", (), {"insts_artifact": FakeArtifact()})()),
                ("b", type("Gemm", (), {"insts_artifact": FakeArtifact()})()),
            ]
            self.shared_xclbin_artifact = shared_xclbin

    workload = get_case("baseline_768", 64).workload
    dataflow_metadata = _metadata_for_operator(
        "dataflow",
        FakeOperator(),
        workload,
        compile_setup_time_ms=1.5,
    )
    assert dataflow_metadata["npu_dispatch_count"] == 3
    assert dataflow_metadata["npu_unique_instruction_binary_count"] == 2
    assert dataflow_metadata["npu_unique_xclbin_count"] == 2

    offload_metadata = _metadata_for_operator(
        "offload",
        FakeOperator(),
        workload,
        compile_setup_time_ms=1.5,
    )
    assert (
        offload_metadata["npu_dispatch_count"] == (2 * workload.num_attention_heads) + 6
    )
    assert offload_metadata["npu_unique_instruction_binary_count"] == 2
    assert offload_metadata["npu_unique_xclbin_count"] == 1


def test_effective_gflops_helpers():
    assert (
        effective_dense_layer_flop_count(
            seq_len=512,
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
        )
        == 8_053_063_680
    )
    effective_gflops = effective_gflops_per_sec(
        seq_len=512,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        avg_latency_ms=2.0,
    )
    assert effective_gflops == 4026.53184
    assert effective_gflops_per_sec_per_watt(effective_gflops, 50.0) == 80.5306368
    assert effective_gflops_per_sec_per_watt(effective_gflops, None) is None


def test_elementwise_mul_buffers_support_scalar_broadcast():
    input_buffers, output_buffers = _elementwise_mul_buffers(
        {
            "size": 16,
            "scalar_broadcast": 0.5,
        },
        seed=42,
    )
    assert set(input_buffers) == {"input1"}
    assert output_buffers["output"].shape == input_buffers["input1"].shape
    assert output_buffers["output"].equal(input_buffers["input1"] * 0.5)


def test_offload_long_sequence_tuning_uses_pattern_path(monkeypatch):
    case = get_case("baseline_768", 16384)

    called = {}

    def fake_long_seq(
        workload, candidate_config, *, warmup_runs, runs_per_sample, seed
    ):
        called["workload"] = workload
        called["candidate_config"] = candidate_config
        called["warmup_runs"] = warmup_runs
        called["runs_per_sample"] = runs_per_sample
        called["seed"] = seed
        return {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": "",
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.modes._benchmark_offload_long_seq_candidate",
        fake_long_seq,
    )

    result = _benchmark_offload_shared_gemm(
        case.workload,
        {"tile_m": 64, "tile_k": 64, "tile_n": 16, "num_aie_columns": 4},
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert called["workload"] == case.workload
    assert called["warmup_runs"] == 1
    assert called["runs_per_sample"] == 1
    assert called["seed"] == 42


def test_mark_best_rows_marks_fastest_successful_mode():
    rows = [
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "avg_latency_ms": 2.0,
            "run_status": "passed",
        },
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "avg_latency_ms": 1.0,
            "run_status": "passed",
        },
        {
            "study_case_id": "baseline_768",
            "seq_len": 64,
            "avg_latency_ms": None,
            "run_status": "failed_exception",
        },
    ]
    mark_best_rows(rows)
    assert rows[0]["is_best"] is False
    assert rows[1]["is_best"] is True
    assert rows[2]["is_best"] is False


def test_tune_mode_selects_fastest_passing_candidate(monkeypatch):
    case = get_case("baseline_768", 64)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.candidate_table_for_case",
        lambda family_id, seq_len: {
            "dataflow": {
                "qkv_proj": [
                    {"candidate_id": "slow", "config": {"tile_m": 32}},
                    {"candidate_id": "fast", "config": {"tile_m": 16}},
                ],
                "mha_out_proj": [
                    {"candidate_id": "only", "config": {"parallel_seq": 1}}
                ],
                "add_norm1": [{"candidate_id": "only", "config": {"tile_size": 768}}],
                "ffn": [{"candidate_id": "only", "config": {"tile_m": 16}}],
                "add_norm2": [{"candidate_id": "only", "config": {"tile_size": 768}}],
            },
            "runlist": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["runlist"]
            },
            "offload": {"shared_gemm": [{"candidate_id": "only", "config": {}}]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, config: {
            next(iter(config)): {"resolved": next(iter(config.values()))}
        },
    )

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        candidate_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        if operator_name == "qkv_proj":
            latency = 2.0 if candidate_config["tile_m"] == 32 else 1.0
        else:
            latency = 3.0
        return {
            "avg_latency_ms": latency,
            "bandwidth_gbps": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )

    tuning_rows, selected_candidate_ids, selected_config, tuning_failure = tune_mode(
        case,
        execution_mode="dataflow",
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert tuning_failure == ""
    assert selected_candidate_ids["qkv_proj"] == "fast"
    assert selected_config["qkv_proj"] == {"resolved": {"tile_m": 16}}
    qkv_rows = [row for row in tuning_rows if row["internal_operator"] == "qkv_proj"]
    assert [row for row in qkv_rows if row["is_operator_best"] is True][0][
        "candidate_id"
    ] == "fast"


def test_tune_mode_runlist_long_seq_skips_singleton_defaults_and_smokes(monkeypatch):
    case = get_case("baseline_768", 8192)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.candidate_table_for_case",
        lambda family_id, seq_len: {
            "dataflow": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["dataflow"]
            },
            "runlist": {
                "qkvo_proj": [
                    {"candidate_id": "slow", "config": {"tile_m": 64}},
                    {"candidate_id": "fast", "config": {"tile_m": 32}},
                ],
                "k_transpose": [{"candidate_id": "transpose", "config": {"m": 64}}],
                **{
                    operator: [{"candidate_id": "only", "config": {}}]
                    for operator in MODE_OPERATORS["runlist"]
                    if operator not in {"qkvo_proj", "k_transpose"}
                },
            },
            "offload": {"shared_gemm": [{"candidate_id": "only", "config": {}}]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, config: {
            next(iter(config)): {"resolved": next(iter(config.values()))}
        },
    )

    benchmarked = []

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        candidate_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        benchmarked.append(operator_name)
        latency = 2.0 if candidate_config.get("tile_m") == 64 else 1.0
        return {
            "avg_latency_ms": latency,
            "bandwidth_gbps": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    smoke_calls = []

    def fake_benchmark_mode(
        execution_mode,
        workload,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
        operator_config,
    ):
        smoke_calls.append((execution_mode, workload.seq_len, operator_config))
        return {
            "timed_total_sec": 0.5,
            "measured_inference_count": runs_per_sample,
            "avg_latency_ms": 5.0,
            "compile_setup_time_ms": 100.0,
            "host_qkv_precompute_ms": None,
            "effective_gflops_per_sec": 1.0,
            "power_backend": power_backend,
            "avg_power_w": None,
            "effective_gflops_per_sec_per_watt": None,
            "npu_dispatch_count": 0,
            "npu_unique_instruction_binary_count": 0,
            "npu_unique_xclbin_count": 0,
            "process_model": "in_process",
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_mode",
        fake_benchmark_mode,
    )

    tuning_rows, selected_candidate_ids, selected_config, tuning_failure = tune_mode(
        case,
        execution_mode="runlist",
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert tuning_failure == ""
    assert benchmarked == ["qkvo_proj", "qkvo_proj"]
    assert selected_candidate_ids["qkvo_proj"] == "fast"
    assert selected_candidate_ids["k_transpose"] == "transpose"
    assert selected_candidate_ids["attn_scores"] == "only"
    skipped = [
        row
        for row in tuning_rows
        if row["internal_operator"] in {"k_transpose", "attn_scores"}
        and row["run_status"] == "skipped_long_seq_default"
    ]
    assert len(skipped) == 2
    assert smoke_calls == [("runlist", 8192, selected_config)]


def test_tune_mode_runlist_long_seq_smoke_failure_returns_tuning_failure(monkeypatch):
    case = get_case("baseline_768", 8192)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.candidate_table_for_case",
        lambda family_id, seq_len: {
            "dataflow": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["dataflow"]
            },
            "runlist": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["runlist"]
            },
            "offload": {"shared_gemm": [{"candidate_id": "only", "config": {}}]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, config: {
            next(iter(config)): {"resolved": next(iter(config.values()))}
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_operator_candidate",
        lambda *args, **kwargs: {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_mode",
        lambda *args, **kwargs: {
            "run_status": "failed_exception",
            "failure_message": "timeout",
        },
    )

    _, _, _, tuning_failure = tune_mode(
        case,
        execution_mode="runlist",
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert (
        tuning_failure
        == "tuning_failed: long-sequence runlist mode smoke failed: timeout"
    )


def test_tune_mode_runlist_short_seq_skips_singleton_defaults(monkeypatch):
    case = get_case("baseline_768", 64)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.candidate_table_for_case",
        lambda family_id, seq_len: {
            "dataflow": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["dataflow"]
            },
            "runlist": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["runlist"]
            },
            "offload": {"shared_gemm": [{"candidate_id": "only", "config": {}}]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, config: {
            next(iter(config)): {"resolved": next(iter(config.values()))}
        },
    )

    benchmarked = []

    def fake_benchmark_operator_candidate(*args, **kwargs):
        benchmarked.append("called")
        return {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )

    tuning_rows, selected_candidate_ids, selected_config, tuning_failure = tune_mode(
        case,
        execution_mode="runlist",
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert tuning_failure == ""
    assert benchmarked == []
    assert set(selected_candidate_ids) == set(MODE_OPERATORS["runlist"])
    assert set(selected_config) == set(MODE_OPERATORS["runlist"])
    assert all(row["run_status"] == "skipped_singleton_default" for row in tuning_rows)


def test_tune_mode_dataflow_singletons_are_still_benchmarked(monkeypatch):
    case = get_case("baseline_768", 64)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.candidate_table_for_case",
        lambda family_id, seq_len: {
            "dataflow": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["dataflow"]
            },
            "runlist": {
                operator: [{"candidate_id": "only", "config": {}}]
                for operator in MODE_OPERATORS["runlist"]
            },
            "offload": {"shared_gemm": [{"candidate_id": "only", "config": {}}]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, config: {
            next(iter(config)): {"resolved": next(iter(config.values()))}
        },
    )

    benchmarked = []

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        candidate_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        benchmarked.append(operator_name)
        return {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )

    tuning_rows, _, _, tuning_failure = tune_mode(
        case,
        execution_mode="dataflow",
        warmup_runs=1,
        runs_per_sample=1,
        seed=42,
    )

    assert tuning_failure == ""
    assert benchmarked == list(MODE_OPERATORS["dataflow"])
    assert all(row["run_status"] == "passed" for row in tuning_rows)


def test_build_rows_returns_tuning_and_final_rows(monkeypatch):
    case = get_case("baseline_768", 128)

    def fake_tune_mode(
        case,
        *,
        execution_mode,
        warmup_runs,
        runs_per_sample,
        seed,
        validate_long_seq_runlist,
    ):
        return (
            [
                {
                    "study_id": "end_to_end_tuning",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "execution_mode": execution_mode,
                    "internal_operator": "qkv_proj",
                    "candidate_id": "picked",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "warmup_runs": warmup_runs,
                    "runs_per_sample": runs_per_sample,
                    "avg_latency_ms": 1.0,
                    "bandwidth_gbps": 1.0,
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "operator_config_json": json_dumps({"tile_m": 16}),
                    "is_operator_best": True,
                }
            ],
            {"qkv_proj": "picked"},
            {"qkv_proj": {"tile_m": 16}},
            "",
        )

    def fake_benchmark_mode(
        execution_mode,
        workload,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
        operator_config,
    ):
        return {
            "timed_total_sec": 0.5,
            "measured_inference_count": runs_per_sample,
            "avg_latency_ms": 5.0 if execution_mode == "dataflow" else 6.0,
            "compile_setup_time_ms": 100.0,
            "host_qkv_precompute_ms": None,
            "effective_gflops_per_sec": 25600.0,
            "power_backend": "none",
            "avg_power_w": None,
            "effective_gflops_per_sec_per_watt": None,
            "npu_dispatch_count": 3,
            "npu_unique_instruction_binary_count": 2,
            "npu_unique_xclbin_count": 2,
            "process_model": "in_process",
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.tune_mode",
        fake_tune_mode,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.benchmark_mode",
        fake_benchmark_mode,
    )

    tuning_rows, final_rows = build_rows(
        case,
        mode_filter="all",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        power_backend="none",
    )

    assert len(tuning_rows) == len(EXECUTION_MODES)
    assert len(final_rows) == len(EXECUTION_MODES)
    assert all(row["warmup_runs"] == 1 for row in final_rows)
    assert all(row["runs_per_sample"] == 10 for row in final_rows)
    assert all(
        row["selected_candidate_ids_json"] == '{"qkv_proj": "picked"}'
        for row in final_rows
    )


def test_main_writes_results_and_tuning_csv(monkeypatch, tmp_path):
    case = get_case("baseline_768", 64)

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.iter_cases",
        lambda family, seq_len: (case,),
    )

    def fake_build_rows(
        case,
        *,
        mode_filter,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
    ):
        return (
            [
                {
                    "study_id": "end_to_end_tuning",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "execution_mode": "dataflow",
                    "internal_operator": "qkv_proj",
                    "candidate_id": "picked",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "avg_latency_ms": 1.0,
                    "bandwidth_gbps": 1.0,
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "operator_config_json": "{}",
                    "is_operator_best": True,
                }
            ],
            [
                {
                    "study_id": "end_to_end",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "backend": "npu",
                    "execution_mode": "dataflow",
                    "pattern_label": "dataflow",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "batch_size": 1,
                    "dtype": "bf16",
                    "use_bias": False,
                    "weights_source": "synthetic",
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "measured_inference_count": 1,
                    "timed_total_sec": 0.1,
                    "avg_latency_ms": 1.0,
                    "compile_setup_time_ms": 10.0,
                    "host_qkv_precompute_ms": None,
                    "effective_gflops_per_sec": 64000.0,
                    "power_backend": "none",
                    "avg_power_w": None,
                    "effective_gflops_per_sec_per_watt": None,
                    "npu_dispatch_count": 3,
                    "npu_unique_instruction_binary_count": 2,
                    "npu_unique_xclbin_count": 2,
                    "process_model": "in_process",
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "selected_candidate_ids_json": '{"qkv_proj": "picked"}',
                    "selected_config_json": '{"qkv_proj": {"tile_m": 16}}',
                }
            ],
        )

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run.build_rows",
        fake_build_rows,
    )

    output_path = tmp_path / "end_to_end_results.csv"
    tuning_output_path = tmp_path / "end_to_end_tuning.csv"
    exit_code = main(
        [
            "--family",
            "baseline_768",
            "--seq-len",
            "64",
            "--mode",
            "dataflow",
            "--output",
            str(output_path),
            "--tuning-output",
            str(tuning_output_path),
            "--power-backend",
            "none",
        ]
    )
    assert exit_code == 0

    with output_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with tuning_output_path.open(newline="", encoding="utf-8") as handle:
        tuning_rows = list(csv.DictReader(handle))

    assert len(result_rows) == 1
    assert len(tuning_rows) == 1
    assert result_rows[0]["is_best"] == "True"
    assert tuning_rows[0]["is_operator_best"] == "True"


def test_power_sweep_main_writes_merged_results_and_tuning_csv(monkeypatch, tmp_path):
    cases = (get_case("baseline_768", 64), get_case("baseline_768", 128))

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep.iter_selected_cases",
        lambda family_filter, max_seq_len: cases,
    )

    def fake_build_rows(
        case,
        *,
        mode_filter,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
    ):
        del mode_filter, warmup_runs, runs_per_sample, seed, power_backend
        return (
            [
                {
                    "study_id": "end_to_end_tuning",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "execution_mode": "offload",
                    "internal_operator": "shared_gemm",
                    "candidate_id": f"picked_{case.seq_len}",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "avg_latency_ms": 1.0,
                    "bandwidth_gbps": 1.0,
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "operator_config_json": "{}",
                    "is_operator_best": True,
                }
            ],
            [
                {
                    "study_id": "end_to_end",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "backend": "npu",
                    "execution_mode": "offload",
                    "pattern_label": "offload",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "batch_size": 1,
                    "dtype": "bf16",
                    "use_bias": False,
                    "weights_source": "synthetic",
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "measured_inference_count": 1,
                    "timed_total_sec": 0.1,
                    "avg_latency_ms": float(case.seq_len),
                    "compile_setup_time_ms": 10.0,
                    "host_qkv_precompute_ms": 0.0,
                    "effective_gflops_per_sec": 1.0,
                    "power_backend": "turbostat_pkgwatt",
                    "avg_power_w": 2.0,
                    "effective_gflops_per_sec_per_watt": 0.5,
                    "npu_dispatch_count": 30,
                    "npu_unique_instruction_binary_count": 8,
                    "npu_unique_xclbin_count": 1,
                    "process_model": "in_process",
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "selected_candidate_ids_json": '{"shared_gemm": "picked"}',
                    "selected_config_json": '{"shared_gemm": {"tile_m": 16}}',
                }
            ],
        )

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep.build_rows",
        fake_build_rows,
    )

    output_path = tmp_path / "results_upto256_power.csv"
    tuning_output_path = tmp_path / "tuning_upto256_power.csv"

    exit_code = power_sweep_main(
        [
            "--family",
            "baseline_768",
            "--max-seq-len",
            "256",
            "--mode",
            "all",
            "--output",
            str(output_path),
            "--tuning-output",
            str(tuning_output_path),
        ]
    )
    assert exit_code == 0

    with output_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with tuning_output_path.open(newline="", encoding="utf-8") as handle:
        tuning_rows = list(csv.DictReader(handle))

    assert len(result_rows) == 2
    assert len(tuning_rows) == 2
    assert {row["seq_len"] for row in result_rows} == {"64", "128"}
    assert all(row["is_best"] == "True" for row in result_rows)


def test_power_sweep_main_checkpoints_after_each_case(monkeypatch, tmp_path):
    cases = (get_case("baseline_768", 64), get_case("baseline_768", 128))

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep.iter_selected_cases",
        lambda family_filter, max_seq_len: cases,
    )

    def fake_build_rows(
        case,
        *,
        mode_filter,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
    ):
        del mode_filter, warmup_runs, runs_per_sample, seed, power_backend
        return (
            [
                {
                    "study_id": "end_to_end_tuning",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "execution_mode": "offload",
                    "internal_operator": "shared_gemm",
                    "candidate_id": f"picked_{case.seq_len}",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "avg_latency_ms": 1.0,
                    "bandwidth_gbps": 1.0,
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "operator_config_json": "{}",
                    "is_operator_best": True,
                }
            ],
            [
                {
                    "study_id": "end_to_end",
                    "study_case_id": case.study_case_id,
                    "study_case_label": case.study_case_label,
                    "backend": "npu",
                    "execution_mode": "offload",
                    "pattern_label": "offload",
                    "seq_len": case.seq_len,
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_attention_heads": case.num_attention_heads,
                    "attention_head_size": case.attention_head_size,
                    "batch_size": 1,
                    "dtype": "bf16",
                    "use_bias": False,
                    "weights_source": "synthetic",
                    "warmup_runs": 1,
                    "runs_per_sample": 1,
                    "measured_inference_count": 1,
                    "timed_total_sec": 0.1,
                    "avg_latency_ms": float(case.seq_len),
                    "compile_setup_time_ms": 10.0,
                    "host_qkv_precompute_ms": 0.0,
                    "effective_gflops_per_sec": 1.0,
                    "power_backend": "turbostat_pkgwatt",
                    "avg_power_w": 2.0,
                    "effective_gflops_per_sec_per_watt": 0.5,
                    "npu_dispatch_count": 30,
                    "npu_unique_instruction_binary_count": 8,
                    "npu_unique_xclbin_count": 1,
                    "process_model": "in_process",
                    "validation_error_count": 0,
                    "run_status": "passed",
                    "failure_message": "",
                    "selected_candidate_ids_json": '{"shared_gemm": "picked"}',
                    "selected_config_json": '{"shared_gemm": {"tile_m": 16}}',
                }
            ],
        )

    write_events: list[tuple[str, int]] = []

    def fake_write_rows(path, *, fieldnames, rows):
        del fieldnames
        write_events.append((Path(path).name, len(rows)))

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep.build_rows",
        fake_build_rows,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep.write_rows",
        fake_write_rows,
    )

    exit_code = power_sweep_main(
        [
            "--family",
            "baseline_768",
            "--max-seq-len",
            "256",
            "--mode",
            "all",
            "--output",
            str(tmp_path / "results_upto256_power.csv"),
            "--tuning-output",
            str(tmp_path / "tuning_upto256_power.csv"),
        ]
    )

    assert exit_code == 0
    assert write_events == [
        ("results_upto256_power.csv", 1),
        ("tuning_upto256_power.csv", 1),
        ("results_upto256_power.csv", 2),
        ("tuning_upto256_power.csv", 2),
        ("results_upto256_power.csv", 2),
        ("tuning_upto256_power.csv", 2),
    ]


def test_benchmark_operator_candidate_uses_subprocess_for_long_sequences(monkeypatch):
    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.modes._benchmark_operator_candidate_isolated_subprocess",
        lambda execution_mode, operator_name, workload, candidate_config, *, warmup_runs, runs_per_sample, seed: (
            calls.append((execution_mode, operator_name, workload.seq_len))
            or {
                "avg_latency_ms": 1.0,
                "bandwidth_gbps": 1.0,
                "validation_error_count": 0,
                "run_status": "passed",
                "failure_message": "",
            }
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.modes._benchmark_operator_candidate_in_process",
        lambda *args, **kwargs: {
            "avg_latency_ms": 2.0,
            "bandwidth_gbps": 2.0,
            "validation_error_count": 1,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    workload = get_case("baseline_768", 8192).workload
    result = benchmark_operator_candidate(
        "offload",
        "shared_gemm",
        workload,
        {"tile_m": 64},
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert calls == [("offload", "shared_gemm", 8192)]


def test_benchmark_operator_candidate_uses_in_process_for_short_sequences(monkeypatch):
    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.modes._benchmark_operator_candidate_in_process",
        lambda execution_mode, operator_name, workload, candidate_config, *, warmup_runs, runs_per_sample, seed: (
            calls.append((execution_mode, operator_name, workload.seq_len))
            or {
                "avg_latency_ms": 1.0,
                "bandwidth_gbps": 1.0,
                "validation_error_count": 0,
                "run_status": "passed",
                "failure_message": "",
            }
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.end_to_end.modes._benchmark_operator_candidate_isolated_subprocess",
        lambda *args, **kwargs: {
            "avg_latency_ms": 2.0,
            "bandwidth_gbps": 2.0,
            "validation_error_count": 1,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    workload = get_case("baseline_768", 4096).workload
    result = benchmark_operator_candidate(
        "offload",
        "shared_gemm",
        workload,
        {"tile_m": 64},
        warmup_runs=1,
        runs_per_sample=5,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert calls == [("offload", "shared_gemm", 4096)]
