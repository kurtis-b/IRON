#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json

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
    get_case,
    iter_cases,
    load_candidate_payload,
    load_default_candidate_payloads,
)
from iron.applications.transformer_layer_new.study.end_to_end.modes import (
    FINAL_ERROR_THRESHOLD,
    _benchmark_offload_shared_gemm,
    _elementwise_mul_buffers,
    _metadata_for_operator,
    _tokens_per_sec,
    _tokens_per_sec_per_watt,
    _validate_output,
)
from iron.applications.transformer_layer_new.study.end_to_end.run import (
    build_rows,
    iteration_schedule,
    json_dumps,
    main,
    mark_best_rows,
    tune_mode,
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
                assert actual == expected

        all_payload = payload[family_id]["all"]
        block_case = family_cases[1024]
        for operator_name in MODE_OPERATORS["dataflow"]:
            actual = [candidate["config"] for candidate in all_payload[operator_name]]
            expected = _expected_dataflow_config_rows(block_case, operator_name)
            assert actual == expected


def test_iteration_schedule_matches_block_study():
    assert iteration_schedule(64) == (10, 100)
    assert iteration_schedule(128) == (10, 100)
    assert iteration_schedule(256) == (5, 50)
    assert iteration_schedule(512) == (5, 50)
    assert iteration_schedule(1024) == (3, 20)
    assert iteration_schedule(2048) == (3, 20)
    assert iteration_schedule(4096) == (1, 10)


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


def test_throughput_helpers():
    tokens_per_sec = _tokens_per_sec(512, 2.0)
    assert tokens_per_sec == 256000.0
    assert _tokens_per_sec_per_watt(tokens_per_sec, 50.0) == 5120.0
    assert _tokens_per_sec_per_watt(tokens_per_sec, None) is None


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


def test_build_rows_returns_tuning_and_final_rows(monkeypatch):
    case = get_case("baseline_768", 128)

    def fake_tune_mode(
        case,
        *,
        execution_mode,
        warmup_runs,
        runs_per_sample,
        seed,
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
            "tokens_per_sec": 25600.0,
            "power_backend": "none",
            "avg_power_w": None,
            "tokens_per_sec_per_watt": None,
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
    assert all(row["warmup_runs"] == 10 for row in final_rows)
    assert all(row["runs_per_sample"] == 100 for row in final_rows)
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
                    "tokens_per_sec": 64000.0,
                    "power_backend": "none",
                    "avg_power_w": None,
                    "tokens_per_sec_per_watt": None,
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
