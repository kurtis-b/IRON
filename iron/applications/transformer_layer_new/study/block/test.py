#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

from iron.applications.transformer_layer_new.study.block.cases import (
    BLOCK_CASES,
    BLOCK_KINDS,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    get_case,
)
from iron.applications.transformer_layer_new.study.block.run import (
    BLOCK_CONFIG_COLUMNS,
    _threshold_validation_result,
    main,
    mark_best_rows,
    operator_kwargs,
)


def test_block_case_table_covers_retained_surface():
    assert tuple(BLOCK_CASES) == FAMILY_IDS

    expected_candidate_sizes = {
        "qkv_proj": 5,
        "mha_out_proj": 6,
        "addnorm": 2,
        "ffn": 11,
    }

    for family_id in FAMILY_IDS:
        assert tuple(BLOCK_CASES[family_id]) == SEQUENCE_LADDER
        for seq_len in SEQUENCE_LADDER:
            case = get_case(family_id, seq_len)
            assert case.family_id == family_id
            assert case.seq_len == seq_len
            for block_kind in BLOCK_KINDS:
                candidates = case.candidates(block_kind)
                assert candidates
                assert all(
                    len(candidate) == expected_candidate_sizes[block_kind]
                    for candidate in candidates
                )


def test_operator_kwargs_expand_shared_workload_correctly():
    case = get_case("baseline_768", 512)
    workload = case.workload

    qkv_kwargs = operator_kwargs(workload, "qkv_proj", case.qkv_proj[0])
    assert qkv_kwargs["seq_len"] == 512
    assert qkv_kwargs["hidden_size"] == 768

    mha_kwargs = operator_kwargs(workload, "mha_out_proj", case.mha_out_proj[0])
    assert mha_kwargs["seq_len"] == 512
    assert mha_kwargs["num_heads"] == 12
    assert mha_kwargs["d"] == 64

    addnorm_kwargs = operator_kwargs(workload, "addnorm", case.addnorm[0])
    assert addnorm_kwargs["size"] == 512 * 768
    assert addnorm_kwargs["tile_size"] == 768

    ffn_kwargs = operator_kwargs(workload, "ffn", case.ffn[0])
    assert ffn_kwargs["M"] == 512
    assert ffn_kwargs["K"] == 768
    assert ffn_kwargs["N"] == 3072


def test_mark_best_rows_marks_fastest_successful_candidate():
    rows = [
        {
            "family_id": "baseline_768",
            "seq_len": 64,
            "block_kind": "qkv_proj",
            "avg_latency_ms": 2.0,
            "run_status": "passed",
        },
        {
            "family_id": "baseline_768",
            "seq_len": 64,
            "block_kind": "qkv_proj",
            "avg_latency_ms": 1.0,
            "run_status": "passed",
        },
        {
            "family_id": "baseline_768",
            "seq_len": 64,
            "block_kind": "qkv_proj",
            "avg_latency_ms": "",
            "run_status": "failed_exception",
        },
    ]

    mark_best_rows(rows)

    assert rows[0]["is_best"] is False
    assert rows[1]["is_best"] is True
    assert rows[2]["is_best"] is False


def test_threshold_validation_result_uses_max_acceptable_errors():
    passing = _threshold_validation_result(
        {"output": 3},
        max_acceptable_errors=10,
    )
    assert passing["validation_error_count"] == 3
    assert passing["run_status"] == "passed"
    assert passing["error_message"] == ""

    failing = _threshold_validation_result(
        {"output": 11},
        max_acceptable_errors=10,
    )
    assert failing["validation_error_count"] == 11
    assert failing["run_status"] == "failed_validation"
    assert "max_acceptable_errors=10" in failing["error_message"]


def test_main_writes_csv_for_selected_case(monkeypatch, tmp_path):
    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        latency_by_block = {
            "qkv_proj": 1.0,
            "mha_out_proj": 2.0,
            "addnorm": 3.0,
            "ffn": 4.0,
        }
        return {
            "avg_latency_ms": latency_by_block[block_kind],
            "bandwidth_gbps": 10.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.block.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    output_path = tmp_path / "block_results.csv"
    exit_code = main(
        [
            "--family",
            "baseline_768",
            "--seq-len",
            "64",
            "--block",
            "all",
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    case = get_case("baseline_768", 64)
    expected_row_count = sum(
        len(case.candidates(block_kind)) for block_kind in BLOCK_KINDS
    )

    assert len(rows) == expected_row_count
    assert {row["block_kind"] for row in rows} == set(BLOCK_KINDS)
    assert all(row["family_id"] == "baseline_768" for row in rows)
    assert all(row["seq_len"] == "64" for row in rows)
    assert all(row["run_status"] == "passed" for row in rows)

    best_rows = [row for row in rows if row["is_best"] == "True"]
    assert len(best_rows) == len(BLOCK_KINDS)
    assert {row["block_kind"] for row in best_rows} == set(BLOCK_KINDS)

    config_columns = {
        column
        for block_columns in BLOCK_CONFIG_COLUMNS.values()
        for column in block_columns
    }
    assert config_columns.issubset(rows[0].keys())
