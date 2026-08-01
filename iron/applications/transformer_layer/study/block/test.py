#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

from matplotlib import pyplot as plt
import pandas as pd

from iron.applications.transformer_layer.study.block.cases import (
    BLOCK_CASES,
    BLOCK_KINDS,
    DECODER_FAMILY_IDS,
    ENCODER_FAMILY_IDS,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    get_case,
)
from iron.applications.transformer_layer.study.block.plot_best_latency import (
    plot_best_latency,
)
from iron.applications.transformer_layer.study.block.run import (
    CSV_FIELDNAMES,
    BLOCK_CONFIG_COLUMNS,
    _threshold_validation_result,
    benchmark_candidate,
    iteration_schedule,
    main,
    mark_best_rows,
    operator_kwargs,
)


def test_block_case_table_covers_retained_surface():
    assert ENCODER_FAMILY_IDS == (
        "tinybert_512",
        "baseline_768",
        "baseline_1024",
    )
    assert DECODER_FAMILY_IDS == (
        "gpt2_512",
        "gpt2_small_768",
        "gpt2_medium_1024",
    )
    assert tuple(BLOCK_CASES) == FAMILY_IDS

    expected_candidate_sizes = {
        "qkv_proj": 5,
        "mha_out_proj": 6,
        "mha_out_proj_causal": 6,
        "addnorm": 2,
        "layer_norm": 2,
        "elementwise_add": 2,
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
                if candidates:
                    assert all(
                        len(candidate) == expected_candidate_sizes[block_kind]
                        for candidate in candidates
                    )


def test_decoder_only_blocks_are_present_only_for_decoder_like_shapes():
    tinybert_case = get_case("tinybert_512", 256)
    assert tinybert_case.candidates("mha_out_proj_causal") == ()
    assert tinybert_case.candidates("layer_norm") == ()
    assert tinybert_case.candidates("elementwise_add") == ()

    bert_case = get_case("baseline_768", 256)
    assert bert_case.candidates("mha_out_proj_causal") == ()
    assert bert_case.candidates("layer_norm") == ()
    assert bert_case.candidates("elementwise_add") == ()

    decoder_case = get_case("gpt2_512", 256)
    assert len(decoder_case.candidates("mha_out_proj_causal")) == 1
    assert len(decoder_case.candidates("layer_norm")) == 1
    assert len(decoder_case.candidates("elementwise_add")) == 1

    long_seq_case = get_case("gpt2_512", 8192)
    assert len(long_seq_case.candidates("mha_out_proj_causal")) == 1
    assert len(long_seq_case.candidates("layer_norm")) == 1
    assert len(long_seq_case.candidates("elementwise_add")) == 1


def test_removed_cases_manifest_prunes_known_bad_tinybert_qkv_candidate():
    case = get_case("tinybert_512", 256)
    candidates = case.candidates("qkv_proj")
    assert candidates == ((64, 64, 64, 4, 8),)


def test_operator_kwargs_expand_shared_workload_correctly():
    case = get_case("gpt2_small_768", 512)
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

    causal_mha_kwargs = operator_kwargs(
        workload, "mha_out_proj_causal", case.mha_out_proj_causal[0]
    )
    assert causal_mha_kwargs["is_causal"] is True
    assert causal_mha_kwargs["q_seq_tile"] == causal_mha_kwargs["kv_seq_tile"]

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


def test_iteration_schedule_matches_end_to_end_defaults():
    assert iteration_schedule(64, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(128, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(256, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(512, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(1024, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(2048, warmup_iters=None, timed_iters=None) == (1, 10)
    assert iteration_schedule(4096, warmup_iters=None, timed_iters=None) == (1, 5)
    assert iteration_schedule(8192, warmup_iters=None, timed_iters=None) == (1, 2)
    assert iteration_schedule(16384, warmup_iters=None, timed_iters=None) == (1, 2)


def test_long_sequence_mha_out_proj_cases_keep_only_stable_candidate():
    for family_id in FAMILY_IDS:
        for seq_len in (8192, 16384):
            case = get_case(family_id, seq_len)
            assert len(case.mha_out_proj) == 1


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
            "mha_out_proj_causal": 2.5,
            "addnorm": 3.0,
            "layer_norm": 3.5,
            "elementwise_add": 3.75,
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
        "iron.applications.transformer_layer.study.block.run.benchmark_candidate",
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
            "--no-resume",
        ]
    )

    assert exit_code == 0

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    case = get_case("baseline_768", 64)
    expected_row_count = sum(
        len(case.candidates(block_kind)) for block_kind in BLOCK_KINDS
    )
    expected_block_kinds = {
        block_kind for block_kind in BLOCK_KINDS if case.candidates(block_kind)
    }

    assert len(rows) == expected_row_count
    assert {row["block_kind"] for row in rows} == expected_block_kinds
    assert all(row["family_id"] == "baseline_768" for row in rows)
    assert all(row["seq_len"] == "64" for row in rows)
    assert all(row["run_status"] == "passed" for row in rows)

    best_rows = [row for row in rows if row["is_best"] == "True"]
    assert len(best_rows) == len(expected_block_kinds)
    assert {row["block_kind"] for row in best_rows} == expected_block_kinds

    config_columns = {
        column
        for block_columns in BLOCK_CONFIG_COLUMNS.values()
        for column in block_columns
    }
    assert config_columns.issubset(rows[0].keys())


def test_main_checkpoints_after_each_case(monkeypatch, tmp_path):
    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        return {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": 10.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    checkpoint_row_counts: list[int] = []

    def fake_write_rows(output_path, rows):
        checkpoint_row_counts.append(len(rows))

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.benchmark_candidate",
        fake_benchmark_candidate,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.write_rows",
        fake_write_rows,
    )

    exit_code = main(
        [
            "--family",
            "all",
            "--seq-len",
            "64",
            "--block",
            "qkv_proj",
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
            "--output",
            str(tmp_path / "ignored.csv"),
            "--no-resume",
        ]
    )

    assert exit_code == 0

    tinybert_rows = len(get_case("tinybert_512", 64).candidates("qkv_proj"))
    baseline_768_rows = len(get_case("baseline_768", 64).candidates("qkv_proj"))
    baseline_1024_rows = len(get_case("baseline_1024", 64).candidates("qkv_proj"))
    gpt2_512_rows = len(get_case("gpt2_512", 64).candidates("qkv_proj"))
    gpt2_small_rows = len(get_case("gpt2_small_768", 64).candidates("qkv_proj"))
    gpt2_medium_rows = len(get_case("gpt2_medium_1024", 64).candidates("qkv_proj"))
    expected_counts = [
        tinybert_rows,
        tinybert_rows + baseline_768_rows,
        tinybert_rows + baseline_768_rows + baseline_1024_rows,
        tinybert_rows + baseline_768_rows + baseline_1024_rows + gpt2_512_rows,
        tinybert_rows
        + baseline_768_rows
        + baseline_1024_rows
        + gpt2_512_rows
        + gpt2_small_rows,
        tinybert_rows
        + baseline_768_rows
        + baseline_1024_rows
        + gpt2_512_rows
        + gpt2_small_rows
        + gpt2_medium_rows,
    ]
    assert checkpoint_row_counts == [*expected_counts, expected_counts[-1]]


def test_plot_best_latency_uses_shared_2x3_family_layout():
    best = pd.DataFrame(
        [
            {
                "family_id": "baseline_768",
                "seq_len": 64,
                "block_kind": "qkv_proj",
                "avg_latency_ms": 1.0,
                "block_label": "QKV Proj",
            },
            {
                "family_id": "baseline_768",
                "seq_len": 64,
                "block_kind": "ffn",
                "avg_latency_ms": 2.0,
                "block_label": "FFN",
            },
            {
                "family_id": "gpt2_512",
                "seq_len": 64,
                "block_kind": "mha_out_proj_causal",
                "avg_latency_ms": 3.0,
                "block_label": "MHAO",
            },
            {
                "family_id": "gpt2_512",
                "seq_len": 64,
                "block_kind": "layer_norm",
                "avg_latency_ms": 4.0,
                "block_label": "Add + Norm",
            },
        ]
    )

    fig = plot_best_latency(
        best,
        "Latency Comparison of Encoder and Decoder Blocks",
    )
    try:
        axes_count = len(fig.axes)
        visible_axes = [ax for ax in fig.axes if ax.axison]
        visible_titles = [ax.get_title(loc="left") for ax in visible_axes]
        text_labels = [text.get_text() for text in fig.texts]
    finally:
        plt.close(fig)

    assert axes_count == 6
    assert visible_titles == ["B-M", "G-S"]
    assert "Encoder" in text_labels
    assert "Decoder" in text_labels


def test_main_reuses_matching_passed_rows(monkeypatch, tmp_path):
    output_path = tmp_path / "block_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "study_id": "block",
                "family_id": "baseline_768",
                "family_label": "768 / 3072 / 12",
                "seq_len": 64,
                "block_kind": "addnorm",
                "candidate_index": 0,
                "head_dim": 64,
                "num_heads": 12,
                "hidden_size": 768,
                "ffn_dim": 3072,
                "avg_latency_ms": 1.0,
                "latency_sample_count": 2,
                "min_latency_ms": 1.0,
                "max_latency_ms": 1.0,
                "bandwidth_gbps": 10.0,
                "warmup_iters": 1,
                "timed_iters": 2,
                "validation_error_count": 0,
                "run_status": "passed",
                "is_best": True,
                "error_message": "",
                "addnorm_num_aie_columns": 8,
                "addnorm_tile_size": 768,
            }
        )

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark_candidate should not be called")

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.benchmark_candidate",
        fail_benchmark,
    )

    exit_code = main(
        [
            "--family",
            "baseline_768",
            "--seq-len",
            "64",
            "--block",
            "addnorm",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0


def test_benchmark_candidate_runs_aggressive_cleanup_for_long_sequences(monkeypatch):
    cleanup_seq_lens: list[int] = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._benchmark_candidate_isolated_subprocess",
        lambda block_kind, workload, candidate, *, warmup_iters, timed_iters, seed: {
            "avg_latency_ms": 1.0,
            "bandwidth_gbps": 10.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._aggressive_cleanup",
        lambda seq_len: cleanup_seq_lens.append(seq_len),
    )

    case = get_case("baseline_1024", 8192)
    result = benchmark_candidate(
        "qkv_proj",
        case.workload,
        case.qkv_proj[0],
        warmup_iters=1,
        timed_iters=2,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert cleanup_seq_lens == []


def test_benchmark_candidate_uses_subprocess_for_long_sequences(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._benchmark_candidate_isolated_subprocess",
        lambda block_kind, workload, candidate, *, warmup_iters, timed_iters, seed: (
            calls.append((block_kind, workload.seq_len))
            or {
                "avg_latency_ms": 1.0,
                "bandwidth_gbps": 10.0,
                "validation_error_count": 0,
                "run_status": "passed",
                "error_message": "",
            }
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._benchmark_candidate_in_process",
        lambda *args, **kwargs: {
            "avg_latency_ms": 2.0,
            "bandwidth_gbps": 9.0,
            "validation_error_count": 1,
            "run_status": "passed",
            "error_message": "",
        },
    )

    case = get_case("baseline_1024", 16384)
    result = benchmark_candidate(
        "qkv_proj",
        case.workload,
        case.qkv_proj[0],
        warmup_iters=1,
        timed_iters=2,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert calls == [("qkv_proj", 16384)]


def test_benchmark_candidate_uses_in_process_for_short_sequences(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run.require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._benchmark_candidate_in_process",
        lambda block_kind, workload, candidate, *, warmup_iters, timed_iters, seed: (
            calls.append((block_kind, workload.seq_len))
            or {
                "avg_latency_ms": 1.0,
                "bandwidth_gbps": 10.0,
                "validation_error_count": 0,
                "run_status": "passed",
                "error_message": "",
            }
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.block.run._benchmark_candidate_isolated_subprocess",
        lambda *args, **kwargs: {
            "avg_latency_ms": 2.0,
            "bandwidth_gbps": 9.0,
            "validation_error_count": 1,
            "run_status": "passed",
            "error_message": "",
        },
    )

    case = get_case("baseline_768", 64)
    result = benchmark_candidate(
        "qkv_proj",
        case.workload,
        case.qkv_proj[0],
        warmup_iters=1,
        timed_iters=1,
        seed=42,
    )

    assert result["run_status"] == "passed"
    assert calls == [("qkv_proj", 64)]
