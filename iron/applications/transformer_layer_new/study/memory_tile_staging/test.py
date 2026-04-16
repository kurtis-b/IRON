#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

from matplotlib import pyplot as plt

from iron.applications.transformer_layer_new.study.memory_tile_staging.plot_staging_depth import (
    load_plot_rows,
    render_plot,
    write_canonical_plots,
)
from iron.applications.transformer_layer_new.study.memory_tile_staging.run import (
    STAGING_SEQUENCE_LENGTHS,
    build_selection_rows,
    main,
    supported_staging_depths,
)
from iron.applications.transformer_layer_new.study.memory_tile_staging.select import (
    CONFIG_COLUMNS_BY_BLOCK_KIND,
    ReferenceSelection,
    select_reference_rows,
)


def _reference_row(
    block_kind: str,
    *,
    family_id: str = "baseline_768",
    seq_len: str = "512",
    candidate_index: str = "0",
    avg_latency_ms: str = "5.0",
    run_status: str = "passed",
    is_best: str = "True",
) -> dict[str, str]:
    family_info = {
        "tinybert_512": ("TinyBERT", "8", "512", "2048"),
        "baseline_768": ("BERT-Base", "12", "768", "3072"),
        "baseline_1024": ("BERT-Large", "16", "1024", "4096"),
        "gpt2_small_768": ("GPT-2 Small", "12", "768", "3072"),
        "gpt2_medium_1024": ("GPT-2 Medium", "16", "1024", "4096"),
    }[family_id]
    row = {
        "study_id": "block",
        "family_id": family_id,
        "family_label": family_info[0],
        "seq_len": seq_len,
        "block_kind": block_kind,
        "candidate_index": candidate_index,
        "head_dim": "64",
        "num_heads": family_info[1],
        "hidden_size": family_info[2],
        "ffn_dim": family_info[3],
        "avg_latency_ms": avg_latency_ms,
        "bandwidth_gbps": "10.0",
        "warmup_iters": "1",
        "timed_iters": "2",
        "validation_error_count": "0",
        "run_status": run_status,
        "is_best": is_best,
        "error_message": "",
    }
    for supported_block_kind, columns in CONFIG_COLUMNS_BY_BLOCK_KIND.items():
        for column in columns:
            row[column] = ""

    if block_kind == "mha_out_proj":
        row.update(
            {
                "mha_out_proj_parallel_seq": "8",
                "mha_out_proj_q_seq_tile": "32",
                "mha_out_proj_kv_seq_tile": "64",
                "mha_out_proj_emb_tile": (
                    "64"
                    if family_id == "tinybert_512"
                    else ("96" if family_id == "baseline_768" else "128")
                ),
                "mha_out_proj_parallel_heads": "1",
                "mha_out_proj_o_proj_acc_depth": "8",
            }
        )
    elif block_kind == "mha_out_proj_causal":
        row.update(
            {
                "mha_out_proj_causal_parallel_seq": "8",
                "mha_out_proj_causal_q_seq_tile": "32",
                "mha_out_proj_causal_kv_seq_tile": "64",
                "mha_out_proj_causal_emb_tile": (
                    "64"
                    if family_id == "tinybert_512"
                    else (
                        "96"
                        if family_id in {"baseline_768", "gpt2_small_768"}
                        else "128"
                    )
                ),
                "mha_out_proj_causal_parallel_heads": "1",
                "mha_out_proj_causal_o_proj_acc_depth": "8",
            }
        )
    elif block_kind == "ffn":
        row.update(
            {
                "ffn_num_aie_columns": "8",
                "ffn_b_col_maj": "False",
                "ffn_c_col_maj": "False",
                "ffn_tile_m": "64",
                "ffn_tile_k": "64" if family_id != "baseline_1024" else "128",
                "ffn_tile_n": "64",
                "ffn_down_proj_depth": (
                    "8"
                    if family_id == "tinybert_512"
                    else ("6" if family_id == "baseline_768" else "8")
                ),
                "ffn_n_a_tiles_distributed": "8",
                "ffn_n_b_tiles_distributed": "2",
                "ffn_stage_only": "",
                "ffn_gelu_stage": "1",
            }
        )

    return row


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_select_reference_rows_keeps_only_best_supported_blocks():
    rows = [
        _reference_row("mha_out_proj", avg_latency_ms="8.0", candidate_index="1"),
        _reference_row("mha_out_proj", avg_latency_ms="6.0", candidate_index="0"),
        _reference_row("ffn", avg_latency_ms="4.0"),
        _reference_row("ffn", avg_latency_ms="3.0", run_status="failed_validation"),
        _reference_row("qkv_proj"),
        _reference_row("mha_out_proj", avg_latency_ms="2.0", is_best="False"),
    ]

    selections = select_reference_rows(rows, family_filter="baseline_768")

    assert [selection.block_kind for selection in selections] == [
        "mha_out_proj",
        "ffn",
    ]
    assert selections[0].source_candidate_index == 0
    assert selections[0].source_candidate == (8, 32, 64, 96, 1, 8)
    assert selections[1].source_candidate == (
        8,
        False,
        False,
        64,
        64,
        64,
        6,
        8,
        2,
        None,
        1,
    )


def test_select_reference_rows_generates_decoder_targets():
    rows = [
        _reference_row("ffn", family_id="baseline_768", avg_latency_ms="4.0"),
        _reference_row(
            "mha_out_proj_causal",
            family_id="baseline_768",
            avg_latency_ms="5.0",
        ),
    ]

    selections = select_reference_rows(rows)
    selection_keys = {
        (
            selection.family_id,
            selection.block_kind,
            selection.benchmark_block_kind or selection.block_kind,
        )
        for selection in selections
    }

    assert ("baseline_768", "ffn", "ffn") in selection_keys
    assert ("gpt2_small_768", "ffn", "ffn") in selection_keys
    assert (
        "gpt2_small_768",
        "mha_out_proj",
        "mha_out_proj_causal",
    ) in selection_keys


def test_supported_staging_depths_cover_all_divisors_for_mha_out_proj():
    selection = ReferenceSelection(
        family_id="baseline_768",
        family_label="768 / 3072 / 12",
        seq_len=512,
        block_kind="mha_out_proj",
        source_candidate_index=0,
        source_avg_latency_ms=5.0,
        head_dim=64,
        num_heads=12,
        hidden_size=768,
        ffn_dim=3072,
        source_candidate=(8, 32, 64, 96, 1, 8),
    )

    assert supported_staging_depths(selection) == (1, 2, 4, 8)


def test_supported_staging_depths_cover_all_divisors_for_ffn():
    selection = ReferenceSelection(
        family_id="baseline_768",
        family_label="768 / 3072 / 12",
        seq_len=512,
        block_kind="ffn",
        source_candidate_index=0,
        source_avg_latency_ms=5.0,
        head_dim=64,
        num_heads=12,
        hidden_size=768,
        ffn_dim=3072,
        source_candidate=(8, False, False, 64, 64, 64, 6, 8, 2, None, 1),
    )

    assert supported_staging_depths(selection) == (1, 2, 3, 4, 6, 12)


def test_build_selection_rows_sweeps_only_depth_and_computes_speedup(monkeypatch):
    selection = ReferenceSelection(
        family_id="baseline_768",
        family_label="768 / 3072 / 12",
        seq_len=512,
        block_kind="ffn",
        source_candidate_index=0,
        source_avg_latency_ms=5.0,
        head_dim=64,
        num_heads=12,
        hidden_size=768,
        ffn_dim=3072,
        source_candidate=(8, False, False, 64, 64, 64, 6, 8, 2, None, 1),
    )

    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        depth = int(candidate[6])
        return {
            "avg_latency_ms": 120.0 / depth,
            "bandwidth_gbps": 10.0 + depth,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    rows = build_selection_rows(
        selection,
        warmup_iters=1,
        timed_iters=2,
        seed=42,
    )

    assert [row["staging_depth"] for row in rows] == [1, 2, 3, 4, 6, 12]
    assert {row["ffn_tile_k"] for row in rows} == {64}
    assert {row["ffn_tile_n"] for row in rows} == {64}
    assert {row["ffn_down_proj_depth"] for row in rows} == {1, 2, 3, 4, 6, 12}
    assert rows[0]["speedup_vs_depth1"] == 1.0
    assert rows[-1]["speedup_vs_depth1"] == 12.0
    best_rows = [row for row in rows if row["is_best_depth"] is True]
    assert len(best_rows) == 1
    assert best_rows[0]["staging_depth"] == 12


def test_build_selection_rows_uses_causal_mha_benchmark_for_decoder(monkeypatch):
    selection = ReferenceSelection(
        family_id="gpt2_small_768",
        family_label="GPT-2 Small",
        seq_len=512,
        block_kind="mha_out_proj",
        source_candidate_index=0,
        source_avg_latency_ms=5.0,
        head_dim=64,
        num_heads=12,
        hidden_size=768,
        ffn_dim=3072,
        source_candidate=(8, 32, 64, 96, 1, 8),
        benchmark_block_kind="mha_out_proj_causal",
    )
    seen_block_kinds: list[str] = []

    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        seen_block_kinds.append(block_kind)
        return {
            "avg_latency_ms": 100.0 / int(candidate[5]),
            "bandwidth_gbps": 5.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    rows = build_selection_rows(
        selection,
        warmup_iters=1,
        timed_iters=2,
        seed=42,
    )

    assert rows
    assert {row["block_kind"] for row in rows} == {"mha_out_proj"}
    assert seen_block_kinds
    assert set(seen_block_kinds) == {"mha_out_proj_causal"}


def test_build_selection_rows_reuses_existing_rows_and_skips_removed(monkeypatch):
    selection = ReferenceSelection(
        family_id="baseline_768",
        family_label="768 / 3072 / 12",
        seq_len=512,
        block_kind="mha_out_proj",
        source_candidate_index=0,
        source_avg_latency_ms=5.0,
        head_dim=64,
        num_heads=12,
        hidden_size=768,
        ffn_dim=3072,
        source_candidate=(8, 32, 64, 96, 1, 8),
    )
    benchmark_calls: list[int] = []

    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        benchmark_calls.append(int(candidate[5]))
        return {
            "avg_latency_ms": 100.0 / int(candidate[5]),
            "bandwidth_gbps": 5.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    rows = build_selection_rows(
        selection,
        warmup_iters=1,
        timed_iters=2,
        seed=42,
        existing_rows={
            ("baseline_768", 512, "mha_out_proj", 2): {
                "study_id": "memory_tile_staging",
                "family_id": "baseline_768",
                "family_label": "768 / 3072 / 12",
                "seq_len": "512",
                "block_kind": "mha_out_proj",
                "source_candidate_index": "0",
                "source_staging_depth": "8",
                "staging_depth": "2",
                "head_dim": "64",
                "num_heads": "12",
                "hidden_size": "768",
                "ffn_dim": "3072",
                "warmup_iters": "1",
                "timed_iters": "2",
                "avg_latency_ms": "50.0",
                "latency_sample_count": "2",
                "min_latency_ms": "50.0",
                "max_latency_ms": "50.0",
                "bandwidth_gbps": "5.0",
                "speedup_vs_depth1": "",
                "validation_error_count": "0",
                "run_status": "passed",
                "is_best_depth": "False",
                "error_message": "",
                "mha_out_proj_parallel_seq": "8",
                "mha_out_proj_q_seq_tile": "32",
                "mha_out_proj_kv_seq_tile": "64",
                "mha_out_proj_emb_tile": "96",
                "mha_out_proj_parallel_heads": "1",
                "mha_out_proj_o_proj_acc_depth": "2",
            }
        },
        removed_case_notes={
            ("baseline_768", 512, "mha_out_proj", 1): "known timeout",
        },
    )

    assert [row["staging_depth"] for row in rows] == ["2", 4, 8]
    assert benchmark_calls == [4, 8]


def test_main_writes_csv_and_canonical_plots(monkeypatch, tmp_path):
    reference_input = tmp_path / "block_results.csv"
    fieldnames = list(_reference_row("mha_out_proj").keys())
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_reference_row("mha_out_proj"))
        writer.writerow(_reference_row("ffn"))
        writer.writerow(_reference_row("ffn"))

    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        depth = int(candidate[5] if block_kind == "mha_out_proj" else candidate[6])
        return {
            "avg_latency_ms": 100.0 / depth,
            "bandwidth_gbps": 5.0 * depth,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    output_path = tmp_path / "memory_tile_staging.csv"
    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--output",
            str(output_path),
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert rows
    assert {row["block_kind"] for row in rows} == {"mha_out_proj", "ffn"}
    assert {
        row["staging_depth"] for row in rows if row["block_kind"] == "mha_out_proj"
    } == {"1", "2", "4", "8"}
    assert {row["staging_depth"] for row in rows if row["block_kind"] == "ffn"} == {
        "1",
        "2",
        "3",
        "4",
        "6",
        "12",
    }

    svg_path = tmp_path / "mha_out_proj_latency_by_staging_depth.svg"
    speedup_path = tmp_path / "ffn_speedup_by_staging_depth.svg"
    assert svg_path.exists()
    assert speedup_path.exists()
    assert (
        "Hybrid Block MHA + Output Projection Latency by Memory-Tile Staging Depth"
        in svg_path.read_text(encoding="utf-8")
    )
    assert (
        "Hybrid Block FFN Speedup by Memory-Tile Staging Depth"
        in speedup_path.read_text(encoding="utf-8")
    )


def test_main_reuses_existing_results_by_default(monkeypatch, tmp_path):
    reference_input = tmp_path / "block_results.csv"
    fieldnames = list(_reference_row("mha_out_proj").keys())
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_reference_row("mha_out_proj"))

    output_path = tmp_path / "memory_tile_staging.csv"
    existing_rows = []
    for family_id in ("baseline_768", "gpt2_small_768"):
        for depth in ("1", "2", "4", "8"):
            existing_rows.append(
                {
                    "study_id": "memory_tile_staging",
                    "family_id": family_id,
                    "family_label": "768 / 3072 / 12",
                    "seq_len": "512",
                    "block_kind": "mha_out_proj",
                    "source_candidate_index": "0",
                    "source_staging_depth": "8",
                    "staging_depth": depth,
                    "head_dim": "64",
                    "num_heads": "12",
                    "hidden_size": "768",
                    "ffn_dim": "3072",
                    "warmup_iters": "1",
                    "timed_iters": "2",
                    "avg_latency_ms": "10.0",
                    "latency_sample_count": "2",
                    "min_latency_ms": "10.0",
                    "max_latency_ms": "10.0",
                    "bandwidth_gbps": "5.0",
                    "speedup_vs_depth1": "1.0",
                    "validation_error_count": "0",
                    "run_status": "passed",
                    "is_best_depth": "False",
                    "error_message": "",
                    "mha_out_proj_parallel_seq": "8",
                    "mha_out_proj_q_seq_tile": "32",
                    "mha_out_proj_kv_seq_tile": "64",
                    "mha_out_proj_emb_tile": "96",
                    "mha_out_proj_parallel_heads": "1",
                    "mha_out_proj_o_proj_acc_depth": depth,
                    "ffn_num_aie_columns": "",
                    "ffn_b_col_maj": "",
                    "ffn_c_col_maj": "",
                    "ffn_tile_m": "",
                    "ffn_tile_k": "",
                    "ffn_tile_n": "",
                    "ffn_down_proj_depth": "",
                    "ffn_n_a_tiles_distributed": "",
                    "ffn_n_b_tiles_distributed": "",
                    "ffn_stage_only": "",
                    "ffn_gelu_stage": "",
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(existing_rows[0]))
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark_candidate should not be called")

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fail_benchmark,
    )

    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0


def test_main_preserves_unrelated_existing_results_when_running_filtered_selection(
    monkeypatch, tmp_path
):
    reference_input = tmp_path / "block_results.csv"
    fieldnames = list(_reference_row("mha_out_proj").keys())
    with reference_input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_reference_row("mha_out_proj", family_id="baseline_768"))

    output_path = tmp_path / "memory_tile_staging.csv"
    existing_rows = [
        {
            "study_id": "memory_tile_staging",
            "family_id": "gpt2_medium_1024",
            "family_label": "1024 / 4096 / 16",
            "seq_len": "8192",
            "block_kind": "ffn",
            "source_candidate_index": "0",
            "source_staging_depth": "8",
            "staging_depth": "1",
            "head_dim": "64",
            "num_heads": "16",
            "hidden_size": "1024",
            "ffn_dim": "4096",
            "warmup_iters": "1",
            "timed_iters": "2",
            "avg_latency_ms": "10.0",
            "latency_sample_count": "2",
            "min_latency_ms": "10.0",
            "max_latency_ms": "10.0",
            "bandwidth_gbps": "5.0",
            "speedup_vs_depth1": "1.0",
            "validation_error_count": "0",
            "run_status": "passed",
            "is_best_depth": "False",
            "error_message": "",
            "mha_out_proj_parallel_seq": "",
            "mha_out_proj_q_seq_tile": "",
            "mha_out_proj_kv_seq_tile": "",
            "mha_out_proj_emb_tile": "",
            "mha_out_proj_parallel_heads": "",
            "mha_out_proj_o_proj_acc_depth": "",
            "ffn_num_aie_columns": "8",
            "ffn_b_col_maj": "False",
            "ffn_c_col_maj": "False",
            "ffn_tile_m": "64",
            "ffn_tile_k": "128",
            "ffn_tile_n": "64",
            "ffn_down_proj_depth": "1",
            "ffn_n_a_tiles_distributed": "8",
            "ffn_n_b_tiles_distributed": "2",
            "ffn_stage_only": "",
            "ffn_gelu_stage": "1",
        }
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(existing_rows[0]))
        writer.writeheader()
        writer.writerows(existing_rows)

    def fake_benchmark_candidate(
        block_kind,
        workload,
        candidate,
        *,
        warmup_iters,
        timed_iters,
        seed,
    ):
        depth = int(candidate[5] if block_kind == "mha_out_proj" else candidate[6])
        return {
            "avg_latency_ms": 100.0 / depth,
            "latency_sample_count": timed_iters,
            "min_latency_ms": 100.0 / depth,
            "max_latency_ms": 100.0 / depth,
            "bandwidth_gbps": 5.0 * depth,
            "validation_error_count": 0,
            "run_status": "passed",
            "error_message": "",
        }

    monkeypatch.setattr(
        "iron.applications.transformer_layer_new.study.memory_tile_staging.run.benchmark_candidate",
        fake_benchmark_candidate,
    )

    exit_code = main(
        [
            "--reference-input",
            str(reference_input),
            "--family",
            "baseline_768",
            "--seq-len",
            "512",
            "--block",
            "mha_out_proj",
            "--output",
            str(output_path),
            "--warmup-iters",
            "1",
            "--timed-iters",
            "2",
        ]
    )

    assert exit_code == 0
    rows = _read_csv_rows(output_path)
    assert {(row["family_id"], row["seq_len"], row["block_kind"]) for row in rows} == {
        ("baseline_768", "512", "mha_out_proj"),
        ("gpt2_medium_1024", "8192", "ffn"),
    }


def test_write_canonical_plots_renders_all_expected_outputs(tmp_path):
    results_csv = tmp_path / "results.csv"
    fieldnames = [
        "study_id",
        "family_id",
        "family_label",
        "seq_len",
        "block_kind",
        "source_candidate_index",
        "source_staging_depth",
        "staging_depth",
        "head_dim",
        "num_heads",
        "hidden_size",
        "ffn_dim",
        "warmup_iters",
        "timed_iters",
        "avg_latency_ms",
        "bandwidth_gbps",
        "speedup_vs_depth1",
        "validation_error_count",
        "run_status",
        "is_best_depth",
        "error_message",
        *(
            column
            for columns in CONFIG_COLUMNS_BY_BLOCK_KIND.values()
            for column in columns
        ),
    ]
    rows = [
        {
            **{field: "" for field in fieldnames},
            "study_id": "memory_tile_staging",
            "family_id": "baseline_768",
            "family_label": "768 / 3072 / 12",
            "seq_len": "512",
            "block_kind": "mha_out_proj",
            "source_candidate_index": "0",
            "source_staging_depth": "8",
            "staging_depth": "1",
            "head_dim": "64",
            "num_heads": "12",
            "hidden_size": "768",
            "ffn_dim": "3072",
            "warmup_iters": "1",
            "timed_iters": "2",
            "avg_latency_ms": "10.0",
            "bandwidth_gbps": "5.0",
            "speedup_vs_depth1": "1.0",
            "validation_error_count": "0",
            "run_status": "passed",
            "is_best_depth": "False",
            "mha_out_proj_parallel_seq": "8",
            "mha_out_proj_q_seq_tile": "32",
            "mha_out_proj_kv_seq_tile": "64",
            "mha_out_proj_emb_tile": "96",
            "mha_out_proj_parallel_heads": "1",
            "mha_out_proj_o_proj_acc_depth": "1",
        },
        {
            **{field: "" for field in fieldnames},
            "study_id": "memory_tile_staging",
            "family_id": "baseline_768",
            "family_label": "768 / 3072 / 12",
            "seq_len": "512",
            "block_kind": "ffn",
            "source_candidate_index": "0",
            "source_staging_depth": "6",
            "staging_depth": "1",
            "head_dim": "64",
            "num_heads": "12",
            "hidden_size": "768",
            "ffn_dim": "3072",
            "warmup_iters": "1",
            "timed_iters": "2",
            "avg_latency_ms": "12.0",
            "bandwidth_gbps": "6.0",
            "speedup_vs_depth1": "1.0",
            "validation_error_count": "0",
            "run_status": "passed",
            "is_best_depth": "False",
            "ffn_num_aie_columns": "8",
            "ffn_b_col_maj": "False",
            "ffn_c_col_maj": "False",
            "ffn_tile_m": "64",
            "ffn_tile_k": "64",
            "ffn_tile_n": "64",
            "ffn_down_proj_depth": "1",
            "ffn_n_a_tiles_distributed": "8",
            "ffn_n_b_tiles_distributed": "2",
            "ffn_stage_only": "",
            "ffn_gelu_stage": "1",
        },
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_canonical_plots(results_csv, tmp_path)

    assert (tmp_path / "mha_out_proj_latency_by_staging_depth.svg").exists()
    assert (tmp_path / "mha_out_proj_speedup_by_staging_depth.svg").exists()
    assert (tmp_path / "ffn_latency_by_staging_depth.svg").exists()
    assert (tmp_path / "ffn_speedup_by_staging_depth.svg").exists()


def test_render_plot_filters_sequence_lengths_to_256_through_8192(tmp_path):
    results_csv = tmp_path / "results.csv"
    fieldnames = [
        "study_id",
        "family_id",
        "family_label",
        "seq_len",
        "block_kind",
        "source_candidate_index",
        "source_staging_depth",
        "staging_depth",
        "head_dim",
        "num_heads",
        "hidden_size",
        "ffn_dim",
        "warmup_iters",
        "timed_iters",
        "avg_latency_ms",
        "bandwidth_gbps",
        "speedup_vs_depth1",
        "validation_error_count",
        "run_status",
        "is_best_depth",
        "error_message",
        *(
            column
            for columns in CONFIG_COLUMNS_BY_BLOCK_KIND.values()
            for column in columns
        ),
    ]
    rows = []
    for seq_len in ("64", "256", "8192", "16384"):
        rows.append(
            {
                **{field: "" for field in fieldnames},
                "study_id": "memory_tile_staging",
                "family_id": "baseline_768",
                "family_label": "768 / 3072 / 12",
                "seq_len": seq_len,
                "block_kind": "mha_out_proj",
                "source_candidate_index": "0",
                "source_staging_depth": "8",
                "staging_depth": "1",
                "head_dim": "64",
                "num_heads": "12",
                "hidden_size": "768",
                "ffn_dim": "3072",
                "warmup_iters": "1",
                "timed_iters": "2",
                "avg_latency_ms": "10.0",
                "bandwidth_gbps": "5.0",
                "speedup_vs_depth1": "1.0",
                "validation_error_count": "0",
                "run_status": "passed",
                "is_best_depth": "False",
                "mha_out_proj_parallel_seq": "8",
                "mha_out_proj_q_seq_tile": "32",
                "mha_out_proj_kv_seq_tile": "64",
                "mha_out_proj_emb_tile": "96",
                "mha_out_proj_parallel_heads": "1",
                "mha_out_proj_o_proj_acc_depth": "1",
            }
        )
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plot_rows = load_plot_rows(results_csv)
    fig = render_plot(plot_rows, block_kind="mha_out_proj", metric="latency")
    legend = fig.legends[0]
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["256", "8192"]
    panel_titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert panel_titles == ["BERT-Base"]
    plt.close(fig)


def test_memory_tile_staging_sequence_window_is_256_through_8192():
    assert STAGING_SEQUENCE_LENGTHS == (256, 512, 1024, 2048, 4096, 8192)


def test_render_plot_keeps_ffn_dimension_only_for_ffn_titles(tmp_path):
    results_csv = tmp_path / "results.csv"
    fieldnames = [
        "study_id",
        "family_id",
        "family_label",
        "seq_len",
        "block_kind",
        "source_candidate_index",
        "source_staging_depth",
        "staging_depth",
        "head_dim",
        "num_heads",
        "hidden_size",
        "ffn_dim",
        "warmup_iters",
        "timed_iters",
        "avg_latency_ms",
        "bandwidth_gbps",
        "speedup_vs_depth1",
        "validation_error_count",
        "run_status",
        "is_best_depth",
        "error_message",
        *(
            column
            for columns in CONFIG_COLUMNS_BY_BLOCK_KIND.values()
            for column in columns
        ),
    ]
    rows = []
    for family_id, num_heads, hidden_size, ffn_dim in (
        ("baseline_768", "12", "768", "3072"),
        ("baseline_1024", "16", "1024", "4096"),
    ):
        rows.append(
            {
                **{field: "" for field in fieldnames},
                "study_id": "memory_tile_staging",
                "family_id": family_id,
                "family_label": "",
                "seq_len": "512",
                "block_kind": "ffn",
                "source_candidate_index": "0",
                "source_staging_depth": "8",
                "staging_depth": "1",
                "head_dim": "64",
                "num_heads": num_heads,
                "hidden_size": hidden_size,
                "ffn_dim": ffn_dim,
                "warmup_iters": "1",
                "timed_iters": "2",
                "avg_latency_ms": "10.0",
                "bandwidth_gbps": "5.0",
                "speedup_vs_depth1": "1.0",
                "validation_error_count": "0",
                "run_status": "passed",
                "is_best_depth": "False",
                "ffn_num_aie_columns": "8",
                "ffn_b_col_maj": "False",
                "ffn_c_col_maj": "False",
                "ffn_tile_m": "64",
                "ffn_tile_k": "64" if family_id == "baseline_768" else "128",
                "ffn_tile_n": "64",
                "ffn_down_proj_depth": "1",
                "ffn_n_a_tiles_distributed": "8",
                "ffn_n_b_tiles_distributed": "2",
                "ffn_stage_only": "",
                "ffn_gelu_stage": "1",
            }
        )

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plot_rows = load_plot_rows(results_csv)
    fig = render_plot(plot_rows, block_kind="ffn", metric="latency")
    panel_titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert panel_titles == [
        "BERT-Base",
        "BERT-Large",
    ]
    plt.close(fig)
