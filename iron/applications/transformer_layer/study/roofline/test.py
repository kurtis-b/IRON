#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer.study.roofline import run as roofline_run


def _write_csv(
    path: Path, *, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tuning_row(
    *,
    study_case_id: str,
    execution_mode: str,
    logical_operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    candidate_id: str,
    avg_latency_ms: float | None,
    run_status: str,
    operator_config: dict[str, object],
    bandwidth_gbps: float | None = None,
) -> dict[str, object]:
    return {
        "study_case_id": study_case_id,
        "execution_mode": execution_mode,
        "internal_operator": logical_operator,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "candidate_id": candidate_id,
        "avg_latency_ms": avg_latency_ms,
        "bandwidth_gbps": bandwidth_gbps,
        "run_status": run_status,
        "operator_config_json": json.dumps(operator_config, sort_keys=True),
    }


def _resource_usage_row(
    *,
    execution_mode: str,
    study_case_id: str,
    workload_variant: str,
    logical_operator: str,
    seq_len: int,
    candidate_id: str,
    compute_tiles_used: int | None,
    shim_tiles_with_dma: int | None,
) -> dict[str, object]:
    return {
        "execution_mode": execution_mode,
        "study_case_id": study_case_id,
        "workload_variant": workload_variant,
        "logical_operator": logical_operator,
        "seq_len": seq_len,
        "selected_candidate_id": candidate_id,
        "compute_tiles_used": compute_tiles_used,
        "shim_tiles_with_dma": shim_tiles_with_dma,
    }


def _hybrid_selected_config(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, dict[str, object]]:
    size = seq_len * hidden_size
    return {
        "qkv_proj": {
            "M": seq_len,
            "K": hidden_size,
            "N": hidden_size,
        },
        "mha_out_proj": {
            "size": size,
        },
        "add_norm1": {
            "size": size,
            "tile_size": hidden_size,
        },
        "ffn": {
            "M": seq_len,
            "K": hidden_size,
            "N": intermediate_size,
        },
        "add_norm2": {
            "size": size,
            "tile_size": hidden_size,
        },
    }


def _runlist_selected_config(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    query_block_size: int,
) -> dict[str, dict[str, object]]:
    head_dim = hidden_size // num_heads
    block_elems = query_block_size * seq_len
    size = seq_len * hidden_size
    return {
        "qkvo_proj": {"M": seq_len, "K": hidden_size, "N": hidden_size},
        "k_transpose": {"M": seq_len, "N": hidden_size},
        "attn_scores": {"M": query_block_size, "K": head_dim, "N": seq_len},
        "attn_scale": {"size": block_elems},
        "attn_softmax": {"rows": query_block_size, "cols": seq_len},
        "attn_output": {"M": query_block_size, "K": seq_len, "N": head_dim},
        "add": {"size": size},
        "ln1": {"size": size, "tile_size": hidden_size},
        "up_proj": {"M": seq_len, "K": hidden_size, "N": intermediate_size},
        "gelu": {"size": seq_len * intermediate_size},
        "down_proj": {"M": seq_len, "K": intermediate_size, "N": hidden_size},
        "ln2": {"size": size, "tile_size": hidden_size},
    }


def _result_row(
    *,
    study_case_id: str,
    execution_mode: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    avg_latency_ms: float,
    effective_gflops_per_sec: float,
    selected_candidate_ids: dict[str, str],
    selected_config: dict[str, dict[str, object]],
) -> dict[str, object]:
    workload_variant = (
        "decoder_gpt2"
        if study_case_id in {"gpt2_512", "gpt2_small_768", "gpt2_medium_1024"}
        else "encoder_bert"
    )
    return {
        "backend": "npu",
        "run_status": "passed",
        "execution_mode": execution_mode,
        "study_case_id": study_case_id,
        "study_case_label": f"{hidden_size} / {intermediate_size} / {num_attention_heads}",
        "workload_variant": workload_variant,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "attention_head_size": hidden_size // num_attention_heads,
        "warmup_runs": 1,
        "runs_per_sample": 10,
        "avg_latency_ms": avg_latency_ms,
        "effective_gflops_per_sec": effective_gflops_per_sec,
        "selected_candidate_ids_json": json.dumps(
            selected_candidate_ids, sort_keys=True
        ),
        "selected_config_json": json.dumps(selected_config, sort_keys=True),
    }


def test_default_paths_prefer_results_final_when_results_tree_is_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_root = tmp_path / "iron" / "applications" / "transformer_layer"
    final_root = app_root / "results_final"
    (final_root / "end_to_end").mkdir(parents=True)
    (final_root / "memcpy_bandwidth").mkdir(parents=True)
    (final_root / "end_to_end" / "results_all_power.csv").write_text(
        "header\n",
        encoding="utf-8",
    )
    (final_root / "end_to_end" / "tuning_all_power.csv").write_text(
        "header\n",
        encoding="utf-8",
    )
    (final_root / "memcpy_bandwidth" / "results.csv").write_text(
        "header\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(roofline_run, "APP_ROOT", app_root)

    assert roofline_run.default_output_dir() == final_root / "roofline"
    assert (
        roofline_run.default_end_to_end_results_path()
        == final_root / "end_to_end" / "results_all_power.csv"
    )
    assert (
        roofline_run.default_end_to_end_tuning_path()
        == final_root / "end_to_end" / "tuning_all_power.csv"
    )
    assert (
        roofline_run.default_memcpy_results_path()
        == final_root / "memcpy_bandwidth" / "results.csv"
    )


def test_peak_memcpy_bandwidth_prefers_overall_peak_marker() -> None:
    rows = [
        {
            "run_status": "passed",
            "bandwidth_gbps": "90.0",
            "is_overall_peak": "false",
        },
        {
            "run_status": "passed",
            "bandwidth_gbps": "88.0",
            "is_overall_peak": "true",
        },
        {
            "run_status": "failed_exception",
            "bandwidth_gbps": "120.0",
            "is_overall_peak": "false",
        },
    ]

    assert roofline_run.peak_memcpy_bandwidth_gbps(rows) == 88.0


def test_peak_memcpy_bandwidth_by_shim_tiles_groups_rows() -> None:
    rows = [
        {
            "run_status": "passed",
            "bandwidth_gbps": "40.0",
            "num_cores": "4",
            "num_channels": "2",
        },
        {
            "run_status": "passed",
            "bandwidth_gbps": "60.0",
            "num_cores": "8",
            "num_channels": "2",
        },
        {
            "run_status": "passed",
            "bandwidth_gbps": "55.0",
            "num_cores": "8",
            "num_channels": "2",
        },
    ]

    assert roofline_run.peak_memcpy_bandwidth_by_shim_tiles_gbps(rows) == {
        2: 40.0,
        4: 60.0,
    }


def test_theoretical_compute_peak_matches_bf16_device_formula() -> None:
    expected = 32 * 64 * 2 * 1.8
    assert roofline_run.theoretical_compute_peak_gflops_per_sec() == expected
    assert roofline_run.scaled_compute_peak_gflops_per_sec(8) == expected * 8 / 32


def test_runlist_query_block_count_uses_blocked_attention_tile_shape() -> None:
    assert (
        roofline_run._runlist_query_block_count(
            seq_len=16384,
            selected_config={"attn_scores": {"M": 4096}},
        )
        == 4
    )


def test_operator_compute_tiles_used_matches_operator_structure() -> None:
    assert (
        roofline_run.operator_compute_tiles_used(
            execution_mode="hybrid",
            logical_operator="ffn",
            operator_config={"num_aie_columns": 8},
        )
        == 16
    )
    assert (
        roofline_run.operator_compute_tiles_used(
            execution_mode="runlist",
            logical_operator="attn_scale",
            operator_config={"num_aie_columns": 8, "num_channels": 2},
        )
        == 16
    )


def test_build_kernel_points_uses_best_measured_row_and_omits_zero_flop_kernel() -> (
    None
):
    tuning_rows = [
        _tuning_row(
            study_case_id="baseline_768",
            execution_mode="hybrid",
            logical_operator="qkv_proj",
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            candidate_id="qkv_fast",
            avg_latency_ms=0.50,
            run_status="passed",
            operator_config={"M": 64, "K": 768, "N": 768},
            bandwidth_gbps=120.0,
        ),
        _tuning_row(
            study_case_id="baseline_768",
            execution_mode="hybrid",
            logical_operator="qkv_proj",
            seq_len=256,
            hidden_size=768,
            intermediate_size=3072,
            candidate_id="qkv_slow",
            avg_latency_ms=3.00,
            run_status="passed",
            operator_config={"M": 256, "K": 768, "N": 768},
            bandwidth_gbps=140.0,
        ),
        _tuning_row(
            study_case_id="baseline_768",
            execution_mode="runlist",
            logical_operator="k_transpose",
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            candidate_id="kt",
            avg_latency_ms=0.25,
            run_status="passed",
            operator_config={"M": 64, "N": 768},
            bandwidth_gbps=80.0,
        ),
        _tuning_row(
            study_case_id="baseline_768",
            execution_mode="runlist",
            logical_operator="gelu",
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            candidate_id="gelu_default",
            avg_latency_ms=None,
            run_status="skipped_singleton_default",
            operator_config={"size": 64 * 3072},
        ),
    ]

    rows = roofline_run.build_kernel_points(
        tuning_rows=tuning_rows,
        family_filter="baseline_768",
    )

    qkv_row = next(
        row
        for row in rows
        if row["execution_mode"] == "hybrid" and row["logical_operator"] == "qkv_proj"
    )
    assert qkv_row["candidate_id"] == "qkv_fast"
    assert qkv_row["seq_len"] == 64
    assert qkv_row["compute_tiles_used"] == 1
    assert qkv_row["shim_tiles_used"] == 1
    assert float(qkv_row["effective_gflops_per_sec"]) > 0.0

    transpose_row = next(
        row
        for row in rows
        if row["execution_mode"] == "runlist"
        and row["logical_operator"] == "k_transpose"
    )
    assert transpose_row["run_status"] == "passed"
    assert transpose_row["compute_tiles_used"] == 1
    assert transpose_row["shim_tiles_used"] == 1
    assert (
        transpose_row["omission_note"]
        == "zero-flop kernel omitted from log-scale roofline"
    )

    gelu_row = next(
        row
        for row in rows
        if row["execution_mode"] == "runlist" and row["logical_operator"] == "gelu"
    )
    assert gelu_row["run_status"] == "missing_measurement"
    assert "skipped_singleton_default" in str(gelu_row["omission_note"])


def test_build_kernel_points_prefers_resource_usage_tile_counts() -> None:
    tuning_rows = [
        _tuning_row(
            study_case_id="baseline_768",
            execution_mode="hybrid",
            logical_operator="qkv_proj",
            seq_len=64,
            hidden_size=768,
            intermediate_size=3072,
            candidate_id="qkv_fast",
            avg_latency_ms=0.50,
            run_status="passed",
            operator_config={
                "parallel_seq": 1,
                "parallel_emb": 1,
            },
            bandwidth_gbps=120.0,
        ),
    ]
    resource_usage_index = roofline_run.build_resource_usage_index(
        [
            _resource_usage_row(
                execution_mode="hybrid",
                study_case_id="baseline_768",
                workload_variant="encoder_bert",
                logical_operator="qkv_proj",
                seq_len=64,
                candidate_id="qkv_fast",
                compute_tiles_used=16,
                shim_tiles_with_dma=4,
            )
        ]
    )

    rows = roofline_run.build_kernel_points(
        tuning_rows=tuning_rows,
        family_filter="baseline_768",
        resource_usage_index=resource_usage_index,
    )

    qkv_row = next(
        row
        for row in rows
        if row["execution_mode"] == "hybrid" and row["logical_operator"] == "qkv_proj"
    )
    assert qkv_row["compute_tiles_used"] == 16
    assert qkv_row["shim_tiles_used"] == 4


def test_build_implementation_points_keeps_hybrid_rows() -> None:
    seq_len = 16384
    hidden_size = 512
    intermediate_size = 2048
    num_heads = 8
    runlist_config = _runlist_selected_config(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        query_block_size=4096,
    )
    result_rows = [
        _result_row(
            study_case_id="tinybert_512",
            execution_mode="hybrid",
            seq_len=8192,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            avg_latency_ms=190.0,
            effective_gflops_per_sec=980.0,
            selected_candidate_ids={"qkv_proj": "best"},
            selected_config=_hybrid_selected_config(
                seq_len=8192,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            ),
        ),
        _result_row(
            study_case_id="tinybert_512",
            execution_mode="runlist",
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            avg_latency_ms=700.0,
            effective_gflops_per_sec=600.0,
            selected_candidate_ids={"attn_scores": "best"},
            selected_config=runlist_config,
        ),
    ]

    rows = roofline_run.build_implementation_points(
        result_rows=result_rows,
        family_filter="tinybert_512",
    )

    hybrid_row = next(row for row in rows if row["execution_mode"] == "hybrid")
    assert hybrid_row["run_status"] == "passed"
    assert hybrid_row["seq_len"] == 8192
    assert hybrid_row["compute_tiles_used"] == 2
    assert hybrid_row["shim_tiles_used"] == 1

    runlist_row = next(row for row in rows if row["execution_mode"] == "runlist")
    assert runlist_row["run_status"] == "passed"
    assert runlist_row["compute_tiles_used"] == 4
    assert runlist_row["shim_tiles_used"] == 1
    assert float(runlist_row["implementation_byte_count"]) > 0.0

    single_attn_scores_bytes = roofline_run.operator_bytes_and_flops(
        execution_mode="runlist",
        logical_operator="attn_scores",
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        operator_config=runlist_config["attn_scores"],
    )[1]
    assert (
        float(runlist_row["implementation_byte_count"]) > single_attn_scores_bytes * 4.0
    )


def test_build_implementation_points_prefers_resource_usage_tile_counts() -> None:
    result_rows = [
        _result_row(
            study_case_id="tinybert_512",
            execution_mode="hybrid",
            seq_len=8192,
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            avg_latency_ms=190.0,
            effective_gflops_per_sec=980.0,
            selected_candidate_ids={"qkv_proj": "best"},
            selected_config={
                "qkv_proj": {
                    "parallel_seq": 1,
                    "parallel_emb": 1,
                }
            },
        ),
    ]
    resource_usage_index = roofline_run.build_resource_usage_index(
        [
            _resource_usage_row(
                execution_mode="hybrid",
                study_case_id="tinybert_512",
                workload_variant="encoder_bert",
                logical_operator="qkv_proj",
                seq_len=8192,
                candidate_id="best",
                compute_tiles_used=32,
                shim_tiles_with_dma=8,
            )
        ]
    )

    rows = roofline_run.build_implementation_points(
        result_rows=result_rows,
        family_filter="tinybert_512",
        resource_usage_index=resource_usage_index,
    )

    assert len(rows) == 1
    assert rows[0]["compute_tiles_used"] == 32
    assert rows[0]["shim_tiles_used"] == 8


def test_render_roofline_plot_and_main_write_outputs(tmp_path: Path) -> None:
    results_path = tmp_path / "results_all_power.csv"
    tuning_path = tmp_path / "tuning_all_power.csv"
    memcpy_path = tmp_path / "memcpy_results.csv"
    output_dir = tmp_path / "roofline"

    _write_csv(
        results_path,
        fieldnames=[
            "backend",
            "run_status",
            "execution_mode",
            "study_case_id",
            "study_case_label",
            "workload_variant",
            "seq_len",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "attention_head_size",
            "warmup_runs",
            "runs_per_sample",
            "avg_latency_ms",
            "effective_gflops_per_sec",
            "selected_candidate_ids_json",
            "selected_config_json",
        ],
        rows=[
            _result_row(
                study_case_id="tinybert_512",
                execution_mode="hybrid",
                seq_len=512,
                hidden_size=512,
                intermediate_size=2048,
                num_attention_heads=8,
                avg_latency_ms=25.0,
                effective_gflops_per_sec=650.0,
                selected_candidate_ids={"qkv_proj": "best"},
                selected_config=_hybrid_selected_config(
                    seq_len=512,
                    hidden_size=512,
                    intermediate_size=2048,
                ),
            ),
            _result_row(
                study_case_id="tinybert_512",
                execution_mode="runlist",
                seq_len=512,
                hidden_size=512,
                intermediate_size=2048,
                num_attention_heads=8,
                avg_latency_ms=35.0,
                effective_gflops_per_sec=500.0,
                selected_candidate_ids={"qkvo_proj": "best"},
                selected_config=_runlist_selected_config(
                    seq_len=512,
                    hidden_size=512,
                    intermediate_size=2048,
                    num_heads=8,
                    query_block_size=512,
                ),
            ),
        ],
    )
    _write_csv(
        tuning_path,
        fieldnames=[
            "study_case_id",
            "execution_mode",
            "internal_operator",
            "seq_len",
            "hidden_size",
            "intermediate_size",
            "candidate_id",
            "avg_latency_ms",
            "bandwidth_gbps",
            "run_status",
            "operator_config_json",
        ],
        rows=[
            _tuning_row(
                study_case_id="tinybert_512",
                execution_mode="hybrid",
                logical_operator="qkv_proj",
                seq_len=512,
                hidden_size=512,
                intermediate_size=2048,
                candidate_id="qkv",
                avg_latency_ms=1.2,
                run_status="passed",
                operator_config={"M": 512, "K": 512, "N": 512},
                bandwidth_gbps=90.0,
            ),
            _tuning_row(
                study_case_id="tinybert_512",
                execution_mode="runlist",
                logical_operator="qkvo_proj",
                seq_len=512,
                hidden_size=512,
                intermediate_size=2048,
                candidate_id="qkvo",
                avg_latency_ms=1.5,
                run_status="passed",
                operator_config={"M": 512, "K": 512, "N": 512},
                bandwidth_gbps=85.0,
            ),
        ],
    )
    _write_csv(
        memcpy_path,
        fieldnames=["run_status", "bandwidth_gbps", "is_overall_peak"],
        rows=[
            {"run_status": "passed", "bandwidth_gbps": 80.0, "is_overall_peak": False},
            {"run_status": "passed", "bandwidth_gbps": 96.0, "is_overall_peak": True},
        ],
    )

    exit_code = roofline_run.main(
        [
            "--family",
            "tinybert_512",
            "--end-to-end-results-input",
            str(results_path),
            "--end-to-end-tuning-input",
            str(tuning_path),
            "--memcpy-results-input",
            str(memcpy_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    kernel_points = output_dir / "kernel_points.csv"
    implementation_points = output_dir / "implementation_points.csv"
    assert kernel_points.exists()
    assert implementation_points.exists()
    kernel_plots = sorted(output_dir.glob("kernel_roofline_*_tiles.svg"))
    implementation_plots = sorted(
        output_dir.glob("implementation_roofline_*_tiles.svg")
    )
    assert [path.name for path in kernel_plots] == [
        "kernel_roofline_01_compute_tiles_01_shim_tiles.svg",
        "kernel_roofline_04_compute_tiles_01_shim_tiles.svg",
    ]
    assert [path.name for path in implementation_plots] == [
        "implementation_roofline_02_compute_tiles_01_shim_tiles.svg",
        "implementation_roofline_04_compute_tiles_01_shim_tiles.svg",
    ]
    for path in [*kernel_plots, *implementation_plots]:
        assert path.with_suffix(".png").exists()

    kernel_text = kernel_plots[0].read_text(encoding="utf-8")
    implementation_text = implementation_plots[0].read_text(encoding="utf-8")
    assert "Kernel Roofline" in kernel_text
    assert "Implementation Roofline" in implementation_text
    assert "Compute Tile" in kernel_text
    assert "B-S" in kernel_text
    assert "Hybrid" in implementation_text
