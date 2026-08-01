#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from iron.applications.transformer_layer.study import plot_families, regenerate_plots
from iron.applications.transformer_layer.study.end_to_end import (
    export_fixed_winner_candidates,
    plot_dataflow_blocks_vs_pattern,
    plot_selected_component_groups_vs_pattern,
    plot_tps_by_pattern,
    power as power_module,
    run_selected_component_aggregates,
)
from iron.applications.transformer_layer.study.end_to_end import modes
from iron.applications.transformer_layer.study.end_to_end import (
    run_staging_ablation,
)
from iron.applications.transformer_layer.study.end_to_end.cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    FAMILY_SPECS,
    EndToEndWorkload,
    candidate_table_for_case,
    get_case,
    load_default_candidate_payloads,
    mode_operators,
)
from iron.applications.transformer_layer.pattern.runlist.op import (
    resolve_runlist_operator_config,
)
from iron.applications.transformer_layer.study.end_to_end.modes import (
    benchmark_mode,
)
from iron.applications.transformer_layer.study.end_to_end.remeasure_power_only import (
    updated_row_with_power_measurement,
)
from iron.applications.transformer_layer.study.end_to_end.run import (
    build_rows as build_end_to_end_rows,
    candidate_payloads_from_dir,
    iteration_schedule,
)
from iron.applications.transformer_layer.study.end_to_end.run_correctness_spot_checks import (
    build_rows as build_correctness_rows,
    merge_rows as merge_correctness_rows,
)
from iron.applications.transformer_layer.study.end_to_end.run_fairness_repeatability import (
    build_rows as build_fairness_rows,
)
from iron.applications.transformer_layer.study.end_to_end.run_latency_variation import (
    build_rows as build_latency_rows,
    merge_rows as merge_latency_rows,
    summarize_latency_samples,
)
from iron.applications.transformer_layer.study.end_to_end.run_staging_ablation import (
    STAGING_ABLATION_SEQUENCE_LENGTHS,
    ablation_iteration_schedule,
    build_rows as build_staging_rows,
    merge_rows as merge_staging_rows,
)
from iron.applications.transformer_layer.study.end_to_end.select import (
    select_result_rows,
)


def _result_row(
    execution_mode: str,
    *,
    seq_len: str = "64",
    study_case_id: str = "baseline_768",
    workload_variant: str | None = None,
    avg_latency_ms: str = "5.0",
    power_backend: str = "none",
    backend: str = "npu",
    study_case_label: str | None = None,
    hidden_size: str | None = None,
    intermediate_size: str | None = None,
    num_attention_heads: str | None = None,
    selected_candidate_ids: dict[str, str] | None = None,
    selected_config: dict[str, dict[str, object]] | None = None,
    run_status: str = "passed",
    failure_message: str = "",
) -> dict[str, str]:
    family_spec = FAMILY_SPECS.get(study_case_id)
    resolved_workload_variant = (
        workload_variant
        if workload_variant is not None
        else (
            family_spec.workload_variant if family_spec is not None else "encoder_bert"
        )
    )
    resolved_hidden_size = (
        hidden_size
        if hidden_size is not None
        else str(family_spec.hidden_size if family_spec is not None else 768)
    )
    resolved_intermediate_size = (
        intermediate_size
        if intermediate_size is not None
        else str(family_spec.intermediate_size if family_spec is not None else 3072)
    )
    resolved_num_attention_heads = (
        num_attention_heads
        if num_attention_heads is not None
        else str(family_spec.num_attention_heads if family_spec is not None else 12)
    )
    resolved_selected_candidate_ids = (
        {"candidate": execution_mode}
        if selected_candidate_ids is None
        else selected_candidate_ids
    )
    resolved_selected_config = (
        {"operator": {"tile_m": 16}} if selected_config is None else selected_config
    )
    return {
        "study_id": "end_to_end",
        "study_case_id": study_case_id,
        "study_case_label": study_case_label or study_case_id,
        "workload_variant": resolved_workload_variant,
        "backend": backend,
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": seq_len,
        "hidden_size": resolved_hidden_size,
        "intermediate_size": resolved_intermediate_size,
        "num_attention_heads": resolved_num_attention_heads,
        "attention_head_size": str(
            int(resolved_hidden_size) // int(resolved_num_attention_heads)
        ),
        "batch_size": "1",
        "dtype": "bf16",
        "use_bias": "False",
        "weights_source": "synthetic",
        "warmup_runs": "1",
        "runs_per_sample": "100" if int(seq_len) <= 256 else "10",
        "measured_inference_count": "10",
        "latency_sample_count": "10",
        "timed_total_sec": "0.5",
        "avg_latency_ms": avg_latency_ms,
        "min_latency_ms": avg_latency_ms,
        "max_latency_ms": avg_latency_ms,
        "compile_setup_time_ms": "1.0",
        "host_qkv_precompute_ms": "",
        "effective_gflops_per_sec": "1000.0",
        "power_backend": power_backend,
        "avg_power_w": "12.0",
        "min_power_w": "11.0",
        "max_power_w": "13.0",
        "power_sample_count": "6",
        "raw_avg_power_w": "12.5",
        "raw_min_power_w": "10.5",
        "raw_max_power_w": "14.0",
        "raw_power_sample_count": "7",
        "power_std_w": "0.4",
        "raw_power_std_w": "0.7",
        "power_outlier_sample_count": "1",
        "power_outlier_filter_applied": "True",
        "effective_gflops_per_sec_per_watt": "80.0",
        "npu_dispatch_count": "8",
        "npu_unique_instruction_binary_count": "8",
        "npu_unique_xclbin_count": "8",
        "process_model": "in_process",
        "validation_error_count": "0",
        "run_status": run_status,
        "failure_message": failure_message,
        "selected_candidate_ids_json": json.dumps(
            resolved_selected_candidate_ids,
            sort_keys=True,
        ),
        "selected_config_json": json.dumps(
            resolved_selected_config,
            sort_keys=True,
        ),
        "is_best": "False",
    }


def _visible_axes(fig) -> list[object]:
    return [ax for ax in fig.axes if ax.axison]


def _selected_row(
    *,
    study_case_id: str,
    execution_mode: str,
    seq_len: int = 64,
    avg_latency_ms: str = "5.0",
    selected_candidate_ids: dict[str, str] | None = None,
    selected_config: dict[str, dict[str, object]] | None = None,
):
    family_spec = FAMILY_SPECS[study_case_id]
    workload_variant = family_spec.workload_variant
    operators = mode_operators(execution_mode, workload_variant)
    row = _result_row(
        execution_mode,
        seq_len=str(seq_len),
        study_case_id=study_case_id,
        workload_variant=workload_variant,
        avg_latency_ms=avg_latency_ms,
        selected_candidate_ids=(
            selected_candidate_ids
            if selected_candidate_ids is not None
            else {operator: f"{operator}_best" for operator in operators}
        ),
        selected_config=(
            selected_config
            if selected_config is not None
            else {operator: {"tile_m": 16} for operator in operators}
        ),
    )
    return select_result_rows(
        [row],
        family_filter=study_case_id,
        seq_len_filter=str(seq_len),
        mode_filter=execution_mode,
    )[0]


def test_plot_tps_by_pattern_keeps_offload_rows(tmp_path):
    results_path = tmp_path / "results_all_power.csv"
    rows = [
        _result_row("hybrid"),
        _result_row("runlist"),
        _result_row("offload"),
    ]
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = plot_tps_by_pattern.load_plot_rows(results_path)

    assert list(df["execution_mode"].cat.categories) == [
        "hybrid",
        "runlist",
        "offload",
    ]
    assert set(df["execution_mode"].astype(str)) == {"hybrid", "runlist", "offload"}

    fig = plot_tps_by_pattern.render_plot(df, metric="throughput")
    try:
        legend_labels = [text.get_text() for text in fig.legends[0].texts]
        visible_axes = _visible_axes(fig)
    finally:
        plot_tps_by_pattern.plt.close(fig)

    assert legend_labels == ["Coarse runlist", "Runlist", "Offload"]
    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-M"]


def test_plot_tps_by_pattern_after_1024_tokens_filters_short_sequences(tmp_path):
    results_path = tmp_path / "results.csv"
    rows = []
    for seq_len in ("512", "1024", "2048", "4096"):
        rows.extend(
            [
                _result_row("hybrid", seq_len=seq_len),
                _result_row("runlist", seq_len=seq_len),
                _result_row("offload", seq_len=seq_len),
            ]
        )
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = plot_tps_by_pattern.load_plot_rows(results_path)

    fig = plot_tps_by_pattern.render_plot(
        df,
        metric="throughput",
        min_seq_len_exclusive=1024,
    )
    try:
        visible_axes = _visible_axes(fig)
        tick_labels = [text.get_text() for text in visible_axes[0].get_xticklabels()]
    finally:
        plot_tps_by_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-M"]
    assert tick_labels == ["2048", "4096"]
    assert (
        plot_tps_by_pattern.variant_stem(
            "throughput",
            "standard",
            min_seq_len_exclusive=1024,
        )
        == "effective_gflops_per_second_by_pattern_after_1024_tokens"
    )


def test_dataflow_plot_after_1024_tokens_filters_short_sequences(tmp_path):
    results_path = tmp_path / "results.csv"
    tuning_path = tmp_path / "tuning.csv"
    _write_csv(
        results_path,
        [
            {
                "study_case_id": "baseline_768",
                "execution_mode": "hybrid",
                "run_status": "passed",
                "seq_len": seq_len,
                "avg_latency_ms": latency_ms,
            }
            for seq_len, latency_ms in (
                ("512", "8.0"),
                ("1024", "16.0"),
                ("2048", "32.0"),
                ("4096", "64.0"),
            )
        ],
    )
    _write_csv(
        tuning_path,
        [
            {
                "study_case_id": "baseline_768",
                "execution_mode": "hybrid",
                "is_operator_best": "True",
                "run_status": "passed",
                "seq_len": seq_len,
                "avg_latency_ms": latency_ms,
                "internal_operator": "qkv_proj",
            }
            for seq_len, latency_ms in (
                ("512", "2.0"),
                ("1024", "4.0"),
                ("2048", "8.0"),
                ("4096", "16.0"),
            )
        ],
    )

    pattern_df, block_df = plot_dataflow_blocks_vs_pattern.load_plot_rows(
        results_path,
        tuning_path,
    )
    fig = plot_dataflow_blocks_vs_pattern.render_plot(
        pattern_df,
        block_df,
        min_seq_len_exclusive=1024,
    )
    try:
        visible_axes = _visible_axes(fig)
        tick_labels = [text.get_text() for text in visible_axes[0].get_xticklabels()]
    finally:
        plot_dataflow_blocks_vs_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-M"]
    assert tick_labels == ["2048", "4096"]
    assert (
        plot_dataflow_blocks_vs_pattern.variant_stem(
            "standard",
            min_seq_len_exclusive=1024,
        )
        == "dataflow_selected_blocks_vs_pattern_latency_after_1024_tokens"
    )


def test_plot_family_helper_uses_canonical_order_and_short_labels():
    assert plot_families.PLOT_FAMILY_ORDER == (
        "tinybert_512",
        "baseline_768",
        "baseline_1024",
        "gpt2_512",
        "gpt2_small_768",
        "gpt2_medium_1024",
    )
    assert plot_families.ordered_plot_family_ids(workload_variant="encoder_bert") == (
        "tinybert_512",
        "baseline_768",
        "baseline_1024",
    )
    assert plot_families.ordered_plot_family_ids(workload_variant="decoder_gpt2") == (
        "gpt2_512",
        "gpt2_small_768",
        "gpt2_medium_1024",
    )
    assert plot_families.plot_family_label("tinybert_512") == "B-S"
    assert plot_families.plot_family_label("baseline_768") == "B-M"
    assert plot_families.plot_family_label("baseline_1024") == "B-L"
    assert plot_families.plot_family_label("gpt2_512") == "G-S"
    assert plot_families.plot_family_label("gpt2_small_768") == "G-M"
    assert plot_families.plot_family_label("gpt2_medium_1024") == "G-L"
    assert plot_families.plot_family_grid_position("gpt2_512") == (1, 0)


def test_plot_tps_by_pattern_uses_shared_2x3_layout_for_encoder_and_decoder(tmp_path):
    results_path = tmp_path / "results_all_power.csv"
    rows = [
        _result_row("hybrid", study_case_id="tinybert_512"),
        _result_row("hybrid", study_case_id="gpt2_512"),
    ]
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = plot_tps_by_pattern.load_plot_rows(results_path)
    fig = plot_tps_by_pattern.render_plot(df, metric="throughput")
    try:
        visible_titles = [axis.get_title(loc="left") for axis in _visible_axes(fig)]
    finally:
        plot_tps_by_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert visible_titles == ["B-S", "G-S"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_summarize_power_samples_filters_large_outlier_conservatively():
    samples_w = [5.0, 5.1, 5.0, 5.2, 4.9, 5.1, 5.0, 5.2, 5.1, 20.0]

    stats = power_module.summarize_power_samples(samples_w, elapsed_sec=2.0)

    assert stats["power_outlier_filter_applied"] is True
    assert stats["power_outlier_sample_count"] == 1
    assert stats["raw_power_sample_count"] == 10
    assert stats["power_sample_count"] == 9
    assert float(stats["raw_avg_power_w"]) > float(stats["avg_power_w"])
    assert float(stats["raw_power_std_w"]) > float(stats["power_std_w"])
    assert float(stats["energy_j"]) == float(stats["avg_power_w"]) * 2.0


def test_summarize_power_samples_keeps_small_sample_sets_unfiltered():
    samples_w = [5.0, 5.1, 5.0, 20.0]

    stats = power_module.summarize_power_samples(samples_w, elapsed_sec=1.0)

    assert stats["power_outlier_filter_applied"] is False
    assert stats["power_outlier_sample_count"] == 0
    assert stats["raw_power_sample_count"] == 4
    assert stats["power_sample_count"] == 4
    assert float(stats["raw_avg_power_w"]) == float(stats["avg_power_w"])


def test_updated_row_with_power_measurement_copies_raw_and_filtered_fields():
    existing_row = _result_row("hybrid")
    power_result = {
        "power_backend": "turbostat_pkgwatt",
        "avg_power_w": 9.5,
        "min_power_w": 9.1,
        "max_power_w": 10.2,
        "power_sample_count": 12,
        "raw_avg_power_w": 10.0,
        "raw_min_power_w": 9.1,
        "raw_max_power_w": 18.0,
        "raw_power_sample_count": 13,
        "power_std_w": 0.3,
        "raw_power_std_w": 2.2,
        "power_outlier_sample_count": 1,
        "power_outlier_filter_applied": True,
    }

    updated = updated_row_with_power_measurement(existing_row, power_result)

    assert updated["avg_power_w"] == 9.5
    assert updated["raw_avg_power_w"] == 10.0
    assert updated["power_outlier_filter_applied"] is True
    assert updated["power_outlier_sample_count"] == 1
    assert float(updated["effective_gflops_per_sec_per_watt"]) == 105.26315789473684


def test_candidate_table_covers_all_execution_modes():
    payloads = load_default_candidate_payloads()
    assert set(payloads) == set(EXECUTION_MODES)


def test_fixed_winner_exporter_writes_candidate_dir_payloads(tmp_path):
    rows = []
    expected_hybrid_operators = mode_operators("hybrid", "encoder_bert")
    for execution_mode in EXECUTION_MODES:
        operators = mode_operators(execution_mode, "encoder_bert")
        rows.append(
            _result_row(
                execution_mode,
                seq_len="512",
                selected_candidate_ids={
                    operator: f"{operator}_winner" for operator in operators
                },
                selected_config={operator: {"tile_m": 16} for operator in operators},
            )
        )

    payloads = export_fixed_winner_candidates.build_fixed_winner_payloads(rows)
    export_fixed_winner_candidates.write_fixed_winner_payloads(tmp_path, payloads)
    loaded_payloads = candidate_payloads_from_dir(tmp_path)
    table = candidate_table_for_case("baseline_768", 512, payloads=loaded_payloads)

    assert {
        operator: tuple(candidate["candidate_id"] for candidate in candidates)
        for operator, candidates in table["hybrid"].items()
    } == {operator: (f"{operator}_winner",) for operator in expected_hybrid_operators}


def test_decoder_512_family_is_present_in_family_tables_and_candidate_payloads():
    payloads = load_default_candidate_payloads()

    assert "gpt2_512" in FAMILY_IDS
    assert FAMILY_SPECS["gpt2_512"].workload_variant == "decoder_gpt2"
    assert FAMILY_SPECS["gpt2_512"].hidden_size == 512
    assert FAMILY_SPECS["gpt2_512"].intermediate_size == 2048
    assert FAMILY_SPECS["gpt2_512"].num_attention_heads == 8
    for execution_mode in EXECUTION_MODES:
        assert "gpt2_512" in payloads[execution_mode]

    table = candidate_table_for_case("gpt2_512", 64, payloads=payloads)
    assert set(table["hybrid"]) == set(mode_operators("hybrid", "decoder_gpt2"))
    assert set(table["runlist"]) == set(mode_operators("runlist", "decoder_gpt2"))
    assert set(table["offload"]) == set(mode_operators("offload", "decoder_gpt2"))


@torch.no_grad()
def test_hybrid_low_seq_candidates_are_augmented():
    payloads = load_default_candidate_payloads()

    for seq_len in (64, 128, 256):
        table = candidate_table_for_case("baseline_768", seq_len, payloads=payloads)
        hybrid = table["hybrid"]
        assert any(
            candidate["candidate_id"] == "qkv_low_emb"
            for candidate in hybrid["qkv_proj"]
        )
        assert any(
            candidate["candidate_id"] == "mha_low_footprint"
            for candidate in hybrid["mha_out_proj"]
        )
        assert any(
            candidate["candidate_id"] == "ffn_low_cols4" for candidate in hybrid["ffn"]
        )
        assert any(
            candidate["candidate_id"] == "addnorm_cols4"
            for candidate in hybrid["add_norm1"]
        )
        assert any(
            candidate["candidate_id"] == "addnorm_cols4"
            for candidate in hybrid["add_norm2"]
        )


@torch.no_grad()
def test_gpt2_small_long_seq_hybrid_qkv_candidates_include_parallel_emb_6():
    payloads = load_default_candidate_payloads()

    expected = {
        "tile_m": 64,
        "tile_k": 64,
        "tile_n": 64,
        "parallel_seq": 4,
        "parallel_emb": 6,
    }

    for seq_len in (256, 16384):
        table = candidate_table_for_case("gpt2_small_768", seq_len, payloads=payloads)
        assert any(
            candidate["candidate_id"] == "qkv_1" and candidate["config"] == expected
            for candidate in table["hybrid"]["qkv_proj"]
        )


@torch.no_grad()
def test_1024_hidden_hybrid_qkv_candidates_include_k32_n128_variant():
    payloads = load_default_candidate_payloads()

    expected = {
        "tile_m": 64,
        "tile_k": 32,
        "tile_n": 128,
        "parallel_seq": 4,
        "parallel_emb": 8,
    }

    for family in ("baseline_1024", "gpt2_medium_1024"):
        for seq_len in (512, 16384):
            table = candidate_table_for_case(family, seq_len, payloads=payloads)
            assert any(
                candidate["candidate_id"] == "qkv_1" and candidate["config"] == expected
                for candidate in table["hybrid"]["qkv_proj"]
            )


@torch.no_grad()
def test_runlist_low_seq_candidates_are_augmented():
    payloads = load_default_candidate_payloads()

    for seq_len in (64, 128, 256):
        table = candidate_table_for_case("baseline_768", seq_len, payloads=payloads)
        runlist = table["runlist"]
        assert any(
            candidate["candidate_id"] == "qkvo_proj_cols4"
            for candidate in runlist["qkvo_proj"]
        )
        assert any(
            candidate["candidate_id"] == "scores_low_cols"
            for candidate in runlist["attn_scores"]
        )
        assert any(
            candidate["candidate_id"] == "output_low_cols"
            for candidate in runlist["attn_output"]
        )
        assert any(
            candidate["candidate_id"] == "attn_scale_low_footprint"
            for candidate in runlist["attn_scale"]
        )
        assert any(
            candidate["candidate_id"] == "attn_softmax_low_footprint"
            for candidate in runlist["attn_softmax"]
        )
        assert any(
            candidate["candidate_id"] == "up_proj_cols4"
            for candidate in runlist["up_proj"]
        )
        assert any(
            candidate["candidate_id"] == "down_proj_cols4"
            for candidate in runlist["down_proj"]
        )


@torch.no_grad()
def test_runlist_long_attn_output_candidates_include_cols8_k128():
    payloads = load_default_candidate_payloads()

    for family in (
        "tinybert_512",
        "baseline_768",
        "baseline_1024",
        "gpt2_512",
        "gpt2_small_768",
        "gpt2_medium_1024",
    ):
        table = candidate_table_for_case(family, 16384, payloads=payloads)
        assert any(
            candidate["candidate_id"] == "output_cols8_k128_n8"
            and candidate["config"]
            == {
                "tile_m": 64,
                "tile_k": 128,
                "tile_n": 8,
                "num_aie_columns": 8,
            }
            for candidate in table["runlist"]["attn_output"]
        )


@torch.no_grad()
def test_runlist_cols8_k128_attn_output_is_restricted_to_seq16384():
    payloads = load_default_candidate_payloads()

    for seq_len in (512, 4096, 8192):
        table = candidate_table_for_case("baseline_768", seq_len, payloads=payloads)
        assert all(
            candidate["candidate_id"] != "output_cols8_k128_n8"
            for candidate in table["runlist"]["attn_output"]
        )


@torch.no_grad()
def test_offload_candidates_remain_singleton_only():
    payloads = load_default_candidate_payloads()

    for seq_len in (64, 128, 256, 16384):
        table = candidate_table_for_case("baseline_768", seq_len, payloads=payloads)
        assert all(len(candidates) == 1 for candidates in table["offload"].values())


@torch.no_grad()
def test_offload_long_seq_candidates_remain_default():
    payloads = load_default_candidate_payloads()

    expected = ({"candidate_id": "default", "config": {}},)

    for family in ("baseline_768", "gpt2_512", "gpt2_small_768"):
        table = candidate_table_for_case(family, 16384, payloads=payloads)["offload"]
        for candidates in table.values():
            assert candidates == expected


def test_hybrid_selected_component_groups_sum_dataflow_blocks(monkeypatch):
    operator_latencies_ms = {
        "qkv_proj": 1.0,
        "mha_out_proj": 2.0,
        "add_norm1": 3.0,
        "ffn": 4.0,
        "add_norm2": 5.0,
    }

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        operator_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        assert execution_mode == "hybrid"
        latency_ms = operator_latencies_ms[operator_name]
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": latency_ms,
            "min_latency_ms": latency_ms,
            "max_latency_ms": latency_ms,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )

    selected_rows = (
        _selected_row(
            study_case_id="baseline_768",
            execution_mode="hybrid",
            avg_latency_ms="20.0",
        ),
    )
    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        selected_rows,
        warmup_runs=1,
        runs_per_sample=2,
    )

    assert {row["group_label"] for row in detailed_rows} == set(
        run_selected_component_aggregates.HYBRID_GROUP_ORDER
    )

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        selected_rows,
        detailed_rows,
    )
    group_totals = {
        row["group_label"]: float(row["avg_latency_ms"])
        for row in aggregate_rows
        if row["row_kind"] == "isolated_group"
    }
    assert group_totals == {
        "QKV Proj": 1.0,
        "MHA + Output": 2.0,
        "Residual + Norm": 8.0,
        "FFN": 4.0,
    }
    full_pattern_row = next(
        row for row in aggregate_rows if row["row_kind"] == "full_pattern"
    )
    assert float(full_pattern_row["avg_latency_ms"]) == 20.0


def test_runlist_selected_component_groups_sum_expected_operators(monkeypatch):
    operator_latencies_ms = {
        "qkvo_proj": 1.0,
        "k_transpose": 2.0,
        "attn_scores": 3.0,
        "attn_scale": 4.0,
        "causal_mask": 5.0,
        "attn_softmax": 6.0,
        "attn_output": 7.0,
        "add": 8.0,
        "ln1": 9.0,
        "ln2": 10.0,
        "up_proj": 11.0,
        "gelu": 12.0,
        "down_proj": 13.0,
    }

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        operator_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        assert execution_mode == "runlist"
        assert warmup_runs == 2
        assert runs_per_sample == 3
        latency_ms = operator_latencies_ms[operator_name]
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": latency_ms,
            "min_latency_ms": latency_ms,
            "max_latency_ms": latency_ms,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )

    selected_rows = (
        _selected_row(
            study_case_id="gpt2_512",
            execution_mode="runlist",
            avg_latency_ms="30.0",
        ),
    )
    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        selected_rows,
        warmup_runs=2,
        runs_per_sample=3,
    )

    assert {
        row["group_label"]
        for row in detailed_rows
        if row["study_case_id"] == "gpt2_512"
    } == set(run_selected_component_aggregates.RUNLIST_GROUP_ORDER)
    assert (
        next(row for row in detailed_rows if row["component_name"] == "causal_mask")[
            "group_label"
        ]
        == "Causal Mask"
    )

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        selected_rows,
        detailed_rows,
    )
    isolated_rows = [
        row for row in aggregate_rows if row["row_kind"] == "isolated_group"
    ]
    group_totals = {
        row["group_label"]: float(row["avg_latency_ms"]) for row in isolated_rows
    }
    assert group_totals == {
        "LN1": 9.0,
        "QKVO Proj": 1.0,
        "K Transpose": 2.0,
        "Attention Scores": 3.0,
        "Attention Scale": 4.0,
        "Causal Mask": 5.0,
        "Attention Softmax": 6.0,
        "Attention Output": 7.0,
        "Residual Add": 8.0,
        "LN2": 10.0,
        "Up Proj": 11.0,
        "GELU": 12.0,
        "Down Proj": 13.0,
    }
    assert {row["expected_component_count"] for row in isolated_rows} == {1}
    full_pattern_row = next(
        row for row in aggregate_rows if row["row_kind"] == "full_pattern"
    )
    assert full_pattern_row["group_label"] == "Full Pattern"
    assert float(full_pattern_row["avg_latency_ms"]) == 30.0


def test_runlist_selected_component_aggregates_follow_workload_variant():
    encoder_row = _selected_row(
        study_case_id="baseline_768",
        execution_mode="runlist",
    )
    decoder_row = _selected_row(
        study_case_id="gpt2_512",
        execution_mode="runlist",
    )

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        (encoder_row, decoder_row),
        detailed_rows=[],
    )
    isolated_groups_by_family = {
        family_id: {
            row["group_label"]
            for row in aggregate_rows
            if row["row_kind"] == "isolated_group" and row["study_case_id"] == family_id
        }
        for family_id in ("baseline_768", "gpt2_512")
    }

    assert "Causal Mask" not in isolated_groups_by_family["baseline_768"]
    assert "Causal Mask" in isolated_groups_by_family["gpt2_512"]
    assert (
        run_selected_component_aggregates.expected_group_component_count(
            "runlist",
            "encoder_bert",
            "Causal Mask",
        )
        is None
    )
    assert (
        run_selected_component_aggregates.expected_group_component_count(
            "runlist",
            "decoder_gpt2",
            "Causal Mask",
        )
        == 1
    )


def test_offload_selected_component_groups_emit_host_and_gemm_rows(monkeypatch):
    gemm_latencies_ms = {
        "q_proj": 1.0,
        "k_proj": 2.0,
        "v_proj": 3.0,
        "attn_scores": 4.0,
        "attn_output": 5.0,
        "output_proj": 6.0,
        "up_proj": 7.0,
        "down_proj": 8.0,
    }
    host_latencies_ms = [11.0, 12.0, 13.0, 11.0, 12.0, 13.0]

    def fake_benchmark_operator_candidate(
        execution_mode,
        operator_name,
        workload,
        operator_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        assert execution_mode == "offload"
        latency_ms = gemm_latencies_ms[operator_name]
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": latency_ms,
            "min_latency_ms": latency_ms,
            "max_latency_ms": latency_ms,
            "run_status": "passed",
            "failure_message": "",
        }

    def fake_host_group_callables(selected_row, *, seed):
        return {
            "attention_host": lambda: None,
            "residual_norm": lambda: None,
            "activation": lambda: None,
        }

    def fake_benchmark_callable(_timed_callable, *, warmup_runs, runs_per_sample):
        latency_ms = host_latencies_ms.pop(0)
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": latency_ms,
            "min_latency_ms": latency_ms,
            "max_latency_ms": latency_ms,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fake_benchmark_operator_candidate,
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "_offload_host_group_callables",
        fake_host_group_callables,
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "_benchmark_callable",
        fake_benchmark_callable,
    )

    selected_rows = (
        _selected_row(study_case_id="baseline_768", execution_mode="offload"),
        _selected_row(study_case_id="gpt2_512", execution_mode="offload"),
    )
    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        selected_rows,
        warmup_runs=1,
        runs_per_sample=2,
    )

    for family_id in ("baseline_768", "gpt2_512"):
        family_rows = [
            row for row in detailed_rows if row["study_case_id"] == family_id
        ]
        assert {row["group_label"] for row in family_rows} == set(
            run_selected_component_aggregates.OFFLOAD_GROUP_ORDER
        )
        assert {row["component_kind"] for row in family_rows} == {
            "host_group",
            "logical_operator",
        }

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        selected_rows,
        detailed_rows,
    )
    for family_id in ("baseline_768", "gpt2_512"):
        family_groups = {
            row["group_label"]: row
            for row in aggregate_rows
            if row["row_kind"] == "isolated_group" and row["study_case_id"] == family_id
        }
        assert set(family_groups) == {
            "GEMMs (NPU)",
            "Non-linear operations (CPU)",
        }
        assert float(family_groups["GEMMs (NPU)"]["avg_latency_ms"]) == 36.0
        assert (
            float(family_groups["Non-linear operations (CPU)"]["avg_latency_ms"])
            == 36.0
        )
        assert family_groups["GEMMs (NPU)"]["expected_component_count"] == 8
        assert (
            family_groups["Non-linear operations (CPU)"]["expected_component_count"]
            == 3
        )


def _selected_component_tuning_rows(
    selected_row,
    *,
    omitted_operator: str | None = None,
    latency_base_ms: float = 1.0,
) -> dict[tuple[str, str, str, str, int, str], dict[str, object]]:
    rows: dict[tuple[str, str, str, str, int, str], dict[str, object]] = {}
    for index, operator_name in enumerate(
        mode_operators(selected_row.execution_mode, selected_row.workload_variant),
        start=1,
    ):
        if operator_name == omitted_operator:
            continue
        row: dict[str, object] = {
            "study_case_id": selected_row.study_case_id,
            "study_case_label": selected_row.study_case_label,
            "workload_variant": selected_row.workload_variant,
            "execution_mode": selected_row.execution_mode,
            "internal_operator": operator_name,
            "candidate_id": selected_row.selected_candidate_ids[operator_name],
            "seq_len": selected_row.seq_len,
            "warmup_runs": 2,
            "runs_per_sample": 3,
            "latency_sample_count": 3,
            "avg_latency_ms": latency_base_ms + index,
            "min_latency_ms": latency_base_ms + index,
            "max_latency_ms": latency_base_ms + index,
            "run_status": "passed",
            "failure_message": "",
            "operator_config_json": json.dumps(
                selected_row.selected_config[operator_name],
                sort_keys=True,
            ),
            "_source_csv": "/tmp/tuning_all_power.csv",
        }
        rows[run_selected_component_aggregates._tuning_row_key(row)] = row
    return rows


def _selected_component_detail_row(
    selected_row,
    *,
    operator_name: str,
    latency_ms: float = 1.0,
    run_status: str = "passed",
    selected_config: dict[str, object] | None = None,
    warmup_runs: int = 2,
    runs_per_sample: int = 3,
    measurement_source: str = "rebenchmark",
) -> dict[str, object]:
    config = (
        dict(selected_row.selected_config[operator_name])
        if selected_config is None
        else selected_config
    )
    return {
        "study_case_id": selected_row.study_case_id,
        "study_case_label": selected_row.study_case_label,
        "workload_variant": selected_row.workload_variant,
        "execution_mode": selected_row.execution_mode,
        "seq_len": selected_row.seq_len,
        "component_kind": "logical_operator",
        "component_name": operator_name,
        "group_label": run_selected_component_aggregates.runlist_group_label(
            operator_name
        ),
        "selected_candidate_id": selected_row.selected_candidate_ids[operator_name],
        "selected_config_json": json.dumps(config, sort_keys=True),
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "latency_sample_count": runs_per_sample if run_status == "passed" else "",
        "avg_latency_ms": latency_ms if run_status == "passed" else "",
        "min_latency_ms": latency_ms if run_status == "passed" else "",
        "max_latency_ms": latency_ms if run_status == "passed" else "",
        "run_status": run_status,
        "failure_message": "" if run_status == "passed" else "boom",
        "measurement_source": measurement_source,
        "source_csv": "",
        "source_row_status": "",
    }


def test_selected_components_reuse_matching_tuning_rows_without_benchmark(monkeypatch):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    tuning_rows = _selected_component_tuning_rows(selected_row)

    def fail_benchmark(*_args, **_kwargs):
        raise AssertionError("matching tuning rows should avoid rebenchmarking")

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fail_benchmark,
    )

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        npu_source="tuning",
    )

    assert {row["measurement_source"] for row in detailed_rows} == {"tuning"}
    assert all(row["source_row_status"] == "passed" for row in detailed_rows)


def test_selected_components_rebenchmark_source_bypasses_tuning(monkeypatch):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    tuning_rows = _selected_component_tuning_rows(selected_row)
    called_operators: list[str] = []

    def fake_benchmark(
        execution_mode,
        operator_name,
        workload,
        operator_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        called_operators.append(operator_name)
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": 9.0,
            "min_latency_ms": 9.0,
            "max_latency_ms": 9.0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fake_benchmark,
    )

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        npu_source="rebenchmark",
    )

    assert called_operators == list(
        mode_operators("runlist", selected_row.workload_variant)
    )
    assert {row["measurement_source"] for row in detailed_rows} == {"rebenchmark"}


def test_selected_components_tuning_source_marks_missing_without_rebenchmark(
    monkeypatch,
):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    missing_operator = "attn_scores"
    tuning_rows = _selected_component_tuning_rows(
        selected_row,
        omitted_operator=missing_operator,
    )

    def fail_benchmark(*_args, **_kwargs):
        raise AssertionError("tuning source should not rebenchmark misses")

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fail_benchmark,
    )

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        npu_source="tuning",
    )

    missing_row = next(
        row for row in detailed_rows if row["component_name"] == missing_operator
    )
    assert missing_row["run_status"] == "missing_tuning_row"
    assert missing_row["measurement_source"] == "tuning"


def test_selected_components_auto_rebenchmarks_only_missing_tuning_rows(monkeypatch):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    missing_operator = "attn_scores"
    tuning_rows = _selected_component_tuning_rows(
        selected_row,
        omitted_operator=missing_operator,
    )
    called_operators: list[str] = []

    def fake_benchmark(
        execution_mode,
        operator_name,
        workload,
        operator_config,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
    ):
        called_operators.append(operator_name)
        return {
            "latency_sample_count": runs_per_sample,
            "avg_latency_ms": 4.0,
            "min_latency_ms": 4.0,
            "max_latency_ms": 4.0,
            "run_status": "passed",
            "failure_message": "",
        }

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fake_benchmark,
    )

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        npu_source="auto",
    )

    assert called_operators == [missing_operator]
    source_by_operator = {
        row["component_name"]: row["measurement_source"] for row in detailed_rows
    }
    assert source_by_operator[missing_operator] == "rebenchmark"
    assert source_by_operator["qkvo_proj"] == "tuning"


def test_selected_components_reuse_matching_detailed_rows(monkeypatch):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    tuning_rows = _selected_component_tuning_rows(selected_row)
    resume_row = _selected_component_detail_row(
        selected_row,
        operator_name="qkvo_proj",
        latency_ms=22.0,
    )
    resume_row["_source_csv"] = "/tmp/selected_component_timings.csv"
    resume_rows = {
        run_selected_component_aggregates._detail_row_key(resume_row): resume_row
    }

    def fail_benchmark(*_args, **_kwargs):
        raise AssertionError("resume/tuning rows should cover every component")

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "benchmark_operator_candidate",
        fail_benchmark,
    )

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        resume_rows=resume_rows,
        npu_source="auto",
    )

    qkvo_row = next(
        row for row in detailed_rows if row["component_name"] == "qkvo_proj"
    )
    assert qkvo_row["measurement_source"] == "reused_detail"
    assert qkvo_row["avg_latency_ms"] == 22.0


def test_selected_components_ignore_stale_detailed_rows(monkeypatch):
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    tuning_rows = _selected_component_tuning_rows(selected_row)
    stale_resume_row = _selected_component_detail_row(
        selected_row,
        operator_name="qkvo_proj",
        latency_ms=22.0,
        selected_config={"tile_m": 32},
    )
    stale_resume_row["_source_csv"] = "/tmp/selected_component_timings.csv"
    resume_rows = {
        run_selected_component_aggregates._detail_row_key(
            stale_resume_row
        ): stale_resume_row
    }

    detailed_rows = run_selected_component_aggregates.build_detailed_rows(
        (selected_row,),
        warmup_runs=2,
        runs_per_sample=3,
        tuning_rows=tuning_rows,
        resume_rows=resume_rows,
        npu_source="auto",
    )

    qkvo_row = next(
        row for row in detailed_rows if row["component_name"] == "qkvo_proj"
    )
    assert qkvo_row["measurement_source"] == "tuning"
    assert qkvo_row["avg_latency_ms"] != 22.0


def test_selected_component_aggregate_marks_missing_component_incomplete():
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    detailed_rows = [
        _selected_component_detail_row(
            selected_row,
            operator_name="k_transpose",
            latency_ms=1.0,
            measurement_source="tuning",
        )
    ]

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        (selected_row,),
        detailed_rows,
    )

    attention_scale = next(
        row for row in aggregate_rows if row["group_label"] == "Attention Scale"
    )
    assert attention_scale["run_status"] == "missing_group_rows"
    assert attention_scale["is_complete"] is False
    assert "logical_operator:attn_scale" in json.loads(
        attention_scale["missing_components_json"]
    )
    assert json.loads(attention_scale["measurement_sources_json"]) == []


def test_selected_component_aggregate_marks_failed_component_incomplete():
    selected_row = _selected_row(study_case_id="baseline_768", execution_mode="runlist")
    detailed_rows = [
        _selected_component_detail_row(
            selected_row,
            operator_name="attn_scale",
            latency_ms=1.0,
            run_status="failed_exception",
        )
    ]

    aggregate_rows = run_selected_component_aggregates.build_aggregate_rows(
        (selected_row,),
        detailed_rows,
    )

    attention_scale = next(
        row for row in aggregate_rows if row["group_label"] == "Attention Scale"
    )
    assert attention_scale["run_status"] == "failed_component_benchmark"
    assert attention_scale["is_complete"] is False
    assert attention_scale["avg_latency_ms"] == ""


def test_selected_component_aggregate_main_uses_default_lock_path(
    tmp_path, monkeypatch
):
    results_path = tmp_path / "results_all_power.csv"
    detailed_output = tmp_path / "selected_component_timings.csv"
    aggregate_output = tmp_path / "selected_component_aggregates.csv"
    tuning_results = tmp_path / "tuning_all_power.csv"

    monkeypatch.setattr(
        run_selected_component_aggregates, "load_result_rows", lambda _: []
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "select_result_rows",
        lambda *_, **__: [],
    )

    assert (
        run_selected_component_aggregates.main(
            [
                "--results",
                str(results_path),
                "--detailed-output",
                str(detailed_output),
                "--aggregate-output",
                str(aggregate_output),
                "--tuning-results",
                str(tuning_results),
                "--npu-source",
                "rebenchmark",
            ]
        )
        == 0
    )
    assert detailed_output.exists()
    assert aggregate_output.exists()
    assert aggregate_output.with_suffix(".csv.lock").exists()


def test_selected_component_main_checkpoints_detail_rows_incrementally(
    tmp_path, monkeypatch
):
    selected_rows = (
        _selected_row(
            study_case_id="baseline_768",
            execution_mode="runlist",
            seq_len=512,
        ),
        _selected_row(
            study_case_id="baseline_768",
            execution_mode="runlist",
            seq_len=2048,
        ),
    )
    results_path = tmp_path / "results_all_power.csv"
    detailed_output = tmp_path / "selected_component_timings.csv"
    aggregate_output = tmp_path / "selected_component_aggregates.csv"
    tuning_results = tmp_path / "tuning_all_power.csv"
    calls: list[int] = []

    monkeypatch.setattr(
        run_selected_component_aggregates, "load_result_rows", lambda _: []
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "select_result_rows",
        lambda *_, **__: selected_rows,
    )
    monkeypatch.setattr(
        run_selected_component_aggregates, "load_tuning_rows", lambda _: {}
    )

    def fake_build_detailed_rows(selected_subset, **_kwargs):
        calls.append(selected_subset[0].seq_len)
        if len(calls) == 2:
            with detailed_output.open("r", newline="", encoding="utf-8") as handle:
                assert len(list(csv.DictReader(handle))) == 1
        return [
            _selected_component_detail_row(
                selected_subset[0],
                operator_name="qkvo_proj",
                measurement_source="rebenchmark",
            )
        ]

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "build_detailed_rows",
        fake_build_detailed_rows,
    )

    assert (
        run_selected_component_aggregates.main(
            [
                "--results",
                str(results_path),
                "--detailed-output",
                str(detailed_output),
                "--aggregate-output",
                str(aggregate_output),
                "--tuning-results",
                str(tuning_results),
                "--npu-source",
                "rebenchmark",
                "--seq-len",
                "512,2048",
                "--detailed-only",
            ]
        )
        == 0
    )

    with detailed_output.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert calls == [512, 2048]
    assert [row["seq_len"] for row in rows] == ["512", "2048"]
    assert not aggregate_output.exists()


def test_selected_component_main_reuses_checkpoint_without_duplication(
    tmp_path, monkeypatch
):
    selected_row = _selected_row(
        study_case_id="baseline_768",
        execution_mode="runlist",
        seq_len=512,
    )
    results_path = tmp_path / "results_all_power.csv"
    detailed_output = tmp_path / "selected_component_timings.csv"
    aggregate_output = tmp_path / "selected_component_aggregates.csv"
    tuning_results = tmp_path / "tuning_all_power.csv"

    monkeypatch.setattr(
        run_selected_component_aggregates, "load_result_rows", lambda _: []
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "select_result_rows",
        lambda *_, **__: (selected_row,),
    )
    monkeypatch.setattr(
        run_selected_component_aggregates, "load_tuning_rows", lambda _: {}
    )

    first_row = _selected_component_detail_row(
        selected_row,
        operator_name="qkvo_proj",
        measurement_source="rebenchmark",
    )

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "build_detailed_rows",
        lambda *_args, **_kwargs: [first_row],
    )
    assert (
        run_selected_component_aggregates.main(
            [
                "--results",
                str(results_path),
                "--detailed-output",
                str(detailed_output),
                "--aggregate-output",
                str(aggregate_output),
                "--tuning-results",
                str(tuning_results),
                "--npu-source",
                "rebenchmark",
                "--seq-len",
                "512",
                "--detailed-only",
            ]
        )
        == 0
    )

    def fake_reuse_build(_selected_rows, *, resume_rows, **_kwargs):
        assert (
            run_selected_component_aggregates._detail_row_key(first_row) in resume_rows
        )
        reused = dict(first_row)
        reused["measurement_source"] = "reused_detail"
        return [reused]

    monkeypatch.setattr(
        run_selected_component_aggregates,
        "build_detailed_rows",
        fake_reuse_build,
    )
    assert (
        run_selected_component_aggregates.main(
            [
                "--results",
                str(results_path),
                "--detailed-output",
                str(detailed_output),
                "--aggregate-output",
                str(aggregate_output),
                "--tuning-results",
                str(tuning_results),
                "--npu-source",
                "rebenchmark",
                "--seq-len",
                "512",
                "--detailed-only",
            ]
        )
        == 0
    )

    with detailed_output.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["measurement_source"] == "rebenchmark"


def test_selected_component_aggregate_only_reads_checkpoint_without_benchmark(
    tmp_path, monkeypatch
):
    selected_row = _selected_row(
        study_case_id="baseline_768",
        execution_mode="runlist",
        seq_len=512,
    )
    results_path = tmp_path / "results_all_power.csv"
    detailed_output = tmp_path / "selected_component_timings.csv"
    aggregate_output = tmp_path / "selected_component_aggregates.csv"
    tuning_results = tmp_path / "tuning_all_power.csv"
    run_selected_component_aggregates._write_rows(
        detailed_output,
        fieldnames=run_selected_component_aggregates.DETAILED_FIELDNAMES,
        rows=[
            _selected_component_detail_row(
                selected_row,
                operator_name="qkvo_proj",
                measurement_source="rebenchmark",
            )
        ],
    )

    monkeypatch.setattr(
        run_selected_component_aggregates, "load_result_rows", lambda _: []
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "select_result_rows",
        lambda *_, **__: (selected_row,),
    )
    monkeypatch.setattr(
        run_selected_component_aggregates, "load_tuning_rows", lambda _: {}
    )
    monkeypatch.setattr(
        run_selected_component_aggregates,
        "build_detailed_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("aggregate-only must not benchmark")
        ),
    )

    assert (
        run_selected_component_aggregates.main(
            [
                "--results",
                str(results_path),
                "--detailed-output",
                str(detailed_output),
                "--aggregate-output",
                str(aggregate_output),
                "--tuning-results",
                str(tuning_results),
                "--npu-source",
                "rebenchmark",
                "--seq-len",
                "512",
                "--aggregate-only",
            ]
        )
        == 0
    )
    assert aggregate_output.exists()


def test_selected_component_plotter_omits_incomplete_aggregate_rows(tmp_path):
    aggregate_path = tmp_path / "selected_component_aggregates.csv"
    rows = [
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "runlist",
            "seq_len": 64,
            "row_kind": "isolated_group",
            "group_label": group_label,
            "source_component_count": 0 if group_label == "Attention Scale" else 1,
            "avg_latency_ms": "" if group_label == "Attention Scale" else 1.0,
            "run_status": (
                "missing_group_rows" if group_label == "Attention Scale" else "passed"
            ),
            "failure_message": "",
        }
        for group_label in run_selected_component_aggregates.RUNLIST_GROUP_ORDER
    ]
    rows.append(
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "runlist",
            "seq_len": 64,
            "row_kind": "full_pattern",
            "group_label": run_selected_component_aggregates.FULL_PATTERN_GROUP_LABEL,
            "source_component_count": 1,
            "avg_latency_ms": 32.0,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    _write_csv(aggregate_path, rows)

    pattern_df, group_df = plot_selected_component_groups_vs_pattern.load_plot_rows(
        aggregate_path,
        execution_mode="runlist",
    )
    fig = plot_selected_component_groups_vs_pattern.render_plot(
        pattern_df,
        group_df,
        execution_mode="runlist",
    )
    try:
        figure_text = "\n".join(text.get_text() for text in fig.texts)
    finally:
        plot_selected_component_groups_vs_pattern.plt.close(fig)

    assert pattern_df.empty
    assert group_df.empty
    assert pattern_df.attrs["incomplete_comparison_count"] == 1
    assert "Incomplete selected-component data" in figure_text
    assert "Incomplete comparisons omitted: 1" in figure_text


def test_selected_component_plotter_handles_no_rows_for_execution_mode(tmp_path):
    """A results tree with no rows for the requested mode must not raise.

    Selecting with a plain list mask would make pandas read the empty mask as a
    column selection and drop every column, so the next `_pair_key` lookup blew
    up with a KeyError. Sparse trees like a smoke-test run hit this routinely.
    """
    aggregate_path = tmp_path / "selected_component_aggregates.csv"
    _write_csv(
        aggregate_path,
        [
            {
                "study_case_id": "tinybert_512",
                "study_case_label": "tinybert_512",
                "workload_variant": "encoder_bert",
                "execution_mode": "runlist",
                "seq_len": 64,
                "row_kind": "full_pattern",
                "group_label": (
                    run_selected_component_aggregates.FULL_PATTERN_GROUP_LABEL
                ),
                "source_component_count": 1,
                "avg_latency_ms": 32.0,
                "run_status": "passed",
                "failure_message": "",
            }
        ],
    )

    pattern_df, group_df = plot_selected_component_groups_vs_pattern.load_plot_rows(
        aggregate_path,
        execution_mode="offload",
    )

    assert pattern_df.empty
    assert group_df.empty
    fig = plot_selected_component_groups_vs_pattern.render_plot(
        pattern_df,
        group_df,
        execution_mode="offload",
    )
    plot_selected_component_groups_vs_pattern.plt.close(fig)


def test_selected_component_plotter_renders_synthetic_hybrid_aggregate_rows(
    tmp_path,
):
    aggregate_path = tmp_path / "selected_component_aggregates.csv"
    rows = [
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "hybrid",
            "seq_len": 64,
            "row_kind": "isolated_group",
            "group_label": group_label,
            "source_component_count": run_selected_component_aggregates.expected_group_component_count(
                "hybrid",
                "encoder_bert",
                group_label,
            ),
            "avg_latency_ms": index + 1,
            "run_status": "passed",
            "failure_message": "",
        }
        for index, group_label in enumerate(
            run_selected_component_aggregates.HYBRID_GROUP_ORDER
        )
    ]
    rows.append(
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "hybrid",
            "seq_len": 64,
            "row_kind": "full_pattern",
            "group_label": run_selected_component_aggregates.FULL_PATTERN_GROUP_LABEL,
            "source_component_count": 1,
            "avg_latency_ms": 24.0,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    _write_csv(aggregate_path, rows)

    pattern_df, group_df = plot_selected_component_groups_vs_pattern.load_plot_rows(
        aggregate_path,
        execution_mode="hybrid",
    )
    fig = plot_selected_component_groups_vs_pattern.render_plot(
        pattern_df,
        group_df,
        execution_mode="hybrid",
    )
    try:
        visible_axes = _visible_axes(fig)
        legend_labels = [text.get_text() for text in fig.legends[0].texts]
    finally:
        plot_selected_component_groups_vs_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-S"]
    assert legend_labels == [
        *run_selected_component_aggregates.HYBRID_GROUP_ORDER,
        "Coarse Runlist End-to-End",
    ]


def test_selected_component_plotter_renders_synthetic_runlist_aggregate_rows(
    tmp_path,
):
    aggregate_path = tmp_path / "selected_component_aggregates.csv"
    rows = [
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "runlist",
            "seq_len": 64,
            "row_kind": "isolated_group",
            "group_label": group_label,
            "source_component_count": run_selected_component_aggregates.expected_group_component_count(
                "runlist",
                "encoder_bert",
                group_label,
            ),
            "avg_latency_ms": index + 1,
            "run_status": "passed",
            "failure_message": "",
        }
        for index, group_label in enumerate(
            run_selected_component_aggregates.expected_group_order(
                "runlist",
                "encoder_bert",
            )
        )
    ]
    rows.append(
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "runlist",
            "seq_len": 64,
            "row_kind": "full_pattern",
            "group_label": run_selected_component_aggregates.FULL_PATTERN_GROUP_LABEL,
            "source_component_count": 1,
            "avg_latency_ms": 32.0,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    _write_csv(aggregate_path, rows)

    pattern_df, group_df = plot_selected_component_groups_vs_pattern.load_plot_rows(
        aggregate_path,
        execution_mode="runlist",
    )
    fig = plot_selected_component_groups_vs_pattern.render_plot(
        pattern_df,
        group_df,
        execution_mode="runlist",
    )
    try:
        visible_axes = _visible_axes(fig)
        legend_labels = [text.get_text() for text in fig.legends[0].texts]
    finally:
        plot_selected_component_groups_vs_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-S"]
    assert legend_labels == [
        *run_selected_component_aggregates.RUNLIST_GROUP_ORDER,
        "Runlist End-to-End",
    ]


def test_selected_component_plotter_renders_synthetic_offload_aggregate_rows(
    tmp_path,
):
    aggregate_path = tmp_path / "selected_component_aggregates.csv"
    rows = [
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "offload",
            "seq_len": 64,
            "row_kind": "isolated_group",
            "group_label": group_label,
            "source_component_count": run_selected_component_aggregates.expected_group_component_count(
                "offload",
                "encoder_bert",
                group_label,
            ),
            "avg_latency_ms": index + 1,
            "run_status": "passed",
            "failure_message": "",
        }
        for index, group_label in enumerate(
            run_selected_component_aggregates.OFFLOAD_GROUP_ORDER
        )
    ]
    rows.append(
        {
            "study_case_id": "tinybert_512",
            "study_case_label": "tinybert_512",
            "workload_variant": "encoder_bert",
            "execution_mode": "offload",
            "seq_len": 64,
            "row_kind": "full_pattern",
            "group_label": run_selected_component_aggregates.FULL_PATTERN_GROUP_LABEL,
            "source_component_count": 1,
            "avg_latency_ms": 32.0,
            "run_status": "passed",
            "failure_message": "",
        }
    )
    _write_csv(aggregate_path, rows)

    pattern_df, group_df = plot_selected_component_groups_vs_pattern.load_plot_rows(
        aggregate_path,
        execution_mode="offload",
    )
    fig = plot_selected_component_groups_vs_pattern.render_plot(
        pattern_df,
        group_df,
        execution_mode="offload",
    )
    try:
        visible_axes = _visible_axes(fig)
        legend_labels = [text.get_text() for text in fig.legends[0].texts]
    finally:
        plot_selected_component_groups_vs_pattern.plt.close(fig)

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in visible_axes] == ["B-S"]
    assert legend_labels == [
        *run_selected_component_aggregates.OFFLOAD_GROUP_ORDER,
        "Offload End-to-End",
    ]


def test_regenerate_plots_redraws_existing_and_selected_component_suites(
    monkeypatch,
    tmp_path,
):
    results_root = tmp_path / "results"
    end_to_end_dir = results_root / "end_to_end"
    end_to_end_dir.mkdir(parents=True)
    for name in (
        "results_all_power.csv",
        "tuning_all_power.csv",
        "selected_component_aggregates.csv",
    ):
        (end_to_end_dir / name).write_text("header\n", encoding="utf-8")

    rendered_stems: list[str] = []
    end_to_end_calls: list[tuple[str, str, str | None, int | None]] = []
    dataflow_calls: list[tuple[str, str, int | None]] = []
    selected_component_calls: list[tuple[str, str, str, int | None]] = []

    monkeypatch.setattr(
        regenerate_plots,
        "load_end_to_end_rows",
        lambda _path: "pattern_rows",
    )
    monkeypatch.setattr(
        regenerate_plots,
        "render_end_to_end_plot",
        lambda _rows, *, metric, variant, y_scale="log", min_seq_len_exclusive=None: (
            end_to_end_calls.append((metric, variant, y_scale, min_seq_len_exclusive))
            or regenerate_plots.plt.figure()
        ),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "load_dataflow_rows",
        lambda _results_path, _tuning_path: ("pattern_rows", "block_rows"),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "render_dataflow_plot",
        lambda _pattern_rows, _block_rows, *, variant, y_scale, min_seq_len_exclusive: (
            dataflow_calls.append((variant, y_scale, min_seq_len_exclusive))
            or regenerate_plots.plt.figure()
        ),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "load_selected_component_rows",
        lambda _path, *, execution_mode: (
            f"{execution_mode}_pattern",
            f"{execution_mode}_groups",
        ),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "render_selected_component_plot",
        lambda _pattern_rows, _group_rows, *, execution_mode, variant, y_scale, min_seq_len_exclusive: (
            selected_component_calls.append(
                (execution_mode, variant, y_scale, min_seq_len_exclusive)
            )
            or regenerate_plots.plt.figure()
        ),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "_write_figure",
        lambda fig, *, output_dir, stem: (
            rendered_stems.append(stem),
            regenerate_plots.plt.close(fig),
        ),
    )

    regenerate_plots.regenerate_end_to_end_summary_plots(results_root)

    assert len(end_to_end_calls) == 12
    assert len(dataflow_calls) == 8
    assert len(selected_component_calls) == 24
    assert any(
        stem.startswith("hybrid_selected_groups_vs_pattern_latency")
        for stem in rendered_stems
    )
    assert any(
        stem.startswith("runlist_selected_groups_vs_pattern_latency")
        for stem in rendered_stems
    )
    assert any(
        stem.startswith("offload_selected_groups_vs_pattern_latency")
        for stem in rendered_stems
    )


def test_regenerate_plots_can_require_selected_component_aggregates(
    monkeypatch,
    tmp_path,
):
    results_root = tmp_path / "results"
    end_to_end_dir = results_root / "end_to_end"
    end_to_end_dir.mkdir(parents=True)
    for name in ("results_all_power.csv", "tuning_all_power.csv"):
        (end_to_end_dir / name).write_text("header\n", encoding="utf-8")

    monkeypatch.setattr(
        regenerate_plots,
        "load_end_to_end_rows",
        lambda _path: "pattern_rows",
    )
    monkeypatch.setattr(
        regenerate_plots,
        "render_end_to_end_plot",
        lambda *_args, **_kwargs: regenerate_plots.plt.figure(),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "load_dataflow_rows",
        lambda _results_path, _tuning_path: ("pattern_rows", "block_rows"),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "render_dataflow_plot",
        lambda *_args, **_kwargs: regenerate_plots.plt.figure(),
    )
    monkeypatch.setattr(
        regenerate_plots,
        "_write_figure",
        lambda fig, *, output_dir, stem: regenerate_plots.plt.close(fig),
    )

    try:
        regenerate_plots.regenerate_end_to_end_summary_plots(
            results_root,
            require_selected_components=True,
        )
    except FileNotFoundError as exc:
        assert "selected_component_aggregates.csv" in str(exc)
    else:
        raise AssertionError("missing selected-component aggregates should fail")


def test_reset_pattern_run_buffers_respects_operator_opt_out():
    class DummyOperator:
        def __init__(self, *, enable_benchmark_buffer_reset: bool):
            self.enable_benchmark_buffer_reset = enable_benchmark_buffer_reset
            self.reset_buffer_names = ("scratch",)
            self.buffers = {"scratch": 16}
            self.writes: list[tuple[str, int]] = []

        def write_buffer(self, name, data):
            self.writes.append((name, len(data)))

    enabled = DummyOperator(enable_benchmark_buffer_reset=True)
    modes._reset_pattern_run_buffers(enabled)
    assert enabled.writes == [("scratch", 16)]

    disabled = DummyOperator(enable_benchmark_buffer_reset=False)
    modes._reset_pattern_run_buffers(disabled)
    assert disabled.writes == []


def test_warm_up_pattern_runtime_prefers_runlist_warmup():
    class DummyOperator:
        def __init__(self):
            self.calls = 0

        def run_runlist(self):
            self.calls += 1

    operator = DummyOperator()
    assert modes._warm_up_pattern_runtime(operator, 3) is True
    assert operator.calls == 3


def test_warm_up_pattern_runtime_falls_back_when_runlist_missing():
    class DummyOperator:
        pass

    assert modes._warm_up_pattern_runtime(DummyOperator(), 1) is False


def test_reset_pattern_output_buffer_zeros_output_when_present():
    class DummyOperator:
        def __init__(self):
            self.buffers = {"output": 32}
            self.buffer_static_data = {}
            self.writes: list[tuple[str, int]] = []

        def write_buffer(self, name, data):
            self.writes.append((name, len(data)))

    operator = DummyOperator()
    modes._reset_pattern_output_buffer(operator)
    assert operator.writes == [("output", 32)]


def test_candidate_table_covers_every_operator_for_every_mode():
    payloads = load_default_candidate_payloads()
    table = candidate_table_for_case("baseline_768", 64, payloads=payloads)
    assert set(table) == set(EXECUTION_MODES)
    for execution_mode in EXECUTION_MODES:
        assert set(table[execution_mode]) == set(
            mode_operators(execution_mode, "encoder_bert")
        )


def test_baseline_768_runlist_long_k_transpose_uses_large_tile_candidates():
    payloads = load_default_candidate_payloads()
    table = candidate_table_for_case("baseline_768", 512, payloads=payloads)
    candidates = table["runlist"]["k_transpose"]

    assert [candidate["candidate_id"] for candidate in candidates] == [
        "transpose_512_8c2ch_64x96",
        "transpose_512_6c2ch_64x128",
        "transpose_512_6c2ch_64x64",
        "transpose_512_4c2ch_64x96",
    ]


def test_baseline_1024_runlist_uses_64x128_k_transpose_for_long_sequences():
    payloads = load_default_candidate_payloads()
    table = candidate_table_for_case("baseline_1024", 1024, payloads=payloads)
    candidate = table["runlist"]["k_transpose"][0]

    assert candidate["candidate_id"] == "transpose_1024_64x128_2ch"
    assert candidate["config"] == {
        "num_aie_columns": 8,
        "num_channels": 2,
        "m": 64,
        "n": 128,
        "s": 8,
    }


def test_removed_runlist_manifest_prunes_bad_baseline_768_long_4c_transpose():
    payloads = load_default_candidate_payloads()
    table = candidate_table_for_case("baseline_768", 1024, payloads=payloads)

    assert [
        candidate["candidate_id"] for candidate in table["runlist"]["k_transpose"]
    ] == [
        "transpose_1024_8c2ch_64x96",
        "transpose_1024_6c2ch_64x128",
    ]


def test_decoder_runlist_attn_scores_uses_row_batched_k_transpose_layout():
    resolved = resolve_runlist_operator_config(
        64,
        768,
        3072,
        12,
        workload_variant="decoder_gpt2",
    )

    assert resolved["attn_scores"]["batch_A"] == (12, 1)
    assert resolved["attn_scores"]["batch_B"] == (12, 0)
    assert resolved["attn_scores"]["batch_C"] == (12, 0)


def test_staging_ablation_uses_lighter_default_iteration_schedule():
    assert ablation_iteration_schedule(64) == (1, 10)
    assert ablation_iteration_schedule(512) == (1, 5)
    assert ablation_iteration_schedule(4096) == (1, 3)
    assert ablation_iteration_schedule(8192) == (1, 2)


def test_end_to_end_power_defaults_target_at_least_twenty_samples():
    avg_iteration_sec = 0.008
    (
        min_measurement_duration_sec,
        min_power_sample_count,
        target_power_sample_count,
    ) = modes.resolve_power_probe_policy(seq_len=512)
    power_probe_runs = modes.resolve_power_probe_runs(
        avg_iteration_sec=avg_iteration_sec,
        baseline_runs=100,
        min_measurement_duration_sec=min_measurement_duration_sec,
    )
    estimated_window_sec = avg_iteration_sec * float(power_probe_runs)
    sample_interval_sec = modes.resolve_power_sample_interval_sec(
        requested_interval_sec=modes.DEFAULT_POWER_SAMPLE_INTERVAL_SEC,
        estimated_timed_window_sec=estimated_window_sec,
        min_sample_count=target_power_sample_count,
    )

    assert min_measurement_duration_sec >= 2.0
    assert min_power_sample_count >= 16
    assert target_power_sample_count >= 20
    assert estimated_window_sec / sample_interval_sec >= float(
        target_power_sample_count
    )


def test_end_to_end_short_sequences_use_longer_power_probe_policy():
    (
        min_measurement_duration_sec,
        min_power_sample_count,
        target_power_sample_count,
    ) = modes.resolve_power_probe_policy(seq_len=64)

    assert min_measurement_duration_sec >= 2.5
    assert min_power_sample_count >= 20
    assert target_power_sample_count >= 24


def test_end_to_end_long_sequences_use_stronger_power_probe_policy():
    (
        min_measurement_duration_sec,
        min_power_sample_count,
        target_power_sample_count,
    ) = modes.resolve_power_probe_policy(seq_len=8192)

    assert min_measurement_duration_sec >= 3.0
    assert min_power_sample_count >= 20
    assert target_power_sample_count >= 24


def test_power_probe_is_complete_requires_runs_duration_and_samples():
    assert not modes.power_probe_is_complete(
        completed_runs=12,
        min_runs=12,
        elapsed_sec=1.2,
        min_measurement_duration_sec=1.2,
        observed_sample_count=9,
        min_sample_count=10,
    )
    assert not modes.power_probe_is_complete(
        completed_runs=11,
        min_runs=12,
        elapsed_sec=1.2,
        min_measurement_duration_sec=1.2,
        observed_sample_count=10,
        min_sample_count=10,
    )
    assert modes.power_probe_is_complete(
        completed_runs=12,
        min_runs=12,
        elapsed_sec=1.2,
        min_measurement_duration_sec=1.2,
        observed_sample_count=10,
        min_sample_count=10,
    )


def test_iteration_schedule_uses_100_timed_iterations_through_256():
    assert iteration_schedule(64) == (1, 100)
    assert iteration_schedule(128) == (1, 100)
    assert iteration_schedule(256) == (1, 100)
    assert iteration_schedule(512) == (1, 10)
    assert iteration_schedule(4096) == (1, 10)
    assert iteration_schedule(8192) == (1, 5)
    assert iteration_schedule(16384) == (1, 5)


def test_new_benchmark_context_reuses_stable_build_scope():
    first = modes._new_benchmark_context("mode hybrid 768 64 cfg_deadbeef")
    second = modes._new_benchmark_context("mode hybrid 768 64 cfg_deadbeef")
    different = modes._new_benchmark_context("mode runlist 768 64")

    assert first.build_dir == second.build_dir
    assert first.build_dir.name == "mode_hybrid_768_64_cfg_deadbeef"
    assert different.build_dir != first.build_dir


def test_isolated_candidate_scope_changes_with_candidate_config():
    first = modes._new_benchmark_context(
        modes._isolated_candidate_scope("hybrid_qkv_proj", {"tile_m": 16})
    )
    second = modes._new_benchmark_context(
        modes._isolated_candidate_scope("hybrid_qkv_proj", {"tile_m": 32})
    )
    repeated = modes._new_benchmark_context(
        modes._isolated_candidate_scope("hybrid_qkv_proj", {"tile_m": 16})
    )

    assert first.build_dir == repeated.build_dir
    assert first.build_dir != second.build_dir


def test_offload_selected_component_rebenchmark_uses_dynamic_gemm(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_resolve_offload_operator_config(*args, operator_config, **kwargs):
        assert operator_config == {"attn_scores": {"tile_m": 64}}
        return {
            "attn_scores": {
                "M": 256,
                "K": 64,
                "N": 768,
                "tile_m": 64,
                "tile_k": 64,
                "tile_n": 64,
                "num_aie_columns": 8,
                "use_static_weight": False,
                "query_block_size": 256,
                "num_heads": 12,
            }
        }

    def fake_benchmark_dynamic_gemm(gemm_kwargs, **kwargs):
        calls.append({"gemm_kwargs": gemm_kwargs, **kwargs})
        return {"run_status": "passed", "avg_latency_ms": 1.0}

    def unexpected_benchmark_gemm(*args, **kwargs):
        raise AssertionError("offload components should use DynamicGEMM")

    monkeypatch.setattr(
        modes, "resolve_offload_operator_config", fake_resolve_offload_operator_config
    )
    monkeypatch.setattr(modes, "_benchmark_dynamic_gemm", fake_benchmark_dynamic_gemm)
    monkeypatch.setattr(modes, "_benchmark_gemm", unexpected_benchmark_gemm)

    result = modes._benchmark_gemm_from_offload(
        EndToEndWorkload("encoder_bert", 512, 768, 3072, 12),
        {"tile_m": 64},
        operator_name="attn_scores",
        warmup_runs=1,
        runs_per_sample=2,
        seed=3,
    )

    assert result["run_status"] == "passed"
    assert calls == [
        {
            "gemm_kwargs": {
                "M": 256,
                "K": 64,
                "N": 768,
                "tile_m": 64,
                "tile_k": 64,
                "tile_n": 64,
                "num_aie_columns": 8,
            },
            "warmup_runs": 1,
            "runs_per_sample": 2,
            "seed": 3,
            "use_static_weight": False,
            "scope_prefix": "dynamic_offload_attn_scores",
        }
    ]


def test_select_result_rows_parses_selected_config_and_filters_modes():
    rows = [
        _result_row("hybrid", seq_len="64"),
        _result_row("runlist", seq_len="64"),
        _result_row("hybrid", seq_len="64", backend="gpu"),
    ]

    selected_rows = select_result_rows(rows)

    assert len(selected_rows) == 2
    assert tuple(row.execution_mode for row in selected_rows) == ("hybrid", "runlist")
    assert selected_rows[0].selected_config == {"operator": {"tile_m": 16}}


def test_benchmark_mode_capture_latencies_and_forced_reference(monkeypatch):
    fake_output = torch.ones((2, 2), dtype=torch.float32)
    include_output_calls: list[bool] = []

    class FakeContext:
        def compile_all(self):
            return None

        def prepare_runtime(self):
            return None

        def reset_runtime(self):
            return None

    class FakeOperator:
        def __init__(self):
            self.context = FakeContext()

        def forward(self, _input):
            return fake_output

    def fake_generate_golden_reference(*args, **kwargs):
        include_output_calls.append(bool(kwargs["include_output"]))
        return {
            "input": torch.ones((2, 2), dtype=torch.float32),
            "output": fake_output,
            "weights": {},
        }

    monkeypatch.setattr(
        modes, "generate_golden_reference", fake_generate_golden_reference
    )
    monkeypatch.setattr(
        modes, "_build_operator", lambda *args, **kwargs: FakeOperator()
    )
    monkeypatch.setattr(modes, "_metadata_for_operator", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        modes,
        "_measure_power",
        lambda *args, **kwargs: {"power_backend": "none", "avg_power_w": None},
    )
    monkeypatch.setattr(
        modes,
        "require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    result = benchmark_mode(
        "hybrid",
        EndToEndWorkload("encoder_bert", 64, 768, 3072, 12),
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        power_backend="none",
        include_reference_output=True,
        capture_latencies=True,
    )

    assert include_output_calls == [True]
    assert result["run_status"] == "passed"
    assert len(result["latency_samples_ms"]) == 2


def test_benchmark_operator_candidate_uses_in_process_path_for_long_sequences(
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(
        modes,
        "require_npu_power_mode_turbo",
        lambda *, study_name: None,
    )

    def fake_in_process(*args, **kwargs):
        calls.append("in_process")
        return {
            "avg_latency_ms": 1.0,
            "validation_error_count": 0,
            "run_status": "passed",
        }

    def fake_subprocess(*args, **kwargs):
        calls.append("subprocess")
        raise AssertionError("long-sequence candidate tuning should stay in-process")

    monkeypatch.setattr(
        modes, "_benchmark_operator_candidate_in_process", fake_in_process
    )
    monkeypatch.setattr(
        modes,
        "_benchmark_operator_candidate_isolated_subprocess",
        fake_subprocess,
    )

    result = modes.benchmark_operator_candidate(
        "runlist",
        "k_transpose",
        EndToEndWorkload("decoder_gpt2", 8192, 768, 3072, 12),
        {"num_aie_columns": 6, "num_channels": 2, "m": 64, "n": 128, "s": 8},
        warmup_runs=1,
        runs_per_sample=2,
        seed=123,
    )

    assert result["run_status"] == "passed"
    assert calls == ["in_process"]


def test_correctness_rows_only_include_spot_check_sequences(monkeypatch, tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(
        results_input,
        [
            _result_row("hybrid", seq_len="512"),
            _result_row("runlist", seq_len="2048"),
            _result_row("offload", seq_len="8192"),
            _result_row("hybrid", seq_len="4096"),
        ],
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run_correctness_spot_checks.benchmark_mode",
        lambda *args, **kwargs: {
            "avg_latency_ms": 3.5,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_correctness_rows(
        results_input=results_input,
        workload_variant_filter="all",
        family_filter="all",
        mode_filter="all",
        seed=42,
    )

    assert len(rows) == 3
    assert {row["seq_len"] for row in rows} == {512, 2048, 8192}
    assert {row["validation_mode"] for row in rows} == {"reused_end_to_end_validation"}


def test_correctness_rows_reuse_matching_existing_row(tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(results_input, [_result_row("hybrid", seq_len="512")])

    existing_row = {
        "study_id": "end_to_end_correctness_spot_checks",
        "study_case_id": "baseline_768",
        "study_case_label": "baseline_768",
        "workload_variant": "encoder_bert",
        "execution_mode": "hybrid",
        "validation_mode": "exact_reference",
        "seq_len": "512",
        "hidden_size": "768",
        "intermediate_size": "3072",
        "num_attention_heads": "12",
        "attention_head_size": "64",
        "warmup_runs": "1",
        "runs_per_sample": "10",
        "avg_latency_ms": "3.5",
        "latency_sample_count": "10",
        "min_latency_ms": "3.0",
        "max_latency_ms": "4.0",
        "validation_error_count": "0",
        "run_status": "passed",
        "failure_message": "",
        "selected_candidate_ids_json": json.dumps(
            {"candidate": "hybrid"}, sort_keys=True
        ),
        "selected_config_json": json.dumps(
            {"operator": {"tile_m": 16}}, sort_keys=True
        ),
    }

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "benchmark should not run when correctness row is reusable"
        )

    rows = build_correctness_rows(
        results_input=results_input,
        workload_variant_filter="all",
        family_filter="all",
        mode_filter="all",
        seed=42,
        existing_rows={
            ("baseline_768", "encoder_bert", "hybrid", 512): existing_row,
        },
        benchmark_fn=fail_benchmark,
    )

    assert len(rows) == 1
    assert rows[0]["avg_latency_ms"] == "3.5"


def test_correctness_merge_rows_preserves_unrelated_existing_rows():
    existing = {
        ("baseline_768", "encoder_bert", "hybrid", 512): {
            "study_case_id": "baseline_768",
            "workload_variant": "encoder_bert",
            "execution_mode": "hybrid",
            "seq_len": "512",
            "avg_latency_ms": "3.5",
        }
    }
    current = [
        {
            "study_case_id": "baseline_1024",
            "workload_variant": "encoder_bert",
            "execution_mode": "runlist",
            "seq_len": 2048,
            "avg_latency_ms": 4.2,
        }
    ]

    merged = merge_correctness_rows(existing, current)

    assert len(merged) == 2
    assert {str(row["study_case_id"]) for row in merged} == {
        "baseline_768",
        "baseline_1024",
    }


def test_summarize_latency_samples_reports_expected_statistics():
    summary = summarize_latency_samples([1.0, 2.0, 3.0])

    assert summary["sample_count"] == 3
    assert summary["mean_latency_ms"] == 2.0
    assert summary["min_latency_ms"] == 1.0
    assert summary["max_latency_ms"] == 3.0


def test_latency_variation_rows_capture_sample_statistics(monkeypatch, tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(results_input, [_result_row("hybrid", seq_len="64")])

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run_latency_variation.benchmark_mode",
        lambda *args, **kwargs: {
            "latency_samples_ms": [1.0, 2.0, 3.0],
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        },
    )

    rows = build_latency_rows(
        results_input=results_input,
        workload_variant_filter="all",
        family_filter="all",
        mode_filter="all",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
    )

    assert len(rows) == 1
    assert rows[0]["runs_per_sample"] == 100
    assert rows[0]["sample_count"] == 3
    assert rows[0]["mean_latency_ms"] == 2.0


def test_latency_variation_rows_reuse_matching_existing_row(tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(results_input, [_result_row("hybrid", seq_len="64")])

    existing_row = {
        "study_id": "end_to_end_latency_variation",
        "study_case_id": "baseline_768",
        "study_case_label": "baseline_768",
        "workload_variant": "encoder_bert",
        "execution_mode": "hybrid",
        "seq_len": "64",
        "hidden_size": "768",
        "intermediate_size": "3072",
        "num_attention_heads": "12",
        "attention_head_size": "64",
        "warmup_runs": "1",
        "runs_per_sample": "100",
        "sample_count": "3",
        "mean_latency_ms": "2.0",
        "stddev_latency_ms": "0.0",
        "min_latency_ms": "2.0",
        "max_latency_ms": "2.0",
        "validation_error_count": "0",
        "run_status": "passed",
        "failure_message": "",
        "selected_candidate_ids_json": json.dumps(
            {"candidate": "hybrid"}, sort_keys=True
        ),
        "selected_config_json": json.dumps(
            {"operator": {"tile_m": 16}}, sort_keys=True
        ),
    }

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "benchmark should not run when latency-variation row is reusable"
        )

    rows = build_latency_rows(
        results_input=results_input,
        workload_variant_filter="all",
        family_filter="all",
        mode_filter="all",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        existing_rows={
            ("baseline_768", "encoder_bert", "hybrid", 64): existing_row,
        },
        benchmark_fn=fail_benchmark,
    )

    assert len(rows) == 1
    assert rows[0]["mean_latency_ms"] == "2.0"


def test_latency_merge_rows_preserves_unrelated_existing_rows():
    existing = {
        ("baseline_768", "encoder_bert", "hybrid", 64): {
            "study_case_id": "baseline_768",
            "workload_variant": "encoder_bert",
            "execution_mode": "hybrid",
            "seq_len": "64",
            "mean_latency_ms": "2.0",
        }
    }
    current = [
        {
            "study_case_id": "gpt2_medium_1024",
            "workload_variant": "decoder_gpt2",
            "execution_mode": "hybrid",
            "seq_len": 1024,
            "mean_latency_ms": 8.0,
        }
    ]

    merged = merge_latency_rows(existing, current)

    assert len(merged) == 2
    assert {str(row["study_case_id"]) for row in merged} == {
        "baseline_768",
        "gpt2_medium_1024",
    }


def test_staging_ablation_rows_join_staging_and_end_to_end_results(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    result_row = _result_row("hybrid", seq_len="512", avg_latency_ms="8.0")
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
        sort_keys=True,
    )
    result_row["selected_config_json"] = json.dumps(
        {
            "qkv_proj": {"parallel_seq": 4},
            "mha_out_proj": {
                "parallel_seq": 2,
                "q_seq_tile": 32,
                "kv_seq_tile": 64,
                "emb_tile": 96,
                "parallel_heads": 4,
                "o_proj_acc_depth": 2,
            },
            "add_norm1": {"tile_size": 768},
            "ffn": {
                "tile_m": 16,
                "tile_k": 96,
                "tile_n": 96,
                "down_proj_depth": 4,
            },
            "add_norm2": {"tile_size": 768},
        },
        sort_keys=True,
    )
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "2",
                "avg_latency_ms": "3.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "4",
                "avg_latency_ms": "2.0",
                "run_status": "passed",
            },
        ],
    )

    def fake_benchmark(
        execution_mode,
        workload,
        *,
        warmup_runs,
        runs_per_sample,
        seed,
        power_backend,
        operator_config,
        **kwargs,
    ):
        assert execution_mode == "hybrid"
        assert power_backend == "none"
        assert kwargs["scope_key_override"] in {
            "staginggrp_encoder_bert_baseline_768_512_ffn_d1",
            "staginggrp_encoder_bert_baseline_768_512_ffn_d2",
        }
        assert "scope_suffix" not in kwargs or kwargs["scope_suffix"] is None
        depth = int(operator_config["ffn"]["down_proj_depth"])
        return {
            "avg_latency_ms": float(16 // depth),
            "compile_setup_time_ms": 1.0,
            "effective_gflops_per_sec": 100.0 + depth,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="ffn",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        benchmark_fn=fake_benchmark,
    )

    assert len(rows) == 3
    assert [row["staging_depth"] for row in rows] == [1, 2, 4]
    assert all(row["source_staging_depth"] == 4 for row in rows)
    assert rows[0]["speedup_vs_depth1"] == 1.0
    assert rows[-1]["speedup_vs_source_depth"] == 1.0
    assert rows[-1]["avg_latency_ms"] == "8.0"
    assert rows[1]["is_best_depth"] is True


def test_staging_merge_rows_replaces_targeted_scope_and_preserves_others():
    existing = {
        ("baseline_768", "encoder_bert", 512, "ffn", 1): {
            "study_case_id": "baseline_768",
            "workload_variant": "encoder_bert",
            "seq_len": "512",
            "block_kind": "ffn",
            "staging_depth": "1",
            "avg_latency_ms": "16.0",
        },
        ("baseline_768", "encoder_bert", 512, "ffn", 2): {
            "study_case_id": "baseline_768",
            "workload_variant": "encoder_bert",
            "seq_len": "512",
            "block_kind": "ffn",
            "staging_depth": "2",
            "avg_latency_ms": "8.0",
        },
        ("baseline_1024", "encoder_bert", 512, "ffn", 1): {
            "study_case_id": "baseline_1024",
            "workload_variant": "encoder_bert",
            "seq_len": "512",
            "block_kind": "ffn",
            "staging_depth": "1",
            "avg_latency_ms": "20.0",
        },
    }
    current = [
        {
            "study_case_id": "baseline_768",
            "workload_variant": "encoder_bert",
            "seq_len": 512,
            "block_kind": "ffn",
            "staging_depth": 4,
            "avg_latency_ms": 4.0,
        }
    ]

    merged = merge_staging_rows(
        existing,
        current,
        active_scopes={("baseline_768", "encoder_bert", 512, "ffn")},
    )

    assert len(merged) == 2
    assert {
        (
            str(row["study_case_id"]),
            str(row["workload_variant"]),
            int(float(str(row["seq_len"]))),
            str(row["block_kind"]),
            int(float(str(row["staging_depth"]))),
        )
        for row in merged
    } == {
        ("baseline_768", "encoder_bert", 512, "ffn", 4),
        ("baseline_1024", "encoder_bert", 512, "ffn", 1),
    }


def test_gpt2_small_seq128_hybrid_uses_parallel_heads_1():
    payloads = load_default_candidate_payloads()
    mha_candidates = payloads["hybrid"]["gpt2_small_768"]["128"]["mha_out_proj"]

    assert mha_candidates == [
        {
            "candidate_id": "mha_128_ps4_ph1",
            "config": {
                "parallel_seq": 4,
                "q_seq_tile": 32,
                "kv_seq_tile": 32,
                "emb_tile": 96,
                "parallel_heads": 1,
                "o_proj_acc_depth": 8,
                "is_causal": True,
            },
        }
    ]


def test_staging_ablation_rows_reuse_matching_existing_row(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    result_row = _result_row("hybrid", seq_len="512", avg_latency_ms="8.0")
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
        sort_keys=True,
    )
    selected_config = {
        "qkv_proj": {"parallel_seq": 4},
        "mha_out_proj": {
            "parallel_seq": 2,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 4,
            "o_proj_acc_depth": 2,
        },
        "add_norm1": {"tile_size": 768},
        "ffn": {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 4,
        },
        "add_norm2": {"tile_size": 768},
    }
    result_row["selected_config_json"] = json.dumps(selected_config, sort_keys=True)
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "4",
                "avg_latency_ms": "2.0",
                "run_status": "passed",
            },
        ],
    )

    reused_config = dict(selected_config)
    reused_config["ffn"] = dict(selected_config["ffn"])
    depth1_config = dict(selected_config)
    depth1_config["ffn"] = dict(selected_config["ffn"])
    depth1_config["ffn"]["down_proj_depth"] = 1
    depth4_config = dict(selected_config)
    depth4_config["ffn"] = dict(selected_config["ffn"])
    depth4_config["ffn"]["down_proj_depth"] = 4
    existing_row_depth1 = {
        "study_id": "end_to_end_staging_ablation",
        "study_case_id": "baseline_768",
        "study_case_label": "baseline_768",
        "workload_variant": "encoder_bert",
        "execution_mode": "hybrid",
        "seq_len": "512",
        "hidden_size": "768",
        "intermediate_size": "3072",
        "num_attention_heads": "12",
        "attention_head_size": "64",
        "block_kind": "ffn",
        "source_staging_depth": "4",
        "staging_depth": "1",
        "warmup_runs": "1",
        "runs_per_sample": "5",
        "avg_latency_ms": "16.0",
        "compile_setup_time_ms": "1.0",
        "effective_gflops_per_sec": "100.0",
        "speedup_vs_source_depth": "0.25",
        "speedup_vs_depth1": "1.0",
        "latency_sample_count": "10",
        "min_latency_ms": "16.0",
        "max_latency_ms": "16.0",
        "validation_error_count": "0",
        "run_status": "passed",
        "failure_message": "",
        "selected_candidate_ids_json": json.dumps(
            {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
            sort_keys=True,
        ),
        "selected_config_json": json.dumps(depth1_config, sort_keys=True),
        "is_best_depth": "False",
    }
    existing_row_depth4 = {
        **existing_row_depth1,
        "staging_depth": "4",
        "avg_latency_ms": "4.0",
        "effective_gflops_per_sec": "101.0",
        "speedup_vs_source_depth": "1.0",
        "speedup_vs_depth1": "4.0",
        "latency_sample_count": "10",
        "min_latency_ms": "4.0",
        "max_latency_ms": "4.0",
        "selected_config_json": json.dumps(depth4_config, sort_keys=True),
        "is_best_depth": "True",
    }

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "benchmark should not run when staging-ablation row is reusable"
        )

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="ffn",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        existing_rows={
            ("baseline_768", "encoder_bert", 512, "ffn", 1): existing_row_depth1,
            ("baseline_768", "encoder_bert", 512, "ffn", 4): existing_row_depth4,
        },
        benchmark_fn=fail_benchmark,
    )

    assert len(rows) == 2
    assert {row["avg_latency_ms"] for row in rows} == {"16.0", "8.0"}


def test_staging_ablation_does_not_reuse_failed_exception_rows(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    result_row = _result_row("hybrid", seq_len="512", avg_latency_ms="8.0")
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
        sort_keys=True,
    )
    selected_config = {
        "qkv_proj": {"parallel_seq": 4},
        "mha_out_proj": {
            "parallel_seq": 2,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 4,
            "o_proj_acc_depth": 2,
        },
        "add_norm1": {"tile_size": 768},
        "ffn": {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 4,
        },
        "add_norm2": {"tile_size": 768},
    }
    result_row["selected_config_json"] = json.dumps(selected_config, sort_keys=True)
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "4",
                "avg_latency_ms": "2.0",
                "run_status": "passed",
            },
        ],
    )

    failed_config = dict(selected_config)
    failed_config["ffn"] = dict(selected_config["ffn"])
    failed_config["ffn"]["down_proj_depth"] = 1
    failed_row = {
        "study_id": "end_to_end_staging_ablation",
        "study_case_id": "baseline_768",
        "study_case_label": "baseline_768",
        "workload_variant": "encoder_bert",
        "execution_mode": "hybrid",
        "seq_len": "512",
        "hidden_size": "768",
        "intermediate_size": "3072",
        "num_attention_heads": "12",
        "attention_head_size": "64",
        "block_kind": "ffn",
        "source_staging_depth": "4",
        "staging_depth": "1",
        "warmup_runs": "1",
        "runs_per_sample": "5",
        "avg_latency_ms": "99.0",
        "compile_setup_time_ms": "1.0",
        "effective_gflops_per_sec": "100.0",
        "speedup_vs_source_depth": "0.25",
        "speedup_vs_depth1": "1.0",
        "validation_error_count": "0",
        "run_status": "failed_exception",
        "failure_message": "stale failure",
        "selected_candidate_ids_json": json.dumps(
            {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
            sort_keys=True,
        ),
        "selected_config_json": json.dumps(failed_config, sort_keys=True),
        "is_best_depth": "False",
    }
    benchmark_calls = []

    def fake_benchmark(*args, **kwargs):
        benchmark_calls.append(1)
        return {
            "avg_latency_ms": 7.0,
            "compile_setup_time_ms": 1.0,
            "effective_gflops_per_sec": 101.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="ffn",
        warmup_runs=1,
        runs_per_sample=5,
        seed=42,
        existing_rows={
            ("baseline_768", "encoder_bert", 512, "ffn", 1): failed_row,
        },
        benchmark_fn=fake_benchmark,
    )

    assert rows[0]["avg_latency_ms"] == 7.0
    assert benchmark_calls == [1]


def test_staging_ablation_skips_removed_cases_manifest(tmp_path, monkeypatch):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    removed_cases = tmp_path / "removed.csv"
    result_row = _result_row("hybrid", seq_len="8192", avg_latency_ms="800.0")
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0"},
        sort_keys=True,
    )
    result_row["selected_config_json"] = json.dumps(
        {
            "qkv_proj": {"parallel_seq": 4},
            "mha_out_proj": {
                "parallel_seq": 8,
                "q_seq_tile": 32,
                "kv_seq_tile": 64,
                "emb_tile": 96,
                "parallel_heads": 1,
                "o_proj_acc_depth": 8,
            },
            "add_norm1": {"tile_size": 768},
            "ffn": {
                "tile_m": 32,
                "tile_k": 96,
                "tile_n": 48,
                "down_proj_depth": 4,
            },
            "add_norm2": {"tile_size": 768},
        },
        sort_keys=True,
    )
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "8192",
                "block_kind": "mha_out_proj",
                "staging_depth": "1",
                "avg_latency_ms": "10.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "8192",
                "block_kind": "mha_out_proj",
                "staging_depth": "2",
                "avg_latency_ms": "9.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "8192",
                "block_kind": "mha_out_proj",
                "staging_depth": "4",
                "avg_latency_ms": "8.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "8192",
                "block_kind": "mha_out_proj",
                "staging_depth": "8",
                "avg_latency_ms": "7.0",
                "run_status": "passed",
            },
        ],
    )
    _write_csv(
        removed_cases,
        [
            {
                "study_case_id": "baseline_768",
                "seq_len": "8192",
                "block_kind": "mha_out_proj",
                "staging_depth": "1",
                "reason": "known timeout",
            }
        ],
    )
    monkeypatch.setattr(
        run_staging_ablation, "removed_cases_path", lambda: removed_cases
    )

    benchmark_depths: list[int] = []

    def fake_benchmark(*args, **kwargs):
        benchmark_depths.append(
            int(kwargs["operator_config"]["mha_out_proj"]["o_proj_acc_depth"])
        )
        return {
            "avg_latency_ms": 20.0,
            "compile_setup_time_ms": 1.0,
            "effective_gflops_per_sec": 100.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="mha_out_proj",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        benchmark_fn=fake_benchmark,
    )

    assert [row["staging_depth"] for row in rows] == [2, 4, 8]
    assert benchmark_depths == [2, 4]


def test_staging_ablation_rows_reuse_source_depth_from_end_to_end_result(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    result_row = _result_row("hybrid", seq_len="512", avg_latency_ms="8.0")
    result_row["warmup_runs"] = "1"
    result_row["runs_per_sample"] = "100"
    result_row["compile_setup_time_ms"] = "12.5"
    result_row["effective_gflops_per_sec"] = "123.0"
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
        sort_keys=True,
    )
    selected_config = {
        "qkv_proj": {"parallel_seq": 4},
        "mha_out_proj": {
            "parallel_seq": 2,
            "q_seq_tile": 32,
            "kv_seq_tile": 64,
            "emb_tile": 96,
            "parallel_heads": 4,
            "o_proj_acc_depth": 2,
        },
        "add_norm1": {"tile_size": 768},
        "ffn": {
            "tile_m": 16,
            "tile_k": 96,
            "tile_n": 96,
            "down_proj_depth": 4,
        },
        "add_norm2": {"tile_size": 768},
    }
    result_row["selected_config_json"] = json.dumps(selected_config, sort_keys=True)
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "4",
                "avg_latency_ms": "2.0",
                "run_status": "passed",
            },
        ],
    )

    benchmark_calls = []

    def fake_benchmark(*args, **kwargs):
        benchmark_calls.append((args, kwargs))
        operator_config = kwargs["operator_config"]
        return {
            "avg_latency_ms": 16.0,
            "compile_setup_time_ms": 1.0,
            "effective_gflops_per_sec": 100.0,
            "validation_error_count": 0,
            "run_status": "passed",
            "failure_message": "",
        }

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="ffn",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        benchmark_fn=fake_benchmark,
    )

    assert len(rows) == 2
    source_depth_row = next(row for row in rows if row["staging_depth"] == 4)
    assert source_depth_row["avg_latency_ms"] == "8.0"
    assert source_depth_row["compile_setup_time_ms"] == "12.5"
    assert source_depth_row["effective_gflops_per_sec"] == "123.0"
    assert source_depth_row["warmup_runs"] == 1
    assert source_depth_row["runs_per_sample"] == 100
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0][1]["operator_config"]["ffn"]["down_proj_depth"] == 1


def test_staging_ablation_skips_families_missing_staging_results(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    result_row = _result_row(
        "hybrid",
        study_case_id="gpt2_small_768",
        workload_variant="decoder_gpt2",
        seq_len="512",
        avg_latency_ms="8.0",
    )
    result_row["study_case_label"] = "GPT-2 Small"
    result_row["selected_candidate_ids_json"] = json.dumps(
        {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
        sort_keys=True,
    )
    result_row["selected_config_json"] = json.dumps(
        {
            "ln1": {"tile_size": 768},
            "qkv_proj": {"parallel_seq": 4},
            "mha_out_proj": {
                "parallel_seq": 2,
                "q_seq_tile": 32,
                "kv_seq_tile": 32,
                "emb_tile": 96,
                "parallel_heads": 1,
                "o_proj_acc_depth": 8,
                "is_causal": True,
            },
            "add_norm": {"tile_size": 768},
            "ffn": {
                "tile_m": 16,
                "tile_k": 96,
                "tile_n": 96,
                "down_proj_depth": 4,
            },
            "add": {"tile_size": 768},
        },
        sort_keys=True,
    )
    _write_csv(results_input, [result_row])
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "512",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
        ],
    )

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "benchmark should not run when staging results are missing for the family"
        )

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="all",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        benchmark_fn=fail_benchmark,
    )

    assert rows == []


def test_staging_ablation_filters_sequence_lengths_to_256_through_8192(tmp_path):
    results_input = tmp_path / "end_to_end.csv"
    staging_results = tmp_path / "staging.csv"
    allowed_seq_len = str(STAGING_ABLATION_SEQUENCE_LENGTHS[1])
    rows_in = [
        _result_row("hybrid", seq_len="64", avg_latency_ms="8.0"),
        _result_row("hybrid", seq_len=allowed_seq_len, avg_latency_ms="8.0"),
        _result_row("hybrid", seq_len="16384", avg_latency_ms="8.0"),
    ]
    for row in rows_in:
        row["selected_candidate_ids_json"] = json.dumps(
            {"mha_out_proj": "mha_0", "ffn": "ffn_0"},
            sort_keys=True,
        )
        row["selected_config_json"] = json.dumps(
            {
                "qkv_proj": {"parallel_seq": 4},
                "mha_out_proj": {
                    "parallel_seq": 2,
                    "q_seq_tile": 32,
                    "kv_seq_tile": 64,
                    "emb_tile": 96,
                    "parallel_heads": 1,
                    "o_proj_acc_depth": 4,
                },
                "add_norm1": {"tile_size": 768},
                "ffn": {
                    "tile_m": 16,
                    "tile_k": 96,
                    "tile_n": 96,
                    "down_proj_depth": 1,
                },
                "add_norm2": {"tile_size": 768},
            },
            sort_keys=True,
        )
    _write_csv(results_input, rows_in)
    _write_csv(
        staging_results,
        [
            {
                "family_id": "baseline_768",
                "seq_len": "64",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": allowed_seq_len,
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
            {
                "family_id": "baseline_768",
                "seq_len": "16384",
                "block_kind": "ffn",
                "staging_depth": "1",
                "avg_latency_ms": "4.0",
                "run_status": "passed",
            },
        ],
    )

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("no benchmark should run for the source-depth-only case")

    rows = build_staging_rows(
        results_input=results_input,
        staging_results=staging_results,
        workload_variant_filter="all",
        family_filter="all",
        seq_len_filter="all",
        block_filter="ffn",
        warmup_runs=None,
        runs_per_sample=None,
        seed=42,
        benchmark_fn=fail_benchmark,
    )

    assert [row["seq_len"] for row in rows] == [int(allowed_seq_len)]


def test_fairness_rows_report_all_modes_and_new_schedule(tmp_path):
    results_input = tmp_path / "results.csv"
    _write_csv(
        results_input,
        [
            _result_row("hybrid", seq_len="64", power_backend="none"),
            _result_row("runlist", seq_len="64", power_backend="turbostat_pkgwatt"),
        ],
    )

    latency_variation_input = tmp_path / "latency_variation.csv"
    _write_csv(
        latency_variation_input,
        [
            {
                "study_id": "end_to_end_latency_variation",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "workload_variant": "encoder_bert",
                "execution_mode": "hybrid",
                "seq_len": "64",
                "hidden_size": "768",
                "intermediate_size": "3072",
                "num_attention_heads": "12",
                "attention_head_size": "64",
                "warmup_runs": "1",
                "runs_per_sample": "100",
                "sample_count": "3",
                "mean_latency_ms": "10.0",
                "stddev_latency_ms": "1.0",
                "min_latency_ms": "9.0",
                "max_latency_ms": "11.0",
                "validation_error_count": "0",
                "run_status": "passed",
                "failure_message": "",
                "selected_candidate_ids_json": "{}",
                "selected_config_json": "{}",
            }
        ],
    )

    rows = build_fairness_rows(results_input, latency_variation_input)

    assert {row["execution_mode"] for row in rows} == set(EXECUTION_MODES)
    assert {row["workload_variant"] for row in rows} == {
        "encoder_bert",
        "decoder_gpt2",
    }
    schedule = json.loads(rows[0]["iteration_schedule_json"])
    assert schedule["64-256"]["runs_per_sample"] == 100
    assert schedule["4096"]["runs_per_sample"] == 10
    assert schedule["8192-16384"]["runs_per_sample"] == 5
    hybrid_encoder_row = next(
        row
        for row in rows
        if row["workload_variant"] == "encoder_bert"
        and row["execution_mode"] == "hybrid"
    )
    assert hybrid_encoder_row["latency_variation_row_count"] == 1
    assert hybrid_encoder_row["latency_variation_sample_count_min"] == 3
    assert hybrid_encoder_row["latency_variation_max_cv_pct"] == 10.0
    assert hybrid_encoder_row["selected_candidate_source"]


def test_build_rows_reuses_matching_tuning_and_final_rows(monkeypatch):
    case = get_case("baseline_768", 64)
    execution_mode = "hybrid"
    operator_name = "op"
    candidate = {"candidate_id": "cand0", "config": {"tile_m": 16}}
    resolved_config = {"tile_m": 16}
    resolved_config_json = json.dumps(resolved_config, sort_keys=True)
    selected_candidate_ids_json = json.dumps({operator_name: "cand0"}, sort_keys=True)
    selected_config_json = json.dumps({operator_name: resolved_config}, sort_keys=True)

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run.candidate_table_for_case",
        lambda study_case_id, seq_len: {
            "hybrid": {operator_name: [candidate]},
            "runlist": {operator_name: [candidate]},
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run.mode_operators",
        lambda execution_mode, workload_variant: (operator_name,),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run.resolve_mode_operator_config",
        lambda execution_mode, workload, operator_config: {
            operator_name: resolved_config
        },
    )

    def fail_benchmark_operator_candidate(*args, **kwargs):
        raise AssertionError("benchmark_operator_candidate should not be called")

    def fail_benchmark_mode(*args, **kwargs):
        raise AssertionError("benchmark_mode should not be called")

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run.benchmark_operator_candidate",
        fail_benchmark_operator_candidate,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.end_to_end.run.benchmark_mode",
        fail_benchmark_mode,
    )

    tuning_rows, final_rows = build_end_to_end_rows(
        case,
        mode_filter=execution_mode,
        warmup_runs=1,
        runs_per_sample=2,
        seed=42,
        power_backend="none",
        existing_tuning_rows={
            (
                case.study_case_id,
                case.workload_variant,
                execution_mode,
                operator_name,
                case.seq_len,
                "cand0",
            ): {
                "study_id": "end_to_end_tuning",
                "study_case_id": case.study_case_id,
                "study_case_label": case.study_case_label,
                "workload_variant": case.workload_variant,
                "execution_mode": execution_mode,
                "internal_operator": operator_name,
                "candidate_id": "cand0",
                "seq_len": str(case.seq_len),
                "hidden_size": str(case.hidden_size),
                "intermediate_size": str(case.intermediate_size),
                "num_attention_heads": str(case.num_attention_heads),
                "attention_head_size": str(case.attention_head_size),
                "warmup_runs": "1",
                "runs_per_sample": "2",
                "latency_sample_count": "2",
                "avg_latency_ms": "5.0",
                "min_latency_ms": "5.0",
                "max_latency_ms": "5.0",
                "bandwidth_gbps": "",
                "validation_error_count": "0",
                "run_status": "passed",
                "failure_message": "",
                "operator_config_json": resolved_config_json,
                "is_operator_best": "True",
            }
        },
        existing_final_rows={
            (
                case.study_case_id,
                case.workload_variant,
                execution_mode,
                case.seq_len,
            ): {
                "study_id": "end_to_end",
                "study_case_id": case.study_case_id,
                "study_case_label": case.study_case_label,
                "workload_variant": case.workload_variant,
                "backend": "npu",
                "execution_mode": execution_mode,
                "pattern_label": "Coarse runlist",
                "seq_len": str(case.seq_len),
                "hidden_size": str(case.hidden_size),
                "intermediate_size": str(case.intermediate_size),
                "num_attention_heads": str(case.num_attention_heads),
                "attention_head_size": str(case.attention_head_size),
                "batch_size": "1",
                "dtype": "bf16",
                "use_bias": "False",
                "weights_source": "synthetic",
                "warmup_runs": "1",
                "runs_per_sample": "2",
                "measured_inference_count": "2",
                "latency_sample_count": "2",
                "timed_total_sec": "0.1",
                "avg_latency_ms": "5.0",
                "min_latency_ms": "5.0",
                "max_latency_ms": "5.0",
                "compile_setup_time_ms": "1.0",
                "host_qkv_precompute_ms": "",
                "effective_gflops_per_sec": "100.0",
                "power_backend": "none",
                "avg_power_w": "",
                "min_power_w": "",
                "max_power_w": "",
                "power_sample_count": "",
                "effective_gflops_per_sec_per_watt": "",
                "npu_dispatch_count": "8",
                "npu_unique_instruction_binary_count": "8",
                "npu_unique_xclbin_count": "8",
                "process_model": "in_process",
                "validation_error_count": "0",
                "run_status": "passed",
                "failure_message": "",
                "selected_candidate_ids_json": selected_candidate_ids_json,
                "selected_config_json": selected_config_json,
                "is_best": "False",
            }
        },
    )

    assert len(tuning_rows) == 1
    assert tuning_rows[0]["candidate_id"] == "cand0"
    assert tuning_rows[0]["run_status"] == "passed"
    assert len(final_rows) == 1
    assert final_rows[0]["execution_mode"] == execution_mode
    assert final_rows[0]["selected_candidate_ids_json"] == selected_candidate_ids_json
