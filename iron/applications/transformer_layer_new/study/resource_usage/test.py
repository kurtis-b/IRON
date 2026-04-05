#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer_new.study.resource_usage import (
    run as resource_usage_run,
)
from iron.applications.transformer_layer_new.study.resource_usage.analysis import (
    parse_input_physical_mlir,
)


def _write_input_physical(prj_dir: Path, content: str) -> None:
    prj_dir.mkdir(parents=True, exist_ok=True)
    (prj_dir / "input_physical.mlir").write_text(content, encoding="utf-8")


def _sample_physical_mlir() -> str:
    return """
module {
  %tile_1_2 = aie.tile(1, 2)
  %tile_2_3 = aie.tile(2, 3)
  %mem_tile_2_1 = aie.tile(2, 1)
  %shim_noc_tile_0_0 = aie.tile(0, 0)
  %shim_noc_tile_1_0 = aie.tile(1, 0)
  %core_1_2 = aie.core(%tile_1_2) { }
  %core_2_3 = aie.core(%tile_2_3) { }
  %buf0 = aie.buffer(%tile_1_2) {sym_name = "buf0"} : memref<4x4xbf16>
  %buf1 = aie.buffer(%mem_tile_2_1) {sym_name = "buf1"} : memref<8xi32>
  aie.shim_dma_allocation @alloc_in(%shim_noc_tile_0_0, MM2S, 1)
  aie.shim_dma_allocation @alloc_out(%shim_noc_tile_1_0, S2MM, 0)
}
"""


def _write_block_results(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "family_id",
        "family_label",
        "seq_len",
        "block_kind",
        "candidate_index",
        "head_dim",
        "num_heads",
        "hidden_size",
        "ffn_dim",
        "is_best",
        "run_status",
        "qkv_proj_tile_m",
        "qkv_proj_tile_k",
        "qkv_proj_tile_n",
        "qkv_proj_parallel_seq",
        "qkv_proj_parallel_emb",
        "mha_out_proj_parallel_seq",
        "mha_out_proj_q_seq_tile",
        "mha_out_proj_kv_seq_tile",
        "mha_out_proj_emb_tile",
        "mha_out_proj_parallel_heads",
        "mha_out_proj_o_proj_acc_depth",
        "addnorm_num_aie_columns",
        "addnorm_tile_size",
        "ffn_num_aie_columns",
        "ffn_b_col_maj",
        "ffn_c_col_maj",
        "ffn_tile_m",
        "ffn_tile_k",
        "ffn_tile_n",
        "ffn_down_proj_depth",
        "ffn_n_a_tiles_distributed",
        "ffn_n_b_tiles_distributed",
        "ffn_stage_only",
        "ffn_gelu_stage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_end_to_end_results(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "run_status",
        "execution_mode",
        "study_case_id",
        "study_case_label",
        "seq_len",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "attention_head_size",
        "warmup_runs",
        "runs_per_sample",
        "selected_candidate_ids_json",
        "selected_config_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_input_physical_mlir_extracts_resource_usage(tmp_path: Path) -> None:
    prj_dir = tmp_path / "artifact.mlir.prj"
    _write_input_physical(prj_dir, _sample_physical_mlir())

    usage = parse_input_physical_mlir(prj_dir / "input_physical.mlir")

    assert usage.compute_tiles_used == 2
    assert usage.aie_tiles_with_buffers == 1
    assert usage.aie_tile_allocated_bytes == 32
    assert usage.mem_tiles_with_buffers == 1
    assert usage.mem_tile_allocated_bytes == 32
    assert usage.shim_tiles_with_mm2s == 1
    assert usage.shim_mm2s_channels_used == 1
    assert usage.shim_tiles_with_s2mm == 1
    assert usage.shim_s2mm_channels_used == 1


def test_default_results_paths_fall_back_to_snapshot(tmp_path: Path) -> None:
    app_root = tmp_path / "iron" / "applications" / "transformer_layer_new"
    snapshot_root = app_root / "results_commitabc"
    block_path = snapshot_root / "block" / "results.csv"
    end_to_end_path = snapshot_root / "end_to_end" / "results_all_power.csv"
    _write_block_results(block_path, [])
    _write_end_to_end_results(end_to_end_path, [])

    assert resource_usage_run.default_block_results_path(app_root) == block_path
    assert (
        resource_usage_run.default_end_to_end_results_path(app_root) == end_to_end_path
    )


def test_find_mode_scope_dir_prefers_stable_then_highest_prefixed(
    tmp_path: Path,
) -> None:
    build_root = tmp_path / "build"
    stable = build_root / "mode_runlist_768_64"
    prefixed_old = build_root / "0001_mode_runlist_768_64"
    prefixed_new = build_root / "0002_mode_runlist_768_64"
    prefixed_old.mkdir(parents=True)
    prefixed_new.mkdir(parents=True)

    assert (
        resource_usage_run._find_mode_scope_dir(
            build_root,
            execution_mode="runlist",
            hidden_size=768,
            seq_len=64,
        )
        == prefixed_new
    )

    stable.mkdir(parents=True)
    assert (
        resource_usage_run._find_mode_scope_dir(
            build_root,
            execution_mode="runlist",
            hidden_size=768,
            seq_len=64,
        )
        == stable
    )


def test_export_dataflow_blocks_dedupes_and_marks_missing_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    block_results = tmp_path / "results.csv"
    row = {
        "family_id": "baseline_768",
        "family_label": "768 / 3072 / 12",
        "seq_len": "64",
        "block_kind": "qkv_proj",
        "candidate_index": "0",
        "head_dim": "64",
        "num_heads": "12",
        "hidden_size": "768",
        "ffn_dim": "3072",
        "is_best": "True",
        "run_status": "passed",
        "qkv_proj_tile_m": "16",
        "qkv_proj_tile_k": "64",
        "qkv_proj_tile_n": "96",
        "qkv_proj_parallel_seq": "4",
        "qkv_proj_parallel_emb": "8",
        "mha_out_proj_parallel_seq": "",
        "mha_out_proj_q_seq_tile": "",
        "mha_out_proj_kv_seq_tile": "",
        "mha_out_proj_emb_tile": "",
        "mha_out_proj_parallel_heads": "",
        "mha_out_proj_o_proj_acc_depth": "",
        "addnorm_num_aie_columns": "",
        "addnorm_tile_size": "",
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
    _write_block_results(block_results, [row, dict(row)])
    monkeypatch.setattr(
        resource_usage_run,
        "_dataflow_block_artifact_specs_from_row",
        lambda row: (resource_usage_run.ArtifactSpec("exact", "missing_qkv.mlir.prj"),),
    )

    rows = resource_usage_run.export_dataflow_block_best_configs(
        block_results_input=block_results,
        family_filter="all",
        seq_len_filter="all",
        build_index=resource_usage_run.build_artifact_index(tmp_path / "missing_build"),
    )

    assert len(rows) == 1
    assert rows[0]["source_row_count"] == 2
    assert rows[0]["artifact_missing"] is True
    assert rows[0]["run_status"] == "missing_artifact"


def test_export_runlist_rows_prefers_mode_scope_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_path = tmp_path / "results_all_power.csv"
    _write_end_to_end_results(
        results_path,
        [
            {
                "backend": "npu",
                "run_status": "passed",
                "execution_mode": "runlist",
                "study_case_id": "baseline_768",
                "study_case_label": "768 / 3072 / 12",
                "seq_len": "64",
                "hidden_size": "768",
                "intermediate_size": "3072",
                "num_attention_heads": "12",
                "attention_head_size": "64",
                "warmup_runs": "1",
                "runs_per_sample": "10",
                "selected_candidate_ids_json": json.dumps({"add": "add_default"}),
                "selected_config_json": json.dumps(
                    {
                        "add": {
                            "size": 49152,
                            "num_aie_columns": 8,
                            "num_channels": 2,
                            "tile_size": 4096,
                        }
                    }
                ),
            }
        ],
    )

    build_root = tmp_path / "build"
    mode_scope = build_root / "mode_runlist_768_64"
    _write_input_physical(
        mode_scope / "encoder_runlist_add_exact.mlir.prj",
        _sample_physical_mlir(),
    )
    monkeypatch.setattr(
        resource_usage_run,
        "_runlist_operator_artifact_specs",
        lambda workload, logical_operator, operator_config: (
            resource_usage_run.ArtifactSpec(
                "exact", "encoder_runlist_add_exact.mlir.prj"
            ),
        ),
    )

    rows = resource_usage_run.export_runlist_selected_ops(
        end_to_end_results_input=results_path,
        family_filter="all",
        seq_len_filter="all",
        build_index=resource_usage_run.build_artifact_index(build_root),
    )

    assert len(rows) == 1
    assert rows[0]["artifact_missing"] is False
    assert rows[0]["logical_operator"] == "add"
    assert rows[0]["artifact_group_dir"] == str(mode_scope)


def test_export_runlist_layer_norm_can_match_hashed_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_path = tmp_path / "results_all_power.csv"
    _write_end_to_end_results(
        results_path,
        [
            {
                "backend": "npu",
                "run_status": "passed",
                "execution_mode": "runlist",
                "study_case_id": "baseline_768",
                "study_case_label": "768 / 3072 / 12",
                "seq_len": "64",
                "hidden_size": "768",
                "intermediate_size": "3072",
                "num_attention_heads": "12",
                "attention_head_size": "64",
                "warmup_runs": "1",
                "runs_per_sample": "10",
                "selected_candidate_ids_json": json.dumps({"ln1": "ln_default"}),
                "selected_config_json": json.dumps(
                    {
                        "ln1": {
                            "size": 49152,
                            "num_aie_columns": 8,
                            "num_channels": 2,
                            "tile_size": 768,
                        }
                    }
                ),
            }
        ],
    )

    build_root = tmp_path / "build"
    mode_scope = build_root / "mode_runlist_768_64"
    _write_input_physical(
        mode_scope / "encoder_runlist_ln1_8c_2ch_49152_768t_deadbeef1234.mlir.prj",
        _sample_physical_mlir(),
    )
    monkeypatch.setattr(
        resource_usage_run,
        "_runlist_operator_artifact_specs",
        lambda workload, logical_operator, operator_config: (
            resource_usage_run.ArtifactSpec(
                "glob", "encoder_runlist_ln1_8c_2ch_49152_768t_*.mlir.prj"
            ),
        ),
    )

    rows = resource_usage_run.export_runlist_selected_ops(
        end_to_end_results_input=results_path,
        family_filter="all",
        seq_len_filter="all",
        build_index=resource_usage_run.build_artifact_index(build_root),
    )

    assert len(rows) == 1
    assert rows[0]["artifact_missing"] is False
    assert rows[0]["artifact_prj_dir"].endswith("deadbeef1234.mlir.prj")
