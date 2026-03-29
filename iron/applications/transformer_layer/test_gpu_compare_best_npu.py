# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from iron.applications.transformer_layer.gpu_compare_best_npu import (
    benchmark_best_npu_vs_gpu,
    load_compare_config,
    select_best_npu_rows,
)
from iron.applications.transformer_layer.src.analysis.gpu_compare_best_npu import (
    benchmark_best_npu_vs_gpu as structured_benchmark_best_npu_vs_gpu,
)
from iron.applications.transformer_layer.src.analysis.gpu_compare_best_npu import (
    load_compare_config as structured_load_compare_config,
)
from iron.applications.transformer_layer.src.analysis.gpu_compare_best_npu import (
    select_best_npu_rows as structured_select_best_npu_rows,
)


def test_gpu_compare_restructure_preserves_legacy_imports_and_selection(
    tmp_path: Path,
):
    assert load_compare_config is structured_load_compare_config
    assert select_best_npu_rows is structured_select_best_npu_rows
    assert benchmark_best_npu_vs_gpu is structured_benchmark_best_npu_vs_gpu

    config_path = tmp_path / "compare.json"
    config_path.write_text(
        json.dumps(
            {
                "reference_npu_csv": "suite.csv",
                "output_csv": "gpu.csv",
                "sampling_schedule": {"64": {"warmup_runs": 1, "runs_per_sample": 2}},
            }
        ),
        encoding="utf-8",
    )
    config = load_compare_config(config_path)
    assert config["reference_npu_csv"] == str((tmp_path / "suite.csv").resolve())
    assert config["output_csv"] == str((tmp_path / "gpu.csv").resolve())
    assert config["sampling_schedule"][64] == {"warmup_runs": 1, "runs_per_sample": 2}

    rows = [
        {"study_case_id": "a", "seq_len": "64", "avg_latency_ms": "5.0"},
        {"study_case_id": "a", "seq_len": "64", "avg_latency_ms": "3.0"},
        {"study_case_id": "a", "seq_len": "128", "avg_latency_ms": "7.0"},
        {"study_case_id": "b", "seq_len": "64", "avg_latency_ms": "4.0"},
    ]
    selected = select_best_npu_rows(rows)
    assert [row["avg_latency_ms"] for row in selected] == ["3.0", "7.0", "4.0"]


def test_benchmark_best_npu_vs_gpu_preserves_reference_npu_topology_metadata(
    tmp_path: Path, monkeypatch
):
    npu_csv = tmp_path / "npu.csv"
    npu_csv.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "backend",
                        "study_case_id",
                        "study_case_label",
                        "execution_mode",
                        "seq_len",
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "batch_size",
                        "dtype",
                        "use_bias",
                        "weights_source",
                        "avg_latency_ms",
                        "run_status",
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
                    ]
                ),
                ",".join(
                    [
                        "npu",
                        "baseline_768",
                        "baseline_768",
                        "dataflow",
                        "64",
                        "768",
                        "3072",
                        "12",
                        "1",
                        "bfloat16",
                        "False",
                        "synthetic",
                        "3.5",
                        "completed",
                        "m64_k64_n16_c8_ps1_ph1_pd1",
                        "shared_runtime_qkv_proj",
                        "q32_kv64_e96_ps1_ph1_acc1",
                        "fused_mha_out_proj",
                        "m32_k96_n64_ps4_pi3_d8_g1",
                        "pipelined_addnorm_ffn_addnorm",
                        "m32_k256_n24_c8_ps2_ph1_pd4",
                        "shared_runtime_qkv_proj_practical",
                        "q32_kv64_e96_ps1_ph6_acc1",
                        "fused_mha_out_proj_practical",
                        "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1",
                        "pipelined_addnorm_ffn_addnorm_practical",
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_benchmark_gpu_layer(**kwargs):
        return [
            {
                "study_id": "gpu_compare",
                "backend": "gpu",
                "execution_mode": "amd_igpu_reference",
                "pattern_label": "amd_igpu_reference",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "batch_size": 1,
                "dtype": "bfloat16",
                "use_bias": False,
                "weights_source": "synthetic",
                "warmup_runs": 1,
                "runs_per_sample": 2,
                "measured_inference_count": 2,
                "timed_total_sec": 0.01,
                "avg_latency_ms": 5.0,
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.src.analysis.gpu_compare_best_npu.benchmark_gpu_layer",
        fake_benchmark_gpu_layer,
    )

    rows = benchmark_best_npu_vs_gpu(
        {
            "reference_npu_csv": str(npu_csv),
            "output_csv": str(tmp_path / "gpu.csv"),
            "study_id": "gpu_compare",
            "execution_mode": "amd_igpu_reference",
            "pattern_label": "amd_igpu_reference",
            "device": "cuda:0",
            "power_backend": "none",
            "power_sample_interval_sec": 0.2,
            "warmup_runs": 1,
            "runs_per_sample": 2,
        }
    )

    row = rows[0]
    assert row["reference_npu_block1_topology_id"] == "m64_k64_n16_c8_ps1_ph1_pd1"
    assert row["reference_npu_block2_topology_family"] == "fused_mha_out_proj"
    assert (
        row["reference_npu_block3_topology_family"] == "pipelined_addnorm_ffn_addnorm"
    )
    assert (
        row["reference_npu_exploration_block1_topology_id"]
        == "m32_k256_n24_c8_ps2_ph1_pd4"
    )
    assert (
        row["reference_npu_exploration_block2_topology_family"]
        == "fused_mha_out_proj_practical"
    )
    assert (
        row["reference_npu_exploration_block3_topology_id"]
        == "cr128_m32_k96_n64_c8_ps4_pi3_d8_g1"
    )
