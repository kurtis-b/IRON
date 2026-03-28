# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from iron.applications.transformer_layer.analyze_design_pattern_bottlenecks import (
    analyze_results,
)
from iron.applications.transformer_layer.analyze_design_pattern_bottlenecks import (
    build_row_summary as legacy_build_row_summary,
)
from iron.applications.transformer_layer.src.analysis import (
    analyze_results as structured_analyze_results,
)
from iron.applications.transformer_layer.src.analysis import (
    build_row_summary as structured_build_row_summary,
)


def test_bottleneck_analysis_restructure_preserves_legacy_imports_and_behavior(
    tmp_path: Path,
):
    assert legacy_build_row_summary is structured_build_row_summary
    assert analyze_results is structured_analyze_results

    input_csv = tmp_path / "suite.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "study_id",
                "study_case_id",
                "study_case_label",
                "execution_mode",
                "seq_len",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "avg_latency_ms",
                "avg_block1_qkv_proj_latency_ms",
                "avg_block2_mha_out_proj_latency_ms",
                "npu_dispatch_count",
                "npu_unique_instruction_binary_count",
                "npu_unique_xclbin_count",
                "process_model",
                "run_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "study_id": "study",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "execution_mode": "dataflow",
                "seq_len": "64",
                "hidden_size": "768",
                "intermediate_size": "3072",
                "num_attention_heads": "12",
                "avg_latency_ms": "10.0",
                "avg_block1_qkv_proj_latency_ms": "3.0",
                "avg_block2_mha_out_proj_latency_ms": "5.0",
                "npu_dispatch_count": "3",
                "npu_unique_instruction_binary_count": "3",
                "npu_unique_xclbin_count": "3",
                "process_model": "in_process",
                "run_status": "completed",
            }
        )

    summary_rows, aggregates, text_summary = analyze_results(input_csv)
    assert len(summary_rows) == 1
    assert summary_rows[0]["dominant_component"] == "block2_mha_out_proj"
    assert "baseline_768:dataflow" in aggregates
    assert "block2_mha_out_proj dominates" in text_summary
