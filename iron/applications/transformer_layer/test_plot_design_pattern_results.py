# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from iron.applications.transformer_layer.plot_design_pattern_results import (
    generate_plots,
)
from iron.applications.transformer_layer.src.analysis import (
    generate_plots as structured_generate_plots,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plot_generation_restructure_preserves_legacy_imports_and_outputs(
    tmp_path: Path,
):
    assert generate_plots is structured_generate_plots

    suite_csv = tmp_path / "suite.csv"
    bottleneck_csv = tmp_path / "bottlenecks.csv"
    gpu_csv = tmp_path / "gpu.csv"
    output_dir = tmp_path / "plots"

    _write_csv(
        suite_csv,
        [
            {
                "backend": "npu",
                "run_status": "completed",
                "execution_mode": "dataflow",
                "pattern_label": "dataflow",
                "seq_len": "64",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "avg_latency_ms": "10.0",
                "throughput_flops_per_sec": "1000.0",
                "avg_power_w": "5.0",
                "flops_per_joule": "200.0",
                "backend_pct_of_peak": "0.25",
                "roofline_pct": "0.5",
            },
            {
                "backend": "npu",
                "run_status": "completed",
                "execution_mode": "runlist",
                "pattern_label": "runlist",
                "seq_len": "64",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "avg_latency_ms": "12.0",
                "throughput_flops_per_sec": "900.0",
                "avg_power_w": "6.0",
                "flops_per_joule": "150.0",
                "backend_pct_of_peak": "0.2",
                "roofline_pct": "0.4",
            },
        ],
    )
    _write_csv(
        bottleneck_csv,
        [
            {
                "execution_mode": "dataflow",
                "seq_len": "64",
                "study_case_label": "baseline_768",
                "dominant_component_fraction": "0.5",
                "dominant_component": "block2_mha_out_proj",
            }
        ],
    )
    _write_csv(
        gpu_csv,
        [
            {
                "backend": "gpu",
                "run_status": "completed",
                "execution_mode": "amd_igpu_reference",
                "pattern_label": "amd_igpu_reference",
                "seq_len": "64",
                "study_case_id": "baseline_768",
                "study_case_label": "baseline_768",
                "avg_latency_ms": "8.0",
                "throughput_flops_per_sec": "1100.0",
            }
        ],
    )

    sections = generate_plots(
        input_csv=suite_csv,
        output_dir=output_dir,
        bottleneck_csv=bottleneck_csv,
        gpu_compare_csv=gpu_csv,
    )

    assert "main" in sections
    assert "gpu_compare" in sections
    assert (output_dir / "index.html").exists()
    assert any((output_dir / name).exists() for name in sections["main"])
    assert any((output_dir / name).exists() for name in sections["gpu_compare"])
