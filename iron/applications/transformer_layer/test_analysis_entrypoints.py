# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from iron.applications.transformer_layer.annotate_roofline import annotate_results_file
from iron.applications.transformer_layer.calibrate_backend_peaks import (
    write_backend_peak_reference,
)
from iron.applications.transformer_layer.src.analysis import (
    annotate_results_file as structured_annotate_results_file,
)
from iron.applications.transformer_layer.src.analysis import (
    write_backend_peak_reference as structured_write_backend_peak_reference,
)


def test_analysis_entrypoint_restructure_preserves_legacy_imports_and_helpers(
    tmp_path: Path,
):
    assert annotate_results_file is structured_annotate_results_file
    assert write_backend_peak_reference is structured_write_backend_peak_reference

    peak_path = tmp_path / "peak.json"
    write_backend_peak_reference(
        backend="npu",
        peak_ops_per_sec=1.0e12,
        peak_bytes_per_sec=2.0e11,
        output=str(peak_path),
        source_note="unit test",
    )
    assert peak_path.exists()

    input_csv = tmp_path / "suite.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "execution_mode",
                "seq_len",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "attention_head_size",
                "estimated_flops_per_inference",
                "estimated_bytes_per_inference",
                "throughput_flops_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "backend": "npu",
                "execution_mode": "dataflow",
                "seq_len": 64,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "attention_head_size": 64,
                "estimated_flops_per_inference": 1000.0,
                "estimated_bytes_per_inference": 100.0,
                "throughput_flops_per_sec": 500.0,
            }
        )

    output_csv = tmp_path / "annotated.csv"
    annotate_results_file(
        input_csv=str(input_csv),
        peak_reference_path=str(peak_path),
        output_csv=str(output_csv),
    )
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        annotated = list(csv.DictReader(handle))
    assert float(annotated[0]["backend_peak_ops_per_sec"]) == 1.0e12
