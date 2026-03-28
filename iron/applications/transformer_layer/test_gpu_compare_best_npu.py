# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from iron.applications.transformer_layer.gpu_compare_best_npu import (
    load_compare_config,
    select_best_npu_rows,
)
from iron.applications.transformer_layer.src.analysis import (
    load_compare_config as structured_load_compare_config,
)
from iron.applications.transformer_layer.src.analysis import (
    select_best_npu_rows as structured_select_best_npu_rows,
)


def test_gpu_compare_restructure_preserves_legacy_imports_and_selection(
    tmp_path: Path,
):
    assert load_compare_config is structured_load_compare_config
    assert select_best_npu_rows is structured_select_best_npu_rows

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
