# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from iron.applications.transformer_layer.gpu_power import (
    parse_rocm_smi_average_power_w,
)
from iron.applications.transformer_layer.measurement_log import (
    MeasurementAuditLogger,
    default_measurement_log_path,
)
from iron.applications.transformer_layer.src.bench import (
    MeasurementAuditLogger as StructuredMeasurementAuditLogger,
)
from iron.applications.transformer_layer.src.bench import (
    default_measurement_log_path as structured_default_measurement_log_path,
)
from iron.applications.transformer_layer.src.bench import (
    parse_rocm_smi_average_power_w as structured_parse_rocm_smi_average_power_w,
)


def test_bench_support_restructure_preserves_legacy_imports_and_behavior(
    tmp_path: Path,
):
    assert default_measurement_log_path is structured_default_measurement_log_path
    assert MeasurementAuditLogger is StructuredMeasurementAuditLogger
    assert parse_rocm_smi_average_power_w is structured_parse_rocm_smi_average_power_w

    output_csv = tmp_path / "suite.csv"
    log_path = default_measurement_log_path(output_csv)
    assert log_path.endswith("suite_measurements.jsonl")

    logger = MeasurementAuditLogger(log_path, context={"study_id": "study"})
    logger.append_event(event_kind="timing", measurement_name="run")
    logger.flush()
    lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["study_id"] == "study"
    assert payload["measurement_name"] == "run"

    parsed = parse_rocm_smi_average_power_w(
        json.dumps(
            {
                "card0": {
                    "Average Graphics Package Power (W)": "12.5W",
                }
            }
        ),
        card_label="card0",
    )
    assert parsed == 12.5
