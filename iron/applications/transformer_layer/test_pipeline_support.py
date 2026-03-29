# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from iron.applications.transformer_layer.run_automated_benchmark_job import (
    build_command,
    load_job_config,
)
from iron.applications.transformer_layer.run_study_pipeline import (
    load_pipeline_config,
    resolve_path,
)
from iron.applications.transformer_layer.src.pipeline import (
    build_command as structured_build_command,
)
from iron.applications.transformer_layer.src.pipeline import (
    load_job_config as structured_load_job_config,
)
from iron.applications.transformer_layer.src.pipeline import (
    load_pipeline_config as structured_load_pipeline_config,
)
from iron.applications.transformer_layer.src.pipeline import (
    resolve_path as structured_resolve_path,
)


def test_pipeline_restructure_preserves_legacy_imports_and_path_resolution(
    tmp_path: Path,
):
    assert load_pipeline_config is structured_load_pipeline_config
    assert resolve_path is structured_resolve_path
    assert load_job_config is structured_load_job_config
    assert build_command is structured_build_command

    pipeline_config_path = tmp_path / "pipeline.json"
    pipeline_config_path.write_text(
        json.dumps(
            {
                "pipeline_id": "designpats",
                "steps": [
                    {
                        "kind": "npu_study",
                        "manifest": "study/design_patterns_end_to_end.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pipeline = load_pipeline_config(pipeline_config_path)
    assert pipeline["pipeline_id"] == "designpats"
    assert pipeline["steps"][0]["step_id"] == "step_0"
    assert pipeline["steps"][0]["benchmarking_step"] is True
    assert pipeline["steps"][0]["manifest"] == str(
        (tmp_path / "study/design_patterns_end_to_end.json").resolve()
    )

    job_config_path = tmp_path / "job.json"
    job_config_path.write_text(
        json.dumps(
            {
                "study_manifest": "study/design_patterns_end_to_end.json",
                "output_csv": "results/out.csv",
                "block1_topology_id": "m64_k64_n16_ps1_ph1_pd1",
                "block2_topology_id": "q32_kv64_e96_ps1_ph1_acc1",
                "block3_topology_id": "m32_k96_n64_ps4_pi3_d8_g1",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    job = load_job_config(str(job_config_path))
    command = build_command(str(job_config_path), job)
    assert "--study-manifest" in command
    assert (
        str((tmp_path / "study/design_patterns_end_to_end.json").resolve()) in command
    )
    assert str((tmp_path / "results/out.csv").resolve()) in command
    assert "--block1-topology-id" in command
    assert "--block2-topology-id" in command
    assert "--block3-topology-id" in command
    assert command[-2:] == ["--seed", "7"]
