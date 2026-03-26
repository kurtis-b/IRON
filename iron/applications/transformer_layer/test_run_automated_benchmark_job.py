import json

from iron.applications.transformer_layer.run_automated_benchmark_job import (
    build_command,
)


def test_build_command_resolves_manifest_and_optional_flags(tmp_path):
    manifest = tmp_path / "study.json"
    manifest.write_text("{}", encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "study_manifest": "study.json",
                "output_csv": "suite.csv",
                "run_parity_check": True,
                "parity_output_csv": "parity.csv",
            }
        ),
        encoding="utf-8",
    )

    command = build_command(str(job), json.loads(job.read_text(encoding="utf-8")))

    assert "--study-manifest" in command
    assert str(manifest.resolve()) in command
    assert "--output-csv" in command
    assert str((tmp_path / "suite.csv").resolve()) in command
    assert "--run-parity-check" in command
    assert str((tmp_path / "parity.csv").resolve()) in command
