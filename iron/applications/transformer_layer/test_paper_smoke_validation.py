import json
from pathlib import Path

from iron.applications.transformer_layer.paper_smoke_validation import (
    main,
    run_paper_smoke_validation,
)


def test_run_paper_smoke_validation_writes_expected_outputs(tmp_path: Path):
    summary = run_paper_smoke_validation(tmp_path)

    assert summary["annotated_row_count"] == 6
    assert summary["bottleneck_row_count"] == 6
    assert "main" in summary["plot_sections"]
    assert "gpu_compare" in summary["plot_sections"]
    assert (tmp_path / "paper_smoke_manifest.json").exists()
    assert (tmp_path / "paper_smoke_suite.csv").exists()
    assert (tmp_path / "paper_smoke_suite_annotated.csv").exists()
    assert (tmp_path / "paper_smoke_bottlenecks.csv").exists()
    assert (tmp_path / "paper_smoke_gpu_compare.csv").exists()
    assert (tmp_path / "plots" / "index.html").exists()
    assert (tmp_path / "plots" / "latency_by_sequence_length.svg").exists()
    assert (tmp_path / "plots" / "best_npu_vs_amd_gpu_latency.svg").exists()


def test_paper_smoke_cli_writes_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_smoke_validation.py",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    summary = json.loads(
        (tmp_path / "paper_smoke_summary.json").read_text(encoding="utf-8")
    )
    assert summary["study_id"] == "paper_smoke"
    assert summary["seq_lens"] == [64, 128]
