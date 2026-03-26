from iron.applications.transformer_layer.debug_log import (
    append_debug_event,
    classify_debug_exception,
)


def test_append_debug_event_writes_header_and_row(tmp_path):
    path = tmp_path / "debug.csv"

    append_debug_event(
        path,
        study_id="design_patterns_main",
        event_kind="study_started",
        component="automation",
        pattern="encoder_pipeline",
        seq_len=64,
        challenge="study_execution",
        symptom="starting",
        impact_on_experiment="study begins",
        mitigation="none",
        status="started",
        supporting_log_path="suite.csv",
        date="2026-03-25T00:00:00+00:00",
    )

    text = path.read_text(encoding="utf-8")
    assert "event_kind" in text
    assert "study_started" in text
    assert "encoder_pipeline" in text


def test_classify_debug_exception_detects_topology_issue():
    event = classify_debug_exception(
        RuntimeError("unsupported topology for requested placement")
    )

    assert event["component"] == "npu_compile"
    assert event["challenge"] == "unsupported_topology_or_placement"
