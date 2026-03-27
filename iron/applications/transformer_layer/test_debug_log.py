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


def test_classify_debug_exception_detects_unsupported_pattern_surface():
    event = classify_debug_exception(
        ValueError(
            "encoder_pipeline thesis pattern currently supports only the 768/3072/12 and 1024/4096/16 families"
        )
    )

    assert event["component"] == "pattern_surface"
    assert event["challenge"] == "unsupported_pattern_surface"


def test_classify_debug_exception_detects_dma_descriptor_limit():
    event = classify_debug_exception(
        RuntimeError("'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.")
    )

    assert event["component"] == "npu_compile"
    assert event["challenge"] == "unsupported_dma_descriptor_limits"
