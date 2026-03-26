from pathlib import Path

from iron.applications.transformer_layer.peak_reference import (
    BackendPeakReference,
    load_peak_reference,
    load_peak_references,
    save_peak_reference,
    upsert_peak_reference,
)


def test_peak_reference_round_trip(tmp_path: Path):
    path = tmp_path / "peak.json"
    peak = BackendPeakReference(
        backend="npu",
        peak_ops_per_sec=123.0,
        peak_bytes_per_sec=456.0,
    )
    save_peak_reference(path, peak)
    assert load_peak_reference(path) == peak


def test_upsert_peak_reference_builds_multi_backend_artifact(tmp_path: Path):
    path = tmp_path / "peaks.json"
    upsert_peak_reference(
        path,
        BackendPeakReference(
            backend="npu",
            peak_ops_per_sec=123.0,
            peak_bytes_per_sec=456.0,
        ),
    )
    upsert_peak_reference(
        path,
        BackendPeakReference(
            backend="gpu",
            peak_ops_per_sec=789.0,
            peak_bytes_per_sec=321.0,
        ),
    )

    peaks = load_peak_references(path)

    assert set(peaks) == {"gpu", "npu"}
    assert peaks["npu"].peak_ops_per_sec == 123.0
    assert peaks["gpu"].peak_bytes_per_sec == 321.0
