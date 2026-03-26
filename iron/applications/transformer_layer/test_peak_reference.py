from pathlib import Path

from iron.applications.transformer_layer.peak_reference import (
    BackendPeakReference,
    load_peak_reference,
    save_peak_reference,
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
