from iron.applications.transformer_layer.peak_reference import BackendPeakReference
from iron.applications.transformer_layer.roofline import (
    annotate_result_row_with_peak,
    estimate_layer_bytes,
    estimate_layer_flops,
    roofline_bound_ops_per_sec,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def test_roofline_estimates_are_positive():
    spec = TransformerLayerSpec()
    assert estimate_layer_flops(spec) > 0
    assert estimate_layer_bytes(spec) > 0


def test_roofline_bound_is_capped_by_peak():
    spec = TransformerLayerSpec()
    peak = BackendPeakReference(
        backend="npu",
        peak_ops_per_sec=10.0,
        peak_bytes_per_sec=1.0e12,
    )
    assert roofline_bound_ops_per_sec(spec, peak) <= peak.peak_ops_per_sec


def test_annotate_result_row_with_peak_fills_percent_fields():
    row = {
        "backend": "npu",
        "throughput_flops_per_sec": 20.0,
        "estimated_flops_per_inference": 10.0,
        "estimated_bytes_per_inference": 5.0,
        "operational_intensity_flops_per_byte": 2.0,
    }
    peak = BackendPeakReference(
        backend="npu",
        peak_ops_per_sec=40.0,
        peak_bytes_per_sec=8.0,
    )

    annotated = annotate_result_row_with_peak(row, peak)

    assert annotated["backend_peak_ops_per_sec"] == 40.0
    assert annotated["roofline_bound_ops_per_sec"] == 16.0
    assert annotated["backend_pct_of_peak"] == 0.5
    assert annotated["roofline_pct"] == 1.25
