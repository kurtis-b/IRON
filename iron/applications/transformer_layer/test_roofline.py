from iron.applications.transformer_layer.peak_reference import BackendPeakReference
from iron.applications.transformer_layer.roofline import (
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
