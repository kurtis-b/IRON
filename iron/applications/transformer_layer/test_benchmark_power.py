import pytest

from iron.applications.transformer_layer.benchmark_power import (
    TurbostatPackagePowerMonitor,
    create_power_monitor,
    empty_power_stats,
    parse_turbostat_pkgwatt_samples,
    resolve_power_sample_interval_sec,
    resolve_power_probe_runs,
)


def test_empty_power_stats_exposes_extended_power_fields():
    stats = empty_power_stats()

    assert stats["power_backend"] is None
    assert stats["raw_package_avg_power_w"] is None
    assert stats["quiescent_package_power_w"] is None


def test_parse_turbostat_pkgwatt_samples_reads_numeric_rows():
    stdout = "PkgWatt\n12.61\n13.81\n17.07\n"

    assert parse_turbostat_pkgwatt_samples(stdout) == [12.61, 13.81, 17.07]


def test_turbostat_monitor_stats_subtracts_quiescent_baseline():
    monitor = TurbostatPackagePowerMonitor()
    monitor.quiescent_package_power_w = 10.0
    monitor.raw_package_samples_w = [12.0, 18.0]
    monitor._pseudo_samples_w = [2.0, 8.0]
    monitor.baseline_sample_events = [{"sample_index": 1, "raw_package_power_w": 10.0}]
    monitor.probe_sample_events = [{"sample_index": 1, "raw_package_power_w": 12.0}]

    stats = monitor.stats(elapsed_sec=2.0)

    assert stats["power_backend"] == "turbostat_pkgwatt"
    assert stats["raw_package_avg_power_w"] == pytest.approx(15.0)
    assert stats["avg_power_w"] == pytest.approx(5.0)
    assert stats["max_power_w"] == pytest.approx(8.0)
    assert stats["energy_j"] == pytest.approx(10.0)
    assert stats["quiescent_package_power_w"] == pytest.approx(10.0)
    details = monitor.measurement_details()
    assert details["baseline"]["sample_count"] == 1
    assert details["probe"]["sample_count"] == 1


def test_create_power_monitor_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_power_monitor(power_backend="unknown")


def test_resolve_power_sample_interval_sec_uses_shorter_interval_for_short_runs():
    interval = resolve_power_sample_interval_sec(
        requested_interval_sec=0.2,
        estimated_timed_window_sec=0.3,
        min_sample_count=6,
        min_interval_sec=0.02,
    )

    assert interval == pytest.approx(0.05)


def test_resolve_power_sample_interval_sec_honors_floor():
    interval = resolve_power_sample_interval_sec(
        requested_interval_sec=0.2,
        estimated_timed_window_sec=0.01,
        min_sample_count=6,
        min_interval_sec=0.02,
    )

    assert interval == pytest.approx(0.02)


def test_resolve_power_probe_runs_extends_short_measurements():
    runs = resolve_power_probe_runs(
        avg_iteration_sec=0.01,
        baseline_runs=20,
        min_measurement_duration_sec=0.25,
    )

    assert runs == 25
