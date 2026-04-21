#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
import math
import shutil
import statistics
import subprocess
import threading
import time

SUPPORTED_POWER_BACKENDS: tuple[str, ...] = ("auto", "none", "turbostat_pkgwatt")
POWER_OUTLIER_FILTER_MIN_SAMPLE_COUNT = 10
POWER_OUTLIER_FILTER_MIN_RETAINED_SAMPLE_COUNT = 6
POWER_OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
PERSISTED_POWER_RESULT_FIELDS: tuple[str, ...] = (
    "power_boundary",
    "power_estimation_method",
    "baseline_policy",
    "baseline_avg_power_w",
    "active_avg_power_w",
    "sensor_source",
    "temperature_source",
    "avg_power_w",
    "min_power_w",
    "max_power_w",
    "power_sample_count",
    "raw_avg_power_w",
    "raw_min_power_w",
    "raw_max_power_w",
    "raw_power_sample_count",
    "power_std_w",
    "raw_power_std_w",
    "power_outlier_sample_count",
    "power_outlier_filter_applied",
    "raw_package_avg_power_w",
    "raw_package_min_power_w",
    "raw_package_max_power_w",
)


def empty_power_stats() -> dict[str, float | str | None]:
    return {
        "power_backend": None,
        "power_boundary": None,
        "power_estimation_method": None,
        "baseline_policy": None,
        "baseline_avg_power_w": None,
        "active_avg_power_w": None,
        "sensor_source": None,
        "temperature_source": None,
        "avg_power_w": None,
        "raw_avg_power_w": None,
        "raw_min_power_w": None,
        "raw_max_power_w": None,
        "raw_power_sample_count": None,
        "power_std_w": None,
        "raw_power_std_w": None,
        "power_outlier_sample_count": None,
        "power_outlier_filter_applied": None,
        "raw_package_avg_power_w": None,
        "raw_package_min_power_w": None,
        "raw_package_max_power_w": None,
        "quiescent_package_power_w": None,
        "min_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "power_sample_count": None,
    }


def parse_turbostat_pkgwatt_samples(stdout: str) -> list[float]:
    samples: list[float] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line == "PkgWatt" or line.endswith(" sec"):
            continue
        try:
            samples.append(float(line))
        except ValueError:
            continue
    return samples


def resolve_requested_power_backend(requested_backend: str) -> str:
    if requested_backend not in SUPPORTED_POWER_BACKENDS:
        raise ValueError(f"Unsupported power backend: {requested_backend}")
    if requested_backend == "auto":
        return "turbostat_pkgwatt" if shutil.which("turbostat") else "none"
    return requested_backend


def resolve_power_sample_interval_sec(
    *,
    requested_interval_sec: float,
    estimated_timed_window_sec: float | None,
    min_sample_count: int = 6,
    min_interval_sec: float = 0.1,
) -> float:
    interval = float(requested_interval_sec)
    if estimated_timed_window_sec is None or estimated_timed_window_sec <= 0:
        return interval
    target_interval = float(estimated_timed_window_sec) / float(min_sample_count)
    return max(float(min_interval_sec), min(interval, target_interval))


def resolve_power_probe_runs(
    *,
    avg_iteration_sec: float | None,
    baseline_runs: int,
    min_measurement_duration_sec: float = 0.25,
) -> int:
    runs = max(1, int(baseline_runs))
    if avg_iteration_sec is None or avg_iteration_sec <= 0:
        return runs
    return max(runs, int(math.ceil(min_measurement_duration_sec / avg_iteration_sec)))


def power_probe_is_complete(
    *,
    completed_runs: int,
    min_runs: int,
    elapsed_sec: float,
    min_measurement_duration_sec: float,
    observed_sample_count: int,
    min_sample_count: int,
) -> bool:
    return (
        completed_runs >= int(min_runs)
        and elapsed_sec >= float(min_measurement_duration_sec)
        and observed_sample_count >= int(min_sample_count)
    )


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(fraction)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def detect_power_sample_outliers(samples_w: list[float]) -> list[bool]:
    if len(samples_w) < 5:
        return [False] * len(samples_w)

    median_sample = statistics.median(samples_w)
    absolute_deviations = [abs(sample - median_sample) for sample in samples_w]
    median_absolute_deviation = statistics.median(absolute_deviations)
    if median_absolute_deviation > 0:
        return [
            abs(0.6745 * (sample - median_sample) / median_absolute_deviation)
            > POWER_OUTLIER_MODIFIED_Z_THRESHOLD
            for sample in samples_w
        ]

    sorted_samples = sorted(samples_w)
    q1 = _percentile(sorted_samples, 0.25)
    q3 = _percentile(sorted_samples, 0.75)
    if q1 is None or q3 is None:
        return [False] * len(samples_w)
    interquartile_range = q3 - q1
    if interquartile_range <= 0:
        return [False] * len(samples_w)
    lower_bound = q1 - 1.5 * interquartile_range
    upper_bound = q3 + 1.5 * interquartile_range
    return [sample < lower_bound or sample > upper_bound for sample in samples_w]


def summarize_power_samples(
    samples_w: list[float],
    *,
    elapsed_sec: float | None = None,
    min_filter_sample_count: int = POWER_OUTLIER_FILTER_MIN_SAMPLE_COUNT,
    min_retained_sample_count: int = POWER_OUTLIER_FILTER_MIN_RETAINED_SAMPLE_COUNT,
) -> dict[str, float | str | None]:
    stats = empty_power_stats()
    if not samples_w:
        return stats

    raw_avg_power_w = statistics.fmean(samples_w)
    raw_std_power_w = statistics.stdev(samples_w) if len(samples_w) >= 2 else 0.0
    filtered_samples = list(samples_w)
    outlier_mask = [False] * len(samples_w)
    power_outlier_filter_applied = False

    if len(samples_w) >= int(min_filter_sample_count):
        candidate_mask = detect_power_sample_outliers(samples_w)
        candidate_filtered_samples = [
            sample
            for sample, is_outlier in zip(samples_w, candidate_mask)
            if not is_outlier
        ]
        if len(candidate_filtered_samples) >= int(min_retained_sample_count) and len(
            candidate_filtered_samples
        ) < len(samples_w):
            candidate_std_power_w = (
                statistics.stdev(candidate_filtered_samples)
                if len(candidate_filtered_samples) >= 2
                else 0.0
            )
            if candidate_std_power_w < raw_std_power_w:
                filtered_samples = candidate_filtered_samples
                outlier_mask = candidate_mask
                power_outlier_filter_applied = True

    avg_power_w = statistics.fmean(filtered_samples)
    power_std_w = (
        statistics.stdev(filtered_samples) if len(filtered_samples) >= 2 else 0.0
    )
    stats.update(
        {
            "avg_power_w": avg_power_w,
            "raw_avg_power_w": raw_avg_power_w,
            "raw_min_power_w": min(samples_w),
            "raw_max_power_w": max(samples_w),
            "raw_power_sample_count": len(samples_w),
            "power_std_w": power_std_w,
            "raw_power_std_w": raw_std_power_w,
            "power_outlier_sample_count": sum(
                1 for is_outlier in outlier_mask if is_outlier
            ),
            "power_outlier_filter_applied": power_outlier_filter_applied,
            "min_power_w": min(filtered_samples),
            "max_power_w": max(filtered_samples),
            "power_sample_count": len(filtered_samples),
        }
    )
    if elapsed_sec is not None:
        stats["energy_j"] = avg_power_w * float(elapsed_sec)
    return stats


def _run_turbostat_pkgwatt_samples(
    *,
    sample_interval_sec: float,
    num_iterations: int,
) -> list[float]:
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "turbostat",
            "--quiet",
            "--Summary",
            "--show",
            "PkgWatt",
            "--interval",
            str(sample_interval_sec),
            "--num_iterations",
            str(num_iterations),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    samples = parse_turbostat_pkgwatt_samples(result.stdout)
    if not samples:
        raise RuntimeError("turbostat did not return any PkgWatt samples")
    return samples


class TurbostatPackagePowerMonitor:
    def __init__(
        self,
        *,
        sample_interval_sec: float = 0.1,
        quiescent_baseline_duration_sec: float = 0.5,
        estimated_timed_window_sec: float | None = None,
    ):
        self.sample_interval_sec = float(sample_interval_sec)
        self.quiescent_baseline_duration_sec = float(quiescent_baseline_duration_sec)
        self.estimated_timed_window_sec = estimated_timed_window_sec
        self.quiescent_package_power_w: float | None = None
        self.raw_package_samples_w: list[float] = []
        self._pseudo_samples_w: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._use_streaming_process = estimated_timed_window_sec is None or (
            estimated_timed_window_sec >= max(0.08, self.sample_interval_sec * 3.0)
        )

    def __enter__(self):
        self.quiescent_package_power_w = self._measure_quiescent_baseline()
        if self._use_streaming_process:
            self._process = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "turbostat",
                    "--quiet",
                    "--Summary",
                    "--show",
                    "PkgWatt",
                    "--interval",
                    str(self.sample_interval_sec),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            target = self._sample_loop_streaming
        else:
            target = self._sample_loop_burst
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=max(1.0, self.sample_interval_sec * 4.0))
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4.0))
        return False

    def _measure_quiescent_baseline(self) -> float:
        iterations = max(
            1,
            int(
                math.ceil(
                    self.quiescent_baseline_duration_sec / self.sample_interval_sec
                )
            ),
        )
        samples = _run_turbostat_pkgwatt_samples(
            sample_interval_sec=self.sample_interval_sec,
            num_iterations=iterations,
        )
        return sum(samples) / len(samples)

    def _sample_loop_streaming(self) -> None:
        baseline = self.quiescent_package_power_w or 0.0
        if self._process is None or self._process.stdout is None:
            return
        try:
            for raw_line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                for sample in parse_turbostat_pkgwatt_samples(raw_line):
                    self.raw_package_samples_w.append(sample)
                    self._pseudo_samples_w.append(sample - baseline)
        except Exception:
            return

    def _sample_loop_burst(self) -> None:
        baseline = self.quiescent_package_power_w or 0.0
        while not self._stop_event.is_set():
            try:
                samples = _run_turbostat_pkgwatt_samples(
                    sample_interval_sec=self.sample_interval_sec,
                    num_iterations=1,
                )
                for sample in samples:
                    self.raw_package_samples_w.append(sample)
                    self._pseudo_samples_w.append(sample - baseline)
            except Exception:
                return

    def current_sample_count(self) -> int:
        return len(self._pseudo_samples_w)

    def stats(self, elapsed_sec: float) -> dict[str, float | str | None]:
        stats = empty_power_stats()
        stats["power_backend"] = "turbostat_pkgwatt"
        stats["power_boundary"] = "package"
        stats["power_estimation_method"] = "delta_package_power"
        stats["baseline_policy"] = "quiescent_package_power"
        stats["quiescent_package_power_w"] = self.quiescent_package_power_w
        stats["baseline_avg_power_w"] = self.quiescent_package_power_w
        stats["sensor_source"] = "turbostat:PkgWatt"
        stats["temperature_source"] = "unavailable"

        if not self.raw_package_samples_w:
            return stats

        avg_raw_w = sum(self.raw_package_samples_w) / len(self.raw_package_samples_w)
        stats["active_avg_power_w"] = avg_raw_w
        filtered_stats = summarize_power_samples(
            self._pseudo_samples_w,
            elapsed_sec=elapsed_sec,
        )
        stats.update(
            {
                "raw_package_avg_power_w": avg_raw_w,
                "raw_package_min_power_w": min(self.raw_package_samples_w),
                "raw_package_max_power_w": max(self.raw_package_samples_w),
            }
        )
        for key, value in filtered_stats.items():
            # Keep monitor-level metadata such as estimation method and sensor
            # source when the sample summary leaves those fields unset.
            if value is None and stats.get(key) is not None:
                continue
            stats[key] = value
        return stats


def create_power_monitor(
    *,
    power_backend: str,
    sample_interval_sec: float = 0.1,
    quiescent_baseline_duration_sec: float = 0.5,
    estimated_timed_window_sec: float | None = None,
):
    if power_backend == "none":
        return nullcontext(None)
    if power_backend != "turbostat_pkgwatt":
        raise ValueError(f"Unsupported power backend: {power_backend}")
    if shutil.which("turbostat") is None:
        raise RuntimeError(
            "turbostat is required for power_backend='turbostat_pkgwatt'"
        )
    return TurbostatPackagePowerMonitor(
        sample_interval_sec=sample_interval_sec,
        quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
        estimated_timed_window_sec=estimated_timed_window_sec,
    )
