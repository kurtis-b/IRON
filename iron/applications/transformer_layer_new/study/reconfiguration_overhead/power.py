#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
import math
import shutil
import subprocess
import threading
import time

SUPPORTED_POWER_BACKENDS: tuple[str, ...] = ("auto", "none", "turbostat_pkgwatt")


def empty_power_stats() -> dict[str, float | str | None]:
    return {
        "power_backend": None,
        "avg_power_w": None,
        "raw_package_avg_power_w": None,
        "raw_package_max_power_w": None,
        "quiescent_package_power_w": None,
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
                    self._pseudo_samples_w.append(max(sample - baseline, 0.0))
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
                    self._pseudo_samples_w.append(max(sample - baseline, 0.0))
            except Exception:
                return

    def stats(self, elapsed_sec: float) -> dict[str, float | str | None]:
        stats = empty_power_stats()
        stats["power_backend"] = "turbostat_pkgwatt"
        stats["quiescent_package_power_w"] = self.quiescent_package_power_w
        if not self.raw_package_samples_w:
            return stats

        avg_raw_w = sum(self.raw_package_samples_w) / len(self.raw_package_samples_w)
        avg_pseudo_w = sum(self._pseudo_samples_w) / len(self._pseudo_samples_w)
        stats.update(
            {
                "raw_package_avg_power_w": avg_raw_w,
                "raw_package_max_power_w": max(self.raw_package_samples_w),
                "avg_power_w": avg_pseudo_w,
                "max_power_w": max(self._pseudo_samples_w),
                "energy_j": avg_pseudo_w * elapsed_sec,
                "power_sample_count": len(self._pseudo_samples_w),
            }
        )
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
