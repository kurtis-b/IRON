# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import math
import shutil
import subprocess
import threading
import time

from iron.applications.transformer_layer.measurement_log import utc_now_iso_precise


def empty_power_stats() -> dict[str, float | None]:
    return {
        "power_backend": None,
        "raw_package_avg_power_w": None,
        "raw_package_max_power_w": None,
        "quiescent_package_power_w": None,
        "avg_power_w": None,
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


def _utc_iso_from_epoch(epoch_sec: float) -> str:
    return datetime.fromtimestamp(epoch_sec, timezone.utc).isoformat(
        timespec="microseconds"
    )


def resolve_power_sample_interval_sec(
    *,
    requested_interval_sec: float,
    estimated_timed_window_sec: float | None,
    min_sample_count: int = 6,
    min_interval_sec: float = 0.02,
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
    *, sample_interval_sec: float, num_iterations: int
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


def _run_turbostat_pkgwatt_capture(
    *, sample_interval_sec: float, num_iterations: int
) -> dict[str, object]:
    started_epoch_sec = time.time()
    started_perf_counter_sec = time.perf_counter()
    started_at_utc = utc_now_iso_precise()
    samples = _run_turbostat_pkgwatt_samples(
        sample_interval_sec=sample_interval_sec,
        num_iterations=num_iterations,
    )
    ended_perf_counter_sec = time.perf_counter()
    ended_epoch_sec = time.time()
    ended_at_utc = utc_now_iso_precise()
    elapsed_sec = ended_perf_counter_sec - started_perf_counter_sec
    sample_events = []
    sample_window_sec = max(elapsed_sec, sample_interval_sec * len(samples))
    sample_window_start_epoch_sec = ended_epoch_sec - sample_window_sec
    sample_window_start_perf_counter_sec = ended_perf_counter_sec - sample_window_sec
    for sample_index, sample in enumerate(samples, start=1):
        sample_offset_sec = min(sample_window_sec, sample_index * sample_interval_sec)
        sample_events.append(
            {
                "sample_index": sample_index,
                "captured_at_utc": _utc_iso_from_epoch(
                    sample_window_start_epoch_sec + sample_offset_sec
                ),
                "captured_perf_counter_sec": (
                    sample_window_start_perf_counter_sec + sample_offset_sec
                ),
                "raw_package_power_w": sample,
            }
        )
    return {
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_perf_counter_sec": started_perf_counter_sec,
        "ended_perf_counter_sec": ended_perf_counter_sec,
        "elapsed_sec": elapsed_sec,
        "samples": samples,
        "sample_events": sample_events,
    }


class TurbostatPackagePowerMonitor:
    def __init__(
        self,
        *,
        sample_interval_sec: float = 0.05,
        quiescent_baseline_duration_sec: float = 0.5,
        estimated_timed_window_sec: float | None = None,
    ):
        self.sample_interval_sec = float(sample_interval_sec)
        self.quiescent_baseline_duration_sec = float(quiescent_baseline_duration_sec)
        self.estimated_timed_window_sec = estimated_timed_window_sec
        self.raw_package_samples_w: list[float] = []
        self._pseudo_samples_w: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._first_sample_event = threading.Event()
        self.quiescent_package_power_w: float | None = None
        self.baseline_started_at_utc: str | None = None
        self.baseline_ended_at_utc: str | None = None
        self.baseline_started_perf_counter_sec: float | None = None
        self.baseline_ended_perf_counter_sec: float | None = None
        self.baseline_elapsed_sec: float | None = None
        self.baseline_sample_events: list[dict[str, object]] = []
        self.probe_sample_events: list[dict[str, object]] = []
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
        if self._use_streaming_process:
            self._first_sample_event.wait(
                timeout=max(0.1, self.sample_interval_sec * 4.0)
            )
            self.raw_package_samples_w.clear()
            self._pseudo_samples_w.clear()
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
        capture = _run_turbostat_pkgwatt_capture(
            sample_interval_sec=self.sample_interval_sec,
            num_iterations=iterations,
        )
        self.baseline_started_at_utc = str(capture["started_at_utc"])
        self.baseline_ended_at_utc = str(capture["ended_at_utc"])
        self.baseline_started_perf_counter_sec = float(
            capture["started_perf_counter_sec"]
        )
        self.baseline_ended_perf_counter_sec = float(capture["ended_perf_counter_sec"])
        self.baseline_elapsed_sec = float(capture["elapsed_sec"])
        self.baseline_sample_events = list(capture["sample_events"])
        samples = list(capture["samples"])
        return sum(samples) / len(samples)

    def _sample_loop_streaming(self):
        baseline = self.quiescent_package_power_w or 0.0
        if self._process is None or self._process.stdout is None:
            return
        try:
            for raw_line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                samples = parse_turbostat_pkgwatt_samples(raw_line)
                for sample in samples:
                    captured_perf_counter_sec = time.perf_counter()
                    captured_at_utc = utc_now_iso_precise()
                    self.raw_package_samples_w.append(sample)
                    pseudo_sample_w = max(sample - baseline, 0.0)
                    self._pseudo_samples_w.append(pseudo_sample_w)
                    self.probe_sample_events.append(
                        {
                            "sample_index": len(self.probe_sample_events) + 1,
                            "captured_at_utc": captured_at_utc,
                            "captured_perf_counter_sec": captured_perf_counter_sec,
                            "raw_package_power_w": sample,
                            "pseudo_power_w": pseudo_sample_w,
                        }
                    )
                    self._first_sample_event.set()
        except Exception:
            return

    def _sample_loop_burst(self):
        baseline = self.quiescent_package_power_w or 0.0
        while not self._stop_event.is_set():
            try:
                capture = _run_turbostat_pkgwatt_capture(
                    sample_interval_sec=self.sample_interval_sec,
                    num_iterations=1,
                )
                for sample_event in capture["sample_events"]:
                    sample = float(sample_event["raw_package_power_w"])
                    pseudo_sample_w = max(sample - baseline, 0.0)
                    self.raw_package_samples_w.append(sample)
                    self._pseudo_samples_w.append(pseudo_sample_w)
                    self.probe_sample_events.append(
                        {
                            **sample_event,
                            "pseudo_power_w": pseudo_sample_w,
                        }
                    )
            except Exception:
                pass

    def stats(self, elapsed_sec: float) -> dict[str, float | None]:
        if not self.raw_package_samples_w:
            stats = empty_power_stats()
            stats["power_backend"] = "turbostat_pkgwatt"
            stats["quiescent_package_power_w"] = self.quiescent_package_power_w
            return stats

        avg_raw_w = sum(self.raw_package_samples_w) / len(self.raw_package_samples_w)
        avg_pseudo_w = sum(self._pseudo_samples_w) / len(self._pseudo_samples_w)
        return {
            "power_backend": "turbostat_pkgwatt",
            "raw_package_avg_power_w": avg_raw_w,
            "raw_package_max_power_w": max(self.raw_package_samples_w),
            "quiescent_package_power_w": self.quiescent_package_power_w,
            "avg_power_w": avg_pseudo_w,
            "max_power_w": max(self._pseudo_samples_w),
            "energy_j": avg_pseudo_w * elapsed_sec,
            "power_sample_count": len(self._pseudo_samples_w),
        }

    def measurement_details(self) -> dict[str, object]:
        return {
            "baseline": {
                "start_time_utc": self.baseline_started_at_utc,
                "end_time_utc": self.baseline_ended_at_utc,
                "elapsed_sec": self.baseline_elapsed_sec,
                "start_perf_counter_sec": self.baseline_started_perf_counter_sec,
                "end_perf_counter_sec": self.baseline_ended_perf_counter_sec,
                "sample_interval_sec": self.sample_interval_sec,
                "average_power_w": self.quiescent_package_power_w,
                "sample_count": len(self.baseline_sample_events),
                "samples": list(self.baseline_sample_events),
            },
            "probe": {
                "sample_interval_sec": self.sample_interval_sec,
                "sample_count": len(self.probe_sample_events),
                "samples": list(self.probe_sample_events),
            },
        }


def create_power_monitor(
    *,
    power_backend: str = "none",
    sample_interval_sec: float = 0.05,
    quiescent_baseline_duration_sec: float = 0.5,
    estimated_timed_window_sec: float | None = None,
):
    if power_backend == "none":
        return nullcontext(None)
    if power_backend != "turbostat_pkgwatt":
        raise ValueError(f"Unsupported NPU power backend: {power_backend}")
    if shutil.which("turbostat") is None:
        raise RuntimeError(
            "turbostat is required for power_backend='turbostat_pkgwatt'"
        )
    return TurbostatPackagePowerMonitor(
        sample_interval_sec=sample_interval_sec,
        quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
        estimated_timed_window_sec=estimated_timed_window_sec,
    )
