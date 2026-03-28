#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
import json
import shutil
import subprocess
import threading
import time

from .benchmark_power import empty_power_stats
from .measurement_log import utc_now_iso_precise


def _parse_power_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Watts", "").replace("W", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_rocm_smi_average_power_w(
    stdout: str, *, card_label: str | None
) -> float | None:
    payload = json.loads(stdout)
    candidates = [card_label] if card_label is not None else list(payload.keys())
    for candidate in candidates:
        card_payload = payload.get(candidate)
        if not isinstance(card_payload, dict):
            continue
        for key, value in card_payload.items():
            lowered = key.lower()
            if "power" in lowered and ("socket" in lowered or "package" in lowered):
                parsed = _parse_power_value(value)
                if parsed is not None:
                    return parsed
    return None


class RocmSMIPowerMonitor:
    def __init__(
        self,
        *,
        device_index: int = 0,
        sample_interval_sec: float = 0.2,
    ):
        self.device_index = int(device_index)
        self.sample_interval_sec = float(sample_interval_sec)
        self.samples_w: list[float] = []
        self.sample_events: list[dict[str, object]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._card_label = f"card{self.device_index}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4.0))
        return False

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showpower", "--json"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                sample = parse_rocm_smi_average_power_w(
                    result.stdout,
                    card_label=self._card_label,
                )
                if sample is not None:
                    self.samples_w.append(sample)
                    self.sample_events.append(
                        {
                            "sample_index": len(self.sample_events) + 1,
                            "captured_at_utc": utc_now_iso_precise(),
                            "captured_perf_counter_sec": time.perf_counter(),
                            "power_w": sample,
                        }
                    )
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_sec)

    def stats(self, elapsed_sec: float) -> dict[str, float | None]:
        if not self.samples_w:
            return empty_power_stats()
        avg_power_w = sum(self.samples_w) / len(self.samples_w)
        return {
            "avg_power_w": avg_power_w,
            "max_power_w": max(self.samples_w),
            "energy_j": avg_power_w * elapsed_sec,
            "power_sample_count": len(self.samples_w),
        }

    def measurement_details(self) -> dict[str, object]:
        return {
            "probe": {
                "sample_interval_sec": self.sample_interval_sec,
                "sample_count": len(self.sample_events),
                "samples": list(self.sample_events),
            }
        }


def create_rocm_power_monitor(
    *,
    power_backend: str,
    device_index: int = 0,
    sample_interval_sec: float = 0.2,
):
    if power_backend == "none":
        return nullcontext(None)
    if power_backend != "rocm-smi":
        raise ValueError(f"Unsupported AMD GPU power backend: {power_backend}")
    if shutil.which("rocm-smi") is None:
        raise RuntimeError("rocm-smi is required for power_backend='rocm-smi'")
    return RocmSMIPowerMonitor(
        device_index=device_index,
        sample_interval_sec=sample_interval_sec,
    )
