# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


def utc_now_iso_precise() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def default_measurement_log_path(output_csv: str | Path) -> str:
    output_path = Path(output_csv)
    base_path = output_path.with_suffix("") if output_path.suffix else output_path
    return str((base_path.parent / f"{base_path.name}_measurements.jsonl").resolve())


def capture_timed_call(fn):
    started_perf_counter_sec = time.perf_counter()
    started_at_utc = utc_now_iso_precise()
    result = fn()
    ended_perf_counter_sec = time.perf_counter()
    ended_at_utc = utc_now_iso_precise()
    return result, {
        "start_time_utc": started_at_utc,
        "end_time_utc": ended_at_utc,
        "elapsed_sec": ended_perf_counter_sec - started_perf_counter_sec,
        "start_perf_counter_sec": started_perf_counter_sec,
        "end_perf_counter_sec": ended_perf_counter_sec,
    }


@dataclass
class MeasurementAuditLogger:
    output_path: str | Path | None
    session_id: str = field(default_factory=lambda: uuid4().hex)
    context: dict[str, Any] = field(default_factory=dict)
    _buffer: list[dict[str, Any]] = field(default_factory=list)
    _event_index: int = 0

    def __post_init__(self) -> None:
        if self.output_path is None:
            return
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(
        self,
        *,
        event_kind: str,
        phase: str | None = None,
        measurement_name: str | None = None,
        run_index: int | None = None,
        sample_index: int | None = None,
        start_time_utc: str | None = None,
        end_time_utc: str | None = None,
        elapsed_sec: float | None = None,
        start_perf_counter_sec: float | None = None,
        end_perf_counter_sec: float | None = None,
        start_point: str | None = None,
        end_point: str | None = None,
        measurement_value: float | int | str | None = None,
        units: str | None = None,
        details: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        if self.output_path is None:
            return
        payload = {
            "event_index": self._event_index,
            "event_timestamp_utc": utc_now_iso_precise(),
            "measurement_session_id": self.session_id,
            "pid": os.getpid(),
            **self.context,
            "event_kind": event_kind,
            "phase": phase,
            "measurement_name": measurement_name,
            "run_index": run_index,
            "sample_index": sample_index,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "elapsed_sec": elapsed_sec,
            "start_perf_counter_sec": start_perf_counter_sec,
            "end_perf_counter_sec": end_perf_counter_sec,
            "start_point": start_point,
            "end_point": end_point,
            "measurement_value": measurement_value,
            "units": units,
            "details": details or {},
            **extra,
        }
        self._buffer.append(payload)
        self._event_index += 1

    def flush(self) -> None:
        if self.output_path is None or not self._buffer:
            return
        assert isinstance(self.output_path, Path)
        with self.output_path.open("a", encoding="utf-8") as handle:
            for payload in self._buffer:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        self._buffer.clear()
