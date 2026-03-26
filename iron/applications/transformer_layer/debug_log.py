#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

DEBUG_LOG_FIELD_ORDER = [
    "date",
    "study_id",
    "event_kind",
    "component",
    "pattern",
    "seq_len",
    "challenge",
    "symptom",
    "impact_on_experiment",
    "mitigation",
    "status",
    "supporting_log_path",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_debug_event(
    output_csv: str | Path | None,
    *,
    study_id: str | None = None,
    event_kind: str,
    component: str,
    pattern: str | None = None,
    seq_len: int | None = None,
    challenge: str | None = None,
    symptom: str | None = None,
    impact_on_experiment: str | None = None,
    mitigation: str | None = None,
    status: str,
    supporting_log_path: str | None = None,
    date: str | None = None,
) -> None:
    if output_csv is None:
        return
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": utc_now_iso() if date is None else date,
        "study_id": study_id,
        "event_kind": event_kind,
        "component": component,
        "pattern": pattern,
        "seq_len": seq_len,
        "challenge": challenge,
        "symptom": symptom,
        "impact_on_experiment": impact_on_experiment,
        "mitigation": mitigation,
        "status": status,
        "supporting_log_path": supporting_log_path,
    }
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_LOG_FIELD_ORDER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def classify_debug_exception(exc: Exception) -> dict[str, str]:
    symptom = f"{type(exc).__name__}: {exc}"
    lowered = symptom.lower()

    if "unsupported" in lowered and ("topology" in lowered or "placement" in lowered):
        return {
            "component": "npu_compile",
            "challenge": "unsupported_topology_or_placement",
            "symptom": symptom,
            "impact_on_experiment": "The requested benchmark case could not compile or run.",
            "mitigation": "Restrict the manifest to supported placements or adjust the layer specification.",
            "status": "open",
        }
    if "memlock" in lowered or "failed to allocate host memory buffer" in lowered:
        return {
            "component": "runtime_startup",
            "challenge": "memlock_or_host_buffer_allocation",
            "symptom": symptom,
            "impact_on_experiment": "The run stopped before the case could complete.",
            "mitigation": "Increase the memlock budget or reduce host-buffer pressure for the failing case.",
            "status": "open",
        }
    if "child" in lowered and ("crash" in lowered or "failed" in lowered):
        return {
            "component": "process_orchestration",
            "challenge": "child_process_failure",
            "symptom": symptom,
            "impact_on_experiment": "The harness lost a subprocess needed for the study.",
            "mitigation": "Inspect child-process logs and rerun the affected case in isolation.",
            "status": "open",
        }
    if "compile" in lowered or "xclbin" in lowered or "insts" in lowered:
        return {
            "component": "npu_compile",
            "challenge": "compile_failure",
            "symptom": symptom,
            "impact_on_experiment": "The requested benchmark case did not produce runnable NPU artifacts.",
            "mitigation": "Inspect generated MLIR, compilation artifacts, and the failing topology or pattern configuration.",
            "status": "open",
        }
    return {
        "component": "automation",
        "challenge": "benchmark_failure",
        "symptom": symptom,
        "impact_on_experiment": "The study halted before all requested outputs were produced.",
        "mitigation": "Inspect the stack trace and rerun the failing stage in isolation.",
        "status": "open",
    }
