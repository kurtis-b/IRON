# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

PEAK_REFERENCE_ARTIFACT_VERSION = "1"


@dataclass(frozen=True)
class BackendPeakReference:
    backend: str
    peak_ops_per_sec: float
    peak_bytes_per_sec: float
    source_note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_peak_reference(path: str | Path, peak: BackendPeakReference) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(peak.to_dict(), indent=2), encoding="utf-8")


def load_peak_reference(path: str | Path) -> BackendPeakReference:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "references" in payload:
        references = payload["references"]
        if len(references) != 1:
            raise ValueError(
                "Peak-reference artifact contains multiple backends; use load_peak_references()"
            )
        payload = references[0]
    return BackendPeakReference(**payload)


def save_peak_references(
    path: str | Path,
    peaks: list[BackendPeakReference] | dict[str, BackendPeakReference],
) -> None:
    if isinstance(peaks, dict):
        ordered_peaks = [peaks[key] for key in sorted(peaks)]
    else:
        ordered_peaks = sorted(peaks, key=lambda peak: peak.backend)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": PEAK_REFERENCE_ARTIFACT_VERSION,
        "references": [peak.to_dict() for peak in ordered_peaks],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_peak_references(path: str | Path) -> dict[str, BackendPeakReference]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "references" not in payload:
        peak = BackendPeakReference(**payload)
        return {peak.backend: peak}
    return {
        reference["backend"]: BackendPeakReference(**reference)
        for reference in payload["references"]
    }


def upsert_peak_reference(path: str | Path, peak: BackendPeakReference) -> None:
    output_path = Path(path)
    if output_path.exists():
        peaks = load_peak_references(output_path)
    else:
        peaks = {}
    peaks[peak.backend] = peak
    save_peak_references(output_path, peaks)
