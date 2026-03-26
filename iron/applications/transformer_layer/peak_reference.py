# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BackendPeakReference:
    backend: str
    peak_ops_per_sec: float
    peak_bytes_per_sec: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_peak_reference(path: str | Path, peak: BackendPeakReference) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(peak.to_dict(), indent=2), encoding="utf-8")


def load_peak_reference(path: str | Path) -> BackendPeakReference:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BackendPeakReference(**payload)
