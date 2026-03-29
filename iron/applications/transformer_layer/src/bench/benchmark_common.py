# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.layer_spec import TransformerLayerSpec
from ..core.result_schema import RESULT_FIELD_ORDER, normalize_result_row


def parse_seq_lens(seq_lens: str) -> list[int]:
    return [int(part.strip()) for part in seq_lens.split(",") if part.strip()]


def parse_execution_modes(raw_modes: str) -> list[str]:
    return [part.strip() for part in str(raw_modes).split(",") if part.strip()]


def summarize_latency_measurements(latencies_sec: list[float]) -> dict[str, float]:
    if not latencies_sec:
        raise ValueError("No latency measurements were provided")
    total_sec = sum(latencies_sec)
    return {
        "timed_total_sec": total_sec,
        "avg_latency_ms": (total_sec / len(latencies_sec)) * 1000.0,
        "measured_inference_count": len(latencies_sec),
    }


def write_results_csv(output_csv: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_result_row(row) for row in rows]
    extra_fields = sorted(
        {key for row in normalized for key in row.keys()}.difference(RESULT_FIELD_ORDER)
    )
    fieldnames = RESULT_FIELD_ORDER + extra_fields
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def write_dict_rows_csv(
    output_csv: str | Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError("No rows were provided")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_study_path(manifest_path: str | Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(manifest_path).resolve().parent / path).resolve())


def _normalize_layer_spec_dict(payload: dict[str, object]) -> dict[str, object]:
    return TransformerLayerSpec.from_dict(dict(payload)).to_dict()


def load_study_manifest(manifest_path: str | Path) -> dict[str, object]:
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    required = {"study_id", "execution_modes", "seq_lens"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise KeyError(f"Missing required study-manifest fields: {', '.join(missing)}")
    has_layer_spec = "layer_spec" in manifest
    has_study_cases = "study_cases" in manifest
    if has_layer_spec == has_study_cases:
        raise KeyError(
            "Study manifest must include exactly one of layer_spec or study_cases"
        )

    manifest["manifest_path"] = str(manifest_file)
    execution_modes = manifest["execution_modes"]
    if isinstance(execution_modes, str):
        manifest["execution_modes"] = parse_execution_modes(execution_modes)
    seq_lens = manifest["seq_lens"]
    if isinstance(seq_lens, str):
        manifest["seq_lens"] = parse_seq_lens(seq_lens)
    manifest.setdefault("warmup_runs", 5)
    manifest.setdefault("runs_per_sample", 20)
    manifest.setdefault("seed", 0)
    manifest.setdefault("continue_on_error", False)

    if "output_csv" in manifest:
        manifest["output_csv"] = resolve_study_path(
            manifest_file, manifest["output_csv"]
        )
    if "debug_log_csv" in manifest:
        manifest["debug_log_csv"] = resolve_study_path(
            manifest_file, manifest["debug_log_csv"]
        )
    if "peak_reference" in manifest:
        manifest["peak_reference"] = resolve_study_path(
            manifest_file, manifest["peak_reference"]
        )
    if "annotated_output_csv" in manifest:
        manifest["annotated_output_csv"] = resolve_study_path(
            manifest_file, manifest["annotated_output_csv"]
        )
    if "support_matrix_csv" in manifest:
        manifest["support_matrix_csv"] = resolve_study_path(
            manifest_file, manifest["support_matrix_csv"]
        )
    if "support_matrix_text" in manifest:
        manifest["support_matrix_text"] = resolve_study_path(
            manifest_file, manifest["support_matrix_text"]
        )

    parity = manifest.get("parity")
    if isinstance(parity, dict):
        parity = dict(parity)
        parity.setdefault("enabled", False)
        if "execution_modes" in parity and isinstance(parity["execution_modes"], str):
            parity["execution_modes"] = parse_execution_modes(parity["execution_modes"])
        if "seq_lens" in parity and isinstance(parity["seq_lens"], str):
            parity["seq_lens"] = parse_seq_lens(parity["seq_lens"])
        if "output_csv" in parity:
            parity["output_csv"] = resolve_study_path(
                manifest_file, parity["output_csv"]
            )
        manifest["parity"] = parity

    sampling_schedule = manifest.get("sampling_schedule")
    if isinstance(sampling_schedule, dict):
        normalized_schedule = {}
        for raw_seq_len, payload in sampling_schedule.items():
            seq_len = int(raw_seq_len)
            if not isinstance(payload, dict):
                raise TypeError(f"sampling_schedule[{raw_seq_len!r}] must be an object")
            normalized_schedule[seq_len] = {
                "warmup_runs": int(payload.get("warmup_runs", manifest["warmup_runs"])),
                "runs_per_sample": int(
                    payload.get("runs_per_sample", manifest["runs_per_sample"])
                ),
            }
        manifest["sampling_schedule"] = normalized_schedule

    if has_study_cases:
        normalized_cases = []
        for index, raw_case in enumerate(manifest["study_cases"]):
            if not isinstance(raw_case, dict):
                raise TypeError("Each study_cases entry must be an object")
            if "layer_spec" not in raw_case:
                raise KeyError("Each study_cases entry must include layer_spec")
            case = dict(raw_case)
            if not isinstance(case["layer_spec"], dict):
                raise TypeError("Each study_cases layer_spec must be an object")
            case["layer_spec"] = _normalize_layer_spec_dict(case["layer_spec"])
            case.setdefault("case_id", f"case_{index}")
            case.setdefault("case_label", case["case_id"])
            if "execution_modes" in case and isinstance(case["execution_modes"], str):
                case["execution_modes"] = parse_execution_modes(case["execution_modes"])
            if "seq_lens" in case and isinstance(case["seq_lens"], str):
                case["seq_lens"] = parse_seq_lens(case["seq_lens"])
            normalized_cases.append(case)
        manifest["study_cases"] = normalized_cases
    else:
        if not isinstance(manifest["layer_spec"], dict):
            raise TypeError("layer_spec must be an object")
        manifest["layer_spec"] = _normalize_layer_spec_dict(manifest["layer_spec"])

    return manifest
