#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from iron.applications.transformer_layer.gpu_inference import benchmark_gpu_layer
from iron.applications.transformer_layer.src.bench import write_results_csv
from iron.applications.transformer_layer.src.core.layer_spec import TransformerLayerSpec


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def load_compare_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("reference_npu_csv", "output_csv"):
        if key not in config:
            raise KeyError(f"Missing required compare config field: {key}")
        raw_path = Path(str(config[key]))
        if not raw_path.is_absolute():
            config[key] = str((config_path.parent / raw_path).resolve())
    sampling_schedule = config.get("sampling_schedule")
    if isinstance(sampling_schedule, dict):
        config["sampling_schedule"] = {
            int(seq_len): {
                "warmup_runs": int(payload["warmup_runs"]),
                "runs_per_sample": int(payload["runs_per_sample"]),
            }
            for seq_len, payload in sampling_schedule.items()
        }
    config.setdefault("study_id", "gpu_compare")
    config.setdefault("execution_mode", "amd_igpu_reference")
    config.setdefault("pattern_label", config["execution_mode"])
    config.setdefault("device", "cuda:0")
    config.setdefault("power_backend", "none")
    config.setdefault("power_sample_interval_sec", 0.2)
    config.setdefault("warmup_runs", 5)
    config.setdefault("runs_per_sample", 20)
    return config


def _load_npu_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("backend") == "npu"
        and row.get("run_status", "completed") == "completed"
    ]


def select_best_npu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_group: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("study_case_id") or "default")
        seq_len = _optional_int(row.get("seq_len"))
        latency = _optional_float(row.get("avg_latency_ms"))
        if seq_len is None or latency is None:
            continue
        group = (case_id, seq_len)
        current = best_by_group.get(group)
        if current is None or latency < _optional_float(current.get("avg_latency_ms")):
            best_by_group[group] = row
    return [best_by_group[key] for key in sorted(best_by_group)]


def _sampling_for_seq(config: dict[str, object], seq_len: int) -> tuple[int, int]:
    schedule = config.get("sampling_schedule")
    if isinstance(schedule, dict) and seq_len in schedule:
        entry = schedule[seq_len]
        return int(entry["warmup_runs"]), int(entry["runs_per_sample"])
    return int(config["warmup_runs"]), int(config["runs_per_sample"])


def _spec_from_row(row: dict[str, str]) -> TransformerLayerSpec:
    return TransformerLayerSpec(
        hidden_size=int(row["hidden_size"]),
        intermediate_size=int(row["intermediate_size"]),
        num_attention_heads=int(row["num_attention_heads"]),
        seq_len=int(row["seq_len"]),
        batch_size=int(row.get("batch_size", 1)),
        dtype=str(row.get("dtype", "bfloat16")),
        use_bias=str(row.get("use_bias", "False")).lower() == "true",
        weights_source=str(row.get("weights_source", "synthetic")),
        source_model_name=(
            row.get("source_model_name")
            if row.get("source_model_name") not in ("", "None")
            else None
        ),
        source_layer_index=_optional_int(row.get("source_layer_index")),
    )


def _unsupported_gpu_row(
    *,
    config: dict[str, object],
    npu_row: dict[str, str],
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    failure_message: str,
) -> dict[str, object]:
    return {
        "study_id": config["study_id"],
        "study_case_id": npu_row.get("study_case_id"),
        "study_case_label": npu_row.get("study_case_label"),
        "backend": "gpu",
        "execution_mode": config["execution_mode"],
        "pattern_label": config["pattern_label"],
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "use_bias": spec.use_bias,
        "weights_source": spec.weights_source,
        "source_model_name": spec.source_model_name,
        "source_layer_index": spec.source_layer_index,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "measured_inference_count": 0,
        "timed_total_sec": 0.0,
        "avg_latency_ms": None,
        "run_status": "unsupported",
        "failure_component": "gpu_runtime",
        "failure_category": "gpu_unavailable",
        "failure_message": failure_message,
        "power_backend": config["power_backend"],
        "reference_npu_execution_mode": npu_row.get("execution_mode"),
        "reference_npu_avg_latency_ms": _optional_float(npu_row.get("avg_latency_ms")),
    }


def benchmark_best_npu_vs_gpu(config: dict[str, object]) -> list[dict[str, object]]:
    selected_rows = select_best_npu_rows(_load_npu_rows(config["reference_npu_csv"]))
    output_rows = []
    for npu_row in selected_rows:
        spec = _spec_from_row(npu_row)
        warmup_runs, runs_per_sample = _sampling_for_seq(config, spec.seq_len)
        try:
            gpu_rows = benchmark_gpu_layer(
                spec=spec,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                output_csv=config["output_csv"],
                seed=int(config.get("seed", 0)),
                device_name=str(config["device"]),
                power_backend=str(config["power_backend"]),
                power_sample_interval_sec=float(config["power_sample_interval_sec"]),
                write_immediately=False,
            )
            row = dict(gpu_rows[0])
            row["study_id"] = config["study_id"]
            row["execution_mode"] = config["execution_mode"]
            row["pattern_label"] = config["pattern_label"]
            row["study_case_id"] = npu_row.get("study_case_id")
            row["study_case_label"] = npu_row.get("study_case_label")
            row["reference_npu_execution_mode"] = npu_row.get("execution_mode")
            row["reference_npu_avg_latency_ms"] = _optional_float(
                npu_row.get("avg_latency_ms")
            )
        except Exception as exc:
            row = _unsupported_gpu_row(
                config=config,
                npu_row=npu_row,
                spec=spec,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                failure_message=str(exc),
            )
        output_rows.append(row)
    write_results_csv(config["output_csv"], output_rows)
    return output_rows
