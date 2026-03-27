#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import torch

from iron.applications.transformer_layer.benchmark_common import (
    summarize_latency_measurements,
)
from iron.applications.transformer_layer.benchmark_power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
)
from iron.applications.transformer_layer.measurement_log import (
    MeasurementAuditLogger,
    capture_timed_call,
)
from iron.applications.transformer_layer.npu_inference import (
    INFERENCE_END_POINT,
    INFERENCE_START_POINT,
    _average_stage_timings_ms,
    _log_power_measurements,
    _measurement_context,
    _pattern_metadata,
)
from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
    flops_per_joule,
    gflops_per_joule,
    operational_intensity,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.pattern_operator_runlist import (
    OperatorRunlistPattern,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)


def _ensure_finite_output(
    output: torch.Tensor, *, mode: str, spec: TransformerLayerSpec
):
    if torch.isfinite(output).all():
        return
    raise RuntimeError(
        "operator_runlist produced non-finite output on the current pattern surface "
        f"(mode={mode}, input_boundary={spec.input_boundary}, seq_len={spec.seq_len}, "
        f"hidden_size={spec.hidden_size}, num_attention_heads={spec.num_attention_heads})"
    )


def benchmark_operator_runlist_request(request: dict[str, object]) -> dict[str, object]:
    spec = TransformerLayerSpec.from_dict(request["spec"])
    warmup_runs = int(request["warmup_runs"])
    runs_per_sample = int(request["runs_per_sample"])
    seed = int(request["seed"])
    power_backend = str(request.get("power_backend", "none"))
    power_sample_interval_sec = float(request.get("power_sample_interval_sec", 0.05))
    quiescent_baseline_duration_sec = float(
        request.get("quiescent_baseline_duration_sec", 0.5)
    )
    study_id = str(request.get("study_id", "synthetic_transformer_layer"))
    study_case_id = request.get("study_case_id")
    study_case_label = request.get("study_case_label")
    measurement_log_path_raw = request.get("measurement_log_path")
    measurement_log_path = (
        None
        if measurement_log_path_raw in (None, "", "None")
        else str(measurement_log_path_raw)
    )
    measurement_session_id = request.get("measurement_session_id")

    audit_logger = MeasurementAuditLogger(
        measurement_log_path,
        session_id=(
            str(measurement_session_id)
            if measurement_session_id not in (None, "", "None")
            else uuid4().hex
        ),
        context=_measurement_context(
            study_id=study_id,
            study_case_id=(
                None if study_case_id in (None, "", "None") else str(study_case_id)
            ),
            study_case_label=(
                None
                if study_case_label in (None, "", "None")
                else str(study_case_label)
            ),
            backend="npu",
            execution_mode="operator_runlist",
            pattern_label="operator_runlist",
            spec=spec,
            measurement_log_path=measurement_log_path,
        ),
    )
    audit_logger.append_event(
        event_kind="benchmark_session_started",
        phase="session",
        details={
            "warmup_runs": warmup_runs,
            "runs_per_sample": runs_per_sample,
            "seed": seed,
            "power_backend": power_backend,
            "power_sample_interval_sec": power_sample_interval_sec,
            "quiescent_baseline_duration_sec": quiescent_baseline_duration_sec,
            "process_model": "child_process",
        },
    )

    pattern = OperatorRunlistPattern(spec)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)
    pattern.assign_weights(weights)
    pattern._prepare_runtime()

    def _run_once() -> dict[str, float]:
        output, stage_timings = pattern.forward_with_stage_timings(layer_inputs)
        _ensure_finite_output(output, mode="benchmark", spec=spec)
        return stage_timings

    try:
        for warmup_index in range(1, warmup_runs + 1):
            _, timing = capture_timed_call(_run_once)
            audit_logger.append_event(
                event_kind="timing_measurement",
                phase="warmup",
                measurement_name="end_to_end_inference_latency",
                run_index=warmup_index,
                start_time_utc=timing["start_time_utc"],
                end_time_utc=timing["end_time_utc"],
                elapsed_sec=timing["elapsed_sec"],
                start_perf_counter_sec=timing["start_perf_counter_sec"],
                end_perf_counter_sec=timing["end_perf_counter_sec"],
                start_point=INFERENCE_START_POINT,
                end_point=INFERENCE_END_POINT,
                measurement_value=timing["elapsed_sec"],
                units="sec",
            )

        latencies = []
        latency_measurements_sec: list[float] = []
        stage_sums_sec: dict[str, float] = {}
        for run_index in range(1, runs_per_sample + 1):
            stage_timings, timing = capture_timed_call(_run_once)
            latencies.append(timing["elapsed_sec"])
            latency_measurements_sec.append(timing["elapsed_sec"])
            for stage_name, stage_sec in stage_timings.items():
                stage_sums_sec[stage_name] = stage_sums_sec.get(
                    stage_name, 0.0
                ) + float(stage_sec)
            audit_logger.append_event(
                event_kind="timing_measurement",
                phase="timed",
                measurement_name="end_to_end_inference_latency",
                run_index=run_index,
                start_time_utc=timing["start_time_utc"],
                end_time_utc=timing["end_time_utc"],
                elapsed_sec=timing["elapsed_sec"],
                start_perf_counter_sec=timing["start_perf_counter_sec"],
                end_perf_counter_sec=timing["end_perf_counter_sec"],
                start_point=INFERENCE_START_POINT,
                end_point=INFERENCE_END_POINT,
                measurement_value=timing["elapsed_sec"],
                units="sec",
                details={"reported_stage_timings_sec": stage_timings},
            )

        summary = summarize_latency_measurements(latencies)
        estimated_flops = estimate_layer_flops(spec)
        estimated_bytes = estimate_layer_bytes(spec)
        throughput_flops_per_sec = (
            estimated_flops
            * summary["measured_inference_count"]
            / summary["timed_total_sec"]
        )
        avg_iteration_sec = (
            summary["timed_total_sec"] / summary["measured_inference_count"]
        )
        power_probe_runs = resolve_power_probe_runs(
            avg_iteration_sec=avg_iteration_sec,
            baseline_runs=runs_per_sample,
            min_measurement_duration_sec=0.25,
        )
        power_probe_window_sec = avg_iteration_sec * power_probe_runs
        effective_power_sample_interval_sec = resolve_power_sample_interval_sec(
            requested_interval_sec=power_sample_interval_sec,
            estimated_timed_window_sec=power_probe_window_sec,
        )
        power_phase_timing = None
        if power_backend == "none":
            power_stats = empty_power_stats()
            _log_power_measurements(
                audit_logger,
                power_monitor=None,
                power_backend=power_backend,
                power_phase_timing={
                    "start_time_utc": None,
                    "end_time_utc": None,
                    "elapsed_sec": 0.0,
                    "start_perf_counter_sec": None,
                    "end_perf_counter_sec": None,
                },
                effective_power_sample_interval_sec=effective_power_sample_interval_sec,
                power_probe_runs=power_probe_runs,
                power_stats=power_stats,
            )
        else:
            with create_power_monitor(
                power_backend=power_backend,
                sample_interval_sec=effective_power_sample_interval_sec,
                quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
                estimated_timed_window_sec=power_probe_window_sec,
            ) as power_monitor:
                for run_index in range(1, power_probe_runs + 1):
                    _, timing = capture_timed_call(_run_once)
                    if power_phase_timing is None:
                        power_phase_timing = dict(timing)
                    else:
                        power_phase_timing["end_time_utc"] = timing["end_time_utc"]
                        power_phase_timing["end_perf_counter_sec"] = timing[
                            "end_perf_counter_sec"
                        ]
                        power_phase_timing["elapsed_sec"] = float(
                            power_phase_timing["end_perf_counter_sec"]
                        ) - float(power_phase_timing["start_perf_counter_sec"])
                    audit_logger.append_event(
                        event_kind="timing_measurement",
                        phase="power_probe",
                        measurement_name="end_to_end_inference_latency",
                        run_index=run_index,
                        start_time_utc=timing["start_time_utc"],
                        end_time_utc=timing["end_time_utc"],
                        elapsed_sec=timing["elapsed_sec"],
                        start_perf_counter_sec=timing["start_perf_counter_sec"],
                        end_perf_counter_sec=timing["end_perf_counter_sec"],
                        start_point=INFERENCE_START_POINT,
                        end_point=INFERENCE_END_POINT,
                        measurement_value=timing["elapsed_sec"],
                        units="sec",
                    )
                    time.sleep(0)
            if power_phase_timing is None:
                power_phase_timing = {
                    "start_time_utc": None,
                    "end_time_utc": None,
                    "elapsed_sec": 0.0,
                    "start_perf_counter_sec": None,
                    "end_perf_counter_sec": None,
                }
            power_stats = power_monitor.stats(float(power_phase_timing["elapsed_sec"]))
            if power_stats.get("avg_power_w") is not None:
                power_stats["energy_j"] = (
                    float(power_stats["avg_power_w"]) * summary["timed_total_sec"]
                )
            _log_power_measurements(
                audit_logger,
                power_monitor=power_monitor,
                power_backend=power_backend,
                power_phase_timing=power_phase_timing,
                effective_power_sample_interval_sec=effective_power_sample_interval_sec,
                power_probe_runs=power_probe_runs,
                power_stats=power_stats,
            )
        if power_stats.get("power_backend") is None:
            power_stats["power_backend"] = power_backend
        energy_efficiency_flops_per_joule = flops_per_joule(
            throughput_flops_per_sec=throughput_flops_per_sec,
            avg_power_w=power_stats.get("avg_power_w"),
        )

        row = {
            "study_id": study_id,
            "study_case_id": (
                None if study_case_id in (None, "", "None") else str(study_case_id)
            ),
            "study_case_label": (
                None
                if study_case_label in (None, "", "None")
                else str(study_case_label)
            ),
            "backend": "npu",
            "execution_mode": "operator_runlist",
            "pattern_label": "operator_runlist",
            "input_boundary": spec.input_boundary,
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
            **_average_stage_timings_ms(
                stage_sums_sec,
                summary["measured_inference_count"],
            ),
            "throughput_flops_per_sec": throughput_flops_per_sec,
            "estimated_flops_per_inference": estimated_flops,
            "estimated_bytes_per_inference": estimated_bytes,
            "operational_intensity_flops_per_byte": operational_intensity(
                estimated_flops,
                estimated_bytes,
            ),
            "backend_peak_ops_per_sec": None,
            "roofline_bound_ops_per_sec": None,
            "backend_pct_of_peak": None,
            "roofline_pct": None,
            "flops_per_joule": energy_efficiency_flops_per_joule,
            "gflops_per_joule": gflops_per_joule(
                throughput_flops_per_sec=throughput_flops_per_sec,
                avg_power_w=power_stats.get("avg_power_w"),
            ),
            "run_status": "completed",
            "failure_component": None,
            "failure_category": None,
            "failure_message": None,
            "measurement_log_path": measurement_log_path,
            "measurement_session_id": (
                audit_logger.session_id if measurement_log_path is not None else None
            ),
            **summary,
            **_pattern_metadata(pattern),
            **power_stats,
        }
        row["process_model"] = "child_process"
        audit_logger.append_event(
            event_kind="benchmark_summary",
            phase="summary",
            measurement_name="benchmark_row_summary",
            details={
                "latency_measurements_sec": latency_measurements_sec,
                "summary_row": row,
            },
        )
        audit_logger.append_event(
            event_kind="benchmark_session_completed",
            phase="session",
            details={
                "measured_inference_count": summary["measured_inference_count"],
                "timed_total_sec": summary["timed_total_sec"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "process_model": "child_process",
            },
        )
        return row
    except Exception as exc:
        audit_logger.append_event(
            event_kind="benchmark_session_failed",
            phase="session",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "process_model": "child_process",
            },
        )
        raise
    finally:
        audit_logger.flush()


def parity_operator_runlist_request(request: dict[str, object]) -> dict[str, object]:
    spec = TransformerLayerSpec.from_dict(request["spec"])
    seed = int(request["seed"])
    study_id = str(request.get("study_id", "synthetic_transformer_layer"))

    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)

    reference = ReferenceTransformerLayer(spec)
    reference.assign_weights(weights)
    reference_output = reference(layer_inputs)

    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(weights)
    candidate_output = pattern(layer_inputs)
    _ensure_finite_output(candidate_output, mode="parity", spec=spec)
    diff = (reference_output - candidate_output).abs().to(candidate_output.dtype)

    return {
        "study_id": study_id,
        "execution_mode": "operator_runlist",
        "input_boundary": spec.input_boundary,
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "weights_source": spec.weights_source,
        "seed": seed,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.float().mean().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run operator_runlist benchmark in an isolated child process."
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        request_path = Path(args.request_json)
        response_path = Path(args.response_json)
        request = json.loads(request_path.read_text())
        mode = str(request.get("mode", "benchmark"))
        if mode == "benchmark":
            row = benchmark_operator_runlist_request(request)
        elif mode == "parity":
            row = parity_operator_runlist_request(request)
        else:
            raise ValueError(f"Unsupported operator_runlist worker mode: {mode}")
        response_path.write_text(json.dumps(row))
        os._exit(0)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
