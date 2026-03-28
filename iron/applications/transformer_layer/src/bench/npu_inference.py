# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .benchmark_common import summarize_latency_measurements, write_results_csv
from .benchmark_power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
)
from .measurement_log import (
    MeasurementAuditLogger,
    capture_timed_call,
    default_measurement_log_path,
)
from ..analysis.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
    flops_per_joule,
    gflops_per_joule,
    operational_intensity,
)
from ..core.layer_spec import TransformerLayerSpec
from ..patterns.dataflow import DataflowPattern
from ..patterns.dataflow_blocks import (
    Block1QKVProjPattern,
    Block2MHAOutProjPattern,
    Block3AddNormFFNAddNormPattern,
)
from ..patterns.gemm_sequences import (
    GemmOffloadGemmSequencePattern,
    RunlistGemmSequencePattern,
)
from ..patterns.gemm_offload import GemmOnlyPattern
from ..patterns.runlist import RunlistPattern
from ..utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)

SUPPORTED_EXECUTION_MODES = (
    "dataflow",
    "runlist",
    "gemm_offload",
    "block1_qkv_proj",
    "block2_mha_out_proj",
    "block3_addnorm_ffn_addnorm",
    "gemm_offload_gemm_sequence",
    "runlist_gemm_sequence",
)

INFERENCE_START_POINT = (
    "Immediately before invoking the selected NPU pattern on the synthetic "
    "transformer-layer inputs."
)
INFERENCE_END_POINT = (
    "Immediately after the selected NPU pattern returned control to the benchmark "
    "harness."
)
POWER_BASELINE_START_POINT = (
    "Immediately before collecting quiescent package-power samples with the "
    "benchmark workload idle."
)
POWER_BASELINE_END_POINT = (
    "Immediately after the quiescent package-power sampler returned its final "
    "baseline sample."
)
POWER_PROBE_START_POINT = (
    "Immediately before entering the separate same-workload power probe loop."
)
POWER_PROBE_END_POINT = (
    "Immediately after the separate same-workload power probe loop completed."
)


def _measurement_context(
    *,
    study_id: str,
    study_case_id: str | None,
    study_case_label: str | None,
    backend: str,
    execution_mode: str,
    pattern_label: str,
    spec: TransformerLayerSpec,
    measurement_log_path: str,
) -> dict[str, object]:
    return {
        "study_id": study_id,
        "study_case_id": study_case_id,
        "study_case_label": study_case_label,
        "backend": backend,
        "execution_mode": execution_mode,
        "pattern_label": pattern_label,
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "measurement_log_path": measurement_log_path,
    }


def _log_power_measurements(
    audit_logger: MeasurementAuditLogger,
    *,
    power_monitor,
    power_backend: str,
    power_phase_timing: dict[str, float | str],
    effective_power_sample_interval_sec: float,
    power_probe_runs: int,
    power_stats: dict[str, object],
) -> None:
    if power_backend == "none":
        audit_logger.append_event(
            event_kind="power_measurement_skipped",
            phase="power",
            measurement_name="power_probe",
            details={"reason": "power_backend=none"},
        )
        return
    measurement_details = {}
    if power_monitor is not None and hasattr(power_monitor, "measurement_details"):
        measurement_details = power_monitor.measurement_details()
    baseline = measurement_details.get("baseline") if measurement_details else None
    if isinstance(baseline, dict):
        audit_logger.append_event(
            event_kind="power_measurement",
            phase="power_baseline",
            measurement_name="quiescent_package_power",
            start_time_utc=baseline.get("start_time_utc"),
            end_time_utc=baseline.get("end_time_utc"),
            elapsed_sec=baseline.get("elapsed_sec"),
            start_perf_counter_sec=baseline.get("start_perf_counter_sec"),
            end_perf_counter_sec=baseline.get("end_perf_counter_sec"),
            start_point=POWER_BASELINE_START_POINT,
            end_point=POWER_BASELINE_END_POINT,
            measurement_value=baseline.get("average_power_w"),
            units="W",
            details={
                "sample_interval_sec": baseline.get("sample_interval_sec"),
                "sample_count": baseline.get("sample_count"),
                "samples": baseline.get("samples", []),
            },
        )
    probe = measurement_details.get("probe") if measurement_details else None
    audit_logger.append_event(
        event_kind="power_measurement",
        phase="power_probe",
        measurement_name="pseudo_npu_power",
        start_time_utc=power_phase_timing["start_time_utc"],
        end_time_utc=power_phase_timing["end_time_utc"],
        elapsed_sec=power_phase_timing["elapsed_sec"],
        start_perf_counter_sec=power_phase_timing["start_perf_counter_sec"],
        end_perf_counter_sec=power_phase_timing["end_perf_counter_sec"],
        start_point=POWER_PROBE_START_POINT,
        end_point=POWER_PROBE_END_POINT,
        measurement_value=power_stats.get("avg_power_w"),
        units="W",
        details={
            "power_backend": power_backend,
            "effective_power_sample_interval_sec": effective_power_sample_interval_sec,
            "power_probe_runs": power_probe_runs,
            "raw_package_avg_power_w": power_stats.get("raw_package_avg_power_w"),
            "raw_package_max_power_w": power_stats.get("raw_package_max_power_w"),
            "quiescent_package_power_w": power_stats.get("quiescent_package_power_w"),
            "max_power_w": power_stats.get("max_power_w"),
            "energy_j": power_stats.get("energy_j"),
            "power_sample_count": power_stats.get("power_sample_count"),
            "sample_count": (
                probe.get("sample_count") if isinstance(probe, dict) else None
            ),
            "sample_interval_sec": (
                probe.get("sample_interval_sec") if isinstance(probe, dict) else None
            ),
            "samples": probe.get("samples", []) if isinstance(probe, dict) else [],
        },
    )


def _run_pattern_with_stage_timings(pattern, layer_inputs):
    if hasattr(pattern, "forward_with_stage_timings"):
        return pattern.forward_with_stage_timings(layer_inputs)
    start = time.perf_counter()
    output = pattern(layer_inputs)
    end = time.perf_counter()
    return output, {"pattern_sec": end - start}


def _average_stage_timings_ms(
    stage_sums_sec: dict[str, float], measured_inference_count: int
) -> dict[str, float]:
    if measured_inference_count <= 0:
        return {}
    row = {}
    for stage_name, total_sec in stage_sums_sec.items():
        if not stage_name.endswith("_sec"):
            continue
        metric_name = f"avg_{stage_name.removesuffix('_sec')}_latency_ms"
        row[metric_name] = (total_sec / measured_inference_count) * 1000.0
    return row


def _pattern_metadata(pattern) -> dict[str, object]:
    if hasattr(pattern, "get_benchmark_metadata"):
        return pattern.get_benchmark_metadata()
    return {}


def build_pattern(execution_mode: str, spec: TransformerLayerSpec):
    if execution_mode == "dataflow":
        return DataflowPattern(spec)
    if execution_mode == "runlist":
        return RunlistPattern(spec)
    if execution_mode == "gemm_offload":
        return GemmOnlyPattern(spec)
    if execution_mode == "block1_qkv_proj":
        return Block1QKVProjPattern(spec)
    if execution_mode == "block2_mha_out_proj":
        return Block2MHAOutProjPattern(spec)
    if execution_mode == "block3_addnorm_ffn_addnorm":
        return Block3AddNormFFNAddNormPattern(spec)
    if execution_mode == "gemm_offload_gemm_sequence":
        return GemmOffloadGemmSequencePattern(spec)
    if execution_mode == "runlist_gemm_sequence":
        return RunlistGemmSequencePattern(spec)
    raise ValueError(f"Unsupported execution_mode: {execution_mode}")


def _cleanup_pattern_runtime(pattern) -> None:
    context = getattr(pattern, "context", None)
    if context is None or not hasattr(context, "reset_runtime"):
        return
    try:
        context.reset_runtime()
    except Exception:
        logging.exception("Failed to reset AIE runtime after benchmark run")


def _benchmark_runlist_isolated(
    *,
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    output_csv: str,
    seed: int,
    power_backend: str,
    power_sample_interval_sec: float,
    quiescent_baseline_duration_sec: float,
    study_id: str,
    study_case_id: str | None,
    study_case_label: str | None,
    measurement_log_path: str | None,
    measurement_session_id: str | None,
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    request = {
        "spec": spec.to_dict(),
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "seed": seed,
        "power_backend": power_backend,
        "power_sample_interval_sec": power_sample_interval_sec,
        "quiescent_baseline_duration_sec": quiescent_baseline_duration_sec,
        "study_id": study_id,
        "study_case_id": study_case_id,
        "study_case_label": study_case_label,
    }
    if measurement_log_path is not None:
        request["measurement_log_path"] = measurement_log_path
    if measurement_session_id is not None:
        request["measurement_session_id"] = measurement_session_id
    with tempfile.TemporaryDirectory(
        prefix="transformer_layer_operator_runlist_"
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        request_path = temp_dir_path / "request.json"
        response_path = temp_dir_path / "response.json"
        request_path.write_text(json.dumps(request))
        command = [
            sys.executable,
            "-m",
            "iron.applications.transformer_layer.src.pipeline.operator_runlist_worker",
            "--request-json",
            str(request_path),
            "--response-json",
            str(response_path),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout
            raise RuntimeError(
                "runlist child process failed with exit code "
                f"{result.returncode}" + (f": {detail}" if detail else "")
            )
        if not response_path.exists():
            raise RuntimeError(
                "runlist child process did not produce a response payload"
            )
        row = json.loads(response_path.read_text())
    if write_immediately:
        write_results_csv(output_csv, [row])
    return [row]


def benchmark_pattern(
    *,
    execution_mode: str,
    spec: TransformerLayerSpec,
    warmup_runs: int,
    runs_per_sample: int,
    output_csv: str,
    seed: int,
    power_backend: str = "none",
    power_sample_interval_sec: float = 0.05,
    quiescent_baseline_duration_sec: float = 0.5,
    study_id: str = "synthetic_transformer_layer",
    study_case_id: str | None = None,
    study_case_label: str | None = None,
    enable_measurement_log: bool = False,
    measurement_log_path: str | None = None,
    write_immediately: bool = True,
) -> list[dict[str, object]]:
    resolved_measurement_log_path = measurement_log_path
    if resolved_measurement_log_path is None and enable_measurement_log:
        resolved_measurement_log_path = default_measurement_log_path(output_csv)
    audit_logger = MeasurementAuditLogger(
        resolved_measurement_log_path,
        context=_measurement_context(
            study_id=study_id,
            study_case_id=study_case_id,
            study_case_label=study_case_label,
            backend="npu",
            execution_mode=execution_mode,
            pattern_label=execution_mode,
            spec=spec,
            measurement_log_path=resolved_measurement_log_path,
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
            "output_csv": output_csv,
        },
    )
    if execution_mode == "runlist":
        try:
            return _benchmark_runlist_isolated(
                spec=spec,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                output_csv=output_csv,
                seed=seed,
                power_backend=power_backend,
                power_sample_interval_sec=power_sample_interval_sec,
                quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
                study_id=study_id,
                study_case_id=study_case_id,
                study_case_label=study_case_label,
                measurement_log_path=resolved_measurement_log_path,
                measurement_session_id=(
                    audit_logger.session_id
                    if resolved_measurement_log_path is not None
                    else None
                ),
                write_immediately=write_immediately,
            )
        except Exception as exc:
            audit_logger.append_event(
                event_kind="benchmark_session_failed",
                phase="session",
                details={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            audit_logger.flush()
            raise

    pattern = build_pattern(execution_mode, spec)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)
    try:
        pattern.assign_weights(weights)
        if hasattr(pattern, "prepare_benchmark_inputs"):
            pattern.prepare_benchmark_inputs(layer_inputs)

        for warmup_index in range(1, warmup_runs + 1):
            _, timing = capture_timed_call(lambda: pattern(layer_inputs))
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
        stage_sums_sec: dict[str, float] = {}
        latency_measurements_sec: list[float] = []
        for run_index in range(1, runs_per_sample + 1):
            (_, stage_timings), timing = capture_timed_call(
                lambda: _run_pattern_with_stage_timings(pattern, layer_inputs)
            )
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
        estimated_flops = estimate_layer_flops(spec, execution_mode=execution_mode)
        estimated_bytes = estimate_layer_bytes(spec, execution_mode=execution_mode)
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
        power_monitor = None
        if power_backend == "none":
            power_stats = empty_power_stats()
        else:
            with create_power_monitor(
                power_backend=power_backend,
                sample_interval_sec=effective_power_sample_interval_sec,
                quiescent_baseline_duration_sec=quiescent_baseline_duration_sec,
                estimated_timed_window_sec=power_probe_window_sec,
            ) as power_monitor:
                for run_index in range(1, power_probe_runs + 1):
                    _, timing = capture_timed_call(lambda: pattern(layer_inputs))
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
            "study_case_id": study_case_id,
            "study_case_label": study_case_label,
            "backend": "npu",
            "execution_mode": execution_mode,
            "pattern_label": execution_mode,
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
            "measurement_log_path": resolved_measurement_log_path,
            "measurement_session_id": (
                audit_logger.session_id
                if resolved_measurement_log_path is not None
                else None
            ),
            **summary,
            **_pattern_metadata(pattern),
            **power_stats,
        }
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
            },
        )
        if write_immediately:
            write_results_csv(output_csv, [row])
        return [row]
    except Exception as exc:
        audit_logger.append_event(
            event_kind="benchmark_session_failed",
            phase="session",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise
    finally:
        audit_logger.flush()
        _cleanup_pattern_runtime(pattern)


__all__ = [
    "SUPPORTED_EXECUTION_MODES",
    "INFERENCE_START_POINT",
    "INFERENCE_END_POINT",
    "POWER_BASELINE_START_POINT",
    "POWER_BASELINE_END_POINT",
    "POWER_PROBE_START_POINT",
    "POWER_PROBE_END_POINT",
    "build_pattern",
    "benchmark_pattern",
    "_measurement_context",
    "_log_power_measurements",
    "_average_stage_timings_ms",
    "_pattern_metadata",
]
