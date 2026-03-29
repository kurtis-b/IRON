# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile

from ..analysis import (
    annotate_results_csv,
    render_support_summary_text,
    summarize_support_rows,
)
from ..bench import (
    append_debug_event,
    classify_debug_exception,
    load_study_manifest,
    parse_execution_modes,
    parse_seq_lens,
    write_dict_rows_csv,
    write_results_csv,
)
from ..bench.npu_inference import benchmark_pattern
from ..core.layer_spec import TransformerLayerSpec

APP_DIR = Path(__file__).resolve().parents[2]


def _resolve_spec(
    args, layer_spec: dict[str, object], seq_len: int
) -> TransformerLayerSpec:
    resolved_spec = dict(layer_spec)
    resolved_spec["seq_len"] = seq_len
    if getattr(args, "hidden_size", None) is not None:
        resolved_spec["hidden_size"] = args.hidden_size
    if getattr(args, "intermediate_size", None) is not None:
        resolved_spec["intermediate_size"] = args.intermediate_size
    if getattr(args, "num_attention_heads", None) is not None:
        resolved_spec["num_attention_heads"] = args.num_attention_heads
    if getattr(args, "block1_topology_id", None) is not None:
        resolved_spec["block1_topology_id"] = args.block1_topology_id
    if getattr(args, "block2_topology_id", None) is not None:
        resolved_spec["block2_topology_id"] = args.block2_topology_id
    if getattr(args, "block3_topology_id", None) is not None:
        resolved_spec["block3_topology_id"] = args.block3_topology_id
    return TransformerLayerSpec.from_dict(resolved_spec)


def _resolve_output_csv(args, manifest: dict[str, object]) -> str:
    if args.output_csv != "transformer_layer_npu_suite.csv":
        return args.output_csv
    return manifest.get("output_csv", args.output_csv)


def _resolve_peak_reference(args, manifest: dict[str, object]) -> str | None:
    if args.peak_reference is not None:
        return args.peak_reference
    return manifest.get("peak_reference")


def _resolve_annotated_output_csv(args, manifest: dict[str, object]) -> str | None:
    if args.annotated_output_csv is not None:
        return args.annotated_output_csv
    return manifest.get("annotated_output_csv")


def _resolve_debug_log_csv(args, manifest: dict[str, object]) -> str | None:
    if args.debug_log_csv is not None:
        return args.debug_log_csv
    return manifest.get("debug_log_csv")


def _resolve_parity_config(
    args, manifest: dict[str, object]
) -> dict[str, object] | None:
    if args.skip_parity_check:
        return None
    parity = manifest.get("parity")
    if args.run_parity_check:
        parity = dict(parity or {})
        parity["enabled"] = True
    if not isinstance(parity, dict) or not parity.get("enabled", False):
        return None
    if args.parity_output_csv is not None:
        parity = dict(parity)
        parity["output_csv"] = args.parity_output_csv
    return parity


def _record_parity_results(
    *,
    debug_log_csv: str | None,
    study_id: str,
    parity: dict[str, object],
    parity_rows: list[dict[str, object]],
) -> None:
    parity_output = parity.get("output_csv")
    if parity_rows:
        if parity_output:
            write_dict_rows_csv(parity_output, parity_rows)
        append_debug_event(
            debug_log_csv,
            study_id=study_id,
            event_kind="parity_completed",
            component="parity",
            challenge="parity_validation",
            symptom="Layer-level parity completed successfully.",
            impact_on_experiment="Correctness-check outputs are available for the selected execution modes.",
            mitigation="None required.",
            status="completed",
            supporting_log_path=parity_output,
        )
        return

    append_debug_event(
        debug_log_csv,
        study_id=study_id,
        event_kind="parity_skipped",
        component="parity",
        challenge="parity_validation",
        symptom="No parity rows matched the selected execution modes and sequence lengths.",
        impact_on_experiment="No parity CSV was written for this filtered run.",
        mitigation="Run a study configuration that overlaps the manifest parity surface if parity output is required.",
        status="completed",
        supporting_log_path=parity_output,
    )


def _iter_study_cases(manifest: dict[str, object]) -> list[dict[str, object]]:
    if "study_cases" in manifest:
        return list(manifest["study_cases"])
    return [
        {
            "case_id": "default",
            "case_label": "default",
            "layer_spec": dict(manifest["layer_spec"]),
        }
    ]


def _case_execution_modes(
    case: dict[str, object], fallback_execution_modes: list[str]
) -> list[str]:
    raw = case.get("execution_modes")
    if raw is None:
        return list(fallback_execution_modes)
    return list(raw)


def _case_seq_lens(case: dict[str, object], fallback_seq_lens: list[int]) -> list[int]:
    raw = case.get("seq_lens")
    if raw is None:
        return list(fallback_seq_lens)
    return list(raw)


def _resolve_sampling(
    manifest: dict[str, object], seq_len: int, args
) -> tuple[int, int]:
    if args.warmup_runs is not None or args.runs_per_sample is not None:
        return (
            int(
                manifest["warmup_runs"]
                if args.warmup_runs is None
                else args.warmup_runs
            ),
            int(
                manifest["runs_per_sample"]
                if args.runs_per_sample is None
                else args.runs_per_sample
            ),
        )
    sampling_schedule = manifest.get("sampling_schedule")
    if isinstance(sampling_schedule, dict) and seq_len in sampling_schedule:
        payload = sampling_schedule[seq_len]
        return int(payload["warmup_runs"]), int(payload["runs_per_sample"])
    return int(manifest["warmup_runs"]), int(manifest["runs_per_sample"])


def _decorate_row_for_case(
    row: dict[str, object],
    *,
    study_id: str,
    case: dict[str, object],
    spec: TransformerLayerSpec,
) -> dict[str, object]:
    normalized = dict(row)
    normalized["study_id"] = study_id
    normalized["study_case_id"] = case["case_id"]
    normalized["study_case_label"] = case["case_label"]
    normalized["hidden_size"] = spec.hidden_size
    normalized["intermediate_size"] = spec.intermediate_size
    normalized["num_attention_heads"] = spec.num_attention_heads
    normalized["attention_head_size"] = spec.attention_head_size
    normalized.setdefault("block1_topology_id", spec.block1_topology_id)
    normalized.setdefault("block2_topology_id", spec.block2_topology_id)
    normalized.setdefault("block3_topology_id", spec.block3_topology_id)
    normalized.setdefault("run_status", "completed")
    normalized.setdefault("failure_component", None)
    normalized.setdefault("failure_category", None)
    normalized.setdefault("failure_message", None)
    return normalized


def _failure_result_row(
    *,
    study_id: str,
    case: dict[str, object],
    spec: TransformerLayerSpec,
    execution_mode: str,
    warmup_runs: int,
    runs_per_sample: int,
    exc: Exception,
) -> dict[str, object]:
    event = classify_debug_exception(exc)
    run_status = (
        "unsupported"
        if event["challenge"]
        in {
            "unsupported_topology_or_placement",
            "unsupported_pattern_surface",
            "unsupported_dma_descriptor_limits",
        }
        else "failed"
    )
    return {
        "study_id": study_id,
        "study_case_id": case["case_id"],
        "study_case_label": case["case_label"],
        "backend": "npu",
        "execution_mode": execution_mode,
        "pattern_label": execution_mode,
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "block1_topology_id": spec.block1_topology_id,
        "block2_topology_id": spec.block2_topology_id,
        "block3_topology_id": spec.block3_topology_id,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "use_bias": spec.use_bias,
        "weights_source": spec.weights_source,
        "source_model_name": spec.source_model_name,
        "source_layer_index": spec.source_layer_index,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "measured_inference_count": 0,
        "timed_total_sec": None,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "throughput_flops_per_sec": None,
        "estimated_flops_per_inference": None,
        "estimated_bytes_per_inference": None,
        "operational_intensity_flops_per_byte": None,
        "backend_peak_ops_per_sec": None,
        "roofline_bound_ops_per_sec": None,
        "backend_pct_of_peak": None,
        "roofline_pct": None,
        "power_backend": None,
        "raw_package_avg_power_w": None,
        "raw_package_max_power_w": None,
        "quiescent_package_power_w": None,
        "avg_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "flops_per_joule": None,
        "gflops_per_joule": None,
        "power_sample_count": None,
        "run_status": run_status,
        "failure_component": event["component"],
        "failure_category": event["challenge"],
        "failure_message": event["symptom"],
    }


def _load_csv_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_parity_checks(
    *,
    parity: dict[str, object],
    cases: list[dict[str, object]],
    execution_modes: list[str],
    seq_lens: list[int],
    benchmark_rows: list[dict[str, object]],
    seed: int,
    study_id: str,
    args,
) -> list[dict[str, object]]:
    validate_script = APP_DIR / "validate_npu_parity.py"
    rows: list[dict[str, object]] = []
    completed_points = {
        (
            str(row.get("study_case_id")),
            str(row.get("execution_mode")),
            int(row.get("seq_len")),
        )
        for row in benchmark_rows
        if row.get("run_status") == "completed"
    }
    with tempfile.TemporaryDirectory(prefix="transformer_layer_parity_") as temp_dir:
        temp_root = Path(temp_dir)
        for case in cases:
            case_seq_lens = _case_seq_lens(case, seq_lens)
            case_execution_modes = _case_execution_modes(case, execution_modes)
            parity_seq_lens = list(parity.get("seq_lens", case_seq_lens))
            parity_execution_modes = list(
                parity.get("execution_modes", case_execution_modes)
            )
            if not parity_seq_lens or not parity_execution_modes:
                continue
            base_spec = _resolve_spec(args, case["layer_spec"], parity_seq_lens[0])
            for execution_mode in parity_execution_modes:
                eligible_seq_lens = [
                    seq_len
                    for seq_len in parity_seq_lens
                    if (case["case_id"], execution_mode, seq_len) in completed_points
                ]
                if not eligible_seq_lens:
                    continue
                output_csv = (
                    temp_root / f"{case['case_id']}_{execution_mode}_parity.csv"
                )
                command = [
                    sys.executable,
                    str(validate_script),
                    "--execution-mode",
                    execution_mode,
                    "--seq-lens",
                    ",".join(str(seq_len) for seq_len in eligible_seq_lens),
                    "--hidden-size",
                    str(base_spec.hidden_size),
                    "--intermediate-size",
                    str(base_spec.intermediate_size),
                    "--num-attention-heads",
                    str(base_spec.num_attention_heads),
                    "--seed",
                    str(seed),
                    "--output-csv",
                    str(output_csv),
                ]
                if base_spec.block1_topology_id is not None:
                    command.extend(
                        ["--block1-topology-id", base_spec.block1_topology_id]
                    )
                if base_spec.block2_topology_id is not None:
                    command.extend(
                        ["--block2-topology-id", base_spec.block2_topology_id]
                    )
                if base_spec.block3_topology_id is not None:
                    command.extend(
                        ["--block3-topology-id", base_spec.block3_topology_id]
                    )
                subprocess.run(command, check=True)
                mode_rows = _load_csv_rows(output_csv)
                for row in mode_rows:
                    row["study_id"] = study_id
                    row["study_case_id"] = case["case_id"]
                    row["study_case_label"] = case["case_label"]
                    row["hidden_size"] = base_spec.hidden_size
                    row["intermediate_size"] = base_spec.intermediate_size
                    row["num_attention_heads"] = base_spec.num_attention_heads
                    row["attention_head_size"] = base_spec.attention_head_size
                rows.extend(mode_rows)
    return rows


def run_manifest_benchmark(args) -> None:
    manifest = load_study_manifest(args.study_manifest)
    study_id = str(manifest["study_id"])
    execution_modes = (
        parse_execution_modes(args.execution_modes)
        if args.execution_modes != "dataflow,runlist,gemm_offload"
        else list(manifest["execution_modes"])
    )
    seq_lens = (
        parse_seq_lens(args.seq_lens)
        if args.seq_lens != "64,128,256,512"
        else list(manifest["seq_lens"])
    )
    output_csv = _resolve_output_csv(args, manifest)
    debug_log_csv = _resolve_debug_log_csv(args, manifest)
    peak_reference = _resolve_peak_reference(args, manifest)
    annotated_output_csv = _resolve_annotated_output_csv(args, manifest)
    study_cases = _iter_study_cases(manifest)
    append_debug_event(
        debug_log_csv,
        study_id=study_id,
        event_kind="study_started",
        component="automation",
        challenge="study_execution",
        symptom=f"Starting manifest-driven run for {study_id}",
        impact_on_experiment="The requested design-pattern study is beginning.",
        mitigation="None required.",
        status="started",
        supporting_log_path=output_csv,
    )
    all_rows = []
    try:
        for case in study_cases:
            case_seq_lens = _case_seq_lens(case, seq_lens)
            case_execution_modes = _case_execution_modes(case, execution_modes)
            for seq_len in case_seq_lens:
                warmup_runs, runs_per_sample = _resolve_sampling(
                    manifest, seq_len, args
                )
                spec = _resolve_spec(args, case["layer_spec"], seq_len)
                for execution_mode in case_execution_modes:
                    try:
                        rows = benchmark_pattern(
                            execution_mode=execution_mode,
                            spec=spec,
                            warmup_runs=warmup_runs,
                            runs_per_sample=runs_per_sample,
                            output_csv=output_csv,
                            seed=args.seed,
                            power_backend=args.power_backend,
                            power_sample_interval_sec=args.power_sample_interval_sec,
                            quiescent_baseline_duration_sec=args.quiescent_baseline_duration_sec,
                            study_id=study_id,
                            study_case_id=case["case_id"],
                            study_case_label=case["case_label"],
                            enable_measurement_log=args.enable_measurement_log,
                            measurement_log_path=args.measurement_log_path,
                            write_immediately=False,
                        )
                    except Exception as exc:
                        event = classify_debug_exception(exc)
                        append_debug_event(
                            debug_log_csv,
                            study_id=study_id,
                            event_kind="benchmark_case_failed",
                            pattern=execution_mode,
                            seq_len=seq_len,
                            supporting_log_path=output_csv,
                            **event,
                        )
                        if not manifest.get("continue_on_error", False):
                            raise
                        all_rows.append(
                            _failure_result_row(
                                study_id=study_id,
                                case=case,
                                spec=spec,
                                execution_mode=execution_mode,
                                warmup_runs=warmup_runs,
                                runs_per_sample=runs_per_sample,
                                exc=exc,
                            )
                        )
                        continue
                    decorated_rows = [
                        _decorate_row_for_case(
                            row,
                            study_id=study_id,
                            case=case,
                            spec=spec,
                        )
                        for row in rows
                    ]
                    all_rows.extend(decorated_rows)
                    if decorated_rows:
                        row = decorated_rows[-1]
                        append_debug_event(
                            debug_log_csv,
                            study_id=study_id,
                            event_kind="benchmark_case_completed",
                            component="benchmark_runner",
                            pattern=execution_mode,
                            seq_len=seq_len,
                            challenge="binary_generation_or_reuse",
                            symptom=(
                                f"Completed with {row.get('npu_dispatch_count')} dispatches, "
                                f"{row.get('npu_unique_instruction_binary_count')} instruction binaries, "
                                f"avg_latency_ms={row.get('avg_latency_ms')}"
                            ),
                            impact_on_experiment="The requested case completed and produced a benchmark row.",
                            mitigation="Use the recorded dispatch, topology, and instruction-binary counts when interpreting programmability overhead.",
                            status="completed",
                            supporting_log_path=output_csv,
                        )
        if all_rows:
            write_results_csv(output_csv, all_rows)
            if peak_reference and annotated_output_csv:
                try:
                    annotate_results_csv(
                        input_csv=output_csv,
                        peak_reference_path=peak_reference,
                        output_csv=annotated_output_csv,
                    )
                except Exception as exc:
                    event = classify_debug_exception(exc)
                    append_debug_event(
                        debug_log_csv,
                        study_id=study_id,
                        event_kind="roofline_annotation_failed",
                        pattern=None,
                        seq_len=None,
                        supporting_log_path=annotated_output_csv,
                        **event,
                    )
                    raise
                append_debug_event(
                    debug_log_csv,
                    study_id=study_id,
                    event_kind="roofline_annotation_completed",
                    component="roofline",
                    challenge="roofline_annotation",
                    symptom="Annotated suite CSV was written successfully.",
                    impact_on_experiment="Percent-of-peak and roofline metrics are available for analysis.",
                    mitigation="None required.",
                    status="completed",
                    supporting_log_path=annotated_output_csv,
                )
            support_matrix_csv = manifest.get("support_matrix_csv")
            support_matrix_text = manifest.get("support_matrix_text")
            if support_matrix_csv or support_matrix_text:
                support_rows = summarize_support_rows(all_rows)
                if support_matrix_csv:
                    write_dict_rows_csv(support_matrix_csv, support_rows)
                if support_matrix_text:
                    output_path = Path(support_matrix_text)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        render_support_summary_text(support_rows), encoding="utf-8"
                    )

        parity = _resolve_parity_config(args, manifest)
        if parity is not None:
            try:
                parity_rows = _run_parity_checks(
                    parity=parity,
                    cases=study_cases,
                    execution_modes=execution_modes,
                    seq_lens=seq_lens,
                    benchmark_rows=all_rows,
                    seed=args.seed,
                    study_id=study_id,
                    args=args,
                )
            except Exception as exc:
                event = classify_debug_exception(exc)
                append_debug_event(
                    debug_log_csv,
                    study_id=study_id,
                    event_kind="parity_failed",
                    pattern=None,
                    seq_len=None,
                    supporting_log_path=parity.get("output_csv"),
                    **event,
                )
                raise
            _record_parity_results(
                debug_log_csv=debug_log_csv,
                study_id=study_id,
                parity=parity,
                parity_rows=parity_rows,
            )
    except Exception:
        append_debug_event(
            debug_log_csv,
            study_id=study_id,
            event_kind="study_failed",
            component="automation",
            challenge="study_execution",
            symptom=f"Study {study_id} failed before all requested outputs were produced.",
            impact_on_experiment="The requested manifest run did not complete.",
            mitigation="Inspect earlier failure rows in the same debug log and rerun the failing stage in isolation.",
            status="failed",
            supporting_log_path=output_csv,
        )
        raise
    append_debug_event(
        debug_log_csv,
        study_id=study_id,
        event_kind="study_completed",
        component="automation",
        challenge="study_execution",
        symptom=f"Study {study_id} completed successfully.",
        impact_on_experiment="The requested suite outputs are available for thesis analysis.",
        mitigation="None required.",
        status="completed",
        supporting_log_path=output_csv,
    )


__all__ = [
    "run_manifest_benchmark",
    "_record_parity_results",
]
