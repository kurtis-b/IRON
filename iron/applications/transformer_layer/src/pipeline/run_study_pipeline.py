# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..bench import append_debug_event

APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parents[2]
RESULTS_ROOT = APP_DIR / "results"
AUTOMATED_BENCHMARK = APP_DIR / "automated_benchmark.py"
BOTTLENECK_ANALYSIS = APP_DIR / "analyze_design_pattern_bottlenecks.py"
GPU_COMPARE = APP_DIR / "gpu_compare_best_npu.py"
PLOT_RESULTS = APP_DIR / "plot_design_pattern_results.py"
BENCHMARK_STEP_KINDS = {"npu_study", "gpu_compare"}
BLOCK_TOPOLOGY_FLAG_MAP = {
    "block1_topology_id": "--block1-topology-id",
    "block2_topology_id": "--block2-topology-id",
    "block3_topology_id": "--block3-topology-id",
}


def resolve_path(base_path: str | Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(base_path).resolve().parent / path).resolve())


def load_pipeline_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "pipeline_id" not in config or "steps" not in config:
        raise KeyError("Pipeline config must include pipeline_id and steps")
    config["config_path"] = str(config_path)
    config["fail_fast"] = bool(config.get("fail_fast", True))
    config["debug_log_csv"] = resolve_path(config_path, config.get("debug_log_csv"))

    thermal_recovery = dict(
        config.get("thermal_recovery", config.get("power_cycle", {}))
    )
    thermal_recovery.setdefault("enabled", True)
    thermal_recovery.setdefault("skip_in_smoke", True)
    thermal_recovery.setdefault("temperature_sensor_name", "k10temp")
    thermal_recovery.setdefault("temperature_sensor_input", "temp1_input")
    thermal_recovery.setdefault("recovery_tolerance_fraction", 0.05)
    thermal_recovery.setdefault("recovery_timeout_sec", 15.0)
    thermal_recovery.setdefault("recovery_poll_interval_sec", 1.0)
    config["thermal_recovery"] = thermal_recovery

    npu_power = dict(config.get("npu_power", {}))
    npu_power.setdefault("power_backend", "turbostat_pkgwatt")
    npu_power.setdefault("power_sample_interval_sec", 0.05)
    npu_power.setdefault("quiescent_baseline_duration_sec", 0.5)
    config["npu_power"] = npu_power

    normalized_steps = []
    for index, raw_step in enumerate(config["steps"]):
        step = dict(raw_step)
        step.setdefault("step_id", f"step_{index}")
        step.setdefault("enabled", True)
        step.setdefault("benchmarking_step", step.get("kind") in BENCHMARK_STEP_KINDS)
        if step.get("manifest") is not None:
            step["manifest"] = resolve_path(config_path, step["manifest"])
        if step.get("config") is not None:
            step["config"] = resolve_path(config_path, step["config"])
        for key in (
            "input_csv",
            "summary_csv",
            "summary_json",
            "summary_text",
            "bottleneck_csv",
            "gpu_compare_csv",
            "output_dir",
        ):
            if step.get(key) is not None:
                step[key] = resolve_path(config_path, step[key])
        normalized_steps.append(step)
    config["steps"] = normalized_steps
    return config


def _sanitize_relative_output_path(raw_value: str) -> Path:
    path = Path(str(raw_value))
    if path.is_absolute():
        try:
            return Path("results") / path.relative_to(RESULTS_ROOT)
        except ValueError:
            pass
        try:
            return path.relative_to(REPO_ROOT)
        except ValueError:
            return Path(path.name)
    parts = [part for part in path.parts if part not in (".", "..", "/")]
    return Path(*parts) if parts else Path(path.name)


def _rewrite_output_path(raw_value: str, output_root: Path | None) -> str:
    if output_root is None:
        return raw_value
    return str((output_root / _sanitize_relative_output_path(raw_value)).resolve())


def _rewrite_study_manifest(
    *,
    manifest_path: Path,
    output_root: Path | None,
    smoke: bool,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "output_csv",
        "debug_log_csv",
        "annotated_output_csv",
        "support_matrix_csv",
        "support_matrix_text",
    ):
        if key in manifest and manifest[key] is not None:
            if output_root is None:
                manifest[key] = resolve_path(manifest_path, str(manifest[key]))
            else:
                manifest[key] = _rewrite_output_path(str(manifest[key]), output_root)
    if manifest.get("peak_reference") is not None:
        manifest["peak_reference"] = resolve_path(
            manifest_path, str(manifest["peak_reference"])
        )
    parity = manifest.get("parity")
    if isinstance(parity, dict) and parity.get("output_csv") is not None:
        manifest["parity"] = dict(parity)
        if output_root is None:
            manifest["parity"]["output_csv"] = resolve_path(
                manifest_path,
                str(parity["output_csv"]),
            )
        else:
            manifest["parity"]["output_csv"] = _rewrite_output_path(
                str(parity["output_csv"]),
                output_root,
            )
        parity = manifest["parity"]

    if smoke:
        if "seq_lens" in manifest and manifest["seq_lens"]:
            manifest["seq_lens"] = [int(manifest["seq_lens"][0])]
        if "study_cases" in manifest:
            patched_cases = []
            for raw_case in manifest["study_cases"]:
                case = dict(raw_case)
                if case.get("seq_lens"):
                    case["seq_lens"] = [int(case["seq_lens"][0])]
                patched_cases.append(case)
            manifest["study_cases"] = patched_cases
        if isinstance(parity, dict) and parity.get("enabled"):
            parity_seq_lens = [
                int(seq_len)
                for seq_len in parity.get("seq_lens", manifest.get("seq_lens", []))
            ]
            allowed_seq_lens = {
                int(seq_len) for seq_len in manifest.get("seq_lens", parity_seq_lens)
            }
            intersected = [
                seq_len for seq_len in parity_seq_lens if seq_len in allowed_seq_lens
            ]
            manifest["parity"] = dict(parity)
            manifest["parity"]["seq_lens"] = intersected or list(allowed_seq_lens)
    return manifest


def _rewrite_gpu_compare_config(
    *,
    config_path: Path,
    output_root: Path | None,
    smoke: bool,
    warmup_runs: int | None,
    runs_per_sample: int | None,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("reference_npu_csv", "output_csv"):
        if config.get(key) is not None:
            if output_root is None:
                config[key] = resolve_path(config_path, str(config[key]))
            else:
                config[key] = _rewrite_output_path(str(config[key]), output_root)
    if smoke or warmup_runs is not None or runs_per_sample is not None:
        if warmup_runs is not None:
            config["warmup_runs"] = int(warmup_runs)
        if runs_per_sample is not None:
            config["runs_per_sample"] = int(runs_per_sample)
        if isinstance(config.get("sampling_schedule"), dict):
            updated = {}
            for raw_seq_len, payload in config["sampling_schedule"].items():
                updated[str(raw_seq_len)] = {
                    "warmup_runs": int(
                        warmup_runs
                        if warmup_runs is not None
                        else payload.get("warmup_runs", config["warmup_runs"])
                    ),
                    "runs_per_sample": int(
                        runs_per_sample
                        if runs_per_sample is not None
                        else payload.get("runs_per_sample", config["runs_per_sample"])
                    ),
                }
            config["sampling_schedule"] = updated
    return config


def _rewrite_step_io_paths(
    step: dict[str, object], *, output_root: Path | None
) -> dict[str, object]:
    patched = dict(step)
    for key in (
        "input_csv",
        "summary_csv",
        "summary_json",
        "summary_text",
        "bottleneck_csv",
        "gpu_compare_csv",
        "output_dir",
    ):
        if patched.get(key) is not None:
            patched[key] = _rewrite_output_path(str(patched[key]), output_root)
    return patched


def _write_temp_json(
    directory: Path, file_name: str, payload: dict[str, object]
) -> str:
    path = directory / file_name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _resolve_hwmon_temperature_path(sensor_name: str, sensor_input: str) -> Path:
    for hwmon_dir in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = hwmon_dir / "name"
        if not name_path.exists():
            continue
        if name_path.read_text(encoding="utf-8").strip() != sensor_name:
            continue
        input_path = hwmon_dir / sensor_input
        if input_path.exists():
            return input_path
    raise FileNotFoundError(
        f"Could not locate hwmon sensor {sensor_name!r} with input {sensor_input!r}"
    )


def _read_temperature_c(sensor_path: Path) -> float:
    raw_value = sensor_path.read_text(encoding="utf-8").strip()
    return float(raw_value) / 1000.0


def _wait_for_temperature_recovery(
    *,
    sensor_path: Path,
    initial_temp_c: float,
    tolerance_fraction: float,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[bool, float]:
    deadline = time.monotonic() + timeout_sec
    while True:
        current_temp_c = _read_temperature_c(sensor_path)
        if abs(current_temp_c - initial_temp_c) <= initial_temp_c * tolerance_fraction:
            return True, current_temp_c
        if time.monotonic() >= deadline:
            return False, current_temp_c
        time.sleep(poll_interval_sec)


def _should_skip_thermal_recovery(args, pipeline_config: dict[str, object]) -> bool:
    if args.skip_thermal_recovery:
        return True
    thermal_recovery = dict(pipeline_config["thermal_recovery"])
    return bool(args.smoke and thermal_recovery.get("skip_in_smoke", False))


def _step_supports_npu_power(step: dict[str, object]) -> bool:
    return str(step["kind"]) == "npu_study"


def _build_step_command(
    *,
    step: dict[str, object],
    args,
    output_root: Path | None,
    temp_dir: Path,
) -> list[str]:
    kind = str(step["kind"])
    if kind == "npu_study":
        manifest_path = Path(str(step["manifest"]))
        patched_manifest = _rewrite_study_manifest(
            manifest_path=manifest_path,
            output_root=output_root,
            smoke=args.smoke,
        )
        manifest_arg = _write_temp_json(
            temp_dir,
            f"{step['step_id']}_manifest.json",
            patched_manifest,
        )
        command = [
            sys.executable,
            str(AUTOMATED_BENCHMARK),
            "--study-manifest",
            manifest_arg,
        ]
        if args.warmup_runs is not None:
            command.extend(["--warmup-runs", str(args.warmup_runs)])
        if args.runs_per_sample is not None:
            command.extend(["--runs-per-sample", str(args.runs_per_sample)])
        command.extend(
            [
                "--power-backend",
                str(args.power_backend),
                "--power-sample-interval-sec",
                str(args.power_sample_interval_sec),
                "--quiescent-baseline-duration-sec",
                str(args.quiescent_baseline_duration_sec),
            ]
        )
        for key, flag in BLOCK_TOPOLOGY_FLAG_MAP.items():
            if step.get(key) is not None:
                command.extend([flag, str(step[key])])
        return command
    if kind == "gpu_compare":
        config_path = Path(str(step["config"]))
        patched_config = _rewrite_gpu_compare_config(
            config_path=config_path,
            output_root=output_root,
            smoke=args.smoke,
            warmup_runs=args.warmup_runs,
            runs_per_sample=args.runs_per_sample,
        )
        config_arg = _write_temp_json(
            temp_dir,
            f"{step['step_id']}_gpu_compare.json",
            patched_config,
        )
        return [
            sys.executable,
            str(GPU_COMPARE),
            "--config",
            config_arg,
        ]
    if kind == "bottleneck":
        patched_step = _rewrite_step_io_paths(step, output_root=output_root)
        return [
            sys.executable,
            str(BOTTLENECK_ANALYSIS),
            "--input-csv",
            str(patched_step["input_csv"]),
            "--summary-csv",
            str(patched_step["summary_csv"]),
            "--summary-json",
            str(patched_step["summary_json"]),
            "--summary-text",
            str(patched_step["summary_text"]),
        ]
    if kind == "plot":
        patched_step = _rewrite_step_io_paths(step, output_root=output_root)
        command = [
            sys.executable,
            str(PLOT_RESULTS),
            "--input-csv",
            str(patched_step["input_csv"]),
            "--output-dir",
            str(patched_step["output_dir"]),
            "--x-axis",
            str(patched_step.get("x_axis", "seq_len")),
        ]
        if patched_step.get("bottleneck_csv") is not None:
            command.extend(["--bottleneck-csv", str(patched_step["bottleneck_csv"])])
        if patched_step.get("gpu_compare_csv") is not None:
            command.extend(["--gpu-compare-csv", str(patched_step["gpu_compare_csv"])])
        return command
    raise ValueError(f"Unsupported pipeline step kind: {kind}")


def run_study_pipeline(config: dict[str, object], args) -> None:
    output_root = Path(args.output_root).resolve() if args.output_root else None
    debug_log_csv = config.get("debug_log_csv")
    if debug_log_csv is not None and output_root is not None:
        debug_log_csv = _rewrite_output_path(str(debug_log_csv), output_root)
    initial_temp_c: float | None = None
    sensor_path: Path | None = None
    benchmarking_steps_seen = 0
    thermal_recovery_cfg = dict(config["thermal_recovery"])
    skip_thermal_recovery = _should_skip_thermal_recovery(args, config)

    append_debug_event(
        debug_log_csv,
        study_id=str(config["pipeline_id"]),
        event_kind="pipeline_started",
        component="pipeline",
        challenge="pipeline_execution",
        symptom=f"Starting unattended pipeline {config['pipeline_id']}",
        impact_on_experiment="The full study pipeline is beginning.",
        mitigation="None required.",
        status="started",
    )

    if thermal_recovery_cfg.get("enabled") and not skip_thermal_recovery:
        sensor_path = _resolve_hwmon_temperature_path(
            str(thermal_recovery_cfg["temperature_sensor_name"]),
            str(thermal_recovery_cfg["temperature_sensor_input"]),
        )
        initial_temp_c = _read_temperature_c(sensor_path)
        append_debug_event(
            debug_log_csv,
            study_id=str(config["pipeline_id"]),
            event_kind="temperature_baseline_captured",
            component="thermal_recovery",
            challenge="temperature_recovery",
            symptom=f"Captured initial package temperature baseline: {initial_temp_c:.2f}C",
            impact_on_experiment="Subsequent benchmark steps can be gated on returning near the initial thermal state.",
            mitigation="None required.",
            status="completed",
        )

    with tempfile.TemporaryDirectory(prefix="transformer_layer_pipeline_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for step in config["steps"]:
            if not step.get("enabled", True):
                continue

            if (
                bool(step.get("benchmarking_step"))
                and benchmarking_steps_seen > 0
                and thermal_recovery_cfg.get("enabled")
                and not skip_thermal_recovery
            ):
                assert sensor_path is not None
                assert initial_temp_c is not None
                recovered, current_temp_c = _wait_for_temperature_recovery(
                    sensor_path=sensor_path,
                    initial_temp_c=initial_temp_c,
                    tolerance_fraction=float(
                        thermal_recovery_cfg["recovery_tolerance_fraction"]
                    ),
                    timeout_sec=float(thermal_recovery_cfg["recovery_timeout_sec"]),
                    poll_interval_sec=float(
                        thermal_recovery_cfg["recovery_poll_interval_sec"]
                    ),
                )
                append_debug_event(
                    debug_log_csv,
                    study_id=str(config["pipeline_id"]),
                    event_kind=(
                        "temperature_recovered"
                        if recovered
                        else "temperature_recovery_timeout"
                    ),
                    component="thermal_recovery",
                    challenge="temperature_recovery",
                    symptom=(
                        f"Current package temperature is {current_temp_c:.2f}C "
                        f"before step {step['step_id']}"
                    ),
                    impact_on_experiment=(
                        "The next benchmark step is starting near the initial thermal state."
                        if recovered
                        else "The next benchmark step is proceeding after the recovery timeout."
                    ),
                    mitigation="Proceed with the requested benchmark step.",
                    status="completed" if recovered else "open",
                )

            command = _build_step_command(
                step=step,
                args=args,
                output_root=output_root,
                temp_dir=temp_dir_path,
            )
            append_debug_event(
                debug_log_csv,
                study_id=str(config["pipeline_id"]),
                event_kind="pipeline_step_started",
                component="pipeline",
                challenge="pipeline_execution",
                symptom=f"Starting step {step['step_id']}: {' '.join(shlex.quote(part) for part in command)}",
                impact_on_experiment="A pipeline stage is starting.",
                mitigation="None required.",
                status="started",
            )
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if result.returncode != 0:
                append_debug_event(
                    debug_log_csv,
                    study_id=str(config["pipeline_id"]),
                    event_kind="pipeline_step_failed",
                    component="pipeline",
                    challenge="pipeline_execution",
                    symptom=f"Step {step['step_id']} exited with return code {result.returncode}",
                    impact_on_experiment="The unattended pipeline did not finish all requested steps.",
                    mitigation="Inspect the failing step outputs and rerun that stage in isolation.",
                    status="failed",
                )
                if config.get("fail_fast", True):
                    raise SystemExit(result.returncode)
                continue
            append_debug_event(
                debug_log_csv,
                study_id=str(config["pipeline_id"]),
                event_kind="pipeline_step_completed",
                component="pipeline",
                challenge="pipeline_execution",
                symptom=f"Step {step['step_id']} completed successfully.",
                impact_on_experiment="The pipeline produced the requested outputs for this step.",
                mitigation="None required.",
                status="completed",
            )
            if step.get("benchmarking_step"):
                benchmarking_steps_seen += 1

    append_debug_event(
        debug_log_csv,
        study_id=str(config["pipeline_id"]),
        event_kind="pipeline_completed",
        component="pipeline",
        challenge="pipeline_execution",
        symptom=f"Pipeline {config['pipeline_id']} completed successfully.",
        impact_on_experiment="All requested study and post-processing steps finished.",
        mitigation="None required.",
        status="completed",
    )


__all__ = [
    "resolve_path",
    "load_pipeline_config",
    "run_study_pipeline",
]
