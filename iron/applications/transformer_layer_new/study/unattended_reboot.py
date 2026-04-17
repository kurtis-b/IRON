#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
from pathlib import Path
import pwd
import shutil
import shlex
import subprocess
import time
from typing import Any

from .block.cases import BLOCK_KINDS, FAMILY_IDS as BLOCK_FAMILY_IDS, get_case
from .end_to_end.cases import (
    EXECUTION_MODES,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    WORKLOAD_VARIANTS,
)
from .end_to_end.run_correctness_spot_checks import SPOT_CHECK_SEQ_LENS
from .end_to_end.run_staging_ablation import STAGING_ABLATION_SEQUENCE_LENGTHS
from .memcpy_bandwidth.cases import iter_cases as iter_memcpy_cases
from .memory_tile_staging.run import STAGING_SEQUENCE_LENGTHS
from .memory_tile_staging.select import STAGING_BLOCK_KINDS

LOGGER = logging.getLogger(__name__)

STUDY_PACKAGE = "iron.applications.transformer_layer_new.study"
STATE_VERSION = 3
DEFAULT_HOST_COMPARISON_16384_TTM_GB = 26
CRON_MARKER_PREFIX = "# transformer-layer-unattended:"
DEFAULT_RETRY_LIMIT = 2
DEFAULT_TEMPERATURE_THRESHOLD_RATIO = 1.05
DEFAULT_TEMPERATURE_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TEMPERATURE_MAX_WAIT_SECONDS = 5.0
TTM_CONFIG_PATH = Path("/etc/modprobe.d/ttm.conf")
TTM_PAGES_LIMIT_PATH = Path("/sys/module/ttm/parameters/pages_limit")
TEMPERATURE_SENSOR_PREFIXES = ("k10temp-", "amdgpu-", "acpitz-")
TEMPERATURE_LOG_FIELDNAMES = [
    "timestamp",
    "event",
    "run_id",
    "job_index",
    "job_id",
    "job_status",
    "baseline_temperature_c",
    "threshold_temperature_c",
    "pre_run_temperature_c",
    "post_run_temperature_c",
    "pre_run_threshold_met",
    "pre_run_check_count",
    "pre_run_wait_seconds",
    "temperature_source",
]
JOB_SPEC_KEYS = {
    "id",
    "description",
    "module",
    "argv",
    "privileged_setup",
    "max_attempts",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_results_root(run_id: str) -> Path:
    return app_root() / f"results_unattended_{run_id}"


def default_state_path(run_id: str) -> Path:
    return default_results_root(run_id) / "automation" / "state.json"


def default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_run_user() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def _user_home(run_user: str) -> Path:
    return Path(pwd.getpwnam(run_user).pw_dir)


def _ensure_results_root_owned_by_run_user(results_root: Path, run_user: str) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    if os.geteuid() != 0:
        return

    run_user_entry = pwd.getpwnam(run_user)
    uid = int(run_user_entry.pw_uid)
    gid = int(run_user_entry.pw_gid)

    for dirpath, dirnames, filenames in os.walk(results_root):
        os.chown(dirpath, uid, gid, follow_symlinks=False)
        for dirname in dirnames:
            os.chown(
                Path(dirpath) / dirname,
                uid,
                gid,
                follow_symlinks=False,
            )
        for filename in filenames:
            os.chown(
                Path(dirpath) / filename,
                uid,
                gid,
                follow_symlinks=False,
            )


def _state_lock_path(state_path: Path) -> Path:
    return state_path.with_suffix(".lock")


def _job_log_path(results_root: Path, job_index: int, job_id: str) -> Path:
    safe_job_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in job_id
    )
    return results_root / "automation" / "logs" / f"{job_index:04d}_{safe_job_id}.log"


def _automation_log_path(results_root: Path) -> Path:
    return results_root / "automation" / "runner.log"


def _temperature_log_path(results_root: Path) -> Path:
    return results_root / "automation" / "temperature_log.csv"


def _cron_marker(state_path: Path) -> str:
    return f"{CRON_MARKER_PREFIX}{state_path}"


def _bash_command(repo: Path, argv: list[str]) -> str:
    shell_parts = [
        "source /opt/xilinx/xrt/setup.sh",
        f"source {shlex.quote(str(repo / 'ironenv' / 'bin' / 'activate'))}",
        f"cd {shlex.quote(str(repo))}",
        shlex.join(argv),
    ]
    return " && ".join(shell_parts)


def _python_module_command(repo: Path, module: str, argv: list[str]) -> str:
    return _bash_command(repo, ["python3", "-m", module, *argv])


def _resolve_amd_ttm_path(run_user: str) -> str | None:
    resolved = shutil.which("amd-ttm")
    if resolved:
        return resolved
    candidates = [
        _user_home(run_user) / ".local" / "bin" / "amd-ttm",
        Path("/usr/local/bin/amd-ttm"),
        Path("/usr/bin/amd-ttm"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _amd_ttm_path_from_state(state: dict[str, Any]) -> str:
    stored = str(state.get("amd_ttm_path") or "").strip()
    if stored and Path(stored).is_file() and os.access(stored, os.X_OK):
        return stored
    resolved = _resolve_amd_ttm_path(str(state.get("run_user") or default_run_user()))
    if resolved is None:
        raise FileNotFoundError(
            "amd-ttm was not found. Install amd-debug-tools and ensure amd-ttm "
            "is available on PATH or at ~/.local/bin for the configured run user."
        )
    return resolved


def _ttm_config_exists() -> bool:
    return TTM_CONFIG_PATH.exists()


def _current_ttm_pages_limit() -> int | None:
    try:
        return int(TTM_PAGES_LIMIT_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _ttm_pages_for_gb(gb_value: int) -> int:
    return int(gb_value) * 262144


def _is_ttm_action(action: dict[str, Any]) -> bool:
    return str(action.get("action")) in {"set_ttm_gb", "clear_ttm"}


def _ttm_action_requires_reboot(action: dict[str, Any], state: dict[str, Any]) -> bool:
    current_pages = _current_ttm_pages_limit()
    action_name = str(action.get("action"))
    if action_name == "set_ttm_gb":
        target_pages = _ttm_pages_for_gb(int(action["value"]))
        return current_pages != target_pages
    if action_name == "clear_ttm":
        normal_pages = state.get("normal_ttm_pages_limit")
        try:
            normal_pages = int(normal_pages)
        except Exception:
            normal_pages = None
        if normal_pages is None or current_pages is None:
            return _ttm_config_exists()
        return current_pages != normal_pages or _ttm_config_exists()
    return False


def _module_job(
    *,
    job_id: str,
    description: str,
    module: str,
    argv: list[str],
    privileged_setup: list[dict[str, Any]] | None = None,
    max_attempts: int = DEFAULT_RETRY_LIMIT,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "description": description,
        "module": module,
        "argv": argv,
        "privileged_setup": list(privileged_setup or []),
        "max_attempts": int(max_attempts),
        "attempts": 0,
        "status": "pending",
        "log_path": "",
        "last_exit_code": None,
        "last_error": "",
        "started_at": "",
        "finished_at": "",
    }


def _setup_turbo() -> dict[str, Any]:
    return {"action": "set_turbo"}


def _setup_ttm(gb: int | None) -> dict[str, Any]:
    if gb is None:
        return {"action": "clear_ttm"}
    return {"action": "set_ttm_gb", "value": int(gb)}


def _output_paths(results_root: Path) -> dict[str, Path]:
    return {
        "block_results": results_root / "block" / "results.csv",
        "memory_tile_staging_results": results_root
        / "memory_tile_staging"
        / "results.csv",
        "end_to_end_results": results_root / "end_to_end" / "results_all_power.csv",
        "end_to_end_tuning": results_root / "end_to_end" / "tuning_all_power.csv",
        "correctness_results": results_root
        / "end_to_end"
        / "correctness_spot_checks.csv",
        "latency_variation_results": results_root
        / "end_to_end"
        / "latency_variation.csv",
        "latency_variation_plot": results_root
        / "end_to_end"
        / "latency_variation_by_pattern.svg",
        "staging_ablation_results": results_root
        / "end_to_end"
        / "staging_ablation.csv",
        "staging_ablation_plot": results_root / "end_to_end" / "staging_ablation.svg",
        "end_to_end_fairness": results_root
        / "end_to_end"
        / "fairness_repeatability.csv",
        "host_comparison_results": results_root / "host_comparison" / "results.csv",
        "host_comparison_tps_plot": results_root
        / "host_comparison"
        / "effective_gflops_comparison.svg",
        "host_comparison_tps_per_watt_plot": results_root
        / "host_comparison"
        / "effective_gflops_per_watt_comparison.svg",
        "host_comparison_fairness": results_root
        / "host_comparison"
        / "fairness_repeatability.csv",
        "memcpy_results": results_root / "memcpy_bandwidth" / "results.csv",
        "resource_usage_dir": results_root / "resource_usage",
        "roofline_dir": results_root / "roofline",
    }


def _collect_temperature_inputs(node: Any) -> list[float]:
    values: list[float] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.startswith("temp") and key.endswith("_input"):
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
            else:
                values.extend(_collect_temperature_inputs(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(_collect_temperature_inputs(item))
    return values


def _read_pc_temperature_c() -> tuple[float, str]:
    sensors_path = shutil.which("sensors")
    if sensors_path is not None:
        result = subprocess.run(
            [sensors_path, "-j"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                chip_temps: dict[str, float] = {}
                for chip_name, chip_payload in payload.items():
                    if not str(chip_name).startswith(TEMPERATURE_SENSOR_PREFIXES):
                        continue
                    readings = _collect_temperature_inputs(chip_payload)
                    if readings:
                        chip_temps[str(chip_name)] = max(readings)
                if chip_temps:
                    hottest_chip = max(chip_temps, key=chip_temps.get)
                    return float(chip_temps[hottest_chip]), f"sensors:{hottest_chip}"

    rocm_smi_path = shutil.which("rocm-smi")
    if rocm_smi_path is not None:
        result = subprocess.run(
            [rocm_smi_path, "--showtemp", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                hottest_value: float | None = None
                hottest_source = ""
                for card_name, card_payload in payload.items():
                    if not isinstance(card_payload, dict):
                        continue
                    for field_name, value in card_payload.items():
                        if "temperature" not in str(field_name).lower():
                            continue
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        if hottest_value is None or numeric > hottest_value:
                            hottest_value = numeric
                            hottest_source = f"rocm-smi:{card_name}:{field_name}"
                if hottest_value is not None:
                    return hottest_value, hottest_source

    raise RuntimeError("Unable to determine PC temperature from sensors or rocm-smi")


def _append_temperature_log_row(results_root: Path, row: dict[str, Any]) -> None:
    output_path = _temperature_log_path(results_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPERATURE_LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {field: row.get(field, "") for field in TEMPERATURE_LOG_FIELDNAMES}
        )


def _record_baseline_temperature(results_root: Path, state: dict[str, Any]) -> None:
    _append_temperature_log_row(
        results_root,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": "baseline",
            "run_id": state["run_id"],
            "baseline_temperature_c": state.get("baseline_temperature_c", ""),
            "threshold_temperature_c": (
                float(state.get("baseline_temperature_c", 0.0))
                * float(state.get("temperature_threshold_ratio", 1.0))
                if state.get("baseline_temperature_c") not in (None, "")
                else ""
            ),
            "temperature_source": state.get("temperature_source", ""),
        },
    )


def _ensure_baseline_temperature(
    state_path: Path, state: dict[str, Any]
) -> tuple[float, str]:
    baseline = state.get("baseline_temperature_c")
    source = str(state.get("temperature_source") or "")
    if baseline not in (None, ""):
        return float(baseline), source
    baseline_value, baseline_source = _read_pc_temperature_c()
    state["baseline_temperature_c"] = baseline_value
    state["temperature_source"] = baseline_source
    write_state(state_path, state)
    _record_baseline_temperature(Path(state["results_root"]), state)
    return baseline_value, baseline_source


def _wait_for_temperature_gate(
    *,
    baseline_temperature_c: float,
    threshold_ratio: float,
    poll_interval_seconds: float,
    max_wait_seconds: float,
    log_handle,
) -> dict[str, Any]:
    threshold_temperature_c = baseline_temperature_c * threshold_ratio
    max_checks = max(0, int(round(max_wait_seconds / poll_interval_seconds)))
    sample_count = 0
    current_temperature_c = float("nan")
    current_source = ""
    threshold_met = False

    for attempt in range(max_checks + 1):
        current_temperature_c, current_source = _read_pc_temperature_c()
        sample_count += 1
        if current_temperature_c <= threshold_temperature_c:
            threshold_met = True
            break
        if attempt < max_checks:
            log_handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Temperature gate waiting: current={current_temperature_c:.3f}C "
                f"threshold={threshold_temperature_c:.3f}C source={current_source}\n"
            )
            time.sleep(poll_interval_seconds)

    return {
        "temperature_source": current_source,
        "threshold_temperature_c": threshold_temperature_c,
        "pre_run_temperature_c": current_temperature_c,
        "pre_run_threshold_met": threshold_met,
        "pre_run_check_count": sample_count,
        "pre_run_wait_seconds": max(0.0, (sample_count - 1) * poll_interval_seconds),
    }


def build_job_plan(
    *,
    results_root: Path,
    host_comparison_16384_ttm_gb: int,
) -> list[dict[str, Any]]:
    paths = _output_paths(results_root)
    jobs: list[dict[str, Any]] = []
    normal_host_comparison_jobs: list[dict[str, Any]] = []
    high_ttm_host_comparison_jobs: list[dict[str, Any]] = []

    for family_id in BLOCK_FAMILY_IDS:
        for seq_len in SEQUENCE_LADDER:
            case = get_case(family_id, seq_len)
            for block_kind in BLOCK_KINDS:
                if not case.candidates(block_kind):
                    continue
                jobs.append(
                    _module_job(
                        job_id=f"block_{family_id}_{seq_len}_{block_kind}",
                        description=f"block {family_id} seq={seq_len} block={block_kind}",
                        module=f"{STUDY_PACKAGE}.block.run",
                        argv=[
                            "--family",
                            family_id,
                            "--seq-len",
                            str(seq_len),
                            "--block",
                            block_kind,
                            "--output",
                            str(paths["block_results"]),
                            "--resume-input",
                            str(paths["block_results"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    for family_id in FAMILY_IDS:
        for seq_len in STAGING_SEQUENCE_LENGTHS:
            for block_kind in STAGING_BLOCK_KINDS:
                jobs.append(
                    _module_job(
                        job_id=f"memory_tile_staging_{family_id}_{seq_len}_{block_kind}",
                        description=f"memory_tile_staging {family_id} seq={seq_len} block={block_kind}",
                        module=f"{STUDY_PACKAGE}.memory_tile_staging.run",
                        argv=[
                            "--reference-input",
                            str(paths["block_results"]),
                            "--family",
                            family_id,
                            "--seq-len",
                            str(seq_len),
                            "--block",
                            block_kind,
                            "--output",
                            str(paths["memory_tile_staging_results"]),
                            "--resume-input",
                            str(paths["memory_tile_staging_results"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    for family_id in FAMILY_IDS:
        workload_variant = (
            "decoder_gpt2" if family_id.startswith("gpt2_") else "encoder_bert"
        )
        for seq_len in SEQUENCE_LADDER:
            for execution_mode in EXECUTION_MODES:
                jobs.append(
                    _module_job(
                        job_id=f"end_to_end_{family_id}_{seq_len}_{execution_mode}",
                        description=f"end_to_end {family_id} seq={seq_len} mode={execution_mode}",
                        module=f"{STUDY_PACKAGE}.end_to_end.run",
                        argv=[
                            "--workload-variant",
                            workload_variant,
                            "--family",
                            family_id,
                            "--seq-len",
                            str(seq_len),
                            "--mode",
                            execution_mode,
                            "--power-backend",
                            "turbostat_pkgwatt",
                            "--output",
                            str(paths["end_to_end_results"]),
                            "--tuning-output",
                            str(paths["end_to_end_tuning"]),
                            "--resume-input",
                            str(paths["end_to_end_results"]),
                            "--resume-tuning-input",
                            str(paths["end_to_end_tuning"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    for family_id in FAMILY_IDS:
        workload_variant = (
            "decoder_gpt2" if family_id.startswith("gpt2_") else "encoder_bert"
        )
        for execution_mode in EXECUTION_MODES:
            for seq_len in SPOT_CHECK_SEQ_LENS:
                jobs.append(
                    _module_job(
                        job_id=f"correctness_{family_id}_{seq_len}_{execution_mode}",
                        description=f"correctness {family_id} seq={seq_len} mode={execution_mode}",
                        module=f"{STUDY_PACKAGE}.end_to_end.run_correctness_spot_checks",
                        argv=[
                            "--results-input",
                            str(paths["end_to_end_results"]),
                            "--workload-variant",
                            workload_variant,
                            "--family",
                            family_id,
                            "--mode",
                            execution_mode,
                            "--seq-len",
                            str(seq_len),
                            "--output",
                            str(paths["correctness_results"]),
                            "--resume-input",
                            str(paths["correctness_results"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    for family_id in FAMILY_IDS:
        workload_variant = (
            "decoder_gpt2" if family_id.startswith("gpt2_") else "encoder_bert"
        )
        for seq_len in SEQUENCE_LADDER:
            for execution_mode in EXECUTION_MODES:
                jobs.append(
                    _module_job(
                        job_id=f"latency_variation_{family_id}_{seq_len}_{execution_mode}",
                        description=f"latency_variation {family_id} seq={seq_len} mode={execution_mode}",
                        module=f"{STUDY_PACKAGE}.end_to_end.run_latency_variation",
                        argv=[
                            "--results-input",
                            str(paths["end_to_end_results"]),
                            "--workload-variant",
                            workload_variant,
                            "--family",
                            family_id,
                            "--mode",
                            execution_mode,
                            "--seq-len",
                            str(seq_len),
                            "--output",
                            str(paths["latency_variation_results"]),
                            "--plot-output",
                            str(paths["latency_variation_plot"]),
                            "--resume-input",
                            str(paths["latency_variation_results"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    for family_id in FAMILY_IDS:
        workload_variant = (
            "decoder_gpt2" if family_id.startswith("gpt2_") else "encoder_bert"
        )
        for seq_len in STAGING_ABLATION_SEQUENCE_LENGTHS:
            for block_kind in STAGING_BLOCK_KINDS:
                jobs.append(
                    _module_job(
                        job_id=f"staging_ablation_{family_id}_{seq_len}_{block_kind}",
                        description=f"staging_ablation {family_id} seq={seq_len} block={block_kind}",
                        module=f"{STUDY_PACKAGE}.end_to_end.run_staging_ablation",
                        argv=[
                            "--results-input",
                            str(paths["end_to_end_results"]),
                            "--staging-results",
                            str(paths["memory_tile_staging_results"]),
                            "--workload-variant",
                            workload_variant,
                            "--family",
                            family_id,
                            "--seq-len",
                            str(seq_len),
                            "--block",
                            block_kind,
                            "--output",
                            str(paths["staging_ablation_results"]),
                            "--plot-output",
                            str(paths["staging_ablation_plot"]),
                            "--resume-input",
                            str(paths["staging_ablation_results"]),
                        ],
                        privileged_setup=[_setup_turbo(), _setup_ttm(None)],
                    )
                )

    jobs.append(
        _module_job(
            job_id="end_to_end_fairness",
            description="end_to_end fairness_repeatability",
            module=f"{STUDY_PACKAGE}.end_to_end.run_fairness_repeatability",
            argv=[
                "--results-input",
                str(paths["end_to_end_results"]),
                "--output",
                str(paths["end_to_end_fairness"]),
            ],
            privileged_setup=[_setup_turbo(), _setup_ttm(None)],
            max_attempts=1,
        )
    )

    for family_id in FAMILY_IDS:
        for seq_len in SEQUENCE_LADDER:
            privileged_setup = [_setup_turbo()]
            if seq_len == 16384:
                privileged_setup.append(_setup_ttm(host_comparison_16384_ttm_gb))
            else:
                privileged_setup.append(_setup_ttm(None))
            job = _module_job(
                job_id=f"host_comparison_igpu_{family_id}_{seq_len}",
                description=f"host_comparison igpu {family_id} seq={seq_len}",
                module=f"{STUDY_PACKAGE}.host_comparison.run",
                argv=[
                    "--reference-input",
                    str(paths["end_to_end_results"]),
                    "--family",
                    family_id,
                    "--seq-len",
                    str(seq_len),
                    "--host-backends",
                    "igpu",
                    "--igpu-power-backend",
                    "rocm-smi",
                    "--output",
                    str(paths["host_comparison_results"]),
                    "--resume-input",
                    str(paths["host_comparison_results"]),
                ],
                privileged_setup=privileged_setup,
            )
            if seq_len == 16384:
                high_ttm_host_comparison_jobs.append(job)
            else:
                normal_host_comparison_jobs.append(job)

    jobs.extend(normal_host_comparison_jobs)
    jobs.append(
        _module_job(
            job_id="host_comparison_fairness",
            description="host_comparison fairness_repeatability",
            module=f"{STUDY_PACKAGE}.host_comparison.run_fairness_repeatability",
            argv=[
                "--output",
                str(paths["host_comparison_fairness"]),
            ],
            privileged_setup=[_setup_ttm(None)],
            max_attempts=1,
        )
    )

    for case in iter_memcpy_cases(
        size_filter="all",
        num_cores_filter="all",
        num_channels_filter="all",
        bypass_filter="all",
    ):
        jobs.append(
            _module_job(
                job_id=f"memcpy_{case.case_id}",
                description=f"memcpy_bandwidth {case.case_id}",
                module=f"{STUDY_PACKAGE}.memcpy_bandwidth.run",
                argv=[
                    "--size",
                    str(case.size_elements),
                    "--num-cores",
                    str(case.num_cores),
                    "--num-channels",
                    str(case.num_channels),
                    "--bypass",
                    "true" if case.bypass else "false",
                    "--output",
                    str(paths["memcpy_results"]),
                    "--resume-input",
                    str(paths["memcpy_results"]),
                ],
                privileged_setup=[_setup_turbo(), _setup_ttm(None)],
            )
        )

    jobs.extend(
        [
            _module_job(
                job_id="resource_usage_all",
                description="resource_usage all",
                module=f"{STUDY_PACKAGE}.resource_usage.run",
                argv=[
                    "--scope",
                    "all",
                    "--block-results-input",
                    str(paths["block_results"]),
                    "--end-to-end-results-input",
                    str(paths["end_to_end_results"]),
                    "--output-dir",
                    str(paths["resource_usage_dir"]),
                ],
                privileged_setup=[_setup_ttm(None)],
                max_attempts=1,
            ),
            _module_job(
                job_id="roofline_all",
                description="roofline all",
                module=f"{STUDY_PACKAGE}.roofline.run",
                argv=[
                    "--end-to-end-results-input",
                    str(paths["end_to_end_results"]),
                    "--end-to-end-tuning-input",
                    str(paths["end_to_end_tuning"]),
                    "--memcpy-results-input",
                    str(paths["memcpy_results"]),
                    "--output-dir",
                    str(paths["roofline_dir"]),
                ],
                privileged_setup=[_setup_ttm(None)],
                max_attempts=1,
            ),
        ]
    )
    jobs.extend(high_ttm_host_comparison_jobs)
    jobs.append(
        _module_job(
            job_id="regenerate_plots_all",
            description="regenerate plots all",
            module=f"{STUDY_PACKAGE}.regenerate_plots",
            argv=[
                "--results-root",
                str(results_root),
            ],
            privileged_setup=[_setup_ttm(None)],
            max_attempts=1,
        )
    )

    return jobs


def build_smoke_test_job_plan(
    *,
    results_root: Path,
    source_results_root: Path,
) -> list[dict[str, Any]]:
    return [
        _module_job(
            job_id="smoke_prepare_results",
            description="smoke prepare results",
            module=f"{STUDY_PACKAGE}.unattended_smoke_job",
            argv=[
                "prepare-results",
                "--source-root",
                str(source_results_root),
                "--target-root",
                str(results_root),
            ],
            privileged_setup=[],
            max_attempts=1,
        ),
        _module_job(
            job_id="smoke_regenerate_plots",
            description="smoke regenerate plots",
            module=f"{STUDY_PACKAGE}.regenerate_plots",
            argv=[
                "--results-root",
                str(results_root),
            ],
            privileged_setup=[],
            max_attempts=1,
        ),
        _module_job(
            job_id="smoke_verify_results",
            description="smoke verify regenerated outputs",
            module=f"{STUDY_PACKAGE}.unattended_smoke_job",
            argv=[
                "verify-results",
                "--results-root",
                str(results_root),
            ],
            privileged_setup=[],
            max_attempts=1,
        ),
    ]


def build_execution_smoke_job_plan(
    *,
    results_root: Path,
    source_results_root: Path,
) -> list[dict[str, Any]]:
    paths = _output_paths(results_root)
    family_id = "baseline_768"
    workload_variant = "encoder_bert"
    seq_len = 512

    jobs = [
        _module_job(
            job_id="execution_smoke_prepare_fixtures",
            description="execution smoke prepare fixtures",
            module=f"{STUDY_PACKAGE}.unattended_smoke_job",
            argv=[
                "prepare-execution-fixtures",
                "--source-root",
                str(source_results_root),
                "--target-root",
                str(results_root),
            ],
            privileged_setup=[],
            max_attempts=1,
        )
    ]

    for execution_mode in EXECUTION_MODES:
        jobs.append(
            _module_job(
                job_id=f"execution_smoke_end_to_end_{execution_mode}",
                description=f"execution smoke end_to_end mode={execution_mode}",
                module=f"{STUDY_PACKAGE}.end_to_end.run",
                argv=[
                    "--workload-variant",
                    workload_variant,
                    "--family",
                    family_id,
                    "--seq-len",
                    str(seq_len),
                    "--mode",
                    execution_mode,
                    "--power-backend",
                    "turbostat_pkgwatt",
                    "--output",
                    str(paths["end_to_end_results"]),
                    "--tuning-output",
                    str(paths["end_to_end_tuning"]),
                    "--resume-input",
                    str(paths["end_to_end_results"]),
                    "--resume-tuning-input",
                    str(paths["end_to_end_tuning"]),
                ],
                privileged_setup=[],
            )
        )

    for execution_mode in EXECUTION_MODES:
        jobs.append(
            _module_job(
                job_id=f"execution_smoke_correctness_{execution_mode}",
                description=f"execution smoke correctness mode={execution_mode}",
                module=f"{STUDY_PACKAGE}.end_to_end.run_correctness_spot_checks",
                argv=[
                    "--results-input",
                    str(paths["end_to_end_results"]),
                    "--workload-variant",
                    workload_variant,
                    "--family",
                    family_id,
                    "--mode",
                    execution_mode,
                    "--seq-len",
                    str(seq_len),
                    "--output",
                    str(paths["correctness_results"]),
                    "--resume-input",
                    str(paths["correctness_results"]),
                ],
                privileged_setup=[],
            )
        )

    for execution_mode in EXECUTION_MODES:
        jobs.append(
            _module_job(
                job_id=f"execution_smoke_latency_variation_{execution_mode}",
                description=f"execution smoke latency variation mode={execution_mode}",
                module=f"{STUDY_PACKAGE}.end_to_end.run_latency_variation",
                argv=[
                    "--results-input",
                    str(paths["end_to_end_results"]),
                    "--workload-variant",
                    workload_variant,
                    "--family",
                    family_id,
                    "--mode",
                    execution_mode,
                    "--seq-len",
                    str(seq_len),
                    "--output",
                    str(paths["latency_variation_results"]),
                    "--plot-output",
                    str(paths["latency_variation_plot"]),
                    "--resume-input",
                    str(paths["latency_variation_results"]),
                ],
                privileged_setup=[],
            )
        )

    jobs.extend(
        [
            _module_job(
                job_id="execution_smoke_end_to_end_fairness",
                description="execution smoke end_to_end fairness",
                module=f"{STUDY_PACKAGE}.end_to_end.run_fairness_repeatability",
                argv=[
                    "--results-input",
                    str(paths["end_to_end_results"]),
                    "--output",
                    str(paths["end_to_end_fairness"]),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
            _module_job(
                job_id="execution_smoke_host_comparison",
                description="execution smoke host comparison",
                module=f"{STUDY_PACKAGE}.host_comparison.run",
                argv=[
                    "--reference-input",
                    str(paths["end_to_end_results"]),
                    "--family",
                    family_id,
                    "--seq-len",
                    str(seq_len),
                    "--host-backends",
                    "igpu",
                    "--igpu-power-backend",
                    "rocm-smi",
                    "--output",
                    str(paths["host_comparison_results"]),
                    "--resume-input",
                    str(paths["host_comparison_results"]),
                ],
                privileged_setup=[],
            ),
            _module_job(
                job_id="execution_smoke_host_comparison_fairness",
                description="execution smoke host comparison fairness",
                module=f"{STUDY_PACKAGE}.host_comparison.run_fairness_repeatability",
                argv=[
                    "--output",
                    str(paths["host_comparison_fairness"]),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
            _module_job(
                job_id="execution_smoke_resource_usage",
                description="execution smoke resource usage",
                module=f"{STUDY_PACKAGE}.resource_usage.run",
                argv=[
                    "--scope",
                    "all",
                    "--family",
                    family_id,
                    "--seq-len",
                    str(seq_len),
                    "--block-results-input",
                    str(paths["block_results"]),
                    "--end-to-end-results-input",
                    str(paths["end_to_end_results"]),
                    "--output-dir",
                    str(paths["resource_usage_dir"]),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
            _module_job(
                job_id="execution_smoke_roofline",
                description="execution smoke roofline",
                module=f"{STUDY_PACKAGE}.roofline.run",
                argv=[
                    "--family",
                    family_id,
                    "--end-to-end-results-input",
                    str(paths["end_to_end_results"]),
                    "--end-to-end-tuning-input",
                    str(paths["end_to_end_tuning"]),
                    "--memcpy-results-input",
                    str(paths["memcpy_results"]),
                    "--output-dir",
                    str(paths["roofline_dir"]),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
            _module_job(
                job_id="execution_smoke_regenerate_plots",
                description="execution smoke regenerate plots",
                module=f"{STUDY_PACKAGE}.regenerate_plots",
                argv=[
                    "--results-root",
                    str(results_root),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
            _module_job(
                job_id="execution_smoke_verify_results",
                description="execution smoke verify results",
                module=f"{STUDY_PACKAGE}.unattended_smoke_job",
                argv=[
                    "verify-execution-results",
                    "--results-root",
                    str(results_root),
                ],
                privileged_setup=[],
                max_attempts=1,
            ),
        ]
    )
    return jobs


def create_state(
    *,
    run_id: str,
    repo: Path,
    results_root: Path,
    run_user: str,
    host_comparison_16384_ttm_gb: int,
    reboot_command: list[str],
    jobs: list[dict[str, Any]] | None = None,
    baseline_temperature_c: float | None = None,
    temperature_source: str = "",
    amd_ttm_path: str = "",
    normal_ttm_pages_limit: int | None = None,
    plan_kind: str = "full",
    source_results_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "run_id": run_id,
        "repo_root": str(repo),
        "results_root": str(results_root),
        "source_results_root": (
            "" if source_results_root is None else str(source_results_root)
        ),
        "plan_kind": plan_kind,
        "run_user": run_user,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "pending",
        "current_job_index": 0,
        "host_comparison_16384_ttm_gb": int(host_comparison_16384_ttm_gb),
        "reboot_command": reboot_command,
        "baseline_temperature_c": baseline_temperature_c,
        "temperature_source": temperature_source,
        "amd_ttm_path": amd_ttm_path,
        "normal_ttm_pages_limit": normal_ttm_pages_limit,
        "temperature_threshold_ratio": DEFAULT_TEMPERATURE_THRESHOLD_RATIO,
        "temperature_poll_interval_seconds": DEFAULT_TEMPERATURE_POLL_INTERVAL_SECONDS,
        "temperature_max_wait_seconds": DEFAULT_TEMPERATURE_MAX_WAIT_SECONDS,
        "jobs": (
            list(jobs)
            if jobs is not None
            else build_job_plan(
                results_root=results_root,
                host_comparison_16384_ttm_gb=host_comparison_16384_ttm_gb,
            )
        ),
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cron_command(state_path: Path) -> str:
    state = load_state(state_path)
    repo = Path(state["repo_root"])
    return (
        "@reboot "
        f"bash -lc {shlex.quote(_python_module_command(repo, f'{STUDY_PACKAGE}.unattended_reboot', ['run-next', '--state', str(state_path)]))} >> "
        f"{shlex.quote(str(_automation_log_path(Path(state['results_root']))))} 2>&1"
    )


def _current_crontab_lines() -> list[str]:
    result = subprocess.run(
        ["crontab", "-l"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def install_crontab_entry(state_path: Path) -> None:
    state = load_state(state_path)
    marker = _cron_marker(state_path)
    cron_line = _cron_command(state_path)
    lines = [
        line
        for line in _current_crontab_lines()
        if marker not in line and not line.startswith(f"{CRON_MARKER_PREFIX}")
    ]
    lines.extend([marker, cron_line])
    payload = "\n".join(lines).rstrip() + "\n"
    subprocess.run(["crontab", "-"], input=payload, text=True, check=True)


def remove_crontab_entry(state_path: Path) -> None:
    marker = _cron_marker(state_path)
    lines = _current_crontab_lines()
    filtered: list[str] = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line == marker:
            skip_next = True
            continue
        if marker in line:
            continue
        filtered.append(line)
    payload = "\n".join(filtered).rstrip()
    subprocess.run(
        ["crontab", "-"],
        input=(payload + "\n") if payload else "",
        text=True,
        check=True,
    )


def _run_subprocess(
    argv: list[str],
    *,
    as_root: bool,
    log_handle,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> int:
    LOGGER.info("Running command: %s", shlex.join(argv))
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if input_text is not None:
        process.communicate(input_text)
        return int(process.returncode or 0)
    return process.wait()


def _run_privileged_action(
    action: dict[str, Any], *, log_handle, state: dict[str, Any]
) -> None:
    action_name = str(action["action"])
    input_text: str | None = None
    if os.geteuid() == 0:
        prefix: list[str] = []
    else:
        prefix = ["sudo", "-n"]

    if action_name == "set_turbo":
        argv = [*prefix, "xrt-smi", "configure", "--pmode", "turbo"]
    elif action_name == "clear_ttm":
        if not _ttm_config_exists():
            log_handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                "Skipping amd-ttm --clear because no ttm.conf is present\n"
            )
            return
        argv = [*prefix, _amd_ttm_path_from_state(state), "--clear"]
        input_text = "n\n"
    elif action_name == "set_ttm_gb":
        argv = [
            *prefix,
            _amd_ttm_path_from_state(state),
            "--set",
            str(int(action["value"])),
        ]
        input_text = "n\n"
    else:
        raise ValueError(f"Unsupported privileged setup action: {action_name}")

    exit_code = _run_subprocess(
        argv,
        as_root=(os.geteuid() == 0),
        log_handle=log_handle,
        input_text=input_text,
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Privileged setup failed for {action_name} with exit code {exit_code}"
        )


def _job_command(repo: Path, run_user: str, module: str, argv: list[str]) -> list[str]:
    shell_command = _python_module_command(repo, module, argv)
    if os.geteuid() == 0:
        return ["sudo", "-u", run_user, "-H", "bash", "-lc", shell_command]
    return ["bash", "-lc", shell_command]


def _load_lock_handle(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _next_incomplete_job_index(state: dict[str, Any]) -> int | None:
    jobs = state["jobs"]
    current_index = int(state.get("current_job_index", 0))
    if current_index < len(jobs):
        current_job = jobs[current_index]
        if str(current_job.get("status", "pending")) in {
            "pending",
            "running",
            "failed_retryable",
        }:
            return current_index
    for index, job in enumerate(jobs):
        if str(job.get("status", "pending")) in {
            "pending",
            "running",
            "failed_retryable",
        }:
            return index
    return None


def _advance_past_completed_jobs(state: dict[str, Any]) -> None:
    index = int(state.get("current_job_index", 0))
    jobs = state["jobs"]
    while index < len(jobs) and str(jobs[index].get("status")) == "passed":
        index += 1
    state["current_job_index"] = index


def _prepare_state_for_resume(
    state: dict[str, Any], *, retry_failed_current: bool = False
) -> None:
    for job in state["jobs"]:
        if str(job.get("status")) == "running":
            job["status"] = "pending"
            job["started_at"] = ""
            job["finished_at"] = ""
            job["last_exit_code"] = None
            job["last_error"] = ""
    current_index = int(state.get("current_job_index", 0))
    if retry_failed_current and current_index < len(state["jobs"]):
        job = state["jobs"][current_index]
        if str(job.get("status")) in {"failed", "failed_retryable"}:
            job["status"] = "pending"
            job["started_at"] = ""
            job["finished_at"] = ""
            job["last_exit_code"] = None
            job["last_error"] = ""
    _advance_past_completed_jobs(state)


def _ensure_normal_ttm_pages_limit(state: dict[str, Any]) -> bool:
    changed = False
    normal_pages = state.get("normal_ttm_pages_limit")
    try:
        normalized_pages = int(normal_pages)
        if normal_pages != normalized_pages:
            state["normal_ttm_pages_limit"] = normalized_pages
            changed = True
        return changed
    except Exception:
        pass
    current_pages = _current_ttm_pages_limit()
    if not _ttm_config_exists() and current_pages is not None:
        state["normal_ttm_pages_limit"] = current_pages
        return True
    legacy_pages = state.get("baseline_ttm_pages_limit")
    try:
        state["normal_ttm_pages_limit"] = int(legacy_pages)
        return True
    except Exception:
        pass
    if current_pages is not None:
        state["normal_ttm_pages_limit"] = current_pages
        return True
    return changed


def _merge_job_progress(
    expected_job: dict[str, Any], existing_job: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(expected_job)
    merged.update(
        {key: value for key, value in existing_job.items() if key not in JOB_SPEC_KEYS}
    )
    return merged


def _job_specs_match(
    expected_job: dict[str, Any], existing_job: dict[str, Any]
) -> bool:
    return {key: expected_job.get(key) for key in JOB_SPEC_KEYS} == {
        key: existing_job.get(key) for key in JOB_SPEC_KEYS
    }


def _expected_jobs_for_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    plan_kind = str(state.get("plan_kind") or "full")
    results_root = Path(state["results_root"])
    host_comparison_16384_ttm_gb = int(
        state.get(
            "host_comparison_16384_ttm_gb",
            DEFAULT_HOST_COMPARISON_16384_TTM_GB,
        )
    )
    if plan_kind == "plot_smoke":
        source_results_root = str(state.get("source_results_root") or "").strip()
        if not source_results_root:
            source_results_root = str(app_root() / "results")
        return build_smoke_test_job_plan(
            results_root=results_root,
            source_results_root=Path(source_results_root),
        )
    if plan_kind == "execution_smoke":
        source_results_root = str(state.get("source_results_root") or "").strip()
        if not source_results_root:
            source_results_root = str(app_root() / "results")
        return build_execution_smoke_job_plan(
            results_root=results_root,
            source_results_root=Path(source_results_root),
        )
    return build_job_plan(
        results_root=results_root,
        host_comparison_16384_ttm_gb=host_comparison_16384_ttm_gb,
    )


def _migrate_state_for_current_runner(state: dict[str, Any]) -> bool:
    changed = _ensure_normal_ttm_pages_limit(state)
    expected_jobs = _expected_jobs_for_state(state)
    expected_ids = [str(job["id"]) for job in expected_jobs]
    existing_jobs = list(state.get("jobs", []))
    existing_ids = [str(job.get("id", "")) for job in existing_jobs]
    specs_match = len(existing_jobs) == len(expected_jobs) and all(
        _job_specs_match(expected_job, existing_job)
        for expected_job, existing_job in zip(expected_jobs, existing_jobs)
    )
    needs_job_migration = (
        int(state.get("version", 1)) < STATE_VERSION
        or existing_ids != expected_ids
        or not specs_match
    )
    if needs_job_migration:
        old_current_job_id = None
        current_job_index = int(state.get("current_job_index", 0))
        if 0 <= current_job_index < len(existing_jobs):
            old_current_job_id = str(existing_jobs[current_job_index].get("id", ""))
        existing_jobs_by_id = {
            str(job.get("id", "")): job
            for job in existing_jobs
            if str(job.get("id", ""))
        }
        state["jobs"] = [
            (
                _merge_job_progress(
                    expected_job, existing_jobs_by_id[expected_job["id"]]
                )
                if expected_job["id"] in existing_jobs_by_id
                else expected_job
            )
            for expected_job in expected_jobs
        ]
        if old_current_job_id:
            for index, job in enumerate(state["jobs"]):
                if str(job.get("id", "")) == old_current_job_id:
                    state["current_job_index"] = index
                    break
        changed = True
    if int(state.get("version", 1)) != STATE_VERSION:
        state["version"] = STATE_VERSION
        changed = True
    _advance_past_completed_jobs(state)
    return changed


def _schedule_reboot(state: dict[str, Any]) -> None:
    command = [str(part) for part in state["reboot_command"]]
    subprocess.run(command, check=True)


def run_next_job(state_path: Path) -> int:
    state = load_state(state_path)
    if _migrate_state_for_current_runner(state):
        write_state(state_path, state)
    repo = Path(state["repo_root"])
    results_root = Path(state["results_root"])
    log_path = _automation_log_path(results_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_temperature_c, baseline_source = _ensure_baseline_temperature(
        state_path, state
    )

    while True:
        _advance_past_completed_jobs(state)
        job_index = _next_incomplete_job_index(state)
        if job_index is None:
            state["status"] = "completed"
            write_state(state_path, state)
            remove_crontab_entry(state_path)
            LOGGER.info("All unattended study jobs are complete")
            return 0

        state["status"] = "running"
        state["current_job_index"] = job_index
        job = state["jobs"][job_index]
        job_log_path = _job_log_path(results_root, job_index, str(job["id"]))
        job["log_path"] = str(job_log_path)
        write_state(state_path, state)

        job_log_path.parent.mkdir(parents=True, exist_ok=True)
        with job_log_path.open("a", encoding="utf-8") as handle:
            pending_ttm_actions = [
                action
                for action in job.get("privileged_setup", [])
                if _is_ttm_action(action) and _ttm_action_requires_reboot(action, state)
            ]
            if pending_ttm_actions:
                handle.write(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Preparing TTM state for "
                    f"{job['description']} before reboot\n"
                )
                for action in pending_ttm_actions:
                    _run_privileged_action(action, log_handle=handle, state=state)
                write_state(state_path, state)
                _schedule_reboot(state)
                return 0

            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["status"] = "running"
            job["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_state(state_path, state)
            handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {job['description']} "
                f"(attempt {job['attempts']}/{job['max_attempts']})\n"
            )
            exit_code = None
            temperature_context: dict[str, Any] = {}
            try:
                temperature_context = _wait_for_temperature_gate(
                    baseline_temperature_c=baseline_temperature_c,
                    threshold_ratio=float(
                        state.get(
                            "temperature_threshold_ratio",
                            DEFAULT_TEMPERATURE_THRESHOLD_RATIO,
                        )
                    ),
                    poll_interval_seconds=float(
                        state.get(
                            "temperature_poll_interval_seconds",
                            DEFAULT_TEMPERATURE_POLL_INTERVAL_SECONDS,
                        )
                    ),
                    max_wait_seconds=float(
                        state.get(
                            "temperature_max_wait_seconds",
                            DEFAULT_TEMPERATURE_MAX_WAIT_SECONDS,
                        )
                    ),
                    log_handle=handle,
                )
                job.update(temperature_context)
                handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Temperature gate proceeding: baseline={baseline_temperature_c:.3f}C "
                    f"threshold={float(temperature_context['threshold_temperature_c']):.3f}C "
                    f"current={float(temperature_context['pre_run_temperature_c']):.3f}C "
                    f"source={temperature_context['temperature_source']} "
                    f"threshold_met={temperature_context['pre_run_threshold_met']} "
                    f"checks={temperature_context['pre_run_check_count']}\n"
                )
                for action in job.get("privileged_setup", []):
                    if _is_ttm_action(action):
                        continue
                    _run_privileged_action(action, log_handle=handle, state=state)
                argv = _job_command(
                    repo,
                    str(state["run_user"]),
                    str(job["module"]),
                    [str(value) for value in job["argv"]],
                )
                exit_code = _run_subprocess(
                    argv,
                    as_root=(os.geteuid() == 0),
                    log_handle=handle,
                )
                job["last_error"] = ""
            except Exception as exc:
                handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Runner exception: {exc}\n"
                )
                job["last_error"] = str(exc)
                exit_code = 1
            try:
                post_run_temperature_c, post_run_temperature_source = (
                    _read_pc_temperature_c()
                )
                job["post_run_temperature_c"] = post_run_temperature_c
                job["post_run_temperature_source"] = post_run_temperature_source
            except Exception as exc:
                job["post_run_temperature_c"] = ""
                job["post_run_temperature_source"] = ""
                handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Post-run temperature read failed: {exc}\n"
                )
            job["last_exit_code"] = exit_code
            job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if exit_code == 0:
                job["status"] = "passed"
                state["current_job_index"] = job_index + 1
            else:
                if not str(job.get("last_error") or "").strip():
                    job["last_error"] = f"exit_code={exit_code}"
                if int(job["attempts"]) < int(job["max_attempts"]):
                    job["status"] = "failed_retryable"
                else:
                    job["status"] = "failed"
                    state["status"] = "failed"
            write_state(state_path, state)
            _append_temperature_log_row(
                results_root,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "event": "job",
                    "run_id": state["run_id"],
                    "job_index": job_index,
                    "job_id": job["id"],
                    "job_status": job["status"],
                    "baseline_temperature_c": baseline_temperature_c,
                    "threshold_temperature_c": job.get("threshold_temperature_c", ""),
                    "pre_run_temperature_c": job.get("pre_run_temperature_c", ""),
                    "post_run_temperature_c": job.get("post_run_temperature_c", ""),
                    "pre_run_threshold_met": job.get("pre_run_threshold_met", ""),
                    "pre_run_check_count": job.get("pre_run_check_count", ""),
                    "pre_run_wait_seconds": job.get("pre_run_wait_seconds", ""),
                    "temperature_source": (
                        str(job.get("temperature_source") or "")
                        or str(job.get("post_run_temperature_source") or "")
                        or baseline_source
                    ),
                },
            )

        if str(job["status"]) == "passed":
            continue

        if str(job["status"]) == "failed_retryable":
            continue

        remove_crontab_entry(state_path)
        LOGGER.error("Job failed permanently: %s", job["description"])
        return int(job.get("last_exit_code") or 1)


def render_status(state: dict[str, Any]) -> str:
    total = len(state["jobs"])
    passed = sum(1 for job in state["jobs"] if str(job.get("status")) == "passed")
    failed = [
        job
        for job in state["jobs"]
        if str(job.get("status")) in {"failed", "failed_retryable"}
    ]
    current_index = _next_incomplete_job_index(state)
    lines = [
        f"run_id: {state['run_id']}",
        f"status: {state['status']}",
        f"results_root: {state['results_root']}",
        f"jobs: {passed}/{total} passed",
    ]
    if current_index is not None:
        job = state["jobs"][current_index]
        lines.append(
            f"next_job: {current_index + 1}/{total} {job['id']} ({job['description']})"
        )
    if failed:
        lines.append("failed_jobs:")
        for job in failed[:5]:
            lines.append(
                f"  - {job['id']}: status={job['status']} attempts={job['attempts']} error={job['last_error']}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transformer_layer_new studies unattended with reboot-aware hardware-state transitions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start",
        help="Create the unattended plan, install the boot hook, and start the first job.",
    )
    start.add_argument("--run-id", default=default_run_id())
    start.add_argument("--state", type=Path, default=None)
    start.add_argument("--results-root", type=Path, default=None)
    start.add_argument("--run-user", default=default_run_user())
    start.add_argument(
        "--host-comparison-16384-ttm-gb",
        type=int,
        default=DEFAULT_HOST_COMPARISON_16384_TTM_GB,
    )
    start.add_argument(
        "--reboot-command",
        default="sudo -n reboot",
        help="Command used to reboot after each job.",
    )
    start.add_argument("--log-level", default="INFO")

    run_next = subparsers.add_parser(
        "run-next", help="Run the next queued job once. Intended for @reboot use."
    )
    run_next.add_argument("--state", type=Path, required=True)
    run_next.add_argument("--log-level", default="INFO")

    status = subparsers.add_parser("status", help="Print unattended queue status.")
    status.add_argument("--state", type=Path, required=True)

    stop = subparsers.add_parser(
        "stop", help="Remove the boot hook and stop the unattended queue."
    )
    stop.add_argument("--state", type=Path, required=True)

    resume = subparsers.add_parser(
        "resume",
        help="Resume a previously stopped unattended queue from the next incomplete job.",
    )
    resume.add_argument("--state", type=Path, required=True)
    resume.add_argument("--log-level", default="INFO")

    smoke = subparsers.add_parser(
        "smoke-test",
        help="Run a small unattended smoke test without installing a boot hook.",
    )
    smoke.add_argument("--run-id", default=f"smoke_{default_run_id()}")
    smoke.add_argument("--state", type=Path, default=None)
    smoke.add_argument("--results-root", type=Path, default=None)
    smoke.add_argument(
        "--source-results-root",
        type=Path,
        default=None,
        help="Existing results root used as the input fixture for the smoke test",
    )
    smoke.add_argument("--run-user", default=default_run_user())
    smoke.add_argument(
        "--reboot-command",
        default="true",
        help="Command used between smoke-test jobs. Defaults to a no-op.",
    )
    smoke.add_argument("--log-level", default="INFO")

    execution_smoke = subparsers.add_parser(
        "execution-smoke-test",
        help="Run a reduced unattended execution smoke without installing a boot hook.",
    )
    execution_smoke.add_argument(
        "--run-id", default=f"execution_smoke_{default_run_id()}"
    )
    execution_smoke.add_argument("--state", type=Path, default=None)
    execution_smoke.add_argument("--results-root", type=Path, default=None)
    execution_smoke.add_argument(
        "--source-results-root",
        type=Path,
        default=None,
        help="Existing results root used as the input fixture source for ancillary outputs.",
    )
    execution_smoke.add_argument("--run-user", default=default_run_user())
    execution_smoke.add_argument(
        "--reboot-command",
        default="true",
        help="Command used between smoke-test jobs. Defaults to a no-op.",
    )
    execution_smoke.add_argument("--log-level", default="INFO")

    return parser.parse_args(argv)


def _start(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    repo = repo_root()
    run_id = str(args.run_id)
    results_root = (
        args.results_root.expanduser()
        if args.results_root is not None
        else default_results_root(run_id)
    )
    state_path = (
        args.state.expanduser()
        if args.state is not None
        else default_state_path(run_id)
    )
    reboot_command = shlex.split(str(args.reboot_command))
    if os.geteuid() == 0 and reboot_command == ["sudo", "-n", "reboot"]:
        reboot_command = ["reboot"]
    if _ttm_config_exists():
        raise RuntimeError(
            "Refusing to start unattended execution while /etc/modprobe.d/ttm.conf "
            "already exists. Clear amd-ttm first so the runner can capture the normal "
            "TTM state explicitly."
        )
    baseline_temperature_c, temperature_source = _read_pc_temperature_c()
    amd_ttm_path = _resolve_amd_ttm_path(str(args.run_user))
    if amd_ttm_path is None:
        raise FileNotFoundError(
            "amd-ttm was not found for the unattended runner. Install amd-debug-tools "
            "and ensure amd-ttm is available on PATH or at ~/.local/bin before starting."
        )
    normal_ttm_pages_limit = _current_ttm_pages_limit()
    _ensure_results_root_owned_by_run_user(results_root, str(args.run_user))
    state = create_state(
        run_id=run_id,
        repo=repo,
        results_root=results_root,
        run_user=str(args.run_user),
        host_comparison_16384_ttm_gb=int(args.host_comparison_16384_ttm_gb),
        reboot_command=reboot_command,
        baseline_temperature_c=baseline_temperature_c,
        temperature_source=temperature_source,
        amd_ttm_path=amd_ttm_path,
        normal_ttm_pages_limit=normal_ttm_pages_limit,
        plan_kind="full",
    )
    write_state(state_path, state)
    _record_baseline_temperature(results_root, state)
    install_crontab_entry(state_path)
    return run_next_job(state_path)


def _run_next(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    state_path = args.state.expanduser()
    lock_handle = None
    try:
        lock_handle = _load_lock_handle(_state_lock_path(state_path))
    except BlockingIOError:
        LOGGER.info("Another unattended runner instance is already active")
        return 0
    try:
        return run_next_job(state_path)
    finally:
        if lock_handle is not None:
            lock_handle.close()


def _status(args: argparse.Namespace) -> int:
    state = load_state(args.state.expanduser())
    print(render_status(state))
    return 0


def _stop(args: argparse.Namespace) -> int:
    state_path = args.state.expanduser()
    state = load_state(state_path)
    state["status"] = "stopped"
    write_state(state_path, state)
    remove_crontab_entry(state_path)
    LOGGER.info("Stopped unattended queue for %s", state_path)
    return 0


def _resume(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    state_path = args.state.expanduser()
    state = load_state(state_path)
    if str(state.get("status")) == "completed":
        raise RuntimeError("Cannot resume a completed unattended queue")
    _migrate_state_for_current_runner(state)
    retry_failed_current = str(state.get("status")) == "failed"
    _prepare_state_for_resume(state, retry_failed_current=retry_failed_current)
    state["status"] = "pending"
    write_state(state_path, state)
    install_crontab_entry(state_path)
    return run_next_job(state_path)


def _smoke_test(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    repo = repo_root()
    run_id = str(args.run_id)
    results_root = (
        args.results_root.expanduser()
        if args.results_root is not None
        else default_results_root(run_id)
    )
    state_path = (
        args.state.expanduser()
        if args.state is not None
        else default_state_path(run_id)
    )
    source_results_root = (
        args.source_results_root.expanduser()
        if args.source_results_root is not None
        else app_root() / "results"
    )
    reboot_command = shlex.split(str(args.reboot_command))
    jobs = build_smoke_test_job_plan(
        results_root=results_root,
        source_results_root=source_results_root,
    )
    baseline_temperature_c, temperature_source = _read_pc_temperature_c()
    state = create_state(
        run_id=run_id,
        repo=repo,
        results_root=results_root,
        run_user=str(args.run_user),
        host_comparison_16384_ttm_gb=DEFAULT_HOST_COMPARISON_16384_TTM_GB,
        reboot_command=reboot_command,
        jobs=jobs,
        baseline_temperature_c=baseline_temperature_c,
        temperature_source=temperature_source,
        plan_kind="plot_smoke",
        source_results_root=source_results_root,
    )
    write_state(state_path, state)
    _record_baseline_temperature(results_root, state)

    max_steps = len(jobs) + 2
    for _ in range(max_steps):
        exit_code = run_next_job(state_path)
        state = load_state(state_path)
        if str(state.get("status")) == "completed":
            print(render_status(state))
            return 0
        if str(state.get("status")) == "failed":
            print(render_status(state))
            return int(exit_code or 1)

    raise RuntimeError(
        "Smoke test did not converge within the expected number of steps"
    )


def _execution_smoke_test(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    repo = repo_root()
    run_id = str(args.run_id)
    results_root = (
        args.results_root.expanduser()
        if args.results_root is not None
        else default_results_root(run_id)
    )
    state_path = (
        args.state.expanduser()
        if args.state is not None
        else default_state_path(run_id)
    )
    source_results_root = (
        args.source_results_root.expanduser()
        if args.source_results_root is not None
        else app_root() / "results"
    )
    reboot_command = shlex.split(str(args.reboot_command))
    jobs = build_execution_smoke_job_plan(
        results_root=results_root,
        source_results_root=source_results_root,
    )
    baseline_temperature_c, temperature_source = _read_pc_temperature_c()
    state = create_state(
        run_id=run_id,
        repo=repo,
        results_root=results_root,
        run_user=str(args.run_user),
        host_comparison_16384_ttm_gb=DEFAULT_HOST_COMPARISON_16384_TTM_GB,
        reboot_command=reboot_command,
        jobs=jobs,
        baseline_temperature_c=baseline_temperature_c,
        temperature_source=temperature_source,
        plan_kind="execution_smoke",
        source_results_root=source_results_root,
    )
    write_state(state_path, state)
    _record_baseline_temperature(results_root, state)

    max_steps = len(jobs) + 2
    for _ in range(max_steps):
        exit_code = run_next_job(state_path)
        state = load_state(state_path)
        if str(state.get("status")) == "completed":
            print(render_status(state))
            return 0
        if str(state.get("status")) == "failed":
            print(render_status(state))
            return int(exit_code or 1)

    raise RuntimeError(
        "Execution smoke test did not converge within the expected number of steps"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "start":
        return _start(args)
    if args.command == "run-next":
        return _run_next(args)
    if args.command == "status":
        return _status(args)
    if args.command == "stop":
        return _stop(args)
    if args.command == "resume":
        return _resume(args)
    if args.command == "smoke-test":
        return _smoke_test(args)
    if args.command == "execution-smoke-test":
        return _execution_smoke_test(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
