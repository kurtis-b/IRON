#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import io
import json
import types
from pathlib import Path

from iron.applications.transformer_layer.study.unattended_reboot import (
    _cron_command,
    _current_npu_power_mode,
    _discover_default_smoke_source_results_root,
    _job_command,
    _load_lock_handle,
    _migrate_state_for_current_runner,
    _prepare_state_for_resume,
    _resolve_amd_ttm_path,
    _results_root_contains_required_files,
    _record_baseline_temperature,
    _resume,
    _run_privileged_action,
    _ttm_action_requires_reboot,
    _temperature_log_path,
    _wait_for_temperature_gate,
    _start,
    STATE_VERSION,
    build_execution_smoke_job_plan,
    build_job_plan,
    build_smoke_test_job_plan,
    create_state,
    run_next_job,
    write_state,
    render_status,
)
from iron.applications.transformer_layer.study.unattended_smoke_job import (
    REQUIRED_EXECUTION_FIXTURE_FILES,
    REQUIRED_RESULTS_FILES,
)


def test_build_job_plan_sets_ttm_only_for_host_comparison_16384():
    jobs = build_job_plan(
        results_root=Path("/tmp/results_unattended"),
        host_comparison_16384_ttm_gb=26,
    )

    decoder_16384 = [
        job
        for job in jobs
        if job["id"] == "host_comparison_igpu_gpt2_medium_1024_16384"
    ][0]
    assert decoder_16384["privileged_setup"] == [
        {"action": "set_turbo"},
        {"action": "set_ttm_gb", "value": 26},
    ]

    decoder_8192 = [
        job for job in jobs if job["id"] == "host_comparison_igpu_gpt2_medium_1024_8192"
    ][0]
    assert decoder_8192["privileged_setup"] == [
        {"action": "set_turbo"},
        {"action": "clear_ttm"},
    ]


def test_build_job_plan_uses_per_seq_latency_and_correctness_jobs():
    jobs = build_job_plan(
        results_root=Path("/tmp/results_unattended"),
        host_comparison_16384_ttm_gb=26,
    )

    correctness_job = [
        job for job in jobs if job["id"] == "correctness_gpt2_medium_1024_2048_hybrid"
    ][0]
    assert "--seq-len" in correctness_job["argv"]
    assert (
        correctness_job["argv"][correctness_job["argv"].index("--seq-len") + 1]
        == "2048"
    )

    latency_job = [
        job
        for job in jobs
        if job["id"] == "latency_variation_baseline_768_4096_runlist"
    ][0]
    assert "--seq-len" in latency_job["argv"]
    assert latency_job["argv"][latency_job["argv"].index("--seq-len") + 1] == "4096"


def test_build_job_plan_ends_with_plot_regeneration():
    jobs = build_job_plan(
        results_root=Path("/tmp/results_unattended"),
        host_comparison_16384_ttm_gb=26,
    )

    assert jobs[-1]["id"] == "regenerate_plots_all"
    assert jobs[-1]["module"] == (
        "iron.applications.transformer_layer.study.regenerate_plots"
    )
    assert jobs[-1]["argv"] == [
        "--results-root",
        "/tmp/results_unattended",
    ]


def test_build_job_plan_places_16384_host_comparison_jobs_at_tail():
    jobs = build_job_plan(
        results_root=Path("/tmp/results_unattended"),
        host_comparison_16384_ttm_gb=26,
    )

    fairness_index = next(
        index
        for index, job in enumerate(jobs)
        if job["id"] == "host_comparison_fairness"
    )
    regenerate_index = next(
        index for index, job in enumerate(jobs) if job["id"] == "regenerate_plots_all"
    )
    roofline_index = next(
        index for index, job in enumerate(jobs) if job["id"] == "roofline_all"
    )
    tinybert_16384_index = next(
        index
        for index, job in enumerate(jobs)
        if job["id"] == "host_comparison_igpu_tinybert_512_16384"
    )

    assert roofline_index < tinybert_16384_index < fairness_index < regenerate_index


def test_build_smoke_test_job_plan_is_small_and_self_verifying():
    jobs = build_smoke_test_job_plan(
        results_root=Path("/tmp/results_unattended_smoke"),
        source_results_root=Path("/tmp/results_fixture"),
    )

    assert [job["id"] for job in jobs] == [
        "smoke_prepare_results",
        "smoke_regenerate_plots",
        "smoke_verify_results",
    ]
    assert jobs[1]["module"] == (
        "iron.applications.transformer_layer.study.regenerate_plots"
    )
    assert jobs[2]["argv"] == [
        "verify-results",
        "--results-root",
        "/tmp/results_unattended_smoke",
    ]


def test_build_execution_smoke_job_plan_covers_three_patterns_and_exports():
    jobs = build_execution_smoke_job_plan(
        results_root=Path("/tmp/results_unattended_exec_smoke"),
        source_results_root=Path("/tmp/results_fixture"),
    )

    assert jobs[0]["id"] == "execution_smoke_prepare_fixtures"
    assert jobs[0]["argv"] == [
        "prepare-execution-fixtures",
        "--source-root",
        "/tmp/results_fixture",
        "--target-root",
        "/tmp/results_unattended_exec_smoke",
    ]
    assert [
        job["id"]
        for job in jobs
        if str(job["id"])
        in {
            "execution_smoke_end_to_end_hybrid",
            "execution_smoke_end_to_end_runlist",
            "execution_smoke_end_to_end_offload",
        }
    ] == [
        "execution_smoke_end_to_end_hybrid",
        "execution_smoke_end_to_end_runlist",
        "execution_smoke_end_to_end_offload",
    ]
    assert [
        job["id"]
        for job in jobs
        if str(job["id"])
        in {
            "execution_smoke_correctness_hybrid",
            "execution_smoke_correctness_runlist",
            "execution_smoke_correctness_offload",
        }
    ] == [
        "execution_smoke_correctness_hybrid",
        "execution_smoke_correctness_runlist",
        "execution_smoke_correctness_offload",
    ]
    assert [
        job["id"]
        for job in jobs
        if str(job["id"])
        in {
            "execution_smoke_latency_variation_hybrid",
            "execution_smoke_latency_variation_runlist",
            "execution_smoke_latency_variation_offload",
        }
    ] == [
        "execution_smoke_latency_variation_hybrid",
        "execution_smoke_latency_variation_runlist",
        "execution_smoke_latency_variation_offload",
    ]
    assert any(job["id"] == "execution_smoke_host_comparison" for job in jobs)
    assert any(job["id"] == "execution_smoke_resource_usage" for job in jobs)
    assert any(job["id"] == "execution_smoke_roofline" for job in jobs)
    assert jobs[-1]["id"] == "execution_smoke_verify_results"
    assert jobs[-1]["argv"] == [
        "verify-execution-results",
        "--results-root",
        "/tmp/results_unattended_exec_smoke",
    ]


def test_results_root_contains_required_files(tmp_path):
    results_root = tmp_path / "results"
    (results_root / "block").mkdir(parents=True)
    (results_root / "end_to_end").mkdir(parents=True)
    (results_root / "memory_tile_staging").mkdir(parents=True)
    (results_root / "host_comparison").mkdir(parents=True)
    (results_root / "block" / "results.csv").write_text("x\n", encoding="utf-8")
    (results_root / "end_to_end" / "results_all_power.csv").write_text(
        "x\n", encoding="utf-8"
    )
    (results_root / "end_to_end" / "tuning_all_power.csv").write_text(
        "x\n", encoding="utf-8"
    )
    (results_root / "memory_tile_staging" / "results.csv").write_text(
        "x\n", encoding="utf-8"
    )
    (results_root / "host_comparison" / "results.csv").write_text(
        "x\n", encoding="utf-8"
    )

    assert _results_root_contains_required_files(results_root, REQUIRED_RESULTS_FILES)
    assert not _results_root_contains_required_files(
        results_root, REQUIRED_EXECUTION_FIXTURE_FILES
    )


def test_discover_default_smoke_source_results_root_prefers_valid_unattended_root(
    monkeypatch, tmp_path
):
    tracked_results = tmp_path / "results"
    tracked_results.mkdir()

    invalid_unattended = tmp_path / "results_unattended_incomplete"
    (invalid_unattended / "block").mkdir(parents=True)
    (invalid_unattended / "block" / "results.csv").write_text("x\n", encoding="utf-8")

    valid_unattended = tmp_path / "results_unattended_valid"
    for rel_path in REQUIRED_EXECUTION_FIXTURE_FILES:
        target = valid_unattended.joinpath(*rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.app_root",
        lambda: tmp_path,
    )

    assert (
        _discover_default_smoke_source_results_root(REQUIRED_EXECUTION_FIXTURE_FILES)
        == valid_unattended
    )


def test_render_status_reports_progress_and_failures():
    state = create_state(
        run_id="test",
        repo=Path("/repo"),
        results_root=Path("/results"),
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["sudo", "-n", "reboot"],
    )
    state["jobs"][0]["status"] = "passed"
    state["jobs"][1]["status"] = "failed_retryable"
    state["jobs"][1]["attempts"] = 1
    state["jobs"][1]["last_error"] = "exit_code=1"
    state["current_job_index"] = 1
    state["status"] = "running"

    text = render_status(state)

    assert "jobs: 1/" in text
    assert "next_job: 2/" in text
    assert "failed_jobs:" in text
    assert "exit_code=1" in text


def test_wait_for_temperature_gate_stops_when_threshold_is_met(monkeypatch):
    readings = iter(
        [
            (60.0, "sensors:k10temp-pci-00c3"),
            (52.0, "sensors:k10temp-pci-00c3"),
        ]
    )
    sleeps: list[float] = []

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: next(readings),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = _wait_for_temperature_gate(
        baseline_temperature_c=50.0,
        threshold_ratio=1.05,
        poll_interval_seconds=1.0,
        max_wait_seconds=5.0,
        log_handle=io.StringIO(),
    )

    assert result["pre_run_threshold_met"] is True
    assert result["pre_run_temperature_c"] == 52.0
    assert result["pre_run_check_count"] == 2
    assert result["pre_run_wait_seconds"] == 1.0
    assert sleeps == [1.0]


def test_wait_for_temperature_gate_proceeds_after_timeout(monkeypatch):
    readings = iter([(60.0, "sensors:k10temp-pci-00c3")] * 6)
    sleeps: list[float] = []

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: next(readings),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = _wait_for_temperature_gate(
        baseline_temperature_c=50.0,
        threshold_ratio=1.05,
        poll_interval_seconds=1.0,
        max_wait_seconds=5.0,
        log_handle=io.StringIO(),
    )

    assert result["pre_run_threshold_met"] is False
    assert result["pre_run_temperature_c"] == 60.0
    assert result["pre_run_check_count"] == 6
    assert result["pre_run_wait_seconds"] == 5.0
    assert sleeps == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_record_baseline_temperature_writes_temperature_log(tmp_path):
    state = create_state(
        run_id="test",
        repo=Path("/repo"),
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["true"],
        baseline_temperature_c=48.5,
        temperature_source="sensors:k10temp-pci-00c3",
    )

    _record_baseline_temperature(Path(state["results_root"]), state)

    log_path = _temperature_log_path(Path(state["results_root"]))
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["event"] == "baseline"
    assert rows[0]["run_id"] == "test"
    assert rows[0]["baseline_temperature_c"] == "48.5"
    assert rows[0]["temperature_source"] == "sensors:k10temp-pci-00c3"


def test_job_command_uses_sourced_environment_python():
    command = _job_command(
        Path("/tmp/repo"),
        "runner",
        "example.module",
        ["--flag", "value"],
    )

    assert command == [
        "bash",
        "-lc",
        "source /opt/xilinx/xrt/setup.sh && source /tmp/repo/ironenv/bin/activate && cd /tmp/repo && python3 -m example.module --flag value",
    ]


def test_cron_command_uses_sourced_environment_python(tmp_path):
    repo = tmp_path / "repo"
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=repo,
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    write_state(state_path, state)

    command = _cron_command(state_path)

    assert "@reboot bash -lc " in command
    assert "source /opt/xilinx/xrt/setup.sh" in command
    assert "source /home" not in command
    assert f"source {repo}/ironenv/bin/activate" in command
    assert (
        "python3 -m iron.applications.transformer_layer.study.unattended_reboot run-next"
        in command
    )
    assert "/usr/bin/python3" not in command


def test_resolve_amd_ttm_path_uses_run_user_local_bin(monkeypatch, tmp_path):
    user_home = tmp_path / "runner-home"
    local_bin = user_home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    amd_ttm = local_bin / "amd-ttm"
    amd_ttm.write_text("#!/bin/sh\n", encoding="utf-8")
    amd_ttm.chmod(0o755)

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.pwd.getpwnam",
        lambda name: types.SimpleNamespace(pw_dir=str(user_home)),
    )

    assert _resolve_amd_ttm_path("runner") == str(amd_ttm)


def test_start_persists_resolved_amd_ttm_path(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    results_root = tmp_path / "results"
    created: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._resolve_amd_ttm_path",
        lambda run_user: "/tmp/fake-amd-ttm",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: 3993375,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.install_crontab_entry",
        lambda path: created.setdefault("installed", str(path)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.run_next_job",
        lambda path: 0,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._record_baseline_temperature",
        lambda results_root, state: created.setdefault("baseline_logged", True),
    )

    args = types.SimpleNamespace(
        log_level="INFO",
        run_id="test",
        results_root=results_root,
        state=state_path,
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command="true",
    )

    assert _start(args) == 0
    saved = Path(state_path).read_text(encoding="utf-8")
    assert '"amd_ttm_path": "/tmp/fake-amd-ttm"' in saved
    assert created["installed"] == str(state_path)
    assert created["baseline_logged"] is True


def test_start_rejects_unreadable_current_ttm_pages_limit(monkeypatch, tmp_path):
    args = types.SimpleNamespace(
        log_level="INFO",
        run_id="test",
        results_root=tmp_path / "results",
        state=tmp_path / "state.json",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command="true",
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._resolve_amd_ttm_path",
        lambda run_user: "/tmp/fake-amd-ttm",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: None,
    )

    try:
        _start(args)
    except RuntimeError as exc:
        assert "Unable to read the current TTM pages limit" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_clear_ttm_is_noop_when_ttm_config_is_absent(monkeypatch):
    invoked: list[list[str]] = []

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        lambda argv, **kwargs: invoked.append(list(argv)) or 0,
    )

    _run_privileged_action(
        {"action": "clear_ttm"},
        log_handle=io.StringIO(),
        state={"amd_ttm_path": "/tmp/fake-amd-ttm"},
    )

    assert invoked == []


def test_current_npu_power_mode_parses_xrt_smi_output(monkeypatch):
    recorded: dict[str, object] = {}

    def _fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        return types.SimpleNamespace(
            returncode=0,
            stdout="Platform\n  Power Mode             : Turbo \n",
        )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._resolve_xrt_smi_path",
        lambda: "/opt/xilinx/xrt/bin/xrt-smi",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.subprocess.run",
        _fake_run,
    )

    assert _current_npu_power_mode() == "Turbo"
    assert recorded["argv"] == ["/opt/xilinx/xrt/bin/xrt-smi", "examine", "-r", "all"]


def test_set_turbo_is_noop_when_device_already_in_turbo(monkeypatch):
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_npu_power_mode",
        lambda: "Turbo",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("set_turbo should not invoke xrt-smi when already turbo")
        ),
    )

    log_handle = io.StringIO()
    _run_privileged_action(
        {"action": "set_turbo"},
        log_handle=log_handle,
        state={},
    )

    assert "already in Turbo mode" in log_handle.getvalue()


def test_set_turbo_uses_resolved_xrt_smi_path(monkeypatch):
    recorded: dict[str, object] = {}

    def _fake_run_subprocess(argv, **kwargs):
        recorded["argv"] = list(argv)
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_npu_power_mode",
        lambda: None,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._resolve_xrt_smi_path",
        lambda: "/opt/xilinx/xrt/bin/xrt-smi",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        _fake_run_subprocess,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.geteuid",
        lambda: 1000,
    )

    _run_privileged_action(
        {"action": "set_turbo"},
        log_handle=io.StringIO(),
        state={},
    )

    assert recorded["argv"] == [
        "sudo",
        "-n",
        "/opt/xilinx/xrt/bin/xrt-smi",
        "configure",
        "--pmode",
        "turbo",
    ]


def test_set_ttm_declines_immediate_reboot_prompt(monkeypatch):
    recorded: dict[str, object] = {}

    def _fake_run_subprocess(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["input_text"] = kwargs.get("input_text")
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        _fake_run_subprocess,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._amd_ttm_path_from_state",
        lambda state: "/tmp/fake-amd-ttm",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.geteuid",
        lambda: 0,
    )

    _run_privileged_action(
        {"action": "set_ttm_gb", "value": 26},
        log_handle=io.StringIO(),
        state={"amd_ttm_path": "/tmp/fake-amd-ttm"},
    )

    assert recorded["argv"] == ["/tmp/fake-amd-ttm", "--set", "26"]
    assert recorded["input_text"] == "n\n"


def test_set_ttm_reports_passwordless_sudo_requirement(monkeypatch):
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._amd_ttm_path_from_state",
        lambda state: "/tmp/fake-amd-ttm",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.geteuid",
        lambda: 1000,
    )

    try:
        _run_privileged_action(
            {"action": "set_ttm_gb", "value": 26},
            log_handle=io.StringIO(),
            state={"amd_ttm_path": "/tmp/fake-amd-ttm"},
        )
    except RuntimeError as exc:
        assert "passwordless sudo is required" in str(exc)
        assert "set_ttm_gb" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_clear_ttm_reboot_check_allows_one_page_normal_drift(monkeypatch):
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: 3993373,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: False,
    )

    assert (
        _ttm_action_requires_reboot(
            {"action": "clear_ttm"},
            {"normal_ttm_pages_limit": 3993374},
        )
        is False
    )


def test_clear_ttm_reboot_check_requires_reboot_for_large_normal_drift(monkeypatch):
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: 6815744,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: False,
    )

    assert (
        _ttm_action_requires_reboot(
            {"action": "clear_ttm"},
            {"normal_ttm_pages_limit": 3993374},
        )
        is True
    )


def test_load_lock_handle_reassigns_root_created_lock_to_state_owner(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / "state.lock"
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.geteuid",
        lambda: 0,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.fchown",
        lambda fd, uid, gid: recorded.setdefault("fchown", (uid, gid)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.os.fchmod",
        lambda fd, mode: recorded.setdefault("fchmod", mode),
    )

    handle = _load_lock_handle(lock_path, owner_path=state_path)
    try:
        owner = state_path.stat()
        assert recorded["fchown"] == (owner.st_uid, owner.st_gid)
        assert recorded["fchmod"] == 0o664
    finally:
        handle.close()


def test_run_next_job_reboots_before_running_when_ttm_change_is_needed(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=[
            {
                "id": "job_with_ttm",
                "description": "job with ttm",
                "module": "example.module",
                "argv": [],
                "privileged_setup": [{"action": "set_ttm_gb", "value": 26}],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            }
        ],
        normal_ttm_pages_limit=3993373,
    )
    write_state(state_path, state)

    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ensure_baseline_temperature",
        lambda state_path, state: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._migrate_state_for_current_runner",
        lambda state: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_action_requires_reboot",
        lambda action, state: True,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_privileged_action",
        lambda action, **kwargs: recorded.setdefault("action", action),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._schedule_reboot",
        lambda state: recorded.setdefault("reboot", True),
    )

    assert run_next_job(state_path) == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "running"
    assert saved_state["jobs"][0]["status"] == "pending"
    assert saved_state["jobs"][0]["attempts"] == 0
    assert saved_state["jobs"][0]["started_at"] == ""
    assert recorded["action"] == {"action": "set_ttm_gb", "value": 26}
    assert recorded["reboot"] is True


def test_run_next_job_stops_on_repeated_ttm_reboot_boundary(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=[
            {
                "id": "job_with_ttm",
                "description": "job with ttm",
                "module": "example.module",
                "argv": [],
                "privileged_setup": [{"action": "set_ttm_gb", "value": 26}],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            }
        ],
        normal_ttm_pages_limit=3993373,
    )
    state["pending_reboot_job_id"] = "job_with_ttm"
    state["pending_reboot_actions_json"] = '[{"action": "set_ttm_gb", "value": 26}]'
    write_state(state_path, state)

    removed: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ensure_baseline_temperature",
        lambda state_path, state: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._migrate_state_for_current_runner",
        lambda state: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_action_requires_reboot",
        lambda action, state: True,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_privileged_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "should not rerun privileged action when reboot loop is detected"
            )
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._schedule_reboot",
        lambda state: (_ for _ in ()).throw(
            AssertionError(
                "should not schedule another reboot when reboot loop is detected"
            )
        ),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.remove_crontab_entry",
        lambda path: removed.setdefault("path", str(path)),
    )

    assert run_next_job(state_path) == 1
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "failed"
    assert saved_state["jobs"][0]["status"] == "failed"
    assert "avoid a reboot loop" in saved_state["jobs"][0]["last_error"]
    assert removed["path"] == str(state_path)


def test_run_next_job_continues_without_reboot_for_normal_jobs(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=[
            {
                "id": "job_a",
                "description": "job a",
                "module": "example.module",
                "argv": ["--first"],
                "privileged_setup": [{"action": "set_turbo"}, {"action": "clear_ttm"}],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            },
            {
                "id": "job_b",
                "description": "job b",
                "module": "example.module",
                "argv": ["--second"],
                "privileged_setup": [{"action": "set_turbo"}, {"action": "clear_ttm"}],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            },
        ],
        normal_ttm_pages_limit=3993373,
    )
    write_state(state_path, state)

    executed: list[list[str]] = []
    privileged_actions: list[dict[str, object]] = []
    removed: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ensure_baseline_temperature",
        lambda state_path, state: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._migrate_state_for_current_runner",
        lambda state: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_action_requires_reboot",
        lambda action, state: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._wait_for_temperature_gate",
        lambda **kwargs: {
            "temperature_source": "sensors:k10temp-pci-00c3",
            "threshold_temperature_c": 52.5,
            "pre_run_temperature_c": 50.0,
            "pre_run_threshold_met": True,
            "pre_run_check_count": 1,
            "pre_run_wait_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_privileged_action",
        lambda action, **kwargs: privileged_actions.append(dict(action)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._job_command",
        lambda repo, run_user, module, argv: ["bash", "-lc", "true", *argv],
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        lambda argv, **kwargs: executed.append(list(argv)) or 0,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: (51.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._schedule_reboot",
        lambda state: (_ for _ in ()).throw(AssertionError("unexpected reboot")),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.remove_crontab_entry",
        lambda path: removed.setdefault("path", str(path)),
    )

    assert run_next_job(state_path) == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "completed"
    assert [job["status"] for job in saved_state["jobs"]] == ["passed", "passed"]
    assert len(executed) == 2
    assert privileged_actions == [
        {"action": "set_turbo"},
        {"action": "set_turbo"},
    ]
    assert removed["path"] == str(state_path)


def test_run_next_job_reboots_at_ttm_boundary_after_normal_job(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=[
            {
                "id": "job_normal",
                "description": "job normal",
                "module": "example.module",
                "argv": ["--normal"],
                "privileged_setup": [{"action": "set_turbo"}, {"action": "clear_ttm"}],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            },
            {
                "id": "job_ttm",
                "description": "job ttm",
                "module": "example.module",
                "argv": ["--ttm"],
                "privileged_setup": [
                    {"action": "set_turbo"},
                    {"action": "set_ttm_gb", "value": 26},
                ],
                "max_attempts": 2,
                "attempts": 0,
                "status": "pending",
                "log_path": "",
                "last_exit_code": None,
                "last_error": "",
                "started_at": "",
                "finished_at": "",
            },
        ],
        normal_ttm_pages_limit=3993373,
    )
    write_state(state_path, state)

    recorded: dict[str, object] = {"privileged": []}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ensure_baseline_temperature",
        lambda state_path, state: (50.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._migrate_state_for_current_runner",
        lambda state: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._wait_for_temperature_gate",
        lambda **kwargs: {
            "temperature_source": "sensors:k10temp-pci-00c3",
            "threshold_temperature_c": 52.5,
            "pre_run_temperature_c": 50.0,
            "pre_run_threshold_met": True,
            "pre_run_check_count": 1,
            "pre_run_wait_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_action_requires_reboot",
        lambda action, state: str(action.get("action")) == "set_ttm_gb",
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_privileged_action",
        lambda action, **kwargs: recorded["privileged"].append(dict(action)),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._job_command",
        lambda repo, run_user, module, argv: ["bash", "-lc", "true", *argv],
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._run_subprocess",
        lambda argv, **kwargs: 0,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._read_pc_temperature_c",
        lambda: (51.0, "sensors:k10temp-pci-00c3"),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._schedule_reboot",
        lambda state: recorded.setdefault("reboot", True),
    )

    assert run_next_job(state_path) == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "running"
    assert saved_state["jobs"][0]["status"] == "passed"
    assert saved_state["jobs"][1]["status"] == "pending"
    assert saved_state["jobs"][1]["attempts"] == 0
    assert recorded["privileged"] == [
        {"action": "set_turbo"},
        {"action": "set_ttm_gb", "value": 26},
    ]
    assert recorded["reboot"] is True


def test_migrate_state_for_current_runner_reorders_pending_suffix(
    monkeypatch, tmp_path
):
    results_root = tmp_path / "results"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=results_root,
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    expected_jobs = list(state["jobs"])
    prefix = [
        job
        for job in expected_jobs
        if not str(job["id"]).startswith("host_comparison_igpu_")
        and str(job["id"]) != "host_comparison_fairness"
        and str(job["id"]) != "regenerate_plots_all"
    ]
    host_jobs_by_id = {
        str(job["id"]): job
        for job in expected_jobs
        if str(job["id"]).startswith("host_comparison_igpu_")
    }
    fairness_job = [
        job for job in expected_jobs if job["id"] == "host_comparison_fairness"
    ][0]
    regenerate_job = [
        job for job in expected_jobs if job["id"] == "regenerate_plots_all"
    ][0]

    legacy_host_jobs: list[dict[str, object]] = []
    for family_id in [
        "tinybert_512",
        "baseline_768",
        "baseline_1024",
        "gpt2_small_768",
        "gpt2_medium_1024",
    ]:
        for seq_len in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            legacy_host_jobs.append(
                host_jobs_by_id[f"host_comparison_igpu_{family_id}_{seq_len}"]
            )

    state["version"] = 1
    state["jobs"] = prefix + legacy_host_jobs + [fairness_job, regenerate_job]
    state["jobs"] = [dict(job) for job in state["jobs"]]
    state["baseline_ttm_pages_limit"] = 6815744
    state.pop("normal_ttm_pages_limit", None)
    state.pop("plan_layout", None)

    baseline_job_id = "host_comparison_igpu_baseline_768_64"
    baseline_index = next(
        index for index, job in enumerate(state["jobs"]) if job["id"] == baseline_job_id
    )
    for index, job in enumerate(state["jobs"]):
        if index < baseline_index:
            job["status"] = "passed"
            job["attempts"] = 1
    tinybert_16384 = next(
        job
        for job in state["jobs"]
        if job["id"] == "host_comparison_igpu_tinybert_512_16384"
    )
    tinybert_16384["status"] = "passed"
    tinybert_16384["attempts"] = 3
    state["current_job_index"] = baseline_index

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: 3993375,
    )

    assert _migrate_state_for_current_runner(state) is True
    assert state["version"] == STATE_VERSION
    assert state["normal_ttm_pages_limit"] == 6815744
    assert state["jobs"][state["current_job_index"]]["id"] == baseline_job_id

    baseline_16384_index = next(
        index
        for index, job in enumerate(state["jobs"])
        if job["id"] == "host_comparison_igpu_baseline_768_16384"
    )
    fairness_index = next(
        index
        for index, job in enumerate(state["jobs"])
        if job["id"] == "host_comparison_fairness"
    )
    roofline_index = next(
        index for index, job in enumerate(state["jobs"]) if job["id"] == "roofline_all"
    )
    regenerate_index = next(
        index
        for index, job in enumerate(state["jobs"])
        if job["id"] == "regenerate_plots_all"
    )
    tinybert_16384_index = next(
        index
        for index, job in enumerate(state["jobs"])
        if job["id"] == "host_comparison_igpu_tinybert_512_16384"
    )
    assert fairness_index < roofline_index < baseline_16384_index < regenerate_index
    assert state["jobs"][tinybert_16384_index]["status"] == "passed"
    assert state["jobs"][tinybert_16384_index]["attempts"] == 3


def test_migrate_state_for_current_runner_rejects_missing_normal_ttm_state():
    state = create_state(
        run_id="test",
        repo=Path("/repo"),
        results_root=Path("/results"),
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    state.pop("normal_ttm_pages_limit", None)
    state.pop("baseline_ttm_pages_limit", None)

    try:
        _migrate_state_for_current_runner(state)
    except RuntimeError as exc:
        assert "Unable to determine the normal TTM pages limit" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_migrate_state_for_current_runner_refreshes_stale_execution_smoke_job_spec(
    monkeypatch, tmp_path
):
    results_root = tmp_path / "results"
    source_results_root = tmp_path / "source_results"
    jobs = build_execution_smoke_job_plan(
        results_root=results_root,
        source_results_root=source_results_root,
    )
    state = create_state(
        run_id="execution_smoke_test",
        repo=tmp_path / "repo",
        results_root=results_root,
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=jobs,
        plan_kind="execution_smoke",
        source_results_root=source_results_root,
    )
    state["normal_ttm_pages_limit"] = 3993375
    roofline_job = next(
        job for job in state["jobs"] if job["id"] == "execution_smoke_roofline"
    )
    roofline_job["argv"] = ["--family", "baseline_768", "--seq-len", "512"]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._current_ttm_pages_limit",
        lambda: 3993375,
    )

    assert _migrate_state_for_current_runner(state) is True
    migrated_roofline_job = next(
        job for job in state["jobs"] if job["id"] == "execution_smoke_roofline"
    )
    assert "--seq-len" not in migrated_roofline_job["argv"]


def test_migrate_state_for_current_runner_allows_execution_smoke_without_ttm_state(
    tmp_path,
):
    results_root = tmp_path / "results"
    source_results_root = tmp_path / "source_results"
    jobs = build_execution_smoke_job_plan(
        results_root=results_root,
        source_results_root=source_results_root,
    )
    state = create_state(
        run_id="execution_smoke_test",
        repo=tmp_path / "repo",
        results_root=results_root,
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
        jobs=jobs,
        plan_kind="execution_smoke",
        source_results_root=source_results_root,
    )
    state.pop("normal_ttm_pages_limit", None)
    state.pop("baseline_ttm_pages_limit", None)

    assert _migrate_state_for_current_runner(state) is False


def test_start_rejects_existing_ttm_config(monkeypatch, tmp_path):
    args = types.SimpleNamespace(
        log_level="INFO",
        run_id="test",
        results_root=tmp_path / "results",
        state=tmp_path / "state.json",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command="true",
    )

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot._ttm_config_exists",
        lambda: True,
    )

    try:
        _start(args)
    except RuntimeError as exc:
        assert "ttm.conf" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_prepare_state_for_resume_resets_running_job_to_pending():
    state = create_state(
        run_id="test",
        repo=Path("/repo"),
        results_root=Path("/results"),
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    state["jobs"][0]["status"] = "passed"
    state["jobs"][1]["status"] = "running"
    state["jobs"][1]["attempts"] = 1
    state["jobs"][1]["started_at"] = "2026-04-14T20:15:07"
    state["jobs"][1]["last_exit_code"] = 9
    state["jobs"][1]["last_error"] = "interrupted"
    state["current_job_index"] = 1

    _prepare_state_for_resume(state)

    assert state["current_job_index"] == 1
    assert state["jobs"][1]["status"] == "pending"
    assert state["jobs"][1]["attempts"] == 1
    assert state["jobs"][1]["started_at"] == ""
    assert state["jobs"][1]["finished_at"] == ""
    assert state["jobs"][1]["last_exit_code"] is None
    assert state["jobs"][1]["last_error"] == ""


def test_resume_reinstalls_cron_and_requeues_running_job(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    state["status"] = "stopped"
    state["jobs"][0]["status"] = "passed"
    state["jobs"][1]["status"] = "running"
    state["jobs"][1]["attempts"] = 1
    state["jobs"][1]["started_at"] = "2026-04-14T20:15:07"
    state["current_job_index"] = 1
    state["normal_ttm_pages_limit"] = 3993375
    write_state(state_path, state)

    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.install_crontab_entry",
        lambda path: recorded.setdefault("installed", str(path)),
    )

    def _fake_run_next(path):
        recorded["ran"] = str(path)
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.run_next_job",
        _fake_run_next,
    )

    args = types.SimpleNamespace(
        state=state_path,
        log_level="INFO",
    )

    assert _resume(args) == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "pending"
    assert saved_state["current_job_index"] == 1
    assert saved_state["jobs"][1]["status"] == "pending"
    assert saved_state["jobs"][1]["attempts"] == 1
    assert recorded["installed"] == str(state_path)
    assert recorded["ran"] == str(state_path)


def test_resume_retries_current_failed_job(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    state["status"] = "failed"
    state["jobs"][0]["status"] = "passed"
    state["jobs"][1]["status"] = "failed"
    state["jobs"][1]["attempts"] = 2
    state["jobs"][1]["started_at"] = "2026-04-15T03:36:35"
    state["jobs"][1]["finished_at"] = "2026-04-15T03:36:37"
    state["jobs"][1]["last_exit_code"] = 1
    state["jobs"][1]["last_error"] = "Privileged setup failed for set_ttm_gb"
    state["current_job_index"] = 1
    state["normal_ttm_pages_limit"] = 3993375
    write_state(state_path, state)

    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.install_crontab_entry",
        lambda path: recorded.setdefault("installed", str(path)),
    )

    def _fake_run_next(path):
        recorded["ran"] = str(path)
        return 0

    monkeypatch.setattr(
        "iron.applications.transformer_layer.study.unattended_reboot.run_next_job",
        _fake_run_next,
    )

    args = types.SimpleNamespace(
        state=state_path,
        log_level="INFO",
    )

    assert _resume(args) == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "pending"
    assert saved_state["current_job_index"] == 1
    assert saved_state["jobs"][1]["status"] == "pending"
    assert saved_state["jobs"][1]["attempts"] == 2
    assert saved_state["jobs"][1]["started_at"] == ""
    assert saved_state["jobs"][1]["finished_at"] == ""
    assert saved_state["jobs"][1]["last_exit_code"] is None
    assert saved_state["jobs"][1]["last_error"] == ""
    assert recorded["installed"] == str(state_path)
    assert recorded["ran"] == str(state_path)


def test_resume_rejects_completed_state(tmp_path):
    state_path = tmp_path / "state.json"
    state = create_state(
        run_id="test",
        repo=tmp_path / "repo",
        results_root=tmp_path / "results",
        run_user="runner",
        host_comparison_16384_ttm_gb=26,
        reboot_command=["reboot"],
    )
    state["status"] = "completed"
    write_state(state_path, state)

    args = types.SimpleNamespace(
        state=state_path,
        log_level="INFO",
    )

    try:
        _resume(args)
    except RuntimeError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
