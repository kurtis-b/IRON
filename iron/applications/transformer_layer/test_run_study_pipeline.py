import json
from pathlib import Path

from iron.applications.transformer_layer.run_study_pipeline import (
    _build_step_command,
    load_pipeline_config,
    run_study_pipeline,
)


class _Args:
    smoke = True
    skip_power_cycle = False
    warmup_runs = 0
    runs_per_sample = 1
    output_root = None
    power_backend = "turbostat_pkgwatt"
    power_sample_interval_sec = 0.05
    quiescent_baseline_duration_sec = 0.5


def test_load_pipeline_config_resolves_step_paths(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    manifest = tmp_path / "study.json"
    manifest.write_text("{}", encoding="utf-8")
    pipeline.write_text(
        json.dumps(
            {
                "pipeline_id": "full",
                "debug_log_csv": "debug.csv",
                "steps": [
                    {
                        "step_id": "study",
                        "kind": "npu_study",
                        "manifest": "study.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_pipeline_config(pipeline)

    assert loaded["debug_log_csv"] == str((tmp_path / "debug.csv").resolve())
    assert loaded["steps"][0]["manifest"] == str(manifest.resolve())


def test_build_step_command_rewrites_manifest_outputs_for_smoke(tmp_path):
    manifest = tmp_path / "study.json"
    manifest.write_text(
        json.dumps(
            {
                "study_id": "main",
                "layer_spec": {
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "batch_size": 1,
                    "seq_len": 64,
                    "dtype": "bfloat16",
                    "activation": "gelu",
                    "use_bias": False,
                    "layer_norm_eps": 1e-12,
                    "attention_mask_mode": "none",
                    "weights_source": "synthetic",
                    "source_model_name": None,
                    "source_layer_index": None,
                },
                "execution_modes": ["encoder_pipeline"],
                "seq_lens": [64, 128],
                "output_csv": "../results/main.csv",
                "annotated_output_csv": "../results/main_annotated.csv",
                "parity": {
                    "enabled": True,
                    "seq_lens": [64, 128],
                    "output_csv": "../results/main_parity.csv",
                },
            }
        ),
        encoding="utf-8",
    )
    args = _Args()
    args.output_root = str((tmp_path / "smoke_outputs").resolve())
    step = {"step_id": "study", "kind": "npu_study", "manifest": str(manifest)}

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    command = _build_step_command(
        step=step,
        args=args,
        output_root=Path(args.output_root),
        temp_dir=temp_dir,
    )

    patched_manifest = Path(command[command.index("--study-manifest") + 1])
    payload = json.loads(patched_manifest.read_text(encoding="utf-8"))
    assert payload["seq_lens"] == [64]
    assert payload["output_csv"].endswith("/smoke_outputs/results/main.csv")
    assert payload["parity"]["output_csv"].endswith(
        "/smoke_outputs/results/main_parity.csv"
    )
    assert "--power-backend" in command


def test_run_study_pipeline_skips_power_cycle_in_smoke(monkeypatch, tmp_path):
    config = {
        "pipeline_id": "full",
        "debug_log_csv": None,
        "fail_fast": True,
        "npu_power": {
            "power_backend": "turbostat_pkgwatt",
            "power_sample_interval_sec": 0.05,
            "quiescent_baseline_duration_sec": 0.5,
        },
        "power_cycle": {
            "enabled": True,
            "command": "false",
            "off_command": None,
            "on_command": None,
            "skip_in_smoke": True,
            "temperature_sensor_name": "k10temp",
            "temperature_sensor_input": "temp1_input",
            "recovery_tolerance_fraction": 0.05,
            "recovery_timeout_sec": 150.0,
            "recovery_poll_interval_sec": 1.0,
        },
        "steps": [
            {
                "step_id": "first",
                "kind": "bottleneck",
                "benchmarking_step": True,
                "input_csv": "a.csv",
                "summary_csv": "b.csv",
                "summary_json": "c.json",
                "summary_text": "d.txt",
            },
            {
                "step_id": "second",
                "kind": "plot",
                "benchmarking_step": False,
                "input_csv": "a.csv",
                "output_dir": "plots",
            },
        ],
    }
    args = _Args()

    commands = []

    def fake_subprocess_run(command, cwd=None, check=False, shell=False):
        commands.append((command, shell))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline.subprocess.run",
        fake_subprocess_run,
    )

    run_study_pipeline(config, args)

    assert len(commands) == 2
    assert all(not shell for _, shell in commands)


def test_run_study_pipeline_power_cycles_between_benchmark_steps(monkeypatch):
    config = {
        "pipeline_id": "full",
        "debug_log_csv": None,
        "fail_fast": True,
        "npu_power": {
            "power_backend": "turbostat_pkgwatt",
            "power_sample_interval_sec": 0.05,
            "quiescent_baseline_duration_sec": 0.5,
        },
        "power_cycle": {
            "enabled": True,
            "command": "power-cycle-now",
            "off_command": None,
            "on_command": None,
            "skip_in_smoke": False,
            "temperature_sensor_name": "k10temp",
            "temperature_sensor_input": "temp1_input",
            "recovery_tolerance_fraction": 0.05,
            "recovery_timeout_sec": 150.0,
            "recovery_poll_interval_sec": 0.0,
        },
        "steps": [
            {
                "step_id": "first",
                "kind": "bottleneck",
                "benchmarking_step": True,
                "input_csv": "a.csv",
                "summary_csv": "b.csv",
                "summary_json": "c.json",
                "summary_text": "d.txt",
            },
            {
                "step_id": "second",
                "kind": "bottleneck",
                "benchmarking_step": True,
                "input_csv": "e.csv",
                "summary_csv": "f.csv",
                "summary_json": "g.json",
                "summary_text": "h.txt",
            },
        ],
    }
    args = _Args()
    args.smoke = False

    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._resolve_hwmon_temperature_path",
        lambda sensor_name, sensor_input: Path("/tmp/fake_temp"),
    )
    temperatures = iter([40.0, 41.0])
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._read_temperature_c",
        lambda sensor_path: next(temperatures),
    )

    power_cycles = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._run_power_cycle_command",
        lambda command: power_cycles.append(command),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline.subprocess.run",
        lambda command, cwd=None, check=False, shell=False: type(
            "Completed", (), {"returncode": 0}
        )(),
    )

    run_study_pipeline(config, args)

    assert power_cycles == ["power-cycle-now"]


def test_run_study_pipeline_power_cycles_with_off_and_on_commands(monkeypatch):
    config = {
        "pipeline_id": "full",
        "debug_log_csv": None,
        "fail_fast": True,
        "npu_power": {
            "power_backend": "turbostat_pkgwatt",
            "power_sample_interval_sec": 0.05,
            "quiescent_baseline_duration_sec": 0.5,
        },
        "power_cycle": {
            "enabled": True,
            "command": None,
            "off_command": ["relayctl", "off"],
            "on_command": ["relayctl", "on"],
            "off_wait_sec": 0.0,
            "on_wait_sec": 0.0,
            "skip_in_smoke": False,
            "temperature_sensor_name": "k10temp",
            "temperature_sensor_input": "temp1_input",
            "recovery_tolerance_fraction": 0.05,
            "recovery_timeout_sec": 150.0,
            "recovery_poll_interval_sec": 0.0,
        },
        "steps": [
            {
                "step_id": "first",
                "kind": "bottleneck",
                "benchmarking_step": True,
                "input_csv": "a.csv",
                "summary_csv": "b.csv",
                "summary_json": "c.json",
                "summary_text": "d.txt",
            },
            {
                "step_id": "second",
                "kind": "bottleneck",
                "benchmarking_step": True,
                "input_csv": "e.csv",
                "summary_csv": "f.csv",
                "summary_json": "g.json",
                "summary_text": "h.txt",
            },
        ],
    }
    args = _Args()
    args.smoke = False

    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._resolve_hwmon_temperature_path",
        lambda sensor_name, sensor_input: Path("/tmp/fake_temp"),
    )
    temperatures = iter([40.0, 40.0])
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._read_temperature_c",
        lambda sensor_path: next(temperatures),
    )

    power_cycle_commands = []
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline._run_power_cycle_command",
        lambda command: power_cycle_commands.append(command),
    )
    monkeypatch.setattr(
        "iron.applications.transformer_layer.run_study_pipeline.subprocess.run",
        lambda command, cwd=None, check=False, shell=False: type(
            "Completed", (), {"returncode": 0}
        )(),
    )

    run_study_pipeline(config, args)

    assert power_cycle_commands == [["relayctl", "off"], ["relayctl", "on"]]
