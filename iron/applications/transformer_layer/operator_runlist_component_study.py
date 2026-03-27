#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from iron.applications.transformer_layer.benchmark_common import write_dict_rows_csv
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def load_component_study_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "study_cases" not in config or "seq_lens" not in config:
        raise KeyError("Component study config must include study_cases and seq_lens")
    if "output_csv" in config:
        raw_path = Path(str(config["output_csv"]))
        if not raw_path.is_absolute():
            config["output_csv"] = str((config_path.parent / raw_path).resolve())
    return config


def _run_component_check_isolated(
    *,
    spec: TransformerLayerSpec,
    seed: int,
    repeats: int,
    targets: str | None,
) -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    validate_script = Path(__file__).with_name("validate_operator_runlist_stability.py")
    with tempfile.TemporaryDirectory(
        prefix="transformer_layer_operator_runlist_component_"
    ) as temp_dir:
        output_csv = Path(temp_dir) / "component_rows.csv"
        command = [
            sys.executable,
            str(validate_script),
            "--seq-len",
            str(spec.seq_len),
            "--hidden-size",
            str(spec.hidden_size),
            "--intermediate-size",
            str(spec.intermediate_size),
            "--num-attention-heads",
            str(spec.num_attention_heads),
            "--input-boundary",
            spec.input_boundary,
            "--seed",
            str(seed),
            "--repeats",
            str(repeats),
            "--components-only",
            "--output-csv",
            str(output_csv),
        ]
        if targets:
            command.extend(["--targets", targets])
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or (result.stdout or "").strip()
            raise RuntimeError(
                "operator_runlist component study subprocess failed with exit code "
                f"{result.returncode}" + (f": {detail}" if detail else "")
            )
        with output_csv.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def run_component_study(config: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    targets = str(config["targets"]) if config.get("targets") else None
    repeats = int(config.get("repeats", 3))
    seed = int(config.get("seed", 0))
    for case in config["study_cases"]:
        case_id = case.get("case_id", "default")
        case_label = case.get("case_label", case_id)
        layer_spec = dict(case["layer_spec"])
        for seq_len in config["seq_lens"]:
            spec = TransformerLayerSpec.from_dict(
                {**layer_spec, "seq_len": int(seq_len)}
            )
            case_rows = _run_component_check_isolated(
                spec=spec,
                seed=seed,
                repeats=repeats,
                targets=targets,
            )
            for row in case_rows:
                row["study_case_id"] = case_id
                row["study_case_label"] = case_label
                row["hidden_size"] = spec.hidden_size
                row["intermediate_size"] = spec.intermediate_size
                row["num_attention_heads"] = spec.num_attention_heads
                row["attention_head_size"] = spec.attention_head_size
            rows.extend(case_rows)
    if config.get("output_csv"):
        write_dict_rows_csv(config["output_csv"], rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run operator_runlist component-boundary validation across a study config."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_component_study(load_component_study_config(args.config))
    for row in rows:
        print(
            f"{row['study_case_label']} seq_len={row['seq_len']} {row['target']}: "
            f"stable={row['stable']} max_abs_diff={float(row['max_abs_diff']):.6f}"
        )


if __name__ == "__main__":
    main()
