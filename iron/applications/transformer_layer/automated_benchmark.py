#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

from iron.applications.transformer_layer.benchmark_common import (
    load_study_manifest,
    parse_execution_modes,
    parse_seq_lens,
    resolve_study_path,
    write_results_csv,
)
from iron.applications.transformer_layer.npu_inference import benchmark_pattern
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven synthetic transformer-layer NPU sweep."
    )
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument(
        "--execution-modes", default="encoder_pipeline,gemm_only,operator_runlist"
    )
    parser.add_argument("--seq-lens", default="64,128,256,512")
    parser.add_argument("--output-csv", default="transformer_layer_npu_suite.csv")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs-per-sample", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-parity-check",
        action="store_true",
        help="Run layer-level parity checks after the NPU sweep.",
    )
    parser.add_argument(
        "--parity-output-csv",
        default=None,
        help="Optional parity-summary CSV path. Defaults to the manifest parity output.",
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip manifest-configured parity validation for this run.",
    )
    return parser.parse_args()


def _resolve_spec(
    args, manifest: dict[str, object], seq_len: int
) -> TransformerLayerSpec:
    layer_spec = dict(manifest["layer_spec"])
    layer_spec.update(
        {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_attention_heads": args.num_attention_heads,
            "seq_len": seq_len,
        }
    )
    return TransformerLayerSpec.from_dict(layer_spec)


def _resolve_output_csv(args, manifest: dict[str, object]) -> str:
    if args.output_csv != "transformer_layer_npu_suite.csv":
        return args.output_csv
    return manifest.get("output_csv", args.output_csv)


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


def _load_csv_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_parity_checks(
    *,
    parity: dict[str, object],
    execution_modes: list[str],
    seq_lens: list[int],
    base_spec: TransformerLayerSpec,
    seed: int,
    study_id: str,
) -> list[dict[str, object]]:
    parity_seq_lens = list(parity.get("seq_lens", seq_lens))
    parity_execution_modes = list(parity.get("execution_modes", execution_modes))
    validate_script = Path(__file__).with_name("validate_npu_parity.py")
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="transformer_layer_parity_") as temp_dir:
        temp_root = Path(temp_dir)
        for execution_mode in parity_execution_modes:
            output_csv = temp_root / f"{execution_mode}_parity.csv"
            command = [
                sys.executable,
                str(validate_script),
                "--execution-mode",
                execution_mode,
                "--seq-lens",
                ",".join(str(seq_len) for seq_len in parity_seq_lens),
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
            subprocess.run(command, check=True)
            mode_rows = _load_csv_rows(output_csv)
            for row in mode_rows:
                row["study_id"] = study_id
            rows.extend(mode_rows)
    return rows


def main():
    args = parse_args()
    manifest = load_study_manifest(args.study_manifest)
    execution_modes = (
        parse_execution_modes(args.execution_modes)
        if args.execution_modes != "encoder_pipeline,gemm_only,operator_runlist"
        else list(manifest["execution_modes"])
    )
    seq_lens = (
        parse_seq_lens(args.seq_lens)
        if args.seq_lens != "64,128,256,512"
        else list(manifest["seq_lens"])
    )
    output_csv = _resolve_output_csv(args, manifest)
    all_rows = []
    for seq_len in seq_lens:
        spec = _resolve_spec(args, manifest, seq_len)
        for execution_mode in execution_modes:
            all_rows.extend(
                benchmark_pattern(
                    execution_mode=execution_mode,
                    spec=spec,
                    warmup_runs=args.warmup_runs,
                    runs_per_sample=args.runs_per_sample,
                    output_csv=output_csv,
                    seed=args.seed,
                    write_immediately=False,
                )
            )
    if all_rows:
        write_results_csv(output_csv, all_rows)

    parity = _resolve_parity_config(args, manifest)
    if parity is not None:
        parity_rows = _run_parity_checks(
            parity=parity,
            execution_modes=execution_modes,
            seq_lens=seq_lens,
            base_spec=_resolve_spec(args, manifest, seq_lens[0]),
            seed=args.seed,
            study_id=str(manifest["study_id"]),
        )
        parity_output = parity.get("output_csv")
        if parity_output:
            from iron.applications.transformer_layer.benchmark_common import (
                write_dict_rows_csv,
            )

            write_dict_rows_csv(parity_output, parity_rows)


if __name__ == "__main__":
    main()
