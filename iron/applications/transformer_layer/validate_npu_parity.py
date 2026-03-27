#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from iron.applications.transformer_layer.benchmark_common import parse_seq_lens
from iron.applications.transformer_layer.benchmark_common import write_dict_rows_csv
from iron.applications.transformer_layer.npu_inference import build_pattern
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (reference - candidate).abs().to(torch.float32)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _validate_operator_runlist_parity_isolated(
    *,
    spec: TransformerLayerSpec,
    seed: int,
    study_id: str,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(
        prefix="transformer_layer_operator_runlist_parity_"
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        request_path = temp_dir_path / "request.json"
        response_path = temp_dir_path / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "mode": "parity",
                    "spec": spec.to_dict(),
                    "seed": seed,
                    "study_id": study_id,
                }
            )
        )
        command = [
            sys.executable,
            "-m",
            "iron.applications.transformer_layer.operator_runlist_worker",
            "--request-json",
            str(request_path),
            "--response-json",
            str(response_path),
        ]
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "operator_runlist parity child process failed with exit code "
                f"{result.returncode}"
            )
        if not response_path.exists():
            raise RuntimeError(
                "operator_runlist parity child process did not produce a response payload"
            )
        return json.loads(response_path.read_text())


def validate_pattern_parity(
    *,
    execution_mode: str,
    spec: TransformerLayerSpec,
    seed: int,
    study_id: str = "synthetic_transformer_layer",
) -> dict[str, object]:
    if execution_mode == "operator_runlist":
        return _validate_operator_runlist_parity_isolated(
            spec=spec,
            seed=seed,
            study_id=study_id,
        )

    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)

    reference = ReferenceTransformerLayer(spec)
    reference.assign_weights(weights)
    reference_output = reference(layer_inputs)

    pattern = build_pattern(execution_mode, spec)
    pattern.assign_weights(weights)
    candidate_output = pattern(layer_inputs)
    stats = error_stats(reference_output, candidate_output)
    return {
        "study_id": study_id,
        "execution_mode": execution_mode,
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "weights_source": spec.weights_source,
        "seed": seed,
        **stats,
    }


def run_parity_suite(
    *,
    execution_modes: list[str],
    seq_lens: list[int],
    base_spec: TransformerLayerSpec,
    seed: int,
    study_id: str = "synthetic_transformer_layer",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seq_len in seq_lens:
        spec = TransformerLayerSpec.from_dict(
            {
                **base_spec.to_dict(),
                "seq_len": seq_len,
            }
        )
        for execution_mode in execution_modes:
            rows.append(
                validate_pattern_parity(
                    execution_mode=execution_mode,
                    spec=spec,
                    seed=seed,
                    study_id=study_id,
                )
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate single-layer NPU parity against the CPU reference."
    )
    parser.add_argument(
        "--execution-mode",
        choices=("encoder_pipeline", "gemm_only", "operator_runlist"),
        required=True,
    )
    parser.add_argument("--seq-lens", default="128")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_parity_suite(
        execution_modes=[args.execution_mode],
        seq_lens=parse_seq_lens(args.seq_lens),
        base_spec=TransformerLayerSpec(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            use_bias=False,
            weights_source="synthetic",
        ),
        seed=args.seed,
    )
    for row in rows:
        print(
            "seq_len="
            f"{row['seq_len']} max_abs_diff={row['max_abs_diff']:.6f} "
            f"mean_abs_diff={row['mean_abs_diff']:.6f}"
        )
    if args.output_csv:
        write_dict_rows_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
