#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import torch

from iron.applications.transformer_layer.benchmark_common import parse_seq_lens
from iron.applications.transformer_layer.npu_inference import build_pattern
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_hidden_states,
    make_synthetic_layer_weights,
)


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (reference - candidate).abs().to(torch.float32)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


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
    return parser.parse_args()


def main():
    args = parse_args()
    for seq_len in parse_seq_lens(args.seq_lens):
        spec = TransformerLayerSpec(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            seq_len=seq_len,
            use_bias=False,
            weights_source="synthetic",
        )
        weights = make_synthetic_layer_weights(spec, seed=args.seed)
        hidden_states = make_synthetic_hidden_states(spec, seed=args.seed + 1)

        reference = ReferenceTransformerLayer(spec)
        reference.assign_weights(weights)
        reference_output = reference(hidden_states)

        pattern = build_pattern(args.execution_mode, spec)
        pattern.assign_weights(weights)
        candidate_output = pattern(hidden_states)
        stats = error_stats(reference_output, candidate_output)
        print(
            f"seq_len={seq_len} max_abs_diff={stats['max_abs_diff']:.6f} mean_abs_diff={stats['mean_abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
