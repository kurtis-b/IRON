#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import torch

from iron.applications.transformer_layer.benchmark_common import write_dict_rows_csv
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.pattern_operator_runlist import (
    OperatorRunlistPattern,
)
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)
from iron.common import AIEContext
from iron.operators.encoder_runlist.op import AIEEncoderRunlist
from iron.applications.transformer_layer.validate_npu_parity import error_stats


def _parse_target_names(raw_targets: str | None) -> tuple[str, ...]:
    if raw_targets is None or raw_targets.strip() == "":
        return AIEEncoderRunlist.component_order
    names = tuple(target.strip() for target in raw_targets.split(",") if target.strip())
    return AIEEncoderRunlist._normalize_component_names(names)


def _run_component_once(
    *,
    name: str,
    operator: AIEEncoderRunlist,
    component_op: object,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    output = component_op(*args)
    if name == "k_transpose":
        return output.reshape(operator.hidden_size, operator.seq_len)
    return output


def _run_single_check(
    *,
    name: str,
    repeats: int,
    expected: torch.Tensor,
    fn: callable,
) -> dict[str, object]:
    first_output = None
    stable = True
    worst_max_abs = 0.0
    worst_mean_abs = 0.0
    for _ in range(repeats):
        output = fn()
        stats = error_stats(expected, output)
        worst_max_abs = max(worst_max_abs, stats["max_abs_diff"])
        worst_mean_abs = max(worst_mean_abs, stats["mean_abs_diff"])
        if first_output is None:
            first_output = output.detach().clone()
        else:
            stable = stable and torch.equal(first_output, output)
    return {
        "target": name,
        "repeats": repeats,
        "stable": stable,
        "max_abs_diff": worst_max_abs,
        "mean_abs_diff": worst_mean_abs,
    }


def _component_context_row(
    *,
    spec: TransformerLayerSpec,
    weights: dict[str, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    name: str,
    repeats: int,
) -> dict[str, object]:
    component_context = AIEContext(use_runlist=True)
    component_operator = AIEEncoderRunlist(
        seq_len=spec.seq_len,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_heads=spec.num_attention_heads,
        context=component_context,
        skip_add_to_list=True,
    )
    component_operator.assign_weights(
        w_o=weights["out_proj_weight"],
        b_up=weights["ffn_up_weight"],
        b_down=weights["ffn_down_weight"],
        ln1_weight=weights["ln1_weight"],
        ln2_weight=weights["ln2_weight"],
    )
    component_ops = component_operator.make_validation_component_operators(
        context=component_context,
        names=(name,),
    )
    component_context.compile_all()
    component_context.prepare_runtime()
    component_case = component_operator.validation_component_case(
        name,
        q,
        k,
        v,
        r,
        eps=eps,
    )
    fn = lambda: _run_component_once(
        name=name,
        operator=component_operator,
        component_op=component_ops[name],
        args=component_case["args"],
    )
    return {
        "study_id": "operator_runlist_stability",
        "seq_len": spec.seq_len,
        "validation_scope": "component_boundary",
        **_run_single_check(
            name=name,
            repeats=repeats,
            expected=component_case["expected"],
            fn=fn,
        ),
    }


def validate_operator_runlist_stability(
    *,
    spec: TransformerLayerSpec,
    seed: int,
    repeats: int,
    components_only: bool = False,
    targets: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)

    q = layer_inputs.q.squeeze(0)
    k = layer_inputs.k.squeeze(0)
    v = layer_inputs.v.squeeze(0)
    r = layer_inputs.r.squeeze(0)

    component_names = AIEEncoderRunlist._normalize_component_names(targets)
    component_rows = [
        _component_context_row(
            spec=spec,
            weights=weights,
            q=q,
            k=k,
            v=v,
            r=r,
            eps=spec.layer_norm_eps,
            name=name,
            repeats=repeats,
        )
        for name in component_names
    ]
    if components_only:
        return component_rows

    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(weights)
    pattern._prepare_runtime()
    reference = ReferenceTransformerLayer(spec)
    reference.assign_weights(weights)
    reference_output = reference(layer_inputs)
    full_pattern_row = {
        "study_id": "operator_runlist_stability",
        "seq_len": spec.seq_len,
        "validation_scope": "full_pattern",
        **_run_single_check(
            name="operator_runlist_full_pattern",
            repeats=repeats,
            expected=reference_output,
            fn=lambda: pattern(layer_inputs),
        ),
    }
    return component_rows + [full_pattern_row]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated stability checks for the operator_runlist component graph."
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--components-only", action="store_true")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated subset of component names to validate. Defaults to all component operators.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = validate_operator_runlist_stability(
        spec=TransformerLayerSpec(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            seq_len=args.seq_len,
            use_bias=False,
            weights_source="synthetic",
        ),
        seed=args.seed,
        repeats=args.repeats,
        components_only=args.components_only,
        targets=_parse_target_names(args.targets),
    )
    for row in rows:
        print(
            f"{row['target']}: stable={row['stable']} "
            f"max_abs_diff={row['max_abs_diff']:.6f} "
            f"mean_abs_diff={row['mean_abs_diff']:.6f}"
        )
    if args.output_csv:
        write_dict_rows_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
