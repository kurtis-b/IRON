#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from iron.applications.transformer_layer.benchmark_common import (
    summarize_latency_measurements,
)
from iron.applications.transformer_layer.benchmark_power import (
    create_power_monitor,
    empty_power_stats,
)
from iron.applications.transformer_layer.npu_inference import (
    _average_stage_timings_ms,
    _pattern_metadata,
)
from iron.applications.transformer_layer.roofline import (
    estimate_layer_bytes,
    estimate_layer_flops,
    operational_intensity,
)
from iron.applications.transformer_layer.src.layer_spec import TransformerLayerSpec
from iron.applications.transformer_layer.src.reference_layer import (
    ReferenceTransformerLayer,
)
from iron.applications.transformer_layer.src.pattern_operator_runlist import (
    OperatorRunlistPattern,
)
from iron.applications.transformer_layer.src.utils import (
    make_synthetic_layer_inputs,
    make_synthetic_layer_weights,
)


def benchmark_operator_runlist_request(request: dict[str, object]) -> dict[str, object]:
    spec = TransformerLayerSpec.from_dict(request["spec"])
    warmup_runs = int(request["warmup_runs"])
    runs_per_sample = int(request["runs_per_sample"])
    seed = int(request["seed"])

    pattern = OperatorRunlistPattern(spec)
    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)
    pattern.assign_weights(weights)
    pattern._prepare_runtime()

    def _run_once() -> dict[str, float]:
        _, stage_timings = pattern.forward_with_stage_timings(layer_inputs)
        return stage_timings

    for _ in range(warmup_runs):
        _run_once()

    latencies = []
    stage_sums_sec: dict[str, float] = {}
    with create_power_monitor() as power_stats:
        for _ in range(runs_per_sample):
            start = time.perf_counter()
            stage_timings = _run_once()
            latencies.append(time.perf_counter() - start)
            for stage_name, stage_sec in stage_timings.items():
                stage_sums_sec[stage_name] = stage_sums_sec.get(
                    stage_name, 0.0
                ) + float(stage_sec)

    summary = summarize_latency_measurements(latencies)
    estimated_flops = estimate_layer_flops(spec)
    estimated_bytes = estimate_layer_bytes(spec)
    throughput_flops_per_sec = (
        estimated_flops
        * summary["measured_inference_count"]
        / summary["timed_total_sec"]
    )
    row = {
        "study_id": "synthetic_transformer_layer",
        "backend": "npu",
        "execution_mode": "operator_runlist",
        "pattern_label": "operator_runlist",
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "use_bias": spec.use_bias,
        "weights_source": spec.weights_source,
        "source_model_name": spec.source_model_name,
        "source_layer_index": spec.source_layer_index,
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        **_average_stage_timings_ms(
            stage_sums_sec,
            summary["measured_inference_count"],
        ),
        "throughput_flops_per_sec": throughput_flops_per_sec,
        "estimated_flops_per_inference": estimated_flops,
        "estimated_bytes_per_inference": estimated_bytes,
        "operational_intensity_flops_per_byte": operational_intensity(
            estimated_flops,
            estimated_bytes,
        ),
        "backend_peak_ops_per_sec": None,
        "roofline_bound_ops_per_sec": None,
        "backend_pct_of_peak": None,
        "roofline_pct": None,
        "run_status": "completed",
        "failure_component": None,
        "failure_category": None,
        "failure_message": None,
        **summary,
        **_pattern_metadata(pattern),
        **(power_stats if power_stats is not None else empty_power_stats()),
    }
    row["process_model"] = "child_process"
    return row


def parity_operator_runlist_request(request: dict[str, object]) -> dict[str, object]:
    spec = TransformerLayerSpec.from_dict(request["spec"])
    seed = int(request["seed"])
    study_id = str(request.get("study_id", "synthetic_transformer_layer"))

    weights = make_synthetic_layer_weights(spec, seed=seed)
    layer_inputs = make_synthetic_layer_inputs(spec, seed=seed + 1)

    reference = ReferenceTransformerLayer(spec)
    reference.assign_weights(weights)
    reference_output = reference(layer_inputs)

    pattern = OperatorRunlistPattern(spec)
    pattern.assign_weights(weights)
    candidate_output = pattern(layer_inputs)
    diff = (reference_output - candidate_output).abs().to(candidate_output.dtype)

    return {
        "study_id": study_id,
        "execution_mode": "operator_runlist",
        "seq_len": spec.seq_len,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "attention_head_size": spec.attention_head_size,
        "batch_size": spec.batch_size,
        "dtype": spec.dtype,
        "weights_source": spec.weights_source,
        "seed": seed,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.float().mean().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run operator_runlist benchmark in an isolated child process."
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    request_path = Path(args.request_json)
    response_path = Path(args.response_json)
    request = json.loads(request_path.read_text())
    mode = str(request.get("mode", "benchmark"))
    if mode == "benchmark":
        row = benchmark_operator_runlist_request(request)
    elif mode == "parity":
        row = parity_operator_runlist_request(request)
    else:
        raise ValueError(f"Unsupported operator_runlist worker mode: {mode}")
    response_path.write_text(json.dumps(row))
    os._exit(0)


if __name__ == "__main__":
    main()
