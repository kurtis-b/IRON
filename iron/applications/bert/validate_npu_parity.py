#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from benchmark_common import (
    add_cooldown_args,
    build_benchmark_texts,
    configure_cpu_thread_env,
    detect_max_physical_core_count,
    parse_seq_lens,
    prepare_benchmark_samples,
)
from cpu_inference import load_hf_encoder_model
from model_support import model_uses_token_type_ids
from npu_inference import (
    SUPPORTED_NPU_EXECUTION_MODES,
    apply_topology_to_config,
    build_npu_encoder_model,
    execution_autograd_context,
    load_encoder_pipeline_config,
    resolve_input_paths,
    resolve_topology,
)

DEFAULT_PARITY_SEQ_LENS = "64,128,512"
DEFAULT_MAX_ERROR_FRACTION = 5.0e-2
PARITY_FIELDNAMES = [
    "study_id",
    "seq_len",
    "sample_index",
    "benchmark_mode",
    "execution_mode",
    "disable_all_biases",
    "cpu_dtype",
    "npu_runtime_dtype",
    "topology_id",
    "topology_selection_time_ms",
    "topology_cache_status",
    "compile_setup_time_ms",
    "hidden_shape",
    "cosine_similarity",
    "max_abs_error",
    "mean_abs_error",
    "rel_tol",
    "abs_tol",
    "error_count",
    "total_element_count",
    "error_fraction",
    "max_acceptable_errors",
    "passes_thresholds",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Validate CPU-vs-NPU final hidden-state parity on deterministic "
            "synthetic encoder inputs."
        )
    )
    parser.add_argument(
        "weights_file_path",
        type=str,
        nargs="?",
        help="Path to model.safetensors containing encoder backbone weights.",
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        nargs="?",
        help="Path to the application config JSON that contains model_config.",
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Resolve weights/config from the study manifest instead of passing paths.",
    )
    parser.add_argument(
        "--study-manifest",
        type=str,
        default=None,
        help="Optional path to the study manifest JSON.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Optional root directory that holds downloaded study model artifacts.",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=DEFAULT_PARITY_SEQ_LENS,
        help="Comma-separated sequence lengths to validate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of deterministic samples to validate per sequence length.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Override the host CPU thread count. Defaults to the machine's maximum "
            "physical core count."
        ),
    )
    parser.add_argument(
        "--cpu-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Execution dtype for the CPU reference model.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=SUPPORTED_NPU_EXECUTION_MODES,
        default="encoder_pipeline",
        help="NPU execution path to validate against the CPU reference.",
    )
    parser.add_argument(
        "--disable-all-biases",
        action="store_true",
        help="Zero every model/backbone bias tensor on both CPU and NPU before comparison.",
    )
    parser.add_argument(
        "--topology-policy",
        choices=("fixed", "cache", "autotune"),
        default="fixed",
        help=(
            "How to choose encoder_pipeline topology per sequence length. "
            "fixed uses the config as-is. cache uses a per-seq cache and autotunes on cache misses. "
            "autotune always retunes."
        ),
    )
    parser.add_argument(
        "--topology-cache",
        type=str,
        default="npu_topology_cache_latest.json",
        help="JSON file used by --topology-policy cache/autotune.",
    )
    parser.add_argument(
        "--candidate-topologies",
        type=str,
        default=None,
        help=(
            "Optional comma-separated topology ids to consider during autotune, "
            "for example 2ps_4pffn or seq32_kv64__ps2_ph1_pffn4."
        ),
    )
    parser.add_argument(
        "--autotune-warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per candidate topology during autotune.",
    )
    parser.add_argument(
        "--autotune-runs",
        type=int,
        default=5,
        help="Timed runs per candidate topology during autotune.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=4.0e-2,
        help="Relative tolerance for elementwise parity checks.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1.5e-1,
        help="Absolute tolerance for elementwise parity checks.",
    )
    parser.add_argument(
        "--max-error-fraction",
        type=float,
        default=DEFAULT_MAX_ERROR_FRACTION,
        help=(
            "Maximum allowed fraction of hidden-state elements that may fall "
            "outside the rel/abs tolerance window."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="npu_parity_latest.csv",
        help="CSV file to write parity results to.",
    )
    add_cooldown_args(parser)
    return parser.parse_args(argv)


def compute_hidden_state_metrics(
    reference_hidden_state,
    candidate_hidden_state,
    *,
    rel_tol,
    abs_tol,
):
    import torch

    reference = reference_hidden_state.detach().float().reshape(-1)
    candidate = candidate_hidden_state.detach().float().reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            "Hidden state shapes do not match: "
            f"{tuple(reference_hidden_state.shape)} vs {tuple(candidate_hidden_state.shape)}"
        )

    cosine_similarity = torch.nn.functional.cosine_similarity(
        reference.unsqueeze(0),
        candidate.unsqueeze(0),
        dim=1,
    ).item()
    abs_error = (reference - candidate).abs()
    mismatch_mask = ~torch.isclose(
        reference,
        candidate,
        rtol=rel_tol,
        atol=abs_tol,
    )
    total_element_count = int(reference.numel())
    error_count = int(mismatch_mask.sum().item())
    return {
        "cosine_similarity": cosine_similarity,
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "error_count": error_count,
        "total_element_count": total_element_count,
        "error_fraction": (
            (error_count / total_element_count) if total_element_count else 0.0
        ),
    }


def find_budget_failures(rows):
    failures = []
    for row in rows:
        if int(row["error_count"]) > int(row["max_acceptable_errors"]):
            failures.append(row)
    return failures


def write_parity_results_csv(output_csv, rows):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PARITY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_topology_args(args, weights_file_path, config_file_path):
    return SimpleNamespace(
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
        topology_policy=args.topology_policy,
        topology_cache=args.topology_cache,
        candidate_topologies=args.candidate_topologies,
        autotune_warmup_runs=args.autotune_warmup_runs,
        autotune_runs=args.autotune_runs,
        benchmark_mode="synthetic_dense",
        cooldown_sec=args.cooldown_sec,
        cooldown_until_temp_c=args.cooldown_until_temp_c,
        cooldown_temp_source=args.cooldown_temp_source,
        cooldown_temp_tolerance_frac=args.cooldown_temp_tolerance_frac,
        cooldown_poll_sec=args.cooldown_poll_sec,
        cooldown_timeout_sec=args.cooldown_timeout_sec,
    )


def validate_seq_len(
    *,
    args,
    weights_file_path,
    config_file_path,
    seq_len,
    texts,
):
    import torch

    _, cpu_model = load_hf_encoder_model(
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
        seq_len=seq_len,
        dtype_name=args.cpu_dtype,
        disable_all_biases=args.disable_all_biases,
    )
    cpu_uses_token_type_ids = model_uses_token_type_ids(cpu_model.config.to_dict())
    encoded_samples = prepare_benchmark_samples(
        benchmark_mode="synthetic_dense",
        execution_mode=args.execution_mode,
        texts=texts,
        seq_len=seq_len,
        vocab_size=cpu_model.config.vocab_size,
        pad_token_id=cpu_model.config.pad_token_id,
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
    )

    config = load_encoder_pipeline_config(config_file_path, seq_len)
    if args.execution_mode == "encoder_pipeline":
        topology, topology_selection = resolve_topology(
            build_topology_args(args, weights_file_path, config_file_path),
            seq_len,
            texts,
        )
        apply_topology_to_config(config, topology)
    else:
        topology = None
        topology_selection = {
            "topology_selection_time_ms": 0.0,
            "topology_cache_status": "",
        }
    npu_model, context, compile_setup_time_ms = build_npu_encoder_model(
        weights_file_path=weights_file_path,
        config=config,
        seq_len=seq_len,
        execution_mode=args.execution_mode,
        disable_all_biases=args.disable_all_biases,
    )

    rows = []
    try:
        with execution_autograd_context(args.execution_mode):
            for sample_index, sample in enumerate(encoded_samples):
                cpu_kwargs = {
                    "input_ids": sample["input_ids"],
                    "attention_mask": None,
                }
                if cpu_uses_token_type_ids:
                    cpu_kwargs["token_type_ids"] = sample["token_type_ids"]
                cpu_hidden_state = cpu_model(**cpu_kwargs).last_hidden_state
                npu_hidden_state = npu_model(
                    input_ids=sample["input_ids"],
                    token_type_ids=sample["token_type_ids"],
                    attention_mask=None,
                )
                metrics = compute_hidden_state_metrics(
                    cpu_hidden_state,
                    npu_hidden_state,
                    rel_tol=args.rel_tol,
                    abs_tol=args.abs_tol,
                )
                max_acceptable_errors = int(
                    metrics["total_element_count"] * args.max_error_fraction
                )
                passes_thresholds = metrics["error_count"] <= max_acceptable_errors
                rows.append(
                    {
                        "study_id": args.study_id or "",
                        "seq_len": str(seq_len),
                        "sample_index": str(sample_index),
                        "benchmark_mode": "synthetic_dense",
                        "execution_mode": args.execution_mode,
                        "disable_all_biases": ("1" if args.disable_all_biases else "0"),
                        "cpu_dtype": args.cpu_dtype,
                        "npu_runtime_dtype": str(config.aie_config.dtype).replace(
                            "torch.", ""
                        ),
                        "topology_id": topology.topology_id if topology else "",
                        "topology_selection_time_ms": (
                            f"{topology_selection['topology_selection_time_ms']:.6f}"
                        ),
                        "topology_cache_status": topology_selection[
                            "topology_cache_status"
                        ],
                        "compile_setup_time_ms": f"{compile_setup_time_ms:.6f}",
                        "hidden_shape": tuple(cpu_hidden_state.shape),
                        "cosine_similarity": f"{metrics['cosine_similarity']:.8f}",
                        "max_abs_error": f"{metrics['max_abs_error']:.8f}",
                        "mean_abs_error": f"{metrics['mean_abs_error']:.8f}",
                        "rel_tol": f"{args.rel_tol:.6f}",
                        "abs_tol": f"{args.abs_tol:.6f}",
                        "error_count": str(metrics["error_count"]),
                        "total_element_count": str(metrics["total_element_count"]),
                        "error_fraction": f"{metrics['error_fraction']:.8f}",
                        "max_acceptable_errors": str(max_acceptable_errors),
                        "passes_thresholds": "1" if passes_thresholds else "0",
                    }
                )
    finally:
        context.device_manager.reset()
    return rows


def main(argv=None):
    args = parse_args(argv)
    weights_file_path, config_file_path = resolve_input_paths(args)
    seq_lens = parse_seq_lens(args.seq_lens)
    num_threads = args.num_threads or detect_max_physical_core_count()
    configure_cpu_thread_env(num_threads)

    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    texts = build_benchmark_texts(args.num_samples)
    all_rows = []
    for seq_len in seq_lens:
        seq_rows = validate_seq_len(
            args=args,
            weights_file_path=weights_file_path,
            config_file_path=config_file_path,
            seq_len=seq_len,
            texts=texts,
        )
        all_rows.extend(seq_rows)
        for row in seq_rows:
            print(
                f"seq_len={row['seq_len']} sample={row['sample_index']} "
                f"mode={row['execution_mode']} "
                f"topology={row['topology_id'] or '-'} "
                f"cosine={row['cosine_similarity']} "
                f"max_abs={row['max_abs_error']} "
                f"errors={row['error_count']}/{row['max_acceptable_errors']} "
                f"cache={row['topology_cache_status']}",
                flush=True,
            )

    write_parity_results_csv(args.output_csv, all_rows)
    print(f"Results written to {args.output_csv}", flush=True)

    failures = find_budget_failures(all_rows)
    if failures:
        worst_error_fraction = max(float(row["error_fraction"]) for row in failures)
        worst_error_count = max(int(row["error_count"]) for row in failures)
        raise SystemExit(
            "CPU-vs-NPU parity validation failed: "
            f"{len(failures)} rows exceeded the allowed error budget "
            f"(max error fraction {args.max_error_fraction:.6f}, "
            f"worst observed {worst_error_fraction:.6f}; "
            f"worst error count {worst_error_count})"
        )


if __name__ == "__main__":
    main()
