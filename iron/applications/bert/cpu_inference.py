#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time

from benchmark_common import (
    DEFAULT_BENCHMARK_SEQ_LENS,
    add_cooldown_args,
    build_benchmark_texts,
    cooldown_before_benchmark,
    configure_cpu_thread_env,
    detect_logical_thread_count,
    detect_physical_core_count,
    encode_benchmark_texts,
    parse_seq_lens,
    summarize_latency_measurements,
    write_results_csv,
)
from benchmark_power import (
    add_power_measurement_args,
    create_power_monitor,
    derive_power_log_path,
    empty_power_stats,
    format_power_stats_for_csv,
    resolve_power_backend,
)
from model_support import (
    build_hf_encoder_model,
    canonicalize_app_config_dict,
    display_name_for_model,
    estimate_encoder_forward_flops,
    extend_or_trim_position_embeddings,
    extract_backbone_state_dict,
    model_uses_token_type_ids,
    runtime_max_position_embeddings,
    resolve_study_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a Hugging Face encoder-only model on CPU across sequence lengths."
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
        default=DEFAULT_BENCHMARK_SEQ_LENS,
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of text samples to run per sequence length.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Warmup iterations per sequence length before timing.",
    )
    parser.add_argument(
        "--runs-per-sample",
        type=int,
        default=100,
        help="Timed inference runs per sample.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Execution dtype for the CPU model.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Override the number of CPU threads. Defaults to detected physical cores.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="cpu_benchmark_latest.csv",
        help="CSV file to write benchmark results to.",
    )
    add_power_measurement_args(parser)
    add_cooldown_args(parser)
    return parser.parse_args()


def resolve_input_paths(args):
    if args.study_id is not None:
        if args.weights_file_path is not None or args.config_file_path is not None:
            raise ValueError(
                "Use either explicit weights/config paths or --study-id, not both"
            )
        resolved = resolve_study_paths(
            args.study_id,
            manifest_path=args.study_manifest,
            models_root=args.models_root,
        )
        return resolved["weights_file_path"], resolved["config_file_path"]

    if args.weights_file_path is None or args.config_file_path is None:
        raise ValueError(
            "Either pass weights_file_path and config_file_path, or use --study-id"
        )
    return args.weights_file_path, args.config_file_path


def load_app_model_config(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    return canonicalize_app_config_dict(config_json)["model_config"]


def load_hf_encoder_model(weights_file_path, config_file_path, seq_len, dtype_name):
    import torch
    from safetensors.torch import load_file

    model_config = load_app_model_config(config_file_path)
    runtime_position_count = runtime_max_position_embeddings(model_config, seq_len)
    model_config["max_position_embeddings"] = runtime_position_count
    family, _, model = build_hf_encoder_model(model_config)

    raw_weights = load_file(weights_file_path)
    state_dict = extract_backbone_state_dict(raw_weights, model_config)

    extend_or_trim_position_embeddings(state_dict, runtime_position_count)

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"Missing {display_name_for_model(model_config)} weights: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected {display_name_for_model(model_config)} weights: {unexpected}"
        )

    model = model.to(dtype=dtype)
    model.eval()
    return family, model


def benchmark_seq_len(
    *,
    weights_file_path,
    config_file_path,
    seq_len,
    texts,
    warmup_runs,
    runs_per_sample,
    dtype_name,
    power_backend,
    power_interval_sec,
    power_log_path,
):
    model_type, model = load_hf_encoder_model(
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
        seq_len=seq_len,
        dtype_name=dtype_name,
    )
    encoded_samples = encode_benchmark_texts(
        texts=texts,
        seq_len=seq_len,
        vocab_size=model.config.vocab_size,
        pad_token_id=model.config.pad_token_id,
    )
    use_token_type_ids = model_uses_token_type_ids(model_config=model.config.to_dict())

    import torch

    with torch.inference_mode():
        for _ in range(warmup_runs):
            for sample in encoded_samples:
                kwargs = {
                    "input_ids": sample["input_ids"],
                    "attention_mask": None,
                }
                if use_token_type_ids:
                    kwargs["token_type_ids"] = sample["token_type_ids"]
                _ = model(**kwargs).last_hidden_state

        latencies_ms = []
        with create_power_monitor(
            power_backend,
            power_interval_sec,
            log_path=power_log_path,
        ) as power_monitor:
            for sample in encoded_samples:
                for _ in range(runs_per_sample):
                    start = time.perf_counter()
                    kwargs = {
                        "input_ids": sample["input_ids"],
                        "attention_mask": None,
                    }
                    if use_token_type_ids:
                        kwargs["token_type_ids"] = sample["token_type_ids"]
                    output = model(**kwargs).last_hidden_state
                    end = time.perf_counter()
                    latencies_ms.append((end - start) * 1000.0)

    latency_stats = summarize_latency_measurements(
        latencies_ms,
        estimate_encoder_forward_flops(model.config.to_dict(), seq_len),
    )

    return {
        "seq_len": seq_len,
        "model_type": model_type,
        "shape": tuple(output.shape),
        "power_stats": power_monitor.stats,
        **latency_stats,
    }


def main():
    args = parse_args()
    weights_file_path, config_file_path = resolve_input_paths(args)
    seq_lens = parse_seq_lens(args.seq_lens)
    physical_threads = detect_physical_core_count()
    logical_threads = detect_logical_thread_count()
    if args.num_threads is not None:
        thread_counts = [args.num_threads]
    else:
        thread_counts = [physical_threads]
        if logical_threads not in thread_counts:
            thread_counts.append(logical_threads)

    import torch

    torch.set_num_interop_threads(1)

    texts = build_benchmark_texts(args.num_samples)
    model_config = load_app_model_config(config_file_path)
    power_backend = resolve_power_backend(args.power_backend, "cpu")
    multi_case = len(thread_counts) > 1 or len(seq_lens) > 1

    print(f"CPU thread comparison set: {thread_counts}", flush=True)
    print(f"torch inter-op threads: {torch.get_num_interop_threads()}", flush=True)
    print(
        f"Benchmarking Hugging Face {display_name_for_model(model_config)}",
        flush=True,
    )
    print(f"Sequence lengths: {seq_lens}", flush=True)

    csv_rows = []
    study_id = args.study_id or ""
    for num_threads in thread_counts:
        configure_cpu_thread_env(num_threads)
        torch.set_num_threads(num_threads)
        print(f"Running CPU benchmark with {num_threads} threads", flush=True)
        print(f"torch intra-op threads: {torch.get_num_threads()}", flush=True)

        for seq_len in seq_lens:
            cooldown_before_benchmark(
                args,
                mode="cpu",
                label=f"CPU seq_len={seq_len} threads={num_threads}",
            )
            print(
                f"Starting CPU benchmark: threads={num_threads} seq_len={seq_len}",
                flush=True,
            )
            power_log_path = (
                derive_power_log_path(
                    args.power_log_path,
                    f"{num_threads}t" if multi_case else None,
                    f"seq{seq_len}" if multi_case else None,
                )
                if power_backend != "none"
                else None
            )
            result = benchmark_seq_len(
                weights_file_path=weights_file_path,
                config_file_path=config_file_path,
                seq_len=seq_len,
                texts=texts,
                warmup_runs=args.warmup_runs,
                runs_per_sample=args.runs_per_sample,
                dtype_name=args.dtype,
                power_backend=power_backend,
                power_interval_sec=args.power_interval_sec,
                power_log_path=power_log_path,
            )
            print(
                f"threads={num_threads:<2d} "
                f"seq_len={seq_len:<5d} "
                f"shape={result['shape']} "
                f"min={result['min_latency_ms']:.3f} ms "
                f"avg={result['avg_latency_ms']:.3f} ms "
                f"max={result['max_latency_ms']:.3f} ms",
                flush=True,
            )
            csv_rows.append(
                {
                    "study_id": study_id,
                    "seq_len": seq_len,
                    "num_threads": num_threads,
                    "dtype": args.dtype,
                    "num_samples": args.num_samples,
                    "runs_per_sample": args.runs_per_sample,
                    "warmup_runs": args.warmup_runs,
                    "measured_inference_count": result["measured_inference_count"],
                    "timed_total_sec": f"{result['timed_total_sec']:.6f}",
                    "throughput_inferences_per_sec": (
                        f"{result['throughput_inferences_per_sec']:.6f}"
                    ),
                    "model_type": result["model_type"],
                    "shape": result["shape"],
                    "estimated_flops_per_inference": (
                        f"{result['estimated_flops_per_inference']:.6e}"
                    ),
                    "throughput_flops_per_sec": (
                        f"{result['throughput_flops_per_sec']:.6e}"
                    ),
                    "topology_id": "",
                    "parallel_seq": "",
                    "parallel_heads": "",
                    "parallel_ffn": "",
                    "min_latency_ms": f"{result['min_latency_ms']:.6f}",
                    "avg_latency_ms": f"{result['avg_latency_ms']:.6f}",
                    "max_latency_ms": f"{result['max_latency_ms']:.6f}",
                    **format_power_stats_for_csv(
                        power_backend,
                        result.get("power_stats", empty_power_stats()),
                        power_log_path,
                    ),
                }
            )
            write_results_csv(args.output_csv, csv_rows)
            print(f"Checkpointed results to {args.output_csv}", flush=True)

    print(f"Results written to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
