#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time

from benchmark_common import (
    build_benchmark_texts,
    configure_cpu_thread_env,
    detect_logical_thread_count,
    detect_physical_core_count,
    encode_benchmark_texts,
    parse_seq_lens,
    write_results_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Hugging Face BertModel on CPU across sequence lengths."
    )
    parser.add_argument(
        "weights_file_path",
        type=str,
        help="Path to model.safetensors containing BERT sequence-classification weights.",
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Path to the application config JSON that contains model_config.",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="64,128,256,512,1024,2048,4096,8192",
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
    return parser.parse_args()


def load_app_model_config(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    if "model_config" not in config_json:
        raise ValueError(f"{config_file_path} does not contain model_config")
    return config_json["model_config"]


def extend_or_trim_position_embeddings(state_dict, seq_len):
    key = "embeddings.position_embeddings.weight"
    if key not in state_dict:
        return
    weights = state_dict[key]
    if weights.shape[0] == seq_len:
        return
    if weights.shape[0] > seq_len:
        state_dict[key] = weights[:seq_len, :]
        return
    repeats = (seq_len // weights.shape[0]) + 1
    state_dict[key] = weights.repeat(repeats, 1)[:seq_len, :]


def load_hf_bert_model(weights_file_path, config_file_path, seq_len, dtype_name):
    import torch
    from safetensors.torch import load_file
    from transformers import BertConfig, BertModel

    model_config = load_app_model_config(config_file_path)
    model_config["max_position_embeddings"] = seq_len
    hf_config = BertConfig.from_dict(model_config)
    model = BertModel(hf_config, add_pooling_layer=False)

    raw_weights = load_file(weights_file_path)
    bert_state_dict = {}
    for key, value in raw_weights.items():
        if not key.startswith("bert."):
            continue
        stripped_key = key[len("bert.") :]
        if stripped_key.startswith("pooler."):
            continue
        stripped_key = stripped_key.replace("LayerNorm.gamma", "LayerNorm.weight")
        stripped_key = stripped_key.replace("LayerNorm.beta", "LayerNorm.bias")
        bert_state_dict[stripped_key] = value

    extend_or_trim_position_embeddings(bert_state_dict, seq_len)

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    missing, unexpected = model.load_state_dict(bert_state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing BertModel weights: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected BertModel weights: {unexpected}")

    model = model.to(dtype=dtype)
    model.eval()
    return model


def benchmark_seq_len(
    *,
    weights_file_path,
    config_file_path,
    seq_len,
    texts,
    warmup_runs,
    runs_per_sample,
    dtype_name,
):
    model = load_hf_bert_model(
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

    import torch

    with torch.inference_mode():
        for _ in range(warmup_runs):
            for sample in encoded_samples:
                _ = model(
                    input_ids=sample["input_ids"],
                    token_type_ids=sample["token_type_ids"],
                    attention_mask=None,
                ).last_hidden_state

        latencies_ms = []
        for sample in encoded_samples:
            for _ in range(runs_per_sample):
                start = time.perf_counter()
                output = model(
                    input_ids=sample["input_ids"],
                    token_type_ids=sample["token_type_ids"],
                    attention_mask=None,
                ).last_hidden_state
                end = time.perf_counter()
                latencies_ms.append((end - start) * 1000.0)

    return {
        "seq_len": seq_len,
        "shape": tuple(output.shape),
        "min_latency_ms": min(latencies_ms),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "max_latency_ms": max(latencies_ms),
    }


def main():
    args = parse_args()
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

    print(f"CPU thread comparison set: {thread_counts}", flush=True)
    print(f"torch inter-op threads: {torch.get_num_interop_threads()}", flush=True)
    print("Benchmarking Hugging Face BertModel(add_pooling_layer=False)", flush=True)
    print(f"Sequence lengths: {seq_lens}", flush=True)

    csv_rows = []
    for num_threads in thread_counts:
        configure_cpu_thread_env(num_threads)
        torch.set_num_threads(num_threads)
        print(f"Running CPU benchmark with {num_threads} threads", flush=True)
        print(f"torch intra-op threads: {torch.get_num_threads()}", flush=True)

        for seq_len in seq_lens:
            print(
                f"Starting CPU benchmark: threads={num_threads} seq_len={seq_len}",
                flush=True,
            )
            result = benchmark_seq_len(
                weights_file_path=args.weights_file_path,
                config_file_path=args.config_file_path,
                seq_len=seq_len,
                texts=texts,
                warmup_runs=args.warmup_runs,
                runs_per_sample=args.runs_per_sample,
                dtype_name=args.dtype,
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
                    "seq_len": seq_len,
                    "num_threads": num_threads,
                    "dtype": args.dtype,
                    "num_samples": args.num_samples,
                    "runs_per_sample": args.runs_per_sample,
                    "warmup_runs": args.warmup_runs,
                    "shape": result["shape"],
                    "topology_id": "",
                    "parallel_seq": "",
                    "parallel_heads": "",
                    "parallel_ffn": "",
                    "min_latency_ms": f"{result['min_latency_ms']:.6f}",
                    "avg_latency_ms": f"{result['avg_latency_ms']:.6f}",
                    "max_latency_ms": f"{result['max_latency_ms']:.6f}",
                }
            )
            write_results_csv(args.output_csv, csv_rows)
            print(f"Checkpointed results to {args.output_csv}", flush=True)

    print(f"Results written to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
