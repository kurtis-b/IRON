#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SAMPLE_TEXT = """
SCENE I. King Lear's palace.
Enter KENT, GLOUCESTER, and EDMUND
KENT
I thought the king had more affected the Duke of Albany than Cornwall.
GLOUCESTER
It did always seem so to us: but now, in the division of the kingdom, it appears not
which of the dukes he values most; for equalities are so weighed, that curiosity in
neither can make choice of either's moiety. The king is coming. The lords of France
and Burgundy wait in court. We have divided in three our kingdom, and we would know
which daughter loves us most. Sir, I love you more than words can wield the matter;
dearer than eyesight, space, and liberty; beyond what can be valued, rich or rare.
Nothing will come of nothing: speak again. I love your majesty according to my bond,
nor more nor less. Good my lord, you have begot me, bred me, loved me: I return those
duties back as are right fit, obey you, love you, and most honour you.
"""


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
        "--tokenizer-name-or-path",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face tokenizer name or local tokenizer path.",
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
        default=1,
        help="Warmup iterations per sequence length before timing.",
    )
    parser.add_argument(
        "--runs-per-sample",
        type=int,
        default=1,
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


def _detect_physical_cores_from_sysfs():
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return None

    try:
        allowed_cpus = os.sched_getaffinity(0)
    except AttributeError:
        allowed_cpus = None

    physical_cores = set()
    for cpu_dir in cpu_root.glob("cpu[0-9]*"):
        cpu_id = int(cpu_dir.name[3:])
        if allowed_cpus is not None and cpu_id not in allowed_cpus:
            continue
        online_file = cpu_dir / "online"
        if online_file.exists() and online_file.read_text().strip() == "0":
            continue
        core_id_file = cpu_dir / "topology" / "core_id"
        package_id_file = cpu_dir / "topology" / "physical_package_id"
        if not (core_id_file.exists() and package_id_file.exists()):
            continue
        core_id = int(core_id_file.read_text().strip())
        package_id = int(package_id_file.read_text().strip())
        physical_cores.add((package_id, core_id))

    return len(physical_cores) if physical_cores else None


def _detect_physical_cores_from_lscpu():
    try:
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,CORE,SOCKET"], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        allowed_cpus = os.sched_getaffinity(0)
    except AttributeError:
        allowed_cpus = None

    physical_cores = set()
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_id, core_id, socket_id = (int(value) for value in line.split(","))
        if allowed_cpus is not None and cpu_id not in allowed_cpus:
            continue
        physical_cores.add((socket_id, core_id))

    return len(physical_cores) if physical_cores else None


def detect_physical_core_count():
    detected = _detect_physical_cores_from_sysfs()
    if detected:
        return detected
    detected = _detect_physical_cores_from_lscpu()
    if detected:
        return detected
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def configure_cpu_thread_env(num_threads):
    thread_count = str(num_threads)
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = thread_count
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_seq_lens(seq_lens_arg):
    seq_lens = []
    for token in seq_lens_arg.split(","):
        token = token.strip()
        if not token:
            continue
        seq_len = int(token)
        if seq_len <= 0:
            raise ValueError(f"Sequence lengths must be positive (got {seq_len})")
        seq_lens.append(seq_len)
    if not seq_lens:
        raise ValueError("At least one sequence length must be provided")
    return seq_lens


def build_benchmark_texts(num_samples):
    lines = [line.strip() for line in SAMPLE_TEXT.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Benchmark text corpus is empty")

    texts = []
    for sample_idx in range(num_samples):
        rotate = sample_idx % len(lines)
        rotated = lines[rotate:] + lines[:rotate]
        texts.append(" ".join(rotated))
    return texts


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


def load_hf_bert_model(
    weights_file_path,
    config_file_path,
    seq_len,
    dtype_name,
):
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
        bert_state_dict[stripped_key] = value

    extend_or_trim_position_embeddings(bert_state_dict, seq_len)

    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

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
    tokenizer_name_or_path,
    seq_len,
    texts,
    warmup_runs,
    runs_per_sample,
    dtype_name,
):
    import torch
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
    encoded_samples = []
    for text in texts:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        encoded_samples.append(
            {
                "input_ids": encoded["input_ids"],
                "token_type_ids": encoded["token_type_ids"],
            }
        )

    model = load_hf_bert_model(
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
        seq_len=seq_len,
        dtype_name=dtype_name,
    )

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
        "num_measurements": len(latencies_ms),
    }


def write_results_csv(output_csv, rows):
    fieldnames = [
        "seq_len",
        "num_threads",
        "dtype",
        "num_samples",
        "runs_per_sample",
        "warmup_runs",
        "shape",
        "min_latency_ms",
        "avg_latency_ms",
        "max_latency_ms",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seq_lens = parse_seq_lens(args.seq_lens)
    num_threads = args.num_threads or detect_physical_core_count()
    configure_cpu_thread_env(num_threads)

    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    texts = build_benchmark_texts(args.num_samples)

    print(f"Using physical CPU threads: {num_threads}")
    print(f"torch intra-op threads: {torch.get_num_threads()}")
    print(f"torch inter-op threads: {torch.get_num_interop_threads()}")
    print("Benchmarking Hugging Face BertModel(add_pooling_layer=False)")
    print(f"Sequence lengths: {seq_lens}")

    csv_rows = []
    for seq_len in seq_lens:
        result = benchmark_seq_len(
            weights_file_path=args.weights_file_path,
            config_file_path=args.config_file_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            seq_len=seq_len,
            texts=texts,
            warmup_runs=args.warmup_runs,
            runs_per_sample=args.runs_per_sample,
            dtype_name=args.dtype,
        )
        print(
            f"seq_len={seq_len:<5d} "
            f"shape={result['shape']} "
            f"min={result['min_latency_ms']:.3f} ms "
            f"avg={result['avg_latency_ms']:.3f} ms "
            f"max={result['max_latency_ms']:.3f} ms"
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
                "min_latency_ms": f"{result['min_latency_ms']:.6f}",
                "avg_latency_ms": f"{result['avg_latency_ms']:.6f}",
                "max_latency_ms": f"{result['max_latency_ms']:.6f}",
            }
        )

    write_results_csv(args.output_csv, csv_rows)
    print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
