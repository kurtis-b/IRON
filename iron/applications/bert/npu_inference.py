#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark_common import (
    build_benchmark_texts,
    configure_cpu_thread_env,
    detect_physical_core_count,
    encode_benchmark_texts,
    parse_seq_lens,
    write_results_csv,
)
from model_support import (
    canonicalize_app_config_dict,
    canonicalize_local_backbone_weights,
    display_name_for_model,
    extend_or_trim_position_embeddings,
    resolve_study_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark an encoder_pipeline-backed encoder-only model on NPU."
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
        "--num-threads",
        type=int,
        default=None,
        help="Override the host CPU thread count. Defaults to detected physical cores.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="npu_benchmark_latest.csv",
        help="CSV file to write benchmark results to.",
    )
    parser.add_argument(
        "--topology-policy",
        choices=("fixed", "cache", "autotune"),
        default="cache",
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
            "for example: 2ps,4ps,2ps_2ph,2ps_2pffn"
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


def dtype_from_string(inp):
    import torch

    if isinstance(inp, torch.dtype):
        return inp
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        inp, torch.float32
    )


def load_encoder_pipeline_config(config_file_path, seq_len):
    with open(config_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = canonicalize_app_config_dict(data, seq_len=seq_len)
    config = json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))
    config.aie_config.dtype = dtype_from_string(config.aie_config.dtype)
    return config


def current_topology_from_config(config):
    aie_cfg = config.aie_config
    emb_tile = int(getattr(aie_cfg, "encoder_pipeline_emb_tile", 96))
    ffn_tile = int(getattr(aie_cfg, "encoder_pipeline_ffn_tile", 64))
    proj_acc_depth = int(
        getattr(
            aie_cfg,
            "encoder_pipeline_proj_acc_depth",
            config.model_config.hidden_size // emb_tile,
        )
    )
    return {
        "parallel_seq": int(getattr(aie_cfg, "encoder_pipeline_parallel_seq", 1)),
        "parallel_heads": int(getattr(aie_cfg, "encoder_pipeline_parallel_heads", 1)),
        "parallel_ffn": int(getattr(aie_cfg, "encoder_pipeline_parallel_ffn", 1)),
        "seq_tile": int(getattr(aie_cfg, "encoder_pipeline_seq_tile", 32)),
        "kv_seq_tile": int(getattr(aie_cfg, "encoder_pipeline_kv_seq_tile", 64)),
        "emb_tile": emb_tile,
        "ffn_tile": ffn_tile,
        "proj_acc_depth": proj_acc_depth,
        "o_proj_acc_group_size": int(
            getattr(aie_cfg, "encoder_pipeline_o_proj_acc_group_size", 1)
        ),
        "ffn_intermediate_size": int(
            getattr(
                aie_cfg,
                "encoder_pipeline_ffn_intermediate_size",
                config.model_config.intermediate_size,
            )
        ),
    }


def topology_id(topology):
    parts = [f"{topology['parallel_seq']}ps"]
    if topology["parallel_heads"] > 1:
        parts.append(f"{topology['parallel_heads']}ph")
    if topology["parallel_ffn"] > 1:
        parts.append(f"{topology['parallel_ffn']}pffn")
    return "_".join(parts)


def parse_candidate_topology_ids(raw_ids):
    if raw_ids is None:
        return None
    parsed = []
    for token in raw_ids.split(","):
        token = token.strip()
        if token:
            parsed.append(token)
    return set(parsed) if parsed else None


def apply_topology_to_config(config, topology):
    config.aie_config.encoder_pipeline_seq_tile = topology["seq_tile"]
    config.aie_config.encoder_pipeline_kv_seq_tile = topology["kv_seq_tile"]
    config.aie_config.encoder_pipeline_emb_tile = topology["emb_tile"]
    config.aie_config.encoder_pipeline_ffn_tile = topology["ffn_tile"]
    config.aie_config.encoder_pipeline_parallel_seq = topology["parallel_seq"]
    config.aie_config.encoder_pipeline_parallel_heads = topology["parallel_heads"]
    config.aie_config.encoder_pipeline_proj_acc_depth = topology["proj_acc_depth"]
    config.aie_config.encoder_pipeline_o_proj_acc_group_size = topology[
        "o_proj_acc_group_size"
    ]
    config.aie_config.encoder_pipeline_parallel_ffn = topology["parallel_ffn"]
    config.aie_config.encoder_pipeline_ffn_intermediate_size = topology[
        "ffn_intermediate_size"
    ]
    return config


def load_topology_cache(cache_path):
    path = Path(cache_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_topology_cache(cache_path, cache_data):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, sort_keys=True)
        f.write("\n")


def supported_topologies_for_seq_len(config, seq_len, candidate_ids=None):
    from iron.operators.encoder_pipeline.placements import TOPOLOGY_PLACEMENTS

    current = current_topology_from_config(config)
    d = config.model_config.hidden_size // config.model_config.num_attention_heads
    matches = []
    for key in TOPOLOGY_PLACEMENTS:
        if (
            key[0] == config.model_config.num_attention_heads
            and key[1] == seq_len
            and key[2] == d
            and key[3] == current["seq_tile"]
            and key[4] == current["kv_seq_tile"]
            and key[5] == current["emb_tile"]
            and key[6] == current["ffn_tile"]
            and key[9] == current["proj_acc_depth"]
            and key[10] == current["o_proj_acc_group_size"]
            and key[12] == current["ffn_intermediate_size"]
        ):
            topology = {
                "parallel_seq": key[7],
                "parallel_heads": key[8],
                "parallel_ffn": key[11],
                "seq_tile": key[3],
                "kv_seq_tile": key[4],
                "emb_tile": key[5],
                "ffn_tile": key[6],
                "proj_acc_depth": key[9],
                "o_proj_acc_group_size": key[10],
                "ffn_intermediate_size": key[12],
            }
            if candidate_ids is not None and topology_id(topology) not in candidate_ids:
                continue
            matches.append(topology)

    matches.sort(
        key=lambda topo: (
            topo["parallel_seq"],
            topo["parallel_heads"],
            topo["parallel_ffn"],
        )
    )
    if not matches:
        requested = (
            sorted(candidate_ids) if candidate_ids is not None else "all supported"
        )
        raise RuntimeError(
            f"No supported encoder_pipeline topologies found for seq_len={seq_len} "
            f"with candidate filter {requested}"
        )
    return matches


def reset_default_context():
    from iron.common import AIEOperatorBase
    from iron.common.aie_context import AIEContext

    current = getattr(AIEOperatorBase, "_default_context", None)
    if current is not None:
        current.device_manager.reset()
    AIEOperatorBase._default_context = AIEContext()
    return AIEOperatorBase._default_context


def build_npu_encoder_model(weights_file_path, config):
    from safetensors.torch import load_file

    from iron.common import AIEOperatorBase
    from src.model import EncoderBackbone

    seq_len = config.model_config.max_position_embeddings
    reset_default_context()
    model = EncoderBackbone(config, seq_len=seq_len)

    raw_weights = load_file(weights_file_path)
    canonical_weights = canonicalize_local_backbone_weights(
        raw_weights,
        config.model_config,
    )
    extend_or_trim_position_embeddings(canonical_weights, seq_len)
    model.assign_backbone_weights(canonical_weights)
    model.eval()

    context = AIEOperatorBase.get_default_context()
    context.compile_all()
    context.prepare_runtime()
    return model, context


def benchmark_with_config(
    *,
    weights_file_path,
    config,
    seq_len,
    texts,
    warmup_runs,
    runs_per_sample,
    topology,
):
    import torch

    model, context = build_npu_encoder_model(
        weights_file_path=weights_file_path,
        config=config,
    )
    encoded_samples = encode_benchmark_texts(
        texts=texts,
        seq_len=seq_len,
        vocab_size=config.model_config.vocab_size,
        pad_token_id=config.model_config.pad_token_id,
    )

    with torch.inference_mode():
        for _ in range(warmup_runs):
            for sample in encoded_samples:
                _ = model(
                    input_ids=sample["input_ids"],
                    token_type_ids=sample["token_type_ids"],
                    attention_mask=None,
                )

        latencies_ms = []
        for sample in encoded_samples:
            for _ in range(runs_per_sample):
                start = time.perf_counter()
                output = model(
                    input_ids=sample["input_ids"],
                    token_type_ids=sample["token_type_ids"],
                    attention_mask=None,
                )
                end = time.perf_counter()
                latencies_ms.append((end - start) * 1000.0)

    context.device_manager.reset()

    return {
        "seq_len": seq_len,
        "model_type": str(config.model_config.model_type),
        "shape": tuple(output.shape),
        "dtype": str(model.dtype).replace("torch.", ""),
        "topology_id": topology_id(topology),
        "parallel_seq": topology["parallel_seq"],
        "parallel_heads": topology["parallel_heads"],
        "parallel_ffn": topology["parallel_ffn"],
        "min_latency_ms": min(latencies_ms),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "max_latency_ms": max(latencies_ms),
    }


def autotune_topology(
    *,
    weights_file_path,
    config_file_path,
    seq_len,
    texts,
    candidate_ids,
    warmup_runs,
    runs_per_sample,
):
    base_config = load_encoder_pipeline_config(config_file_path, seq_len)
    candidates = supported_topologies_for_seq_len(
        base_config, seq_len, candidate_ids=candidate_ids
    )
    if len(candidates) == 1:
        return candidates[0]

    best_topology = None
    best_latency_ms = None
    for topology in candidates:
        print(
            f"Autotune candidate: seq_len={seq_len} topology={topology_id(topology)}",
            flush=True,
        )
        tuned_config = load_encoder_pipeline_config(config_file_path, seq_len)
        apply_topology_to_config(tuned_config, topology)
        result = benchmark_with_config(
            weights_file_path=weights_file_path,
            config=tuned_config,
            seq_len=seq_len,
            texts=texts,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            topology=topology,
        )
        print(
            f"Autotune result: seq_len={seq_len} topology={result['topology_id']} "
            f"avg={result['avg_latency_ms']:.3f} ms",
            flush=True,
        )
        if best_latency_ms is None or result["avg_latency_ms"] < best_latency_ms:
            best_latency_ms = result["avg_latency_ms"]
            best_topology = topology

    return best_topology


def resolve_topology(args, seq_len, texts):
    base_config = load_encoder_pipeline_config(args.config_file_path, seq_len)
    fixed_topology = current_topology_from_config(base_config)
    if args.topology_policy == "fixed":
        return fixed_topology

    candidate_ids = parse_candidate_topology_ids(args.candidate_topologies)
    cache_data = load_topology_cache(args.topology_cache)
    cache_key = str(seq_len)
    if args.topology_policy == "cache" and cache_key in cache_data:
        return cache_data[cache_key]

    selected = autotune_topology(
        weights_file_path=args.weights_file_path,
        config_file_path=args.config_file_path,
        seq_len=seq_len,
        texts=texts,
        candidate_ids=candidate_ids,
        warmup_runs=args.autotune_warmup_runs,
        runs_per_sample=args.autotune_runs,
    )
    cache_data[cache_key] = selected
    save_topology_cache(args.topology_cache, cache_data)
    return selected


def main():
    args = parse_args()
    weights_file_path, config_file_path = resolve_input_paths(args)
    args.weights_file_path = weights_file_path
    args.config_file_path = config_file_path
    seq_lens = parse_seq_lens(args.seq_lens)
    num_threads = args.num_threads or detect_physical_core_count()
    configure_cpu_thread_env(num_threads)

    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    texts = build_benchmark_texts(args.num_samples)

    print(f"Using host CPU threads: {num_threads}", flush=True)
    print(f"torch intra-op threads: {torch.get_num_threads()}", flush=True)
    print(f"torch inter-op threads: {torch.get_num_interop_threads()}", flush=True)
    base_config = load_encoder_pipeline_config(config_file_path, seq_lens[0])
    print(
        f"Benchmarking local {display_name_for_model(base_config.model_config)} encoder "
        "with encoder_pipeline operator",
        flush=True,
    )
    print(f"Sequence lengths: {seq_lens}", flush=True)
    print(f"Topology policy: {args.topology_policy}", flush=True)

    csv_rows = []
    for seq_len in seq_lens:
        print(f"Starting NPU benchmark: seq_len={seq_len}", flush=True)
        topology = resolve_topology(args, seq_len, texts)
        print(
            f"Selected topology: seq_len={seq_len} topology={topology_id(topology)}",
            flush=True,
        )
        config = load_encoder_pipeline_config(config_file_path, seq_len)
        apply_topology_to_config(config, topology)
        result = benchmark_with_config(
            weights_file_path=weights_file_path,
            config=config,
            seq_len=seq_len,
            texts=texts,
            warmup_runs=args.warmup_runs,
            runs_per_sample=args.runs_per_sample,
            topology=topology,
        )
        print(
            f"seq_len={seq_len:<5d} topology={result['topology_id']:<10s} "
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
                "dtype": result["dtype"],
                "num_samples": args.num_samples,
                "runs_per_sample": args.runs_per_sample,
                "warmup_runs": args.warmup_runs,
                "model_type": result["model_type"],
                "shape": result["shape"],
                "topology_id": result["topology_id"],
                "parallel_seq": result["parallel_seq"],
                "parallel_heads": result["parallel_heads"],
                "parallel_ffn": result["parallel_ffn"],
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
