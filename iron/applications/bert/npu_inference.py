#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark_common import (
    DEFAULT_BENCHMARK_SEQ_LENS,
    add_benchmark_mode_args,
    add_cooldown_args,
    build_benchmark_texts,
    cooldown_before_benchmark,
    configure_cpu_thread_env,
    default_execution_mode_for_backend,
    detect_max_physical_core_count,
    parse_seq_lens,
    prepare_benchmark_samples,
    summarize_latency_measurements,
    validate_benchmark_mode_request,
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
    canonicalize_app_config_dict,
    canonicalize_local_backbone_weights,
    display_name_for_model,
    estimate_encoder_forward_flops,
    extend_or_trim_position_embeddings,
    resolve_study_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
_TOPOLOGY_MODULE = None


def load_encoder_pipeline_topology_module():
    global _TOPOLOGY_MODULE
    if _TOPOLOGY_MODULE is not None:
        return _TOPOLOGY_MODULE

    topology_path = (
        REPO_ROOT / "iron" / "operators" / "encoder_pipeline" / "topology.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_iron_encoder_pipeline_topology",
        topology_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load encoder_pipeline topology helpers from {topology_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TOPOLOGY_MODULE = module
    return _TOPOLOGY_MODULE


def load_encoder_pipeline_topology_placements():
    return (
        load_encoder_pipeline_topology_module().load_encoder_pipeline_topology_placements()
    )


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
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Override the host CPU thread count. Defaults to the machine's maximum "
            "physical core count."
        ),
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
            "for example legacy aliases like 2ps,4ps,2ps_2ph or canonical ids "
            "like seq32_kv64__ps2_ph1_pffn4"
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
    add_benchmark_mode_args(parser)
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


def current_topology_from_config(config, seq_len):
    return load_encoder_pipeline_topology_module().topology_from_config(config, seq_len)


def topology_id(topology):
    return load_encoder_pipeline_topology_module().topology_id(topology)


def parse_candidate_topology_ids(raw_ids):
    return load_encoder_pipeline_topology_module().parse_candidate_topology_filters(
        raw_ids
    )


def apply_topology_to_config(config, topology):
    return load_encoder_pipeline_topology_module().apply_topology_to_config(
        config, topology
    )


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


def topology_signature(topology):
    return load_encoder_pipeline_topology_module().topology_signature(topology)


def topology_cache_key(config, seq_len):
    return load_encoder_pipeline_topology_module().topology_cache_key(config, seq_len)


def supported_topologies_for_seq_len(config, seq_len, candidate_ids=None):
    return load_encoder_pipeline_topology_module().supported_topologies_for_seq_len(
        config, seq_len, candidate_ids=candidate_ids
    )


def supported_topologies_for_config_family(config, seq_len):
    return (
        load_encoder_pipeline_topology_module().supported_topologies_for_config_family(
            config, seq_len
        )
    )


def find_cached_topology(cache_data, config, seq_len, candidate_ids=None):
    return load_encoder_pipeline_topology_module().find_cached_topology(
        cache_data,
        config,
        seq_len,
        candidate_ids=candidate_ids,
    )


def reset_default_context():
    from iron.common import AIEOperatorBase
    from iron.common.aie_context import AIEContext

    current = getattr(AIEOperatorBase, "_default_context", None)
    if current is not None:
        current.device_manager.reset()
    AIEOperatorBase._default_context = AIEContext()
    return AIEOperatorBase._default_context


def build_npu_encoder_model(weights_file_path, config, seq_len):
    from safetensors.torch import load_file

    from iron.common import AIEOperatorBase
    from src.model import EncoderBackbone

    setup_start = time.perf_counter()
    reset_default_context()
    model = EncoderBackbone(config, seq_len=seq_len)

    raw_weights = load_file(weights_file_path)
    canonical_weights = canonicalize_local_backbone_weights(
        raw_weights,
        config.model_config,
    )
    extend_or_trim_position_embeddings(
        canonical_weights,
        int(config.model_config.max_position_embeddings),
    )
    model.assign_backbone_weights(canonical_weights)
    model.eval()

    context = AIEOperatorBase.get_default_context()
    context.compile_all()
    context.prepare_runtime()
    setup_end = time.perf_counter()
    return model, context, (setup_end - setup_start) * 1000.0


def benchmark_with_config(
    *,
    weights_file_path,
    config_file_path,
    config,
    seq_len,
    texts,
    warmup_runs,
    runs_per_sample,
    benchmark_mode,
    topology,
    power_backend="none",
    power_interval_sec=0.5,
    power_log_path=None,
):
    import torch

    model, context, compile_setup_time_ms = build_npu_encoder_model(
        weights_file_path=weights_file_path,
        config=config,
        seq_len=seq_len,
    )
    encoded_samples = prepare_benchmark_samples(
        benchmark_mode=benchmark_mode,
        texts=texts,
        seq_len=seq_len,
        vocab_size=config.model_config.vocab_size,
        pad_token_id=config.model_config.pad_token_id,
        weights_file_path=weights_file_path,
        config_file_path=config_file_path,
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
        embedding_latencies_ms = []
        qkv_projection_latencies_ms = []
        encoder_pipeline_latencies_ms = []
        with create_power_monitor(
            power_backend,
            power_interval_sec,
            log_path=power_log_path,
        ) as power_monitor:
            for sample in encoded_samples:
                for _ in range(runs_per_sample):
                    start = time.perf_counter()
                    output, stage_timings = model.forward_with_stage_timings(
                        input_ids=sample["input_ids"],
                        token_type_ids=sample["token_type_ids"],
                        attention_mask=None,
                    )
                    end = time.perf_counter()
                    latencies_ms.append((end - start) * 1000.0)
                    embedding_latencies_ms.append(
                        stage_timings["embedding_sec"] * 1000.0
                    )
                    qkv_projection_latencies_ms.append(
                        stage_timings["qkv_projection_sec"] * 1000.0
                    )
                    encoder_pipeline_latencies_ms.append(
                        stage_timings["encoder_pipeline_sec"] * 1000.0
                    )

    context.device_manager.reset()
    latency_stats = summarize_latency_measurements(
        latencies_ms,
        estimate_encoder_forward_flops(vars(config.model_config), seq_len),
    )

    return {
        "seq_len": seq_len,
        "model_type": str(config.model_config.model_type),
        "shape": tuple(output.shape),
        "dtype": str(model.dtype).replace("torch.", ""),
        "benchmark_mode": benchmark_mode,
        "execution_mode": default_execution_mode_for_backend("npu"),
        "topology_id": topology_id(topology),
        "parallel_seq": topology.parallel_seq,
        "parallel_heads": topology.parallel_heads,
        "parallel_ffn": topology.parallel_ffn,
        "compute_tile_count": topology.compute_tile_count,
        "compute_tile_utilization_fraction": topology.utilization_fraction,
        "compile_setup_time_ms": compile_setup_time_ms,
        "topology_selection_time_ms": "",
        "topology_cache_status": "",
        "cached_steady_state_avg_latency_ms": latency_stats["avg_latency_ms"],
        "avg_embedding_latency_ms": (
            sum(embedding_latencies_ms) / len(embedding_latencies_ms)
        ),
        "avg_qkv_projection_latency_ms": (
            sum(qkv_projection_latencies_ms) / len(qkv_projection_latencies_ms)
        ),
        "avg_encoder_pipeline_latency_ms": (
            sum(encoder_pipeline_latencies_ms) / len(encoder_pipeline_latencies_ms)
        ),
        "power_stats": power_monitor.stats,
        **latency_stats,
    }


def select_autotune_topology(candidate_results, preferred_family_id):
    if not candidate_results:
        raise RuntimeError("Expected at least one viable autotune candidate")

    best_latency_ms = min(
        candidate_result["avg_latency_ms"] for candidate_result in candidate_results
    )
    latency_band = [
        candidate_result
        for candidate_result in candidate_results
        if candidate_result["avg_latency_ms"] <= (best_latency_ms * 1.01)
    ]
    return max(
        latency_band,
        key=lambda candidate_result: (
            candidate_result["topology"].compute_tile_count,
            int(candidate_result["topology"].family_id == preferred_family_id),
            candidate_result["topology"].topology_id,
        ),
    )


def autotune_topology(
    *,
    args,
    weights_file_path,
    config_file_path,
    seq_len,
    texts,
    candidate_ids,
    warmup_runs,
    runs_per_sample,
    benchmark_mode,
):
    base_config = load_encoder_pipeline_config(config_file_path, seq_len)
    candidates = supported_topologies_for_seq_len(
        base_config, seq_len, candidate_ids=candidate_ids
    )
    if len(candidates) == 1:
        return candidates[0]

    candidate_results = []
    preferred_family_id = current_topology_from_config(base_config, seq_len).family_id
    for topology in candidates:
        cooldown_before_benchmark(
            args=args,
            mode="npu",
            label=f"NPU autotune seq_len={seq_len} topology={topology_id(topology)}",
        )
        print(
            f"Autotune candidate: seq_len={seq_len} topology={topology_id(topology)} "
            f"compute_tiles={topology.compute_tile_count} "
            f"utilization={topology.utilization_fraction:.3f}",
            flush=True,
        )
        tuned_config = load_encoder_pipeline_config(config_file_path, seq_len)
        apply_topology_to_config(tuned_config, topology)
        try:
            result = benchmark_with_config(
                weights_file_path=weights_file_path,
                config_file_path=config_file_path,
                config=tuned_config,
                seq_len=seq_len,
                texts=texts,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                benchmark_mode=benchmark_mode,
                topology=topology,
            )
        except Exception as exc:
            print(
                f"Autotune failed: seq_len={seq_len} "
                f"topology={topology_id(topology)} error={exc}",
                flush=True,
            )
            continue
        print(
            f"Autotune result: seq_len={seq_len} topology={result['topology_id']} "
            f"avg={result['avg_latency_ms']:.3f} ms "
            f"compute_tiles={result['compute_tile_count']} "
            f"utilization={result['compute_tile_utilization_fraction']:.3f}",
            flush=True,
        )
        candidate_results.append(
            {
                "topology": topology,
                "avg_latency_ms": result["avg_latency_ms"],
            }
        )

    if not candidate_results:
        raise RuntimeError(
            f"No viable encoder_pipeline topologies for seq_len={seq_len}"
        )

    selected = select_autotune_topology(candidate_results, preferred_family_id)
    print(
        f"Autotune selected: seq_len={seq_len} topology={selected['topology'].topology_id} "
        f"compute_tiles={selected['topology'].compute_tile_count} "
        f"utilization={selected['topology'].utilization_fraction:.3f} "
        f"avg={selected['avg_latency_ms']:.3f} ms",
        flush=True,
    )
    return selected["topology"]


def resolve_topology(args, seq_len, texts):
    selection_start = time.perf_counter()
    base_config = load_encoder_pipeline_config(args.config_file_path, seq_len)
    fixed_topology = current_topology_from_config(base_config, seq_len)
    if args.topology_policy == "fixed":
        selection_end = time.perf_counter()
        return fixed_topology, {
            "topology_selection_time_ms": (selection_end - selection_start) * 1000.0,
            "topology_cache_status": "fixed",
        }

    candidate_ids = parse_candidate_topology_ids(args.candidate_topologies)
    cache_data = load_topology_cache(args.topology_cache)
    cache_key = topology_cache_key(base_config, seq_len)
    if args.topology_policy == "cache":
        cached_topology = find_cached_topology(
            cache_data, base_config, seq_len, candidate_ids=candidate_ids
        )
        if cached_topology is not None:
            selection_end = time.perf_counter()
            return cached_topology, {
                "topology_selection_time_ms": (selection_end - selection_start)
                * 1000.0,
                "topology_cache_status": "cache_hit",
            }

    selected = autotune_topology(
        args=args,
        weights_file_path=args.weights_file_path,
        config_file_path=args.config_file_path,
        seq_len=seq_len,
        texts=texts,
        candidate_ids=candidate_ids,
        warmup_runs=args.autotune_warmup_runs,
        runs_per_sample=args.autotune_runs,
        benchmark_mode=args.benchmark_mode,
    )
    cache_data[cache_key] = selected.to_runtime_dict()
    save_topology_cache(args.topology_cache, cache_data)
    selection_end = time.perf_counter()
    return selected, {
        "topology_selection_time_ms": (selection_end - selection_start) * 1000.0,
        "topology_cache_status": (
            "cache_miss" if args.topology_policy == "cache" else "autotune"
        ),
    }


def main():
    args = parse_args()
    weights_file_path, config_file_path = resolve_input_paths(args)
    args.weights_file_path = weights_file_path
    args.config_file_path = config_file_path
    seq_lens = parse_seq_lens(args.seq_lens)
    validate_benchmark_mode_request(
        args.benchmark_mode,
        backend_mode="npu",
        seq_lens=seq_lens,
    )
    num_threads = args.num_threads or detect_max_physical_core_count()
    configure_cpu_thread_env(num_threads)

    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    texts = build_benchmark_texts(args.num_samples)
    power_backend = resolve_power_backend(args.power_backend, "npu")
    multi_case = len(seq_lens) > 1

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
    print(f"Benchmark mode: {args.benchmark_mode}", flush=True)
    print(f"Topology policy: {args.topology_policy}", flush=True)

    csv_rows = []
    study_id = args.study_id or ""
    for seq_len in seq_lens:
        cooldown_before_benchmark(
            args,
            mode="npu",
            label=f"NPU seq_len={seq_len}",
        )
        print(f"Starting NPU benchmark: seq_len={seq_len}", flush=True)
        topology, topology_selection = resolve_topology(args, seq_len, texts)
        print(
            f"Selected topology: seq_len={seq_len} topology={topology_id(topology)} "
            f"compute_tiles={topology.compute_tile_count} "
            f"utilization={topology.utilization_fraction:.3f}",
            flush=True,
        )
        config = load_encoder_pipeline_config(config_file_path, seq_len)
        apply_topology_to_config(config, topology)
        power_log_path = (
            derive_power_log_path(
                args.power_log_path,
                f"seq{seq_len}" if multi_case else None,
            )
            if power_backend != "none"
            else None
        )
        result = benchmark_with_config(
            weights_file_path=weights_file_path,
            config_file_path=config_file_path,
            config=config,
            seq_len=seq_len,
            texts=texts,
            warmup_runs=args.warmup_runs,
            runs_per_sample=args.runs_per_sample,
            benchmark_mode=args.benchmark_mode,
            topology=topology,
            power_backend=power_backend,
            power_interval_sec=args.power_interval_sec,
            power_log_path=power_log_path,
        )
        result["topology_selection_time_ms"] = topology_selection[
            "topology_selection_time_ms"
        ]
        result["topology_cache_status"] = topology_selection["topology_cache_status"]
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
                "study_id": study_id,
                "benchmark_mode": result["benchmark_mode"],
                "execution_mode": result["execution_mode"],
                "seq_len": seq_len,
                "num_threads": num_threads,
                "dtype": result["dtype"],
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
                "topology_id": result["topology_id"],
                "parallel_seq": result["parallel_seq"],
                "parallel_heads": result["parallel_heads"],
                "parallel_ffn": result["parallel_ffn"],
                "compute_tile_count": str(result["compute_tile_count"]),
                "compute_tile_utilization_fraction": (
                    f"{result['compute_tile_utilization_fraction']:.6f}"
                ),
                "compile_setup_time_ms": f"{result['compile_setup_time_ms']:.6f}",
                "topology_selection_time_ms": (
                    f"{result['topology_selection_time_ms']:.6f}"
                ),
                "topology_cache_status": result["topology_cache_status"],
                "cached_steady_state_avg_latency_ms": (
                    f"{result['cached_steady_state_avg_latency_ms']:.6f}"
                ),
                "avg_embedding_latency_ms": (
                    f"{result['avg_embedding_latency_ms']:.6f}"
                ),
                "avg_qkv_projection_latency_ms": (
                    f"{result['avg_qkv_projection_latency_ms']:.6f}"
                ),
                "avg_encoder_pipeline_latency_ms": (
                    f"{result['avg_encoder_pipeline_latency_ms']:.6f}"
                ),
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
