#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .cases import ExecutionMode, ReconfigurationWorkload
from .power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
    resolve_requested_power_backend,
)

if TYPE_CHECKING:
    from iron.common import AIEContext

GEMM_REL_TOL = 0.1
GEMM_ABS_TOL = 0.5
GEMM_ERROR_THRESHOLD = 0.05
DEFAULT_POWER_SAMPLE_INTERVAL_SEC = 0.1
DEFAULT_QUIESCENT_BASELINE_DURATION_SEC = 0.5
_CONTEXT_COUNTER = count()


@dataclass
class PreparedSequenceData:
    query_block_size: int
    block_count: int
    hidden_block: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    attn_output_weight: torch.Tensor
    ffn_up_weight: torch.Tensor
    ffn_down_weight: torch.Tensor
    q_proj_expected: torch.Tensor
    k_proj_expected: torch.Tensor
    v_proj_expected: torch.Tensor
    q_heads_block_by_head: tuple[torch.Tensor, ...]
    k_heads_transposed_by_head: tuple[torch.Tensor, ...]
    v_heads_by_head: tuple[torch.Tensor, ...]
    attn_scores_expected_by_head: tuple[torch.Tensor, ...]
    attention_probs_by_head: tuple[torch.Tensor, ...]
    attn_output_expected_by_head: tuple[torch.Tensor, ...]
    attn_scores_runlist_input_a: torch.Tensor
    attn_scores_runlist_input_b: torch.Tensor
    attn_scores_expected_runlist: torch.Tensor
    attn_output_runlist_input_a: torch.Tensor
    attn_output_runlist_input_b: torch.Tensor
    attn_output_expected_runlist: torch.Tensor
    out_proj_input_block: torch.Tensor
    out_proj_expected: torch.Tensor
    ffn_input_block: torch.Tensor
    up_proj_expected: torch.Tensor
    gelu_block: torch.Tensor
    down_proj_expected: torch.Tensor


@dataclass
class ModeRuntime:
    context: Any
    forward_once: callable
    compile_setup_time_ms: float | None
    npu_dispatch_count: int
    npu_unique_instruction_binary_count: int
    npu_unique_xclbin_count: int


def _new_benchmark_context(scope: str):
    from iron.common import AIEContext

    context = AIEContext()
    sanitized_scope = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in scope
    )
    context.build_dir = (
        Path.cwd()
        / "build"
        / "transformer_layer_new_reconfiguration_overhead"
        / f"{next(_CONTEXT_COUNTER):04d}_{sanitized_scope}"
    )
    return context


def resolve_query_block_size(seq_len: int) -> int:
    if seq_len >= 8192:
        return 256
    return seq_len


def resolve_attn_scores_partition_n(seq_len: int) -> int:
    if seq_len < 8192:
        return 1
    if seq_len % 4096 != 0:
        raise ValueError(
            "Long-sequence offload attention-score partitioning requires "
            f"seq_len divisible by 4096; got seq_len={seq_len}"
        )
    return seq_len // 4096


def block_count_for_seq_len(seq_len: int) -> int:
    block_size = resolve_query_block_size(seq_len)
    return (seq_len + block_size - 1) // block_size


def dispatch_count_for_mode(
    execution_mode: ExecutionMode,
    workload: ReconfigurationWorkload,
) -> int:
    block_count = block_count_for_seq_len(workload.seq_len)
    if execution_mode == "offload_gemm_sequence":
        return ((2 * workload.num_attention_heads) + 6) * block_count
    if execution_mode == "runlist_gemm_sequence":
        return 8 * block_count
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def artifact_counts_for_mode(execution_mode: ExecutionMode) -> tuple[int, int]:
    if execution_mode == "offload_gemm_sequence":
        return (8, 1)
    if execution_mode == "runlist_gemm_sequence":
        return (5, 5)
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _strip_keys(
    config: dict[str, object],
    *keys: str,
) -> dict[str, object]:
    return {key: value for key, value in config.items() if key not in set(keys)}


def _default_offload_gemm_config(
    workload: ReconfigurationWorkload,
) -> dict[str, object]:
    if workload.seq_len <= 128:
        return {
            "tile_m": 16,
            "tile_k": 64,
            "tile_n": 8,
            "num_aie_columns": 8,
            "b_col_maj": False,
            "c_col_maj": False,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        }
    return {
        "tile_m": 64,
        "tile_k": 64,
        "tile_n": 16,
        "num_aie_columns": 8,
        "b_col_maj": False,
        "c_col_maj": False,
        "prio_accuracy": False,
        "emulate_bf16_mmul_with_bfp16": True,
    }


def _default_runlist_gemm_config(
    workload: ReconfigurationWorkload,
) -> dict[str, dict[str, object]]:
    num_heads = workload.num_attention_heads
    return {
        "qkvo_proj": {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "attn_scores": {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "num_aie_columns": 8,
            "batch_A": (num_heads, 1),
            "batch_B": (num_heads, 1),
            "batch_C": (num_heads, 0),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "attn_output": {
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 16,
            "num_aie_columns": 4,
            "batch_A": (num_heads, 0),
            "batch_B": (num_heads, 1),
            "batch_C": (num_heads, 1),
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "up_proj": {
            "tile_m": 64,
            "tile_k": 48,
            "tile_n": 96,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
        "down_proj": {
            "tile_m": 64,
            "tile_k": 96,
            "tile_n": 48,
            "num_aie_columns": 8,
            "prio_accuracy": False,
            "emulate_bf16_mmul_with_bfp16": True,
        },
    }


def resolve_mode_operator_config(
    execution_mode: ExecutionMode,
    workload: ReconfigurationWorkload,
    operator_config: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    block_size = resolve_query_block_size(workload.seq_len)
    head_dim = workload.attention_head_size

    if execution_mode == "offload_gemm_sequence":
        shared_gemm = {
            **_default_offload_gemm_config(workload),
            **dict(operator_config["shared_gemm"]),
        }
        return {
            "q_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **shared_gemm,
            },
            "k_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **shared_gemm,
            },
            "v_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **shared_gemm,
            },
            "attn_scores": {
                "M": block_size,
                "K": head_dim,
                "N": workload.seq_len,
                "partition_N": resolve_attn_scores_partition_n(workload.seq_len),
                **shared_gemm,
            },
            "attn_output": {
                "M": block_size,
                "K": workload.seq_len,
                "N": head_dim,
                **shared_gemm,
            },
            "out_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **shared_gemm,
            },
            "ffn_up": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.intermediate_size,
                **shared_gemm,
            },
            "ffn_down": {
                "M": block_size,
                "K": workload.intermediate_size,
                "N": workload.hidden_size,
                **shared_gemm,
            },
        }

    if execution_mode == "runlist_gemm_sequence":
        defaults = _default_runlist_gemm_config(workload)
        qkvo = {
            **defaults["qkvo_proj"],
            **_strip_keys(operator_config["qkvo_proj"], "M", "K", "N"),
        }
        attn_scores = {
            **defaults["attn_scores"],
            **_strip_keys(operator_config["attn_scores"], "M", "K", "N"),
        }
        attn_output = {
            **defaults["attn_output"],
            **_strip_keys(operator_config["attn_output"], "M", "K", "N"),
        }
        up_proj = {
            **defaults["up_proj"],
            **_strip_keys(operator_config["up_proj"], "M", "K", "N"),
        }
        down_proj = {
            **defaults["down_proj"],
            **_strip_keys(operator_config["down_proj"], "M", "K", "N"),
        }
        return {
            "q_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **qkvo,
            },
            "k_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **qkvo,
            },
            "v_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **qkvo,
            },
            "attn_scores": {
                "M": block_size,
                "K": head_dim,
                "N": workload.seq_len,
                **attn_scores,
            },
            "attn_output": {
                "M": block_size,
                "K": workload.seq_len,
                "N": head_dim,
                **attn_output,
            },
            "out_proj": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
                **qkvo,
            },
            "ffn_up": {
                "M": block_size,
                "K": workload.hidden_size,
                "N": workload.intermediate_size,
                **up_proj,
            },
            "ffn_down": {
                "M": block_size,
                "K": workload.intermediate_size,
                "N": workload.hidden_size,
                **down_proj,
            },
        }

    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _summarize_latencies(latencies_sec: list[float]) -> dict[str, float | int | None]:
    measured_inference_count = len(latencies_sec)
    timed_total_sec = sum(latencies_sec)
    avg_latency_ms = None
    if measured_inference_count:
        avg_latency_ms = (timed_total_sec / measured_inference_count) * 1000.0
    return {
        "measured_inference_count": measured_inference_count,
        "timed_total_sec": timed_total_sec,
        "avg_latency_ms": avg_latency_ms,
    }


def _count_mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    mismatches = ~torch.isclose(
        actual.float(),
        expected.float(),
        rtol=GEMM_REL_TOL,
        atol=GEMM_ABS_TOL,
    )
    return int(torch.count_nonzero(mismatches).item())


def _validate_stage_outputs(
    outputs: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    validation_error_count = 0
    failing_stage = ""

    for stage_name, actual, expected in outputs:
        if tuple(actual.shape) != tuple(expected.shape):
            return {
                "validation_error_count": validation_error_count,
                "run_status": "failed_validation",
                "failure_message": (
                    f"{stage_name} output shape {tuple(actual.shape)} does not match "
                    f"expected {tuple(expected.shape)}"
                ),
            }

        error_count = _count_mismatches(actual, expected)
        validation_error_count += error_count
        max_acceptable_errors = int(expected.numel() * GEMM_ERROR_THRESHOLD)
        if error_count > max_acceptable_errors and not failing_stage:
            failing_stage = stage_name

    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if not failing_stage else "failed_validation",
        "failure_message": (
            ""
            if not failing_stage
            else f"{failing_stage} validation errors exceeded threshold"
        ),
    }


def _measure_power(
    forward_once,
    *,
    requested_power_backend: str,
    runs_per_sample: int,
    avg_latency_ms: float | None,
    timed_total_sec: float,
    power_sample_interval_sec: float,
) -> dict[str, object]:
    resolved_power_backend = resolve_requested_power_backend(requested_power_backend)
    if resolved_power_backend == "none":
        stats = empty_power_stats()
        stats["power_backend"] = "none"
        return stats

    avg_iteration_sec = None
    if avg_latency_ms is not None:
        avg_iteration_sec = avg_latency_ms / 1000.0
    power_probe_runs = resolve_power_probe_runs(
        avg_iteration_sec=avg_iteration_sec,
        baseline_runs=runs_per_sample,
        min_measurement_duration_sec=0.25,
    )
    estimated_window_sec = (
        None if avg_iteration_sec is None else avg_iteration_sec * power_probe_runs
    )
    sample_interval_sec = resolve_power_sample_interval_sec(
        requested_interval_sec=power_sample_interval_sec,
        estimated_timed_window_sec=estimated_window_sec,
        min_interval_sec=DEFAULT_POWER_SAMPLE_INTERVAL_SEC,
    )

    try:
        with create_power_monitor(
            power_backend=resolved_power_backend,
            sample_interval_sec=sample_interval_sec,
            quiescent_baseline_duration_sec=DEFAULT_QUIESCENT_BASELINE_DURATION_SEC,
            estimated_timed_window_sec=estimated_window_sec,
        ) as power_monitor:
            started = time.perf_counter()
            for _ in range(power_probe_runs):
                forward_once()
                time.sleep(0)
            elapsed_sec = time.perf_counter() - started
        stats = power_monitor.stats(elapsed_sec)
        if stats.get("avg_power_w") is not None:
            stats["energy_j"] = float(stats["avg_power_w"]) * timed_total_sec
        stats["power_backend"] = resolved_power_backend
        return stats
    except Exception:
        if requested_power_backend == "auto":
            stats = empty_power_stats()
            stats["power_backend"] = "none"
            return stats
        raise


def _prepare_sequence_data(
    workload: ReconfigurationWorkload,
    *,
    seed: int,
) -> PreparedSequenceData:
    from iron.applications.transformer_layer_new.pattern.reference import (
        derive_offload_inputs,
        generate_golden_reference,
    )

    reference = generate_golden_reference(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        seed=seed,
        include_output=False,
        include_attention_mask=False,
    )
    hidden_states = reference["input"]
    weights = reference["weights"]
    if not isinstance(hidden_states, torch.Tensor):
        raise RuntimeError("reference generator did not return an input tensor")
    if not isinstance(weights, dict):
        raise RuntimeError("reference generator did not return a weight mapping")

    derived = derive_offload_inputs(reference, num_heads=workload.num_attention_heads)
    q_heads_full = derived["q"]
    k_heads_full = derived["k"]
    v_heads_full = derived["v"]
    block_size = resolve_query_block_size(workload.seq_len)
    block_count = block_count_for_seq_len(workload.seq_len)
    hidden_block = hidden_states[:block_size].contiguous()

    q_heads_block = q_heads_full[:, :block_size, :].contiguous()
    k_heads_transposed = k_heads_full.transpose(-2, -1).contiguous()
    attn_scores_expected = torch.matmul(q_heads_block, k_heads_transposed)
    attention_probs = F.softmax(
        attn_scores_expected.float() * (workload.attention_head_size**-0.5),
        dim=-1,
    ).to(q_heads_block.dtype)
    attn_output_by_head = tuple(
        torch.matmul(attention_probs[head], v_heads_full[head])
        for head in range(workload.num_attention_heads)
    )
    attn_output_expected_runlist = torch.stack(attn_output_by_head, dim=1).contiguous()

    q_proj_expected = (
        q_heads_block.transpose(0, 1)
        .contiguous()
        .view(block_size, workload.hidden_size)
    )
    k_proj_expected = (
        k_heads_full[:, :block_size, :]
        .transpose(0, 1)
        .contiguous()
        .view(block_size, workload.hidden_size)
    )
    v_proj_expected = (
        v_heads_full[:, :block_size, :]
        .transpose(0, 1)
        .contiguous()
        .view(block_size, workload.hidden_size)
    )
    out_proj_input_block = attn_output_expected_runlist.view(
        block_size, workload.hidden_size
    ).contiguous()
    out_proj_expected = torch.matmul(
        out_proj_input_block, weights["attn_output_weight"]
    )
    ffn_input_block = F.layer_norm(
        out_proj_expected + hidden_block,
        (workload.hidden_size,),
        weights["ln1_weight"],
        None,
    )
    up_proj_expected = torch.matmul(ffn_input_block, weights["ffn_up_weight"])
    gelu_block = F.gelu(up_proj_expected)
    down_proj_expected = torch.matmul(gelu_block, weights["ffn_down_weight"])

    return PreparedSequenceData(
        query_block_size=block_size,
        block_count=block_count,
        hidden_block=hidden_block,
        q_weight=weights["q_weight"],
        k_weight=weights["k_weight"],
        v_weight=weights["v_weight"],
        attn_output_weight=weights["attn_output_weight"],
        ffn_up_weight=weights["ffn_up_weight"],
        ffn_down_weight=weights["ffn_down_weight"],
        q_proj_expected=q_proj_expected,
        k_proj_expected=k_proj_expected,
        v_proj_expected=v_proj_expected,
        q_heads_block_by_head=tuple(
            q_heads_block[head].contiguous()
            for head in range(workload.num_attention_heads)
        ),
        k_heads_transposed_by_head=tuple(
            k_heads_transposed[head].contiguous()
            for head in range(workload.num_attention_heads)
        ),
        v_heads_by_head=tuple(
            v_heads_full[head].contiguous()
            for head in range(workload.num_attention_heads)
        ),
        attn_scores_expected_by_head=tuple(
            attn_scores_expected[head].contiguous()
            for head in range(workload.num_attention_heads)
        ),
        attention_probs_by_head=tuple(
            attention_probs[head].contiguous()
            for head in range(workload.num_attention_heads)
        ),
        attn_output_expected_by_head=tuple(
            output.contiguous() for output in attn_output_by_head
        ),
        attn_scores_runlist_input_a=q_heads_block.permute(1, 0, 2).contiguous(),
        attn_scores_runlist_input_b=k_heads_full.permute(2, 0, 1).contiguous(),
        attn_scores_expected_runlist=attn_scores_expected.contiguous(),
        attn_output_runlist_input_a=attention_probs.contiguous(),
        attn_output_runlist_input_b=v_heads_full.permute(1, 0, 2).contiguous(),
        attn_output_expected_runlist=attn_output_expected_runlist,
        out_proj_input_block=out_proj_input_block,
        out_proj_expected=out_proj_expected,
        ffn_input_block=ffn_input_block,
        up_proj_expected=up_proj_expected,
        gelu_block=gelu_block,
        down_proj_expected=down_proj_expected,
    )


def _compile_context(context: AIEContext) -> float | None:
    started = time.perf_counter()
    context.compile_all()
    context.prepare_runtime()
    return (time.perf_counter() - started) * 1000.0


def _configure_runlist_stage_artifact(
    xclbin_artifact,
    *,
    instance_name: str,
    kernel_id: int,
    xclbin_input=None,
) -> None:
    if xclbin_input is not None:
        xclbin_artifact.xclbin_input = xclbin_input
        xclbin_artifact.depends += [xclbin_input]
    xclbin_artifact.extra_flags += [
        f"--xclbin-instance-name={instance_name}",
        f"--xclbin-kernel-id={hex(kernel_id)}",
    ]
    xclbin_artifact.kernel_name = instance_name


def _build_offload_runtime(
    workload: ReconfigurationWorkload,
    *,
    resolved_config: dict[str, dict[str, object]],
    prepared: PreparedSequenceData,
) -> ModeRuntime:
    from iron.operators.gemm.op import AIEGEMM

    context = _new_benchmark_context("offload_gemm_sequence")

    q_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["q_proj"]
    )
    k_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["k_proj"]
    )
    v_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["v_proj"]
    )
    attn_scores = AIEGEMM(context=context, **resolved_config["attn_scores"])
    attn_output = AIEGEMM(context=context, **resolved_config["attn_output"])
    out_proj = AIEGEMM(
        context=context,
        use_static_weight=True,
        **resolved_config["out_proj"],
    )
    ffn_up = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["ffn_up"]
    )
    ffn_down = AIEGEMM(
        context=context,
        use_static_weight=True,
        **resolved_config["ffn_down"],
    )
    gemm_ops = {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "attn_scores": attn_scores,
        "attn_output": attn_output,
        "out_proj": out_proj,
        "ffn_up": ffn_up,
        "ffn_down": ffn_down,
    }

    q_proj.weight = prepared.q_weight.T.contiguous()
    k_proj.weight = prepared.k_weight.T.contiguous()
    v_proj.weight = prepared.v_weight.T.contiguous()
    out_proj.weight = prepared.attn_output_weight.T.contiguous()
    ffn_up.weight = prepared.ffn_up_weight.T.contiguous()
    ffn_down.weight = prepared.ffn_down_weight.T.contiguous()

    shared_xclbin = out_proj.get_runtime_xclbin_artifact(
        prefix="reconfig_offload_runtime_"
    )
    shared_xclbin.kernel_name = "reconfig_offload_runtime"
    for workload_name, gemm_op in gemm_ops.items():
        insts_artifact = gemm_op.get_insts_artifact(
            prefix=f"reconfig_offload_{workload_name}_",
            xclbin_input=shared_xclbin,
            kernel_name=shared_xclbin.kernel_name,
        )
        gemm_op.bind_artifacts(
            shared_xclbin,
            insts_artifact,
            runtime_xclbin_artifact=shared_xclbin,
            runtime_kernel_name=shared_xclbin.kernel_name,
        )

    compile_setup_time_ms = _compile_context(context)

    def forward_once() -> dict[str, object]:
        q_last = None
        k_last = None
        v_last = None
        attn_scores_last = ()
        attn_output_last = ()
        out_last = None
        up_last = None
        down_last = None
        for _ in range(prepared.block_count):
            q_last = q_proj(prepared.hidden_block)
            k_last = k_proj(prepared.hidden_block)
            v_last = v_proj(prepared.hidden_block)
            attn_scores_last = tuple(
                attn_scores(
                    prepared.q_heads_block_by_head[head],
                    prepared.k_heads_transposed_by_head[head],
                )
                for head in range(workload.num_attention_heads)
            )
            attn_output_last = tuple(
                attn_output(
                    prepared.attention_probs_by_head[head],
                    prepared.v_heads_by_head[head],
                )
                for head in range(workload.num_attention_heads)
            )
            out_last = out_proj(prepared.out_proj_input_block)
            up_last = ffn_up(prepared.ffn_input_block)
            down_last = ffn_down(prepared.gelu_block)
        return {
            "q_proj": q_last,
            "k_proj": k_last,
            "v_proj": v_last,
            "attn_scores": attn_scores_last,
            "attn_output": attn_output_last,
            "out_proj": out_last,
            "ffn_up": up_last,
            "ffn_down": down_last,
        }

    inst_count, xclbin_count = artifact_counts_for_mode("offload_gemm_sequence")
    return ModeRuntime(
        context=context,
        forward_once=forward_once,
        compile_setup_time_ms=compile_setup_time_ms,
        npu_dispatch_count=dispatch_count_for_mode("offload_gemm_sequence", workload),
        npu_unique_instruction_binary_count=inst_count,
        npu_unique_xclbin_count=xclbin_count,
    )


def _build_runlist_runtime(
    workload: ReconfigurationWorkload,
    *,
    resolved_config: dict[str, dict[str, object]],
    prepared: PreparedSequenceData,
) -> ModeRuntime:
    from iron.operators.gemm.op import AIEGEMM

    context = _new_benchmark_context("runlist_gemm_sequence")

    q_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["q_proj"]
    )
    k_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["k_proj"]
    )
    v_proj = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["v_proj"]
    )
    attn_scores = AIEGEMM(context=context, **resolved_config["attn_scores"])
    attn_output = AIEGEMM(context=context, **resolved_config["attn_output"])
    out_proj = AIEGEMM(
        context=context,
        use_static_weight=True,
        **resolved_config["out_proj"],
    )
    ffn_up = AIEGEMM(
        context=context, use_static_weight=True, **resolved_config["ffn_up"]
    )
    ffn_down = AIEGEMM(
        context=context,
        use_static_weight=True,
        **resolved_config["ffn_down"],
    )

    q_proj.weight = prepared.q_weight.T.contiguous()
    k_proj.weight = prepared.k_weight.T.contiguous()
    v_proj.weight = prepared.v_weight.T.contiguous()
    out_proj.weight = prepared.attn_output_weight.T.contiguous()
    ffn_up.weight = prepared.ffn_up_weight.T.contiguous()
    ffn_down.weight = prepared.ffn_down_weight.T.contiguous()

    qkvo_xclbin, qkvo_insts = q_proj.get_artifacts(prefix="reconfig_runlist_qkvo_")
    _configure_runlist_stage_artifact(
        qkvo_xclbin,
        instance_name="reconfig_qkvo_proj",
        kernel_id=0x801,
    )
    attn_scores_xclbin, attn_scores_insts = attn_scores.get_artifacts(
        prefix="reconfig_runlist_attn_scores_"
    )
    _configure_runlist_stage_artifact(
        attn_scores_xclbin,
        instance_name="reconfig_attn_scores",
        kernel_id=0x802,
        xclbin_input=qkvo_xclbin,
    )
    attn_output_xclbin, attn_output_insts = attn_output.get_artifacts(
        prefix="reconfig_runlist_attn_output_"
    )
    _configure_runlist_stage_artifact(
        attn_output_xclbin,
        instance_name="reconfig_attn_output",
        kernel_id=0x803,
        xclbin_input=attn_scores_xclbin,
    )
    up_proj_xclbin, up_proj_insts = ffn_up.get_artifacts(
        prefix="reconfig_runlist_up_proj_"
    )
    _configure_runlist_stage_artifact(
        up_proj_xclbin,
        instance_name="reconfig_up_proj",
        kernel_id=0x804,
        xclbin_input=attn_output_xclbin,
    )
    down_proj_xclbin, down_proj_insts = ffn_down.get_artifacts(
        prefix="reconfig_runlist_down_proj_"
    )
    _configure_runlist_stage_artifact(
        down_proj_xclbin,
        instance_name="reconfig_down_proj",
        kernel_id=0x805,
        xclbin_input=up_proj_xclbin,
    )
    combined_xclbin = down_proj_xclbin

    q_proj.bind_artifacts(
        qkvo_xclbin,
        qkvo_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=qkvo_xclbin.kernel_name,
    )
    k_proj.bind_artifacts(
        qkvo_xclbin,
        qkvo_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=qkvo_xclbin.kernel_name,
    )
    v_proj.bind_artifacts(
        qkvo_xclbin,
        qkvo_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=qkvo_xclbin.kernel_name,
    )
    out_proj.bind_artifacts(
        qkvo_xclbin,
        qkvo_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=qkvo_xclbin.kernel_name,
    )
    attn_scores.bind_artifacts(
        attn_scores_xclbin,
        attn_scores_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=attn_scores_xclbin.kernel_name,
    )
    attn_output.bind_artifacts(
        attn_output_xclbin,
        attn_output_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=attn_output_xclbin.kernel_name,
    )
    ffn_up.bind_artifacts(
        up_proj_xclbin,
        up_proj_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=up_proj_xclbin.kernel_name,
    )
    ffn_down.bind_artifacts(
        down_proj_xclbin,
        down_proj_insts,
        runtime_xclbin_artifact=combined_xclbin,
        runtime_kernel_name=down_proj_xclbin.kernel_name,
    )

    compile_setup_time_ms = _compile_context(context)

    def forward_once() -> dict[str, object]:
        q_last = None
        k_last = None
        v_last = None
        attn_scores_last = None
        attn_output_last = None
        out_last = None
        up_last = None
        down_last = None
        for _ in range(prepared.block_count):
            q_last = q_proj(prepared.hidden_block)
            k_last = k_proj(prepared.hidden_block)
            v_last = v_proj(prepared.hidden_block)
            attn_scores_last = attn_scores(
                prepared.attn_scores_runlist_input_a,
                prepared.attn_scores_runlist_input_b,
            )
            attn_output_last = attn_output(
                prepared.attn_output_runlist_input_a,
                prepared.attn_output_runlist_input_b,
            )
            out_last = out_proj(prepared.out_proj_input_block)
            up_last = ffn_up(prepared.ffn_input_block)
            down_last = ffn_down(prepared.gelu_block)
        return {
            "q_proj": q_last,
            "k_proj": k_last,
            "v_proj": v_last,
            "attn_scores": attn_scores_last,
            "attn_output": attn_output_last,
            "out_proj": out_last,
            "ffn_up": up_last,
            "ffn_down": down_last,
        }

    inst_count, xclbin_count = artifact_counts_for_mode("runlist_gemm_sequence")
    return ModeRuntime(
        context=context,
        forward_once=forward_once,
        compile_setup_time_ms=compile_setup_time_ms,
        npu_dispatch_count=dispatch_count_for_mode("runlist_gemm_sequence", workload),
        npu_unique_instruction_binary_count=inst_count,
        npu_unique_xclbin_count=xclbin_count,
    )


def _stage_outputs_for_validation(
    execution_mode: ExecutionMode,
    outputs: dict[str, object],
    prepared: PreparedSequenceData,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    entries: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("q_proj", outputs["q_proj"], prepared.q_proj_expected),  # type: ignore[arg-type]
        ("k_proj", outputs["k_proj"], prepared.k_proj_expected),  # type: ignore[arg-type]
        ("v_proj", outputs["v_proj"], prepared.v_proj_expected),  # type: ignore[arg-type]
        ("out_proj", outputs["out_proj"], prepared.out_proj_expected),  # type: ignore[arg-type]
        ("ffn_up", outputs["ffn_up"], prepared.up_proj_expected),  # type: ignore[arg-type]
        ("ffn_down", outputs["ffn_down"], prepared.down_proj_expected),  # type: ignore[arg-type]
    ]

    if execution_mode == "runlist_gemm_sequence":
        entries.append(
            (
                "attn_scores",
                outputs["attn_scores"],  # type: ignore[arg-type]
                prepared.attn_scores_expected_runlist,
            )
        )
        entries.append(
            (
                "attn_output",
                outputs["attn_output"],  # type: ignore[arg-type]
                prepared.attn_output_expected_runlist,
            )
        )
        return entries

    attn_scores_outputs = outputs["attn_scores"]  # type: ignore[assignment]
    attn_output_outputs = outputs["attn_output"]  # type: ignore[assignment]
    for head, actual in enumerate(attn_scores_outputs):
        entries.append(
            (
                f"attn_scores_head{head}",
                actual,
                prepared.attn_scores_expected_by_head[head],
            )
        )
    for head, actual in enumerate(attn_output_outputs):
        entries.append(
            (
                f"attn_output_head{head}",
                actual,
                prepared.attn_output_expected_by_head[head],
            )
        )
    return entries


def _cleanup_context(context) -> None:
    try:
        context.reset_runtime()
    except Exception:
        pass


def benchmark_mode(
    execution_mode: ExecutionMode,
    workload: ReconfigurationWorkload,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    power_backend: str,
    operator_config: dict[str, dict[str, object]],
    power_sample_interval_sec: float = DEFAULT_POWER_SAMPLE_INTERVAL_SEC,
) -> dict[str, object]:
    resolved_config = resolve_mode_operator_config(
        execution_mode,
        workload,
        operator_config,
    )
    prepared = _prepare_sequence_data(workload, seed=seed)
    runtime = (
        _build_offload_runtime(
            workload,
            resolved_config=resolved_config,
            prepared=prepared,
        )
        if execution_mode == "offload_gemm_sequence"
        else _build_runlist_runtime(
            workload,
            resolved_config=resolved_config,
            prepared=prepared,
        )
    )
    try:
        for _ in range(warmup_runs):
            runtime.forward_once()

        latencies_sec: list[float] = []
        last_outputs = None
        for _ in range(runs_per_sample):
            started = time.perf_counter()
            last_outputs = runtime.forward_once()
            latencies_sec.append(time.perf_counter() - started)

        summary = _summarize_latencies(latencies_sec)
        power_stats = _measure_power(
            runtime.forward_once,
            requested_power_backend=power_backend,
            runs_per_sample=runs_per_sample,
            avg_latency_ms=summary["avg_latency_ms"],  # type: ignore[arg-type]
            timed_total_sec=float(summary["timed_total_sec"]),
            power_sample_interval_sec=power_sample_interval_sec,
        )

        if last_outputs is None:
            return {
                **summary,
                "compile_setup_time_ms": runtime.compile_setup_time_ms,
                "power_backend": power_stats.get("power_backend"),
                "avg_power_w": power_stats.get("avg_power_w"),
                "max_power_w": power_stats.get("max_power_w"),
                "energy_j": power_stats.get("energy_j"),
                "power_sample_count": power_stats.get("power_sample_count"),
                "npu_dispatch_count": runtime.npu_dispatch_count,
                "npu_unique_instruction_binary_count": (
                    runtime.npu_unique_instruction_binary_count
                ),
                "npu_unique_xclbin_count": runtime.npu_unique_xclbin_count,
                "process_model": "in_process",
                "validation_error_count": 0,
                "run_status": "failed_exception",
                "failure_message": "no output tensor was produced",
            }

        validation = _validate_stage_outputs(
            _stage_outputs_for_validation(execution_mode, last_outputs, prepared)
        )
        return {
            **summary,
            "compile_setup_time_ms": runtime.compile_setup_time_ms,
            "power_backend": power_stats.get("power_backend"),
            "avg_power_w": power_stats.get("avg_power_w"),
            "max_power_w": power_stats.get("max_power_w"),
            "energy_j": power_stats.get("energy_j"),
            "power_sample_count": power_stats.get("power_sample_count"),
            "npu_dispatch_count": runtime.npu_dispatch_count,
            "npu_unique_instruction_binary_count": (
                runtime.npu_unique_instruction_binary_count
            ),
            "npu_unique_xclbin_count": runtime.npu_unique_xclbin_count,
            "process_model": "in_process",
            **validation,
        }
    finally:
        _cleanup_context(runtime.context)


__all__ = [
    "artifact_counts_for_mode",
    "benchmark_mode",
    "block_count_for_seq_len",
    "dispatch_count_for_mode",
    "resolve_mode_operator_config",
    "resolve_query_block_size",
]
