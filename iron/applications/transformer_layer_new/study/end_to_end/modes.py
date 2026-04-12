#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zlib

import numpy as np
import torch

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.addnorm.reference import (
    generate_golden_reference as generate_addnorm_reference,
)
from iron.operators.causal_mask.op import AIECausalMask
from iron.operators.causal_mask.reference import (
    generate_golden_reference as generate_causal_mask_reference,
)
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.elementwise_add.reference import (
    generate_golden_reference as generate_elementwise_add_reference,
)
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.elementwise_mul.reference import (
    generate_golden_reference as generate_elementwise_mul_reference,
)
from iron.operators.ffn.op import AIEFFN
from iron.operators.ffn.reference import (
    generate_golden_reference as generate_ffn_reference,
)
from iron.operators.gelu.op import AIEGELU
from iron.operators.gelu.reference import (
    generate_golden_reference as generate_gelu_reference,
)
from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import (
    generate_golden_reference as generate_gemm_reference,
)
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.layer_norm.reference import (
    generate_golden_reference as generate_layer_norm_reference,
)
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.mha_out_proj.reference import (
    generate_golden_reference as generate_mha_out_proj_reference,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import (
    generate_golden_reference as generate_qkv_proj_reference,
)
from iron.operators.softmax.op import AIESoftmax
from iron.operators.softmax.reference import (
    generate_golden_reference as generate_softmax_reference,
)
from iron.operators.transpose.op import AIETranspose
from iron.operators.transpose.reference import (
    generate_golden_reference as generate_transpose_reference,
)

from iron.applications.transformer_layer_new.pattern.dataflow.op import (
    AIETransformerHybrid,
    resolve_hybrid_operator_config,
)
from iron.applications.transformer_layer_new.pattern.reference import (
    generate_golden_reference,
)
from iron.applications.transformer_layer_new.pattern.runlist.op import (
    AIETransformerRunlist,
    resolve_runlist_operator_config,
)

from ml_dtypes import bfloat16

from .cases import (
    EndToEndWorkload,
    ExecutionMode,
    canonical_execution_mode,
    effective_gflops_per_sec,
    effective_gflops_per_sec_per_watt,
)
from .power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
    resolve_requested_power_backend,
)
from ..npu_runtime_checks import require_npu_power_mode_turbo

GEMM_REL_TOL = 0.1
GEMM_ABS_TOL = 0.5
GEMM_ERROR_THRESHOLD = 0.05
BLOCK_REL_TOL = 4.0e-2
BLOCK_ABS_TOL = 1.5e-1
BLOCK_ERROR_THRESHOLD = 0.005
ADDNORM_REL_TOL = 0.1
ADDNORM_ABS_TOL = 0.1
EXACT_REL_TOL = 0.04
EXACT_ABS_TOL = 1e-6
LAYER_NORM_REL_TOL = 0.1
LAYER_NORM_ABS_TOL = 0.1
FINAL_REL_TOL = 0.1
FINAL_ABS_TOL = 0.5
FINAL_ERROR_THRESHOLD = 0.05
REFERENCE_VALIDATION_MAX_SEQ_LEN = 512
DEFAULT_POWER_SAMPLE_INTERVAL_SEC = 0.1
DEFAULT_QUIESCENT_BASELINE_DURATION_SEC = 0.5
DEFAULT_MIN_POWER_MEASUREMENT_DURATION_SEC = 1.0
_LONG_SEQ_CANDIDATE_SUBPROCESS_MIN_SEQ_LEN = 8192


def _config_scope_key(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    operator_config: dict[str, dict[str, object]] | None = None,
) -> str:
    resolved_config = resolve_mode_operator_config(
        execution_mode,
        workload,
        operator_config=operator_config,
    )
    payload = {
        "execution_mode": execution_mode,
        "workload_variant": workload.workload_variant,
        "seq_len": workload.seq_len,
        "hidden_size": workload.hidden_size,
        "intermediate_size": workload.intermediate_size,
        "num_attention_heads": workload.num_attention_heads,
        "operator_config": resolved_config,
    }
    checksum = zlib.crc32(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return f"cfg_{checksum:08x}"


def _candidate_scope_key(candidate_config: dict[str, object]) -> str:
    checksum = zlib.crc32(
        json.dumps(candidate_config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return f"cfg_{checksum:08x}"


def _isolated_candidate_scope(
    prefix: str,
    candidate_config: dict[str, object],
) -> str:
    return f"{prefix}_{_candidate_scope_key(candidate_config)}"


def _new_benchmark_context(scope: str) -> AIEContext:
    context = AIEContext()
    sanitized_scope = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in scope
    )
    context.build_dir = (
        Path.cwd() / "build" / "transformer_layer_new_end_to_end" / sanitized_scope
    )
    return context


def _fresh_benchmark_context_for_build_dir(build_dir: Path) -> AIEContext:
    context = AIEContext()
    context.build_dir = build_dir
    return context


def _is_retryable_linker_failure(exc: Exception) -> bool:
    message = str(exc)
    return "ld.lld: error:" in message and "symbol not found: core_" in message


def _preferred_subprocess_start_method() -> str:
    start_methods = multiprocessing.get_all_start_methods()
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file not in (None, "<stdin>") and "spawn" in start_methods:
        return "spawn"
    if "fork" in start_methods:
        return "fork"
    return start_methods[0]


def resolve_mode_operator_config(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    operator_config: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    execution_mode = canonical_execution_mode(str(execution_mode))
    if execution_mode == "hybrid":
        return resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config=operator_config,
        )
    if execution_mode == "runlist":
        return resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config=operator_config,
        )
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _build_operator(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    weights: dict[str, torch.Tensor],
    *,
    context: AIEContext,
    operator_config: dict[str, dict[str, object]] | None = None,
):
    execution_mode = canonical_execution_mode(str(execution_mode))
    common_kwargs = {
        "seq_len": workload.seq_len,
        "hidden_size": workload.hidden_size,
        "intermediate_size": workload.intermediate_size,
        "num_heads": workload.num_attention_heads,
        "ln1_weight": weights["ln1_weight"],
        "ln2_weight": weights["ln2_weight"],
        "workload_variant": workload.workload_variant,
        "operator_config": operator_config,
        "context": context,
    }
    if execution_mode == "hybrid":
        operator = AIETransformerHybrid(**common_kwargs)
    elif execution_mode == "runlist":
        operator = AIETransformerRunlist(**common_kwargs)
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    operator.attn_output_weight = weights["attn_output_weight"]
    operator.ffn_up_weight = weights["ffn_up_weight"]
    operator.ffn_down_weight = weights["ffn_down_weight"]
    operator.q_weight = weights["q_weight"]
    operator.k_weight = weights["k_weight"]
    operator.v_weight = weights["v_weight"]
    return operator


def _artifact_key(artifact: object) -> str | int | None:
    if artifact is None:
        return None
    path = getattr(artifact, "path", None)
    return str(path) if path is not None else id(artifact)


def _unique_suffix_artifact_count(operator, suffix: str) -> int:
    keys = {
        _artifact_key(value)
        for name, value in vars(operator).items()
        if name.endswith(suffix) and value is not None
    }
    return len({key for key in keys if key is not None})


def _metadata_for_operator(
    execution_mode: ExecutionMode,
    operator,
    workload: EndToEndWorkload,
    *,
    compile_setup_time_ms: float | None,
) -> dict[str, object]:
    if execution_mode == "runlist" and getattr(
        operator, "use_long_seq_fallback", False
    ):
        query_block_size = int(getattr(operator, "query_block_size", workload.seq_len))
        block_count = (workload.seq_len + query_block_size - 1) // query_block_size
        attention_dispatches = 2 * workload.num_attention_heads * block_count
        extra_ops = (
            getattr(operator, "long_attn_scores_gemm", None),
            getattr(operator, "long_attn_output_gemm", None),
        )
        extra_insts = {
            _artifact_key(getattr(op, "insts_artifact", None))
            for op in extra_ops
            if op is not None
        }
        extra_xclbins = {
            _artifact_key(getattr(op, "xclbin_artifact", None))
            for op in extra_ops
            if op is not None
        }
        base_insts = {
            _artifact_key(value)
            for name, value in vars(operator).items()
            if name.endswith("_insts") and value is not None
        }
        base_xclbins = {
            _artifact_key(value)
            for name, value in vars(operator).items()
            if name.endswith("_xclbin") and value is not None
        }
        return {
            "compile_setup_time_ms": compile_setup_time_ms,
            "npu_dispatch_count": len(getattr(operator, "runlist", ()))
            + attention_dispatches,
            "npu_unique_instruction_binary_count": len(
                {key for key in (base_insts | extra_insts) if key is not None}
            ),
            "npu_unique_xclbin_count": len(
                {key for key in (base_xclbins | extra_xclbins) if key is not None}
            ),
            "process_model": "in_process",
        }

    return {
        "compile_setup_time_ms": compile_setup_time_ms,
        "npu_dispatch_count": len(getattr(operator, "runlist", ())),
        "npu_unique_instruction_binary_count": _unique_suffix_artifact_count(
            operator, "_insts"
        ),
        "npu_unique_xclbin_count": _unique_suffix_artifact_count(operator, "_xclbin"),
        "process_model": "in_process",
    }


def _summarize_latencies(latencies_sec: list[float]) -> dict[str, float | int | None]:
    measured_inference_count = len(latencies_sec)
    timed_total_sec = sum(latencies_sec)
    avg_latency_ms = None
    if measured_inference_count:
        avg_latency_ms = (timed_total_sec / measured_inference_count) * 1000.0
    return {
        "timed_total_sec": timed_total_sec,
        "measured_inference_count": measured_inference_count,
        "avg_latency_ms": avg_latency_ms,
    }


def _threshold_validation_result(
    buffer_error_counts: dict[str, int],
    *,
    max_acceptable_errors: int | None = None,
    skip_validation: bool = False,
) -> dict[str, object]:
    validation_error_count = sum(buffer_error_counts.values())
    if skip_validation:
        return {
            "validation_error_count": validation_error_count,
            "run_status": "passed",
            "failure_message": "",
        }

    if max_acceptable_errors is None:
        passed = validation_error_count == 0
    else:
        passed = all(
            error_count <= max_acceptable_errors
            for error_count in buffer_error_counts.values()
        )

    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if passed else "failed_validation",
        "failure_message": (
            ""
            if passed
            else (
                f"validation_error_count={validation_error_count}"
                + (
                    f" exceeds max_acceptable_errors={max_acceptable_errors}"
                    if max_acceptable_errors is not None
                    else ""
                )
            )
        ),
    }


def _max_acceptable_errors(total_size: int, *, error_threshold: float) -> int:
    return int(total_size * error_threshold)


def _validate_output(
    workload: EndToEndWorkload,
    output: torch.Tensor | None,
    reference_output: torch.Tensor | None,
) -> dict[str, object]:
    if output is None:
        return {
            "validation_error_count": 0,
            "run_status": "failed_exception",
            "failure_message": "no output tensor was produced",
        }

    if reference_output is None:
        non_finite = int(torch.count_nonzero(~torch.isfinite(output)).item())
        return {
            "validation_error_count": non_finite,
            "run_status": "passed" if non_finite == 0 else "failed_validation",
            "failure_message": (
                "" if non_finite == 0 else f"non_finite_output_count={non_finite}"
            ),
        }

    if tuple(output.shape) != tuple(reference_output.shape):
        return {
            "validation_error_count": 0,
            "run_status": "failed_validation",
            "failure_message": (
                f"output shape {tuple(output.shape)} does not match "
                f"expected {tuple(reference_output.shape)}"
            ),
        }

    mismatches = ~torch.isclose(
        output.float(),
        reference_output.float(),
        rtol=FINAL_REL_TOL,
        atol=FINAL_ABS_TOL,
    )
    validation_error_count = int(torch.count_nonzero(mismatches).item())
    max_acceptable_errors = _max_acceptable_errors(
        workload.seq_len * workload.hidden_size,
        error_threshold=FINAL_ERROR_THRESHOLD,
    )
    passed = validation_error_count <= max_acceptable_errors
    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if passed else "failed_validation",
        "failure_message": (
            ""
            if passed
            else (
                f"validation_error_count={validation_error_count} exceeds "
                f"max_acceptable_errors={max_acceptable_errors}"
            )
        ),
    }


def _measure_power(
    forward_once,
    *,
    requested_power_backend: str,
    runs_per_sample: int,
    avg_latency_ms: float | None,
    timed_total_sec: float,
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
        min_measurement_duration_sec=DEFAULT_MIN_POWER_MEASUREMENT_DURATION_SEC,
    )
    estimated_window_sec = (
        None if avg_iteration_sec is None else avg_iteration_sec * power_probe_runs
    )
    sample_interval_sec = resolve_power_sample_interval_sec(
        requested_interval_sec=DEFAULT_POWER_SAMPLE_INTERVAL_SEC,
        estimated_timed_window_sec=estimated_window_sec,
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


def _reset_pattern_run_buffers(operator) -> None:
    if not getattr(operator, "enable_benchmark_buffer_reset", True):
        return
    if not hasattr(operator, "write_buffer") or not hasattr(operator, "buffers"):
        return
    for buffer_name in getattr(operator, "reset_buffer_names", ()):
        operator.write_buffer(
            buffer_name,
            np.zeros(operator.buffers[buffer_name], dtype=np.uint8),
        )


def _reset_pattern_output_buffer(operator) -> None:
    if not hasattr(operator, "write_buffer") or not hasattr(operator, "buffers"):
        return
    if "output" not in operator.buffers:
        return
    if (
        hasattr(operator, "buffer_static_data")
        and "output" in operator.buffer_static_data
    ):
        return
    operator.write_buffer(
        "output",
        np.zeros(operator.buffers["output"], dtype=np.uint8),
    )


def _run_pattern_once(
    operator,
    input_tensor: torch.Tensor,
    *,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    if not (
        hasattr(operator, "write_buffer")
        and hasattr(operator, "run_runlist")
        and hasattr(operator, "read_buffer_as_torch")
    ):
        return operator.forward(input_tensor)
    _reset_pattern_run_buffers(operator)
    _reset_pattern_output_buffer(operator)
    operator.write_buffer("input", input_tensor.view(-1))
    operator.run_runlist()
    return operator.read_buffer_as_torch("output", output_shape, dtype=bfloat16)


def _warm_up_pattern_runtime(operator, warmup_runs: int) -> bool:
    if warmup_runs <= 0:
        return True
    if not hasattr(operator, "run_runlist"):
        return False
    for _ in range(warmup_runs):
        operator.run_runlist()
    return True


def _cleanup_operator_runtime(operator) -> None:
    release_runtime = getattr(operator, "release_runtime", None)
    if callable(release_runtime):
        try:
            release_runtime()
        except Exception:
            pass

    seen_contexts = set()
    for attr_name in (
        "context",
        "qkv_context",
        "q_proj_context",
        "k_proj_context",
        "v_proj_context",
        "attn_context",
        "post_context",
        "o_proj_context",
        "ffn_up_context",
        "ffn_down_context",
    ):
        context = getattr(operator, attr_name, None)
        if context is None or not hasattr(context, "reset_runtime"):
            continue
        if id(context) in seen_contexts:
            continue
        seen_contexts.add(id(context))
        try:
            context.reset_runtime()
        except Exception:
            continue


def _partitioned_static_weight_tensor(
    input_b,
    *,
    b_col_maj: bool,
) -> torch.Tensor:
    if isinstance(input_b, torch.Tensor):
        return input_b.T.contiguous() if not b_col_maj else input_b.contiguous()
    if b_col_maj:
        full_b = torch.cat(input_b, dim=0)
        return full_b.contiguous()
    full_b = torch.cat(input_b, dim=1)
    return full_b.T.contiguous()


def _benchmark_gemm(
    gemm_kwargs: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    use_static_weight: bool = False,
    scope_prefix: str = "isolated_gemm",
) -> dict[str, object]:
    scope_payload = {
        "gemm_kwargs": gemm_kwargs,
        "use_static_weight": use_static_weight,
    }
    context = _new_benchmark_context(
        _isolated_candidate_scope(scope_prefix, scope_payload)
    )
    try:
        batch_A = tuple(gemm_kwargs.get("batch_A", (1, 0)))
        batch_B = tuple(gemm_kwargs.get("batch_B", (1, 0)))
        batch_C = tuple(gemm_kwargs.get("batch_C", (1, 0)))
        b_col_maj = bool(gemm_kwargs.get("b_col_maj", False))
        c_col_maj = bool(gemm_kwargs.get("c_col_maj", False))
        reference = generate_gemm_reference(
            M=int(gemm_kwargs["M"]),
            K=int(gemm_kwargs["K"]),
            N=int(gemm_kwargs["N"]),
            seed=seed,
            b_col_maj=b_col_maj,
            c_col_maj=c_col_maj,
            batch_A=batch_A,
            batch_B=batch_B,
            batch_C=batch_C,
        )
        operator = AIEGEMM(
            context=context,
            use_static_weight=use_static_weight,
            **gemm_kwargs,
        )

        input_buffers = {"A": reference["input"].flatten()}
        output_buffers: dict[str, torch.Tensor] = {}
        partition_N = int(getattr(operator, "partition_N", 1))
        if batch_C[0] > 1:
            if use_static_weight:
                operator.weight = _partitioned_static_weight_tensor(
                    reference["input_b"],
                    b_col_maj=b_col_maj,
                )
            else:
                input_buffers["B"] = reference["input_b"].flatten()
            output_buffers["C"] = reference["output"].flatten()
            total_output_size = (
                int(gemm_kwargs["M"]) * int(gemm_kwargs["N"]) * batch_C[0]
            )
        else:
            if use_static_weight:
                operator.weight = _partitioned_static_weight_tensor(
                    reference["input_b"],
                    b_col_maj=b_col_maj,
                )
            else:
                if partition_N > 1:
                    full_b = reference["input_b"]
                    if b_col_maj:
                        for i in range(partition_N):
                            row_lo = i * int(gemm_kwargs["N"]) // partition_N
                            row_hi = (i + 1) * int(gemm_kwargs["N"]) // partition_N
                            input_buffers[f"B_{i}"] = full_b[row_lo:row_hi, :].flatten()
                    else:
                        for i in range(partition_N):
                            col_lo = i * int(gemm_kwargs["N"]) // partition_N
                            col_hi = (i + 1) * int(gemm_kwargs["N"]) // partition_N
                            input_buffers[f"B_{i}"] = full_b[:, col_lo:col_hi].flatten()
                else:
                    input_buffers["B_0"] = reference["input_b"].flatten()

            if partition_N > 1:
                full_c = reference["output"]
                if c_col_maj:
                    for i in range(partition_N):
                        row_lo = i * int(gemm_kwargs["N"]) // partition_N
                        row_hi = (i + 1) * int(gemm_kwargs["N"]) // partition_N
                        output_buffers[f"C_{i}"] = full_c[row_lo:row_hi, :].flatten()
                else:
                    for i in range(partition_N):
                        col_lo = i * int(gemm_kwargs["N"]) // partition_N
                        col_hi = (i + 1) * int(gemm_kwargs["N"]) // partition_N
                        output_buffers[f"C_{i}"] = full_c[:, col_lo:col_hi].flatten()
            else:
                output_buffers["C_0"] = reference["output"].flatten()
            total_output_size = int(gemm_kwargs["M"]) * int(gemm_kwargs["N"])

        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=GEMM_REL_TOL,
            abs_tol=GEMM_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"C": sum(len(value) for value in errors.values())},
            max_acceptable_errors=_max_acceptable_errors(
                total_output_size,
                error_threshold=GEMM_ERROR_THRESHOLD,
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_qkv_proj(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("hybrid_qkv_proj", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"qkv_proj": candidate_config},
        )["qkv_proj"]
        reference = generate_qkv_proj_reference(
            seq_len=workload.seq_len,
            hidden_size=workload.hidden_size,
            seed=seed,
        )
        operator = AIEQKVProj(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "A": reference["input"].flatten(),
                "B": reference["input_b"].flatten(),
            },
            {
                "Q": reference["output_q"].flatten(),
                "K": reference["output_k"].flatten(),
                "V": reference["output_v"].flatten(),
            },
            rel_tol=BLOCK_REL_TOL,
            abs_tol=BLOCK_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {
                "Q": len(errors.get("Q", [])),
                "K": len(errors.get("K", [])),
                "V": len(errors.get("V", [])),
            },
            max_acceptable_errors=_max_acceptable_errors(
                workload.seq_len * workload.hidden_size,
                error_threshold=BLOCK_ERROR_THRESHOLD,
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_mha_out_proj(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("hybrid_mha_out_proj", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"mha_out_proj": candidate_config},
        )["mha_out_proj"]
        reference = generate_mha_out_proj_reference(
            heads=workload.num_attention_heads,
            seq_len=workload.seq_len,
            d=workload.attention_head_size,
            seed=seed,
            is_causal=workload.workload_variant == "decoder_gpt2",
        )
        operator = AIEMHAOutProj(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "Q": reference["input_q"].flatten(),
                "K": reference["input_k"].flatten(),
                "V": reference["input_v"].flatten(),
                "W_O": reference["input_w_o"].flatten(),
            },
            {"O": reference["output"].flatten()},
            rel_tol=BLOCK_REL_TOL,
            abs_tol=BLOCK_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"O": len(errors.get("O", []))},
            max_acceptable_errors=_max_acceptable_errors(
                workload.seq_len
                * workload.attention_head_size
                * workload.num_attention_heads,
                error_threshold=BLOCK_ERROR_THRESHOLD,
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_addnorm(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    operator_name: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope(f"hybrid_{operator_name}", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={operator_name: candidate_config},
        )[operator_name]
        total_size = int(kwargs["size"])
        tile_size = int(kwargs["tile_size"])
        if total_size % tile_size != 0:
            raise ValueError(
                f"addnorm size {total_size} is not divisible by tile_size {tile_size}"
            )
        reference = generate_addnorm_reference(
            rows=total_size // tile_size,
            cols=tile_size,
            seed=seed,
        )
        operator = AIEAddAndNorm(
            weights=reference["weight"],
            context=context,
            **kwargs,
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "input1": reference["input1"],
                "input2": reference["input2"],
            },
            {"output": reference["output"]},
            rel_tol=ADDNORM_REL_TOL,
            abs_tol=ADDNORM_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=_max_acceptable_errors(
                total_size,
                error_threshold=BLOCK_ERROR_THRESHOLD,
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_ffn(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("hybrid_ffn", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"ffn": candidate_config},
        )["ffn"]
        reference = generate_ffn_reference(
            M=workload.seq_len,
            K=workload.hidden_size,
            N=workload.intermediate_size,
            seed=seed,
            b_col_maj=bool(kwargs["b_col_maj"]),
            c_col_maj=bool(kwargs["c_col_maj"]),
        )
        operator = AIEFFN(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "A": reference["input"].flatten(),
                "B_Up": reference["input_b_up"].flatten(),
                "B_Down": reference["input_b_down"].flatten(),
            },
            {"C": reference["output"].flatten()},
            rel_tol=BLOCK_REL_TOL,
            abs_tol=BLOCK_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"C": len(errors.get("C", []))},
            max_acceptable_errors=_max_acceptable_errors(
                workload.seq_len * workload.hidden_size,
                error_threshold=BLOCK_ERROR_THRESHOLD,
            ),
            skip_validation=kwargs["stage_only"] is not None,
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_transpose(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_k_transpose", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"k_transpose": candidate_config},
        )["k_transpose"]
        reference = generate_transpose_reference(
            rows=int(kwargs["M"]),
            cols=int(kwargs["N"]),
            seed=seed,
        )
        operator = AIETranspose(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input": reference["input"]},
            {"output": reference["output"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_softmax(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_attn_softmax", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"attn_softmax": candidate_config},
        )["attn_softmax"]
        reference = generate_softmax_reference(
            rows=int(kwargs["rows"]),
            cols=int(kwargs["cols"]),
            seed=seed,
        )
        operator = AIESoftmax(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"in": reference["input"]},
            {"output": reference["output"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _elementwise_mul_buffers(
    kwargs: dict[str, object], *, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    reference = generate_elementwise_mul_reference(
        input_length=int(kwargs["size"]),
        seed=seed,
    )
    input_buffers = {"input1": reference["A"]}
    if kwargs.get("scalar_broadcast") is None:
        input_buffers["input2"] = reference["B"]
        expected_output = reference["C"]
    else:
        expected_output = reference["A"] * float(kwargs["scalar_broadcast"])
    return input_buffers, {"output": expected_output}


def _benchmark_elementwise_mul(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_attn_scale", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"attn_scale": candidate_config},
        )["attn_scale"]
        input_buffers, output_buffers = _elementwise_mul_buffers(kwargs, seed=seed)
        operator = AIEElementwiseMul(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            input_buffers,
            output_buffers,
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_elementwise_add(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_add", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"add": candidate_config},
        )["add"]
        reference = generate_elementwise_add_reference(
            input_length=int(kwargs["size"]),
            seed=seed,
        )
        operator = AIEElementwiseAdd(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input1": reference["A"], "input2": reference["B"]},
            {"output": reference["C"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_hybrid_elementwise_add(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("hybrid_add", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"add": candidate_config},
        )["add"]
        reference = generate_elementwise_add_reference(
            input_length=int(kwargs["size"]),
            seed=seed,
        )
        operator = AIEElementwiseAdd(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input1": reference["A"], "input2": reference["B"]},
            {"output": reference["C"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_layer_norm(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    operator_name: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope(f"runlist_{operator_name}", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={operator_name: candidate_config},
        )[operator_name]
        total_size = int(kwargs["size"])
        tile_size = int(kwargs["tile_size"])
        if total_size % tile_size != 0:
            raise ValueError(
                f"layer_norm size {total_size} is not divisible by tile_size {tile_size}"
            )
        reference = generate_layer_norm_reference(
            rows=total_size // tile_size,
            cols=tile_size,
            seed=seed,
        )
        operator = AIELayerNorm(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input": reference["input"]},
            {"output": reference["output"]},
            rel_tol=LAYER_NORM_REL_TOL,
            abs_tol=LAYER_NORM_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_hybrid_layer_norm(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    operator_name: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope(f"hybrid_{operator_name}", candidate_config)
    )
    try:
        kwargs = resolve_hybrid_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={operator_name: candidate_config},
        )[operator_name]
        total_size = int(kwargs["size"])
        tile_size = int(kwargs["tile_size"])
        reference = generate_layer_norm_reference(
            rows=total_size // tile_size,
            cols=tile_size,
            seed=seed,
        )
        operator = AIELayerNorm(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input": reference["input"]},
            {"output": reference["output"]},
            rel_tol=LAYER_NORM_REL_TOL,
            abs_tol=LAYER_NORM_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_gelu(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_gelu", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"gelu": candidate_config},
        )["gelu"]
        reference = generate_gelu_reference(
            input_length=int(kwargs["size"]),
            seed=seed,
        )
        operator = AIEGELU(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input": reference["input"]},
            {"output": reference["output"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_causal_mask(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context(
        _isolated_candidate_scope("runlist_causal_mask", candidate_config)
    )
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            workload_variant=workload.workload_variant,
            operator_config={"causal_mask": candidate_config},
        )["causal_mask"]
        reference = generate_causal_mask_reference(
            query_block_size=int(kwargs["query_block_size"]),
            seq_len=int(kwargs["seq_len"]),
            num_heads=int(kwargs["num_heads"]),
            q_start=int(kwargs.get("q_start", 0)),
            masked_fill_value=float(kwargs["masked_fill_value"]),
            seed=seed,
        )
        operator = AIECausalMask(context=context, **kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"input1": reference["input"]},
            {"output": reference["output"]},
            rel_tol=EXACT_REL_TOL,
            abs_tol=EXACT_ABS_TOL,
            warmup_iters=warmup_runs,
            timed_iters=runs_per_sample,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def benchmark_operator_candidate(
    execution_mode: ExecutionMode,
    operator_name: str,
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    require_npu_power_mode_turbo(study_name="end-to-end tuning")
    execution_mode = canonical_execution_mode(str(execution_mode))
    return _benchmark_operator_candidate_in_process(
        execution_mode,
        operator_name,
        workload,
        candidate_config,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
    )


def _benchmark_operator_candidate_in_process(
    execution_mode: ExecutionMode,
    operator_name: str,
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    execution_mode = canonical_execution_mode(str(execution_mode))
    benchmarkers = {
        ("hybrid", "ln1"): lambda *args, **kwargs: _benchmark_hybrid_layer_norm(
            *args, operator_name="ln1", **kwargs
        ),
        ("hybrid", "qkv_proj"): _benchmark_qkv_proj,
        ("hybrid", "mha_out_proj"): _benchmark_mha_out_proj,
        ("hybrid", "add_norm"): lambda *args, **kwargs: _benchmark_addnorm(
            *args, operator_name="add_norm", **kwargs
        ),
        ("hybrid", "add_norm1"): lambda *args, **kwargs: _benchmark_addnorm(
            *args, operator_name="add_norm1", **kwargs
        ),
        ("hybrid", "ffn"): _benchmark_ffn,
        ("hybrid", "add"): _benchmark_hybrid_elementwise_add,
        ("hybrid", "add_norm2"): lambda *args, **kwargs: _benchmark_addnorm(
            *args, operator_name="add_norm2", **kwargs
        ),
        ("runlist", "qkvo_proj"): lambda *args, **kwargs: _benchmark_gemm_from_runlist(
            *args, operator_name="qkvo_proj", **kwargs
        ),
        ("runlist", "k_transpose"): _benchmark_transpose,
        (
            "runlist",
            "attn_scores",
        ): lambda *args, **kwargs: _benchmark_gemm_from_runlist(
            *args, operator_name="attn_scores", **kwargs
        ),
        ("runlist", "attn_scale"): _benchmark_elementwise_mul,
        ("runlist", "causal_mask"): _benchmark_causal_mask,
        ("runlist", "attn_softmax"): _benchmark_softmax,
        (
            "runlist",
            "attn_output",
        ): lambda *args, **kwargs: _benchmark_gemm_from_runlist(
            *args, operator_name="attn_output", **kwargs
        ),
        ("runlist", "add"): _benchmark_elementwise_add,
        ("runlist", "ln1"): lambda *args, **kwargs: _benchmark_layer_norm(
            *args, operator_name="ln1", **kwargs
        ),
        ("runlist", "up_proj"): lambda *args, **kwargs: _benchmark_gemm_from_runlist(
            *args, operator_name="up_proj", **kwargs
        ),
        ("runlist", "gelu"): _benchmark_gelu,
        ("runlist", "down_proj"): lambda *args, **kwargs: _benchmark_gemm_from_runlist(
            *args, operator_name="down_proj", **kwargs
        ),
        ("runlist", "ln2"): lambda *args, **kwargs: _benchmark_layer_norm(
            *args, operator_name="ln2", **kwargs
        ),
    }

    try:
        return benchmarkers[(execution_mode, operator_name)](
            workload,
            candidate_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
        )
    except Exception as exc:
        return {
            "avg_latency_ms": "",
            "bandwidth_gbps": "",
            "validation_error_count": "",
            "run_status": "failed_exception",
            "failure_message": f"{type(exc).__name__}: {exc}",
        }


def _benchmark_operator_candidate_subprocess_entry(
    result_queue,
    execution_mode: ExecutionMode,
    operator_name: str,
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> None:
    result_queue.put(
        _benchmark_operator_candidate_in_process(
            execution_mode,
            operator_name,
            workload,
            candidate_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
        )
    )


def _benchmark_operator_candidate_isolated_subprocess(
    execution_mode: ExecutionMode,
    operator_name: str,
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    ctx = multiprocessing.get_context(_preferred_subprocess_start_method())
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_benchmark_operator_candidate_subprocess_entry,
        args=(
            result_queue,
            execution_mode,
            operator_name,
            workload,
            candidate_config,
            warmup_runs,
            runs_per_sample,
            seed,
        ),
    )
    process.start()
    process.join()

    result: dict[str, object] | None = None
    if not result_queue.empty():
        result = result_queue.get()
    result_queue.close()
    result_queue.join_thread()

    gc.collect()

    if process.exitcode == 0 and result is not None:
        return result

    failure_message = (
        "isolated candidate benchmark subprocess failed "
        f"with exit code {process.exitcode}"
    )
    if result is not None and result.get("run_status") == "failed_exception":
        failure_message = str(result.get("failure_message", failure_message))

    return {
        "avg_latency_ms": "",
        "bandwidth_gbps": "",
        "validation_error_count": "",
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }


def _benchmark_mode_subprocess_entry(
    result_path: str,
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    power_backend: str,
    operator_config: dict[str, dict[str, object]] | None,
    include_reference_output: bool | None,
    capture_latencies: bool,
    scope_key_override: str | None,
    scope_suffix: str | None,
) -> None:
    result = benchmark_mode(
        execution_mode,
        workload,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
        power_backend=power_backend,
        operator_config=operator_config,
        include_reference_output=include_reference_output,
        capture_latencies=capture_latencies,
        scope_key_override=scope_key_override,
        scope_suffix=scope_suffix,
    )
    with open(result_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def benchmark_mode_subprocess(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    power_backend: str,
    operator_config: dict[str, dict[str, object]] | None = None,
    include_reference_output: bool | None = None,
    capture_latencies: bool = False,
    scope_key_override: str | None = None,
    scope_suffix: str | None = None,
) -> dict[str, object]:
    ctx = multiprocessing.get_context(_preferred_subprocess_start_method())
    with tempfile.NamedTemporaryFile(
        prefix="benchmark_mode_result_",
        suffix=".pkl",
        delete=False,
    ) as temp_result_file:
        result_path = temp_result_file.name
    process = ctx.Process(
        target=_benchmark_mode_subprocess_entry,
        args=(
            result_path,
            execution_mode,
            workload,
            warmup_runs,
            runs_per_sample,
            seed,
            power_backend,
            operator_config,
            include_reference_output,
            capture_latencies,
            scope_key_override,
            scope_suffix,
        ),
    )
    process.start()
    process.join()

    result: dict[str, object] | None = None
    if process.exitcode == 0 and os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                result = pickle.load(handle)
        finally:
            os.unlink(result_path)
    elif os.path.exists(result_path):
        os.unlink(result_path)

    gc.collect()

    if process.exitcode == 0 and result is not None:
        result.setdefault("process_model", "subprocess")
        return result

    failure_message = (
        "full benchmark subprocess failed " f"with exit code {process.exitcode}"
    )
    if result is not None and result.get("run_status") == "failed_exception":
        failure_message = str(result.get("failure_message", failure_message))

    return {
        "timed_total_sec": 0.0,
        "measured_inference_count": 0,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "effective_gflops_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "effective_gflops_per_sec_per_watt": None,
        "host_qkv_precompute_ms": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "subprocess",
        "validation_error_count": "",
        "run_status": "failed_exception",
        "failure_message": failure_message,
    }


def _benchmark_gemm_from_runlist(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    operator_name: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    kwargs = resolve_runlist_operator_config(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        workload_variant=workload.workload_variant,
        operator_config={operator_name: candidate_config},
    )[operator_name]
    return _benchmark_gemm(
        kwargs,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
        scope_prefix=f"runlist_{operator_name}",
    )


def benchmark_mode(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    power_backend: str,
    operator_config: dict[str, dict[str, object]] | None = None,
    include_reference_output: bool | None = None,
    capture_latencies: bool = False,
    scope_key_override: str | None = None,
    scope_suffix: str | None = None,
) -> dict[str, object]:
    require_npu_power_mode_turbo(study_name="end-to-end benchmark")
    execution_mode = canonical_execution_mode(str(execution_mode))
    result = {
        "timed_total_sec": 0.0,
        "measured_inference_count": 0,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "effective_gflops_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "effective_gflops_per_sec_per_watt": None,
        "host_qkv_precompute_ms": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": "",
    }

    include_output = (
        workload.seq_len <= REFERENCE_VALIDATION_MAX_SEQ_LEN
        if include_reference_output is None
        else bool(include_reference_output)
    )
    reference = generate_golden_reference(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        seed=seed,
        workload_variant=workload.workload_variant,
        include_output=include_output,
        include_attention_mask=False,
    )
    scope_key = (
        scope_key_override
        if scope_key_override is not None
        else _config_scope_key(execution_mode, workload, operator_config)
    )
    scope = (
        f"mode_{execution_mode}_{workload.hidden_size}_{workload.seq_len}_{scope_key}"
    )
    if scope_suffix:
        scope = f"{scope}_{scope_suffix}"
    context = _new_benchmark_context(scope)
    operator = None
    compile_attempt = 0
    try:
        while True:
            operator = _build_operator(
                execution_mode,
                workload,
                reference["weights"],
                context=context,
                operator_config=operator_config,
            )
            compile_started = time.perf_counter()
            try:
                operator.context.compile_all()
                operator.context.prepare_runtime()
                compile_setup_time_ms = (time.perf_counter() - compile_started) * 1000.0
                break
            except RuntimeError as exc:
                if compile_attempt == 0 and _is_retryable_linker_failure(exc):
                    logging.warning(
                        "Retrying %s %s seq_len=%s from a clean build scope after linker failure in %s",
                        execution_mode,
                        workload.workload_variant,
                        workload.seq_len,
                        context.build_dir,
                    )
                    context.reset_runtime()
                    shutil.rmtree(context.build_dir, ignore_errors=True)
                    context = _fresh_benchmark_context_for_build_dir(context.build_dir)
                    compile_attempt += 1
                    continue
                raise
        result["compile_setup_time_ms"] = compile_setup_time_ms
        result.update(
            _metadata_for_operator(
                execution_mode,
                operator,
                workload,
                compile_setup_time_ms=compile_setup_time_ms,
            )
        )

        output_shape = (workload.seq_len, workload.hidden_size)

        def forward_once():
            return _run_pattern_once(
                operator,
                reference["input"],
                output_shape=output_shape,
            )

        if not _warm_up_pattern_runtime(operator, warmup_runs):
            for _ in range(warmup_runs):
                forward_once()

        latencies_sec: list[float] = []
        output = None
        for _ in range(runs_per_sample):
            started = time.perf_counter()
            output = forward_once()
            latencies_sec.append(time.perf_counter() - started)

        summary = _summarize_latencies(latencies_sec)
        result.update(summary)
        if capture_latencies:
            result["latency_samples_ms"] = [
                latency * 1000.0 for latency in latencies_sec
            ]
        result["effective_gflops_per_sec"] = effective_gflops_per_sec(
            seq_len=workload.seq_len,
            hidden_size=workload.hidden_size,
            intermediate_size=workload.intermediate_size,
            num_attention_heads=workload.num_attention_heads,
            avg_latency_ms=summary["avg_latency_ms"],
        )

        power_stats = _measure_power(
            forward_once,
            requested_power_backend=power_backend,
            runs_per_sample=runs_per_sample,
            avg_latency_ms=summary["avg_latency_ms"],
            timed_total_sec=float(summary["timed_total_sec"]),
        )
        result["power_backend"] = power_stats.get(
            "power_backend", result["power_backend"]
        )
        result["avg_power_w"] = power_stats.get("avg_power_w")
        result["effective_gflops_per_sec_per_watt"] = effective_gflops_per_sec_per_watt(
            result["effective_gflops_per_sec"],
            result["avg_power_w"],
        )

        result.update(_validate_output(workload, output, reference["output"]))
        return result
    except Exception as exc:
        result["run_status"] = "failed_exception"
        result["failure_message"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if operator is not None:
            _cleanup_operator_runtime(operator)
