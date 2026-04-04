#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import count
from pathlib import Path
import time

import torch

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.addnorm.reference import (
    generate_golden_reference as generate_addnorm_reference,
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
    AIETransformerDataflow,
    resolve_dataflow_operator_config,
)
from iron.applications.transformer_layer_new.pattern.offload.op import (
    AIETransformerOffload,
    resolve_offload_operator_config,
)
from iron.applications.transformer_layer_new.pattern.reference import (
    derive_offload_inputs,
    generate_golden_reference,
)
from iron.applications.transformer_layer_new.pattern.runlist.op import (
    AIETransformerRunlist,
    resolve_runlist_operator_config,
)

from .cases import EndToEndWorkload, ExecutionMode
from .power import (
    create_power_monitor,
    empty_power_stats,
    resolve_power_probe_runs,
    resolve_power_sample_interval_sec,
    resolve_requested_power_backend,
)

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
_CONTEXT_COUNTER = count()


def _new_benchmark_context(scope: str) -> AIEContext:
    context = AIEContext()
    sanitized_scope = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in scope
    )
    context.build_dir = (
        Path.cwd()
        / "build"
        / "transformer_layer_new_end_to_end"
        / f"{next(_CONTEXT_COUNTER):04d}_{sanitized_scope}"
    )
    return context


def resolve_mode_operator_config(
    execution_mode: ExecutionMode,
    workload: EndToEndWorkload,
    operator_config: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    if execution_mode == "dataflow":
        return resolve_dataflow_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            operator_config=operator_config,
        )
    if execution_mode == "runlist":
        return resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            operator_config=operator_config,
        )
    if execution_mode == "offload":
        return resolve_offload_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    common_kwargs = {
        "seq_len": workload.seq_len,
        "hidden_size": workload.hidden_size,
        "intermediate_size": workload.intermediate_size,
        "num_heads": workload.num_attention_heads,
        "ln1_weight": weights["ln1_weight"],
        "ln2_weight": weights["ln2_weight"],
        "operator_config": operator_config,
        "context": context,
    }
    if execution_mode == "dataflow":
        operator = AIETransformerDataflow(**common_kwargs)
    elif execution_mode == "runlist":
        operator = AIETransformerRunlist(**common_kwargs)
    elif execution_mode == "offload":
        operator = AIETransformerOffload(**common_kwargs)
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
    if execution_mode == "offload":
        inst_keys = {
            _artifact_key(gemm_op.insts_artifact)
            for _, gemm_op in operator.gemm_ops
            if getattr(gemm_op, "insts_artifact", None) is not None
        }
        xclbin_keys = (
            {_artifact_key(operator.shared_xclbin_artifact)}
            if operator.shared_xclbin_artifact is not None
            else set()
        )
        query_block_size = int(getattr(operator, "query_block_size", workload.seq_len))
        block_count = (workload.seq_len + query_block_size - 1) // query_block_size
        return {
            "compile_setup_time_ms": compile_setup_time_ms,
            "npu_dispatch_count": ((2 * workload.num_attention_heads) + 6)
            * block_count,
            "npu_unique_instruction_binary_count": len(
                {key for key in inst_keys if key is not None}
            ),
            "npu_unique_xclbin_count": len(
                {key for key in xclbin_keys if key is not None}
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
        min_measurement_duration_sec=0.25,
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


def _tokens_per_sec(seq_len: int, avg_latency_ms: float | None) -> float | None:
    if avg_latency_ms is None or avg_latency_ms <= 0:
        return None
    return float(seq_len) / (avg_latency_ms / 1000.0)


def _tokens_per_sec_per_watt(
    tokens_per_sec: float | None, avg_power_w: float | None
) -> float | None:
    if tokens_per_sec is None or avg_power_w is None or avg_power_w <= 0:
        return None
    return tokens_per_sec / avg_power_w


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
) -> dict[str, object]:
    context = _new_benchmark_context("isolated_gemm")
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
    context = _new_benchmark_context("dataflow_qkv_proj")
    try:
        kwargs = resolve_dataflow_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("dataflow_mha_out_proj")
    try:
        kwargs = resolve_dataflow_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            operator_config={"mha_out_proj": candidate_config},
        )["mha_out_proj"]
        reference = generate_mha_out_proj_reference(
            heads=workload.num_attention_heads,
            seq_len=workload.seq_len,
            d=workload.attention_head_size,
            seed=seed,
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
    context = _new_benchmark_context(f"dataflow_{operator_name}")
    try:
        kwargs = resolve_dataflow_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("dataflow_ffn")
    try:
        kwargs = resolve_dataflow_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("runlist_k_transpose")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("runlist_attn_softmax")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("runlist_attn_scale")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context("runlist_add")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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
    context = _new_benchmark_context(f"runlist_{operator_name}")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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


def _benchmark_gelu(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context("runlist_gelu")
    try:
        kwargs = resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
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


def _benchmark_offload_shared_gemm(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    if workload.seq_len >= 8192:
        return _benchmark_offload_long_seq_candidate(
            workload,
            candidate_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
        )

    resolved = resolve_offload_operator_config(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        operator_config={"shared_gemm": candidate_config},
    )["shared_gemm"]
    query_block_size = AIETransformerOffload._resolve_query_block_size(workload.seq_len)
    attn_scores_partition_n = AIETransformerOffload._resolve_attn_scores_partition_n(
        workload.seq_len
    )
    block_count = (workload.seq_len + query_block_size - 1) // query_block_size

    role_results = []
    role_specs = (
        (
            "q_proj",
            {
                "M": query_block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
            },
            3 * block_count,
            True,
        ),
        (
            "attn_scores",
            {
                "M": query_block_size,
                "K": workload.attention_head_size,
                "N": workload.seq_len,
                "partition_N": attn_scores_partition_n,
            },
            workload.num_attention_heads * block_count,
            False,
        ),
        (
            "attn_output",
            {
                "M": query_block_size,
                "K": workload.seq_len,
                "N": workload.attention_head_size,
            },
            workload.num_attention_heads * block_count,
            False,
        ),
        (
            "out_proj",
            {
                "M": query_block_size,
                "K": workload.hidden_size,
                "N": workload.hidden_size,
            },
            block_count,
            True,
        ),
        (
            "ffn_up",
            {
                "M": query_block_size,
                "K": workload.hidden_size,
                "N": workload.intermediate_size,
            },
            block_count,
            True,
        ),
        (
            "ffn_down",
            {
                "M": query_block_size,
                "K": workload.intermediate_size,
                "N": workload.hidden_size,
            },
            block_count,
            True,
        ),
    )

    weighted_latency_ms = 0.0
    total_validation_error_count = 0
    for role_name, shape_kwargs, weight, use_static_weight in role_specs:
        role_result = _benchmark_gemm(
            {
                **resolved,
                **shape_kwargs,
            },
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
            seed=seed,
            use_static_weight=use_static_weight,
        )
        if role_result["run_status"] != "passed":
            return {
                "avg_latency_ms": "",
                "bandwidth_gbps": "",
                "validation_error_count": role_result["validation_error_count"],
                "run_status": role_result["run_status"],
                "failure_message": f"{role_name}: {role_result['failure_message']}",
            }
        role_results.append(role_result)
        weighted_latency_ms += float(role_result["avg_latency_ms"]) * weight
        total_validation_error_count += int(role_result["validation_error_count"])

    avg_bandwidth = (
        sum(float(result["bandwidth_gbps"]) for result in role_results)
        / len(role_results)
        if role_results
        else ""
    )
    return {
        "avg_latency_ms": weighted_latency_ms,
        "bandwidth_gbps": avg_bandwidth,
        "validation_error_count": total_validation_error_count,
        "run_status": "passed",
        "failure_message": "",
    }


def _benchmark_offload_long_seq_candidate(
    workload: EndToEndWorkload,
    candidate_config: dict[str, object],
    *,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    context = _new_benchmark_context("offload_long_seq_candidate")
    operator = None
    try:
        reference = generate_golden_reference(
            workload.seq_len,
            workload.hidden_size,
            workload.intermediate_size,
            workload.num_attention_heads,
            seed=seed,
            include_output=False,
            include_attention_mask=False,
        )
        operator = _build_operator(
            "offload",
            workload,
            reference["weights"],
            context=context,
            operator_config={"shared_gemm": candidate_config},
        )
        operator.prepare_runtime()

        for _ in range(warmup_runs):
            operator.forward(reference["input"])

        latencies_sec: list[float] = []
        output = None
        for _ in range(runs_per_sample):
            started = time.perf_counter()
            output = operator.forward(reference["input"])
            latencies_sec.append(time.perf_counter() - started)

        if output is None:
            return {
                "avg_latency_ms": "",
                "bandwidth_gbps": "",
                "validation_error_count": "",
                "run_status": "failed_exception",
                "failure_message": "No output produced by offload long-sequence benchmark",
            }

        summary = _summarize_latencies(latencies_sec)
        validation = _validate_output(workload, output, None)
        return {
            "avg_latency_ms": summary["avg_latency_ms"],
            "bandwidth_gbps": "",
            "validation_error_count": validation["validation_error_count"],
            "run_status": validation["run_status"],
            "failure_message": validation["failure_message"],
        }
    finally:
        if operator is not None:
            _cleanup_operator_runtime(operator)
        else:
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
    benchmarkers = {
        ("dataflow", "qkv_proj"): _benchmark_qkv_proj,
        ("dataflow", "mha_out_proj"): _benchmark_mha_out_proj,
        ("dataflow", "add_norm1"): lambda *args, **kwargs: _benchmark_addnorm(
            *args, operator_name="add_norm1", **kwargs
        ),
        ("dataflow", "ffn"): _benchmark_ffn,
        ("dataflow", "add_norm2"): lambda *args, **kwargs: _benchmark_addnorm(
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
        ("offload", "shared_gemm"): _benchmark_offload_shared_gemm,
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
        operator_config={operator_name: candidate_config},
    )[operator_name]
    return _benchmark_gemm(
        kwargs,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
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
) -> dict[str, object]:
    result = {
        "timed_total_sec": 0.0,
        "measured_inference_count": 0,
        "avg_latency_ms": None,
        "compile_setup_time_ms": None,
        "tokens_per_sec": None,
        "power_backend": "none" if power_backend == "auto" else power_backend,
        "avg_power_w": None,
        "tokens_per_sec_per_watt": None,
        "host_qkv_precompute_ms": None,
        "npu_dispatch_count": None,
        "npu_unique_instruction_binary_count": None,
        "npu_unique_xclbin_count": None,
        "process_model": "in_process",
        "validation_error_count": 0,
        "run_status": "failed_exception",
        "failure_message": "",
    }

    reference = generate_golden_reference(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        seed=seed,
        include_output=workload.seq_len <= REFERENCE_VALIDATION_MAX_SEQ_LEN,
        include_attention_mask=False,
    )
    context = _new_benchmark_context(
        f"mode_{execution_mode}_{workload.hidden_size}_{workload.seq_len}"
    )
    operator = None
    try:
        operator = _build_operator(
            execution_mode,
            workload,
            reference["weights"],
            context=context,
            operator_config=operator_config,
        )
        compile_started = time.perf_counter()
        if execution_mode == "offload":
            operator.prepare_runtime()
        else:
            operator.context.compile_all()
            operator.context.prepare_runtime()
        compile_setup_time_ms = (time.perf_counter() - compile_started) * 1000.0
        result["compile_setup_time_ms"] = compile_setup_time_ms
        result.update(
            _metadata_for_operator(
                execution_mode,
                operator,
                workload,
                compile_setup_time_ms=compile_setup_time_ms,
            )
        )

        if execution_mode == "offload":
            result["host_qkv_precompute_ms"] = 0.0

            def forward_once():
                operator.prepare_runtime()
                return operator.forward(reference["input"])

        else:

            def forward_once():
                return operator.forward(reference["input"])

        for _ in range(warmup_runs):
            forward_once()

        latencies_sec: list[float] = []
        output = None
        if execution_mode == "offload":
            for _ in range(runs_per_sample):
                operator.prepare_runtime()
                started = time.perf_counter()
                output = operator.forward(reference["input"])
                latencies_sec.append(time.perf_counter() - started)
        else:
            for _ in range(runs_per_sample):
                started = time.perf_counter()
                output = forward_once()
                latencies_sec.append(time.perf_counter() - started)

        summary = _summarize_latencies(latencies_sec)
        result.update(summary)
        result["tokens_per_sec"] = _tokens_per_sec(
            workload.seq_len,
            summary["avg_latency_ms"],
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
        result["tokens_per_sec_per_watt"] = _tokens_per_sec_per_watt(
            result["tokens_per_sec"],
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
