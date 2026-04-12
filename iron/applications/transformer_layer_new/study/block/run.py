#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import gc
import logging
import multiprocessing
import sys
from collections import defaultdict
from pathlib import Path

from iron.common.aie_device_manager import AIEDeviceManager
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
from iron.operators.ffn.op import AIEFFN
from iron.operators.ffn.reference import (
    generate_golden_reference as generate_ffn_reference,
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
from iron.applications.transformer_layer_new.pattern.runlist.op import (
    resolve_runlist_operator_config,
)
from ..npu_runtime_checks import (
    require_npu_power_mode_turbo,
    warn_if_npu_power_mode_not_turbo,
)
from ..run_lock import default_lock_path, hold_study_lock

from .cases import (
    BLOCK_KINDS,
    FAMILY_IDS,
    SEQUENCE_LADDER,
    BlockKind,
    BlockWorkload,
    iter_cases,
)

LOGGER = logging.getLogger(__name__)
REL_TOL = 4.0e-2
ABS_TOL = 1.5e-1
ERROR_THRESHOLD = 0.005
EXACT_REL_TOL = 4.0e-2
EXACT_ABS_TOL = 1e-6
LAYER_NORM_REL_TOL = 1.0e-1
LAYER_NORM_ABS_TOL = 1.0e-1

BLOCK_CONFIG_COLUMNS: dict[BlockKind, tuple[str, ...]] = {
    "qkv_proj": (
        "qkv_proj_tile_m",
        "qkv_proj_tile_k",
        "qkv_proj_tile_n",
        "qkv_proj_parallel_seq",
        "qkv_proj_parallel_emb",
    ),
    "mha_out_proj": (
        "mha_out_proj_parallel_seq",
        "mha_out_proj_q_seq_tile",
        "mha_out_proj_kv_seq_tile",
        "mha_out_proj_emb_tile",
        "mha_out_proj_parallel_heads",
        "mha_out_proj_o_proj_acc_depth",
    ),
    "mha_out_proj_causal": (
        "mha_out_proj_causal_parallel_seq",
        "mha_out_proj_causal_q_seq_tile",
        "mha_out_proj_causal_kv_seq_tile",
        "mha_out_proj_causal_emb_tile",
        "mha_out_proj_causal_parallel_heads",
        "mha_out_proj_causal_o_proj_acc_depth",
    ),
    "addnorm": (
        "addnorm_num_aie_columns",
        "addnorm_tile_size",
    ),
    "layer_norm": (
        "layer_norm_num_aie_columns",
        "layer_norm_num_channels",
    ),
    "elementwise_add": (
        "elementwise_add_num_aie_columns",
        "elementwise_add_num_channels",
    ),
    "causal_mask": (
        "causal_mask_num_aie_columns",
        "causal_mask_num_channels",
    ),
    "ffn": (
        "ffn_num_aie_columns",
        "ffn_b_col_maj",
        "ffn_c_col_maj",
        "ffn_tile_m",
        "ffn_tile_k",
        "ffn_tile_n",
        "ffn_down_proj_depth",
        "ffn_n_a_tiles_distributed",
        "ffn_n_b_tiles_distributed",
        "ffn_stage_only",
        "ffn_gelu_stage",
    ),
}

CSV_FIELDNAMES = (
    "study_id",
    "family_id",
    "family_label",
    "seq_len",
    "block_kind",
    "candidate_index",
    "head_dim",
    "num_heads",
    "hidden_size",
    "ffn_dim",
    "avg_latency_ms",
    "bandwidth_gbps",
    "warmup_iters",
    "timed_iters",
    "validation_error_count",
    "run_status",
    "is_best",
    "error_message",
    *(column for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns),
)


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "block" / "results.csv"


def default_resume_input_paths(output_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    candidate = (
        Path(__file__).resolve().parents[2] / "results_final" / "block" / "results.csv"
    )
    if candidate.exists():
        paths.append(candidate)
    if output_path.exists() and output_path not in paths:
        paths.append(output_path)
    return tuple(paths)


def _case_descriptor(family_id: str, workload: BlockWorkload) -> str:
    return (
        f"{family_id} seq_len={workload.seq_len} hidden={workload.hidden_size} "
        f"ffn={workload.ffn_dim} heads={workload.num_heads}"
    )


def _should_use_aggressive_cleanup(seq_len: int) -> bool:
    return seq_len >= 8192


def _aggressive_cleanup(seq_len: int) -> None:
    if not _should_use_aggressive_cleanup(seq_len):
        return

    gc.collect()
    try:
        AIEDeviceManager().reset()
    except Exception:
        LOGGER.exception(
            "Failed to reset AIE device manager during long-sequence block cleanup"
        )
    gc.collect()


def _preferred_subprocess_start_method() -> str:
    start_methods = multiprocessing.get_all_start_methods()
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file not in (None, "<stdin>") and "spawn" in start_methods:
        return "spawn"
    if "fork" in start_methods:
        return "fork"
    return start_methods[0]


def operator_kwargs(
    workload: BlockWorkload,
    block_kind: BlockKind,
    candidate: tuple[object, ...],
) -> dict[str, object]:
    if block_kind == "qkv_proj":
        tile_m, tile_k, tile_n, parallel_seq, parallel_emb = candidate
        return {
            "seq_len": workload.seq_len,
            "hidden_size": workload.hidden_size,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "parallel_seq": parallel_seq,
            "parallel_emb": parallel_emb,
        }

    if block_kind == "mha_out_proj":
        (
            parallel_seq,
            q_seq_tile,
            kv_seq_tile,
            emb_tile,
            parallel_heads,
            o_proj_acc_depth,
        ) = candidate
        return {
            "num_heads": workload.num_heads,
            "seq_len": workload.seq_len,
            "d": workload.head_dim,
            "parallel_seq": parallel_seq,
            "q_seq_tile": q_seq_tile,
            "kv_seq_tile": kv_seq_tile,
            "emb_tile": emb_tile,
            "parallel_heads": parallel_heads,
            "o_proj_acc_depth": o_proj_acc_depth,
        }

    if block_kind == "mha_out_proj_causal":
        (
            parallel_seq,
            q_seq_tile,
            kv_seq_tile,
            emb_tile,
            parallel_heads,
            o_proj_acc_depth,
        ) = candidate
        return {
            "num_heads": workload.num_heads,
            "seq_len": workload.seq_len,
            "d": workload.head_dim,
            "parallel_seq": parallel_seq,
            "q_seq_tile": q_seq_tile,
            "kv_seq_tile": kv_seq_tile,
            "emb_tile": emb_tile,
            "parallel_heads": parallel_heads,
            "o_proj_acc_depth": o_proj_acc_depth,
            "is_causal": True,
        }

    if block_kind == "addnorm":
        num_aie_columns, tile_size = candidate
        return {
            "size": workload.seq_len * workload.hidden_size,
            "num_aie_columns": num_aie_columns,
            "tile_size": tile_size,
        }

    if block_kind == "layer_norm":
        num_aie_columns, num_channels = candidate
        return {
            "size": workload.seq_len * workload.hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": num_channels,
            "tile_size": workload.hidden_size,
        }

    if block_kind == "elementwise_add":
        num_aie_columns, num_channels = candidate
        return {
            "size": workload.seq_len * workload.hidden_size,
            "num_aie_columns": num_aie_columns,
            "num_channels": num_channels,
            "tile_size": workload.hidden_size,
        }

    if block_kind == "causal_mask":
        num_aie_columns, num_channels = candidate
        return resolve_runlist_operator_config(
            workload.seq_len,
            workload.hidden_size,
            workload.ffn_dim,
            workload.num_heads,
            workload_variant="decoder_gpt2",
            num_aie_columns=num_aie_columns,
            operator_config={
                "causal_mask": {
                    "num_aie_columns": num_aie_columns,
                    "num_channels": num_channels,
                }
            },
        )["causal_mask"]

    if block_kind == "ffn":
        (
            num_aie_columns,
            b_col_maj,
            c_col_maj,
            tile_m,
            tile_k,
            tile_n,
            down_proj_depth,
            n_a_tiles_distributed,
            n_b_tiles_distributed,
            stage_only,
            gelu_stage,
        ) = candidate
        return {
            "M": workload.seq_len,
            "K": workload.hidden_size,
            "N": workload.ffn_dim,
            "num_aie_columns": num_aie_columns,
            "b_col_maj": b_col_maj,
            "c_col_maj": c_col_maj,
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "down_proj_depth": down_proj_depth,
            "n_a_tiles_distributed": n_a_tiles_distributed,
            "n_b_tiles_distributed": n_b_tiles_distributed,
            "stage_only": stage_only,
            "gelu_stage": gelu_stage,
            "emulate_bf16_mmul_with_bfp16": True,
        }

    raise ValueError(f"Unsupported block kind: {block_kind!r}")


def config_row(
    block_kind: BlockKind, candidate: tuple[object, ...]
) -> dict[str, object]:
    row = {
        column: "" for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns
    }
    for column, value in zip(BLOCK_CONFIG_COLUMNS[block_kind], candidate):
        row[column] = value
    return row


def _threshold_validation_result(
    buffer_error_counts: dict[str, int],
    *,
    max_acceptable_errors: int,
    skip_validation: bool = False,
) -> dict[str, object]:
    validation_error_count = sum(buffer_error_counts.values())
    if skip_validation:
        return {
            "validation_error_count": validation_error_count,
            "run_status": "passed",
            "error_message": "",
        }

    passed = all(
        error_count <= max_acceptable_errors
        for error_count in buffer_error_counts.values()
    )
    return {
        "validation_error_count": validation_error_count,
        "run_status": "passed" if passed else "failed_validation",
        "error_message": (
            ""
            if passed
            else (
                f"validation_error_count={validation_error_count} exceeds "
                f"max_acceptable_errors={max_acceptable_errors}"
            )
        ),
    }


def _benchmark_qkv_proj(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        reference = generate_qkv_proj_reference(
            seq_len=workload.seq_len,
            hidden_size=workload.hidden_size,
            seed=seed,
        )
        operator = AIEQKVProj(
            context=context, **operator_kwargs(workload, "qkv_proj", candidate)
        )
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
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {
                "Q": len(errors.get("Q", [])),
                "K": len(errors.get("K", [])),
                "V": len(errors.get("V", [])),
            },
            max_acceptable_errors=int(
                workload.seq_len * workload.hidden_size * ERROR_THRESHOLD
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
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    return _benchmark_mha_out_proj_variant(
        workload,
        candidate,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        seed=seed,
        is_causal=False,
    )


def _benchmark_mha_out_proj_variant(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    is_causal: bool,
) -> dict[str, object]:
    context = AIEContext()
    try:
        reference = generate_mha_out_proj_reference(
            heads=workload.num_heads,
            seq_len=workload.seq_len,
            d=workload.head_dim,
            seed=seed,
            is_causal=is_causal,
        )
        operator = AIEMHAOutProj(
            context=context,
            **operator_kwargs(
                workload,
                "mha_out_proj_causal" if is_causal else "mha_out_proj",
                candidate,
            ),
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "Q": reference["input_q"].flatten(),
                "K": reference["input_k"].flatten(),
                "V": reference["input_v"].flatten(),
                "W_O": reference["input_w_o"].flatten(),
            },
            {"O": reference["output"].flatten()},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"O": len(errors.get("O", []))},
            max_acceptable_errors=int(
                workload.seq_len
                * workload.head_dim
                * workload.num_heads
                * ERROR_THRESHOLD
            ),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_mha_out_proj_causal(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    return _benchmark_mha_out_proj_variant(
        workload,
        candidate,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        seed=seed,
        is_causal=True,
    )


def _benchmark_addnorm(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        addnorm_kwargs = operator_kwargs(workload, "addnorm", candidate)
        total_size = int(addnorm_kwargs["size"])
        tile_size = int(addnorm_kwargs["tile_size"])
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
            **addnorm_kwargs,
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "input1": reference["input1"],
                "input2": reference["input2"],
            },
            {"output": reference["output"]},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=int(total_size * ERROR_THRESHOLD),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_layer_norm(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        kwargs = operator_kwargs(workload, "layer_norm", candidate)
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
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=int(total_size * ERROR_THRESHOLD),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_elementwise_add(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        kwargs = operator_kwargs(workload, "elementwise_add", candidate)
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
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=int(int(kwargs["size"]) * ERROR_THRESHOLD),
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def _benchmark_causal_mask(
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        kwargs = operator_kwargs(workload, "causal_mask", candidate)
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
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        validation = _threshold_validation_result(
            {"output": len(errors.get("output", []))},
            max_acceptable_errors=int(
                int(kwargs["query_block_size"])
                * int(kwargs["seq_len"])
                * int(kwargs["num_heads"])
                * ERROR_THRESHOLD
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
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    context = AIEContext()
    try:
        ffn_kwargs = operator_kwargs(workload, "ffn", candidate)
        reference = generate_ffn_reference(
            M=workload.seq_len,
            K=workload.hidden_size,
            N=workload.ffn_dim,
            seed=seed,
            b_col_maj=bool(ffn_kwargs["b_col_maj"]),
            c_col_maj=bool(ffn_kwargs["c_col_maj"]),
        )
        operator = AIEFFN(context=context, **ffn_kwargs)
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {
                "A": reference["input"].flatten(),
                "B_Up": reference["input_b_up"].flatten(),
                "B_Down": reference["input_b_down"].flatten(),
            },
            {"C": reference["output"].flatten()},
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        stage_only = ffn_kwargs["stage_only"]
        validation = _threshold_validation_result(
            {"C": len(errors.get("C", []))},
            max_acceptable_errors=int(
                workload.seq_len * workload.hidden_size * ERROR_THRESHOLD
            ),
            skip_validation=stage_only is not None,
        )
        return {
            "avg_latency_ms": latency_us / 1000.0,
            "bandwidth_gbps": bandwidth_gbps,
            **validation,
        }
    finally:
        context.reset_runtime()


def benchmark_candidate(
    block_kind: BlockKind,
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    require_npu_power_mode_turbo(study_name="block study")
    if _should_use_aggressive_cleanup(workload.seq_len):
        return _benchmark_candidate_isolated_subprocess(
            block_kind,
            workload,
            candidate,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            seed=seed,
        )

    return _benchmark_candidate_in_process(
        block_kind,
        workload,
        candidate,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        seed=seed,
    )


def _benchmark_candidate_in_process(
    block_kind: BlockKind,
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    benchmarkers = {
        "qkv_proj": _benchmark_qkv_proj,
        "mha_out_proj": _benchmark_mha_out_proj,
        "mha_out_proj_causal": _benchmark_mha_out_proj_causal,
        "addnorm": _benchmark_addnorm,
        "layer_norm": _benchmark_layer_norm,
        "elementwise_add": _benchmark_elementwise_add,
        "causal_mask": _benchmark_causal_mask,
        "ffn": _benchmark_ffn,
    }
    try:
        return benchmarkers[block_kind](
            workload,
            candidate,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            seed=seed,
        )
    except Exception as exc:
        return {
            "avg_latency_ms": "",
            "bandwidth_gbps": "",
            "validation_error_count": "",
            "run_status": "failed_exception",
            "error_message": str(exc),
        }
    finally:
        _aggressive_cleanup(workload.seq_len)


def _subprocess_benchmark_entry(
    result_queue,
    block_kind: BlockKind,
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> None:
    result = _benchmark_candidate_in_process(
        block_kind,
        workload,
        candidate,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        seed=seed,
    )
    result_queue.put(result)


def _benchmark_candidate_isolated_subprocess(
    block_kind: BlockKind,
    workload: BlockWorkload,
    candidate: tuple[object, ...],
    *,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> dict[str, object]:
    ctx = multiprocessing.get_context(_preferred_subprocess_start_method())
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_subprocess_benchmark_entry,
        args=(
            result_queue,
            block_kind,
            workload,
            candidate,
            warmup_iters,
            timed_iters,
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

    if process.exitcode == 0 and result is not None:
        _aggressive_cleanup(workload.seq_len)
        return result

    failure_message = (
        f"isolated benchmark subprocess failed with exit code {process.exitcode}"
    )
    if result is not None and result.get("run_status") == "failed_exception":
        failure_message = str(result.get("error_message", failure_message))

    _aggressive_cleanup(workload.seq_len)
    return {
        "avg_latency_ms": "",
        "bandwidth_gbps": "",
        "validation_error_count": "",
        "run_status": "failed_exception",
        "error_message": failure_message,
    }


def mark_best_rows(rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        row["is_best"] = False
        grouped_rows[
            (str(row["family_id"]), int(row["seq_len"]), str(row["block_kind"]))
        ].append(row)

    for group_rows in grouped_rows.values():
        successful_rows = [
            row
            for row in group_rows
            if row["run_status"] == "passed" and row["avg_latency_ms"] != ""
        ]
        if not successful_rows:
            continue
        best_row = min(successful_rows, key=lambda row: float(row["avg_latency_ms"]))
        best_row["is_best"] = True


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def _candidate_key(
    family_id: str,
    seq_len: int,
    block_kind: BlockKind,
    candidate_index: int,
) -> tuple[str, int, BlockKind, int]:
    return (family_id, seq_len, block_kind, candidate_index)


def _config_signature_from_row(
    row: dict[str, object],
    block_kind: BlockKind,
) -> tuple[str, ...]:
    return tuple(
        str(row.get(column, "")) for column in BLOCK_CONFIG_COLUMNS[block_kind]
    )


def _config_signature_from_candidate(
    block_kind: BlockKind,
    candidate: tuple[object, ...],
) -> tuple[str, ...]:
    return tuple(str(value) for value in candidate)


def load_existing_rows_from_paths(
    paths: tuple[Path, ...],
) -> dict[tuple[str, int, BlockKind, int], dict[str, object]]:
    expected_keys = {
        _candidate_key(case.family_id, case.seq_len, block_kind, candidate_index)
        for case in iter_cases()
        for block_kind in BLOCK_KINDS
        for candidate_index, _candidate in enumerate(case.candidates(block_kind))
    }
    existing: dict[tuple[str, int, BlockKind, int], dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                block_kind = str(row.get("block_kind") or "")
                if block_kind not in BLOCK_CONFIG_COLUMNS:
                    continue
                try:
                    key = _candidate_key(
                        str(row.get("family_id") or ""),
                        int(row.get("seq_len") or 0),
                        block_kind,  # type: ignore[arg-type]
                        int(row.get("candidate_index") or 0),
                    )
                except ValueError:
                    continue
                if key not in expected_keys:
                    continue
                existing[key] = dict(row)
    return existing


def reusable_existing_row(
    existing_rows: dict[tuple[str, int, BlockKind, int], dict[str, object]],
    *,
    family_id: str,
    seq_len: int,
    block_kind: BlockKind,
    candidate_index: int,
    candidate: tuple[object, ...],
) -> dict[str, object] | None:
    row = existing_rows.get(
        _candidate_key(family_id, seq_len, block_kind, candidate_index)
    )
    if row is None:
        return None
    if row.get("run_status") != "passed":
        return None
    if _config_signature_from_row(row, block_kind) != _config_signature_from_candidate(
        block_kind, candidate
    ):
        return None
    return dict(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark shared block-study candidates."
    )
    parser.add_argument(
        "--family",
        choices=[*FAMILY_IDS, "all"],
        default="all",
        help="Select one retained family or benchmark all families.",
    )
    parser.add_argument(
        "--seq-len",
        default="all",
        help="Select one sequence length from the retained ladder or use all.",
    )
    parser.add_argument(
        "--block",
        choices=[*BLOCK_KINDS, "all"],
        default="all",
        help="Select one block kind or benchmark all block kinds.",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument(
        "--resume-input",
        type=Path,
        default=None,
        help="Reuse matching passed rows from this prior results CSV.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse rows from the current output path or results_final snapshot.",
    )
    return parser.parse_args(argv)


def selected_block_kinds(block_argument: str) -> tuple[BlockKind, ...]:
    return BLOCK_KINDS if block_argument == "all" else (block_argument,)  # type: ignore[return-value]


def selected_seq_len(seq_len_argument: str) -> int | None:
    if seq_len_argument == "all":
        return None
    seq_len = int(seq_len_argument)
    if seq_len not in SEQUENCE_LADDER:
        raise ValueError(
            f"Unsupported seq_len {seq_len}; expected one of {SEQUENCE_LADDER}"
        )
    return seq_len


def iteration_schedule(
    seq_len: int,
    *,
    warmup_iters: int | None,
    timed_iters: int | None,
) -> tuple[int, int]:
    if seq_len <= 2048:
        default_warmup_iters, default_timed_iters = 1, 10
    elif seq_len <= 4096:
        default_warmup_iters, default_timed_iters = 1, 5
    else:
        default_warmup_iters, default_timed_iters = 1, 2

    return (
        default_warmup_iters if warmup_iters is None else warmup_iters,
        default_timed_iters if timed_iters is None else timed_iters,
    )


def build_rows(
    *,
    family_argument: str,
    seq_len_argument: str,
    block_argument: str,
    warmup_iters: int | None,
    timed_iters: int | None,
    seed: int,
    existing_rows: (
        dict[tuple[str, int, BlockKind, int], dict[str, object]] | None
    ) = None,
) -> list[dict[str, object]]:
    family_id = None if family_argument == "all" else family_argument
    seq_len = selected_seq_len(seq_len_argument)
    block_kinds = selected_block_kinds(block_argument)

    rows: list[dict[str, object]] = []
    for case in iter_cases(family_id=family_id, seq_len=seq_len):
        rows.extend(
            build_case_rows(
                case,
                block_kinds=block_kinds,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                seed=seed,
                existing_rows={} if existing_rows is None else existing_rows,
            )
        )

    mark_best_rows(rows)
    return rows


def build_case_rows(
    case,
    *,
    block_kinds: tuple[BlockKind, ...],
    warmup_iters: int | None,
    timed_iters: int | None,
    seed: int,
    existing_rows: dict[tuple[str, int, BlockKind, int], dict[str, object]],
) -> list[dict[str, object]]:
    case_warmup_iters, case_timed_iters = iteration_schedule(
        case.seq_len,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )
    rows: list[dict[str, object]] = []
    for block_kind in block_kinds:
        for candidate_index, candidate in enumerate(case.candidates(block_kind)):
            reused_row = reusable_existing_row(
                existing_rows,
                family_id=case.family_id,
                seq_len=case.seq_len,
                block_kind=block_kind,
                candidate_index=candidate_index,
                candidate=candidate,
            )
            if reused_row is not None:
                LOGGER.info(
                    "Reusing family=%s seq_len=%s block=%s candidate=%s from existing results",
                    case.family_id,
                    case.seq_len,
                    block_kind,
                    candidate_index,
                )
                rows.append(reused_row)
                continue
            LOGGER.info(
                "Benchmarking family=%s seq_len=%s block=%s candidate=%s warmup_iters=%s timed_iters=%s",
                case.family_id,
                case.seq_len,
                block_kind,
                candidate_index,
                case_warmup_iters,
                case_timed_iters,
            )
            result = benchmark_candidate(
                block_kind,
                case.workload,
                candidate,
                warmup_iters=case_warmup_iters,
                timed_iters=case_timed_iters,
                seed=seed,
            )
            rows.append(
                {
                    "study_id": "block",
                    "family_id": case.family_id,
                    "family_label": case.family_label,
                    "seq_len": case.seq_len,
                    "block_kind": block_kind,
                    "candidate_index": candidate_index,
                    "head_dim": case.workload.head_dim,
                    "num_heads": case.workload.num_heads,
                    "hidden_size": case.workload.hidden_size,
                    "ffn_dim": case.workload.ffn_dim,
                    "warmup_iters": case_warmup_iters,
                    "timed_iters": case_timed_iters,
                    **config_row(block_kind, candidate),
                    **result,
                }
            )

    mark_best_rows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)
    warn_if_npu_power_mode_not_turbo(LOGGER, study_name="block study")
    output_path = args.output.expanduser()
    with hold_study_lock(
        default_lock_path(output_path),
        study_name="block study",
    ):
        resume_paths: tuple[Path, ...] = tuple()
        if not args.no_resume:
            if args.resume_input is not None:
                resume_paths = (args.resume_input.expanduser(),)
            else:
                resume_paths = default_resume_input_paths(output_path)
        existing_rows = load_existing_rows_from_paths(resume_paths)
        if resume_paths and existing_rows:
            LOGGER.info(
                "Loaded %d reusable block rows from %s",
                len(existing_rows),
                ", ".join(str(path) for path in resume_paths),
            )
        family_id = None if args.family == "all" else args.family
        seq_len = selected_seq_len(args.seq_len)
        block_kinds = selected_block_kinds(args.block)
        cases = tuple(iter_cases(family_id=family_id, seq_len=seq_len))
        row_map: dict[tuple[str, int, BlockKind, int], dict[str, object]] = dict(
            existing_rows
        )
        rows: list[dict[str, object]] = list(row_map.values())

        LOGGER.info(
            "Starting block study with %d case(s), block=%s", len(cases), args.block
        )
        for case_index, case in enumerate(cases, start=1):
            LOGGER.info(
                "Case %d/%d: %s",
                case_index,
                len(cases),
                _case_descriptor(case.family_id, case.workload),
            )
            case_rows = build_case_rows(
                case,
                block_kinds=block_kinds,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                seed=args.seed,
                existing_rows=existing_rows,
            )
            for row in case_rows:
                row_map[
                    _candidate_key(
                        str(row["family_id"]),
                        int(row["seq_len"]),
                        str(row["block_kind"]),  # type: ignore[arg-type]
                        int(row["candidate_index"]),
                    )
                ] = row
            rows = list(row_map.values())
            mark_best_rows(rows)
            write_rows(output_path, rows)
            LOGGER.info("Checkpointed %d block-study rows", len(rows))
            _aggressive_cleanup(case.seq_len)

        mark_best_rows(rows)
        write_rows(output_path, rows)
        LOGGER.info("Wrote %s rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
