#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from iron.applications.transformer_layer.pattern.offload.op import (
    resolve_offload_operator_config,
)
from iron.applications.transformer_layer.pattern.reference import (
    generate_golden_reference,
)
from iron.applications.transformer_layer.study.end_to_end.cases import (
    FAMILY_IDS,
    EndToEndWorkload,
    mode_operators,
)
from iron.applications.transformer_layer.study.end_to_end.modes import (
    benchmark_operator_candidate,
)
from iron.applications.transformer_layer.study.end_to_end.run import iteration_schedule
from iron.applications.transformer_layer.study.end_to_end.select import (
    SelectedEndToEndRow,
    default_results_path,
    load_result_rows,
    select_result_rows,
)
from iron.applications.transformer_layer.study.run_lock import (
    default_lock_path,
    hold_study_lock,
)

LOGGER = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)
RESULT_MODES: tuple[str, ...] = ("hybrid", "runlist", "offload")
NPU_SOURCE_MODES: tuple[str, ...] = ("auto", "tuning", "rebenchmark")
DETAIL_LATENCY_FIELDS: tuple[str, ...] = (
    "latency_sample_count",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
)
HYBRID_GROUP_ORDER: tuple[str, ...] = (
    "QKV Proj",
    "MHA + Output",
    "Residual + Norm",
    "FFN",
)
HYBRID_GROUP_BY_OPERATOR: dict[str, str] = {
    "qkv_proj": "QKV Proj",
    "mha_out_proj": "MHA + Output",
    "add_norm1": "Residual + Norm",
    "add_norm2": "Residual + Norm",
    "ln1": "Residual + Norm",
    "add_norm": "Residual + Norm",
    "add": "Residual + Norm",
    "ffn": "FFN",
}
RUNLIST_GROUP_ORDER: tuple[str, ...] = (
    "LN1",
    "QKVO Proj",
    "K Transpose",
    "Attention Scores",
    "Attention Scale",
    "Causal Mask",
    "Attention Softmax",
    "Attention Output",
    "Residual Add",
    "LN2",
    "Up Proj",
    "GELU",
    "Down Proj",
)
RUNLIST_GROUP_BY_OPERATOR: dict[str, str] = {
    "ln1": "LN1",
    "qkvo_proj": "QKVO Proj",
    "k_transpose": "K Transpose",
    "attn_scores": "Attention Scores",
    "attn_scale": "Attention Scale",
    "causal_mask": "Causal Mask",
    "attn_softmax": "Attention Softmax",
    "attn_output": "Attention Output",
    "add": "Residual Add",
    "ln2": "LN2",
    "up_proj": "Up Proj",
    "gelu": "GELU",
    "down_proj": "Down Proj",
}
OFFLOAD_GROUP_ORDER: tuple[str, ...] = (
    "GEMMs (NPU)",
    "Non-linear operations (CPU)",
)
OFFLOAD_GEMM_GROUP_BY_OPERATOR: dict[str, str] = {
    "q_proj": "GEMMs (NPU)",
    "k_proj": "GEMMs (NPU)",
    "v_proj": "GEMMs (NPU)",
    "attn_scores": "GEMMs (NPU)",
    "attn_output": "GEMMs (NPU)",
    "output_proj": "GEMMs (NPU)",
    "up_proj": "GEMMs (NPU)",
    "down_proj": "GEMMs (NPU)",
}
OFFLOAD_HOST_GROUP_LABELS: dict[str, str] = {
    "attention_host": "Non-linear operations (CPU)",
    "residual_norm": "Non-linear operations (CPU)",
    "activation": "Non-linear operations (CPU)",
}
FULL_PATTERN_GROUP_LABEL = "Full Pattern"
DETAILED_FIELDNAMES: tuple[str, ...] = (
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "component_kind",
    "component_name",
    "group_label",
    "selected_candidate_id",
    "selected_config_json",
    "warmup_runs",
    "runs_per_sample",
    "latency_sample_count",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "run_status",
    "failure_message",
    "measurement_source",
    "source_csv",
    "source_row_status",
)
AGGREGATE_FIELDNAMES: tuple[str, ...] = (
    "study_case_id",
    "study_case_label",
    "workload_variant",
    "execution_mode",
    "seq_len",
    "row_kind",
    "group_label",
    "source_component_count",
    "avg_latency_ms",
    "run_status",
    "failure_message",
    "expected_component_count",
    "missing_components_json",
    "is_complete",
    "measurement_sources_json",
)


def default_detailed_output_path(results_path: Path | None = None) -> Path:
    base = default_results_path() if results_path is None else results_path
    return Path(base).with_name("selected_component_timings.csv")


def default_aggregate_output_path(results_path: Path | None = None) -> Path:
    base = default_results_path() if results_path is None else results_path
    return Path(base).with_name("selected_component_aggregates.csv")


def default_tuning_results_path(results_path: Path | None = None) -> Path:
    base = default_results_path() if results_path is None else results_path
    return Path(base).with_name("tuning_all_power.csv")


def _mode_sort_key(execution_mode: str) -> int:
    return RESULT_MODES.index(execution_mode)


def _group_order(execution_mode: str) -> tuple[str, ...]:
    if execution_mode == "hybrid":
        return HYBRID_GROUP_ORDER
    if execution_mode == "runlist":
        return RUNLIST_GROUP_ORDER
    if execution_mode == "offload":
        return OFFLOAD_GROUP_ORDER
    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _group_sort_key(execution_mode: str, group_label: str) -> int:
    order = _group_order(execution_mode)
    if group_label == FULL_PATTERN_GROUP_LABEL:
        return len(order)
    return order.index(group_label)


def _resolved_sampling(
    selected_row: SelectedEndToEndRow,
    *,
    warmup_runs: int | None,
    runs_per_sample: int | None,
) -> tuple[int, int]:
    default_warmup_runs, default_runs_per_sample = iteration_schedule(
        selected_row.seq_len
    )
    return (
        int(
            warmup_runs
            if warmup_runs is not None
            else (
                selected_row.warmup_runs
                if selected_row.warmup_runs is not None
                else default_warmup_runs
            )
        ),
        int(
            runs_per_sample
            if runs_per_sample is not None
            else (
                selected_row.runs_per_sample
                if selected_row.runs_per_sample is not None
                else default_runs_per_sample
            )
        ),
    )


def _workload_from_selected_row(selected_row: SelectedEndToEndRow) -> EndToEndWorkload:
    return EndToEndWorkload(
        workload_variant=selected_row.workload_variant,  # type: ignore[arg-type]
        seq_len=selected_row.seq_len,
        hidden_size=selected_row.hidden_size,
        intermediate_size=selected_row.intermediate_size,
        num_attention_heads=selected_row.num_attention_heads,
    )


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def _has_required_latency_fields(row: dict[str, object]) -> bool:
    return all(row.get(field) not in ("", None) for field in DETAIL_LATENCY_FIELDS)


def _json_field_matches(value: object, expected: dict[str, object]) -> bool:
    try:
        loaded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return False
    if not isinstance(loaded, dict):
        return False
    return _json_dumps(loaded) == _json_dumps(expected)


def _component_identifier(component_kind: str, component_name: str) -> str:
    return f"{component_kind}:{component_name}"


def _tuning_row_key(row: dict[str, object]) -> tuple[str, str, str, str, int, str]:
    return (
        str(row.get("study_case_id") or ""),
        str(row.get("workload_variant") or ""),
        str(row.get("execution_mode") or ""),
        str(row.get("internal_operator") or ""),
        int(float(str(row.get("seq_len") or 0))),
        str(row.get("candidate_id") or ""),
    )


def load_tuning_rows(
    path: Path,
) -> dict[tuple[str, str, str, str, int, str], dict[str, object]]:
    rows: dict[tuple[str, str, str, str, int, str], dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("candidate_id") or ""):
                continue
            copied: dict[str, object] = dict(row)
            copied["_source_csv"] = str(path)
            rows[_tuning_row_key(copied)] = copied
    return rows


def _detail_row_key_from_values(
    *,
    study_case_id: str,
    workload_variant: str,
    execution_mode: str,
    seq_len: int,
    component_kind: str,
    component_name: str,
    selected_candidate_id: str,
) -> tuple[str, str, str, int, str, str, str]:
    return (
        study_case_id,
        workload_variant,
        execution_mode,
        int(seq_len),
        component_kind,
        component_name,
        selected_candidate_id,
    )


def _detail_row_key(row: dict[str, object]) -> tuple[str, str, str, int, str, str, str]:
    return _detail_row_key_from_values(
        study_case_id=str(row.get("study_case_id") or ""),
        workload_variant=str(row.get("workload_variant") or ""),
        execution_mode=str(row.get("execution_mode") or ""),
        seq_len=int(float(str(row.get("seq_len") or 0))),
        component_kind=str(row.get("component_kind") or ""),
        component_name=str(row.get("component_name") or ""),
        selected_candidate_id=str(row.get("selected_candidate_id") or ""),
    )


def load_detailed_rows(
    path: Path,
) -> dict[tuple[str, str, str, int, str, str, str], dict[str, object]]:
    rows: dict[tuple[str, str, str, int, str, str, str], dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            copied: dict[str, object] = dict(row)
            copied["_source_csv"] = str(path)
            copied["group_label"] = _canonical_group_label(
                execution_mode=str(copied.get("execution_mode") or ""),
                component_kind=str(copied.get("component_kind") or ""),
                component_name=str(copied.get("component_name") or ""),
                fallback=str(copied.get("group_label") or ""),
            )
            rows[_detail_row_key(copied)] = copied
    return rows


def hybrid_group_label(operator_name: str) -> str:
    return HYBRID_GROUP_BY_OPERATOR[operator_name]


def runlist_group_label(operator_name: str) -> str:
    return RUNLIST_GROUP_BY_OPERATOR[operator_name]


def offload_group_label(
    *,
    component_kind: str,
    component_name: str,
) -> str:
    if component_kind == "host_group":
        return OFFLOAD_HOST_GROUP_LABELS[component_name]
    return OFFLOAD_GEMM_GROUP_BY_OPERATOR[component_name]


def _canonical_group_label(
    *,
    execution_mode: str,
    component_kind: str,
    component_name: str,
    fallback: str = "",
) -> str:
    try:
        if execution_mode == "hybrid" and component_kind == "logical_operator":
            return hybrid_group_label(component_name)
        if execution_mode == "runlist" and component_kind == "logical_operator":
            return runlist_group_label(component_name)
        if execution_mode == "offload":
            return offload_group_label(
                component_kind=component_kind,
                component_name=component_name,
            )
    except KeyError:
        if fallback:
            return fallback
        raise
    if fallback:
        return fallback
    raise ValueError(
        "Unsupported selected-component row: "
        f"execution_mode={execution_mode} "
        f"component_kind={component_kind} component_name={component_name}"
    )


def _write_rows(
    path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sorted_detailed_rows(
    rows: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    family_order = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    return sorted(
        rows,
        key=lambda row: (
            family_order[str(row["study_case_id"])],
            _mode_sort_key(str(row["execution_mode"])),
            int(row["seq_len"]),
            _group_sort_key(str(row["execution_mode"]), str(row["group_label"])),
            str(row["component_name"]),
        ),
    )


def _latency_summary(latencies_sec: list[float]) -> dict[str, object]:
    if not latencies_sec:
        return {
            "latency_sample_count": 0,
            "avg_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
        }
    latencies_ms = [latency * 1000.0 for latency in latencies_sec]
    return {
        "latency_sample_count": len(latencies_ms),
        "avg_latency_ms": sum(latencies_ms) / float(len(latencies_ms)),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
    }


def _benchmark_callable(
    timed_callable,
    *,
    warmup_runs: int,
    runs_per_sample: int,
) -> dict[str, object]:
    try:
        with torch.no_grad():
            for _ in range(warmup_runs):
                timed_callable()
            latencies_sec: list[float] = []
            for _ in range(runs_per_sample):
                started = time.perf_counter()
                timed_callable()
                latencies_sec.append(time.perf_counter() - started)
        return {
            **_latency_summary(latencies_sec),
            "run_status": "passed",
            "failure_message": "",
        }
    except Exception as exc:
        return {
            "latency_sample_count": "",
            "avg_latency_ms": "",
            "min_latency_ms": "",
            "max_latency_ms": "",
            "run_status": "failed_exception",
            "failure_message": f"{type(exc).__name__}: {exc}",
        }


def _offload_host_group_callables(
    selected_row: SelectedEndToEndRow,
    *,
    seed: int,
) -> dict[str, object]:
    workload = _workload_from_selected_row(selected_row)
    reference = generate_golden_reference(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        seed=seed,
        workload_variant=workload.workload_variant,
        include_output=False,
        include_attention_mask=False,
    )
    input_tensor = reference["input"]
    weights = reference["weights"]
    operator_config = resolve_offload_operator_config(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        workload_variant=workload.workload_variant,
        operator_config=selected_row.selected_config,
    )
    query_block_size = int(operator_config["attn_scores"]["query_block_size"])
    head_dim = workload.attention_head_size
    scale = math.sqrt(1.0 / float(head_dim))

    zero_ln1_bias = torch.zeros_like(weights["ln1_weight"])
    zero_ln2_bias = torch.zeros_like(weights["ln2_weight"])

    if workload.workload_variant == "decoder_gpt2":
        attn_input = F.layer_norm(
            input_tensor,
            (workload.hidden_size,),
            weights["ln1_weight"],
            zero_ln1_bias,
        ).to(torch.bfloat16)
    else:
        attn_input = input_tensor

    q = torch.matmul(attn_input, weights["q_weight"])
    k = torch.matmul(attn_input, weights["k_weight"])
    v = torch.matmul(attn_input, weights["v_weight"])

    q_heads = q.view(
        workload.seq_len, workload.num_attention_heads, head_dim
    ).transpose(0, 1)
    k_heads = k.view(
        workload.seq_len, workload.num_attention_heads, head_dim
    ).transpose(0, 1)
    v_heads = v.view(
        workload.seq_len, workload.num_attention_heads, head_dim
    ).transpose(0, 1)

    attn_output_heads = torch.empty_like(v_heads)
    for q_start in range(0, workload.seq_len, query_block_size):
        q_end = min(q_start + query_block_size, workload.seq_len)
        q_block = q_heads[:, q_start:q_end, :].contiguous()
        for head_index in range(workload.num_attention_heads):
            scores = torch.matmul(
                q_block[head_index],
                k_heads[head_index].transpose(-2, -1).contiguous(),
            )
            scaled = scores.float() * scale
            if workload.workload_variant == "decoder_gpt2":
                query_positions = torch.arange(q_start, q_end, device=scaled.device)
                key_positions = torch.arange(workload.seq_len, device=scaled.device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                scaled = scaled.masked_fill(causal_mask, float("-inf"))
            attn_probs = F.softmax(scaled, dim=-1).to(torch.bfloat16)
            attn_output_heads[head_index, q_start:q_end, :] = torch.matmul(
                attn_probs,
                v_heads[head_index],
            )

    attn_output = (
        attn_output_heads.transpose(0, 1)
        .contiguous()
        .view(workload.seq_len, workload.hidden_size)
    )
    projected = torch.matmul(attn_output, weights["attn_output_weight"])
    residual_hidden_states = projected + input_tensor

    if workload.workload_variant == "decoder_gpt2":
        ffn_input = F.layer_norm(
            residual_hidden_states,
            (workload.hidden_size,),
            weights["ln2_weight"],
            zero_ln2_bias,
        ).to(torch.bfloat16)
    else:
        hidden_states = F.layer_norm(
            residual_hidden_states,
            (workload.hidden_size,),
            weights["ln1_weight"],
            zero_ln1_bias,
        ).to(torch.bfloat16)
        ffn_input = hidden_states

    intermediate_pre_activation = torch.matmul(ffn_input, weights["ffn_up_weight"])
    activated = F.gelu(intermediate_pre_activation).to(torch.bfloat16)
    ffn_output = torch.matmul(activated, weights["ffn_down_weight"])
    reusable_scores = torch.matmul(
        q_heads[0, : min(query_block_size, workload.seq_len), :].contiguous(),
        k_heads[0].transpose(-2, -1).contiguous(),
    )

    def attention_host():
        if workload.workload_variant == "decoder_gpt2":
            F.layer_norm(
                input_tensor,
                (workload.hidden_size,),
                weights["ln1_weight"],
                zero_ln1_bias,
            ).to(torch.bfloat16)
        for q_start in range(0, workload.seq_len, query_block_size):
            q_end = min(q_start + query_block_size, workload.seq_len)
            scores = reusable_scores[: q_end - q_start, :]
            for _ in range(workload.num_attention_heads):
                scaled = scores.float() * scale
                if workload.workload_variant == "decoder_gpt2":
                    query_positions = torch.arange(q_start, q_end, device=scaled.device)
                    key_positions = torch.arange(workload.seq_len, device=scaled.device)
                    causal_mask = key_positions.unsqueeze(
                        0
                    ) > query_positions.unsqueeze(1)
                    scaled = scaled.masked_fill(causal_mask, float("-inf"))
                F.softmax(scaled, dim=-1).to(torch.bfloat16)

    def residual_norm():
        residual = projected + input_tensor
        if workload.workload_variant == "decoder_gpt2":
            F.layer_norm(
                residual,
                (workload.hidden_size,),
                weights["ln2_weight"],
                zero_ln2_bias,
            ).to(torch.bfloat16)
            (ffn_output + residual).to(torch.bfloat16)
            return
        hidden = F.layer_norm(
            residual,
            (workload.hidden_size,),
            weights["ln1_weight"],
            zero_ln1_bias,
        ).to(torch.bfloat16)
        F.layer_norm(
            ffn_output + hidden,
            (workload.hidden_size,),
            weights["ln2_weight"],
            zero_ln2_bias,
        ).to(torch.bfloat16)

    def activation():
        F.gelu(intermediate_pre_activation).to(torch.bfloat16)

    return {
        "attention_host": attention_host,
        "residual_norm": residual_norm,
        "activation": activation,
    }


def _selected_component_row(
    *,
    selected_row: SelectedEndToEndRow,
    execution_mode: str,
    component_kind: str,
    component_name: str,
    group_label: str,
    selected_candidate_id: str,
    selected_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
    benchmark_result: dict[str, object],
    measurement_source: str,
    source_csv: str = "",
    source_row_status: str = "",
) -> dict[str, object]:
    return {
        "study_case_id": selected_row.study_case_id,
        "study_case_label": selected_row.study_case_label,
        "workload_variant": selected_row.workload_variant,
        "execution_mode": execution_mode,
        "seq_len": selected_row.seq_len,
        "component_kind": component_kind,
        "component_name": component_name,
        "group_label": group_label,
        "selected_candidate_id": selected_candidate_id,
        "selected_config_json": _json_dumps(selected_config),
        "warmup_runs": warmup_runs,
        "runs_per_sample": runs_per_sample,
        "latency_sample_count": benchmark_result.get("latency_sample_count", ""),
        "avg_latency_ms": benchmark_result.get("avg_latency_ms", ""),
        "min_latency_ms": benchmark_result.get("min_latency_ms", ""),
        "max_latency_ms": benchmark_result.get("max_latency_ms", ""),
        "run_status": benchmark_result.get("run_status", "failed_exception"),
        "failure_message": benchmark_result.get("failure_message", ""),
        "measurement_source": measurement_source,
        "source_csv": source_csv,
        "source_row_status": source_row_status,
    }


def _benchmark_result_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "latency_sample_count": row.get("latency_sample_count", ""),
        "avg_latency_ms": row.get("avg_latency_ms", ""),
        "min_latency_ms": row.get("min_latency_ms", ""),
        "max_latency_ms": row.get("max_latency_ms", ""),
        "run_status": row.get("run_status", ""),
        "failure_message": row.get("failure_message", ""),
    }


def _reusable_detail_component_row(
    resume_rows: dict[tuple[str, str, str, int, str, str, str], dict[str, object]],
    *,
    selected_row: SelectedEndToEndRow,
    component_kind: str,
    component_name: str,
    group_label: str,
    selected_candidate_id: str,
    selected_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
) -> dict[str, object] | None:
    row = resume_rows.get(
        _detail_row_key_from_values(
            study_case_id=selected_row.study_case_id,
            workload_variant=selected_row.workload_variant,
            execution_mode=selected_row.execution_mode,
            seq_len=selected_row.seq_len,
            component_kind=component_kind,
            component_name=component_name,
            selected_candidate_id=selected_candidate_id,
        )
    )
    if row is None:
        return None
    if _optional_int(row.get("warmup_runs")) != warmup_runs:
        return None
    if _optional_int(row.get("runs_per_sample")) != runs_per_sample:
        return None
    if not _json_field_matches(row.get("selected_config_json"), selected_config):
        return None
    source_row_status = str(row.get("run_status") or "")
    if source_row_status != "passed":
        return None
    if not _has_required_latency_fields(row):
        return None
    return _selected_component_row(
        selected_row=selected_row,
        execution_mode=selected_row.execution_mode,
        component_kind=component_kind,
        component_name=component_name,
        group_label=group_label,
        selected_candidate_id=selected_candidate_id,
        selected_config=selected_config,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        benchmark_result=_benchmark_result_from_row(row),
        measurement_source="reused_detail",
        source_csv=str(row.get("_source_csv") or ""),
        source_row_status=source_row_status,
    )


def _tuning_component_row(
    tuning_rows: dict[tuple[str, str, str, str, int, str], dict[str, object]],
    *,
    selected_row: SelectedEndToEndRow,
    component_name: str,
    group_label: str,
    selected_candidate_id: str,
    selected_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
) -> dict[str, object] | None:
    row = tuning_rows.get(
        (
            selected_row.study_case_id,
            selected_row.workload_variant,
            selected_row.execution_mode,
            component_name,
            selected_row.seq_len,
            selected_candidate_id,
        )
    )
    if row is None:
        return None
    source_row_status = str(row.get("run_status") or "")
    if source_row_status != "passed":
        return None
    if not _has_required_latency_fields(row):
        return None
    if not _json_field_matches(row.get("operator_config_json"), selected_config):
        return None
    return _selected_component_row(
        selected_row=selected_row,
        execution_mode=selected_row.execution_mode,
        component_kind="logical_operator",
        component_name=component_name,
        group_label=group_label,
        selected_candidate_id=selected_candidate_id,
        selected_config=selected_config,
        warmup_runs=_optional_int(row.get("warmup_runs")) or warmup_runs,
        runs_per_sample=_optional_int(row.get("runs_per_sample")) or runs_per_sample,
        benchmark_result=_benchmark_result_from_row(row),
        measurement_source="tuning",
        source_csv=str(row.get("_source_csv") or ""),
        source_row_status=source_row_status,
    )


def _missing_tuning_component_row(
    *,
    selected_row: SelectedEndToEndRow,
    component_name: str,
    group_label: str,
    selected_candidate_id: str,
    selected_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
    tuning_results_path: Path | None,
) -> dict[str, object]:
    return _selected_component_row(
        selected_row=selected_row,
        execution_mode=selected_row.execution_mode,
        component_kind="logical_operator",
        component_name=component_name,
        group_label=group_label,
        selected_candidate_id=selected_candidate_id,
        selected_config=selected_config,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        benchmark_result={
            "latency_sample_count": "",
            "avg_latency_ms": "",
            "min_latency_ms": "",
            "max_latency_ms": "",
            "run_status": "missing_tuning_row",
            "failure_message": (
                "no matching passed tuning row for selected component "
                f"{component_name}"
            ),
        },
        measurement_source="tuning",
        source_csv="" if tuning_results_path is None else str(tuning_results_path),
        source_row_status="",
    )


def _benchmark_npu_component_row(
    *,
    selected_row: SelectedEndToEndRow,
    workload: EndToEndWorkload,
    component_name: str,
    group_label: str,
    selected_candidate_id: str,
    selected_config: dict[str, object],
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
) -> dict[str, object]:
    result = benchmark_operator_candidate(
        selected_row.execution_mode,  # type: ignore[arg-type]
        component_name,
        workload,
        dict(selected_config),
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
    )
    return _selected_component_row(
        selected_row=selected_row,
        execution_mode=selected_row.execution_mode,
        component_kind="logical_operator",
        component_name=component_name,
        group_label=group_label,
        selected_candidate_id=selected_candidate_id,
        selected_config=selected_config,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        benchmark_result=result,
        measurement_source="rebenchmark",
    )


def _build_npu_component_row(
    *,
    selected_row: SelectedEndToEndRow,
    workload: EndToEndWorkload,
    component_name: str,
    group_label: str,
    warmup_runs: int,
    runs_per_sample: int,
    seed: int,
    npu_source: str,
    tuning_rows: dict[tuple[str, str, str, str, int, str], dict[str, object]],
    tuning_results_path: Path | None,
    resume_rows: dict[tuple[str, str, str, int, str, str, str], dict[str, object]],
) -> dict[str, object]:
    selected_candidate_id = selected_row.selected_candidate_ids.get(component_name, "")
    selected_config = dict(selected_row.selected_config[component_name])
    if npu_source != "rebenchmark":
        reused_detail = _reusable_detail_component_row(
            resume_rows,
            selected_row=selected_row,
            component_kind="logical_operator",
            component_name=component_name,
            group_label=group_label,
            selected_candidate_id=selected_candidate_id,
            selected_config=selected_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
        )
        if reused_detail is not None:
            return reused_detail

    if npu_source in ("auto", "tuning"):
        tuning_row = _tuning_component_row(
            tuning_rows,
            selected_row=selected_row,
            component_name=component_name,
            group_label=group_label,
            selected_candidate_id=selected_candidate_id,
            selected_config=selected_config,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
        )
        if tuning_row is not None:
            return tuning_row
        if npu_source == "tuning":
            return _missing_tuning_component_row(
                selected_row=selected_row,
                component_name=component_name,
                group_label=group_label,
                selected_candidate_id=selected_candidate_id,
                selected_config=selected_config,
                warmup_runs=warmup_runs,
                runs_per_sample=runs_per_sample,
                tuning_results_path=tuning_results_path,
            )

    return _benchmark_npu_component_row(
        selected_row=selected_row,
        workload=workload,
        component_name=component_name,
        group_label=group_label,
        selected_candidate_id=selected_candidate_id,
        selected_config=selected_config,
        warmup_runs=warmup_runs,
        runs_per_sample=runs_per_sample,
        seed=seed,
    )


def build_detailed_rows(
    selected_rows: tuple[SelectedEndToEndRow, ...],
    *,
    seed: int = 42,
    warmup_runs: int | None = None,
    runs_per_sample: int | None = None,
    tuning_rows: (
        dict[tuple[str, str, str, str, int, str], dict[str, object]] | None
    ) = None,
    tuning_results_path: Path | None = None,
    npu_source: str = "auto",
    resume_rows: (
        dict[tuple[str, str, str, int, str, str, str], dict[str, object]] | None
    ) = None,
    max_host_seq_len: int | None = None,
) -> list[dict[str, object]]:
    if npu_source not in NPU_SOURCE_MODES:
        raise ValueError(f"Unsupported NPU source: {npu_source}")
    tuning_rows = {} if tuning_rows is None else tuning_rows
    resume_rows = {} if resume_rows is None else resume_rows
    detailed_rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        resolved_warmup_runs, resolved_runs_per_sample = _resolved_sampling(
            selected_row,
            warmup_runs=warmup_runs,
            runs_per_sample=runs_per_sample,
        )
        workload = _workload_from_selected_row(selected_row)
        if selected_row.execution_mode in ("hybrid", "runlist"):
            for operator_name in mode_operators(
                selected_row.execution_mode,
                workload.workload_variant,
            ):
                group_label = (
                    hybrid_group_label(operator_name)
                    if selected_row.execution_mode == "hybrid"
                    else runlist_group_label(operator_name)
                )
                detailed_rows.append(
                    _build_npu_component_row(
                        selected_row=selected_row,
                        component_name=operator_name,
                        group_label=group_label,
                        workload=workload,
                        warmup_runs=resolved_warmup_runs,
                        runs_per_sample=resolved_runs_per_sample,
                        seed=seed,
                        npu_source=npu_source,
                        tuning_rows=tuning_rows,
                        tuning_results_path=tuning_results_path,
                        resume_rows=resume_rows,
                    )
                )
            continue

        host_group_callables: dict[str, object] | None = None
        for host_group_name in ("attention_host", "residual_norm", "activation"):
            group_label = offload_group_label(
                component_kind="host_group",
                component_name=host_group_name,
            )
            reused_detail = _reusable_detail_component_row(
                resume_rows,
                selected_row=selected_row,
                component_kind="host_group",
                component_name=host_group_name,
                group_label=group_label,
                selected_candidate_id="",
                selected_config={},
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
            )
            if reused_detail is not None:
                detailed_rows.append(reused_detail)
                continue
            if max_host_seq_len is not None and selected_row.seq_len > int(
                max_host_seq_len
            ):
                LOGGER.info(
                    "Skipping offload host profiling for %s seq_len=%d group=%s "
                    "because --max-host-seq-len=%d",
                    selected_row.study_case_id,
                    selected_row.seq_len,
                    host_group_name,
                    int(max_host_seq_len),
                )
                continue
            if host_group_callables is None:
                host_group_callables = _offload_host_group_callables(
                    selected_row, seed=seed
                )
            result = _benchmark_callable(
                host_group_callables[host_group_name],
                warmup_runs=resolved_warmup_runs,
                runs_per_sample=resolved_runs_per_sample,
            )
            detailed_rows.append(
                _selected_component_row(
                    selected_row=selected_row,
                    execution_mode=selected_row.execution_mode,
                    component_kind="host_group",
                    component_name=host_group_name,
                    group_label=group_label,
                    selected_candidate_id="",
                    selected_config={},
                    warmup_runs=resolved_warmup_runs,
                    runs_per_sample=resolved_runs_per_sample,
                    benchmark_result=result,
                    measurement_source="host_profile",
                )
            )

        for operator_name in mode_operators(
            selected_row.execution_mode,
            workload.workload_variant,
        ):
            detailed_rows.append(
                _build_npu_component_row(
                    selected_row=selected_row,
                    component_name=operator_name,
                    group_label=offload_group_label(
                        component_kind="logical_operator",
                        component_name=operator_name,
                    ),
                    workload=workload,
                    warmup_runs=resolved_warmup_runs,
                    runs_per_sample=resolved_runs_per_sample,
                    seed=seed,
                    npu_source=npu_source,
                    tuning_rows=tuning_rows,
                    tuning_results_path=tuning_results_path,
                    resume_rows=resume_rows,
                )
            )

    return _sorted_detailed_rows(detailed_rows)


def expected_group_components(
    execution_mode: str,
    workload_variant: str,
) -> dict[str, tuple[tuple[str, str], ...]]:
    grouped: dict[str, list[tuple[str, str]]] = {}

    def append_component(group_label: str, component: tuple[str, str]) -> None:
        grouped.setdefault(group_label, []).append(component)

    if execution_mode == "hybrid":
        for operator_name in mode_operators(execution_mode, workload_variant):
            append_component(
                hybrid_group_label(operator_name),
                ("logical_operator", operator_name),
            )
    elif execution_mode == "runlist":
        for operator_name in mode_operators(execution_mode, workload_variant):
            append_component(
                runlist_group_label(operator_name),
                ("logical_operator", operator_name),
            )
    elif execution_mode == "offload":
        for host_group_name in ("attention_host", "residual_norm", "activation"):
            append_component(
                offload_group_label(
                    component_kind="host_group",
                    component_name=host_group_name,
                ),
                ("host_group", host_group_name),
            )
        for operator_name in mode_operators(execution_mode, workload_variant):
            append_component(
                offload_group_label(
                    component_kind="logical_operator",
                    component_name=operator_name,
                ),
                ("logical_operator", operator_name),
            )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    return {
        group_label: tuple(components) for group_label, components in grouped.items()
    }


def expected_group_order(
    execution_mode: str,
    workload_variant: str,
) -> tuple[str, ...]:
    expected = expected_group_components(execution_mode, workload_variant)
    return tuple(
        group_label
        for group_label in _group_order(execution_mode)
        if group_label in expected
    )


def expected_group_component_count(
    execution_mode: str,
    workload_variant: str,
    group_label: str,
) -> int | None:
    components = expected_group_components(execution_mode, workload_variant).get(
        group_label
    )
    if components is None:
        return None
    return len(components)


def build_aggregate_rows(
    selected_rows: tuple[SelectedEndToEndRow, ...],
    detailed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    detail_index: dict[
        tuple[str, str, str, int], dict[tuple[str, str], dict[str, object]]
    ] = {}
    for row in detailed_rows:
        key = (
            str(row["study_case_id"]),
            str(row["workload_variant"]),
            str(row["execution_mode"]),
            int(row["seq_len"]),
        )
        component_key = (
            str(row.get("component_kind") or ""),
            str(row.get("component_name") or ""),
        )
        detail_index.setdefault(key, {})[component_key] = row

    for selected_row in selected_rows:
        key = (
            selected_row.study_case_id,
            selected_row.workload_variant,
            selected_row.execution_mode,
            selected_row.seq_len,
        )
        component_rows_by_key = detail_index.get(key, {})
        expected_by_group = expected_group_components(
            selected_row.execution_mode,
            selected_row.workload_variant,
        )

        for group_label in expected_group_order(
            selected_row.execution_mode,
            selected_row.workload_variant,
        ):
            expected_components = expected_by_group[group_label]
            component_rows: list[dict[str, object]] = []
            missing_components: list[str] = []
            for component_kind, component_name in expected_components:
                row = component_rows_by_key.get((component_kind, component_name))
                if row is None:
                    missing_components.append(
                        _component_identifier(component_kind, component_name)
                    )
                    continue
                component_rows.append(row)

            failed_components: list[str] = []
            for row in component_rows:
                component_name = str(row.get("component_name") or "")
                status = str(row.get("run_status") or "")
                if status != "passed":
                    failure_message = str(row.get("failure_message") or status)
                    failed_components.append(f"{component_name}: {failure_message}")
                    continue
                missing_latency_fields = [
                    field
                    for field in DETAIL_LATENCY_FIELDS
                    if row.get(field) in ("", None)
                ]
                if missing_latency_fields:
                    failed_components.append(
                        f"{component_name}: missing latency fields "
                        f"{','.join(missing_latency_fields)}"
                    )

            if not missing_components and not failed_components:
                avg_latency_ms = sum(
                    float(row["avg_latency_ms"]) for row in component_rows
                )
                run_status = "passed"
                failure_message = ""
                is_complete = True
            elif missing_components:
                avg_latency_ms = ""
                run_status = "missing_group_rows"
                failure_message = "missing detailed rows for " + ", ".join(
                    missing_components
                )
                is_complete = False
            else:
                avg_latency_ms = ""
                run_status = "failed_component_benchmark"
                failure_message = "; ".join(failed_components)
                is_complete = False
            measurement_sources = sorted(
                {
                    str(row.get("measurement_source") or "unknown")
                    for row in component_rows
                }
            )
            aggregates.append(
                {
                    "study_case_id": selected_row.study_case_id,
                    "study_case_label": selected_row.study_case_label,
                    "workload_variant": selected_row.workload_variant,
                    "execution_mode": selected_row.execution_mode,
                    "seq_len": selected_row.seq_len,
                    "row_kind": "isolated_group",
                    "group_label": group_label,
                    "source_component_count": len(component_rows),
                    "avg_latency_ms": avg_latency_ms,
                    "run_status": run_status,
                    "failure_message": failure_message,
                    "expected_component_count": len(expected_components),
                    "missing_components_json": _json_dumps(missing_components),
                    "is_complete": is_complete,
                    "measurement_sources_json": _json_dumps(measurement_sources),
                }
            )

        full_pattern_complete = selected_row.row.get(
            "run_status", "passed"
        ) == "passed" and selected_row.row.get("avg_latency_ms", "") not in ("", None)
        aggregates.append(
            {
                "study_case_id": selected_row.study_case_id,
                "study_case_label": selected_row.study_case_label,
                "workload_variant": selected_row.workload_variant,
                "execution_mode": selected_row.execution_mode,
                "seq_len": selected_row.seq_len,
                "row_kind": "full_pattern",
                "group_label": FULL_PATTERN_GROUP_LABEL,
                "source_component_count": 1,
                "avg_latency_ms": selected_row.row.get("avg_latency_ms", ""),
                "run_status": selected_row.row.get("run_status", "passed"),
                "failure_message": selected_row.row.get("failure_message", ""),
                "expected_component_count": 1,
                "missing_components_json": (
                    "[]" if full_pattern_complete else _json_dumps(["full_pattern"])
                ),
                "is_complete": full_pattern_complete,
                "measurement_sources_json": _json_dumps(["end_to_end"]),
            }
        )

    family_order = {family_id: index for index, family_id in enumerate(FAMILY_IDS)}
    return sorted(
        aggregates,
        key=lambda row: (
            family_order[str(row["study_case_id"])],
            _mode_sort_key(str(row["execution_mode"])),
            int(row["seq_len"]),
            1 if row["row_kind"] == "full_pattern" else 0,
            _group_sort_key(str(row["execution_mode"]), str(row["group_label"])),
        ),
    )


def _parse_seq_len_filter(value: str) -> set[int] | None:
    text = str(value or "all").strip()
    if text == "all":
        return None
    seq_lens: set[int] = set()
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        seq_lens.add(int(stripped))
    if not seq_lens:
        raise ValueError("--seq-len must be 'all' or a comma-separated integer list")
    return seq_lens


def _select_rows_for_filters(
    rows: list[dict[str, str]],
    *,
    workload_variant_filter: str,
    family_filter: str,
    seq_len_filter: str,
    mode_filter: str,
) -> tuple[SelectedEndToEndRow, ...]:
    seq_lens = _parse_seq_len_filter(seq_len_filter)
    selected_rows = select_result_rows(
        rows,
        workload_variant_filter=workload_variant_filter,
        family_filter=family_filter,
        seq_len_filter="all",
        mode_filter=mode_filter,
    )
    return tuple(
        selected_row
        for selected_row in selected_rows
        if selected_row.execution_mode in RESULT_MODES
        and (seq_lens is None or selected_row.seq_len in seq_lens)
    )


def _merge_detail_rows(
    existing_rows: dict[tuple[str, str, str, int, str, str, str], dict[str, object]],
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, str, int, str, str, str], dict[str, object]]:
    merged = {key: dict(value) for key, value in existing_rows.items()}
    for row in rows:
        key = _detail_row_key(row)
        if (
            key in merged
            and str(row.get("measurement_source") or "") == "reused_detail"
        ):
            continue
        merged[key] = dict(row)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark selected hybrid/runlist/offload components from end-to-end results and write grouped aggregate comparison CSVs."
    )
    parser.add_argument("--results", type=Path, default=default_results_path())
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=None,
        help="Detailed selected-component timing CSV path",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Grouped selected-component aggregate CSV path",
    )
    parser.add_argument(
        "--tuning-results",
        type=Path,
        default=None,
        help="Tuning CSV path used to reuse selected NPU operator timings",
    )
    parser.add_argument(
        "--npu-source",
        choices=NPU_SOURCE_MODES,
        default="auto",
        help=(
            "Source for selected NPU component timings. auto reuses matching "
            "tuning rows and rebenchmarks only misses."
        ),
    )
    parser.add_argument(
        "--resume-input",
        type=Path,
        default=None,
        help=(
            "Existing detailed selected-component timing CSV to reuse. Defaults "
            "to --detailed-output when that file already exists."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable detailed selected-component timing row reuse.",
    )
    parser.add_argument(
        "--max-host-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for fresh offload host-group profiling.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--detailed-only",
        action="store_true",
        help="Only produce/checkpoint detailed selected-component timing rows.",
    )
    mode_group.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only read detailed timing rows and write grouped aggregates.",
    )
    parser.add_argument("--family", default="all")
    parser.add_argument("--seq-len", default="all")
    parser.add_argument("--workload-variant", default="all")
    parser.add_argument("--mode", choices=("all", *RESULT_MODES), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--runs-per-sample", type=int, default=None)
    parser.add_argument("--lock-path", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    results_path = Path(args.results).expanduser()
    detailed_output = (
        default_detailed_output_path(results_path)
        if args.detailed_output is None
        else Path(args.detailed_output).expanduser()
    )
    aggregate_output = (
        default_aggregate_output_path(results_path)
        if args.aggregate_output is None
        else Path(args.aggregate_output).expanduser()
    )
    lock_path = (
        default_lock_path(aggregate_output)
        if args.lock_path is None
        else Path(args.lock_path).expanduser()
    )
    tuning_results = (
        default_tuning_results_path(results_path)
        if args.tuning_results is None
        else Path(args.tuning_results).expanduser()
    )
    resume_input: Path | None = None
    if not args.no_resume:
        if args.resume_input is not None:
            resume_input = Path(args.resume_input).expanduser()
        elif detailed_output.exists():
            resume_input = detailed_output

    rows = load_result_rows(results_path)
    selected_rows = _select_rows_for_filters(
        rows,
        workload_variant_filter=args.workload_variant,
        family_filter=args.family,
        seq_len_filter=args.seq_len,
        mode_filter=args.mode,
    )
    LOGGER.info(
        "Benchmarking selected components for %d end-to-end rows", len(selected_rows)
    )
    tuning_rows = (
        load_tuning_rows(tuning_results)
        if args.npu_source in ("auto", "tuning")
        else {}
    )
    if args.npu_source in ("auto", "tuning"):
        LOGGER.info(
            "Loaded %d tuning rows from %s",
            len(tuning_rows),
            tuning_results,
        )
    resume_rows = load_detailed_rows(resume_input) if resume_input is not None else {}
    if resume_input is not None:
        LOGGER.info(
            "Loaded %d reusable detailed rows from %s",
            len(resume_rows),
            resume_input,
        )

    with hold_study_lock(
        lock_path,
        study_name="selected-component aggregate study",
    ):
        detailed_row_map: dict[
            tuple[str, str, str, int, str, str, str], dict[str, object]
        ] = dict(resume_rows)
        if not args.aggregate_only:
            if not selected_rows:
                _write_rows(
                    detailed_output,
                    fieldnames=DETAILED_FIELDNAMES,
                    rows=_sorted_detailed_rows(list(detailed_row_map.values())),
                )
            for selected_row in selected_rows:
                detailed_row_map = _merge_detail_rows(
                    detailed_row_map,
                    build_detailed_rows(
                        (selected_row,),
                        seed=args.seed,
                        warmup_runs=args.warmup_runs,
                        runs_per_sample=args.runs_per_sample,
                        tuning_rows=tuning_rows,
                        tuning_results_path=tuning_results,
                        npu_source=args.npu_source,
                        resume_rows=detailed_row_map,
                        max_host_seq_len=args.max_host_seq_len,
                    ),
                )
                _write_rows(
                    detailed_output,
                    fieldnames=DETAILED_FIELDNAMES,
                    rows=_sorted_detailed_rows(list(detailed_row_map.values())),
                )
                LOGGER.info(
                    "Checkpointed %d selected-component detail rows to %s",
                    len(detailed_row_map),
                    detailed_output,
                )

        detailed_rows = _sorted_detailed_rows(list(detailed_row_map.values()))
        aggregate_rows: list[dict[str, object]] = []
        if not args.detailed_only:
            aggregate_rows = build_aggregate_rows(selected_rows, detailed_rows)
            _write_rows(
                aggregate_output,
                fieldnames=AGGREGATE_FIELDNAMES,
                rows=aggregate_rows,
            )

    incomplete_aggregates = sum(
        1
        for row in aggregate_rows
        if row["row_kind"] == "isolated_group" and row.get("is_complete") is not True
    )
    LOGGER.info(
        "Selected-component summary: selected_rows=%d npu_reused_from_tuning=%d "
        "npu_rebenchmarked=%d host_groups_profiled=%d incomplete_aggregates=%d",
        len(selected_rows),
        sum(
            1
            for row in detailed_rows
            if row.get("component_kind") == "logical_operator"
            and row.get("measurement_source") == "tuning"
            and row.get("run_status") == "passed"
        ),
        sum(
            1
            for row in detailed_rows
            if row.get("component_kind") == "logical_operator"
            and row.get("measurement_source") == "rebenchmark"
        ),
        sum(
            1
            for row in detailed_rows
            if row.get("component_kind") == "host_group"
            and row.get("measurement_source") == "host_profile"
        ),
        incomplete_aggregates,
    )
    LOGGER.info("Wrote %d detailed rows to %s", len(detailed_rows), detailed_output)
    if not args.detailed_only:
        LOGGER.info(
            "Wrote %d aggregate rows to %s", len(aggregate_rows), aggregate_output
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
