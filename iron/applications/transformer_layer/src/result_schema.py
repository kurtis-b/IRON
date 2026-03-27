# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

RESULT_FIELD_ORDER = [
    "study_id",
    "study_case_id",
    "study_case_label",
    "backend",
    "execution_mode",
    "pattern_label",
    "input_boundary",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "batch_size",
    "dtype",
    "use_bias",
    "weights_source",
    "source_model_name",
    "source_layer_index",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
    "compile_setup_time_ms",
    "avg_encoder_pipeline_latency_ms",
    "avg_host_projection_latency_ms",
    "avg_npu_projection_latency_ms",
    "avg_npu_gemm_latency_ms",
    "avg_operator_runlist_latency_ms",
    "avg_host_preprocess_latency_ms",
    "avg_host_postprocess_latency_ms",
    "avg_device_sync_latency_ms",
    "npu_dispatch_count",
    "npu_unique_instruction_binary_count",
    "npu_unique_xclbin_count",
    "topology_id",
    "topology_family",
    "parallel_seq",
    "parallel_heads",
    "parallel_ffn",
    "compute_tile_count",
    "compute_tile_utilization_fraction",
    "process_model",
    "run_status",
    "failure_component",
    "failure_category",
    "failure_message",
    "throughput_flops_per_sec",
    "estimated_flops_per_inference",
    "estimated_bytes_per_inference",
    "operational_intensity_flops_per_byte",
    "backend_peak_ops_per_sec",
    "roofline_bound_ops_per_sec",
    "backend_pct_of_peak",
    "roofline_pct",
    "avg_power_w",
    "max_power_w",
    "energy_j",
    "power_sample_count",
]

REQUIRED_RESULT_FIELDS = {
    "study_id",
    "backend",
    "execution_mode",
    "seq_len",
    "batch_size",
    "dtype",
    "weights_source",
    "warmup_runs",
    "runs_per_sample",
    "measured_inference_count",
    "timed_total_sec",
    "avg_latency_ms",
}


def normalize_result_row(row: dict[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_RESULT_FIELDS.difference(row))
    if missing:
        raise KeyError(f"Missing required result fields: {', '.join(missing)}")
    normalized = {field: row.get(field) for field in RESULT_FIELD_ORDER}
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = value
    return normalized
