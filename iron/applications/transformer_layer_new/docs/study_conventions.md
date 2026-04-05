<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Study Conventions

This note defines the current shared rules for the implemented
`transformer_layer_new` studies.

The implemented scope now includes:

- `block`
- `end_to_end`
- `reconfiguration_overhead`
- `memory_tile_staging`
- `igpu`
- `memcpy_bandwidth`

## Retained Cases

Current implementation families:

- `baseline_768`
  `hidden_size=768`, `intermediate_size=3072`, `num_attention_heads=12`
- `baseline_1024`
  `hidden_size=1024`, `intermediate_size=4096`, `num_attention_heads=16`

Deferred family:

- `baseline_2048`
  `hidden_size=2048`, `intermediate_size=8192`, `num_attention_heads=32`

Full sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

The shared block-study case table lives in:

- `study/block/cases.py`

The end-to-end study uses checked-in default candidate files:

- `study/end_to_end/dataflow_candidates.json`
- `study/end_to_end/runlist_candidates.json`
- `study/end_to_end/offload_candidates.json`

## Shared Block Entry Point

Each retained block-study case starts from the same workload tuple:

- `seq_len`
- `head_dim`
- `num_heads`
- `ffn_dim`

The runner derives:

- `hidden_size = head_dim * num_heads`

Those shared workload values are then mapped into operator-local parameters for:

- `qkv_proj`
- `mha_out_proj`
- `addnorm`
- `ffn`

The candidate tuples stay explicit and manually editable per family and per
sequence length.

## Stable Naming

Stable study IDs:

- `block`
- `end_to_end`
- `reconfiguration_overhead`
- `memory_tile_staging`
- `igpu`
- `memcpy_bandwidth`

Stable family IDs:

- `baseline_768`
- `baseline_1024`

Stable block kinds:

- `qkv_proj`
- `mha_out_proj`
- `addnorm`
- `ffn`

`addnorm` is explored once as the shared residual add-and-norm operator.
It covers the same operator implementation used at both residual positions in
the full transformer layer.

## Synthetic Data Policy

The block study uses each operator's existing reference generator directly.

- `qkv_proj` uses `iron/operators/qkv_proj/reference.py`
- `mha_out_proj` uses `iron/operators/mha_out_proj/reference.py`
- `addnorm` uses `iron/operators/addnorm/reference.py`
- `ffn` uses `iron/operators/ffn/reference.py`

The end-to-end study uses the shared app-level generator in:

- `pattern/reference.py`

This shared generator supports two modes:

- full golden-reference generation for small sequence lengths
- synthetic input/weight generation without materializing the full attention
  output for the long retained ladder

The end-to-end runner uses golden-reference validation through `seq_len=512`
and finite-output validation above that threshold.

## End-to-End Tuning Flow

The end-to-end study tunes internal operators before the final full-pattern run.

For each `(family, seq_len, execution_mode)` point:

- load the default JSON candidate table from the mode-local candidate file
- benchmark each internal operator's candidates in isolation
- select the fastest passing candidate for that operator
- assemble the selected operator configs into the full pattern
- run the final end-to-end benchmark once with that resolved config set

There is no CLI override for the candidate files in this first pass.

The current tuning units are:

- `dataflow`:
  `qkv_proj`, `mha_out_proj`, `add_norm1`, `ffn`, `add_norm2`
- `runlist`:
  `qkvo_proj`, `k_transpose`, `attn_scores`, `attn_scale`, `attn_softmax`,
  `attn_output`, `add`, `ln1`, `up_proj`, `gelu`, `down_proj`, `ln2`
- `offload`:
  `shared_gemm`

The current `offload` pattern starts from hidden states and performs:

- `q_proj`
- `k_proj`
- `v_proj`
- `attn_scores`
- `attn_output`
- `out_proj`
- `ffn_up`
- `ffn_down`

on NPU under one shared xclbin. Host work remains softmax, GeLU, and the two
residual add/layer norm steps.

## Result CSV Contract

Current canonical CSVs:

- `results/block/results.csv`
- `results/end_to_end/tuning.csv`
- `results/end_to_end/results.csv`
- `results/reconfiguration_overhead/results.csv`
- `results/memory_tile_staging/results.csv`
- `results/igpu/results.csv`
- `results/memcpy_bandwidth/results.csv`

Current canonical study plots:

- `results/memory_tile_staging/mha_out_proj_latency_by_staging_depth.svg`
- `results/memory_tile_staging/mha_out_proj_speedup_by_staging_depth.svg`
- `results/memory_tile_staging/ffn_latency_by_staging_depth.svg`
- `results/memory_tile_staging/ffn_speedup_by_staging_depth.svg`
- `results/igpu/effective_gflops_comparison.svg`
- `results/igpu/effective_gflops_per_watt_comparison.svg`
- `results/memcpy_bandwidth/peak_bandwidth_by_size.svg`
- `results/memcpy_bandwidth/latency_by_size.svg`

Required row groups:

- identity:
  `study_id`, `family_id`, `family_label`, `seq_len`, `block_kind`,
  `candidate_index`
- workload:
  `head_dim`, `num_heads`, `hidden_size`, `ffn_dim`
- timing:
  `avg_latency_ms`, `bandwidth_gbps`, `warmup_iters`, `timed_iters`
- result:
  `run_status`, `is_best`
- config:
  named columns for the operator-specific candidate tuple

The CSV keeps one row per candidate and marks the minimum-latency successful
candidate per `(family_id, seq_len, block_kind)` as best.

The end-to-end CSV keeps one row per `(study_case_id, seq_len, execution_mode)`
and marks the minimum-latency successful row per `(study_case_id, seq_len)` as
best.

The iGPU CSV keeps one comparison row per
`(study_case_id, seq_len, metric)` point.
The metric values are `effective_gflops_per_sec` and
`effective_gflops_per_sec_per_watt`.
Its columns are `igpu`, `dataflow`, `runlist`, and `offload`.
It reuses the retained `end_to_end` NPU rows for `dataflow`, `runlist`, and
`offload`, benchmarks the iGPU once per available `(study_case_id, seq_len)`
group, and aggregates all retained NPU rows per execution mode into the clean
comparison table.

The end-to-end tuning CSV keeps one row per
`(study_case_id, seq_len, execution_mode, internal_operator, candidate_id)` and
marks the fastest passing candidate per internal operator with
`is_operator_best=True`.

The reconfiguration-overhead CSV keeps one row per
`(study_case_id, seq_len, execution_mode)` point.
It reads the selected `runlist` and `offload` configs from
`results/end_to_end/results.csv`, skips missing/empty config rows, and
benchmarks a blocked GEMM-only NPU sequence for:

- `offload_gemm_sequence`
- `runlist_gemm_sequence`

Its required metrics are latency and power:

- `avg_latency_ms`
- `avg_power_w`

It also preserves the reconfiguration metadata:

- `npu_dispatch_count`
- `npu_unique_instruction_binary_count`
- `npu_unique_xclbin_count`

The memory-tile staging CSV keeps one row per
`(family_id, seq_len, block_kind, staging_depth)` point.
It reads the selected passing best rows for `mha_out_proj` and `ffn` from
`results/block/results.csv`, freezes every non-depth config field, and sweeps
all supported divisible staging depths for:

- `mha_out_proj_o_proj_acc_depth`
- `ffn_down_proj_depth`

Its required metrics are:

- `avg_latency_ms`
- `bandwidth_gbps`
- `speedup_vs_depth1`

It also preserves the source-selection metadata and staged best-row marker:

- `source_candidate_index`
- `source_staging_depth`
- `is_best_depth`

The memcpy-bandwidth CSV keeps one row per memcpy benchmark case and is not
family-based.
It measures the existing `AIEMemCopy` operator across transfer size,
`num_cores`, `num_channels`, and `bypass`.

Its required metrics are:

- `latency_us`
- `bandwidth_gbps`

It also preserves the case-identifying config and best-row markers:

- `size_elements`
- `size_bytes`
- `total_moved_bytes`
- `num_cores`
- `num_channels`
- `bypass`
- `tile_size`
- `is_size_peak`
- `is_overall_peak`

Required end-to-end row groups:

- identity:
  `study_id`, `study_case_id`, `study_case_label`, `backend`,
  `execution_mode`, `pattern_label`, `seq_len`
- workload:
  `hidden_size`, `intermediate_size`, `num_attention_heads`,
  `attention_head_size`, `batch_size`, `dtype`, `use_bias`, `weights_source`
- timing:
  `warmup_runs`, `runs_per_sample`, `measured_inference_count`,
  `timed_total_sec`, `avg_latency_ms`, `compile_setup_time_ms`,
  `host_qkv_precompute_ms`
- throughput and power:
  `effective_gflops_per_sec`, `power_backend`, `avg_power_w`,
  `effective_gflops_per_sec_per_watt`
- runtime metadata:
  `npu_dispatch_count`, `npu_unique_instruction_binary_count`,
  `npu_unique_xclbin_count`, `process_model`
- result:
  `validation_error_count`, `run_status`, `failure_message`,
  `selected_candidate_ids_json`, `selected_config_json`, `is_best`
