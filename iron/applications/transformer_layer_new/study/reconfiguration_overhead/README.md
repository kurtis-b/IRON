<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reconfiguration-Overhead Study

Status: `implemented`

## Goal

Compare the current shared-xclbin `offload` path against the multi-xclbin
runlist sequence path for the GEMM-dominated portions of the transformer layer.

Compared execution modes:

- `offload_gemm_sequence`
- `runlist_gemm_sequence`

## Case Matrix

Families:

- `baseline_768`
- `baseline_1024`

Representative sequence lengths:

- `256`
- `2048`
- `16384`

Total planned surface:

- `2 families x 3 seq lengths x 2 modes`

## Timed Surface

The timed surface is limited to the NPU GEMM stages:

- `q_proj`
- `k_proj`
- `v_proj`
- `attn_scores`
- `attn_output`
- `out_proj`
- `ffn_up`
- `ffn_down`

This study intentionally excludes:

- softmax
- GeLU
- add/norm

This matches the current offload/runtime comparison focus, even though the
end-to-end offload pattern now also performs `q/k/v` on NPU.

## Artifact Contract

The study should make the reconfiguration difference visible in row metadata.

- `offload_gemm_sequence`:
  one shared runtime xclbin plus multiple `insts.bin` artifacts
- `runlist_gemm_sequence`:
  multiple xclbins plus `insts.bin` artifacts

The result rows should preserve at least:

- `npu_dispatch_count`
- `npu_unique_instruction_binary_count`
- `npu_unique_xclbin_count`

## Dependency

This study depends on:

- `results/end_to_end/results.csv`

It reads the selected `runlist` and `offload` configs from the end-to-end
study output, but the implementation in this directory stays stand-alone and
does not import from other study directories.

If the required end-to-end row is missing, or its `selected_config_json` is
empty, that `(family, seq_len, mode)` point is skipped.

## Benchmark Boundary

The study benchmarks a blocked GEMM-only sequence with precomputed host-side
intermediates so the timed window isolates NPU latency and power.

Timed GEMM stages:

- `q_proj`
- `k_proj`
- `v_proj`
- `attn_scores`
- `attn_output`
- `out_proj`
- `ffn_up`
- `ffn_down`

The non-GEMM intermediates needed between those stages are prepared outside the
timed loop.

## Output Contract

Canonical output is a long-form CSV with one row per:

- `(study_case_id, seq_len, execution_mode)`

The required metrics are:

- `avg_latency_ms`
- `avg_power_w`

The row also keeps:

- `compile_setup_time_ms`
- `max_power_w`
- `energy_j`
- `power_sample_count`
- `source_end_to_end_execution_mode`
- `selected_candidate_ids_json`
- `selected_config_json`
- `run_status`
- `failure_message`

## Planned Files

This study owns:

- `cases.py`
- `modes.py`
- `power.py`
- `run.py`
- `select.py`
- `test.py`

## Outputs

Canonical output:

- `results/reconfiguration_overhead/results.csv`

Canonical visualization:

- `results/reconfiguration_overhead/runlist_vs_offload_reconfiguration_overhead.svg`
- `results/reconfiguration_overhead/runlist_vs_offload_reconfiguration_overhead.png`
