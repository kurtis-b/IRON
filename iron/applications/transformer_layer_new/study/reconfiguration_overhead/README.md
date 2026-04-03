<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reconfiguration-Overhead Study

Status: `planned`

## Goal

Isolate the GEMM-only subgraph and compare the shared-runtime reuse path against
the multi-xclbin reconfiguration path.

Compared execution modes:

- `offload_gemm_sequence`
- `runlist_gemm_sequence`

## Case Matrix

Families:

- `baseline_768`
- `baseline_1024`

Representative sequence lengths:

- `64`
- `512`
- `16384`

Total planned surface:

- `2 families x 3 seq lengths x 2 modes`

## Timed Surface

The timed surface is limited to:

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

## Planned Files

This study will own:

- `cases.py`
- `modes.py`
- `run.py`
- `test.py`

## Outputs

Canonical output:

- `results/reconfiguration_overhead/results.csv`
