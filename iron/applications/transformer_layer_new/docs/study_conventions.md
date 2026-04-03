<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Study Conventions

This note defines the current shared rules for the first
`transformer_layer_new` study implementation pass.

The current implementation scope is the block study only.
Broader shared-study conventions for `end_to_end`, `reconfiguration_overhead`,
and `igpu` are deferred until those study runners land.

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

Stable study ID:

- `block`

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

The first block-study pass uses each operator's existing reference generator
directly.

- `qkv_proj` uses `iron/operators/qkv_proj/reference.py`
- `mha_out_proj` uses `iron/operators/mha_out_proj/reference.py`
- `addnorm` uses `iron/operators/addnorm/reference.py`
- `ffn` uses `iron/operators/ffn/reference.py`

The block runner does not introduce a new app-level synthetic data contract in
this pass.

## Result CSV Contract

The current canonical CSV is:

- `results/block/results.csv`

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
