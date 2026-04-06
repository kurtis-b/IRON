<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Study Conventions

This note defines the current shared rules for the implemented
`transformer_layer_new` studies.

## Implemented Scope

- `block`
- `end_to_end`
- `memory_tile_staging`
- `resource_usage`
- `host_comparison`
- `memcpy_bandwidth`

## Retained Cases

Current families:

- `tinybert_512`
  `hidden_size=512`, `intermediate_size=2048`, `num_attention_heads=8`
- `baseline_768`
  `hidden_size=768`, `intermediate_size=3072`, `num_attention_heads=12`
- `baseline_1024`
  `hidden_size=1024`, `intermediate_size=4096`, `num_attention_heads=16`

Deferred family:

- `baseline_2048`
  `hidden_size=2048`, `intermediate_size=8192`, `num_attention_heads=32`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

## End-to-End Rules

The retained NPU execution modes are:

- `dataflow`
- `runlist`

The end-to-end study uses checked-in candidate files:

- `study/end_to_end/dataflow_candidates.json`
- `study/end_to_end/runlist_candidates.json`

The shared synthetic generator lives in:

- `pattern/reference.py`

The main end-to-end validation policy is:

- exact reference validation through `seq_len=512`
- finite-output validation above `512`

The paper-facing spot-check runner strengthens that surface with exact checks at:

- `seq_len=512`
- `seq_len=2048`

The short-sequence iteration policy is:

- `64`, `128`, `256`
  `warmup_runs=1`, `runs_per_sample=100`

## Result Contracts

Canonical CSV outputs:

- `results/block/results.csv`
- `results/end_to_end/results_all_power.csv`
- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`
- `results/memory_tile_staging/results.csv`
- `results/resource_usage/dataflow_block_best_configs.csv`
- `results/resource_usage/runlist_selected_ops.csv`
- `results/host_comparison/results.csv`
- `results/host_comparison/fairness_repeatability.csv`
- `results/memcpy_bandwidth/results.csv`

The end-to-end CSV keeps one row per `(study_case_id, seq_len, execution_mode)`.

The end-to-end tuning CSV keeps one row per
`(study_case_id, seq_len, execution_mode, internal_operator, candidate_id)` and
marks the fastest passing isolated candidate with `is_operator_best=True`.

The host-comparison CSV keeps one comparison row per `(study_case_id, seq_len, metric)`.
Its comparison columns are:

- `igpu`
- `dataflow`

## Supporting Studies

The memory-tile staging CSV keeps one row per
`(family_id, seq_len, block_kind, staging_depth)`.

The resource-usage exports are compile-artifact summaries. They reuse the
existing `build/transformer_layer_new_end_to_end` tree, do not recompile, and
record missing-artifact notes when the expected physical MLIR is absent.

The end-to-end staging-ablation CSV keeps one row per
`(study_case_id, seq_len, block_kind, staging_depth)` for real `dataflow`
reruns. It sweeps the selected `dataflow` config at different staging depths
and can reuse the memory-tile staging depth ladder when that CSV is present.

The host-comparison study remains separate from `study/end_to_end` by design.
It consumes completed end-to-end NPU rows instead of re-running NPU patterns.
