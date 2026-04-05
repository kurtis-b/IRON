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
- `igpu`
- `memcpy_bandwidth`

## Retained Cases

Current families:

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
- `results/end_to_end/results.csv`
- `results/end_to_end/tuning.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`
- `results/memory_tile_staging/results.csv`
- `results/igpu/results.csv`
- `results/igpu/fairness_repeatability.csv`
- `results/memcpy_bandwidth/results.csv`

The end-to-end CSV keeps one row per `(study_case_id, seq_len, execution_mode)`.

The end-to-end tuning CSV keeps one row per
`(study_case_id, seq_len, execution_mode, internal_operator, candidate_id)` and
marks the fastest passing isolated candidate with `is_operator_best=True`.

The iGPU CSV keeps one comparison row per `(study_case_id, seq_len, metric)`.
Its comparison columns are:

- `igpu`
- `dataflow`
- `runlist`

## Supporting Studies

The memory-tile staging CSV keeps one row per
`(family_id, seq_len, block_kind, staging_depth)` and is the source input for
the staging-ablation summary in `study/end_to_end`.

The iGPU study remains separate from `study/end_to_end` by design. It consumes
completed end-to-end NPU rows instead of re-running NPU patterns.
