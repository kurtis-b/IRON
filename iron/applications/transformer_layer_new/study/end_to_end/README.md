<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Study

Status: `implemented`

## Goal

Benchmark the two retained full-layer NPU implementations from the shared
hidden-states boundary:

- `dataflow`
- `runlist`

## Case Matrix

Families:

- `baseline_768`
- `baseline_1024`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

The default short-sequence schedule is:

- `64`, `128`, `256`
  `warmup_runs=1`, `runs_per_sample=100`

The remaining ladder uses the default policy from `run.py`.

## Tuning Flow

The study tunes internal operators before the final full-pattern benchmark.

Configuration sources:

- `dataflow_candidates.json`
- `runlist_candidates.json`

Flow per `(family, seq_len, execution_mode)`:

- benchmark each internal operator candidate in isolation
- select the fastest passing candidate for that operator
- assemble the selected operator configs
- run the final end-to-end benchmark once with the resolved config set

Singleton-candidate handling:

- `runlist` skips isolated tuning when only one candidate remains
- `dataflow` still benchmarks singleton operators so the block-vs-pattern plot
  can continue to use measured block rows

## Outputs

Canonical outputs:

- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/results_all_power.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`

The main result CSV keeps one row per `(study_case_id, seq_len, execution_mode)`
and is the reference input for the separate `igpu` study.

The helper entrypoints default to `results_all_power.csv` when it is present and
fall back to `results.csv` otherwise.

Validation policy:

- main runner:
  exact reference validation through `seq_len=512`
- main runner above `512`:
  finite-output validation
- correctness spot checks:
  exact validation at `512` and `2048`

## Entry Points

- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_staging_ablation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_fairness_repeatability`

Environment:

- `source /opt/xilinx/xrt/setup.sh`
- `source ironenv/bin/activate`
