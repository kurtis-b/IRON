<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# iGPU Study

Status: `implemented`

## Goal

Compare the ROCm iGPU reference path against the available end-to-end NPU rows
for the same `(study_case_id, seq_len)`.

Compared columns:

- `igpu`
- `dataflow`
- `runlist`

## Dependency

This study depends on completed end-to-end NPU outputs:

- `results/end_to_end/results.csv`
- `results/end_to_end/results_all_power.csv`

It does not re-run NPU patterns. It benchmarks the iGPU once per available
`(study_case_id, seq_len)` and then aggregates the retained NPU rows into the
comparison table.

## Scheduling

The default short-sequence schedule is:

- `64`, `128`, `256`
  `warmup_runs=1`, `runs_per_sample=100`

The remaining ladder follows the default policy in `run.py`.

## Outputs

Canonical outputs:

- `results/igpu/results.csv`
- `results/igpu/effective_gflops_comparison.svg`
- `results/igpu/effective_gflops_per_watt_comparison.svg`
- `results/igpu/fairness_repeatability.csv`

The comparison CSV keeps one row per `(study_case_id, seq_len, metric)` and
uses the columns `igpu`, `dataflow`, and `runlist`.

## Entry Points

- `python -m iron.applications.transformer_layer_new.study.igpu.run`
- `python -m iron.applications.transformer_layer_new.study.igpu.run_fairness_repeatability`

## Python Environment

Keep the repo-root `requirements.txt` generic for the rest of the codebase.

For the ROCm iGPU study on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

1. install the normal repo requirements first:
   `pip install -r requirements.txt`
2. then install the iGPU ROCm torch overlay:
   `pip install -r iron/applications/transformer_layer_new/requirements.txt`
