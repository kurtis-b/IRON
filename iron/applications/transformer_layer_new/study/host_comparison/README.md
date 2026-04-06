<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Comparison Study

Status: `implemented`

## Goal

Compare the ROCm iGPU reference path against the available end-to-end
NPU `dataflow` rows for the same `(study_case_id, seq_len)`.

Compared columns:

- `igpu`
- `dataflow`

## Dependency

This study depends on completed end-to-end NPU outputs:

- `results/end_to_end/results_all_power.csv`

It does not re-run NPU patterns. It benchmarks the requested host backend once
per available `(study_case_id, seq_len)` and then joins those measurements with
the retained NPU `dataflow` rows.

## Scheduling

The default short-sequence schedule is:

- `64`, `128`, `256`
  `warmup_runs=1`, `runs_per_sample=100`

The remaining ladder follows the default policy in `run.py`.

## Outputs

Canonical outputs:

- `results/host_comparison/results.csv`
- `results/host_comparison/effective_gflops_comparison.svg`
- `results/host_comparison/effective_gflops_per_watt_comparison.svg`
- `results/host_comparison/fairness_repeatability.csv`

The comparison CSV keeps one row per `(study_case_id, seq_len, metric)` and
uses the columns `igpu` and `dataflow`.

## Entry Points

- `python -m iron.applications.transformer_layer_new.study.host_comparison.run`
- `python -m iron.applications.transformer_layer_new.study.host_comparison.run_fairness_repeatability`

## Python Environment

Keep the repo-root `requirements.txt` generic for the rest of the codebase.

For the iGPU host comparison on Ubuntu 24.04 / Python 3.12 Ryzen APU
systems:

1. install the normal repo requirements first:
   `pip install -r requirements.txt`
2. then install the ROCm torch overlay used by the iGPU path:
   `pip install -r iron/applications/transformer_layer_new/requirements.txt`
