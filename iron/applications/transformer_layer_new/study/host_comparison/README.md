<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Comparison Study

Status: `implemented`

## Goal

Compare the ROCm iGPU reference path against the available end-to-end
NPU `hybrid` rows for the same `(study_case_id, seq_len)`.

CPU is not part of the canonical host-comparison study.

Compared columns:

- `igpu`
- `igpu_rocm_smi`
- `igpu_turbostat_pkgwatt`
- `hybrid`

## Dependency

This study depends on completed end-to-end NPU outputs:

- `results/end_to_end/results_all_power.csv`

It does not re-run NPU patterns. It benchmarks the requested host backend once
per available `(study_case_id, seq_len)` and then joins those measurements with
the retained NPU `hybrid` rows.

For the iGPU path, throughput is measured once per point. The per-watt outputs
then reuse that throughput value with two separate power measurements:

- `rocm-smi`
- `turbostat_pkgwatt`

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

The comparison CSV keeps one row per `(study_case_id, seq_len, metric)`.
The throughput row uses `igpu` and `hybrid`, and the per-watt row uses
`igpu_rocm_smi`, `igpu_turbostat_pkgwatt`, and `hybrid`.

The fairness summary remains one row for the iGPU backend. It does not duplicate
rows by power backend.

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
