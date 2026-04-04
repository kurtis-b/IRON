<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# iGPU Study

Status: `implemented`

## Goal

Compare the iGPU reference path against the available end-to-end NPU rows for
the same `(study_case_id, seq_len)`.

Compared iGPU execution mode:

- `amd_igpu_reference`

Reference NPU execution modes:

- `dataflow`
- `runlist`
- `offload`

## Dependency

This study depends on:

- `results/end_to_end/results_all_power.csv`

It does not re-run NPU patterns.
Its execution surface is derived from the completed NPU rows present in the
end-to-end study output.

## Python Environment

Keep the repo-root `requirements.txt` generic for the rest of the codebase.

For the ROCm iGPU study on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

1. install the normal repo requirements first:
   `pip install -r requirements.txt`
2. then install the iGPU ROCm torch overlay:
   `pip install -r iron/applications/transformer_layer_new/requirements.txt`

The overlay file is local to `transformer_layer_new` so that unrelated studies
keep using the repo-default Python environment.

## Shared Memory Note

The retained `seq_len=16384` iGPU cases can require a larger ROCm shared-memory
pool than the kernel default on Ryzen APUs.

AMD documents that ROCm on Ryzen uses the kernel Translation Table Manager
(TTM) shared-memory limit, which defaults to roughly half of system RAM.
If the long-sequence iGPU cases fail with out-of-memory errors, increase the
TTM limit before running this study.

Reference:
<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html>

Example workflow:

- query the current setting:
  `amd-ttm`
- increase the shared-memory limit, in GB:
  `amd-ttm --set 18`
- reboot so the new limit takes effect:
  `sudo reboot`

If `amd-ttm` is not installed yet, AMD's documented setup is:

- `sudo apt install pipx`
- `pipx ensurepath`
- `pipx install amd-debug-tools`

After the iGPU study is finished, clear the TTM override so other studies go
back to the kernel default shared-memory behavior:

- `amd-ttm --clear`
- `sudo reboot`

This TTM guidance is specific to the ROCm iGPU study.
The NPU studies do not require it, and clearing the override afterward avoids
changing the shared-memory environment for unrelated runs.

## Selection Rule

For each `(study_case_id, seq_len)` with available end-to-end data:

- keep only passing NPU rows for `dataflow`, `runlist`, and `offload`
- run the iGPU benchmark once for that workload
- aggregate all retained NPU rows per execution mode when building the
  comparison table

If the reference end-to-end CSV is missing, or if a requested case has no
matching NPU rows, the case is skipped and the study continues.

## Output Shape

The canonical CSV is a compact comparison table with rows keyed by:

- `study_case_id`
- `seq_len`
- `metric`

The metric rows are:

- `effective_gflops_per_sec`
- `effective_gflops_per_sec_per_watt`

The comparison columns are:

- `igpu`
- `dataflow`
- `runlist`
- `offload`

If multiple retained end-to-end rows exist for the same
`(study_case_id, seq_len, execution_mode)`, the study uses the mean value for
that execution mode so that the CSV stays one-column-per-pattern.

If no ROCm-visible GPU is available, the study should emit unsupported rows
as blank iGPU cells instead of failing the full run.

## Planned Files

This study owns:

- `select.py`
- `run.py`
- `test.py`

## Outputs

Canonical output:

- `results/igpu/results.csv`
- `results/igpu/effective_gflops_comparison.svg`
  grouped bar chart for effective GFLOP/s across `igpu`, `dataflow`, `runlist`,
  and `offload`
- `results/igpu/effective_gflops_per_watt_comparison.svg`
  grouped bar chart for effective GFLOP/s/W across `igpu`, `dataflow`,
  `runlist`, and `offload`
