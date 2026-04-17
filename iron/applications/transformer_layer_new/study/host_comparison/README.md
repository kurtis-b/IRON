<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Comparison Study

This study benchmarks the ROCm iGPU path and joins those measurements with the
completed end-to-end NPU rows for the same `(study_case_id, seq_len)`.

Dependencies:

- `results/end_to_end/results_all_power.csv`

Reference columns in `results/host_comparison/results.csv`:

- `igpu`
- `igpu_rocm_smi`
- `hybrid`
- `runlist`
- `offload`

The throughput plot uses `igpu` plus the NPU reference modes. The per-watt plot
uses the same measured iGPU throughput numerator with ROCm-SMI power samples in
`igpu_rocm_smi`, again alongside the NPU reference modes.

Canonical outputs:

- `results/host_comparison/results.csv`
- `results/host_comparison/effective_gflops_comparison.svg`
- `results/host_comparison/effective_gflops_per_watt_comparison.svg`
- `results/host_comparison/fairness_repeatability.csv`

Entry points:

- `python -m iron.applications.transformer_layer_new.study.host_comparison.run`
- `python -m iron.applications.transformer_layer_new.study.host_comparison.run_fairness_repeatability`

Environment on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

```bash
pip install -r requirements.txt
pip install -r iron/applications/transformer_layer_new/requirements.txt
```
