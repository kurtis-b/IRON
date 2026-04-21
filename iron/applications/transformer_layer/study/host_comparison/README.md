<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Comparison Study

This study benchmarks the ROCm iGPU path and joins those measurements with the
completed end-to-end NPU rows from the same canonical campaign and repeat. The
join requires exactly one selected NPU row per
`(campaign_id, repeat_index, study_case_id, seq_len, execution_mode)`.
Each joined reference group must contain all three retained NPU modes
(`hybrid`, `runlist`, `offload`) from one campaign only, and those NPU rows
must already be validated with the canonical `reference_tolerance_validation`
policy from the end-to-end study.

Dependencies:

- `results/end_to_end/results_all_power.csv`

Reference columns in `results/host_comparison/results.csv`:

- `igpu`
- `igpu_rocm_smi`
- `hybrid`
- `runlist`
- `offload`

The throughput plot uses `igpu` plus the NPU reference modes. The per-watt plot
uses the same measured iGPU throughput numerator with ROCm-SMI
delta-package-power samples in `igpu_rocm_smi`, again alongside the NPU
reference modes. The CSV records the power-boundary policy family, the NPU/iGPU
estimation methods, and both baseline policies so the comparison does not
silently mix incompatible power semantics.

Resume reuse is manifest-gated. Existing host-comparison rows are only reused
when the sibling `campaign_manifest.json` matches the requested `campaign_id`,
the host-comparison study name, and the current git SHA.

Canonical outputs:

- `results/host_comparison/results.csv`
- `results/host_comparison/effective_gflops_comparison.svg`
- `results/host_comparison/effective_gflops_per_watt_comparison.svg`
- `results/host_comparison/fairness_repeatability.csv`

Entry points:

- `python3 -m iron.applications.transformer_layer.study.host_comparison.run`
- `python3 -m iron.applications.transformer_layer.study.host_comparison.run_fairness_repeatability`
- `python3 -m iron.applications.transformer_layer.study.host_comparison.remeasure_power_only`

`remeasure_power_only` preserves the existing host latency and throughput rows,
remeasures only iGPU power for the selected groups, updates the
`igpu_rocm_smi` power and per-watt columns, and regenerates the host-comparison
plots. It writes exploratory refreshes under `results_exploratory/` and refuses
to overwrite the canonical `results/host_comparison/results.csv`. It remains an
exploratory/support workflow and is not part of the canonical paper-number
generation path.

Environment on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

```bash
pip install -r requirements.txt
pip install -r iron/applications/transformer_layer/requirements.txt
```
