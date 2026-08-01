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

- `python3 -m iron.applications.transformer_layer.study.host_comparison.run`
- `python3 -m iron.applications.transformer_layer.study.host_comparison.run_fairness_repeatability`
- `python3 -m iron.applications.transformer_layer.study.host_comparison.remeasure_power_only`

`remeasure_power_only` preserves the existing host latency and throughput rows,
remeasures only iGPU power for the selected groups, updates the
`igpu_rocm_smi` power and per-watt columns, and regenerates the host-comparison
plots.

## Prerequisites

Environment on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

```bash
pip install -r requirements.txt
pip install -r iron/applications/transformer_layer/requirements.txt
```

This study has requirements the other studies do not. See the application
`README.md` `## Prerequisites` for the full list; the ones specific to here:

- **A ROCm torch build.** The repository `requirements.txt` pins the CPU-only
  wheel, which cannot run this study at all. `resolve_host_device` raises
  `Host comparison iGPU benchmark requires a ROCm-enabled torch build with a
  visible GPU` — and only at benchmark time, not at import.
- **`rocm-smi` on `PATH`**, for the default `--igpu-power-backend rocm-smi`.
- **NPU turbo mode and a sourced XRT**, as for every measured study:

  ```bash
  source /opt/xilinx/xrt/setup.sh
  source /path/to/iron/ironenv/bin/activate
  sudo xrt-smi configure --pmode turbo
  ```

## Why This Study Forces Reboots

This is the only study that needs a TTM page-limit change. The `seq_len=16384`
iGPU rows require a 26 GB TTM limit, which takes effect only after a reboot,
and restoring the normal limit afterwards takes another. In an unattended
suite run those are the two reboots the runner performs; it carries itself
across them with an `@reboot` crontab hook.

Running this study by hand at `seq_len=16384` therefore means managing
`amd-ttm` and the reboots yourself. Below 16384 no TTM change is needed.
