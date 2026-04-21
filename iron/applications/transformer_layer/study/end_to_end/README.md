<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Study

This study benchmarks the retained full-layer NPU modes:

- `hybrid`
- `runlist`
- `offload`

Configuration sources:

- `hybrid_candidates.json`
- `runlist_candidates.json`
- `offload_candidates.json`

For each `(family, seq_len, execution_mode)`, the runner benchmarks internal
operator candidates, selects the fastest passing config for each operator, and
then benchmarks the resolved full pattern once. The main result is
`results/end_to_end/results_all_power.csv`, which is also the reference input
for `study/host_comparison`.

Candidate payloads are family- and sequence-specific. The loader in
`cases.py` also injects a small set of low-footprint short-sequence fallback
candidates for `hybrid` and `runlist`, so the checked-in JSON files are not the
entire effective search space.

Helper outputs:

- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`

Validation policy:

- exact reference validation through `seq_len=512`
- finite-output validation above `512`
- correctness spot checks at `512` and `2048`

Environment:

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all
```

Entry points:

- `python3 -m iron.applications.transformer_layer.study.end_to_end.run`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_power_sweep`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_correctness_spot_checks`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_latency_variation`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_staging_ablation`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_fairness_repeatability`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.remeasure_power_only`

`remeasure_power_only` reuses the existing latency and throughput values from
`results_all_power.csv`, reruns only the NPU power measurement, and rewrites
the per-watt columns without retuning or rerunning latency.
