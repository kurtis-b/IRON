<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Study

This study benchmarks the retained full-layer NPU modes:

- `hybrid`
- `runlist`
- `offload`

Paper terminology maps those repo modes onto offload and runlist boundaries:

| Paper label | Repo evidence | Interpretation |
| --- | --- | --- |
| `offload` | `offload` | Host runtime executes the transformer layer and offloads GEMM kernels to the NPU. |
| `runlist` | `runlist` | The layer is decomposed into fine-grained NPU operators with explicit intermediates. |
| `coarse runlist` | `hybrid` selected blocks | Runlist orchestration over fused/staged NPU kernels that carry bandwidth-heavy regions. |

The proposed coarse runlist path uses the internal `hybrid` mode: it uses
runtime sequencing around fused/staged coarse kernels rather than claiming a
separate pure full-layer dataflow baseline.

Configuration sources:

- `hybrid_candidates.json`
- `runlist_candidates.json`
- `offload_candidates.json`

For each `(family, seq_len, execution_mode)`, the runner benchmarks internal
operator candidates, selects the fastest passing config for each operator, and
then benchmarks the resolved full path once. The main result is
`results/end_to_end/results_all_power.csv`, which is also the reference input
for `study/host_comparison`.

Candidate payloads are family- and sequence-specific. The loader in
`cases.py` also injects a small set of low-footprint short-sequence fallback
candidates for `hybrid` and `runlist`, so the checked-in JSON files are not the
entire effective search space.

A completed end-to-end run can be converted into fixed-winner candidate JSONs
for future reruns:

```bash
python3 -m iron.applications.transformer_layer.study.end_to_end.export_fixed_winner_candidates \
  --results <root>/end_to_end/results_all_power.csv \
  --output-dir <root>/end_to_end/fixed_winner_candidates
```

Pass that directory back to the end-to-end runner with `--candidate-dir` to
time the selected operator configs and full paths without searching the full
candidate matrix.

Helper outputs:

- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/selected_component_timings.csv`
- `results/end_to_end/selected_component_aggregates.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`

Validation policy:

- exact reference validation through `seq_len=512`
- finite-output validation above `512`
- correctness spot checks at `512`, `2048`, and `8192`

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
- `python3 -m iron.applications.transformer_layer.study.end_to_end.run_selected_component_aggregates`
- `python3 -m iron.applications.transformer_layer.study.end_to_end.remeasure_power_only`

`remeasure_power_only` reuses the existing latency and throughput values from
`results_all_power.csv`, reruns only the NPU power measurement, and rewrites
the per-watt columns without retuning or rerunning latency.

`run_selected_component_aggregates` builds the selected-component comparison
inputs used by the coarse runlist, runlist, and offload stacked latency plots.
It reads `results_all_power.csv`, writes detailed component timings to
`selected_component_timings.csv`, and writes grouped aggregates to
`selected_component_aggregates.csv`.

By default, selected NPU operator timings are reused from the sibling
`tuning_all_power.csv` when the selected candidate id, resolved operator
config, family, workload variant, execution mode, operator, and sequence length
match a passed tuning row. Missing NPU rows are rebenchmarked in `auto` mode;
`--npu-source tuning` fails closed with missing rows instead, and
`--npu-source rebenchmark` keeps the old fresh-measurement path. Offload host
groups are still freshly profiled because tuning only covers isolated NPU
GEMM kernels. Coarse runlist rows reuse the selected fused/staged coarse-kernel
timings from tuning and compare their sum to the full coarse runlist path.
Existing detailed rows can be reused with `--resume-input` to avoid restarting
completed host-group profiling. Use `--detailed-only` for checkpointed detail
jobs and a final `--aggregate-only` pass to build the aggregate CSV from the
current detail rows; `--seq-len` accepts comma-separated lists such as
`512,2048,8192`.
