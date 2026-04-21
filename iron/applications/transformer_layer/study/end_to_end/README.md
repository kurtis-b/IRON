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

For each `(campaign_id, repeat_index, family, seq_len, execution_mode)`, the
runner benchmarks every isolated operator candidate, validates the top `3`
estimated joint full-pattern compositions per mode, selects the best passing
joint config found under that declared search budget, and records a paired
final measurement consisting of an uninstrumented latency run plus a matched
power rerun. The canonical outputs are `results/end_to_end/results_all_power.csv`
and `results/end_to_end/tuning_all_power.csv`; the former is also the reference
input for `study/host_comparison`.

Candidate payloads are family- and sequence-specific. The loader in `cases.py`
still injects a small set of low-footprint short-sequence fallback candidates
for `hybrid` and `runlist`, and candidate removals may also be applied from the
checked-in CSV manifests. The canonical campaign manifest publishes both the
mode-level candidate policy and the effective candidate inventory seen by the
runner for every `(family, seq_len, mode)`, including candidate IDs and counts
by operator.

Helper outputs:

- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`

Validation policy:

- full reference-output validation for every paper-facing sequence length
- paper-facing rows use the published `reference_tolerance_validation` policy:
  `rtol=0.1`, `atol=0.5`, `max_error_fraction=0.05`
- long-sequence runlist selections must pass a composed full-pattern smoke
- correctness spot checks remain auxiliary, but `512` and `2048` are both
  full reference-output reruns under the same published tolerances

Power policy:

- `avg_power_w` is an incremental package-power estimate, not an isolated NPU
  rail reading
- NPU power rows export the package boundary, estimation method, baseline
  policy, baseline average power, active average package power, and sensor
  source explicitly
- `remeasure_power_only` remains an exploratory maintenance tool and is not part
  of the canonical paper-number generation path
- final rows publish the joint search policy, the number of joint candidates
  considered, the number evaluated, and whether the selected config is
  exhaustive-best or heuristic-best

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

Resume reuse is manifest-gated. The canonical runner only reuses prior tuning or
final rows when the sibling `campaign_manifest.json` matches the requested
`campaign_id`, the study name, and the current git SHA.

`run_power_sweep` and `remeasure_power_only` are exploratory workflows. They
write under `results_exploratory/` and refuse to overwrite the canonical paper
CSV set under `results/`.

`remeasure_power_only` reuses existing latency and throughput values and reruns
only the NPU power measurement.
