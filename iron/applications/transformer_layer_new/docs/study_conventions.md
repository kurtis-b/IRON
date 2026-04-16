<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Study Conventions

This note defines the current shared rules for the implemented
`transformer_layer_new` studies.

## Implemented Scope

- `block`
- `end_to_end`
- `memory_tile_staging`
- `resource_usage`
- `host_comparison`
- `memcpy_bandwidth`
- `roofline`

## Retained Cases

Current families:

- `tinybert_512`
  `hidden_size=512`, `intermediate_size=2048`, `num_attention_heads=8`
- `baseline_768`
  `hidden_size=768`, `intermediate_size=3072`, `num_attention_heads=12`
- `baseline_1024`
  `hidden_size=1024`, `intermediate_size=4096`, `num_attention_heads=16`
- `gpt2_small_768`
  `hidden_size=768`, `intermediate_size=3072`, `num_attention_heads=12`,
  `workload_variant=decoder_gpt2`
- `gpt2_medium_1024`
  `hidden_size=1024`, `intermediate_size=4096`, `num_attention_heads=16`,
  `workload_variant=decoder_gpt2`

Deferred family:

- `baseline_2048`
  `hidden_size=2048`, `intermediate_size=8192`, `num_attention_heads=32`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

## End-to-End Rules

The retained NPU execution modes are:

- `hybrid`
- `runlist`

The end-to-end study uses checked-in candidate files:

- `study/end_to_end/hybrid_candidates.json`
- `study/end_to_end/runlist_candidates.json`

The shared synthetic generator lives in:

- `pattern/reference.py`

The main end-to-end validation policy is:

- exact reference validation through `seq_len=512`
- finite-output validation above `512`

The paper-facing spot-check runner strengthens that surface with exact checks at:

- `seq_len=512`
- `seq_len=2048`

The short-sequence iteration policy is:

- `64`, `128`, `256`
  `warmup_runs=1`, `runs_per_sample=100`

## Result Contracts

Canonical CSV outputs:

- `results/block/results.csv`
- `results/end_to_end/results_all_power.csv`
- `results/end_to_end/tuning_all_power.csv`
- `results/end_to_end/correctness_spot_checks.csv`
- `results/end_to_end/latency_variation.csv`
- `results/end_to_end/staging_ablation.csv`
- `results/end_to_end/fairness_repeatability.csv`
- `results/memory_tile_staging/results.csv`
- `results/resource_usage/dataflow_block_best_configs.csv`
- `results/resource_usage/hybrid_selected_ops.csv`
- `results/resource_usage/runlist_selected_ops.csv`
- `results/host_comparison/results.csv`
- `results/host_comparison/fairness_repeatability.csv`
- `results/memcpy_bandwidth/results.csv`
- `results/roofline/kernel_points.csv`
- `results/roofline/implementation_points.csv`
- `results/roofline/kernel_roofline_*_tiles.svg`
- `results/roofline/implementation_roofline_*_tiles.svg`

The end-to-end CSV keeps one row per
`(workload_variant, study_case_id, seq_len, execution_mode)`.

The end-to-end tuning CSV keeps one row per
`(workload_variant, study_case_id, seq_len, execution_mode, internal_operator, candidate_id)`
and marks the fastest passing isolated candidate with `is_operator_best=True`.

The host-comparison CSV keeps one comparison row per `(study_case_id, seq_len, metric)`.
Its comparison columns are:

- `igpu`
- `igpu_rocm_smi`
- `hybrid`

The throughput row uses `igpu` and `hybrid`.
The per-watt row uses the same iGPU throughput numerator with the ROCm-SMI
power backend:

- `igpu_rocm_smi`
- `hybrid`

## Supporting Studies

The memory-tile staging CSV keeps one row per
`(family_id, seq_len, block_kind, staging_depth)`.

The resource-usage exports are compile-artifact summaries. They reuse the
existing `build/transformer_layer_new_end_to_end` tree, do not recompile, and
record missing-artifact notes when the expected physical MLIR is absent.

Meaningful full exports therefore require the build tree from the same
end-to-end run set as the selected results CSVs.

The end-to-end staging-ablation CSV keeps one row per
`(study_case_id, seq_len, block_kind, staging_depth)` for real `hybrid`
reruns. It sweeps the selected `hybrid` config at different staging depths
and can reuse the memory-tile staging depth ladder when that CSV is present.

For unattended full-suite execution, use
`study/unattended_reboot.py`. It keeps running normal jobs in one session,
reboots only when a TTM transition is required for the iGPU `16384`
host-comparison rows, and ends with `study/regenerate_plots.py` to refresh the
summary plot suite from the collected results root after the normal TTM state
has been restored. The unattended runner re-sources `/opt/xilinx/xrt/setup.sh`
and the repo-local `ironenv` virtualenv before each resumed `python3`
invocation after reboot.

If a run is stopped intentionally, resume it with
`python3 -m iron.applications.transformer_layer_new.study.unattended_reboot resume --state <state.json>`.
When the run was started with `sudo`, the reboot hook is managed in the root
crontab via `crontab`, not by direct edits under `/var/spool/cron/crontabs`.

The host-comparison study remains separate from `study/end_to_end` by design.
It consumes completed end-to-end NPU rows instead of re-running NPU patterns.

The roofline study is a postprocessing study. It combines:

- final end-to-end throughput rows
- isolated end-to-end tuning rows
- peak memcpy-bandwidth rows

It does not rerun NPU benchmarks and instead derives arithmetic intensity
analytically from the selected workload and operator configs.

## Recommended Execution Order

The retained study order is:

1. `block`
2. `end_to_end`
3. end-to-end helper studies:
   `correctness_spot_checks`, `latency_variation`, `staging_ablation`,
   `fairness_repeatability`
4. `memory_tile_staging`
5. `host_comparison`
6. `resource_usage`
7. `memcpy_bandwidth`
8. `roofline`

The important dependency edges are:

- `memory_tile_staging` depends on `block`
- `host_comparison` depends on `end_to_end`
- `resource_usage` depends on `end_to_end` results plus the matching build tree
- `roofline` depends on `end_to_end`, end-to-end tuning, and `memcpy_bandwidth`

## NPU Benchmark Environment

Before running NPU benchmark studies, set the NPU power mode to `turbo`:

- `sudo xrt-smi configure --pmode turbo`

Verify the setting with:

- `xrt-smi examine -r all`

The NPU-running benchmark helpers perform an `xrt-smi` turbo-mode check before
each measured NPU datapoint. If the reported mode is not `turbo`, the
benchmark call fails before writing that datapoint.
