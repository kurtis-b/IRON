<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer New

`transformer_layer_new` is the study app for the transformer-layer execution
granularity paper work.

It keeps two NPU pattern implementations:

- `pattern/dataflow` for the hybrid runlist+dataflow implementation
  The path still uses the legacy `dataflow` directory name, but the only
  supported execution-mode name is `hybrid`.
- `pattern/runlist`

The current paper-facing studies are:

- `study/block`
- `study/end_to_end`
- `study/memory_tile_staging`
- `study/resource_usage`
- `study/host_comparison`
- `study/memcpy_bandwidth`
- `study/roofline`

## Retained Surface

Workload variants:

- `encoder_bert`
- `decoder_gpt2`

Families:

- `tinybert_512`
- `baseline_768`
- `baseline_1024`
- `gpt2_small_768`
- `gpt2_medium_1024`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

The short-sequence end-to-end policy uses `100` timed iterations for `64`,
`128`, and `256`.

## Study Outputs

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

The end-to-end helper studies default to `results_all_power.csv` when it is
present and fall back to `results.csv` otherwise.

`study/end_to_end/run_staging_ablation.py` is an end-to-end `hybrid`
benchmark sweep, not a metadata-only summary. It reruns the selected `hybrid`
config at different `mha_out_proj` and `ffn` staging depths, and can mirror the
depth ladder from `results/memory_tile_staging/results.csv`.

Current paper-facing figure scripts live in:

- `study/end_to_end/plot_tps_by_pattern.py`
- `study/end_to_end/run_latency_variation.py`
- `study/end_to_end/run_staging_ablation.py`
- `study/memory_tile_staging/plot_staging_depth.py`
- `study/resource_usage/run.py`
- `study/host_comparison/run.py`
- `study/roofline/run.py`

## Recommended Run Order

There is no single wrapper that runs the full paper-facing study suite.
The canonical order is:

1. `python -m iron.applications.transformer_layer_new.study.block.run`
2. `python -m iron.applications.transformer_layer_new.study.end_to_end.run`
3. `python -m iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks`
4. `python -m iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation`
5. `python -m iron.applications.transformer_layer_new.study.end_to_end.run_staging_ablation`
6. `python -m iron.applications.transformer_layer_new.study.end_to_end.run_fairness_repeatability`
7. `python -m iron.applications.transformer_layer_new.study.memory_tile_staging.run`
8. `python -m iron.applications.transformer_layer_new.study.host_comparison.run`
9. `python -m iron.applications.transformer_layer_new.study.host_comparison.run_fairness_repeatability`
10. `python -m iron.applications.transformer_layer_new.study.resource_usage.run`
11. `python -m iron.applications.transformer_layer_new.study.memcpy_bandwidth.run`
12. `python -m iron.applications.transformer_layer_new.study.roofline.run`

Dependency notes:

- `memory_tile_staging` depends on completed block-study results.
- `host_comparison` depends on completed end-to-end results and does not rerun
  NPU patterns.
- `resource_usage` depends on both end-to-end results and the matching
  `build/transformer_layer_new_end_to_end` tree. If those compilation
  artifacts are absent, it will emit `missing_artifact` rows instead of
  recompiling.
- `roofline` depends on completed end-to-end results, end-to-end tuning
  results, and memcpy-bandwidth results. It is a pure postprocessing study
  and now emits separate roofline plots for each `(compute_tiles_used, shim_tiles_used)`
  bucket. It does not rerun NPU benchmarks.

## NPU Performance Mode

Before running NPU benchmark studies, set the NPU power mode to `turbo`:

- `sudo xrt-smi configure --pmode turbo`

Verify the current setting with:

- `xrt-smi examine -r all`

The NPU benchmarking studies perform an `xrt-smi` turbo-mode check before each
measured NPU datapoint. If the reported mode is not `turbo`, the benchmark
call fails before writing that datapoint.

## Entry Points

- `python -m iron.applications.transformer_layer_new.study.block.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_staging_ablation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_fairness_repeatability`
- `python -m iron.applications.transformer_layer_new.study.memory_tile_staging.run`
- `python -m iron.applications.transformer_layer_new.study.resource_usage.run`
- `python -m iron.applications.transformer_layer_new.study.host_comparison.run`
- `python -m iron.applications.transformer_layer_new.study.host_comparison.run_fairness_repeatability`
- `python -m iron.applications.transformer_layer_new.study.memcpy_bandwidth.run`
- `python -m iron.applications.transformer_layer_new.study.roofline.run`

## Host Comparison Environment

The repo-root [requirements.txt](/home/agi-demo/iron/requirements.txt) stays
generic for the wider codebase.

For the iGPU host comparison on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

1. install the normal repo requirements first
2. then install the ROCm overlay used by the iGPU path:
   `pip install -r iron/applications/transformer_layer_new/requirements.txt`

The canonical host comparison study is iGPU-only. It writes one throughput
series and two iGPU per-watt series using `rocm-smi` and `turbostat_pkgwatt`
from the same measured iGPU throughput run.

## Documentation Map

- `docs/study_conventions.md`
- `study/block/README.md`
- `study/end_to_end/README.md`
- `study/memory_tile_staging/README.md`
- `study/resource_usage/README.md`
- `study/host_comparison/README.md`
- `study/memcpy_bandwidth/README.md`
- `study/roofline/README.md`
