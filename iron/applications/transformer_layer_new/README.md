<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer New

`transformer_layer_new` is the study app for the transformer-layer execution
granularity paper work.

It keeps two NPU pattern implementations:

- `pattern/dataflow`
- `pattern/runlist`

The current paper-facing studies are:

- `study/block`
- `study/end_to_end`
- `study/memory_tile_staging`
- `study/igpu`
- `study/memcpy_bandwidth`

## Retained Surface

Families:

- `baseline_768`
- `baseline_1024`

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
- `results/igpu/results.csv`
- `results/igpu/fairness_repeatability.csv`
- `results/memcpy_bandwidth/results.csv`

The end-to-end helper studies default to `results_all_power.csv` when it is
present and fall back to `results.csv` otherwise.

`study/end_to_end/run_staging_ablation.py` is an end-to-end `dataflow`
benchmark sweep, not a metadata-only summary. It reruns the selected `dataflow`
config at different `mha_out_proj` and `ffn` staging depths, and can mirror the
depth ladder from `results/memory_tile_staging/results.csv`.

Current paper-facing figure scripts live in:

- `study/end_to_end/plot_tps_by_pattern.py`
- `study/end_to_end/run_latency_variation.py`
- `study/end_to_end/run_staging_ablation.py`
- `study/memory_tile_staging/plot_staging_depth.py`
- `study/igpu/run.py`

## Entry Points

- `python -m iron.applications.transformer_layer_new.study.block.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_power_sweep`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_correctness_spot_checks`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_latency_variation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_staging_ablation`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run_fairness_repeatability`
- `python -m iron.applications.transformer_layer_new.study.memory_tile_staging.run`
- `python -m iron.applications.transformer_layer_new.study.igpu.run`
- `python -m iron.applications.transformer_layer_new.study.igpu.run_fairness_repeatability`
- `python -m iron.applications.transformer_layer_new.study.memcpy_bandwidth.run`

## iGPU Environment

The repo-root [requirements.txt](/home/agi-demo/iron/requirements.txt) stays
generic for the wider codebase.

For the ROCm iGPU study on Ubuntu 24.04 / Python 3.12 Ryzen APU systems:

1. install the normal repo requirements first
2. then install the iGPU ROCm overlay:
   `pip install -r iron/applications/transformer_layer_new/requirements.txt`

## Documentation Map

- `docs/study_conventions.md`
- `study/block/README.md`
- `study/end_to_end/README.md`
- `study/memory_tile_staging/README.md`
- `study/igpu/README.md`
- `study/memcpy_bandwidth/README.md`
