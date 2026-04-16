<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Memory-Tile Staging Study

Status: `implemented`

## Goal

Benchmark how memory-tile staging depth changes block latency for:

- `mha_out_proj`
- `ffn`

This study is block-only.
It does not retune the full transformer layer.

## Dependency

This study depends on:

- `results/block/results.csv`

It reuses the current fastest passing block-study candidate at each:

- `family_id`
- `seq_len`
- `block_kind`

Then it freezes all non-depth config fields and sweeps only the staging depth:

- `mha_out_proj`: `o_proj_acc_depth`
- `ffn`: `down_proj_depth`

## Scope

The study only benchmarks already supported divisible configurations.

Supported sweep condition per block:

- `mha_out_proj`: `hidden_size % (emb_tile * o_proj_acc_depth) == 0`
- `ffn`: `hidden_size % (tile_k * down_proj_depth) == 0`

Non-divisible tail handling is out of scope for this study.

## Output Shape

Canonical output:

- `results/memory_tile_staging/results.csv`

The CSV keeps one row per:

- `(family_id, seq_len, block_kind, staging_depth)`

Important fields include:

- `source_candidate_index`
- `source_staging_depth`
- `staging_depth`
- `avg_latency_ms`
- `bandwidth_gbps`
- `speedup_vs_depth1`
- `is_best_depth`

Canonical plots:

- `results/memory_tile_staging/mha_out_proj_latency_by_staging_depth.svg`
- `results/memory_tile_staging/mha_out_proj_speedup_by_staging_depth.svg`
- `results/memory_tile_staging/ffn_latency_by_staging_depth.svg`
- `results/memory_tile_staging/ffn_speedup_by_staging_depth.svg`

PNG versions are emitted alongside the SVGs.
The canonical plots focus on `seq_len=256` through `8192` for readability,
while the CSV retains the full sweep.

## Runner

Entrypoint:

- `python -m iron.applications.transformer_layer_new.study.memory_tile_staging.run`

Environment setup:

- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `sudo xrt-smi configure --pmode turbo`
- verify with `xrt-smi examine -r all`

The underlying block benchmark helper checks `xrt-smi` before each measured
staging-depth row and fails that measurement if the reported NPU power mode is
not `turbo`.

Selection flags:

- `--family`
- `--seq-len`
- `--block`
- `--reference-input`
- `--warmup-iters`
- `--timed-iters`
- `--output`

Default reference input:

- `results/block/results.csv`

Default iteration schedule matches the block study:

- `seq_len <= 2048`: `1` warmup, `10` timed
- `seq_len == 4096`: `1` warmup, `5` timed
- `seq_len >= 8192`: `1` warmup, `2` timed

## Plot Helper

Standalone plot entrypoint:

- `python -m iron.applications.transformer_layer_new.study.memory_tile_staging.plot_staging_depth`

Optional focused rendering:

- `--block mha_out_proj|ffn`
- `--metric latency|speedup`
