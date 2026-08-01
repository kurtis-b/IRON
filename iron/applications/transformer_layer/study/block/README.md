<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Block Study

Status: `implemented`

## Goal

Benchmark the unique block operators through one shared workload entry
point and a manually editable candidate table.

Compared block kinds:

- `qkv_proj`
- `mha_out_proj`
- `mha_out_proj_causal`
- `addnorm`
- `layer_norm`
- `elementwise_add`
- `ffn`

`addnorm` is the shared operator used at both residual positions in the full
transformer layer.
This first study pass explores that operator once per workload rather than as
separate logical positions.

## Shared Entry Point

Every block-study case starts from:

- `seq_len`
- `head_dim`
- `num_heads`
- `ffn_dim`

The runner derives `hidden_size = head_dim * num_heads`.

Per-block candidate tuples then map those workload values into operator-local
configuration choices.

The case table lives in:

- `study/block/cases.py`

That file is the intended edit point for manually changing candidate tuples per
family and per sequence length.

## Case Matrix

Families:

- `tinybert_512`
- `baseline_768`
- `baseline_1024`
- `gpt2_512`
- `gpt2_small_768`
- `gpt2_medium_1024`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

Total workload surface:

- `6 families x 9 seq lengths x 7 block kinds`

Each workload point may expand to one or more candidate tuples per block kind.

## Runner

Entrypoint:

- `python -m iron.applications.transformer_layer.study.block.run`

Environment setup:

- `source /opt/xilinx/xrt/setup.sh`
- `source /path/to/iron/ironenv/bin/activate`
- `sudo xrt-smi configure --pmode turbo`
- verify with `xrt-smi examine -r all`

The benchmark helper checks `xrt-smi` before each measured candidate row and
fails that measurement if the reported NPU power mode is not `turbo`.

Selection flags:

- `--family`
- `--seq-len`
- `--block`
- `--warmup-iters`
- `--timed-iters`
- `--output`

Default iteration schedule:

- `seq_len <= 2048`: `1` warmup, `10` timed
- `seq_len == 4096`: `1` warmup, `5` timed
- `seq_len >= 8192`: `1` warmup, `2` timed

## Outputs

Canonical output:

- `results/block/results.csv`

The CSV keeps one row per candidate and marks the minimum-latency successful
candidate per `(family_id, seq_len, block_kind)` as `is_best=True`.

Visualization helper:

- `python -m iron.applications.transformer_layer.study.block.plot_best_latency`
- `python -m iron.applications.transformer_layer.study.block.plot_best_latency --variant slides`

That command renders:

- `results/block/best_latency_by_block.png`
- `results/block/best_latency_by_block.svg`
- `results/block/best_latency_by_block_slides.png`
- `results/block/best_latency_by_block_slides.svg`
