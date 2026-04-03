<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Block Study

Status: `implemented`

## Goal

Benchmark the four unique block operators through one shared workload entry
point and a manually editable candidate table.

Compared block kinds:

- `qkv_proj`
- `mha_out_proj`
- `addnorm`
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

- `baseline_768`
- `baseline_1024`

Sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

Total workload surface:

- `2 families x 9 seq lengths x 4 block kinds`

Each workload point may expand to one or more candidate tuples per block kind.

## Runner

Entrypoint:

- `python -m iron.applications.transformer_layer_new.study.block.run`

Environment setup:

- `source /opt/xilinx/xrt/setup.sh`
- `source ironenv/bin/activate`

Selection flags:

- `--family`
- `--seq-len`
- `--block`
- `--warmup-iters`
- `--timed-iters`
- `--output`

## Outputs

Canonical output:

- `results/block/results.csv`

The CSV keeps one row per candidate and marks the minimum-latency successful
candidate per `(family_id, seq_len, block_kind)` as `is_best=True`.
