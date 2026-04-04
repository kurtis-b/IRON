<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# memcpy-Bandwidth Study

Status: `implemented`

## Goal

Measure and analyze peak NPU memory bandwidth with the existing
`AIEMemCopy` microbenchmark operator.

This study is NPU-only and does not compare against the transformer patterns.

## Sweep Surface

Fixed transfer size:

- `8388608` elements
- `16777216` bytes input
- `33554432` total moved bytes

Default sweep dimensions:

- `2` channels with `num_cores in {2, 4, 8, 16}`
- default run compares only `bypass=True`

This keeps the processed data size fixed while comparing a small, interpretable
set of core/channel configurations. With `2` channels fixed, the sweep maps
directly to shim-tile usage:

- `2` cores -> `1` shim tile used
- `4` cores -> `2` shim tiles used
- `8` cores -> `4` shim tiles used
- `16` cores -> `8` shim tiles used

Each case uses a fixed `4096` element tile size. The transfer size is chosen so
that every tested configuration partitions into an exact number of full tiles.

If needed, kernel-mode rows can still be generated explicitly with
`--bypass all` or `--bypass false`.

## Output Contract

Canonical raw output is:

- `results/memcpy_bandwidth/results.csv`

Canonical plots are:

- `results/memcpy_bandwidth/bandwidth_by_shim_tiles.svg`

The plot is rendered as a single bar chart over shim-tile count, using
bypass-mode rows by default.

The CSV keeps one row per memcpy benchmark case and includes:

- `size_elements`
- `size_bytes`
- `total_moved_bytes`
- `num_cores`
- `num_channels`
- `bypass`
- `tile_size`
- `latency_us`
- `bandwidth_gbps`
- `run_status`
- `failure_message`
- `is_size_peak`
- `is_overall_peak`

`is_size_peak=True` marks the highest-bandwidth successful row for each
transfer size.

`is_overall_peak=True` marks the single highest-bandwidth successful row across
the full study output.

## Measurement

The study uses the existing shared `run_test(...)` helper with `AIEMemCopy`,
so the measured bandwidth matches the operator test convention:

- compile and prepare the runtime
- run warmup iterations
- time the NPU runlist execution
- validate the copied output
- compute effective bandwidth from total input-plus-output bytes

Default iteration schedule:

- `10` warmup iterations
- `500` timed iterations

## Owned Files

This study owns:

- `cases.py`
- `run.py`
- `test.py`
