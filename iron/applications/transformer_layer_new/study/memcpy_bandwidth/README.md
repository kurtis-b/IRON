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

Default transfer-size ladder:

- `1024`
- `2048`
- `4096`
- `8192`
- `16384`
- `32768`
- `65536`
- `131072`

Default sweep dimensions:

- `num_cores=1..16`
- `num_channels in {1, 2}`
- `bypass in {False, True}`

The case generator applies the same validity rules as the existing
`iron/operators/mem_copy/test.py` surface:

- `num_cores <= 8 * num_channels`
- `num_cores >= num_channels`
- `size_elements % num_cores == 0`
- `tile_size = size_elements // num_cores`
- `tile_size <= 8192`

If capping `tile_size` to `8192` would change the requested total size, that
case is skipped.

## Output Contract

Canonical raw output is:

- `results/memcpy_bandwidth/results.csv`

Canonical plots are:

- `results/memcpy_bandwidth/peak_bandwidth_by_size.svg`
- `results/memcpy_bandwidth/latency_by_size.svg`

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

## Owned Files

This study owns:

- `cases.py`
- `run.py`
- `test.py`

