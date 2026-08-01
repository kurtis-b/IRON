<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# memcpy-Bandwidth Study

Status: `implemented`

## Goal

Measure and analyze peak NPU memory bandwidth with the existing
`AIEMemCopy` microbenchmark operator.

This study is NPU-only and does not compare against the transformer execution
paths.

Before running it, set the NPU power mode to `turbo`:

- `sudo xrt-smi configure --pmode turbo`
- verify with `xrt-smi examine -r all`

The benchmark helper checks `xrt-smi` before each measured memcpy row and
fails that measurement if the reported NPU power mode is not `turbo`.

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

## Entry Point

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

sudo xrt-smi configure --pmode turbo
python3 -m iron.applications.transformer_layer.study.memcpy_bandwidth.run
```

A default run measures the four bypass rows and takes well under a minute.
There are no family or sequence-length filters — the transfer size is fixed.
Useful flags:

- `--bypass` — `true` (default), `false`, or `all`; `all` gives the 8 rows an
  unattended suite run produces
- `--num-cores` — one of `2, 4, 8, 16`, or `all` (default)
- `--num-channels` — or `all` (default)
- `--warmup-iters` (default `10`), `--timed-iters` (default `500`)
- `--output`, `--resume-input`, `--no-resume`
- `--bandwidth-plot`

This study has no dependencies on other studies, so it is a good first check
that the NPU and turbo mode are working.

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
