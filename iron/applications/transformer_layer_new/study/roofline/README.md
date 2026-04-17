<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Roofline Study

Status: `implemented`

## Goal

Generate roofline plots from existing study outputs without rerunning the NPU.

This study combines:

- end-to-end final throughput results
- end-to-end tuning results for isolated internal operators
- memcpy-bandwidth peak results

It emits:

- one kernel roofline plot per `(compute_tiles_used, shim_tiles_used)` bucket,
  showing the best measured isolated kernel configuration for each logical
  operator in that bucket
- one implementation roofline plot per `(compute_tiles_used, shim_tiles_used)`
  bucket, showing the best full-pattern `hybrid` and `runlist` results in that
  bucket

## Dependencies

This study depends on:

- `results/end_to_end/results_all_power.csv`
- `results/end_to_end/tuning_all_power.csv`
- `results/memcpy_bandwidth/results.csv`

When the canonical `results/` tree is absent, the runner falls back to the
checked-in `results_final/` tree and then to `results_commit*` snapshots for
the end-to-end CSVs. There is no snapshot fallback for memcpy bandwidth; that
CSV must exist explicitly.

## Output Contract

Canonical outputs:

- `results/roofline/kernel_points.csv`
- `results/roofline/implementation_points.csv`
- `results/roofline/kernel_roofline_*_tiles.svg`
- `results/roofline/implementation_roofline_*_tiles.svg`

PNG variants are emitted alongside the SVGs.

The plot path arguments act as base names. The runner appends
`_<NN>_compute_tiles_<MM>_shim_tiles` to separate the rooflines by resource
bucket.

The kernel CSV keeps one row per:

- `(workload_variant, study_case_id, execution_mode, logical_operator)`

using the best measured isolated throughput row across the retained sequence
ladder. If an operator has no measured isolated throughput row, the study still
emits a CSV row with an omission note.

The implementation CSV keeps one row per:

- `(workload_variant, study_case_id, execution_mode)`

using the best measured full-pattern throughput row across the retained
sequence ladder.

## Method

Roofline arithmetic intensity is derived analytically from the workload shape
and the selected operator config. Compute-tile counts are also derived
analytically from the selected operator config so the plots can be separated by
tile count. The memory-bandwidth ceiling comes from the highest-bandwidth
successful memcpy row.

Kernel throughput comes from isolated tuning rows.
Implementation throughput comes from the final end-to-end rows.

Pure data-movement kernels such as `k_transpose` are recorded in the CSV with an
omission note and are not plotted on the log-scale compute roofline.

## Entry Point

```bash
python -m iron.applications.transformer_layer_new.study.roofline.run
```

Useful filters:

```bash
python -m iron.applications.transformer_layer_new.study.roofline.run \
  --family baseline_768
```
