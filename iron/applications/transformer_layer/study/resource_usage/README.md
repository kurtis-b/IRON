<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Resource Usage Study

`study/resource_usage` is an export-only study that reads the build artifacts
already produced by the end-to-end study and saves structured resource-usage
rows without recompiling.

It writes:

- `results/resource_usage/dataflow_block_best_configs.csv`
- `results/resource_usage/hybrid_selected_ops.csv`
- `results/resource_usage/runlist_selected_ops.csv`
- `results/resource_usage/offload_selected_ops.csv`

Saved metrics come from `input_physical.mlir`:

- compute tiles used
- allocated bytes on AIE tiles
- allocated bytes on memory tiles
- shim DMA `S2MM` channel usage
- shim DMA `MM2S` channel usage
- compute-tile memory utilization min/max/mean/median
- memory-tile memory utilization min/max/mean/median
- memory-tile and compute-tile DMA usage when explicit physical allocations exist

Utilization is normalized to the capacity of the tiles touched by that design:

- AIE tile memory: `65536` bytes per tile
- memory tile memory: `524288` bytes per tile
- shim DMA: `2` channels per direction per shim tile

The runner depends on the end-to-end study in two ways:

- it uses the end-to-end results CSVs to enumerate selected `hybrid`,
  `runlist`, and `offload` rows
- it reuses the existing build tree under `build/transformer_layer_end_to_end`

If the expected compilation artifact is missing, the runner still writes a CSV
row and records the missing-artifact note instead of compiling.

In practice, that means the most useful export is produced immediately after the
matching end-to-end run set. If the selected results CSVs and the available
build tree come from different runs, the export may legitimately contain
`missing_artifact` rows.

Default command:

```bash
python3 -m iron.applications.transformer_layer.study.resource_usage.run
```

Useful filters:

```bash
python3 -m iron.applications.transformer_layer.study.resource_usage.run \
  --scope hybrid_ops \
  --family baseline_768 \
  --seq-len 1024
```
