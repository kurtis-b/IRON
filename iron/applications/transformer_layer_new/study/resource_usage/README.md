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
- `results/resource_usage/runlist_selected_ops.csv`

Saved metrics come from `input_physical.mlir`:

- compute tiles used
- allocated bytes on AIE tiles
- allocated bytes on memory tiles
- shim DMA `S2MM` channel usage
- shim DMA `MM2S` channel usage

Utilization is normalized to the capacity of the tiles touched by that design:

- AIE tile memory: `65536` bytes per tile
- memory tile memory: `524288` bytes per tile
- shim DMA: `2` channels per direction per shim tile

The runner depends on the end-to-end study in two ways:

- it uses the end-to-end results CSVs to enumerate selected `runlist` rows
- it reuses the existing build tree under `build/transformer_layer_new_end_to_end`

If the expected compilation artifact is missing, the runner still writes a CSV
row and records the missing-artifact note instead of compiling.

Default command:

```bash
python -m iron.applications.transformer_layer_new.study.resource_usage.run
```

Useful filters:

```bash
python -m iron.applications.transformer_layer_new.study.resource_usage.run \
  --scope dataflow_blocks \
  --family baseline_768 \
  --seq-len 1024
```
