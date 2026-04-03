<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer New

`transformer_layer_new` is the smaller thesis-local study app for the
Transformer Layer design-pattern work.

It starts from the existing pattern implementations in:

- `pattern/dataflow`
- `pattern/runlist`
- `pattern/offload`

The remaining work in this app is the study harness and study documentation.

## Current Status

- the block implementations are complete
- the first implementation pass is the shared block-study entry point under
  `study/block`
- study infrastructure is being added here instead of extending the older
  `transformer_layer` app

Implementation order:

1. `block`
2. `end_to_end`
3. `reconfiguration_overhead`
4. `igpu`

## Current Retained Surface

Current implementation families:

- `768 / 3072 / 12`
- `1024 / 4096 / 16`

Deferred family:

- `2048 / 8192 / 32`

Full sequence ladder:

- `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

Reconfiguration-overhead subset:

- `64, 512, 16384`

## Study Layout

Each study gets its own directory under `study/`.

Current structure:

- `docs/`
  shared study rules and current block-study conventions
- `pattern/`
  existing end-to-end implementations
- `study/block/`
  shared block entry point, editable case table, tests, and docs
- `study/end_to_end/`
  end-to-end study runner and docs
- `study/reconfiguration_overhead/`
  reconfiguration-overhead study runner and docs
- `study/igpu/`
  iGPU comparison runner and docs
- `results/`
  canonical result locations per study

## Study Outputs

Current canonical result CSV:

- `results/block/results.csv`

Later planned result CSVs:

- `results/end_to_end/results.csv`
- `results/reconfiguration_overhead/results.csv`
- `results/igpu/results.csv`

Current study entrypoint:

- `python -m iron.applications.transformer_layer_new.study.block.run`

Later planned entrypoints:

- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`
- `python -m iron.applications.transformer_layer_new.study.reconfiguration_overhead.run`
- `python -m iron.applications.transformer_layer_new.study.igpu.run`

## Documentation Map

- `docs/study_conventions.md`
  shared study rules, naming, and current block-study result conventions
- `study/block/README.md`
  block-study contract and shared entry point
- `study/end_to_end/README.md`
  end-to-end study contract
- `study/reconfiguration_overhead/README.md`
  reconfiguration-overhead study contract
- `study/igpu/README.md`
  iGPU study contract
