<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer New

`transformer_layer_new` is the thesis-local study app for the Transformer Layer
design-pattern work.

It starts from the existing pattern implementations in:

- `pattern/dataflow`
- `pattern/runlist`
- `pattern/offload`

Current offload semantics:

- one shared xclbin across the offloaded GEMMs
- `q_proj`, `k_proj`, `v_proj`, `attn_scores`, `attn_output`, `out_proj`,
  `ffn_up`, and `ffn_down` on NPU
- host softmax, GeLU, and residual add/layer norm

The remaining work in this app is the study harness and study documentation.

## Current Status

- the block implementations are complete
- the implemented study runners are `study/block` and `study/end_to_end`
- study infrastructure lives here

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
  shared study rules and the current block/end-to-end conventions
- `pattern/`
  existing end-to-end implementations
- `study/block/`
  shared block entry point, editable case table, tests, and docs
- `study/end_to_end/`
  workload-based end-to-end study runner, JSON tuning defaults, power helper,
  tests, and docs
- `study/reconfiguration_overhead/`
  reconfiguration-overhead study runner and docs
- `study/igpu/`
  iGPU comparison runner and docs
- `results/`
  canonical result locations per study

## Study Outputs

Current canonical result CSVs:

- `results/block/results.csv`
- `results/end_to_end/results.csv`
- `results/end_to_end/tuning.csv`

Later planned result CSVs:

- `results/reconfiguration_overhead/results.csv`
- `results/igpu/results.csv`

Current study entrypoints:

- `python -m iron.applications.transformer_layer_new.study.block.run`
- `python -m iron.applications.transformer_layer_new.study.end_to_end.run`

The end-to-end study also uses checked-in default candidate files:

- `study/end_to_end/dataflow_candidates.json`
- `study/end_to_end/runlist_candidates.json`
- `study/end_to_end/offload_candidates.json`

Later planned entrypoints:

- `python -m iron.applications.transformer_layer_new.study.reconfiguration_overhead.run`
- `python -m iron.applications.transformer_layer_new.study.igpu.run`

## Documentation Map

- `docs/study_conventions.md`
  shared study rules, naming, and current block/end-to-end result conventions
- `study/block/README.md`
  block-study contract and shared entry point
- `study/end_to_end/README.md`
  end-to-end study contract
- `study/reconfiguration_overhead/README.md`
  reconfiguration-overhead study contract
- `study/igpu/README.md`
  iGPU study contract
