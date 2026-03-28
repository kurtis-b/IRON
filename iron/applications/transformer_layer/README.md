<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Transformer Layer DesignPats

This thesis app benchmarks three hidden-states-only NPU implementations of one
encoder-style transformer layer:

- `dataflow`
- `runlist`
- `gemm_offload`

It also supports:

- a block-level Dataflow study
- an end-to-end best-NPU-vs-iGPU study
- a reconfiguration-overhead study
- roofline annotation
- bottleneck summaries
- unattended pipeline execution
- NPU pseudo-power and iGPU power measurement

## Minimal-Diff Policy

DesignPats should stay as close to `devel` as practical.

- keep shared-kernel and shared-framework edits to a minimum
- keep `Runlist` and `GEMM offload` as imported baselines with thesis-app glue
  only
- concentrate new thesis-specific work in the fused Dataflow blocks and the
  thesis benchmark harness
- avoid new shared-kernel changes unless a retained thesis case proves they are
  required

## Execution Modes

### End-to-end modes

- `dataflow`: three Dataflow blocks connected in one benchmarked pipeline
- `runlist`: the runlist-based full-layer baseline
- `gemm_offload`: the shared-runtime GEMM-offload baseline

For `dataflow`, each block is intended to be internally fused and uses its own
xclbin. The retained inter-block tensor contract is:

- `Block 1` outputs `q/k/v [num_heads, seq, head_dim]`
- head-major means each head is stored contiguously in memory
- `Block 2` consumes that layout

The detailed block contracts, topology dimensions, and pytest policy are
documented in:

- `docs/dataflow_operator_contracts.md`

## Module Layout

The app library code is being incrementally grouped under:

- `src/core`: shared data structures and reference logic
- `src/patterns`: end-to-end and block pattern implementations
- `src/bench`: shared benchmark-support modules
  including measurement audit, GPU/NPU power, and reusable GPU/NPU inference helpers
- `src/analysis`: reusable roofline and reporting-support modules
  including bottleneck summarization helpers
  and best-NPU-vs-iGPU compare helpers
  plus SVG/HTML plot generation helpers
- `src/pipeline`: unattended orchestration helpers
  including manifest-driven benchmark sweeps,
  study-pipeline orchestration, and JSON job launch support

The older `src/*.py` module paths remain as thin compatibility shims so the
restructure can stay close to `devel` while imports are cleaned up
incrementally.

### Block-study modes

- `block1_qkv_proj`
- `block2_mha_out_proj`
- `block3_addnorm_ffn_addnorm`

### Reconfiguration-overhead modes

- `gemm_offload_gemm_sequence`
- `runlist_gemm_sequence`

## Input Boundary

The DesignPats app starts from hidden states only.

There is no supported public benchmark surface that starts from supplied
`Q/K/V/R`.

The retained operator-test policy is:

- one `generate_test_params()` helper
- one test method
- params composed from workload dimensions and topology dimensions
- the design file owns the constructibility decision

## Studies

Checked-in study manifests:

- `study/dataflow_blocks.json`
- `study/design_patterns_end_to_end.json`
- `study/reconfiguration_overhead.json`
- `study/gpu_compare_end_to_end_igpu.json`
- `study/designpats_pipeline.json`

Retained benchmark surface:

- model families:
  - `768 / 3072 / 12`
  - `1024 / 4096 / 16`
- sequence ladder:
  - `64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

The reconfiguration-overhead study uses representative points:

- `64`
- `512`
- `16384`

## Common Commands

Run one end-to-end NPU study:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_end_to_end.json
```

Run the full unattended DesignPats pipeline:

```bash
python iron/applications/transformer_layer/run_study_pipeline.py \
  --config iron/applications/transformer_layer/study/designpats_pipeline.json
```

Smoke the full unattended pipeline:

```bash
python iron/applications/transformer_layer/run_study_pipeline.py \
  --config iron/applications/transformer_layer/study/designpats_pipeline.json \
  --smoke \
  --warmup-runs 0 \
  --runs-per-sample 1
```

Run end-to-end parity:

```bash
python iron/applications/transformer_layer/validate_npu_parity.py \
  --execution-mode dataflow \
  --seq-lens 64,128
```

## Power

NPU power is measured as pseudo-NPU power:

- package power from `turbostat`
- minus a per-row quiescent baseline

iGPU power is measured through ROCm SMI in the iGPU compare path.

The unattended pipeline uses thermal recovery between benchmark-producing steps.
It does not use a reboot-based or external-controller power-cycle path.
