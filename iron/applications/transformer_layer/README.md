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

## Execution Modes

### End-to-end modes

- `dataflow`: three Dataflow blocks connected in one benchmarked pipeline
- `runlist`: the runlist-based full-layer baseline
- `gemm_offload`: the shared-runtime GEMM-offload baseline

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
