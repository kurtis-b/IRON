<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design Pattern Considerations

## Compared implementations

The DesignPats comparison is:

- `Dataflow`
- `Runlist`
- `GEMM offload`

`encoder_pipeline` is not part of this thesis direction.

## Dataflow

`Dataflow` is organized as three blocks:

1. `Q/K/V` projection
2. `MHA + Output Projection`
3. `Add & Norm 1 + FFN + Add & Norm 2`

Important retained implementation choices:

- `Block 2` uses the pipelined `mha_out_proj` operator
- `Block 3` uses a dedicated DesignPats operator that pipelines the first Add &
  Norm stage into the Up projection cores
- the benchmark boundary is hidden states only

## Runlist

`Runlist` keeps the stitched full-layer operator path as the baseline for a
non-Dataflow composite implementation.

It still begins from hidden states in the thesis app, but it internally
projects to `Q/K/V` before entering the stitched runlist operator.

## GEMM offload

`GEMM offload` is the shared-runtime baseline:

- one runtime xclbin per case
- multiple `insts.bin` artifacts
- runtime-parameter-driven reuse across GEMM workloads

This is the key contrast used again in the reconfiguration-overhead study.

## Reconfiguration-overhead study

The reconfiguration study isolates only the GEMM subgraph.

It compares:

- `gemm_offload_gemm_sequence`
- `runlist_gemm_sequence`

The thesis point is:

- `GEMM offload` uses shared runtime artifacts
- `Runlist` pays a multi-xclbin reconfiguration cost
