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

## Minimal-Diff Direction

The guiding implementation rule is to stay as close to `devel` as practical.

- `Dataflow` carries the thesis-specific new work
- `Runlist` should remain close to the imported baseline, with hidden-states-only
  thesis glue
- `GEMM offload` should remain close to the imported baseline, with hidden-states-only
  thesis glue
- shared-kernel changes should be avoided unless a retained thesis case proves
  they are required

## Dataflow

`Dataflow` is organized as three blocks:

1. `Q/K/V` projection
2. `MHA + Output Projection`
3. `Add & Norm 1 + FFN + Add & Norm 2`

Important retained implementation choices:

- each Dataflow block is intended to remain internally fused
- Dataflow uses device-buffer handoff between blocks rather than inter-block
  streaming
- `Block 1` outputs `q/k/v [num_heads, seq, head_dim]`
- `Block 1` parallelizes the three projection matrices through one fused GEMM
  and splits the result back into head-major `q/k/v` on the wrapper surface
- head-major means each attention head is stored contiguously in memory
- `Block 2` uses the pipelined `mha_out_proj` operator
- `Block 3` uses a dedicated DesignPats operator that pipelines the first Add &
  Norm stage into the Up projection cores
- the benchmark boundary is hidden states only
- operator-local topology testing is a retained policy, with one parametrized
  test method per operator
- the detailed block contracts and topology/test policy are documented in
  `dataflow_operator_contracts.md`

Retained workflow policy:

- the checked-in Dataflow study manifests pin the intended retained block
  topology IDs for the `768 / 3072 / 12` and `1024 / 4096 / 16` families
- benchmark, parity, support-matrix, bottleneck, and iGPU-compare outputs keep
  those topology IDs visible so downstream analysis stays attributable to the
  selected retained Dataflow surface
- unattended Dataflow plot steps facet on retained Block 2 topology so
  generated plots stay grouped by the resolved fused-attention configuration

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
