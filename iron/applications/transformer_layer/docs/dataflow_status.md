<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataflow Status

This note records the current implementation status of the three fused Dataflow
blocks, what is functionally verified today, and what work is still left.

Definitions used here:

- `fused`: the block's intended sub-operations execute within one NPU kernel
  invocation
- `pipelined`: the fused block is split into explicit internal stages, and
  earlier stages feed later stages on device
- `verified`: the implementation is functionally checked against a reference,
  not just constructed or smoke-run

## Block 1

Current state:

- standalone local operator: `iron/operators/qkv_proj`
- `fused`: yes
- `pipelined`: no
- current implementation shape:
  - one fused `QKV` projection invocation
  - host-side split back into `q/k/v`

Current runtime-supported surface:

- retained workload families only:
  - `12 x 64`
  - `16 x 64`
- runtime support is `seq_len`-aware
- for the currently exercised retained workloads, the runtime-supported surface
  equals the practical surface:
  - `seq_len=64, hidden=768, heads=12`: `64` topologies
  - `seq_len=512, hidden=768, heads=12`: `64` topologies
  - `seq_len=64, hidden=1024, heads=16`: `64` topologies
  - `seq_len=512, hidden=1024, heads=16`: `64` topologies

Verified today:

- supported Block 1 runtime topologies are functionally verified in
  `iron/operators/qkv_proj/test.py`

Work left:

- turn the paper topology axes into real lowering axes:
  - `parallel_seq`
  - `parallel_heads`
  - `parallel_head_dim`
- if wider thesis families are needed, generalize beyond the retained `12 x 64`
  and `16 x 64` families
- if a staged Block 1 pipeline is desired, introduce real internal worker
  stages rather than the current single fused GEMM shape

## Block 2

Current state:

- standalone local operator: `iron/operators/mha_out_proj`
- `fused`: yes
- `pipelined`: yes
- current internal stages:
  - `QK`
  - `softmax`
  - `PV`
  - `O-proj`

Current runtime-supported surface:

- `1 x 64`
  - `q32_kv64_e64_ps1_ph1_acc1`
- `12 x 64`
  - `q32_kv64_e96_ps1_ph1_acc1`
  - `q32_kv64_e96_ps1_ph2_acc1`
  - `q32_kv64_e96_ps1_ph4_acc1`
  - `q32_kv64_e96_ps1_ph6_acc1`
- `16 x 64`
  - `q32_kv64_e128_ps1_ph1_acc1`
  - `q32_kv64_e128_ps1_ph2_acc1`
  - `q32_kv64_e128_ps1_ph4_acc1`

Verified today:

- supported Block 2 runtime topologies are functionally verified in
  `iron/operators/mha_out_proj/test.py`
- the packed Block 2 output path used for the Block 2 -> Block 3 handoff has
  focused layout tests and a compile-plus-runtime smoke, but not a broad
  per-topology verification sweep yet

Work left:

- implement real topology-level `parallel_seq` lowering
- broaden the packed Block 2 output mode beyond the current restricted path
  used for Block 3 handoff
- widen runtime support across more of the practical/theoretical Block 2 space,
  especially if higher `o_proj_acc_depth` or wider family support is needed
- remove the remaining pytest skips in the practical matrix as those topologies
  become runnable

## Block 3

Current state:

- standalone local operator: `iron/operators/addnorm_ffn_addnorm`
- `fused`: yes
- `pipelined`: yes
- current internal stages:
  - first `Add + LayerNorm`
  - `Up-proj`
  - `Down-proj`
  - second `Add + LayerNorm`
- the current local design uses a packed Block 2 -> Block 3 handoff and routes
  packed `A/R` ingress through `L3 -> L2 -> L1`

Current runtime-supported surface:

- the runtime catalog is now generated per workload rather than pinned only to a
  short retained ID list
- runtime promotion currently comes from:
  - the established retained Block 3 thesis topologies
  - plus broader practical Block 3 candidates that also satisfy the strict
    current-layout runtime gate
- that gate still reflects the current `design.py` implementation:
  - `num_aie_columns == 8`
  - `tile_m in {32, 64}`
  - `tile_k >= 64`
  - `tile_n >= 16`
  - `down_proj_depth <= 8`
  - placement must satisfy the current horizontal/vertical Block 3 layout
- current exercised runtime-supported counts are:
  - `64 x 768 x 3072`: `6`
  - `512 x 768 x 3072`: `8`
  - `64 x 1024 x 4096`: `4`
  - `512 x 1024 x 4096`: `4`
  - `64 x 2048 x 8192`: `2`
  - `512 x 2048 x 8192`: `8`
- `ps8` remains practical-only today; it does not yet satisfy the current
  runtime placement/dataflow implementation

Current execution contract:

- Block 3 no longer exposes `compile_rows` as a topology dimension
- the runtime requires exact sequence-length fit:
  - `seq_len % (parallel_seq * tile_m) == 0`
- artifacts are compiled for the concrete requested `seq_len`; the current
  Block 3 runtime does not pad or chunk rows internally

Verified today:

- topology construction and packed-buffer contract checks exist in
  `iron/operators/addnorm_ffn_addnorm/test.py`
- the current packed Block 2 -> Block 3 path has:
  - compile coverage
  - focused helper/layout tests
  - a `seq_len=64` runtime smoke through Dataflow
- the full Block 3 runtime-supported surface is numerically verified against
  the Block 3 golden reference in
  `iron/operators/addnorm_ffn_addnorm/test.py`
- Block 3 now fingerprints `ln1_weight` and `ln2_weight` into its artifact
  names because those layer-norm weights are compile-time constants embedded
  into the generated MLIR/xclbin; without that, different test cases with the
  same topology could incorrectly reuse stale compiled artifacts
- the established `768 / 3072` and `1024 / 4096` families, including their
  `gelu_stage=0` retained variants, currently pass the numerical operator test
  matrix
- the retained `2048 / 8192` family, including both `gelu_stage` variants,
  now also passes the numerical operator test matrix
- the older retained/runtime surface passed the full Block 3 numerical matrix in
  `iron/operators/addnorm_ffn_addnorm/test.py::test_addnorm_ffn_addnorm`
- after the new runtime-catalog promotion rule, the application-side
  topology-exploration tests again pass and the expanded Block 3 numerical
  matrix is being re-validated against the generated runtime catalog

Work left:

- continue promoting additional theoretical/practical Block 3 topologies only
  after the standalone runtime proves them constructible and numerically sound
- add real runtime placement/dataflow support for `ps8`
- broaden runtime support below the current `tile_m in {32, 64}` and
  `tile_k >= 64` implementation limits only after those smaller-tile shapes are
  numerically validated
- broaden Dataflow-level Block 2 -> Block 3 verification now that the packed
  handoff and the multi-group Block 3 runtime are both functional

## Cross-block and App-Layer Work

Current state:

- Block 1, Block 2, and Block 3 are all standalone local operators
- practical/theoretical topology policy lives in per-operator `topology.py`
  files
- operator-local `test.py` files generate the current practical pytest matrices
- the app layer preserves topology provenance through benchmark, parity,
  support-matrix, bottleneck, and GPU-compare outputs
- the Dataflow path now supports a packed Block 2 -> Block 3 handoff

Work left:

- add a checked-in study that exercises the new topology-exploration path
  directly instead of only pinned retained IDs
- broaden end-to-end runtime verification once additional practical topologies
  are promoted into true runtime support
- keep the status of each block honest:
  - runtime-supported
  - practical-only
  - theoretical-only

## Verification Priorities

The highest-signal next verification work is:

1. add broader Block 2 packed-output verification when paired with Block 3
2. add a checked-in study that exercises the practical topology-exploration path
3. only then continue promoting additional practical/theoretical topologies into
   runtime-supported surfaces
