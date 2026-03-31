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
  - direct device drains into separate flat `Q/K/V` buffers
  - wrapper can expose either flat `Q/K/V` or head-major `q/k/v`

Current runtime-supported surface:

- runtime support is `seq_len`-aware
- the current local lowering now runtime-supports:
  - a generated/pruned `c6/c8` runtime catalog instead of a retained-only
    family table
  - a real `parallel_seq` sweep with active row counts in `{1, 2, 4}`
  - a real `parallel_emb` sweep on the canonical `c6/c8` paths through
    contiguous embedding-group `B` fills and direct `Q/K/V` drains
  - broader workload-family support beyond the retained `12x64` and `16x64`
    study families
- the runtime-supported surface is now pruned to at most `12` candidates per
  workload, and the practical exploration surface is pruned to at most `20`
  candidates per workload
- for the currently exercised retained workloads, that runtime-supported
  surface is:
  - `seq_len=64, hidden=768, heads=12`: `12` topologies
  - `seq_len=512, hidden=768, heads=12`: `12` topologies
  - `seq_len=64, hidden=1024, heads=16`: `12` topologies
  - `seq_len=512, hidden=1024, heads=16`: `12` topologies
- representative generalized workloads that now produce runtime/practical
  Block 1 catalogs include:
  - `seq_len=64, hidden=1536, heads=24`: runtime `12`, practical `20`
  - `seq_len=64, hidden=960, heads=12`: runtime `12`, practical `20`
  - `seq_len=512, hidden=1536, heads=24`: runtime `12`, practical `20`
  - `seq_len=512, hidden=960, heads=12`: runtime `12`, practical `20`

Verified today:

- supported Block 1 runtime topologies are functionally verified in
  `iron/operators/qkv_proj/test.py`
- the Block 1 -> Block 2 flat `Q/K/V` handoff is functionally verified in the
  local operator tests and used by the structured Dataflow pattern

Work left:

- if higher sequence-lane support is needed, extend the current `parallel_seq`
  runtime lowering beyond `{1, 2, 4}`
- if a wider embedding-group sweep is needed, continue validating and promoting
  additional `parallel_emb` variants as they are selected by studies
- continue validating and promoting additional generalized workload families as
  they get exercised in studies; the practical/theoretical catalogs are still
  broader than the pruned runtime-supported study surface
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

- runtime support is now `seq_len`-aware
- the current local lowering now runtime-supports:
  - a generated/pruned runtime catalog filtered from the broader theoretical
    surface instead of only returning the retained family table when `seq_len`
    is provided
  - the retained `parallel_heads` sweep on the established `ps1` path
  - a validated `parallel_seq` subset on the retained `12x64` and `16x64`
    families, with `parallel_seq in {2, 4, 6, 8}` lowered as real sequence lanes
    when `q_seq_tile == 32`, `kv_seq_tile == 64`, `o_proj_acc_depth == 1`, and
    `parallel_seq * parallel_heads <= 8`; the validated composed subset now
    includes `ps2/ph2`, `ps2/ph4`, and `ps4/ph2` where sequence divisibility
    allows them, `ps6` is currently validated on retained workloads whose
    `seq_len` is divisible by `192`, and `ps8` is currently validated on
    retained workloads whose `seq_len` is divisible by `256`
  - a narrower validated `parallel_seq` subset on the retained `1x64` family:
    `parallel_heads == 1`, `q_seq_tile == 32`, `kv_seq_tile == 64`,
    `emb_tile == 64`, `o_proj_acc_depth == 1`, with `ps2` validated at
    `seq_len=64`, `ps2/ps4/ps6` validated at `seq_len=384`, and
    `ps2/ps4/ps8` validated at `seq_len=512`
- the current exercised runtime-supported surface is:
  - `seq_len=64, heads=1, head_dim=64`: `2` topologies
  - `seq_len=384, heads=1, head_dim=64`: `4` topologies
  - `seq_len=512, heads=1, head_dim=64`: `4` topologies
  - `seq_len=64, heads=12, head_dim=64`: `7` topologies
  - `seq_len=512, heads=12, head_dim=64`: `10` topologies
  - `seq_len=64, heads=16, head_dim=64`: `6` topologies
  - `seq_len=512, heads=16, head_dim=64`: `9` topologies
  - representative generalized `head_dim=64` workloads now also produce
    runtime/practical Block 2 catalogs:
    - `seq_len=64, heads=24`: runtime `12`
    - `seq_len=512, heads=24`: runtime `12`
    - `seq_len=64, heads=8`: runtime `12`
    - `seq_len=512, heads=8`: runtime `12`
- concretely:
  - `1 x 64, seq_len=64`
    - `q32_kv64_e64_ps1_ph1_acc1`
    - `q32_kv64_e64_ps2_ph1_acc1`
  - `1 x 64, seq_len=512`
    - `q32_kv64_e64_ps1_ph1_acc1`
    - `q32_kv64_e64_ps2_ph1_acc1`
    - `q32_kv64_e64_ps4_ph1_acc1`
    - `q32_kv64_e64_ps8_ph1_acc1`
  - `1 x 64, seq_len=384`
    - `q32_kv64_e64_ps1_ph1_acc1`
    - `q32_kv64_e64_ps2_ph1_acc1`
    - `q32_kv64_e64_ps4_ph1_acc1`
    - `q32_kv64_e64_ps6_ph1_acc1`
  - `12 x 64, seq_len=64`
    - `q32_kv64_e96_ps1_ph1_acc1`
    - `q32_kv64_e96_ps1_ph2_acc1`
    - `q32_kv64_e96_ps1_ph4_acc1`
    - `q32_kv64_e96_ps1_ph6_acc1`
    - `q32_kv64_e96_ps2_ph1_acc1`
    - `q32_kv64_e96_ps2_ph2_acc1`
    - `q32_kv64_e96_ps2_ph4_acc1`
  - `12 x 64, seq_len=512`
    - the same set plus `q32_kv64_e96_ps4_ph1_acc1`,
      `q32_kv64_e96_ps4_ph2_acc1`, and `q32_kv64_e96_ps8_ph1_acc1`
  - `12 x 64, seq_len=384`
    - the `ps1` retained set plus `q32_kv64_e96_ps2_ph2_acc1`,
      `q32_kv64_e96_ps2_ph4_acc1`, `q32_kv64_e96_ps4_ph2_acc1`, and
      `q32_kv64_e96_ps6_ph1_acc1`
  - `16 x 64, seq_len=64`
    - `q32_kv64_e128_ps1_ph1_acc1`
    - `q32_kv64_e128_ps1_ph2_acc1`
    - `q32_kv64_e128_ps1_ph4_acc1`
    - `q32_kv64_e128_ps2_ph1_acc1`
    - `q32_kv64_e128_ps2_ph2_acc1`
    - `q32_kv64_e128_ps2_ph4_acc1`
  - `16 x 64, seq_len=512`
    - the same set plus `q32_kv64_e128_ps4_ph1_acc1`,
      `q32_kv64_e128_ps4_ph2_acc1`, and `q32_kv64_e128_ps8_ph1_acc1`
  - `16 x 64, seq_len=384`
    - the `ps1` retained set plus `q32_kv64_e128_ps2_ph2_acc1`,
      `q32_kv64_e128_ps2_ph4_acc1`, `q32_kv64_e128_ps4_ph2_acc1`, and
      `q32_kv64_e128_ps6_ph1_acc1`

Verified today:

- supported Block 2 runtime topologies are functionally verified in
  `iron/operators/mha_out_proj/test.py`
- the packed Block 2 output path used for the Block 2 -> Block 3 handoff has
  focused layout tests, direct dense-vs-packed parity coverage on both the
  retained `ps1` runtime path, a representative composed `ps2/ph2` path, and a
  representative mismatched-target packed case where Block 2 runs at `ps2` and
  writes the canonical packed layout for Block 3 `ps1`, and
  representative app-level parity coverage when it is paired with a
  tile-compatible Block 3 runtime topology, but not a full compatible-pair
  sweep yet

Work left:

- broaden the current real `parallel_seq` lowering beyond the validated
  `q32/kv64/acc1` subset; on retained families that now includes the composed
  `ps/ph` cases listed above, on the `1x64` family the widened support is still
  limited to `parallel_heads == 1`, and on generalized families the current
  generated runtime catalog is still anchored on the `ps1` head-parallel path
- broaden the packed Block 2 output mode beyond the current restricted path
  used for Block 3 handoff; the current packed handoff no longer requires
  `Block 2 parallel_seq == Block 3 parallel_seq`, but it still requires
  matching `(q_seq_tile, emb_tile) == (tile_m, tile_k)` and a Block 3-compatible
  packed row count
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

- the runtime catalog is now generated per workload from the broader
  theoretical surface, then filtered through the strict current-layout runtime
  gate and pruned to a compact runtime-supported set
- the older retained Block 3 thesis IDs are preserved as preferred baselines
  inside that generated runtime pool rather than acting as the sole runtime
  source; retained workloads now also promote additional generated candidates
  when they satisfy the current runtime gate
- that gate still reflects the current `design.py` implementation:
  - `num_aie_columns == 8`
  - `tile_m in {16, 32, 64}`
  - `tile_k >= 16`
  - `tile_n >= 16`
  - the old broad skinny-wide fence is gone; only a short explicit list of
    reproduced unstable shapes remains excluded:
    - `tile_k=24` with `tile_n > 128` still produces runtime NaNs
    - `512 x 1536 x 6144` with `tile_k=16` and `tile_n=512` still fails shim
      BD lowering
  - `down_proj_depth <= 8`
  - placement must satisfy the current horizontal, vertical, or compact-
    sequence Block 3 layout
  - `parallel_seq > 4` currently requires the compact-sequence layout, so the
    promoted `ps8` runtime surface is currently limited to `parallel_int_dim=1`
- current exercised runtime-supported counts are:
  - `64 x 768 x 3072`: `8`
  - `512 x 768 x 3072`: `8`
  - `64 x 1024 x 4096`: `8`
  - `512 x 1024 x 4096`: `8`
  - `64 x 2048 x 8192`: `8`
  - `512 x 2048 x 8192`: `8`
- representative generalized workloads that now also produce runtime/practical
  Block 3 catalogs include:
  - `64 x 1536 x 6144`: runtime `8`, practical `13`
  - `512 x 1536 x 6144`: runtime `8`, practical `16`
  - `64 x 960 x 3840`: runtime `8`, practical `12`
  - `512 x 960 x 3840`: runtime `8`, practical `16`
- `ps8` is now runtime-supported for strict-runtime-feasible `parallel_int_dim=1`
  compact-sequence Block 3 topologies; broader `ps8` shapes still remain
  practical-only until the runtime layout is generalized further

Current execution contract:

- Block 3 no longer exposes `compile_rows` as a topology dimension
- the runtime requires exact sequence-length fit:
  - `seq_len % (parallel_seq * tile_m) == 0`
- for the current packed Block 2 -> Block 3 handoff, the selected Dataflow
  pair must also satisfy:
  - Block 2 `q_seq_tile ==` Block 3 `tile_m`
  - Block 2 `emb_tile ==` Block 3 `tile_k`
  - `(seq_len / Block 2 q_seq_tile)` divisible by Block 3 `parallel_seq`
- artifacts are compiled for the concrete requested `seq_len`; the current
  Block 3 runtime does not pad or chunk rows internally

Verified today:

- topology construction and packed-buffer contract checks exist in
  `iron/operators/addnorm_ffn_addnorm/test.py`
- the current packed Block 2 -> Block 3 path has:
  - compile coverage
  - focused helper/layout tests
  - representative Dataflow parity coverage across compatible `768 / 3072` and
    `1024 / 4096` runtime cases
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
- the current widened runtime surface, including the new `m16` and `ps8`
  promotions plus the generated runtime-catalog refactor, passes:
  - `iron/operators/addnorm_ffn_addnorm/test.py`
  - `iron/applications/transformer_layer/test_patterns.py`
  - `iron/applications/transformer_layer/test_topology_exploration.py`

Work left:

- continue promoting additional theoretical/practical Block 3 topologies only
  after the standalone runtime proves them constructible and numerically sound
- broaden `ps8` runtime support beyond the current compact-sequence `pi1` path
- continue promoting smaller-`k` runtime candidates now that the hard
  `tile_k >= 64` policy gate is gone
- broaden Dataflow-level Block 2 -> Block 3 verification further only as more
  tile-compatible runtime pairs are promoted into the app-visible catalog

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
