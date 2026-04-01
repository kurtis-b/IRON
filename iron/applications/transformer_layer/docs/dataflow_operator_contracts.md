<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataflow Operator Contracts

This note captures the retained public contracts for the three fused Dataflow
blocks, plus the topology and pytest policy that should drive the operator
implementations.

For the current implementation/verification status and the remaining support
work, see `dataflow_status.md`.

## General policy

- `dtype` is `bfloat16` only.
- `head_dim` is retained as `64` for the thesis surface.
- `embedding_dim` is derived as `num_heads * head_dim`.
- `embedding_dim` should not be an independent pytest workload parameter.
- DesignPats should stay as close to `devel` as practical.
- each block is internally fused
- Dataflow uses device-buffer handoff between blocks
- each fused block uses its own xclbin
- topology validity is decided in the operator design file
- pytest should attempt all enumerated workload/topology combinations and fail
  when a combination is not actually constructible or functional
- the lowering should be topology-driven, not split into dedicated structural
  branches such as "mixed parallel" special cases
- each thesis-local operator directory should mirror the `mha` operator layout:
  `design.py`, `op.py`, `reference.py`, and `test.py`
- `design.py` should expose the retained execution entrypoint only; optional
  debug/CLI helpers should live in a separate `design_debug.py`
- `reference.py` should expose only `generate_golden_reference()`
- `test.py` should expose `generate_test_params()` and one test method
- `op.py` should keep the same operator-method surface as `mha`; any small
  layout-adaptation helpers should live at module scope rather than as extra
  operator methods
- thesis-specific weight binding and benchmark metadata should stay in the
  pattern layer, not as extra helper methods on the operator class
- when a block wraps another imported operator, the pattern layer should bind
  weights on the thesis wrapper itself rather than reaching through nested
  `.block` internals
- shared host-side benchmark-preparation helpers for the Dataflow thesis app
  should live in `src/utils.py`, not duplicated across pattern modules
- shared thesis-app weight-binding helpers for Dataflow blocks may also live in
  `src/utils.py`, while the actual operator mutation still happens only from
  the pattern layer
- repeated in-process benchmark-metadata formatting for Dataflow thesis patterns
  may also be centralized in `src/utils.py`

## Block 1

### Public tensor contract

- input:
  - `hidden_states [seq, embedding_dim]`
- weights:
  - `W_Q [embedding_dim, num_heads, head_dim]`
  - `W_K [embedding_dim, num_heads, head_dim]`
  - `W_V [embedding_dim, num_heads, head_dim]`
- outputs written to DDR:
  - canonical handoff layout:
    - `q [seq, embedding_dim]`
    - `k [seq, embedding_dim]`
    - `v [seq, embedding_dim]`
  - wrapper surface:
    - `q [num_heads, seq, head_dim]`
    - `k [num_heads, seq, head_dim]`
    - `v [num_heads, seq, head_dim]`

Block 1 therefore drains separate flat `Q/K/V` buffers directly, and the wrapper
only reshapes to head-major when that public surface is requested.

### Variable workload dimensions

- `seq_len`
- `num_heads`
- `head_dim`

### Variable topology dimensions

- `parallel_seq`
- `parallel_emb`
- `tile_m`
- `tile_k`
- `tile_n`

### Required validity checks

- `embedding_dim == num_heads * head_dim`
- `parallel_seq in {1, 2, 4, 6, 8}`
- `seq_len % (parallel_seq * tile_m) == 0`
- `embedding_dim % parallel_emb == 0`
- `num_aie_columns % parallel_emb == 0`
- the combined topology must fit within the 8-column array

Block 1 currently resolves its retained thesis topologies through the single
design entrypoint in
[`iron/operators/qkv_proj/design.py`](/home/cj/iron/iron/operators/qkv_proj/design.py)
and lowers them through the local standalone fused Block 1 design used by
[`iron/operators/qkv_proj/op.py`](/home/cj/iron/iron/operators/qkv_proj/op.py).
The retained `v2` implementation parallelizes the three independent Q/K/V
projections as one wider GEMM, drains separate `Q/K/V` DDR buffers directly,
and reshapes to head-major `q/k/v` only on the wrapper surface when needed. The
current Block 1
runtime-supported surface is workload-aware over the tile/array part of the
topology space, and it now lowers real sweeps for both Block 1 topology axes on
the canonical `c6/c8` runtime paths:
- `parallel_seq` changes the active compute-row count and the Block 1 row tiling
  used by the runtime sequence
- `parallel_emb` partitions each projection's embedding width into contiguous
  output groups and makes the Block 1 `B` fills and direct `Q/K/V` drains
  embedding-group aware
The runtime-supported Block 1 catalog is now generated from the broader
theoretical/practical space instead of being pinned to the retained `12x64` and
`16x64` thesis families. It still limits `parallel_seq` to `{1, 2, 4}` because
the local design only lowers up to the available four compute rows, while the
broader theoretical/practical catalogs continue to explore the larger thesis
space. The practical surface is pruned to a compact ranked selection, and the
runtime-supported surface is pruned further so studies keep a compact set of
runtime-validated Block 1 candidates per workload.
That design file should expose three distinct topology views:
- a seq-len-aware runtime-supported topology list used by operator tests and the
  workload-specific runtime-selection path
- a broader theoretical topology enumerator that explores every combination
  allowed by the Block 1 contract, fused GEMM tiling, array-column count,
  matmul-kernel divisibility, the current batched-GEMM double-buffered
  compute-tile local-memory limits, and the current `parallel_seq` /
  `parallel_emb` lowering axes for a given workload
- a heuristic-pruned practical exploration surface that keeps only
  microkernel-valid tile sizes on the current `c6/c8` runtime paths and then
  favors higher core/lane parallelism plus fuller compute-tile utilization with
  squarer `MxK` and `KxN` GEMM tiles while still retaining the baseline
  runtime-supported study topologies
The checked-in study manifests should continue to pin the baseline retained
topology IDs for reproducibility even as that broader theoretical exploration
surface grows. The canonical Block 1 surface is expressed in terms of
`parallel_seq` and `parallel_emb`.

## Block 2

### Public tensor contract

- inputs:
  - accepted canonical layout:
    - `q [seq, embedding_dim]`
    - `k [seq, embedding_dim]`
    - `v [seq, embedding_dim]`
  - also accepted:
    - `q [num_heads, seq, head_dim]`
    - `k [num_heads, seq, head_dim]`
    - `v [num_heads, seq, head_dim]`
- output:
  - `hidden_states [seq, embedding_dim]`

### Variable workload dimensions

- `seq_len`
- `num_heads`
- `head_dim`

### Variable topology dimensions

- `parallel_seq`
- `parallel_heads`
- `q_seq_tile`
- `kv_seq_tile`
- `emb_tile`
- `o_proj_acc_depth`

### Required validity checks

- `embedding_dim == num_heads * head_dim`
- `parallel_seq in {1, 2, 4, 6, 8}`
- `seq_len % (parallel_seq * q_seq_tile) == 0`
- `seq_len % kv_seq_tile == 0`
- `num_heads % parallel_heads == 0`
- `parallel_seq * parallel_heads <= 8`
- `embedding_dim % (emb_tile * o_proj_acc_depth) == 0`

Block 2 currently resolves its retained thesis topologies through the single
design entrypoint in
[`iron/operators/mha_out_proj/design.py`](/home/cj/iron/iron/operators/mha_out_proj/design.py)
and translates them into the fused runtime wrapper in
[`iron/operators/mha_out_proj/op.py`](/home/cj/iron/iron/operators/mha_out_proj/op.py).
The retained `v2` surface now lowers two topology axes on the validated runtime
path:
- when `seq_len` is provided, the runtime-supported Block 2 catalog is now
  generated from the broader theoretical surface and pruned back to a compact
  runtime set instead of being limited to the retained family table; retained
  workloads still resolve to the preferred validated retained/promoted IDs,
  while generalized `head_dim=64` workloads can surface broader generated
  runtime candidates
- `parallel_heads` on the established retained `ps1` path
- a validated denser `o_proj_acc_depth` subset on the retained `12x64` and
  `16x64` families for `head_dim == 64`, `q_seq_tile == 32`, and
  `kv_seq_tile == 64`; the current runtime-supported catalog now admits the
  maximum validated `o_proj_acc_depth` for each retained `ps/ph` shape:
  `acc8` on all retained widened `12x64` / `16x64` `q32/kv64` shapes
- a seq-len-aware `parallel_seq` subset on the retained `12x64` and `16x64`
  families, with `parallel_seq in {2, 4, 6, 8}` lowered as real sequence lanes
  when `q_seq_tile == 32`, `kv_seq_tile == 64`, and
  `parallel_seq * parallel_heads <= 8`; the validated composed subset now
  includes `ps2/ph2`, `ps2/ph4`, and `ps4/ph2` where sequence divisibility
  allows them, `ps6` is currently validated on retained workloads whose
  `seq_len` is divisible by `192`, and `ps8` is currently validated on retained
  workloads whose `seq_len` is divisible by `256`; within that retained
  widened surface the runtime selector now prefers the maximum validated
  `o_proj_acc_depth` for each shape before falling back to `acc1`
- a narrower `parallel_seq` subset on the retained `1x64` family, currently
  validated with `parallel_heads == 1`, `q_seq_tile == 32`,
  `kv_seq_tile == 64`, `emb_tile == 64`, and `o_proj_acc_depth == 1`; this
  currently covers `ps2` at `seq_len=64`, `ps2/ps4/ps6` at `seq_len=384`, and
  `ps2/ps4/ps8` at `seq_len=512`
Representative generalized `head_dim=64` workloads such as `24x64` and `8x64`
now also resolve non-empty seq-len-aware runtime/practical catalogs through
that same path, though the honest current runtime gate is still the narrower
validated `q32/kv64` subset: higher `o_proj_acc_depth` is currently admitted
only on the retained `12x64/16x64` families through the max-validated-per-shape
policy above, while generalized families and the retained `1x64` family remain
`acc1`.
For the retained runtime-supported study surface, `emb_tile` is still selected
per retained workload family, and the current lowered design still does not
support the `16x64 / parallel_heads=8` retained-shape variant because it
overruns the available DMA-channel fanout during AIE lowering.
That design file should expose three distinct topology views:
- a seq-len-aware runtime-supported topology list used by operator tests,
  benchmark manifests, and app-level runtime matching
- a broader theoretical topology enumerator that explores every combination
  allowed by the Block 2 contract, microkernel divisibility, lane-count limit,
  and the current staged local-memory working-set limits for a given workload
- a heuristic-pruned practical exploration surface that favors higher sequence
  and head parallelism, larger Q/KV/output tiles, larger
  `o_proj_acc_depth`, larger `emb_tile * o_proj_acc_depth` output chunks to
  reduce recomputation of the earlier stages, and fuller per-stage
  local-memory utilization while still retaining the baseline runtime-supported
  study topologies; practical ranking should continue to prefer real lowered
  axes over paper-only gains
The checked-in study manifests should continue to pin the baseline retained
topology IDs for reproducibility even as that broader theoretical exploration
surface grows.
Study metadata emitted by the in-process Dataflow patterns should include the
selected Block 2 topology ID and family so retained topology choices are visible
in benchmark outputs.
The current packed Block 2 -> Block 3 handoff now uses a canonical tile-packed
buffer order that is independent of Block 2 `parallel_seq`. Compatibility now
requires:
- `Block 2 q_seq_tile == Block 3 tile_m`
- `Block 2 emb_tile == Block 3 tile_k`
- `(seq_len / Block 2 q_seq_tile)` divisible by `Block 3 parallel_seq`
The default Dataflow topology resolver still prefers a matching Block 2
`parallel_seq` when the selected Block 3 topology is sequence-parallel, while
keeping the conservative `ph1` path as the default unless a composed Block 2
topology is explicitly requested.

## Block 3

### Public tensor contract

- inputs:
  - `residual [seq, embedding_dim]`
  - `hidden_states [seq, embedding_dim]`
- weights:
  - `W_Up [embedding_dim, intermediate_size]`
  - `W_Down [intermediate_size, embedding_dim]`
- output:
  - `hidden_states [seq, embedding_dim]`

### Variable workload dimensions

- `seq_len`
- `num_heads`
- `head_dim`
- `intermediate_size`

### Variable topology dimensions

- `parallel_seq`
- `parallel_int_dim`
- `tile_m`
- `tile_k`
- `tile_n`
- `down_proj_depth`
- `gelu_stage`

### Required validity checks

- `embedding_dim == num_heads * head_dim`
- `parallel_seq in {1, 2, 4, 6, 8}`
- `seq_len % (parallel_seq * tile_m) == 0`

Block 3 currently resolves its retained thesis topologies through the single
design entrypoint in
[`iron/operators/addnorm_ffn_addnorm/design.py`](/home/cj/iron/iron/operators/addnorm_ffn_addnorm/design.py)
and translates them into the pipelined wrapper in
[`iron/operators/addnorm_ffn_addnorm/op.py`](/home/cj/iron/iron/operators/addnorm_ffn_addnorm/op.py).
That design file is now paired with a workload-aware Block 3 topology layer that
exposes:
- a seq-len-aware runtime-supported catalog
- a broader practical exploration catalog
- the full theoretical surface for the current Block 3 contract

The current runtime-supported Block 3 catalog is generated from the broader
theoretical surface, filtered through the strict current Block 3 runtime gate,
and then pruned to a compact runtime-supported set while preserving retained
thesis topology IDs as preferred baselines where they still satisfy that gate.
In practice, retained workloads currently stay on those preferred validated
baseline IDs, while generalized workloads use the broader generated runtime
pool.

That runtime gate is intentionally narrower than the practical surface and
still reflects the current `design.py` implementation:
- `num_aie_columns == 8`
- `tile_m in {16, 32, 64}`
- `tile_k >= 16`
- `tile_n >= 16`
- `down_proj_depth <= 8`
- placement must satisfy the current horizontal, vertical, or compact-sequence
  Block 3 layouts
- `parallel_seq > 4` currently requires the compact-sequence layout, so the
  promoted `ps8` runtime surface is currently limited to `parallel_int_dim=1`

So the practical Block 3 surface can still explore broader `ps8`/small-tile
shapes than the current runtime-supported set, but `ps8` is no longer
practical-only: the compact-sequence runtime path now supports strict-runtime-
feasible `ps8, pi1` topologies.
Representative generalized workloads such as `1536/6144` and `960/3840` now
also produce seq-len-aware runtime/practical Block 3 catalogs through that same
generated-runtime path.
For the current packed Dataflow handoff from Block 2 into Block 3, a selected
runtime pair is only structurally compatible when:
- Block 2 `q_seq_tile ==` Block 3 `tile_m`
- Block 2 `emb_tile ==` Block 3 `tile_k`
- `seq_len / q_seq_tile` remains divisible by Block 3 `parallel_seq`

That compatibility rule is stricter than "both operators are independently
runtime-supported." The app-layer Dataflow resolver should enforce those tile
shape checks before it picks a packed Block 2 -> Block 3 pair.
Study metadata emitted by the in-process Dataflow patterns should include the
selected block topology IDs and families so retained topology choices are visible
in benchmark outputs.
Those topology fields, plus any preserved practical-topology provenance fields
for manifest-expanded studies, should also be reserved in the shared
transformer-layer result schema so benchmark CSVs expose them as stable columns
rather than unordered extra metadata.

At the study/app layer, retained topology selection should remain optional.
`TransformerLayerSpec` may carry `block1_topology_id`, `block2_topology_id`, and
`block3_topology_id` overrides so manifests and study cases can pin retained
topologies explicitly; when omitted, each block should resolve its default
retained topology through its local design entrypoint.
When parity validation is enabled, the same requested topology overrides should
be forwarded into the parity path so correctness checks exercise the same
retained topology selection as the benchmark rows.
Parity CSV outputs should preserve the retained block topology IDs and families,
plus any preserved practical-topology provenance fields from manifest-expanded
studies, with a stable parity-column order rather than relying on
insertion-ordered extra fields.
For direct automation, the same retained topology overrides should also be
accepted by the benchmark/job CLI layer so study manifests, ad hoc benchmark
invocations, and unattended jobs all share the same topology-selection surface.
Alongside that strict runtime-supported override path, the application layer
should also expose a pure-Python practical-topology exploration catalog that
cross-products the three block-level practical surfaces into capped
cross-block candidate combinations for future autotune/study generation,
without pretending those broader practical IDs are already runnable through the
retained runtime wrappers; that catalog should annotate which per-block and
cross-block candidates are runtime-supported today, and when a study manifest
requests `topology_exploration` the generated runnable `study_cases` should use
the corresponding runtime topology IDs rather than raw non-runnable practical
IDs, while still preserving the original practical exploration IDs in the study
case metadata and benchmark rows for traceability.
Study manifests may optionally request that same practical surface through a
`topology_exploration` object layered on top of a single base `layer_spec`, but
that expansion should currently be limited to single-`seq_len` manifests so the
generated combinations remain tied to one concrete workload geometry.
The unattended pipeline's temp-manifest rewrite path should preserve that
`topology_exploration` object as-is so the generated study-case expansion still
happens inside the benchmark loader rather than being flattened by the pipeline
wrapper.
For unattended study pipelines, `npu_study` step definitions should be allowed
to carry the same `block1_topology_id`, `block2_topology_id`, and
`block3_topology_id` overrides so pipeline-driven sweeps can pin retained
topologies without patching the underlying study manifest.
The checked-in Dataflow study manifests should pin those retained topology IDs
explicitly for the retained `768/3072/12` and `1024/4096/16` families so the
benchmark surface remains reproducible even if default topology selection
changes later.
Study manifests should also be normalized through `TransformerLayerSpec` during
loading so pinned topology IDs and the rest of the retained layer surface are
validated before any benchmark or parity execution starts.
When those requested topology IDs are present, benchmark rows, failure rows,
and parity rows should also backfill the matching topology-family columns so
the stable result schema remains informative even before runtime metadata is
available.
Support-matrix CSV outputs should preserve the same block topology IDs and
families, plus any preserved practical-topology provenance fields, with stable
support-matrix column ordering, so unsupported or filtered rows remain
attributable to the retained topology that was requested or resolved.
For the iGPU comparison study, GPU result rows should preserve the selected
reference NPU row's block topology IDs and families under explicit
`reference_npu_*` columns so best-NPU-vs-iGPU results remain attributable to
the retained topology that won NPU selection; if the selected NPU row came from
a manifest-expanded practical study, the preserved practical-topology
provenance should also flow through those `reference_npu_*` fields.
Those `reference_npu_*` columns should also be reserved in the shared result
schema so GPU compare CSVs expose them as stable fields rather than unordered
extras.
Bottleneck summary CSV/JSON/text outputs should also preserve the retained
block topology IDs from the analyzed rows, plus any preserved
practical-topology provenance fields, so dominant-component reports remain
attributable to the resolved Dataflow topology surface and the original
exploration candidate when those differ.
Plot generation should be allowed to facet on retained topology columns too, so
latency/throughput/power, bottleneck, and best-NPU-vs-iGPU plots can be grouped
by resolved block topology when needed.
The checked-in unattended pipeline should use that surface for retained
Dataflow-study plot steps rather than relying only on study-case grouping.
Pipeline config loading should also reject invalid plot `facet_key` values
early, using the stable result-schema columns as the supported unattended
faceting surface.
- `intermediate_size % parallel_int_dim == 0`
- `embedding_dim % tile_k == 0`
- `intermediate_size % tile_n == 0`
- the combined topology must fit within the 8-column array

Block 3 currently validates those constraints through the single design
entrypoint in
[`iron/operators/addnorm_ffn_addnorm/design.py`](/home/cj/iron/iron/operators/addnorm_ffn_addnorm/design.py)
and resolves the thesis-facing topology directly into the local pipelined
Block 3 runtime. The current runtime contract is exact rather than padded:
`seq_len` must be divisible by `parallel_seq * tile_m`, and one compiled Block
3 artifact is built for that concrete `seq_len`. The Block 3 runtime catalog is
now generated per workload from the broader theoretical surface, then filtered
through the current layout/runtime gate and pruned back to a compact validated
set. The older retained thesis IDs are still preserved as preferred baselines,
but retained workloads no longer stop there: additional generated candidates
can now fill the remaining runtime slots when they satisfy the current runtime
gate. The old broad skinny-wide fence is gone; the remaining exclusions are
now a short explicit list of reproduced bad shapes instead:
- `tile_k=24` with `tile_n > 128`, which still produces runtime NaNs
- `512 x 1536 x 6144` with `tile_k=16` and `tile_n=512`, which still fails
  shim BD lowering
Because `ln1_weight` and `ln2_weight` are compile-time constants embedded into
the generated Block 3 MLIR/xclbin, Block 3 artifact names must include a
deterministic fingerprint of those two weight tensors so tests and studies do
not accidentally reuse stale compiled artifacts across different retained
workloads.
For grouped Block 3 families where `hidden_size / tile_k` exceeds
`down_proj_depth`, the standalone runtime now uses a row-tile-local three-phase
design rather than a banked LN2 design:
- for one row tile, LN2 first consumes a full-width packed `[A | R]` prepass
  and caches LN1 row statistics locally
- next, for each `col_group`, LN1 regenerates the full-width stage-1 stream for
  FFN while LN2 independently regenerates only the active group and accumulates
  LN2 row statistics locally from `down + stage1`
- finally, for each `col_group`, LN1/FFN recompute that group and LN2
  regenerates only the active group again and emits that group's output slice
Under this contract, `parallel_seq` duplicates the whole
`AN1 -> Up-proj -> Down-proj -> AN2` pipeline across the array. If one pipeline
runs over multiple row tiles because `seq_len / (parallel_seq * tile_m) > 1`,
those row tiles are processed sequentially through the three phases
rather than through separate LN2 stats banks. LN2 therefore owns both LN1 and
LN2 cached row statistics locally, and the grouped runtime no longer depends on
a down-proj-to-LN2 statistics handoff.
That design file should also expose three distinct topology views:
- a seq-len-aware runtime-supported topology list used by operator tests and
  benchmark manifests; for retained workloads that list now keeps the validated
  baseline IDs and also promotes additional generated runtime candidates when
  they pass the current gate
- a broader theoretical topology enumerator that explores every combination
  allowed by the Block 3 contract, imported FFN tiling equalities,
  `seq_len`-fit, lane-count limit, and GeLU staging for a given workload
- a heuristic-pruned practical exploration surface that favors higher sequence
  and intermediate-dimension parallelism, larger reusable tiles, and fuller
  array-column utilization while still retaining the baseline runtime-supported
  study topologies

## Reference policy

Each operator keeps its own `reference.py`.

`reference.py` should stay simple:

- one fused reference path per operator
- public-layout tensors only
- no benchmark logic
- no topology logic
- no extra debug-only execution branches

The retained operator-local references are:

- Block 1:
  - fused `Q/K/V` projection using the Block 1 public layouts
- Block 2:
  - scaled dot-product attention plus output projection using
    `q/k/v [num_heads, seq, head_dim]`
- Block 3:
  - `Add & Norm 1 + FFN + Add & Norm 2` using the Block 3 public layouts

Thresholds should match the retained `mha` operator thresholds.

## Pytest policy

Each operator `test.py` should expose:

- one `generate_test_params()` helper
- one test method

`generate_test_params()` should build parameter tuples from:

- workload dimensions
- topology dimensions

The params should therefore encode both workload and topology information.

The single test method should, for each generated parameter tuple:

- instantiate the operator
- attempt real hardware construction/execution
- compare against the operator-local reference
- fail if the attempted combination is not possible or not functional

The design file, not the test helper, owns the constructibility decision.
