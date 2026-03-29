<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataflow Operator Contracts

This note captures the retained public contracts for the three fused Dataflow
blocks, plus the topology and pytest policy that should drive the operator
implementations.

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
- `design.py` should expose one retained design entrypoint plus optional `main()`
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
  - `q [num_heads, seq, head_dim]`
  - `k [num_heads, seq, head_dim]`
  - `v [num_heads, seq, head_dim]`

Block 1 therefore writes transposed-vs-input sequence/head order when it drains
to DDR.

### Variable workload dimensions

- `seq_len`
- `num_heads`
- `head_dim`

### Variable topology dimensions

- `parallel_seq`
- `parallel_heads`
- `parallel_head_dim`
- `tile_m`
- `tile_k`
- `tile_n`

### Required validity checks

- `embedding_dim == num_heads * head_dim`
- `parallel_seq in {1, 2, 4, 6, 8}`
- `seq_len % (parallel_seq * tile_m) == 0`
- `num_heads % parallel_heads == 0`
- `head_dim % parallel_head_dim == 0`
- the combined topology must fit within the 8-column array

Block 1 currently resolves its retained thesis topologies through the single
design entrypoint in
[`iron/operators/qkv_proj/design.py`](/home/cj/iron/iron/operators/qkv_proj/design.py)
and lowers them through the local fused Block 1 design used by
[`iron/operators/qkv_proj/op.py`](/home/cj/iron/iron/operators/qkv_proj/op.py).
The retained `v2` implementation parallelizes the three independent Q/K/V
projections as one wider GEMM and then reshapes the combined output back to
head-major `q/k/v` tensors on the wrapper surface. The current retained Block 1
topology sweep varies the fused GEMM tiling (`tile_m`, `tile_n`) while keeping
the thesis-facing `parallel_seq`, `parallel_heads`, and `parallel_head_dim`
axes pinned at `1` until the local fused lowering grows real lane-level
parallel splits for those dimensions.
That design file should also be the source of the retained topology list used by
operator-local tests and future topology selection logic.

## Block 2

### Public tensor contract

- inputs:
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
The retained `v2` surface is currently pinned to `parallel_seq=1`,
`q_seq_tile=32`, `kv_seq_tile=64`, `parallel_heads=1`, and
`o_proj_acc_depth=1`, with `emb_tile` selected per retained workload family.
That design file should also be the source of the retained topology list used by
operator-local tests and future topology selection logic.
Study metadata emitted by the in-process Dataflow patterns should include the
selected Block 2 topology ID and family so retained topology choices are visible
in benchmark outputs.

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
- the retained topology's internal `compile_rows` must be divisible by
  `parallel_seq * tile_m`

Block 3 currently resolves its retained thesis topologies through the single
design entrypoint in
[`iron/operators/addnorm_ffn_addnorm/design.py`](/home/cj/iron/iron/operators/addnorm_ffn_addnorm/design.py)
and translates them into the pipelined wrapper in
[`iron/operators/addnorm_ffn_addnorm/op.py`](/home/cj/iron/iron/operators/addnorm_ffn_addnorm/op.py).
That design file should also be the source of the retained topology list used by
operator-local tests and future topology selection logic.
Study metadata emitted by the in-process Dataflow patterns should include the
selected block topology IDs and families so retained topology choices are visible
in benchmark outputs.
Those topology fields should also be reserved in the shared transformer-layer
result schema so benchmark CSVs expose them as stable columns rather than
unordered extra metadata.

At the study/app layer, retained topology selection should remain optional.
`TransformerLayerSpec` may carry `block1_topology_id`, `block2_topology_id`, and
`block3_topology_id` overrides so manifests and study cases can pin retained
topologies explicitly; when omitted, each block should resolve its default
retained topology through its local design entrypoint.
When parity validation is enabled, the same requested topology overrides should
be forwarded into the parity path so correctness checks exercise the same
retained topology selection as the benchmark rows.
Parity CSV outputs should preserve the retained block topology IDs and families
with a stable parity-column order, rather than relying on insertion-ordered
extra fields.
For direct automation, the same retained topology overrides should also be
accepted by the benchmark/job CLI layer so study manifests, ad hoc benchmark
invocations, and unattended jobs all share the same topology-selection surface.
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
families, with stable support-matrix column ordering, so unsupported or
filtered rows remain attributable to the retained topology that was requested or
resolved.
For the iGPU comparison study, GPU result rows should preserve the selected
reference NPU row's block topology IDs and families under explicit
`reference_npu_*` columns so best-NPU-vs-iGPU results remain attributable to
the retained topology that won NPU selection.
Those `reference_npu_*` columns should also be reserved in the shared result
schema so GPU compare CSVs expose them as stable fields rather than unordered
extras.
Bottleneck summary CSV/JSON/text outputs should also preserve the retained
block topology IDs from the analyzed rows so dominant-component reports remain
attributable to the resolved Dataflow topology surface.
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
and then translates the thesis-facing topology into the imported
`ffn_addnorm` runtime parameters. Shorter total `seq_len` values are still
allowed because the imported runtime pads and chunks the final row block.
The current retained Block 3 topology sweep widens only the lane-distribution
axes already exercised by the imported `ffn_addnorm` design coverage, so the
retained `768/3072` family now explores both `parallel_seq=2, parallel_int_dim=6`
and `parallel_seq=4, parallel_int_dim=3` while keeping the proven tile sizes,
depth, and GeLU staging fixed.

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
