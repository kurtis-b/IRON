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
- `intermediate_size % parallel_int_dim == 0`
- `embedding_dim % tile_k == 0`
- `intermediate_size % tile_n == 0`
- the combined topology must fit within the 8-column array

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
