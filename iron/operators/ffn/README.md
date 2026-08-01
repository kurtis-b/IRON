<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined FFN Operator

This operator implements the transformer feed-forward network only:

- input: `hidden_states [seq, hidden_size]`
- weights:
  - `W_up [hidden_size, intermediate_size]`
  - `W_down [intermediate_size, hidden_size]`
- output: `hidden_states [seq, hidden_size]`

The exact math is:

```text
output = GeLU(input @ W_up) @ W_down
```

## Intended Structured Surface

The transformer-layer thesis app uses FFN as a standalone coarse kernel. The
study-facing surface should therefore be topology-driven, with:

- runtime-supported topologies
- practical exploration topologies
- theoretical topologies

Those topologies are expected to capture:

- `parallel_seq`
- `parallel_int_dim`
- `tile_m`
- `tile_k`
- `tile_n`
- `down_proj_depth`
- `gelu_stage`

## Current Implementation Notes

The implementation in this directory is derived from the existing pipelined FFN
core design already used inside larger fused operators.

Important constraints of the current runtime path:

- the operator uses a two-stage pipeline: up projection plus GeLU, then down
  projection
- it currently supports stage-only bringup modes in the low-level operator
  surface
- large `down_proj_depth` configurations should not force repeated runtime
  re-preparation between launches; repeated benchmark/test launches are part of
  the expected operator contract

## Testing Expectations

This operator should eventually match the structured-operator maturity level of
the other thesis coarse kernels:

- one reference module exposing only `generate_golden_reference()`
- one topology module exposing runtime/practical/theoretical catalogs
- one test module covering:
  - topology contracts
  - retained numeric runtime cases
  - generalized runtime spot checks
