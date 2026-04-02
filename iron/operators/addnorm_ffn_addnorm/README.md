<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pipelined AddNorm + FFN + AddNorm

## Semantics

Block 3 computes:

- `preadd = A + R`
- `ln1_out = LN1(preadd)`
- `output = LN2(GeLU(ln1_out @ B_Up) @ B_Down + preadd)`

`stage_only` exposes the copied `addnorm_ffn` bringup ladder:

- `-1`: no compute
- `0`: first AddNorm only
- `1`: up projection only
- `2`: down projection only
- `3`: final AddNorm only
- `None`: full compute

## Topology Axes

Runtime and exploration surfaces use:

- `tile_m`, `tile_k`, `tile_n`
- `parallel_seq`
- `parallel_int_dim`
- `down_proj_depth`
- `gelu_stage`

Canonical topology IDs look like:

- `m16_k96_n96_ps4_pi2_d8_g1`

## Current Design Limits

The current copied-`addnorm_ffn` Block 3 design is intentionally narrow:

- `tile_m = 16`
- `tile_k = 96`
- `tile_n = 96`
- `down_proj_depth in {1, 2, 4, 8}`
- `parallel_seq in {1, 2, 4}`
- `parallel_int_dim in {1, 2, 4}`
- `num_aie_columns = 2 * max(parallel_seq, parallel_int_dim)`

Retained runtime families keep promoted Block 3 runtime IDs for the workloads that are already part of the active bringup ladder. Practical and theoretical topology helpers expose the wider currently-admitted surface around those same copied-design constraints.
