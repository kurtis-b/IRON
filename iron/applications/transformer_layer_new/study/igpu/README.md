<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# iGPU Study

Status: `planned`

## Goal

Compare the best completed NPU row per `(study_case_id, seq_len)` from the
end-to-end study against the iGPU reference path.

Compared execution mode:

- `amd_igpu_reference`

## Dependency

This study depends on:

- `results/end_to_end/results.csv`

It should not define its own independent family and sequence grid.
Its execution surface is derived from the completed NPU rows selected from the
end-to-end study output.

## Selection Rule

For each `(study_case_id, seq_len)`:

- keep only completed NPU rows
- select the minimum-latency NPU row
- benchmark the iGPU reference against that selected point

## Attribution

Each GPU row should preserve the winning NPU metadata under `reference_npu_*`
fields, including:

- `reference_npu_execution_mode`
- `reference_npu_avg_latency_ms`
- `reference_npu_block1_topology_id` through `reference_npu_block5_topology_id`
- matching `reference_npu_block*_topology_family`

If no ROCm-visible GPU is available, the study should emit unsupported rows
instead of failing the full run.

## Planned Files

This study will own:

- `select.py`
- `run.py`
- `test.py`

## Outputs

Canonical output:

- `results/igpu/results.csv`
