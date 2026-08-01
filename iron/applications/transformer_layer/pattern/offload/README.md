<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Offload Transformer Layer (`offload`)

Paper label: **`offload`**. Results rows carry `execution_mode=offload`.

`AIETransformerOffload` (`op.py`) keeps control of the layer on the **host**
and offloads only the GEMMs to the NPU, through `AIEDynamicGEMM`. Everything
between them — reshapes, softmax, normalisation, residuals — runs in torch on
the host. This is the conventional accelerator-offload arrangement, and the
baseline the other two strategies are measured against.

Eight GEMMs are offloaded, listed in `OFFLOAD_GEMM_OPERATOR_NAMES`:

```text
q_proj  k_proj  v_proj  attn_scores  attn_output  output_proj  up_proj  down_proj
```

Because each is a separate host-driven dispatch with its own buffer
synchronisation, this strategy is dominated by host round trips rather than by
NPU compute. Two practical consequences:

- It is the noisiest of the three between runs — roughly ten times the
  run-to-run drift of `hybrid` or `runlist`. `study/compare_results_roots`
  gives it a correspondingly wider tolerance.
- It is the strategy most sensitive to the runtime stack. An XRT version
  change alone has moved `offload` latency by 19-39% at `seq_len >= 4096`
  while leaving the other two within 0.6%.

## Shared Artifacts

All eight GEMMs share a single xclbin, and long sequences reuse one
instruction-sequence buffer across query blocks. `test_offload_artifacts_share_one_xclbin`
and `test_offload_long_seq_reuses_query_block_insts` pin that. Per-operator
supertile overrides are deliberately rejected — see
`test_offload_rejects_per_operator_supertile_overrides`.

Query blocking for long sequences reuses `_resolve_query_block_size` from
[`runlist`](../runlist/), so both strategies block attention the same way.

## Configuration

`study/end_to_end` searches over configurations; the candidates it considers
live in `study/end_to_end/offload_candidates.json`.

## Testing

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

pytest iron/applications/transformer_layer/pattern/offload/
```

`test_transformer_layer_offload_smoke` compiles and executes on the NPU
against the shared golden reference in `pattern/reference.py`, so it needs real
hardware. The configuration-resolution tests are pure Python and do not.
