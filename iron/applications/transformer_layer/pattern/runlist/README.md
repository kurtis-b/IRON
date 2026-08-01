<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runlist Transformer Layer (`runlist`)

Paper label: **`runlist`**. Results rows carry `execution_mode=runlist`.

`AIETransformerRunlist` (`op.py`) runs a full transformer layer as an NPU
runlist of **fine-grained** operators, with intermediates moved explicitly
between them. It is the granular counterpart to
[`hybrid`](../hybrid/), which sequences the same layer over a few coarse fused
kernels instead.

The layer is composed from single-purpose operators:

- `AIEGEMM` — every projection and attention matmul
- `AIETranspose`, `AIESoftmax`, `AIEElementwiseMul`, `AIECausalMask`
- `AIEGELU`
- `AIELayerNorm`, `AIEAddAndNorm`, `AIEElementwiseAdd`

## Blocked Attention

Long sequences cannot materialise the full attention score matrix, so above a
threshold the operator switches to blocked attention over query blocks. Two
constants in `op.py` govern this:

- `MAX_ATTENTION_SCRATCH_BUFFER_BYTES` — the scratch budget that forces
  blocking
- `MIN_BLOCKED_QUERY_BLOCK_SIZE` — the smallest query block considered

`test_runlist_long_seq_uses_blocked_attention` pins that behaviour.

## Configuration

`study/end_to_end` searches over per-operator configurations; the candidates it
considers live in `study/end_to_end/runlist_candidates.json`.

## Testing

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

pytest iron/applications/transformer_layer/pattern/runlist/
```

The end-to-end test compiles and executes on the NPU against the shared golden
reference in `pattern/reference.py`, so it needs real hardware and takes
minutes. `test_runlist_runtime_sync_metadata` is a pure-Python check and does
not.
