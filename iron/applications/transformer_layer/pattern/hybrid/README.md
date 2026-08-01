<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coarse Runlist Transformer Layer (`hybrid`)

Paper label: **`coarse runlist`**. The repo keeps `hybrid` as the internal
module name and CSV/schema key, so results rows for this strategy carry
`execution_mode=hybrid` and `pattern_label=Coarse runlist`.

`AIETransformerHybrid` (`op.py`) runs a full transformer layer as a runlist of
a few **coarse** kernels rather than many fine-grained operators. Each coarse
kernel already fuses what would otherwise be several dispatches, so the
sequence the host submits is short and the intermediates largely stay on the
device.

The layer is composed from these operators:

- `AIEQKVProj` — fused Q/K/V projection
- `AIEMHAOutProj` — attention and output projection as one staged kernel
- `AIEFFN` — up projection, GeLU, and down projection as one staged kernel
- `AIEAddAndNorm`, `AIELayerNorm`, `AIEElementwiseAdd` — residual and
  normalisation positions

Compare with the two other strategies:

| Strategy | Granularity | Who sequences the work |
| --- | --- | --- |
| [`hybrid`](.) | coarse fused kernels | NPU runlist |
| [`runlist`](../runlist/) | fine-grained operators | NPU runlist |
| [`offload`](../offload/) | GEMM-level | host, per operator |

## Configuration

`default_hybrid_operator_config(seq_len, hidden_size, intermediate_size,
num_heads, *, workload_variant="encoder_bert", num_aie_columns=8)` builds a
per-operator configuration from the workload shape. `study/end_to_end` searches
over variations of it; the candidates it considers live in
`study/end_to_end/hybrid_candidates.json`.

## Testing

```bash
source /opt/xilinx/xrt/setup.sh
source /path/to/iron/ironenv/bin/activate
cd /path/to/iron

pytest iron/applications/transformer_layer/pattern/hybrid/
```

The test compiles and executes the layer on the NPU and validates against the
shared golden reference in `pattern/reference.py`, so it needs real hardware
and takes minutes. `test_hybrid_runtime_sync_metadata` is a pure-Python check
and does not.
