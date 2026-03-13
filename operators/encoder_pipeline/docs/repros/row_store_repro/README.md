# Row-Store AddNorm1 Repro

This folder contains the smallest failing `encoder_pipeline` row-store case found so far.

## Failing Case

- mode: `memtile`
- stage/debug mode: `debug=9` (`addnorm1_post_only`)
- topology:
  - `seq_len=512`
  - `d=64`
  - `heads=12`
  - `ffn_intermediate_size=3072`
  - `q_seq_tile=32`
  - `kv_seq_tile=64`
  - `emb_tile=96`
  - `parallel_heads=4`
  - `proj_acc_depth=8`
  - `o_proj_acc_group_size=4`
  - `nB_tiles_distributed=1`

This is the smallest case that still fails after replacing LN1 replay with `aie.memtile_row_store`.
The same timeout also occurs for `pffn=2` and `pffn=4`, so the issue is not tied to multi-group FFN replay.

## Observed Failure

`run_runlist()` fails at runtime with:

```text
RuntimeError: runlist failed execution (ERT_CMD_STATE_TIMEOUT)
txn_op_idx = 0xFFFFFFFF
ctx_pc = 0x28B060AD
fatal_error_type = 0x00000000
fatal_error_exception_type = 0x00000000
fatal_error_exception_pc = 0x00000000
fatal_error_app_module = 0x00000000
```

Phase timings from the isolated run:

- `compile_all()`: about `24s`
- `prepare_runtime()`: about `0.16s`
- `run_runlist()`: timeout

## Files

- [`addnorm1_debug9_pffn1.mlir`](/home/agi-demo/iron/operators/encoder_pipeline/row_store_repro/addnorm1_debug9_pffn1.mlir)
  - source MLIR before row-store lowering
- [`addnorm1_debug9_pffn1.row_store_lowered.mlir`](/home/agi-demo/iron/operators/encoder_pipeline/row_store_repro/addnorm1_debug9_pffn1.row_store_lowered.mlir)
  - MLIR after `aie-opt --aie-lower-memtile-row-stores`
- [`addnorm1_debug9_pffn1.input_with_addresses.mlir`](/home/agi-demo/iron/operators/encoder_pipeline/row_store_repro/addnorm1_debug9_pffn1.input_with_addresses.mlir)
  - lowered project artifact with concrete addresses, locks, flows, and DMA BD assignments

## Useful Anchors

In `addnorm1_debug9_pffn1.row_store_lowered.mlir`:

- row-store declaration and lowered buffers/locks:
  - `ln1Replay_src`
  - `ln1Replay_dst`
  - `ln1Replay_row`
- LN1 norm core:
  - `%core_6_5 = aie.core(%tile_6_5)`
- LN1 post core:
  - `%core_4_2 = aie.core(%tile_4_2)`

In `addnorm1_debug9_pffn1.input_with_addresses.mlir`:

- LN1 tile-local FIFO buffers and locks:
  - `outOProjInput_*` on tile `(6,5)`
  - `ln1Norm_*` on tile `(6,5)` / `(4,2)`
- row-store buffers and locks:
  - `ln1Replay_*`
- row-store memtile DMA:
  - `%memtile_dma_5_1`

## How To Reproduce

From `/home/agi-demo/iron`:

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
pytest -q operators/encoder_pipeline/test.py -k "lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg" -s -x
```

For the isolated AddNorm1-post run used to produce these artifacts:

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
python - <<'PY'
import time
import numpy as np
from operators.common.aie_context import AIEContext
from operators.encoder_pipeline.op import AIEEncoderPipeline
from operators.encoder_pipeline.test import _cached_golden_reference

case = (512, 64, 12, 3072, 32, 64, 96, 4, 1, 8, 4)
debug = 9
seq_len, d, heads, intermediate_size, q_seq_tile, kv_seq_tile, emb_tile, parallel_heads, parallel_ffn, proj_acc_depth, opg = case
ref = _cached_golden_reference(seq_len, d, heads, intermediate_size, debug)
ctx = AIEContext()
op = AIEEncoderPipeline(
    seq_len=seq_len,
    d=d,
    num_heads=heads,
    seq_tile=q_seq_tile,
    kv_seq_tile=kv_seq_tile,
    emb_tile=emb_tile,
    parallel_heads=parallel_heads,
    proj_acc_depth=proj_acc_depth,
    o_proj_acc_group_size=opg,
    ffn_intermediate_size=intermediate_size,
    nB_tiles_distributed=parallel_ffn,
    debug=debug,
    ln1_weight=ref["ln1_weight"].clone(),
    ln2_weight=ref["ln2_weight"].clone(),
    ln1_staging_design="memtile",
    context=ctx,
)
ctx.compile_all()
ctx.prepare_runtime()
for buf_name in ["O"]:
    op.write_buffer(buf_name, np.zeros(op.buffers[buf_name], dtype=np.uint8))
for buf_name in ["QKV", "W_O", "OR", "B_Up", "B_Down"]:
    op.write_buffer(buf_name, ref[buf_name].flatten())
op.run_runlist()
PY
```

That run should end with the timeout shown above.

For direct compiler-side regeneration from the frozen lowered MLIR:

```bash
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
AIECC=$(python - <<'PY'
from pathlib import Path
import aie.utils.config as cfg
print(Path(cfg.root_path()) / "bin" / "aiecc.py")
PY
)
PEANO=$(python - <<'PY'
import aie.utils.config as cfg
print(cfg.peano_install_dir())
PY
)
python "$AIECC" \
  --no-compile-host \
  --no-xchesscc \
  --no-xbridge \
  --peano "$PEANO" \
  --dynamic-objFifos \
  --aie-generate-xclbin \
  --xclbin-name=/tmp/addnorm1_debug9_pffn1.xclbin \
  --xclbin-kernel-name=MLIR_AIE \
  --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=/tmp/addnorm1_debug9_pffn1.bin \
  operators/encoder_pipeline/row_store_repro/addnorm1_debug9_pffn1.row_store_lowered.mlir
```

## Notes

- The compile-side integration is working:
  - source MLIR is generated
  - row-store lowering runs
  - xclbin / NPU instruction generation completes
- The failure boundary is runtime only.
- This frozen minimal case already uses LN1 replay memtile column `5`.
