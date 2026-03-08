# encoder_pipeline Resource Utilization (Current)

Last updated: 2026-03-08

This file documents the current resource model, active pressure points, and
how to collect per-design utilization from build artifacts.

## Capacity model (NPU2)

- Compute-tile L1: 64 KiB
- Mem-tile L2: 512 KiB
- Compute-tile DMA channels: 2 in / 2 out
- Mem-tile DMA channels: 6 in / 6 out
- Compute-tile budget: 32 tiles
- Memtile DMA block budget: 48 blocks per memtile DMA op

## Compute-tile budget rule

`parallel_heads*4 + 3 + 2*effective_ffn_branches <= 32`

Where:
- `parallel_heads*4`: MHA (`QK`, `softmax`, `PV`, `O-proj`)
- `+3`: (`LN1 norm`, `LN1 mul+add`, `LN2`)
- `2*effective_ffn_branches`: (`FFN up`, `FFN down`)

## Current pressure hotspots

- Col `0/1/2/3`: Q/K/V/W_O host ingress
- Col `7`: residual/LN paths and LN outputs
- Cols used by FFN B-stream staging and FFN-down accumulation (topology-dependent)

Current mapping policy now prefers non-col7 placement for FFN B-stream columns,
using col7 only as fallback, to reduce channel contention with residual/LN traffic.

## Current status snapshot

- `1pheads_4pffn_8pacc_1opg` (memtile):
  - prior per-tile DMA channel overflow on col7 was fixed by column-placement updates.
- `6pheads_2pffn_8pacc_2opg` (memtile):
  - currently fails compile with memtile BD overflow:
    - `aie.memtile_dma op has more than 48 blocks`
  - this is a per-memtile BD allocation bottleneck, not a compute-tile count limit.

## How to inspect utilization for a generated design

1. Build a topology (for example with `pytest -x`) so artifacts are generated.
2. Inspect:
   - `build/encoder_pipeline_*.mlir.prj/input_physical.mlir`
   - `build/encoder_pipeline_*.mlir.prj/input_with_addresses.mlir`
3. Record, per relevant tile/memtile:
   - compute-tile usage and L1 high-water marks,
   - memtile channel usage (in/out),
   - memtile BD usage and overflow point.
