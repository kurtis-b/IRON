# encoder_pipeline Resource Utilization (Current)

Last updated: 2026-03-08

## NPU2 capacity model

- Compute tiles: 32 total, 64 KiB L1 each
- Memtiles: 512 KiB L2 each
- Compute-tile DMA channels: 2 in / 2 out
- Memtile DMA channels: 6 in / 6 out
- Memtile DMA block budget: 48 blocks per memtile DMA op

## Budget guards in design

- Compute tiles:
  - `parallel_heads*4 + 3 + 2*effective_ffn_branches <= 32`
- Shim output stream budget:
  - `total = 5 + estimated_B_streams + ln1_ddr_streams`
  - base `5` is `Q/K/V/W_O/R`
  - `ln1_ddr_streams = 1` in DDR mode, `0` in memtile mode
  - require `total <= 16`

## LN1 staging resource effect

- Memtile mode:
  - no LN1 DDR shim refill stream.
- DDR mode:
  - one LN1 stage drain to DDR and one LN1 refill stream from DDR.
  - refill stream is broadcast on-chip to FFN-up branches (no per-branch host refill).

## Common hotspots

- Shim cols `0/1/2/3` from `Q/K/V/W_O`
- Shim col `7` from residual/LN traffic
- Tail memtiles carrying FFN-down accumulation/reduction and replay FIFOs

## Utilization inspection workflow

1. Generate a design (`pytest -x ...` or direct build).
2. Inspect:
   - `build/encoder_pipeline_*.mlir.prj/input_physical.mlir`
   - `build/encoder_pipeline_*.mlir.prj/input_with_addresses.mlir`
3. Record:
   - compute-tile count and mapping,
   - per-memtile channel usage (in/out),
   - per-memtile DMA block usage.
