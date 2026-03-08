# encoder_pipeline Resource Utilization (Current)

Last updated: 2026-03-08

## NPU2 limits used by the design

- Compute tiles: 32 total, 64 KiB L1 each
- Memtiles: 512 KiB L2 each
- Compute-tile DMA channels: 2 in / 2 out
- Memtile DMA channels: 6 in / 6 out
- Memtile DMA block budget: 48 blocks per memtile DMA op

## Enforced budget guards

- Compute tiles:
  - `parallel_heads*4 + 3 + 2*effective_ffn_branches <= 32`
- Shim output stream budget:
  - `total = 5 + estimated_B_streams + ln1_ddr_streams`
  - base `5` is `Q/K/V/W_O/R`
  - `ln1_ddr_streams = 1` in DDR mode, `0` in memtile mode
  - require `total <= 16`

## Mode-specific LN1 impact

- Memtile mode:
  - no LN1 DDR shim refill stream.
- DDR mode:
  - one LN1 stage drain to DDR and one LN1 refill stream from DDR.
  - refill stream is broadcast on-chip to FFN-up branches (no per-branch host refill).

## Current high-pressure areas

- Shim cols `0/1/2/3`: `Q/K/V/W_O` ingress fanout
- Shim col `7`: residual and LN2 traffic
- Tail down-proj tiles with `emb_tile=128`: L1 pressure from B weights + reduction/out FIFOs
- Tail memtiles: FFN-down accumulation/reduction + replay channels/BDs

## Practical notes from current baseline

- DDR and memtile regressions are currently green with clean builds.
- Default FIFO depth adjustments for `emb_tile >= 128` are required to keep FFN tiles within 64 KiB L1.
