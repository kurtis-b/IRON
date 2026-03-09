# encoder_pipeline Resource Utilization

## Device limits used by mapping checks (NPU2)

- Compute tiles: 32 (64 KiB L1 each)
- Memtile DMA channels: 6 in / 6 out
- Compute-tile DMA channels: 2 in / 2 out
- Memtile DMA BD budget: 48 blocks per memtile DMA operation

## Guard formulas in the design

- Compute tile budget:
- `parallel_heads * 4 + 3 + 2 * effective_ffn_branches <= 32`
- Shim-output budget:
- `5 + estimated_B_streams + ln1_ddr_streams <= 16`
- `5` corresponds to `Q/K/V/W_O/R`
- `ln1_ddr_streams = 1` in DDR mode and `0` in memtile mode

## Mode impact on resources

- `memtile` LN1 mode:
- no LN1 DDR refill stream at shim
- `ddr` LN1 mode:
- one LN1 drain stream and one LN1 refill stream at shim
- refill data is broadcast on-chip to FFN-up branches

## Typical pressure points

- Shim columns carrying `Q/K/V/W_O` fanout
- Tail-region memtiles used by FFN down-projection accumulation/reduction
- `emb_tile=128` tails where FIFO depth can push L1 usage over 64 KiB
