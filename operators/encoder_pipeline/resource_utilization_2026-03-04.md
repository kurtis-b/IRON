# encoder_pipeline Resource Utilization

## Hardware limits (NPU2)

- Compute tiles: `32` (64 KiB L1 each)
- Memtile DMA channels: `6 in / 6 out`
- Compute-tile DMA channels: `2 in / 2 out`
- Memtile DMA BD budget: `48` blocks per memtile DMA operation

## Design-side checks

- Compute tiles: `parallel_heads * 4 + 3 + 2 * effective_ffn_branches <= 32`
- Shim outputs: `5 + estimated_B_streams + ln1_ddr_streams <= 16`
- `5` fixed streams = `Q/K/V/W_O/R`
- `ln1_ddr_streams = 0` in `memtile` mode, `1` in `ddr` mode

## Mode impact

- `memtile` LN1 mode: no LN1 DDR refill stream at shim.
- `ddr` LN1 mode: LN1 adds DDR stage traffic at shim (drain + refill pattern in runtime).

## Common pressure points

- Shim output budget with high FFN branch count (`pffn`) plus grouped O-proj (`opg > 1`)
- Tail memtiles carrying FFN down-acc/reduction traffic
- `emb_tile=128` topologies where FIFO depth pushes L1 usage

## Current note

- For memtile topology `4pheads_6pffn_6pacc_2opg`, B-weight split is required when pruning is disabled.
- Forcing no split leads to compile-time shim-output failure: `total_streams=17 > 16`.
