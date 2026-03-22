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

- Same-tile full-row staging uses the stable replay and accumulation transport
  path for the current envelope:
  - LN1 replay in both modes
  - LN2 replay in `memtile` mode only for `parallel_heads == 4` and `effective_ffn_branches <= 4`
  - O-proj accumulation staging only when there is a single O-proj stage core
  - FFN-down accumulation staging only for `effective_ffn_branches <= 4`
- `memtile` LN1 mode: no LN1 DDR refill stream at shim.
- `ddr` LN1 mode: LN1 adds DDR stage traffic at shim (drain + refill pattern in runtime).

## Common pressure points

- Shim output budget with high FFN branch count (`pffn`) plus grouped O-proj (`opg > 1`)
- Tail memtiles carrying FFN down-acc/reduction traffic
- Memtile channel assignment for replay and accumulation sites; BD pressure is
  reduced, but channel pressure is still topology-dependent
- `emb_tile=128` topologies where FIFO depth pushes L1 usage

## Current note

- For memtile topology `4pheads_6pffn_6pacc_2opg`, B-weight split is required when pruning is disabled.
- Forcing no split leads to compile-time shim-output failure: `total_streams=17 > 16`.
- The broader long-sequence complex subset is currently passing with the
  narrowed stable envelope:
  - `6pheads_2pffn_8pacc_2opg`
  - `4pheads_4pffn_8pacc_4opg`
  - `4pheads_6pffn_6pacc_2opg`
  - `16heads_4096ffn_32qseqtile_64kvtile_128embtile_4pheads_4pffn_8pacc_4opg` (`ddr` only)
