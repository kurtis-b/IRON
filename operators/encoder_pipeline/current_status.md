# encoder_pipeline Current Status

## Stable design facts

- `encoder_pipeline` is `MHA + AddNorm1 + FFN + AddNorm2`.
- LN1 and LN2 are both two-pass layer norm stages because they require full-row
  statistics.
- `ln1_staging_design="memtile"` broadcasts LN1 output on-chip.
- `ln1_staging_design="ddr"` stages LN1 output through DDR once, then
  broadcasts on-chip.

## Current row-store envelope

Active row-store use on the current tree:

- `ln1Replay`: always
- `outOProjAccum*`: enabled for the current stable `parallel_heads <= 4`
  envelope, with FIFO fallback when no safe memtile slot exists
- `ffnDownAccum*`: enabled only when `effective_ffn_branches <= 4`, with
  low-head DDR layouts falling back to FIFO by default
- `ln2Replay`: enabled only in `memtile` mode when:
  - `parallel_heads == 4`
  - `effective_ffn_branches <= 4`

Special tail case:

- for `parallel_heads >= 6`, `proj_acc_depth >= 16`,
  `effective_ffn_branches == 1`, AddNorm2 consumes direct FFN-down replay
  instead of creating `ln2Replay`

## Supported topology constraints

- `emb_tile * proj_acc_depth == embed_sz`
- `parallel_heads % o_proj_acc_group_size == 0`
- `o_proj_acc_group_size <= parallel_heads`
- only evenly partitioned FFN layouts are supported
- unsupported uneven FFN partitions fail during design construction

## Current validated state

Tracked selection results that are currently expected:

- `lnstage_memtile`: `100 passed, 40 skipped`
- `lnstage_ddr`: `115 passed, 40 skipped`

Representative passing cases:

- memtile control:
  - `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- fast DDR case:
  - `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
- higher-`pacc` passing memtile case:
  - `encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`

## Current bottlenecks

### Full vs isolated-stage gap

Representative measurements:

- memtile control
  - `full = 16488.12 us`
  - max isolated stage `addnorm1 = 11739.34 us`
  - exposed gap `= 4748.78 us`
- fast DDR case
  - `full = 12226.31 us`
  - max isolated stage `addnorm1 = 8984.99 us`
  - exposed gap `= 3241.32 us`
- higher-`pacc` memtile case
  - `full = 4349 us`
  - max isolated stage `addnorm1 = 3514 us`
  - exposed gap `= 835 us`

Interpretation:

- `AddNorm1` is still the slowest isolated stage on the key passing cases
- the `512seq` memtile and fast DDR cases still expose meaningful
  non-overlapped work above the stage bottleneck
- the high-`pacc` passing memtile case is much closer to ideal overlap and is
  more directly `AddNorm1`-bound

### Known remaining pressure points

- accumulation-core L1 pressure on O-proj accumulation cores
- accumulation-core L1 pressure on FFN-down stage/root cores
- tail memtile output-channel pressure on the more aggressive `64q/48e/16pacc`
  families
- final `memLN2` shim-drain pressure in the `64embtile / 12pacc` experiments

## What is no longer the main blocker

- part-count-driven memtile BD growth on the LN/O-proj/FFN-down row staging
  paths

That was the old forwarded-FIFO limitation. Current row-store lowering removed
that as the dominant issue on the stable envelope.
