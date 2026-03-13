## Row-Store Optimization Plan

### Goal

Use `aie.memtile_row_store` more aggressively on same-core row replay and
accumulation paths so lower `emb_tile` / higher `proj_acc_depth` topologies can
stay on memtile staging without hitting the old part-count-driven BD growth.

This plan is only for paths that match the row-store contract:

- one compute-tile producer
- one compute-tile consumer
- in-order fill and in-order replay of the same logical row

It does not apply to shim transport paths or final drains.

### Current Stable Row-Store Envelope

The design currently uses row-store in these places:

- `ln1Replay`: always
- `outOProjAccum*`: current stable default is `parallel_heads <= 4`, with
  slot-based FIFO fallback when a memtile column has no safe row-store channel
  pair left
- `ffnDownAccum*`: only when `effective_ffn_branches <= 4`, with slot-based
  allocation and DDR low-head layouts falling back to FIFO by default
- `ln2Replay`: only when:
  - memtile mode
  - `parallel_heads == 4`
  - `effective_ffn_branches <= 4`

For the high-`pacc` single-FFN-branch tail (`parallel_heads >= 6`,
`proj_acc_depth >= 16`, `effective_ffn_branches == 1`), the design now avoids
`ln2Replay` entirely and uses the direct replay pass emitted by FFN-down into
AddNorm2.

Everything else falls back to forwarded `ObjectFifo` staging.

### What Row-Store Should Replace

The best row-store clients in `encoder_pipeline` are the same-core row scratch
paths whose old FIFO lowering scaled with `proj_acc_depth`:

1. `outOProjAccum`
2. `ffnDownAccum`
3. `ln2Replay`

`ln1Replay` is already on row-store.

These are not row-store candidates:

- `Q/K/V/W_O`
- `B_Up/B_Down`
- `R`
- `ffnROut -> ffnRIn`
- `memLN2`

Those are transport or final-drain paths, not same-core replay paths.

### Main Design Problem

The current design still decides row-store eligibility using coarse topology
guards, for example:

- single-stage O-proj accumulation only
- `effective_ffn_branches <= 4`

Those guards are workarounds for placement and DMA-channel conflicts. They are
not the real semantic limit of row-store.

The actual question is:

- does the chosen memtile column still have a legal ingress/egress channel pair
  for another row-store client?

### Optimization Plan

#### Step 1: Widen O-proj accumulation row-store

Replace the current single-stage-only default with a channel-availability-based
policy.

Success condition:

- grouped O-proj accumulation can stay on row-store when a legal memtile slot
  exists
- fallback remains in place when no safe slot exists

#### Step 2: Widen FFN-down accumulation row-store

Replace the current `effective_ffn_branches <= 4` default with a placement-based
policy.

Success condition:

- row-store is used for every FFN-down stage branch that has a legal memtile
  slot
- only conflicting stage branches fall back

#### Step 3: Unify row-store channel allocation

Replace the current per-site hardcoded maps with a shared allocator that tracks
per-memtile row-store slots and reserved DMA channels.

The allocator should prefer:

- high channels first for row-store traffic
- keeping row-store off columns already saturated by split host streams
- deterministic fallback when no safe slot remains

#### Step 4: Revisit LN2 row-store envelope

After steps 1-3 are stable, widen `ln2Replay` beyond the current `parallel_heads
== 4` / `effective_ffn_branches <= 4` guard.

This should only happen after targeted tests confirm the compiler/runtime path is
stable on the broader layouts.

### Validation Plan

Validate each step from a clean build with targeted tests first:

- memtile and ddr variants of:
  - `4pheads_4pffn_8pacc_4opg`
  - `6pheads_2pffn_8pacc_2opg`
  - `4pheads_6pffn_6pacc_2opg`

After targeted validation, rerun:

- `pytest -q operators/encoder_pipeline/test.py -k 'lnstage_memtile' -s`
- `pytest -q operators/encoder_pipeline/test.py -k 'lnstage_ddr' -s`

### Current Progress

- Step 1 is partially implemented.
- `outOProjAccum` row-store is now widened from the old single-stage-only
  default to the current tested envelope:
  - `parallel_heads <= 4`
- Allocation is now slot-based per memtile, so later row-store users can avoid
  colliding with earlier O-proj row-store placements on the same column.
- A blanket enablement regressed the `6pheads_2pffn_8pacc_2opg` family with
  runtime timeouts in both memtile and ddr modes, so the widening was narrowed
  to the 4-head envelope instead of staying fully topology-agnostic.
- Clean-build validation for the representative `512seq` subset now passes:
  - `35 passed, 260 deselected`

- Step 2 is partially implemented.
- `ffnDownAccum` row-store now uses slot-aware allocation that is seeded from
  already-placed O-proj row-stores on the same memtile.
- A blanket `parallel_heads <= 4` enablement was too broad.
- The current landed policy is:
  - `effective_ffn_branches <= 4`
  - and in `ddr` mode, `parallel_heads >= 4`
- Reason:
  - low-head DDR topology `1pheads_4pffn_8pacc_1opg` still times out when
    FFN-down row-store is enabled
- Result:
  - full tracked selections are green again:
    - `lnstage_memtile`: `100 passed, 40 skipped`
    - `lnstage_ddr`: `115 passed, 40 skipped`

- AddNorm2 tail replay is now partially optimized for the high-`pacc`
  single-branch case:
  - when `parallel_heads >= 6`, `proj_acc_depth >= 16`, and
    `effective_ffn_branches == 1`, AddNorm2 consumes FFN-down's direct replay
    pass instead of instantiating a separate `ln2Replay` memtile stream
  - this fixes the previously failing `6pheads_1pffn_16pacc_2opg` `64seq`
    cases in both `memtile` and `ddr`
  - updated `64qseqtile / 48embtile / 16pacc` comparison matrix:
    - `120 passed, 95 failed, 295 deselected`

### Expected Outcome

If the plan works, row-store will eliminate more of the old
`proj_acc_depth`-scaled memtile FIFO staging, which should make it easier to:

- reduce `emb_tile`
- increase `proj_acc_depth`
- keep full-row staging on memtile instead of moving it to DDR

This will not fix the final `memLN2` shim-drain limit by itself. That tail path
is not a row-store candidate and must be addressed separately.
