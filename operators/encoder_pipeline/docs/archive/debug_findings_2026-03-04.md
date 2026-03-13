# encoder_pipeline Debug Findings

## Stable facts

- Layer norm remains two-pass in LN1 and LN2 (row-complete statistics requirement).
- Same-tile full-row staging uses `aie.memtile_row_store` with `buffer_count=2` where legal.
- `memtile` mode: LN1 output broadcasts on-chip to FFN-up branches.
- `ddr` mode: LN1 output drains to DDR once, refills once, then broadcasts on-chip.

## Current status

- The live passing path depends on the compiler-side double-buffered row-store lowering.
- Revalidated control cases after converting additional accumulator paths to row-store:
  - `memtile`: `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
  - `ddr`: `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- Full-vs-stage profiling on current passing cases:
  - `memtile` control `512seq / 96e / 4pheads / 4pffn / 8pacc / 4opg`
    - `full = 16488.12 us`
    - bottleneck isolated stage `addnorm1 = 11739.34 us`
    - exposed gap above max stage = `4748.78 us`
  - `ddr` fast case `512seq / 128e / 4pheads / 6pffn / 6pacc / 2opg`
    - `full = 12226.31 us`
    - bottleneck isolated stage `addnorm1 = 8984.99 us`
    - exposed gap above max stage = `3241.32 us`
  - higher-`pacc` passing memtile case `64q / 48e / 16pacc / 6pheads / 1pffn / 2opg`
    - `full = 4349 us`
    - bottleneck isolated stage `addnorm1 = 3514 us`
    - exposed gap above max stage = `835 us`
- Interpretation:
  - the high-`pacc` passing memtile case is much closer to ideal overlap and is
    primarily dominated by `AddNorm1`
  - the `512seq` control and fast DDR topology still expose a few milliseconds
    of sequentialized or otherwise non-overlapped work beyond the bottleneck
    stage

## High-pacc row-store experiment

- Experiment goal: check whether row-store lets `emb_tile` drop from `96/128` to `64` while raising `proj_acc_depth` to `12` (`64 * 12 = 768`) without hitting the old memtile forwarded-FIFO BD wall.
- Topology: `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_64embtile_4pheads_4pffn_12pacc_4opg` (`memtile`).
- Result: compile reaches `aie-assign-bd-ids`, and the first allocator failure is no longer any of the row-store sites. The failing op is the final `memLN2` shim drain (`aie.objectfifo @memLN2`), with:
  - `Allocator exhausted available BD IDs (maximum 24 available for channel 3)`
  - current op: `aie.dma_bd(... memref<32x64xbf16>)` with dimensions `[4,512] [8,8] [8,64] [8,1]`
- Interpretation: row-store did remove the original part-count-driven LN/O-proj/FFN-down staging bottleneck for this experiment. The next exposed limit is the non-row-store tail drain path.
- Follow-up check: simplifying `memLN2` itself to the direct row-major drain form (dropping the expanded `o_dims` layout) did not change the allocator outcome. The failure remains on the same shim-drain channel, which means the dominant pressure is the runtime output tap count on the final `32x64` tile stream, not the vectorized `dimensionsToStream` form of `memLN2`.

- Reduced-topology checks with the same `64embtile / 12pacc` split:
  - `1pheads_1pffn_12pacc`
  - `2pheads_2pffn_12pacc`
- Earlier result: both stopped in MLIR verification with:
  - `'aie.core' op memtile row store accessed by core running on non-compute tile`
- Current status: guarded LN2 row-store usage to the current stable envelope:
  - `parallel_heads == 4`
  - `effective_ffn_branches <= 4`
- After that guard, the reduced topologies also progress to allocator failure instead of verifier failure.
- Interpretation: the small-topology row-store placement issue is currently contained by the LN2 guard. The remaining blocker across the `64embtile / 12pacc` experiments is still the final output-drain allocation path, not LN/O-proj/FFN-down row-store staging.

## Recent resolved issue

- Topology: `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg` (`memtile`).
- Root cause: split B-weight TAP packing order did not match `ObjectFifo.split(...)` consumer ordering.
- Fix: chunk-aware split tap construction and runtime scheduling by `(col_group, local_group)`.
- Outcome: mismatch count dropped from catastrophic (`32702`) to stable low mismatch (`7`, max allowed `245`), and pytest case passes all iterations.

- Topology: `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg` (`memtile`).
- Root cause: grouped FFN-down accumulation was not preserving the forward-FIFO ordering semantics on `ffnDownAccum/ffnDownPart`. The grouped branch logic also made `debug=4` look like an FFN-down failure even though that mode bypasses LN2 and returns the stage-1 path at the output.
- Fix: keep grouped FFN-down accumulation in the baseline `curr_acc -> new_acc` order for remaining groups, and use the grouped branch only to change how per-group partials are formed before the single accumulator update.
- Outcome: full memtile pytest case now passes all 5 iterations with `229` mismatches (max allowed `1966`) and latency around `16.8-17.1 ms`.

- Topology: `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_6pffn_8pacc_2opg` (`memtile`).
- First fix: non-grouped FFN-down path was missing reduction producers after the grouped-accumulation refactor. When grouping is disabled, every branch must remain a stage branch.
- Compile-passing placement rebalance for this topology family:
  - `Q=[0,5]`, `K=[1,2]`, `V=[0,6]`, `W_O=[3,4]`
  - packed `B_Up` chunk columns `[6,4,5]`
  - packed `B_Down` chunk columns `[3,2,1]`
- Current blocker: runtime timeout (`ERT_CMD_STATE_TIMEOUT`) still occurs in normal mode after the compile constraints are removed.

## Recent row-store scope fix

- Compiler-side double-buffered row-store lowering is now used only where the current runtime path is stable:
  - LN1 replay: enabled
  - LN2 replay: only `parallel_heads == 4` and `effective_ffn_branches <= 4`
  - O-proj accumulation: current stable default is `parallel_heads <= 4`, with
    slot-based FIFO fallback when a memtile column runs out of safe row-store
    DMA pairs
  - FFN-down accumulation: only when `effective_ffn_branches <= 4`, with
    slot-based allocation and DDR low-head layouts falling back to FIFO
- Why:
  - slot-aware channel allocation was needed so FFN-down row-store does not
    collide with O-proj row-store on the same memtile
  - DDR low-head FFN-down row-store still times out at runtime on
    `1pheads_4pffn_8pacc_1opg`, so that layout stays on FIFO fallback
  - 6-branch FFN-down row-store is still not enabled by default
- Result:
  - representative failing cases in both `memtile` and `ddr` now pass again
  - full tracked selections are green again:
    - `lnstage_memtile`: `100 passed, 40 skipped`
    - `lnstage_ddr`: `115 passed, 40 skipped`

## High-pacc single-branch LN2 tail

- Topology class:
  - `parallel_heads >= 6`
  - `proj_acc_depth >= 16`
  - `effective_ffn_branches == 1`
- Symptom:
  - the `64qseqtile / 48embtile / 16pacc` single-branch tail was not fixed by
    moving grouped O-proj accumulation or `ln2Replay` between memtiles; the
    output-DMA overflow just moved between tail columns.
- Working fix:
  - stop creating `ln2Replay` for this topology class
  - let AddNorm2 consume the direct replay pass emitted by FFN-down
- Outcome:
  - `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
    now passes
  - `lnstage_ddr-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
    now passes
  - updated `64qseqtile / 16pacc` comparison matrix:
    - `120 passed, 95 failed, 295 deselected`

## Historical note

- Earlier single-row LN1 row-store integration attempts exposed runtime hangs.
- Those results are historical now; the current design uses the compiler-side double-buffered lowering.

## Effective debug patterns

- Keep shim-output budgeting mode-aware (`+1` only for DDR LN1 staging stream).
- Rebalance high-pressure tail memtile placement instead of relaxing numeric thresholds.
- Keep FIFO depths bounded for `emb_tile=128` topologies.
- Validate runtime fill/drain counts against core loop trip counts.

## Guardrails

- Do not relax correctness thresholds to hide routing/liveness issues.
- Preserve layer-norm full-row/two-pass behavior.
