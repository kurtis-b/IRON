# encoder_pipeline Debug Findings

## Stable facts

- Layer norm remains two-pass in LN1 and LN2 (row-complete statistics requirement).
- `memtile` mode: LN1 output broadcasts on-chip to FFN-up branches.
- `ddr` mode: LN1 output drains to DDR once, refills once, then broadcasts on-chip.

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

## Compiler-side repro

- Minimal failing pattern: AddNorm1 post-stage residual path with shim-managed host traffic.
- Smallest confirmed repro shape:
  - `muladd`-only probe
  - `parts=8`, `rows=64`, `groups=1`, `branches=1`, `seq_tile=32`, `emb_tile=96`
- Worker-managed residual input/output passes.
- Shim-managed residual input or shim-managed residual output fails with:
  - `ERT_CMD_STATE_TIMEOUT`
  - `ctx_pc = 0x28B060AD`
- Lowering difference:
  - compute-side AddNorm1 post graph stays effectively unchanged
  - residual endpoint moves to shim
  - runtime lowers that path into many serial awaited shim DMA tasks (`repeat_count = 7` per row task)
- Interpretation:
  - strong compiler/runtime bug candidate on the shim-managed residual path
  - not evidence of a replay-row-store bug

## Effective debug patterns

- Keep shim-output budgeting mode-aware (`+1` only for DDR LN1 staging stream).
- Rebalance high-pressure tail memtile placement instead of relaxing numeric thresholds.
- Keep FIFO depths bounded for `emb_tile=128` topologies.
- Validate runtime fill/drain counts against core loop trip counts.

## Guardrails

- Do not relax correctness thresholds to hide routing/liveness issues.
- Preserve layer-norm full-row/two-pass behavior.
