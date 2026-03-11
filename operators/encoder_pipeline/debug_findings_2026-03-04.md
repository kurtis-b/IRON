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

## Effective debug patterns

- Keep shim-output budgeting mode-aware (`+1` only for DDR LN1 staging stream).
- Rebalance high-pressure tail memtile placement instead of relaxing numeric thresholds.
- Keep FIFO depths bounded for `emb_tile=128` topologies.
- Validate runtime fill/drain counts against core loop trip counts.

## Guardrails

- Do not relax correctness thresholds to hide routing/liveness issues.
- Preserve layer-norm full-row/two-pass behavior.
