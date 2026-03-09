# encoder_pipeline Debug Findings

## Stable design facts

- Layer norm is two-pass in both LN stages because row-complete statistics are required.
- LN1 staging modes:
- `memtile`: LN1 output is broadcast on-chip to FFN-up branches.
- `ddr`: one drain to DDR, one refill from DDR, then on-chip broadcast.

## Failure classes seen during bring-up

- Memtile/channel pressure in high-parallel tails (shim outputs, memtile channels, BD budget).
- Runtime deadlock from stream-order or token-flow mismatch.
- L1 overflow on `emb_tile=128` topologies when FIFO depths are too deep.

## Fix patterns that were effective

- Keep shim-output budgeting mode-aware (`+1` only for DDR LN1 staging stream).
- Rebalance high-pressure tail memtile placement instead of changing numeric thresholds.
- Reduce selected FIFO depths for large-embedding topologies to stay within 64 KiB L1.
- Validate runtime fill/drain counts against core loop trip counts when diagnosing deadlock.

## Guardrails

- Do not relax correctness thresholds to hide routing/liveness issues.
- Preserve layer-norm full-row/two-pass behavior.
