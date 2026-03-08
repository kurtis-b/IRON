# encoder_pipeline Debug Findings (Current)

Last updated: 2026-03-08

## Active design facts

- Layer norm requires full-row statistics, so LN1/LN2 remain two-pass.
- FFN branch count may be reduced by shared placement/resource guards in `design.py`.
- LN1 staging behavior is mode-specific:
  - `memtile`: direct on-chip LN1 broadcast to FFN-up branches.
  - `ddr`: single LN1 stage drain + single LN1 refill stream, then on-chip broadcast to FFN-up branches.

## Recent fixes

1. Removed DDR LN1 per-branch host refill pattern.
2. Replaced with single-stream DDR refill + on-chip broadcast fanout.
3. Updated shim-output stream guard to account for DDR mode (`+1` LN1 DDR refill stream).
4. Updated `op.py` OR host writes to include DDR LN1 scratch rows.

## Validation snapshot

- Generation-level check on regular topology set:
  - no LN1 DDR staging allocator failures after single-stream redesign.
- Targeted DDR hardware tests:
  - `1pheads_4pffn_8pacc_1opg`: pass.
  - `2pheads_2pffn_8pacc_1opg`: pass.

## Guardrails

- Do not relax numerical error thresholds to hide routing/liveness defects.
- Keep LN replay/two-pass semantics intact.
