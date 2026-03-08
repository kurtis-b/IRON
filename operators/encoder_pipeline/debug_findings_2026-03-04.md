# encoder_pipeline Debug Findings (Current)

Last updated: 2026-03-08

## Current facts

- Layer norm requires full-row stats, so LN1/LN2 remain two-pass.
- LN1 staging is mode-specific:
  - `memtile`: direct on-chip LN1 broadcast to FFN-up branches.
  - `ddr`: single LN1 drain + single LN1 refill stream, then on-chip broadcast.

## Fixes in current baseline

1. DDR LN1 replay placement avoids high-pressure memtile combinations in high-acc grouped topologies.
2. DDR FFN-down accumulation memtile placement avoids col3 collision with `W_O` fanout.
3. For `emb_tile >= 128`, default FIFO depths are reduced where needed (`B`-weight, FFN-down reduce, FFN-down out) to prevent L1 over-allocation.
4. Shim-output budgeting remains mode-aware (`+1` LN1 stream only in DDR mode).

## Regression validation (2026-03-08)

- `pytest operators/encoder_pipeline/test.py -k lnstage_ddr -q`:
  - `80 passed, 30 skipped, 95 deselected`
- `pytest operators/encoder_pipeline/test.py -k lnstage_memtile -q`:
  - `65 passed, 30 skipped, 110 deselected`

## Guardrails

- Do not change numerical thresholds to mask liveness/routing issues.
- Preserve LN two-pass semantics and row-complete staging requirements.
