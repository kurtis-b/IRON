# encoder_pipeline

Fused encoder operator: `MHA + AddNorm1 + FFN + AddNorm2`.

## LN1 staging modes

- `ln1_staging_design="memtile"`:
  - LN1 mul+add broadcasts directly on-chip to FFN-up branches.
- `ln1_staging_design="ddr"`:
  - LN1 uses a single stage stream to DDR per tap, then a single DDR refill stream.
  - Refilled LN1 data is broadcast on-chip to all FFN-up branches.
  - No per-branch host refill streams.

## Key constraints

- `emb_tile * proj_acc_depth == embed_sz`
- `o_proj_acc_group_size in {1,2,4}` and divides `parallel_heads`
- Layer norm remains two-pass (full-row stats requirement):
  - pass 1: sum/sumsq
  - pass 2: normalized output

## Resource guards

- Compute tiles:
  - `parallel_heads*4 + 3 + 2*effective_ffn_branches <= 32`
- Shim output stream guard (mode-aware):
  - `total = 5 (Q/K/V/W_O/R) + estimated_B_streams + ln1_ddr_streams`
  - `ln1_ddr_streams = 1` in DDR mode, `0` in memtile mode
  - require `total <= 16`
- Per-column shim and memtile BD/channel limits are validated during allocation/lowering.

## Current regression status (2026-03-08)

- DDR mode (`-k lnstage_ddr`): `80 passed, 30 skipped`
- Memtile mode (`-k lnstage_memtile`): `65 passed, 30 skipped`

## Quick runs

```bash
pytest operators/encoder_pipeline/test.py -q
pytest operators/encoder_pipeline/test.py -q -k stage_profile
python operators/encoder_pipeline/profile_debug_modes.py --clean-build
```
