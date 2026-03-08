# encoder_pipeline

Fused encoder operator: `MHA + AddNorm1 + FFN + AddNorm2`.

## Files

- `design.py`: shared placement, workers, and runtime sequence
- `design_ln1_ddr.py`: DDR-specific LN1 staging hooks
- `design_ln1_memtile.py`: memtile-specific LN1 staging hooks
- `op.py`: operator API, artifact generation, host buffers
- `test.py`: topology matrix and stage-profile tests

## LN1 staging modes

- `ln1_staging_design="memtile"`:
  - LN1 mul+add broadcasts directly on-chip to FFN-up branches.
- `ln1_staging_design="ddr"`:
  - LN1 uses a single stage stream to DDR per tap, then a single DDR refill stream.
  - Refilled LN1 data is broadcast on-chip to all FFN-up branches.
  - No per-branch host refill streams.

## Mapping summary

- MHA lane `i` is fixed at:
  - QK `(i,2)`, softmax `(i,3)`, PV `(i,4)`, O-proj `(i,5)`.
- Tail tiles (`LN1`, `FFN up/down`, `LN2`) are selected by `mapping_validation.py`.
- FFN down reduction supports:
  - full chain through all selected down cores, or
  - chain through all-but-one down core (dual LN2 FFN input case).

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

## Quick runs

```bash
pytest operators/encoder_pipeline/test.py -q
pytest operators/encoder_pipeline/test.py -q -k stage_profile
python operators/encoder_pipeline/profile_debug_modes.py --clean-build
```
