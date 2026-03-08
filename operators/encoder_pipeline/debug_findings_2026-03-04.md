# encoder_pipeline Debug Findings (Current)

Last updated: 2026-03-08

This note keeps only active findings and recently validated fixes.

## Scope

- Operator: `operators/encoder_pipeline`
- Modes: LN1 DDR staging and LN1 memtile staging
- Invariant: layer norm requires full-row statistics, so two-pass replay/staging must be preserved.

## Confirmed invariants

1. `emb_tile * proj_acc_depth == embed_sz`
2. `o_proj_acc_group_size in {1,2,4}` and divides `parallel_heads`
3. Runtime host transfer counts must match core loop object counts
4. LN1/LN2 keep two-pass behavior (sum/sumsq pass + normalized/output pass)

## Verified fixes

### 1) LN1 broadcast over-production in multi-branch memtile FFN

- Fix: broadcast/replay group count is capped to required FFN-up demand
  (`ln1_broadcast_groups = max(ffn_col_group_counts)`).
- Result: removed the repeat-run token imbalance seen in the prior
  `1pheads_2pffn_8pacc_1opg` memtile deadlock path.

### 2) Per-tile DMA channel overflow in `1pheads_4pffn_8pacc_1opg` (memtile)

- Root cause: col7 (`mem_tile_7_1` / `shim_noc_tile_7_0`) was overloaded by
  residual/LN traffic plus FFN B-stream staging.
- Fixes:
  - limited low-head/high-acc memtile remap special-case to exactly 2 branches;
  - made FFN B-stream allocator prefer non-col7 columns, with col7 fallback only.
- Validation:
  - `pytest -q operators/encoder_pipeline/test.py -k "lnstage_memtile and 1pheads_4pffn_8pacc_1opg" -x`
  - Result: pass.

## Current open issue

### Memtile BD overflow in `6pheads_2pffn_8pacc_2opg`

- Failure class: compile-time memtile DMA block limit
- Symptom:
  - `aie.memtile_dma op has more than 48 blocks`
- Interpretation:
  - this is a per-memtile BD pressure issue in the selected topology, not a numerical threshold problem.

### Detailed diagnosis (2026-03-08)

- Repro:
  - `rm -rf ./build && source /opt/xilinx/xrt/setup.sh && source ~/iron/ironenv/bin/activate && pytest -q operators/encoder_pipeline/test.py -k "memtile" -x`
  - first failing test:
    - `iter0-lnstage_memtile-..._6pheads_2pffn_8pacc_2opg`
- Compiler reports:
  - `'aie.memtile_dma' op has more than 48 blocks`
  - failing memtile DMA op operand `%32`
- Mapping check:
  - `%32` resolves to `mem_tile_4_1`.
- Why `mem_tile_4_1` exceeds 48 BDs in this topology:
  - three depth-8 bidirectional replay/staging FIFO pairs are colocated:
    - `ln1Replay/ln1ReplayPart`
    - `outOProjAccumIn1/outOProjAccumOut1`
    - `ffnDownAccum1/ffnDownPart1`
  - plus one depth-2 B-weight pair:
    - `inBDown/memBDown`
  - BD contribution in generated memtile DMA:
    - `3 * (8+8) + (2+2) = 52` BD entries
  - this matches the compiler failure threshold (`>48`).

## Practical debug workflow

1. Reproduce one topology with `-x`.
2. Classify failure first:
   - compile-time (L1/channel/BD) vs runtime timeout.
3. For runtime timeouts, sweep debug modes to isolate stage.
4. For compile failures, inspect generated `input_physical.mlir`:
   - per-memtile channel usage,
   - per-memtile BD allocation,
   - concentrated traffic on col7 and tail-stage columns.

## Guardrails

- Do not change numerical thresholds to mask routing/liveness defects.
- Do not remove LN replay/two-pass semantics; they are required for correctness.
