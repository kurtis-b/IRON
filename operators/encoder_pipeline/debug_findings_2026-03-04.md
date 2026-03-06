# encoder_pipeline Debug Findings (Cleaned)

Last updated: 2026-03-06

## Scope
- Operator: `operators/encoder_pipeline`
- Goal: keep default encoder test matrix stable while investigating FFN parallelization and LN/FFN staging paths.
- Constraint kept throughout: layer norm requires full-row statistics, so replay/staging across rows is mandatory.
- Test thresholds were not changed.

## Files in active path
- `operators/encoder_pipeline/design.py`
- `operators/encoder_pipeline/op.py`
- `operators/encoder_pipeline/test.py`
- `operators/encoder_pipeline/reference.py`
- `aie_kernels/aie2p/encoder.cc`

## Current behavior snapshot

### Stable default path
- Command (2026-03-06):
  - `rm -rf ./build`
  - `source /opt/xilinx/xrt/setup.sh`
  - `source ~/iron/ironenv/bin/activate`
  - `pytest operators/encoder_pipeline/test.py -q --iterations 1`
- Result:
  - `5 passed in 85.06s`

### New LN1 DDR staging experiment (initial state)
- Command (2026-03-06, targeted):
  - `rm -rf ./build`
  - `source /opt/xilinx/xrt/setup.sh`
  - `source ~/iron/ironenv/bin/activate`
  - `ENCODER_STAGE_LN1_TO_DDR=1 pytest operators/encoder_pipeline/test.py -q -s --iterations 1 -k "1pheads_1pffn_6pacc" --maxfail=1`
- Result:
  - runtime timeout: `ERT_CMD_STATE_TIMEOUT`
  - `ctx_pc = 0x28B060AD`

## Implemented changes in this cleanup cycle

### 1) LN1->FFN DDR staging path (opt-in)
- Added opt-in knob in `design.py`:
  - `ENCODER_STAGE_LN1_TO_DDR` (default `false`)
- Dataflow when enabled:
  - `LN1 out -> memtile -> shim -> DRAM`
  - `DRAM -> shim -> memtile -> FFN-up input`
- Runtime sequence uses per-branch interleaved drain/fill ordering for this staging path.

### 2) DRAM buffer layout update for LN1 staging
- Updated `OR` layout to reserve a dedicated LN1 staging region:
  - `OR = [O region | residual R region | LN1 stage scratch]`
- `design.py`:
  - `OR_ty` now includes `ln1_dram_stage_rows = profile_replay_groups * seq_tile`
  - LN1 DDR staging taps target the appended scratch region (not the O output region).
- `op.py`:
  - OR buffer allocation updated to match new shape.
  - Host write path remains backward-compatible (writes only the existing `[O|R]` prefix).

### 3) Runtime-interface correction
- A temporary attempt to add a separate kernel BO for LN1 staging caused runtime arg mismatch (`prepare_runtime` arg-index failure).
- Final implementation keeps the kernel argument list unchanged and stages via the expanded `OR` buffer.

## Timeout root cause and fix (2026-03-06 update)

### Isolation results
- With LN1 DDR staging enabled (`ENCODER_STAGE_LN1_TO_DDR=1`), timeout reproduced in:
  - full mode (`debug=-1`)
  - stage modes (`debug=3`, `debug=6`)
- Signature remained:
  - `ERT_CMD_STATE_TIMEOUT`
  - `ctx_pc = 0x28B060AD`

### Root cause
- Runtime ordering created circular waits in the LN1->FFN handoff:
1. Waiting on tail/main fill task groups before servicing LN1 staged stream could stall LN1 producer progress.
2. Deferring FFN weight fills until *after* all LN1 refills caused another cycle:
   - FFN-up could not consume LN1 refills without `B_Up`,
   - LN1 refill could block on shallow staging FIFO depth.

### Fix implemented
- In `design.py` runtime loop:
1. Added `schedule_ffn_weight_fills(...)` helper to centralize B-stream scheduling.
2. For `ln1_ddr_stage_enabled` path:
   - keep LN1 drain steps explicit (producer relief),
   - schedule LN1 refills and FFN weight fills in the same task group so A/B streams make progress together,
   - finish task groups in an order that avoids the prior circular waits.
3. Added explicit transfer-count guard for staged LN1 taps:
   - staged objects must equal per-branch core-loop expectation
   - `ffn_col_group_counts[branch] * proj_acc_depth`.

### Validation after fix
- Target case:
  - `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc`
- Commands used (with cleanup each run):
  - `rm -rf ./build`
  - `source /opt/xilinx/xrt/setup.sh`
  - `source ~/iron/ironenv/bin/activate`
  - `ENCODER_STAGE_LN1_TO_DDR=1 ENCODER_PIPELINE_DEBUG_MODE=<mode> pytest ... -k "1pheads_1pffn_6pacc"`
- Results:
  - `debug=6`: pass, no timeout
  - `debug=3`: pass, no timeout
  - `debug=4`: pass, no timeout
  - `debug=5`: pass, no timeout
  - `debug=-1`: pass, no timeout

### Current status
- Timeout is fixed for the staged target case above.
- Default non-staged baseline remains stable:
  - `pytest operators/encoder_pipeline/test.py -q --iterations 1` -> `5 passed`.
- Enabling LN1 DDR staging globally still has compile-time resource failures in larger topologies (output DMA channel/BD limits), separate from the timeout issue.

## Larger-topology LN1 DDR staging fixes (2026-03-06 follow-up)

### Reproduced failures
- With `ENCODER_STAGE_LN1_TO_DDR=1`, larger topologies initially failed in two classes:
1. Compile-time shim output channel overflow + invalid flow channel (`source_channel = -1`) in:
   - `1pheads_2pffn_6pacc`
   - `1pheads_6pffn_6pacc` (after pruning to effective 2 branches)
2. Runtime timeout (`ERT_CMD_STATE_TIMEOUT`, `ctx_pc = 0x28B060AD`) after channel fixes.
3. Compile-time memtile BD exhaustion (`maximum 48`) in:
   - `2pheads_2pffn_6pacc`
   - `6pheads_1pffn_6pacc`

### Root causes and focused patches

#### A) Shim output overflow in staged multi-branch FFN
- Cause:
  - B-weight shim stream allocator did not reserve per-branch LN1 DDR refill streams.
  - This overbooked shim output DMA channels in staged mode.
- Fix:
  - Added `reserved_output_load_by_col` support in `_allocate_ffn_weight_mem_tile_cols_by_streams(...)`.
  - In staged mode, reserve one shim output slot per LN1 stage column before B-up/B-down placement.
  - Bias B-down placement away from LN1 stage-reserved columns to reduce contention.

#### B) Runtime timeout after legal compile
- Cause:
  - LN1 DDR branch drains were issued sequentially.
  - With multi-branch LN1 fanout, this can backpressure non-drained branch FIFOs and stall the producer.
- Fix:
  - Issue all LN1 DDR branch drains in a single task group, then finish once.
  - This allows all branch outputs to make progress concurrently and removes the stall.

#### C) Remaining memtile BD exhaustion on larger maps
- Cause:
  - LN1 staged stream placement still concentrated traffic on replay/accumulator hot columns.
- Fixes:
1. Keep LN1 staged FIFO depth shallow (configurable, default `ENCODER_LN1_DDR_STAGE_FIFO_DEPTH=1`) for staged path memtile forwards.
2. Reorder staged branch memtile preference:
   - in staged mode, deprioritize col5 in branch stage ordering;
   - for high-head staged mode, prioritize col4 before col6.
3. In `parallel_heads >= 6` and staged mode, move LN1 replay default from col4 to col5 to avoid stacking replay + staged LN1 traffic on col4.

### Validation results after fixes

#### Staged targeted regressions
- Command pattern:
  - `rm -rf ./build`
  - `source /opt/xilinx/xrt/setup.sh`
  - `source ~/iron/ironenv/bin/activate`
  - `ENCODER_STAGE_LN1_TO_DDR=1 pytest operators/encoder_pipeline/test.py -q --iterations 1 -k "<case>"`
- Results:
  - `1pheads_2pffn_6pacc`: pass
  - `1pheads_6pffn_6pacc`: pass
  - `2pheads_2pffn_6pacc`: pass
  - `6pheads_1pffn_6pacc`: pass

#### Full staged matrix
- `ENCODER_STAGE_LN1_TO_DDR=1 pytest operators/encoder_pipeline/test.py -q --iterations 1`
- Result: `5 passed in 87.64s`

#### Baseline non-staged regression check
- `pytest operators/encoder_pipeline/test.py -q --iterations 1`
- Result: `5 passed in 84.85s`

## LN1 staging design split + debug sweep script (2026-03-06)

### What changed
- Added split design entry files:
  - `operators/encoder_pipeline/design_ln1_ddr.py`
  - `operators/encoder_pipeline/design_ln1_memtile.py`
- Updated `AIEEncoderPipeline` (`op.py`) to select design entrypoint via:
  - ctor arg: `ln1_staging_design`
  - env: `ENCODER_LN1_STAGING_DESIGN`
  - default: `ddr`
- `OR` host buffer sizing now tracks selected LN1 staging design:
  - DDR mode reserves appended LN1 stage scratch rows
  - memtile mode keeps original 2x-seq rows

### Profiling script added
- New script: `operators/encoder_pipeline/profile_debug_modes.py`
- Purpose:
  - run all debug modes (`-1,0,1,2,3,4,5,6,7`) for selected encoder test cases
  - report per-mode latency/errors
  - identify bottleneck stage among `{mha, addnorm1, ffn_up, ffn_down, addnorm2}`
  - optional JSON export

### Validation run
- Command:
  - `source /opt/xilinx/xrt/setup.sh`
  - `source ~/iron/ironenv/bin/activate`
  - `python operators/encoder_pipeline/profile_debug_modes.py --design ddr --case-index 0 --warmup-iters 0 --timed-iters 1 --clean-build --output-json /tmp/encoder_profile_case0_ddr.json`
- Result:
  - Script completed and produced JSON summary.
  - Reported bottleneck stage for case 0: `ffn_up` (`5131.15 us`).

## DDR `pffn<=6` stability pass (2026-03-06)

### Goal
- Ensure DDR-staged design remains runnable for `pffn` values up to 6.

### What was validated
- Target case:
  - `ENCODER_LN1_STAGING_DESIGN=ddr pytest operators/encoder_pipeline/test.py -q -s --iterations 1 -k "1pheads_6pffn_6pacc" -o log_cli=true --log-cli-level=INFO`
  - Result: pass.
- Full suite:
  - `ENCODER_LN1_STAGING_DESIGN=ddr pytest operators/encoder_pipeline/test.py -q --iterations 1`
  - Result: `5 passed`.

### Debug root cause when forcing true 6-way physical FFN
- With high-acc branch-pruning guards disabled and B-weight split enabled, compile reaches `effective=6` but fails due hard resource limits:
  - memtile output-channel overflow on `(col=5,row=1)` and/or
  - down-core L1 overflow on intermediate/output down cores (e.g. `(5,2)`), where `memBDown + ffnDownPart + ffnDownAccum + reduce/out buffers` exceed 64 KiB by a small margin.
- This is a structural per-core buffer budget issue in current down-proj pipeline shape, not a test-threshold issue.

### Focused code updates in this pass
- Kept stable default behavior for high-acc DDR path (prunes physical FFN branches when needed) so `pffn=6` remains runnable.
- Retained topology/mapping scaffolding improvements:
  - pairwise-first reduction planning helper for larger branch counts,
  - merge-tile list plumbing support,
  - explicit FFN down-acc memtile column logging.

## DDR stage-interleave runtime update (2026-03-06)

### Goal
- Overlap `MHA+AddNorm1` with `FFN+AddNorm2` in DDR-staged LN1 mode by interleaving LN1 DDR drain/refill with adjacent tap processing.

### Runtime sequence change
- Added one-step pipeline in DDR path:
  - current tap schedules MHA-side fills first,
  - pending previous tap LN1 refill/output drain is completed while current MHA work is already in flight,
  - current tap LN1 staged drain (`LN1 -> DDR`) is issued and waited,
  - current tap refill+FFN-weight fills (`DDR -> FFN`) are scheduled and carried as pending work to next tap.
- Final output drain is deferred by one tap (pending-output model) and flushed at loop end.
- First staged drain still uses explicit wait/finish before the first refill (bootstrap barrier).

### Validation
- Targeted:
  - `ENCODER_LN1_STAGING_DESIGN=ddr pytest operators/encoder_pipeline/test.py -q -s --iterations 1 -k "1pheads_1pffn_6pacc or 1pheads_6pffn_6pacc"`
  - result: `2 passed`.
- Full suite:
  - `ENCODER_LN1_STAGING_DESIGN=ddr pytest operators/encoder_pipeline/test.py -q --iterations 1`
  - result: `5 passed`.
