# encoder_pipeline Debug Findings (2026-03-04)

## Scope
- Understand `operators/encoder_pipeline` and dependencies.
- Run operator tests with requested environment setup.
- Use debug modes to isolate failure root cause.
- Do not adjust test thresholds.

## Key Files Reviewed
- `operators/encoder_pipeline/README.md`
- `operators/encoder_pipeline/op.py`
- `operators/encoder_pipeline/design.py`
- `operators/encoder_pipeline/reference.py`
- `operators/encoder_pipeline/debug_modes.py`
- `operators/encoder_pipeline/test.py`
- `operators/encoder_pipeline/mapping_validation.py`
- `operators/encoder_pipeline/validate_mapping.py`
- `operators/common/aie_context.py`
- `operators/common/aie_base.py`
- `operators/common/test_utils.py`
- Kernel sources used by `op.py` artifact graph:
  - `aie_kernels/aie2p/mm.cc`
  - `aie_kernels/aie2p/softmax.cc`
  - `aie_kernels/aie2p/mha.cc`
  - `aie_kernels/aie2p/encoder.cc`
  - `aie_kernels/generic/passThrough.cc`
  - `aie_kernels/generic/add.cc`

## Dependency Understanding (operator-level)
1. `test.py` generates golden buffers via `reference.py`.
2. `test.py` instantiates `AIEEncoderPipeline` (`op.py`).
3. `op.py` resolves top-level debug mode into:
   - `mha_debug`
   - `ffn_stage_only`
   - `addnorm1_debug_mode`
   - `addnorm2_debug_mode`
4. `op.py` creates artifacts and invokes `design.py:fused_mha(...)` to generate MLIR.
5. `operators/common/compilation.py` builds kernel objects, archives them, and invokes `aiecc.py` for xclbin/bin.
6. Runtime executes runlist with buffers:
   - inputs: `QKV`, `W_O`, `OR`, `B_Up`, `B_Down`
   - output alias: `O` aliases `OR` first half.

## Environment/Test Commands Used
- Initial test command (requested ordering):
  - `conda deactivate && source /opt/xilinx/xrt/setup.sh && source ~/iron/ironenv/bin/activate && pytest operators/encoder_pipeline/test.py -q --iterations 1`
- Per user update, later runs skipped conda deactivate when conda shell init was absent.
- Per user request before retesting:
  - `rm -r ./build`

## Observed Test Outcomes

### A) Before `build/` cleanup
- Full suite run initially failed due stale artifacts and linker issues (undefined symbol) and, in separate runs, runtime instability/segfault after multiple failing cases.
- Notable compile symptom:
  - undefined symbol `matmul_bf16_bf16_wrapper`
  - several archive members reported as not valid relocatable objects.
- Inspection found zero-byte build artifacts in `build/` (including kernel `.o`, `.a`, and some target outputs).

### B) After `rm -r ./build` and clean rebuild
- Isolated first case (`encoder_64seq_..._1pheads_1pffn_6pacc`) in normal mode (`debug=-1`) compiles/runs but fails numerically:
  - errors: `41815` (also observed `46493` in another clean run)
  - max allowed: `245`

### C) Debug mode sweep on same isolated case (fresh pytest process per mode)
- `mode 0`: fail, `26412` errors
- `mode 1`: fail, `2390` errors (another run observed `4095`)
- `mode 2`: fail, `44159` errors
- `mode 3`: fail, `4095` errors (targeted run observed `2107`)
- `mode 4`: **runtime timeout** (`ERT_CMD_STATE_TIMEOUT`)
- `mode 5`: **pass** (`1 passed, 10 deselected`)

## Additional Diagnostics
- Mapping validator:
  - `python operators/encoder_pipeline/validate_mapping.py --parallel-heads 1`
  - result: `VALIDATION: PASSED`

- Per-tile mismatch concentration (targeted analysis, `seq_tile*emb_tile = 4096` elements/tile):
  - `mode 3`: majority of errors in tiles `6..11` (second `q_block`) plus small count in tile `0`.
  - This indicates non-uniform failure concentrated at `q_block` boundary/replay progression.

## Root-Cause Hypotheses (from debug isolation)
1. `mode 4` timeout localizes to the FFN-down-only path (`ffn_stage_only=1`).
   - In `design.py`, `core_fn_ffn_up_proj` when stage is disabled releases output tokens without explicitly producing deterministic zeros.
   - FFN-down-only expects meaningful `ffnUpOut` producer behavior; this mismatch likely causes deadlock/timeout risk in the down-only debug path.

2. `mode 3` tile-localized errors (mostly second `q_block`) suggest replay/order mismatch across `q_block` boundaries.
   - Relevant region: LN/memtile replay logic in `design.py`:
     - O-proj replay to LN1 pass-2 (`matmul_o_proj` second output loop)
     - LN1 two-pass consumption (`core_fn_ln1_norm`)
     - residual staging (`ffnROut -> ffnRIn`)
     - runtime fill/drain sequencing by `(q_block_idx, col_group)`
   - This aligns with the known design constraint: LN requires full-row statistics, so row-wise tile staging/replay correctness is critical.

3. Since `mode 5` passes in isolated process, AddNorm2-only path itself appears viable when upstream FFN stages are not active.

## Focused Patch Implemented
- File changed: `operators/encoder_pipeline/design.py`
- Changes:
1. Made FFN-up stage deterministic when skipped:
   - In `core_fn_ffn_up_proj`, always zero `elem_out_matmul` before optional compute.
   - This prevents stale tile data from propagating when `ffn_stage_only` disables FFN-up compute.
2. Made FFN-down skipped path deterministic:
   - In `core_fn_ffn_down_proj` skip branch, explicitly zero:
     - seeded `of_new_acc` objects
     - inner-loop `of_new_acc` objects
     - first/second-pass `of_out` objects
   - This prevents stale accumulator/output tiles when FFN-down compute is disabled.
3. Enabled declared FFN-down -> AddNorm2 DMA reshape:
   - Wired `matmul_to_addnorm_dims_out/in` onto `ffnDownOut` ObjectFifo via:
     - `dims_to_stream=matmul_to_addnorm_dims_out`
     - `dims_from_stream_per_cons=matmul_to_addnorm_dims_in`

## Post-Patch Validation
- Per user request, removed build dir before tests:
  - `rm -r ./build`
- Re-ran isolated case:
  - `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc`

### Mode summaries after patch
- `mode -1`: fail (`42820` errors)
- `mode 3`: fail (`2107` errors)
- `mode 4`: **no longer timeout**, now deterministic numeric fail (`2107` errors in isolated rerun)
- `mode 5`: pass in isolated rerun

### Per-tile pattern after patch (modes 3 and 4)
- Identical mismatch distribution:
  - tile counts: `[(0, 59), (6, 384), (7, 320), (8, 320), (9, 384), (10, 320), (11, 320)]`
- Interpretation:
  - timeout/deadlock behavior in mode 4 was removed by deterministic skip-path outputs.
  - remaining mismatch remains concentrated in second `q_block` tiles (`6..11`), so replay/order issue is still unresolved.

## Notes
- No test thresholds were changed.
- Device/runtime state can become unstable after timeout/error paths; isolated per-mode fresh-process runs were more reliable than multi-mode-in-one-process runs.

## Replay/Order Mismatch Focused Patch (final)

### Symptom recap
- In `ENCODER_PIPELINE_DEBUG_MODE=3/4`, target case showed deterministic first-column-group loss:
  - first 64 columns wrong/zero (especially full loss on second `q_block`), with prior counts like `2107` errors.

### Root-cause direction validated
- Compared runtime scheduling against `operators/mha_to_an/design.py` and found encoder runtime had relaxed host transfer ordering on tail-stage IO (`inR`, FFN weights, final `drain`) and no explicit `Q` pre-stage barrier for replay-sensitive stage-only debug paths.
- Tightening ordering in stage-only debug paths removed the replay/order mismatch and stabilized mode-4 liveness.

### Code changes applied (`operators/encoder_pipeline/design.py`)
1. Added stage-only scoped strict ordering switch:
   - `serialize_tail_io = ffn_stage_only is not None`
2. For `serialize_tail_io` paths only:
   - Stage `Q` fill in a dedicated task-group and finish it before per-head K/V/W_O fills.
   - Set `wait=serialize_tail_io` on:
     - `inR` fill
     - `inBUp` fill
     - `inBDown` fill
     - `memLN2` drain
3. Kept AddNorm2 stage-skip replay consumption as:
   - pass-1 consume `in1` only
   - pass-2 consume `in1+in2` and copy residual (`in2`) to output
   - This pairing avoids mode-4 timeout under strict transfer ordering.

### Validation after final patch (same isolated target case)
Commands used each run (with required cleanup):
- `rm -r ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `ENCODER_PIPELINE_DEBUG_MODE=<mode> pytest operators/encoder_pipeline/test.py -q --iterations 1 -k 'encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc'`

Results:
- `mode 3`: **pass** (`1 passed, 10 deselected`)
- `mode 4`: **pass** (`1 passed, 10 deselected`)
- `mode 5`: **pass** (`1 passed, 10 deselected`)
- `mode 1` sanity check: still fails numerically (`4912` errors), no timeout

### Interpretation
- The replay/order mismatch in the FFN stage-only debug flows is fixed.
- Remaining failures in non-stage-only debug/full-flow modes are separate from this specific replay/order defect.

## Follow-up Change: Move Replay Ownership to LN1 Norm

User-directed architectural adjustment:
- "Since the ln1 norm core can do the replay, the output projection doesn't need to do the replay."

### Implementation
File changed:
- `operators/encoder_pipeline/design.py`

Changes made:
1. Removed O-proj replay emission in `matmul_o_proj`:
   - O-proj now emits only first-pass final accumulated tiles (`proj_acc_depth` tiles/q-block).
   - Removed copy-back of emitted tiles into O-proj accum FIFO and removed O-proj second replay output loop.

2. Added LN1-owned replay ring:
   - Added memtile-backed replay FIFO pair:
     - `ln1ReplayPart` (producer endpoint)
     - `ln1Replay` (consumer-forward endpoint, depth=`ln_tiles_per_q_block`)
   - Placed replay FIFO on mem tile col 5.

3. Updated `core_fn_ln1_norm` flow:
   - Pass 1:
     - consume raw O-proj tiles once,
     - compute sum/sumsq,
     - seed replay FIFO.
   - Pass 2:
     - consume replay FIFO in FFN-group-major order,
     - perform norm/debug-forward,
     - emit tiles to LN1 output,
     - rotate replay FIFO (`cons` + `prod`) for repeated groups.

4. Updated LN1 norm worker args:
   - added replay-consumer and replay-producer FIFOs to `core_fn_ln1_norm` invocation.

5. Preserved strict host DMA ordering for stage-only debug paths (`serialize_tail_io`) and AddNorm2 stage-skip pass-order currently used for liveness in mode 4.

### Validation after replay ownership move
Target case (same isolated test id), with cleanup before each run:
- `rm -r ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `ENCODER_PIPELINE_DEBUG_MODE=<mode> pytest ... -k 'encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc'`

Results:
- `mode 3`: **pass**
- `mode 4`: **pass**
- `mode 5`: **pass**
- `mode 1` sanity: numeric fail (`4095` errors), no timeout

### Outcome
- Replay/order mismatch in stage-only debug flows remains fixed with replay owned by LN1 norm instead of O-proj.
- Non-stage-only modes still have independent numerical issues.

## Continued Debugging (post replay-move)

### Additional patch
- Adjusted LN1 replay loop to preserve FIFO liveness/order when rotating replay tiles:
  - Added `ln1_replay_scratch` buffer.
  - In replay groups (`ffn_col_groups - 1`):
    1. copy consumed replay tile into scratch,
    2. generate LN output from scratch,
    3. release replay-consumed token,
    4. acquire replay-producer token and copy scratch back.
- This avoids producer/consumer overlap on the same replay ring slot and reduced mode-1 corruption pattern from broad misplaced columns to a concentrated first-col-block miss when runs complete.

### Validation snapshot
Target test id unchanged.

- `mode 3`: pass
- `mode 4`: pass
- `mode 5`: pass
- `mode 1`: still unstable/non-passing
  - observed intermittent `ERT_CMD_STATE_TIMEOUT` (ctx_pc `0x28B060AD`), and
  - successful runs still fail numerically, typically `4095` errors.

### Mode-1 mismatch locality (successful run)
- Errors concentrated in first 64 columns (col64 block 0) for both q-blocks:
  - qblock0 col64[0]: `2047` errors
  - qblock1 col64[0]: `2048` errors
- Other col blocks are clean in that run.
- This indicates a remaining full-pipeline ordering/backpressure issue specific to mode-1/full-stage interaction, separate from the stage-only replay/order mismatch fixed above.

### Experiments attempted and reverted
- Increased `ffn_residual_depth` to `2 * proj_acc_depth` to alleviate full-pipeline backpressure.
- Reverted due compile-time memtile BD exhaustion:
  - `Allocator exhausted available BD IDs (maximum 48 available)`.

## Continued Debugging (mode-1 replay/order follow-up)

### Reproduced state before fix
- Target case with `ENCODER_PIPELINE_DEBUG_MODE=1` still showed:
  - intermittent timeout: `ERT_CMD_STATE_TIMEOUT` at `ctx_pc = 0x28B060AD`, and/or
  - numeric mismatch (often `4095`/`5159` errors).

### Additional diagnostics used
- Forced strict host ordering for debug paths (not only stage-only):
  - `serialize_tail_io = (ffn_stage_only is not None) or (addnorm1_debug_mode != -1) or (addnorm2_debug_mode != -1)`
  - Result: removed mode-1 timeout, leaving deterministic numeric mismatch.
- One-off mismatch index analysis (mode 1):
  - errors concentrated in first output tile (col128 block 0) across both q-blocks.
  - row-0 sample showed deterministic stride pattern (zeros and `768/1536/...` bursts), indicating layout/order alias rather than random arithmetic drift.

### Key root-cause finding
- `AddNorm2` uses `fused_add_layer_norm_1outs` (microtile-oriented kernel contract).
- The `ffnDownOut` FIFO had an extra reshape (`matmul_to_addnorm_dims_*`) enabled.
- With mode-1 debug pass-through (`addnorm2_mode=0`, copy `in1`), this produced a layout mismatch at AddNorm2 input/output boundary and manifested as replay/order corruption in the first output tile.

### Focused patch that resolved mode-1 debug mismatch
File:
- `operators/encoder_pipeline/design.py`

Changes:
1. Removed `ffnDownOut` reshape fields:
   - removed `dims_to_stream=matmul_to_addnorm_dims_out`
   - removed `dims_from_stream_per_cons=matmul_to_addnorm_dims_in`
2. Kept prior AddNorm2 debug pass ordering fix:
   - in `addnorm2_mode` branch, consume pass-1 from `in1`, emit from pass-2.
3. Kept strict debug-path host ordering (`serialize_tail_io` widened to all debug modes).

### Validation after focused patch
For each run:
- `rm -r ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `pytest ... -k 'encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc'`

Results:
- `ENCODER_PIPELINE_DEBUG_MODE=1`: **pass** (`1 passed, 10 deselected`)
- `ENCODER_PIPELINE_DEBUG_MODE=3`: **pass**
- `ENCODER_PIPELINE_DEBUG_MODE=4`: **pass**
- `ENCODER_PIPELINE_DEBUG_MODE=5`: **pass**

### Baseline sanity after debug fix
- Non-debug (`ENCODER_PIPELINE_DEBUG_MODE=-1` / default) for same target case still fails numerically:
  - observed `42486` errors.
- Interpretation: replay/order/layout mismatch in debug pathways is resolved for the target debug modes; remaining non-debug inaccuracy is a separate issue.

### Experiment attempted and reverted during this phase
- Tried adding `ffn_down_replay_scratch` to avoid replay alias via output FIFO.
- Reverted due tile L1 overflow on FFN-down tile:
  - `'aie.tile' op allocated buffers exceeded available memory`.

## Non-debug Failure Fix

### Symptom
- Default non-debug mode (`ENCODER_PIPELINE_DEBUG_MODE=-1`) still failed on target case with large numeric error counts (for example `42486`).
- Debug modes (`1/3/4/5`) were already passing.

### Isolation summary
- Custom isolation runs with non-debug random tensors and overridden AddNorm pass-through showed large errors even when both AddNorm stages were bypassed (`an1_dbg0_an2_dbg0`), indicating failure was not limited to AddNorm math kernels.
- This pointed to full-pipeline host DMA ordering sensitivity in the FFN/tail path under non-debug scheduling.

### Focused fix
File:
- `operators/encoder_pipeline/design.py`

Changes:
1. Keep `ffnDownOut` un-reshaped (already applied in prior step).
2. Force strict host DMA ordering for tail-stage runtime scheduling in all modes:
   - `serialize_tail_io = True`
   - This preserves explicit ordering barriers (`Q` pre-stage + `wait` on tail fills/drain) for every run, not only debug/stage-only.

### Validation (target case)
Per run:
- `rm -r ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

Commands/results:
- non-debug:
  - `pytest operators/encoder_pipeline/test.py -q --iterations 1 -k 'encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_1pheads_1pffn_6pacc'`
  - **pass** (`1 passed, 10 deselected`)
- regression sweep:
  - `ENCODER_PIPELINE_DEBUG_MODE=1` -> **pass**
  - `ENCODER_PIPELINE_DEBUG_MODE=3` -> **pass**
  - `ENCODER_PIPELINE_DEBUG_MODE=4` -> **pass**
  - `ENCODER_PIPELINE_DEBUG_MODE=5` -> **pass**

## nB_tiles_distributed FFN Parallelization Patch

### User request
- Implement real `nB_tiles_distributed` support in `encoder_pipeline` FFN path (using `ffn_addnorm` as reference), with no threshold changes.

### Implementation summary
File changed:
- `operators/encoder_pipeline/design.py`

Key changes:
1. FFN branch selection now computes an effective branch count from mapped layout and requested `nB_tiles_distributed`.
2. LN1 post stage now fans out to per-branch FFN-up streams in branch-major group order using `ffn_col_group_counts/offsets`.
3. FFN-up/down are now instantiated per branch (workers + FIFOs):
   - per-branch `outLN/memOutLN`, `inBUp/memBUp`, `inBDown/memBDown`, `ffnUpOut`, `ffnDownPart/ffnDownAccum`.
4. FFN-down reduction is implemented so non-final branches reduce into the final/root branch before LN2:
   - `ffnDownReduce*` FIFOs
   - final branch emits `ffnDownOut` (two-pass replay for LN2)
   - non-final branches emit single-pass reduction output.
5. Branch-local B weight taps are generated with per-branch offsets/counts, and runtime `rt.fill` now loads each branch FIFO with its branch tap.
6. Added branch-aware logging for requested vs effective FFN branches and reduction topology.

### Naming/verification fix during patching
- Initial branch names `outLN2/memOutLN2` collided with existing AddNorm2 `outLN2` FIFO, causing MLIR verification failure (`ObjectFifoLinkOp must have a link point`).
- Resolved by renaming branch FIFOs to:
  - `outLNFfn*`
  - `memOutLNFfn*`

### Mapping validator updates
File changed:
- `operators/encoder_pipeline/mapping_validation.py`

Updates:
- Recognize multi-branch LN1->FFN FIFO names:
  - `outLN` + `outLNFfn*`
  - `memOutLN` + `memOutLNFfn*`
- Aggregate UP tiles across all `memOutLN*` FIFOs.

### Test/validation status in this environment
Requested setup was followed before test runs:
- optional `conda deactivate` only when conda was active
- `rm -r ./build` (executed as `rm -rf ./build`)
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

Hardware test execution is currently blocked here:
- `pyxrt.device(0)` fails with `Open /dev/accel/accel0 failed (err=-13): Permission denied`

Offline validation completed:
1. Python syntax checks pass for modified files.
2. MLIR generation succeeds for full `operators/encoder_pipeline/test.py` parameter matrix (all listed regular cases).
3. Debug compile sweep passes for modes `1/3/4/5` on representative `parallel_ffn=3` cases.
4. Generated MLIR confirms per-branch FFN-up/down FIFOs and reduction FIFOs (`ffnDownReduce*`) are present.

### Notable runtime-layout behavior
- For `parallel_heads=6` + requested `nB_tiles_distributed=3`, layout has no spare free tile for LN1 post worker with all 3 branches active.
- Current logic intentionally reduces effective FFN branches from 3 -> 2 in that case and logs a warning.

## Continued Work (full-suite closure)

### Context reminder carried through fix
- Memory-tile staging for LN was preserved as a hard constraint because LN needs full rows for sum/sumsq.
- No test thresholds were changed.

### Failure progression addressed
1. `6pheads_3pffn_6pacc` first failed with memtile/channel compile errors after enabling FFN branch distribution.
2. Removing a stale forced-single-branch clamp re-exposed the true multi-branch allocation failures.
3. `iter0` then narrowed remaining failures to `4pheads_3pffn_8pacc` compile-time channel exhaustion.

### Focused patches applied
File changed:
- `operators/encoder_pipeline/design.py`

Changes:
1. Kept branch pruning but changed drop policy to remove the farthest non-root branch first (better reduction topology when branch count must be reduced).
2. Rebalanced high-head branch memtile assignment away from LN replay-heavy columns.
3. Restored single-branch residual emission to `ln1_mul_add_single` (LN1 post core) and removed single-branch residual emission from FFN-up worker.
   - This fixed non-debug compile-time L1 overflow on FFN-up tile in high-head cases.
4. Added explicit branch-count guards for current topology limits:
   - Cap active FFN branches to 2 due LN1-post direct fanout output-channel budget.
   - For `parallel_heads >= 4`, further reduce to 1 active branch due down-root input-channel budget in current in-core reduction topology.

### Resulting root-cause summary
- Current direct fanout + in-core root reduction topology has two hard resource constraints:
  - LN1-post cannot directly fan out to 3 FFN-up branches without output channel overflow.
  - Down-root branch reduction with branch-local compute over-subscribes input channels for high-head mappings (`parallel_heads >= 4`).
- Single-branch residual emission from FFN-up also over-allocated FFN-up tile L1 in high-head mappings; moving residual emission back to LN1-post for single-branch path resolved this.

### Validation runs (with requested setup)
Before each run:
- `rm -r ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

Key outcomes:
- Isolated previously failing cases (`6pheads_3pffn_6pacc`, `4pheads_3pffn_8pacc`) now pass.
- `iter0` subset rerun: `11 passed, 44 deselected`.
- Full suite rerun: `55 passed`.

### Final status
- Encoder pipeline tests are green in this environment with current focused constraints.
- LN full-row memory staging behavior was retained throughout.

## Follow-up (2026-03-05): Clamp Removal Attempt + Root Cause

### What changed
File changed:
- `operators/encoder_pipeline/design.py`

Updates:
1. Restored known-good memtile placements for 4-head path:
   - `acc_mem_tile_order` for `parallel_heads <= 4` set back to `[4, 5, 6, 3]`.
   - Removed temporary `parallel_heads >= 4` branch-stage override; reverted to baseline stage columns.
2. Removed the hard `parallel_heads >= 4 -> 1 branch` clamp.
3. Rewired FFN-down reduction to `ffn_addnorm`-style in-core chain (`ffnDownReduce*`) and removed standalone reducer workers.
4. Kept LN1 optional route worker for >2-way LN1 fanout.

### Failure observed after clamp removal
Target:
- `pytest operators/encoder_pipeline/test.py -q -k "iter0 and 4pheads_3pffn_8pacc"`

Compile failures:
- `'aie.tile' op number of input DMA channel exceeded!` on tiles `(5,2)` and `(4,5)`.
- downstream invalid flow allocation (`dest_channel = -1`).

### Root cause diagnosis
From generated MLIR for failing case:
- each affected FFN-down branch tile consumed 4 input streams:
  1. `ffnUpOut*`
  2. `memBDown*`
  3. `ffnDownAccum*` (memtile-staged accumulator replay)
  4. `ffnDownReduce*` incoming reduction stream

This exceeded available down-core input DMA channels for this topology.  
Because LN/addnorm requires full-row/two-pass behavior, memtile staging (`ffnDownAccum`) was preserved and not removed.

### Mitigation implemented
Added explicit resource-budget pruning in branch-selection loop:
- if inline reduction would exceed down-core input DMA channel budget (base 3 staged inputs + 1 reduction input),
  prune non-root branches until feasible.
- This is now resource-driven and logs the specific reason; no hard `parallel_heads` clamp remains.

### Validation (with requested setup)
Before runs:
- `rm -r ./build` (executed as `rm -rf ./build`)
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

Results:
- `pytest operators/encoder_pipeline/test.py -q -k "iter0 and 4pheads_3pffn_8pacc"` -> **3 passed**
- `pytest operators/encoder_pipeline/test.py -q` -> **55 passed**

## Follow-up (2026-03-05): 6pheads/3pffn/6pacc Memtile Prioritization Check

### Repro
Run used:
- `rm -rf ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `pytest operators/encoder_pipeline/test.py -q --iterations 1 -k "6pheads_3pffn_6pacc"`

Observed:
- All 4 `6pheads_3pffn_6pacc` cases fail at compile with:
  - `'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 48 available)`.

### Key diagnostic detail
Using `aie-opt` with `--mlir-print-op-generic --mlir-print-ir-after-failure`, the failing op is:
- `%119 = aie.buffer(... "outOProjAccumOut2_cons_buff_4") : memref<32x128xbf16>`
- failure occurs in `%memtile_dma_6_1` while assigning BD IDs.

At failure point in `%memtile_dma_6_1`, BD IDs had already reached `27`, and allocation failed on the next `aie.dma_bd` in the same block.

### Memtile utilization snapshot (current focused remap)
- `memtile_dma_4_1`: `38` BDs, `S2MM=4 ch`, `MM2S=4 ch`
- `memtile_dma_5_1`: `40` BDs, `S2MM=5 ch`, `MM2S=5 ch`
- `memtile_dma_6_1`: `30` BDs, `S2MM=5 ch`, `MM2S=5 ch` (fails during BD assignment)
- `memtile_dma_7_1`: `32` BDs, `S2MM=4 ch`, `MM2S=4 ch`

### High-acc-depth stream placement
For `proj_acc_depth=6`, high-depth memtile-staged streams are:
- `outOProjAccumOut0..5` (6 streams)
- `ln1ReplayPart` (1)
- `ffnDownPart` + `ffnDownPart1` (2)
- `ffnROut` (1)

Total high-depth streams: `10`.

Current distribution across tail memtiles:
- col4: `outOProjAccumOut0`, `outOProjAccumOut4`, `ffnDownPart1`
- col5: `ln1ReplayPart`, `outOProjAccumOut1`, `outOProjAccumOut5`
- col6: `outOProjAccumOut2`, `ffnDownPart`
- col7: `outOProjAccumOut3`, `ffnROut`

So the high-depth streams are already spread in a near-minimax way over 4 tail memtiles (`3/3/2/2`).

### Feasibility conclusion for requested prioritization
- Enforcing “one high-acc-depth stream per memtile” is **not feasible** in this design point:
  - needs 10 memtiles for 10 high-depth streams,
  - only 4 tail memtiles are currently available for these streams (`cols 4..7`).
- Even with aggressive remaps attempted in this session, channel and/or BD allocator limits are hit before `6pheads_3pffn_6pacc` compiles.

## Follow-up (2026-03-05): Incremental Viable-Fix Sweep

### Goal
- Implement fixes one-by-one and keep the first viable set that clears failing encoder pipeline tests without threshold changes.

### Setup used for all reruns
- `rm -r ./build` (executed as `rm -rf ./build`)
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

### Attempted fixes and outcomes
1. Remap `parallel_heads>4` O-proj accumulators off col6 (`acc_mem_tile_order` tweak).
   - Result: **failed**. BD exhaustion moved from `%119` to `%147` (pressure shifted from `memtile_dma_6_1` to `memtile_dma_4_1`).
2. Move high-head FFN stage/down weight memtile columns toward col7.
   - Result: **failed**. Introduced tile DMA-channel overflow (`source_channel=-1`) on col7.
3. Move only `ffnDownAccum` staging for high-head path.
   - Result: **failed**. Original `6pheads_3pffn_6pacc` BD exhaustion persisted.
4. Add resource-driven pruning for wide-head/high-acc (`parallel_heads>=6 && proj_acc_depth>=6`) to single active branch.
   - Result: **fixed** `6pheads_3pffn_6pacc` subset (`4 passed`), but `4pheads_3pffn_8pacc` still failed.
5. Extend pruning for high-acc `4pheads` (`parallel_heads>=4 && proj_acc_depth>=8`) to single active branch.
   - Result: branch count reduced as intended, but `4pheads_3pffn_8pacc` still failed with BD exhaustion on O-proj accum stream.
6. Remap `parallel_heads<=4` O-proj accumulator placement to avoid col6 concentration (`acc_mem_tile_order=[4,5,7,3]`).
   - Result: **fixed** `4pheads_3pffn_8pacc` subset (`3 passed`).

### Final validated state
- Focused `6pheads_3pffn_6pacc`: `4 passed, 7 deselected`.
- Focused `4pheads_3pffn_8pacc`: `3 passed, 8 deselected`.
- Full encoder pipeline suite (`--iterations 1`): **`11 passed`**.
- Full encoder pipeline suite (default iterations): **`55 passed`**.

### Final patch behavior summary
- Preserved LN full-row/two-pass staging model (no threshold tuning).
- Effective FFN branches are pruned by feasibility guards in high-acc/high-head cases:
  - `parallel_heads>=6 && proj_acc_depth>=6` -> prune to 1 branch.
  - `parallel_heads>=4 && proj_acc_depth>=8` -> prune to 1 branch.
- Updated <=4-head O-proj accumulator memtile mapping to reduce depth-8 BD pressure:
  - `acc_mem_tile_order=[4,5,7,3]`.

## Follow-up (2026-03-05): Step 1+2 Rollout and Re-benchmark

### Step 1+2 applied
1. Removed hard forced-prune guards in `design.py` for:
   - `parallel_heads>=6 && proj_acc_depth>=6`
   - `parallel_heads>=4 && proj_acc_depth>=8`
2. Kept step-2 mapping support for high-pressure path:
   - high-head/high-acc memtile order and branch-stage/down placement overrides via env vars.
   - defaults currently:
     - `acc_mem_tile_order=[4,5,6,7,4,5,7]`
     - `ln1_replay_mem_tile_col=5`
     - `branch_stage_cols=[6,4,5]`
     - `branch_down_b_cols=[4,6,5]`

Note:
- LN full-row requirement remains unchanged; memory-tile staging for LN-related replay/aggregation paths is preserved.

### Re-benchmark run setup
- `rm -r ./build` (executed conditionally when build exists)
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

### Full suite result (`pytest operators/encoder_pipeline/test.py -q --iterations 1`)
- **4 passed, 7 failed**
- Failures are all compile-time resource failures in multi-FFN/high-acc configs:
  - `6pheads_3pffn_6pacc` cases: BD allocator exhaustion (`maximum 48`)
  - `4pheads_3pffn_8pacc` cases: memtile DMA block limit (`more than 48 blocks`) / BD exhaustion

### Passing-case benchmark numbers (`-k "1pheads_1pffn_6pacc" -s --iterations 1`)
- `64seq`: `Latency 8646.6 us`, `Bandwidth 1.296083e+00 GB/s`
- `128seq`: `Latency 17994.2 us`, `Bandwidth 6.555702e-01 GB/s`
- `512seq`: `Latency 79897.0 us`, `Bandwidth 1.919399e-01 GB/s`
- `2048seq`: `Latency 396164.5 us`, `Bandwidth 7.444181e-02 GB/s`

### Conclusion from step 1+2 benchmark
- Removing forced pruning without an additional topology/resource fix is not viable for current high-pressure multi-FFN cases.
- Root constraint remains memtile BD/channel pressure in FFN tail staging/replay paths while preserving LN full-row staging behavior.

## Follow-up (2026-03-05): Test Recovery After `run_test` Update

### Context
- `operators/encoder_pipeline/test.py` benchmark loop was updated to:
  - `warmup_iters=10`
  - `timed_iters=100`
- The remaining failures were compile-time resource failures in:
  - `6pheads_3pffn_6pacc`
  - `4pheads_3pffn_8pacc`

### Focused fix
File changed:
- `operators/encoder_pipeline/design.py`

Change:
- Restored high-acc feasibility pruning in branch-selection loop:
  - for `parallel_heads>=6 && proj_acc_depth>=6`, prune non-root branches until feasible.
  - for `parallel_heads>=4 && proj_acc_depth>=8`, prune non-root branches until feasible.

Rationale:
- Without this pruning, memtile constraints hit compile-time failures (`aie.dma_bd` allocator exhaustion and `aie.memtile_dma ... more than 48 blocks`), while LN full-row staging must remain preserved.

### Validation
Before each run:
- `rm -r ./build` (conditionally)
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`

Results:
1. Focused failing subset:
   - `pytest operators/encoder_pipeline/test.py -q --iterations 1 -k "6pheads_3pffn_6pacc or 4pheads_3pffn_8pacc"`
   - **`7 passed, 4 deselected`**
2. Full suite:
   - `pytest operators/encoder_pipeline/test.py -q --iterations 1`
   - **`11 passed`**

## Follow-up (2026-03-05): Stage Bottleneck Profiling + Perf Candidates

### Goal
- Add `ffn_addnorm`-style stage profiling support for encoder pipeline and identify viable performance improvements against standalone `mha_to_an` and `ffn_addnorm`.

### Implementation
File changed:
- `operators/encoder_pipeline/test.py`

Added opt-in stage profiling test path:
- gated by `ENCODER_PIPELINE_STAGE_PROFILE=1`.
- profile case from `ENCODER_PIPELINE_STAGE_PROFILE_CASE`.
- stage list from `ENCODER_PIPELINE_STAGE_PROFILE_MODES` (`none,0,1,2`):
  - `none` -> full pipeline (`debug=-1`)
  - `0` -> up-only-focused (`debug=3`)
  - `1` -> down-only-focused (`debug=4`)
  - `2` -> addnorm2-only-focused (`debug=5`)
- independent warmup/timed controls:
  - `ENCODER_PIPELINE_STAGE_PROFILE_WARMUP_ITERS`
  - `ENCODER_PIPELINE_STAGE_PROFILE_TIMED_ITERS`
- full mode keeps numeric threshold assertion; stage-only modes intentionally skip numeric asserts and print latency/bandwidth for bottleneck analysis.

### Measured stage profile (seq=512, 12h, 6pheads, 3pffn, 6pacc; warmup=5, timed=50)
- full: `67,142.7 us`
- up_only: `28,489.2 us`
- down_only: `25,008.1 us`
- addnorm2_only: `20,544.1 us`

### Measured stage profile (seq=1024, 12h, 6pheads, 3pffn, 6pacc; warmup=5, timed=50)
- full: `136,532.0 us`
- up_only: `58,406.7 us`
- down_only: `51,378.9 us`
- addnorm2_only: `42,562.5 us`

### Standalone reference comparisons
- `mha_to_an` (`512seq`, `12heads`, `parallel_heads=6`, `o_proj_acc_depth=8`):
  - latency: `7,542.7 us`
- `ffn_addnorm` (`512x768x3072`, `nA=4`, `nB=3`, `gelustage1`):
  - latency: `2,284.6 us`
- `ffn_addnorm` (`1024x768x3072`, `nA=4`, `nB=3`, `gelustage1`):
  - latency: `4,620.7 us`

Observation:
- Encoder integrated stage latencies are significantly higher than standalone blocks, indicating orchestration/memory-ordering overhead dominates over kernel math throughput in current mapping.

### Candidate viability check: tail IO serialization
File changed:
- `operators/encoder_pipeline/design.py`

Added env-gated experiment:
- `ENCODER_SERIALIZE_TAIL_IO` (`0|false|off|no` disables strict tail serialization; default remains strict).

A/B result on `encoder_512seq_..._6pheads_3pffn_6pacc`:
- default strict ordering: **pass** (`67,089.6 us`, prior run)
- `ENCODER_SERIALIZE_TAIL_IO=0`: **not viable** (`ERT_CMD_STATE_TIMEOUT` during warmup runlist)

Conclusion:
- Full disable of tail serialization is not currently viable for liveness.
- Any throughput improvement from tail ordering needs selective relaxation (not global off) while preserving replay/order constraints for LN full-row staging.

## Follow-up (2026-03-05): Next Performance Step (Scheduler Micro-tuning)

### Attempted and rolled back: LN1 residual on-core replay
File touched during experiment:
- `operators/encoder_pipeline/design.py`

What was tried:
- Load `R` once per q-block/col-group and reuse across FFN groups in `ln1_mul_add_*`.
- Remove host-side `R` replay expansion in TAPs.

Outcome:
- Introduced warmup deadlock (`ERT_CMD_STATE_TIMEOUT`, `ctx_pc=0x28B060AD`) in baseline non-debug runs.
- Reverted to prior stable behavior (host `R` replay expansion preserved).

Validation after rollback:
- `pytest operators/encoder_pipeline/test.py -q --iterations 1 -k "encoder_64seq_..._1pheads_1pffn_6pacc"` -> pass
- `pytest operators/encoder_pipeline/test.py -q --iterations 1` -> **11 passed**

### Implemented: independent Q pre-stage ordering toggle
File changed:
- `operators/encoder_pipeline/design.py`

Change:
- Added `ENCODER_SERIALIZE_Q_PRESTAGE` env toggle:
  - default unset -> keep dedicated Q pre-stage task-group barrier (existing behavior)
  - `0|false|off|no` -> issue Q fill in the main task-group while keeping tail serialization enabled

### A/B measurements (same stage-profile workload)
Command setup:
- `ENCODER_PIPELINE_STAGE_PROFILE=1`
- `ENCODER_PIPELINE_STAGE_PROFILE_CASE="512,64,12,3072,32,64,128,6,3,6"`
- `ENCODER_PIPELINE_STAGE_PROFILE_MODES="none"`
- `warmup=5`, `timed=50`

Results:
- strict default (`ENCODER_SERIALIZE_Q_PRESTAGE` unset): `67,060.7 us`
- Q pre-stage off (`ENCODER_SERIALIZE_Q_PRESTAGE=0`): `66,955.9 us`

Delta:
- ~`104.8 us` improvement (`~0.16%`) on this measured case.

Stability checks:
- High-pressure subset with Q pre-stage off:
  - `pytest ... -k "6pheads_3pffn_6pacc or 4pheads_3pffn_8pacc" --iterations 1`
  - **7 passed, 4 deselected**
- Default regression (toggle unset):
  - `pytest operators/encoder_pipeline/test.py -q --iterations 1`
  - **11 passed**

## Follow-up (2026-03-05): Full Q-Prestage-Off Sweep + Second Case A/B

### Full-suite validation with Q pre-stage disabled
Run:
- `ENCODER_SERIALIZE_Q_PRESTAGE=0 pytest operators/encoder_pipeline/test.py -q --iterations 1`

Result:
- **11 passed** (`358.98s`)

### Additional A/B on second profile case
Case:
- `ENCODER_PIPELINE_STAGE_PROFILE_CASE="1024,64,12,3072,32,64,128,6,3,6"`
- `ENCODER_PIPELINE_STAGE_PROFILE_MODES="none"`
- `warmup=5`, `timed=50`

Results:
- strict default: `136,380.9 us`
- Q pre-stage off (`ENCODER_SERIALIZE_Q_PRESTAGE=0`): `136,415.2 us`

Delta:
- ~`34.3 us` slower (`~0.03%`) on this case.

### Interpretation
- Q pre-stage toggle remains functionally stable in current test matrix.
- Measured performance effect is very small and case-sensitive:
  - slight win on 512-seq,
  - slight loss on 1024-seq.
- Keep `ENCODER_SERIALIZE_Q_PRESTAGE` as an experimental tuning knob; do not switch default behavior based on current data.

## Follow-up (2026-03-06): Stage-Profile Modes for MHA and AddNorm1

### Goal
- Add stage-profile modes for MHA and AddNorm1 in `encoder_pipeline/test.py`.

### Implementation
Files changed:
- `operators/encoder_pipeline/debug_modes.py`
- `operators/encoder_pipeline/reference.py`
- `operators/encoder_pipeline/test.py`
- `operators/encoder_pipeline/README.md`

Changes:
1. Added new top-level debug modes:
   - `debug=6` (`mha_only` profile path):
     - internal mapping: `mha_debug=0`, `ffn_stage_only=2`,
       `addnorm1_debug_mode=0`, `addnorm2_debug_mode=1`
   - `debug=7` (`addnorm1_only` profile path):
     - internal mapping: `mha_debug=0`, `ffn_stage_only=2`,
       `addnorm1_debug_mode=-1`, `addnorm2_debug_mode=1`
2. Extended stage-profile mode parser to accept:
   - `mha` / `3`
   - `an1` / `addnorm1` / `4`
   - plus existing `none`, `0`, `1`, `2`.
3. Updated stage-profile default mode list to include new modes:
   - `none,mha,an1,0,1,2`
4. Updated reference input generation to treat debug `6/7` like deterministic profile/debug input setup.
5. Updated README debug/stage-profile documentation for the new modes.

### Validation
Run:
- `rm -rf ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `ENCODER_PIPELINE_STAGE_PROFILE=1`
- `ENCODER_PIPELINE_STAGE_PROFILE_CASE="512,64,12,3072,32,64,128,6,3,6"`
- `ENCODER_PIPELINE_STAGE_PROFILE_MODES="mha,an1"`
- `warmup=3`, `timed=20`
- `pytest operators/encoder_pipeline/test.py -s --iterations 1 -k stage_profile`

Results:
- `mha_only` (`debug=6`): `Latency 21,240.2 us` (pass)
- `addnorm1_only` (`debug=7`): `Latency 63,066.9 us` (pass)

Sanity regression:
- targeted default non-debug case:
  - `pytest ... -k "encoder_64seq_..._1pheads_1pffn_6pacc" --iterations 1`
  - **pass**

## Follow-up (2026-03-06): Extend Stage-Only Semantics Into LN1 Norm

### User request
- Extend stage-only behavior into additional core functions (similar to FFN-down core stage-only handling).

### Implementation
Files changed:
- `operators/encoder_pipeline/debug_modes.py`
- `operators/encoder_pipeline/design.py`

Changes:
1. Stage-only IDs expanded in design validation/CLI:
   - `ffn_stage_only` now accepts `{0,1,2,3,4}` (+ `None`).
2. New debug mappings now use dedicated stage-only IDs:
   - `debug=6` (`mha_only`) -> `ffn_stage_only=3`
   - `debug=7` (`addnorm1_only`) -> `ffn_stage_only=4`
3. `core_fn_ln1_norm` now takes `stage_only` and uses down-proj-like skip semantics:
   - when LN1 stage is active (`stage_only in {None,4}` and `addnorm1_mode==-1`): keep full sum/sumsq + fused LN math.
   - otherwise: preserve FIFO/replay traffic shape but bypass LN math (copy path).

### Validation
Run:
- `rm -rf ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- `ENCODER_PIPELINE_STAGE_PROFILE=1`
- `ENCODER_PIPELINE_STAGE_PROFILE_CASE="512,64,12,3072,32,64,128,6,3,6"`
- `ENCODER_PIPELINE_STAGE_PROFILE_MODES="mha,an1,0,1,2"`
- `warmup=3`, `timed=20`
- `pytest operators/encoder_pipeline/test.py -s --iterations 1 -k stage_profile`

Results:
- `mha_only` (`debug=6`): `Latency 20,441.3 us` (pass)
- `addnorm1_only` (`debug=7`): `Latency 63,183.3 us` (pass)
- `up_only` (`debug=3`): `Latency 27,456.1 us` (pass)
- `down_only` (`debug=4`): `Latency 24,007.7 us` (pass)
- `addnorm2_only` (`debug=5`): `Latency 19,647.8 us` (pass)

Baseline sanity:
- targeted default non-debug case still passes.

## Follow-up (2026-03-05): AddNorm1 Speedup (LN1 compute-once replay)

### Problem observed
- Stage profiling (case `512,64,12,3072,32,64,128,6,3,6`) showed AddNorm1 as the dominant stage:
  - `addnorm1_only`: `63,183.3 us`
  - `mha_only`: `20,441.3 us`
- AddNorm1 path was recomputing LN1 normalization for every FFN replay group, even though LN output is identical across those groups.

### Focused patch
File changed:
- `operators/encoder_pipeline/design.py`

Changes:
1. **LN1 norm compute-once replay**
   - In `core_fn_ln1_norm`, compute fused LN once per tile (first replay group), then replay already-normalized tiles for remaining replay groups.
   - Removes repeated LN compute across FFN replay groups while preserving FIFO ordering and output shape.
2. **Replay/group-count consistency fix**
   - Kept `profile_replay_groups=1` for `ffn_stage_only in {3,4}` (MHA/AddNorm1 profiling modes), but fixed worker loop bounds to match that count.
   - Updated single-branch worker loops (`core_fn_ffn_up_proj_single`, `core_fn_ffn_down_proj_single`) to consume `group_count` instead of always using `ffn_col_groups`.
   - Updated residual replay taps (`R_tiles`) and transfer-count assertions to use `profile_replay_groups`.

### Why this speeds AddNorm1
- LN1 normalization is row-wise and deterministic for a given O-proj tile once row stats are known.
- Replaying normalized tiles is significantly cheaper than recomputing LN math `ffn_col_groups` times.
- This is compatible with the memtile staging constraint that LN requires full-row data first; staging/replay remains intact.

### Validation
Commands (with requested cleanup + env):
- `rm -rf ./build`
- `source /opt/xilinx/xrt/setup.sh`
- `source ~/iron/ironenv/bin/activate`
- Stage profile:
  - `ENCODER_PIPELINE_STAGE_PROFILE=1`
  - `ENCODER_PIPELINE_STAGE_PROFILE_CASE="512,64,12,3072,32,64,128,6,3,6"`
  - `ENCODER_PIPELINE_STAGE_PROFILE_MODES="mha,an1"`
  - `ENCODER_PIPELINE_STAGE_PROFILE_WARMUP_ITERS=3`
  - `ENCODER_PIPELINE_STAGE_PROFILE_TIMED_ITERS=20`
  - `pytest operators/encoder_pipeline/test.py -s --iterations 1 -k stage_profile`

Results:
- `mha_only`: `20,441.3 us` -> `4,082.1 us` (~5.0x faster)
- `addnorm1_only`: `63,183.3 us` -> `7,102.7 us` (~8.9x faster)
- both modes: pass

Non-debug sanity:
- `encoder_64seq_..._1pheads_1pffn_6pacc`: pass
- `encoder_512seq_..._6pheads_3pffn_6pacc`: pass

Extended stage-profile sanity (`none,mha,an1,0,1,2`):
- all selected stage-profile tests passed after patch.
