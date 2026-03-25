# encoder_pipeline Performance Improvement Plan

## Goal

Improve `encoder_pipeline` latency with a depth-first workflow that avoids
repeating dead ends and keeps the benchmarkable path stable.

This document covers two related but distinct targets:

1. the current benchmarkable `encoder_pipeline` path
2. the staged `hidden_states -> QKV -> MHA` branch that is intended to become
   the long-term clean integration path for Q/K/V projection on the NPU

The staged branch is not yet competitive with the current benchmark winners, so
it should be treated as a bring-up and architecture branch first, and only later
as a benchmark branch.

## Current State

### Stable mainline facts

- The benchmarkable `encoder_pipeline` operator is still the fused
  `MHA + AddNorm1 + FFN + AddNorm2` path.
- The current unattended benchmark winners are still the sequence-parallel
  topologies:
  - `seq32_kv64__ps2_ph2_pffn2` at `seq64`
  - `seq32_kv64__ps4_ph1_pffn1` at `seq128+`
- The retained local optimization on the mainline path is the single-chunk
  `B_Up` priming change for the single-branch `4ps` family in
  [design.py](/home/cj/iron/iron/operators/encoder_pipeline/design.py).

### Stable staged-QKV facts

- The staged operator now takes semantic input `X` hidden states.
- Q/K/V projection is computed on the NPU inside the fused operator.
- Q/K/V are streamed directly into the MHA portion of the graph.
- The staged path no longer materializes external `QKV`.
- The staged contract now derives the initial residual from `X` internally.
- The staged contract no longer accepts external `r`.
- The staged path now uses a hybrid K/V policy:
  - direct-stream K/V for smaller staged cases
  - OR-backed K/V cache for larger staged cases
- The staged contract now uses:
  - `W_ATTN`
  - `X`
  - `OR`
  - `B_Up`
  - `B_Down`
- The staged contract currently supports only:
  - `parallel_seq = 1`
  - `parallel_heads <= 2`
  - `nB_tiles_distributed = 1`

### Current measured baselines

Whole-encoder measurements with real `bert-base-uncased` weights, using the
non-sequence-parallel staged/projection-compatible slice:

- `seq64`, `packed_host`, `1ps/1ph/1pffn`
  - avg `encoder_pipeline_sec`: `48.34 ms`
  - avg `qkv_projection_sec`: `2.76 ms`
- `seq64`, `staged_npu`, `1ps/1ph/1pffn`
  - current hybrid staged policy:
    - avg `encoder_pipeline_sec`: `59.33 ms`
    - avg `qkv_projection_sec`: `0.0028 ms`
- `seq128`, `staged_npu`, `1ps/1ph/1pffn`
  - current hybrid staged policy:
    - avg `encoder_pipeline_sec`: `114.17 ms`
    - avg `qkv_projection_sec`: `0.0036 ms`
- `seq256`, `staged_npu`, `1ps/1ph/1pffn`
  - current hybrid staged policy:
    - avg `encoder_pipeline_sec`: `254.62 ms`
    - avg `qkv_projection_sec`: `0.0115 ms`
- `seq64`, `staged_npu`, `1ps/2ph/1pffn`
  - direct-stream staged policy:
    - avg `encoder_pipeline_sec`: `55.03 ms`
    - avg `qkv_projection_sec`: `0.0081 ms`
- `seq128`, `staged_npu`, `1ps/2ph/1pffn`
  - direct-stream staged policy:
    - avg `encoder_pipeline_sec`: `107.30 ms`
    - avg `qkv_projection_sec`: `0.0118 ms`

Interpretation:

- The staged branch has already eliminated host-side QKV time.
- The staged branch is now stable through `seq256`.
- The staged branch is still slower overall at `seq64` and `seq128` than the
  `packed_host` control path.
- At `seq256`, the staged branch beats `packed_host` on total time because the
  host-side QKV cost has grown large enough to dominate.
- The staged `2ph` branch is now real and stable at `seq64` and `seq128`.
- The staged `2ph` branch is still slower than `packed_host` on the same
  non-sequence-parallel slice.

## Hard Constraints

These constraints should shape the search space before any more experiments are
run.

### Architectural constraints

- LN1 and LN2 are still full-row two-pass layer-norm stages.
- The current mainline winner already uses essentially all compute tiles for the
  `128+` benchmark path.
- The remaining mainline wins are therefore more likely to come from lower data
  movement and synchronization cost than from more raw parallelism.

### Current staged-contract constraints

- The staged branch must stay within the current fused-operator model. Do not
  move this work into a runlist path.
- The staged branch must keep `X` as the public semantic input.
- The staged branch must not reintroduce external `QKV` materialization.
- The staged branch currently fits only in the existing 5-BO runtime boundary.

### Runtime constraints

- The current lower runtime/kernel boundary does not tolerate the attempted
  6-BO staged design cleanly. That branch was pruned.
- Do not rely on retries for performance work. Timeouts must be treated as real
  failures.
- Only one hardware process should run at a time.

## Design Space That Is Actually Live

### Mainline benchmarkable path

These are still real search dimensions:

- placement family already supported in
  [placements.py](/home/cj/iron/iron/operators/encoder_pipeline/placements.py)
- layout overrides already exposed in
  [op.py](/home/cj/iron/iron/operators/encoder_pipeline/op.py) and
  [topology.py](/home/cj/iron/iron/operators/encoder_pipeline/topology.py):
  - `weight_forward_depth`
  - `o_proj_fifo_depth`
  - `ffn_replay_fifo_depth`
  - `use_fused_replayed_addnorm`
- pruned data-movement toggles already exposed:
  - `use_transport_groups`
  - `use_unified_qr_split`
- structural staging choices inside
  [design.py](/home/cj/iron/iron/operators/encoder_pipeline/design.py)

### Staged-QKV branch

These are still real search dimensions:

- projection worker placement
- staged projection schedule
- direct-stream buffering
- K/V reuse strategy across q-blocks
- how `OR` is used for stage/output scratch
- eventual support expansion to `parallel_heads > 1`
- eventual support expansion to `nB_tiles_distributed > 1`
- eventual support expansion to sequence-parallel variants

## Design Space Already Covered And Pruned

The following directions have already been explored enough to deprioritize or
prune for now.

### Mainline path: covered and rejected

- naive tail `wait=True/False` relaxations
- simple fill-order shuffles for `K/V/W_O`
- local FIFO-depth shaving around FFN-up and FFN-down
- direct `LN1 -> FFN` bypass variants on the current graph
- local LN2 replay-store variations
- shallow O-proj replay variants that only moved the cost around
- small kernel-only LN/AddNorm rewrites that did not hold up in real timings

### Staged-QKV path: covered and pruned

- external `QKV` materialization as the long-term design
- serial combined `K/V` projection worker
- obvious staged FIFO-depth bumps that only increased pressure
- local 6-BO `KV` buffer branch under the current runtime boundary
- direct local replay branch that deadlocked or timed out
- unconditional OR-backed K/V caching for all staged sizes; it regressed `seq64`
- staged `parallel_heads = 4`; current placement budget cannot place the needed
  projection workers cleanly
- staged `nB_tiles_distributed = 2`; current non-sequence-parallel design hits
  shim DMA limits before runtime

## Depth-First Execution Order

The next work should proceed in the following order and should not skip ahead
unless the current branch is clearly exhausted.

### Phase 0: Keep the current benchmark path safe

1. Keep the mainline benchmarkable path working while staged work continues.
2. Do not mix staged-QKV bring-up changes into the benchmark selection path.
3. Re-run the focused staged slices after every staged edit before using
   hardware.

Required checks:

- `py_compile` on touched files
- staged operator slice in
  [test.py](/home/cj/iron/iron/operators/encoder_pipeline/test.py)
- staged BERT slice in
  [test.py](/home/cj/iron/iron/applications/bert/test.py)

### Phase 1: Exhaust the staged `seq128` stability branch

This is the first branch to exhaust because the staged path is not useful until
it clears the current `seq128` timeout boundary.

#### 1.1 Fix `seq128` under the current 5-BO staged direct-stream contract

Target:

- keep `X` as the public input
- keep direct-stream `Q/K/V -> MHA`
- keep external `QKV` removed
- keep the current 5-BO runtime contract

Experiment order:

1. Instrument the staged q-block/head-group schedule inside
   [design.py](/home/cj/iron/iron/operators/encoder_pipeline/design.py)
   enough to localize the first failing fill/worker group.
2. Check whether the failure is tied to:
   - second q-block startup
   - second K/V block
   - weight refill overlap
   - OR stage/output pressure
3. Try schedule-local fixes only:
   - reorder staged `Q/K/V/W_O` fills
   - split or serialize the head task group
   - stage `Q` later relative to `K/V`
   - stage `W_O` later relative to `K/V`
4. Try placement-local fixes only:
   - move staged `K` and `V` source shim columns
   - move staged `W_K` and `W_V` source shim columns
   - keep worker count unchanged
5. Try minimal buffering changes only where the failure localizes:
   - `projKOut`
   - `projVOut`
   - `projQOut`

Stop criteria:

- stop this sub-branch as soon as `seq128` runs cleanly
- if `seq128` still times out after the local schedule/placement/buffering space
  is exhausted, move to Phase 2

Status:

- completed
- the first concrete fix was repeating staged `W_K/W_V` fills across
  `num_kv_seq_blocks`, which matched the worker contract and removed the
  `seq128` timeout

#### 1.2 Validate that the contract cleanup was actually worth keeping

If `seq128` becomes stable, compare:

- `packed_host`
- `staged_npu`

on the same `1ps/1ph/1pffn` slice for:

- `seq64`
- `seq128`

If the contract cleanup did not help stability or did not help the long-term
architecture enough, explicitly document that before moving on.

Status:

- completed
- the staged contract cleanup is worth keeping architecturally:
  - semantic input is now `X`
  - external residual input is gone
  - external `QKV` is gone
- but that contract cleanup alone was not enough for a performance win on the
  short cases

### Phase 2: Remove K/V recomputation across q-blocks

This is the highest-value staged performance branch after `seq128` stability.

#### 2.1 Confirm the exact cost of current K/V recomputation

Measure and document:

- how many times `K` and `V` are reprojected per q-block at `seq64`
- how that scales at `seq128`
- the share of end-to-end staged time attributable to projection-side work

#### 2.2 Pick one clean K/V reuse design

Do not explore all variants in parallel. Pick the cleanest viable design first.

Candidate order:

1. Keep the current public contract and add K/V reuse through existing internal
   scratch/storage if possible.
2. If that is impossible, redesign internal scratch usage around `OR` and stage
   rows before touching the lower runtime boundary.
3. Only if both fail, change the lower runtime/kernel boundary so a sixth BO is
   legal for staged K/V storage.

Status:

- completed for the current single-head staged slice
- the first retained implementation is candidate 2:
  - keep the 5-BO staged contract
  - use `OR` as internal K/V cache for larger staged cases
  - refill K/V from `OR` per q-block
- current measured outcome:
  - `seq64`: worse than direct staged stream, so cache is disabled there
  - `seq128`: better than direct staged stream, so cache remains enabled
  - `seq256`: stable and materially better in total time than `packed_host`
- current scope note:
  - the retained OR-backed K/V cache is only enabled for `parallel_heads = 1`
  - multi-head staged support currently stays on direct-stream K/V

#### 2.3 Exhaust the chosen K/V reuse branch before moving on

For the chosen branch:

1. get a correct `seq64`
2. get a stable `seq128`
3. compare against staged direct-stream baseline at `seq64`
4. only then move to `seq256`

Stop criteria:

- do not move to multi-head staged support until K/V are no longer recomputed
  per q-block

### Phase 3: Expand staged topology support

Only start Phase 3 after Phase 2 is successful.

#### 3.1 Expand to `parallel_heads > 1`

Order:

1. `2ph`
2. `4ph`
3. only then more aggressive variants

Goals:

- projection parallelism without reopening shim-pressure failures
- preserve the clean staged `X` input contract

Status:

- partially completed
- retained:
  - staged `parallel_heads = 2` now compiles and runs at `seq64` and `seq128`
  - the kept design groups staged K/V weights and remaps one staged `X_V`
    source column to stay under the shim budget
- pruned:
  - staged `parallel_heads = 4` currently fails placement on the existing
    non-sequence-parallel layout

#### 3.2 Expand to `nB_tiles_distributed > 1`

Order:

1. enable two FFN branches
2. validate the FFN tail and LN2 residual handoff
3. only then consider higher branch counts

Status:

- attempted and pruned for now
- the smallest staged `1ps/1ph/2pffn` compile hits shim output-DMA limits
- keep staged support restricted to `nB_tiles_distributed = 1` until a larger
  ingress/routing redesign is justified

#### 3.3 Expand to sequence-parallel staged path

This should be last because it is the most structurally invasive.

Preconditions:

- stable `parallel_heads > 1`
- stable staged K/V reuse
- clear evidence that the staged branch is worth extending toward the current
  benchmark winner family

Status:

- not started
- still blocked by the lack of a compelling non-sequence-parallel staged win

### Phase 4: Manual sweep of the exposed mainline operator design space

Do this after the staged branch is either:

- clearly promising and stable, or
- clearly blocked on a larger redesign

This phase should use the now-exposed operator dimensions rather than more blind
local edits.

#### 4.1 Sweep layout overrides on the current winning families

Families to sweep first:

- `seq32_kv64__ps2_ph2_pffn2`
- `seq32_kv64__ps4_ph1_pffn1`

Dimensions to sweep first:

- `weight_forward_depth`
- `o_proj_fifo_depth`
- `ffn_replay_fifo_depth`
- `use_fused_replayed_addnorm`

Status:

- seq64 `seq32_kv64__ps2_ph2_pffn2`: completed
  - live sweep space was effectively `fa = false` only because the current
    local [add.cc](/home/cj/iron/aie_kernels/generic/add.cc) change creates a
    duplicate-symbol link failure for every `fa = true` point
  - the best one-shot point was `wf=2, of=2, fr=1, fa=false`
  - repeated verification did not hold that apparent win; the current default
    `wf=2, of=2, fr=2, fa=false` remained better
- seq128 `seq32_kv64__ps4_ph1_pffn1`: started, but the full sweep is too
  expensive to brute-force casually; a narrow targeted comparison is now done:
  - default `fa=true` vs `fa=false` on the current `4ps` winner does not yield a
    stable keepable `fa=false` win
  - a longer A/B rerun favored the current default
  - combined data-movement fallback `tg0_uq0` is not legal on this family; it
    exceeds memtile input-DMA budget at compile time
  - keep the current default for the seq128 winner family

#### 4.2 Sweep pruned data-movement toggles

Only after 4.1:

- `use_transport_groups`
- `use_unified_qr_split`

Keep the sweep manual and small. Do not expand the autotune matrix until a real
win appears.

### Phase 5: Return to structural mainline wins

Only start this after the manual sweep has no more easy wins.

#### 5.1 AddNorm1 / AddNorm2 staging redesign

This is still the highest-value mainline structural target.

Priority order:

1. redesign LN1 staging first
2. redesign LN2 residual path second
3. keep exact layer-norm semantics unchanged

Rules:

- do not repeat the already-pruned local bypass variants
- prefer a cleaner stage-boundary redesign over another FIFO tweak

Status:

- attempted on the seq-par single-branch winner path
- the first concrete pass tried to let LN2 hold residual tiles across its two
  internal passes so the second host-side residual refill could be removed
- that branch was pruned:
  - it pushed the seq128 `4ps` winner over a memtile input-DMA limit at compile
    time
  - the tree is back on the last known-good mainline state after reverting it

#### 5.2 Attention-to-LN materialization redesign

If AddNorm redesign stalls:

1. reduce attention intermediate materialization
2. keep more score/probability/PV state on-chip
3. avoid a new runlist boundary

#### 5.3 Fold QKV fully into the production winner path

This is the long-term end state:

- `X` as the public input
- QKV projection fully inside the fused production operator
- no external `QKV`
- no staged-only special case

This phase should begin only after the staged branch has proven the internal
projection approach on a broader supported slice.

## Manual Experiment Protocol

Every performance pass should follow the same discipline.

1. Run `py_compile` on touched files.
2. Run the focused staged or mainline pytest slices relevant to the change.
3. Run at most one hardware process at a time.
4. Use real `bert-base-uncased` weights for hardware checks.
5. Measure both:
   - `qkv_projection_sec`
   - `encoder_pipeline_sec`
6. Treat timeouts as real failures. Do not add retries to make numbers look
   better.
7. Revert branches that are slower or unstable unless they are explicitly
   required for a cleaner long-term contract.

## Exit Criteria

### For the staged branch

The staged-QKV branch should be considered successful only when all of the
following are true:

1. `seq128` is stable without retries
2. `K/V` are no longer recomputed per q-block
3. the staged branch is at least competitive with `packed_host` on the same
   topology slice
4. the public contract remains `X` input, not `QKV`

### For the production benchmark path

The production `encoder_pipeline` path should be considered improved only when:

1. latency improves on the current winner family
2. the focused regression slices still pass
3. no old BD/resource failure is reintroduced
4. the change is stable enough for unattended benchmarking

## Non-Goals For Now

- converting the staged work into a runlist implementation
- reviving the old uneven-FFN exploration
- large speculative topology-space expansion before a real manual win exists
- retry-based masking of runtime instability
- spending more time on the already-pruned local wait/fifo tweaks

## Immediate Next Step

The next practical step is no longer a broad sweep.

1. If continuing staged work:
   - either make the staged `2ph` branch competitive with `packed_host`
   - or explicitly pause staged expansion and keep it as architecture work only
2. If returning to the production benchmark path:
   - the narrow targeted seq128 comparison is done
   - the current default still stands
3. If continuing Phase 5:
   - the next real target is a deeper AddNorm boundary redesign than the pruned
     LN2 residual-hold attempt
   - do not reopen the already-pruned local FIFO / refill variants
