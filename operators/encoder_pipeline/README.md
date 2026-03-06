# encoder_pipeline

This directory contains the Python-side implementation and tests for the fused encoder pipeline operator:
MHA + AddNorm1 + FFN + AddNorm2.

## Python files

- `design.py`
  - Generates the AIE MLIR graph (`fused_mha`).
  - Defines object FIFOs, worker kernels, tile placement, tensor access patterns, and runtime fill/drain sequence.
  - Includes CLI entrypoint (`main`) to emit MLIR to `build/encoder_pipeline.mlir` (or a provided output path).

- `design_ln1_ddr.py`
  - DDR-staged LN1 wrapper entrypoint.
  - Forwards to `design.py:fused_mha` with `ln1_stage_mode="ddr"`.

- `design_ln1_memtile.py`
  - Memtile-staged LN1 wrapper entrypoint.
  - Forwards to `design.py:fused_mha` with `ln1_stage_mode="memtile"`.

- `op.py`
  - Runtime-facing operator wrapper class: `AIEEncoderPipeline`.
  - Validates shape/depth constraints, builds compile artifacts, sets up runtime buffers, and runs the kernel.
  - Supports separate AddNorm-1 and AddNorm-2 compile-time LN weights (`ln1_weight`, `ln2_weight`), with `ln_weight` as backward-compatible fallback.
  - Converts tensors to/from host buffers (`QKV`, `W_O`, `OR`, `B_Up`, `B_Down`, `O`).

- `reference.py`
  - Golden reference model used by tests.
  - Mirrors debug/stage-isolation behavior expected by the hardware graph.
  - Produces reference inputs/outputs and packed host buffers.

- `test.py`
  - Parametrized pytest suite for `AIEEncoderPipeline`.
  - Generates golden data from `reference.py`, runs the operator, and checks output error counts within fixed tolerances.

## How files connect

1. `test.py` calls `reference.py` to generate input/output tensors.
2. `test.py` instantiates `AIEEncoderPipeline` from `op.py`.
3. `op.py` invokes `design.py` (via artifact generation) to produce MLIR/XCLBIN/insts.
4. Runtime executes the design and compares `O` against the reference output.

## Debug modes

`op.py` and `reference.py` accept a single top-level `debug` argument:

- `-1`: debug disabled (default full pipeline behavior).
- `0`: self-attention debug path; AddNorm stages pass through their input path.
- `1`: deterministic MHA-input debug path (linear QK/PV flow); AddNorm stages pass input.
- `2`: residual-path debug; AddNorm stages pass residual.
- `3`: FFN up-proj isolation (`ffn_stage_only=0`), deterministic MHA feed.
- `4`: FFN down-proj isolation (`ffn_stage_only=1`), deterministic MHA feed.
- `5`: FFN AddNorm2 isolation (`ffn_stage_only=2`), deterministic MHA feed.
- `6`: MHA-focused profiling path (FFN up/down disabled, AddNorm1 bypass input, AddNorm2 bypass residual).
- `7`: AddNorm1-focused profiling path (FFN up/down disabled, AddNorm2 bypass residual).

Internally, `op.py` resolves `debug` into:
- MHA kernel debug value (`mha_debug`)
- FFN stage-only selector (`ffn_stage_only`)
- per-stage AddNorm pass-through modes (`addnorm1_debug_mode`, `addnorm2_debug_mode`)

## Dataflow mapping and pipelining

- Host-facing buffer packing:
  - `QKV` packs `[Q; K; V]` row-wise.
  - `OR` packs `[output_region; residual; ln1_stage_scratch]` row-wise.
  - `B_Up` and `B_Down` are flat host buffers; `TensorAccessPattern`s in `design.py` define their logical 2D tile order.
  - Runtime aliases `O` to the first half of `OR`.

- Stage-to-stage flow on the array:
  - MHA path:
    - `Q/K -> QK matmul -> softmax -> PV matmul -> O-proj`
    - O-proj accumulates through mem-tile FIFOs (`outOProjAccum*`) and reduces across parallel-head workers (`outOPart*`) to `outO`.
  - AddNorm1 bridge:
    - `outO + R -> AddNorm1`
    - LN1 uses mem-tile replay buffering (`ln1ReplayPart -> ln1Replay`) so full-row statistics
      are computed once and replayed across FFN column groups.
    - residual stream for AddNorm2 is staged through `ffnROut -> ffnRIn`:
      - single-branch FFN path: emitted by LN1 mul+add core.
      - 2-branch path with LN1 two-branch router enabled: emitted by LN1 mul+add core,
        then LN1 route split fans out FFN branch traffic.
      - other multi-branch paths: emitted by FFN-up branch 0.
  - FFN path:
    - `outLN + B_Up -> FFN up (with GeLU) -> ffnUpOut`
    - `ffnUpOut + B_Down -> FFN down`, with accumulation queue in `ffnDownAccum`.
    - when multiple FFN branches are active, down-proj branches reduce in a chain
      (`ffnDownReduce*`) into a final/root branch before AddNorm2.
    - LN1 DDR staging mode (default design path, `ENCODER_LN1_STAGING_DESIGN=ddr`):
      - LN1 branch outputs drain via memtile/shim to host staging buffer.
      - FFN-up A inputs are refilled from host staging buffer via shim/memtile.
      - runtime uses a one-step interleave pipeline:
        - first staged drain is explicitly waited (bootstrap),
        - then previous-tap refill/output and current-tap MHA-side fills are overlapped,
        - current-tap staged refill is carried as pending work to the next tap.
  - Final norm:
    - `ffnDownOut + ffnRIn -> AddNorm2 -> outLN2 -> memLN2 -> drain to OR/O`.

- Runtime pipelining model:
  - All workers are started once and run continuously.
  - Runtime iterates by `(q_block_idx, col_group)` and schedules fills/drain for one tile-group at a time:
    - fill `Q` for the q-block
    - for each head-group, fill `K`, `V`, `W_O`
    - fill residual `R`
    - fill FFN weights `B_Up`, `B_Down` for the same col-group
    - drain final output tile from `memLN2` to `OR`/`O`.
  - Tail-stage host IO is intentionally serialized (`serialize_tail_io=True`) for deterministic
    replay/order behavior.

- Two-pass behavior used for AddNorm stages:
  - O-proj/AddNorm1:
    - pass 1 provides tiles for sum/sumsq accumulation.
    - pass 2 replays tiles for normalized output application.
  - FFN-down/AddNorm2 uses the same pattern:
    - FFN-down emits/replays accumulated tiles so AddNorm2 can do sum/sumsq then normalized apply.

## Current effective-branch behavior (`nB_tiles_distributed`)

`nB_tiles_distributed` is the requested FFN branch count. The generated design may reduce it at
compile time when placement/channel limits are hit:

1. Initial clamp to available mapped branch tiles (mapping now searches up to 6-branch chains).
2. If no spare free tile remains for LN1-post (and optional LN1-route) worker placement,
   non-root branches are pruned.
3. Inline FFN reduction is mapped in `ffn_addnorm` style (`ffnDownReduce*` chain), but
   memtile-staged FFN-down accumulation (kept for LN full-row/two-pass requirements) already
   consumes three input DMA streams per down core.
4. Because of (3), adding the reduction input stream can exceed down-core input DMA channel
   budget, so multi-branch requests are pruned until feasible.
5. Shim-output budget check is applied for FFN weight streams (`B_Up` + `B_Down`).
6. High-acc tail liveness/BD guard is applied for `proj_acc_depth >= 6`, currently capping to at
   most 2 physical FFN branches for stable execution.
7. Standard compute tile budget check (`<= 32` total compute tiles) is also enforced.

When physical branches are pruned below requested `nB_tiles_distributed`, the requested `nB`
value is still used as a logical FFN partition count and those logical parts are mapped onto the
remaining physical branch workers.

The effective branch count is logged by `design.py` at compile time.

## Stage Bottleneck Profiling

`test.py` now includes an opt-in stage profiling path that mirrors `ffn_addnorm`-style
`stage_only` profiling for encoder pipeline stages (MHA/AddNorm1/FFN/AddNorm2 focused modes).

- Enable profiling with `ENCODER_PIPELINE_STAGE_PROFILE=1`.
- Profiled modes are controlled by `ENCODER_PIPELINE_STAGE_PROFILE_MODES`
  (comma-separated from `none,0,1,2,3,4`).
  - `none`: full encoder pipeline (`debug=-1`)
  - `0`: FFN up-proj-focused run (`debug=3`)
  - `1`: FFN down-proj-focused run (`debug=4`)
  - `2`: AddNorm2-focused run (`debug=5`)
  - `3` / `mha`: MHA-focused run (`debug=6`)
  - `4` / `an1`: AddNorm1-focused run (`debug=7`)
- Workload is set by `ENCODER_PIPELINE_STAGE_PROFILE_CASE`:
  - format: `seq_len,d,heads,intermediate_size,q_seq_tile,kv_seq_tile,emb_tile,parallel_heads,parallel_ffn,proj_acc_depth`
- Profiling iteration controls:
  - `ENCODER_PIPELINE_STAGE_PROFILE_WARMUP_ITERS`
  - `ENCODER_PIPELINE_STAGE_PROFILE_TIMED_ITERS`

Example:

```bash
pytest operators/encoder_pipeline/test.py -s --iterations 1 -k stage_profile \
  -o log_cli=true
```

with:

```bash
export ENCODER_PIPELINE_STAGE_PROFILE=1
export ENCODER_PIPELINE_STAGE_PROFILE_CASE="512,64,12,3072,32,64,128,6,3,6"
export ENCODER_PIPELINE_STAGE_PROFILE_MODES="none,mha,an1,0,1,2"
export ENCODER_PIPELINE_STAGE_PROFILE_WARMUP_ITERS=5
export ENCODER_PIPELINE_STAGE_PROFILE_TIMED_ITERS=50
```

Notes:
- Full mode (`none`) keeps the normal numeric threshold check.
- Stage-only modes print latency/bandwidth and intentionally skip numeric assertions.
- This is intended for bottleneck analysis, not functional correctness gating.

### Scripted All-Mode Bottleneck Profiling

For automated per-test-case sweeps across all debug modes (`-1..7`) and bottleneck detection,
use:

```bash
python operators/encoder_pipeline/profile_debug_modes.py --clean-build
```

Useful options:
- `--design ddr|memtile`:
  choose LN1 staging design path for the run (default `ddr`)
- `--case-index 0,2,4`:
  run only selected generated test-case indices
- `--warmup-iters N` / `--timed-iters N`:
  control measurement iterations
- `--output-json /tmp/encoder_profile.json`:
  emit machine-readable results

The script prints each mode latency/error status and reports the bottleneck stage among:
`mha`, `addnorm1`, `ffn_up`, `ffn_down`, `addnorm2`.

## Experimental Tail-IO Toggle

`design.py` supports an opt-in runtime-ordering experiment:
- `ENCODER_SERIALIZE_TAIL_IO`:
  - default unset -> strict serialized tail IO (current stable behavior)
  - set to `0|false|off|no` -> disable strict tail serialization (profiling experiment only)
- `ENCODER_TAIL_WAIT_MODE` (applies when `ENCODER_SERIALIZE_TAIL_IO` is enabled):
  - `strict` (default)
  - `relax_ffn_weights`
  - `relax_ffn_weights_residual`
  - `relax_all_tail` (known liveness risk)
- `ENCODER_SERIALIZE_Q_PRESTAGE`:
  - default unset -> keep dedicated Q pre-stage task-group barrier
  - set to `0|false|off|no` -> schedule Q fill into the main task-group while keeping
    the rest of tail ordering strict (profiling experiment only)

This override is for A/B performance experiments; strict ordering remains the default because
full-pipeline liveness/correctness is sensitive to tail DMA order.

## FFN Tail Mapping Knobs

`design.py` also exposes guarded placement/pruning knobs for FFN-tail resource exploration:
- `ENCODER_LN1_STAGING_DESIGN`:
  - selects which design entry file is used by `AIEEncoderPipeline`:
    - `ddr` (default): LN1 staged through DDR path (`design_ln1_ddr.py`)
    - `memtile`: LN1 staged on memtile path (`design_ln1_memtile.py`)
  - aliases accepted:
    - DDR: `ddr`, `dram`, `host`
    - Memtile: `memtile`, `mt`, `onchip`
- `ENCODER_FORCE_SINGLE_BRANCH_WIDE_ACC`:
  - default unset -> `true` (keep conservative single-branch fallback for `parallel_heads>=6` and `proj_acc_depth>=6`)
  - set to `0|false|off|no` -> allow multi-branch attempt (may fail compile due memtile BD/channel limits)
- `ENCODER_FORCE_SINGLE_BRANCH_HIGH_ACC`:
  - default unset -> `true` (keep conservative single-branch fallback for `parallel_heads>=4` and `proj_acc_depth>=8`)
  - set to `0|false|off|no` -> allow multi-branch attempt (may fail compile due memtile BD/channel limits)
- `ENCODER_FORCE_SINGLE_BRANCH_ACC_GE6`:
  - default unset -> `true` (keep conservative high-acc fallback for `proj_acc_depth>=6`)
  - when enabled, current behavior caps high-acc tails to at most 2 physical FFN branches
    (instead of forcing a single branch)
  - set to `0|false|off|no` -> allow `>2` branch attempts (may fail compile or runtime liveness)
- `ENCODER_LN2_REPLAY_MEM_TILE_COL`:
  - memtile column for LN2 replay staging in the FFN-down/AddNorm2 handoff path
  - default is `4` in current implementation
- `ENCODER_EMIT_LN2_REPLAY_FROM_DOWN`:
  - default unset -> `true` (final FFN-down core emits replay pass directly to LN2 input)
  - set to `0|false|off|no` -> enable LN2 replay memtile FIFO path (higher memtile pressure)
- `ENCODER_BRANCH_STAGE_COLS`:
  - low-head (`parallel_heads<6`) override for FFN stage memtile column pool
  - must provide 6 comma-separated ints (same arity as default pool)
- `ENCODER_BRANCH_BUP_COLS`:
  - low-head (`parallel_heads<6`) override for FFN `B_Up` memtile columns
  - must provide one column per effective FFN branch
- `ENCODER_BRANCH_DOWN_B_COLS`:
  - low-head (`parallel_heads<6`) override for FFN `B_Down` memtile columns
  - must provide one column per effective FFN branch
- `ENCODER_USE_B_WEIGHT_SPLIT`:
  - default unset -> `false`
  - when set to `1|true|on|yes`, enables experimental packed `B_Up`/`B_Down` shim streams with memtile `split(...)` fanout per FFN branch chunk
  - currently intended for bring-up only; runtime liveness is not yet stable on all high-acc profiles
- `ENCODER_USE_BUP_SPLIT`:
  - applies when `ENCODER_USE_B_WEIGHT_SPLIT` is enabled
  - default unset -> `true`
  - controls whether `B_Up` uses packed+split ingress (`1`) or legacy per-branch ingress (`0`)
- `ENCODER_USE_BDOWN_SPLIT`:
  - applies when `ENCODER_USE_B_WEIGHT_SPLIT` is enabled
  - default unset -> `true`
  - controls whether `B_Down` uses packed+split ingress (`1`) or legacy per-branch ingress (`0`)
- `ENCODER_B_WEIGHT_SPLIT_CHUNK_SIZE`:
  - applies only when `ENCODER_USE_B_WEIGHT_SPLIT` is enabled
  - controls branches per packed B stream chunk (default `min(3, effective_ffn_branches)`)
  - must be `> 0`
- `ENCODER_B_WEIGHT_SPLIT_PARENT_DEPTH`:
  - applies when either `ENCODER_USE_BUP_SPLIT` or `ENCODER_USE_BDOWN_SPLIT` is enabled
  - controls packed parent ObjectFifo depth (default `max(2, legacy_depth)`)
  - must be `> 0`
- `ENCODER_FFN_BRANCH_START_IDX`:
  - override starting branch index of the contiguous selected FFN branch subset
  - default selects a suffix ending at the layout-preferred root
- `ENCODER_FFN_GROUP_SPLIT`:
  - override physical FFN group counts per effective branch (comma-separated)
  - values must be `>0`, one per effective branch, summing to `profile_replay_groups`
- `ENCODER_ENABLE_LN1_TWO_BRANCH_ROUTER`:
  - default unset -> `true` for `proj_acc_depth>=6`, else `false`
  - when enabled for effective 2-branch FFN, inserts one LN1 route worker so LN1 can emit
    AddNorm2 residual directly (improves replay/order stability for 2-branch high-acc tails)
- `ENCODER_BYPASS_LN_TO_FFN_STAGE`:
  - controls LN1->FFN memtile staging bypass
  - default unset enables bypass for high-acc multi-branch when LN1 two-branch router mode is active
  - set to `0|false|off|no` to force staging path (can increase memtile BD pressure)
- `ENCODER_STAGE_LN1_TO_DDR`:
  - legacy/low-level override only when invoking `design.py` directly
    (the operator-level knob is `ENCODER_LN1_STAGING_DESIGN` and defaults to DDR).
  - default unset -> `true` in `design.py`
  - when enabled (`1|true|on|yes`), stages LN1->FFN A-stream through host DDR:
    - `LN1 -> memtile -> shim -> host` drain, then
    - `host -> shim -> memtile -> FFN-up` fill.
  - uses the appended `ln1_stage_scratch` region inside `OR` (no extra kernel argument).
  - if enabled, `ENCODER_BYPASS_LN_TO_FFN_STAGE` is forced off.
  - runtime now schedules LN1 refills and FFN weight fills together to avoid handoff deadlocks.
  - staged multi-branch drains are issued in one task-group to avoid LN1 fanout backpressure deadlocks.
- `ENCODER_LN1_DDR_STAGE_FIFO_DEPTH`:
  - override memtile FIFO depth for LN1 DDR staging forwards (`memOutLNStageToDDR*` and staged `memOutLN*`)
  - default `1` to reduce memtile BD pressure in staged larger-topology mappings.
- `ENCODER_FFN_DOWN_REDUCE_DEPTH`:
  - override depth for `ffnDownReduce*` inter-branch reduction FIFOs (default `1`)
- `ENCODER_FFN_DOWN_OUT_DEPTH`:
  - override depth for `ffnDownOut` FIFO (default follows internal heuristic)

Note:
- FFN-down replay-from-down (`ENCODER_EMIT_LN2_REPLAY_FROM_DOWN=1`) keeps full-row LN behavior while freeing LN2 replay memtile channels/BDs.
- LN2 replay FIFO mode (`ENCODER_EMIT_LN2_REPLAY_FROM_DOWN=0`) is still available for targeted experiments.
- Even with force-single overrides disabled, `proj_acc_depth >= 6` currently has an additional
  stability guard path for `>2` physical FFN branches.
- Current staged validation (March 6, 2026):
  - `ENCODER_LN1_STAGING_DESIGN=ddr pytest operators/encoder_pipeline/test.py -q --iterations 1` -> `5 passed`.
