# encoder_pipeline

This directory contains the Python-side implementation and tests for the fused encoder pipeline operator:
MHA + AddNorm1 + FFN + AddNorm2.

## Python files

- `design.py`
  - Generates the AIE MLIR graph (`fused_mha`).
  - Defines object FIFOs, worker kernels, tile placement, tensor access patterns, and runtime fill/drain sequence.
  - Includes CLI entrypoint (`main`) to emit MLIR to `build/encoder_pipeline.mlir` (or a provided output path).

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
  - `OR` packs `[output_region; residual]` row-wise.
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
      - multi-branch FFN path: emitted by FFN-up branch 0.
  - FFN path:
    - `outLN + B_Up -> FFN up (with GeLU) -> ffnUpOut`
    - `ffnUpOut + B_Down -> FFN down`, with accumulation queue in `ffnDownAccum`.
    - when multiple FFN branches are active, down-proj branches reduce in a chain
      (`ffnDownReduce*`) into a final/root branch before AddNorm2.
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

1. Initial clamp to available mapped branch tiles.
2. If no spare free tile remains for LN1-post (and optional LN1-route) worker placement,
   non-root branches are pruned.
3. Inline FFN reduction is mapped in `ffn_addnorm` style (`ffnDownReduce*` chain), but
   memtile-staged FFN-down accumulation (kept for LN full-row/two-pass requirements) already
   consumes three input DMA streams per down core.
4. Because of (3), adding the reduction input stream can exceed down-core input DMA channel
   budget, so multi-branch requests are pruned until feasible.
5. Additional memtile BD/channel feasibility guards are applied for high-acc configurations:
   - `parallel_heads >= 6` with `proj_acc_depth >= 6`: prune to a single branch.
   - `parallel_heads >= 4` with `proj_acc_depth >= 8`: prune to a single branch.
6. Standard compute tile budget check (`<= 32` total compute tiles) is also enforced.

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
- `ENCODER_FORCE_SINGLE_BRANCH_WIDE_ACC`:
  - default unset -> `true` (keep conservative single-branch fallback for `parallel_heads>=6` and `proj_acc_depth>=6`)
  - set to `0|false|off|no` -> allow multi-branch attempt (may fail compile due memtile BD/channel limits)
- `ENCODER_FORCE_SINGLE_BRANCH_HIGH_ACC`:
  - default unset -> `true` (keep conservative single-branch fallback for `parallel_heads>=4` and `proj_acc_depth>=8`)
  - set to `0|false|off|no` -> allow multi-branch attempt (may fail compile due memtile BD/channel limits)
- `ENCODER_LN2_REPLAY_MEM_TILE_COL`:
  - memtile column for LN2 replay staging in the FFN-down/AddNorm2 handoff path
  - default is `4` in current implementation

Note:
- FFN-down now emits one pass to LN2 input and LN2 replays internally from memtile staging for norm statistics/output pass.
- This keeps full-row LN behavior while reducing FFN-down replay pressure in the down-proj core.
