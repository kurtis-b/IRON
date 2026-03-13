# Row-Store Compute Buffer Test Plan

## Goal

Record the compute-side row-store buffering experiments that were already run,
so they do not get repeated while the main optimization work moves to
accumulation-core residency reduction.

The compiler now supports:

- `buffer_count = 2`
- `compute_buffer_count >= 1`
- `compute_produce_buffer_count >= 0`
- `compute_consume_buffer_count >= 0`

The current `iron` wrapper only exposes the shared `compute_buffer_count`
controls today.

The intended tuning direction is:

- keep `buffer_count = 2`
- use `compute_buffer_count` as the shared default
- use `compute_produce_buffer_count` or `compute_consume_buffer_count` to raise
  only the hot side when local L1 is tight
- preserve the compressed memtile-bank lowering so memtile BD usage does not
  grow with `part_count`

## Implementation Status

Implemented in `iron`:

1. [row_store.py](/home/agi-demo/iron/operators/encoder_pipeline/row_store.py)
   - `compute_buffer_count` now validates as `>= 1`
   - the old `compute_buffer_count <= buffer_count` restriction is removed
   - `compute_produce_buffer_count` and `compute_consume_buffer_count` are now
     exposed and validated as `>= 0`

2. [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
   - added env-var overrides:
     - `ENCODER_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_ACCUM_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
     - `ENCODER_ACCUM_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_ACCUM_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
     - `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_LN2_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_OPROJ_ACC_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_BUFFER_COUNT`
     - `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
     - `ENCODER_LN2_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_LN2_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
     - `ENCODER_OPROJ_ACC_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_OPROJ_ACC_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
     - `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
     - `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
   - defaults remain:
     - replay `= 2`
     - accum `= 1`
   - env parsing is now real; the earlier local stubs were removed

## Current Baseline

Validated baseline:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- current default:
  - replay `= 2`
  - accum `= 1`
- active row-stores:
  - `outOProjAccum3`
  - `ln1Replay`
  - `ffnDownAccum1`
  - `ffnDownAccum3`
- `ln2Replay` is not on row-store in this baseline anymore because the current
  design uses direct FFN-down replay on the AddNorm2 path here
- mean latency: `16431.12 us`
- result: pass

## Second-Pass Isolation Results

The site-isolated reruns used the same baseline topology above.

### 1. Raise only `outOProjAccum`

Env:

- `ENCODER_OPROJ_ACC_ROW_STORE_COMPUTE_BUFFER_COUNT=2`

Result:

- compile-time failure
- O-proj accumulation core tile exceeded L1
- first failing tile: `%tile_3_5`
- failing buffers included:
  - `outOProjAccum3_src_0`
  - `outOProjAccum3_src_1`
  - `outOProjAccum3_dst_0`
  - `outOProjAccum3_dst_1`

Conclusion:

- deeper compute-side buffering on O-proj accumulation is not viable in this
  baseline topology

### 2. Raise only `ffnDownAccum`

Env:

- `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_BUFFER_COUNT=2`

Result:

- compile-time failure
- FFN-down core tile exceeded L1
- first failing tile: `%tile_6_3`
- failing buffers included:
  - `ffnDownAccum1_src_0`
  - `ffnDownAccum1_src_1`
  - `ffnDownAccum1_dst_0`
  - `ffnDownAccum1_dst_1`

Conclusion:

- deeper compute-side buffering on FFN-down accumulation is also not viable in
  this baseline topology

### 3. Raise only `ln2Replay`

This result is no longer active on the current tree.

Reason:

- the current `512 / 4pheads / 4pffn / 8pacc / 4opg` memtile baseline no longer
  emits `ln2Replay` as a row-store
- AddNorm2 consumes the direct FFN-down replay path instead

So the older `ln2Replay`-only result should be treated as historical.

## Follow-On Asymmetric Results

Because `ln1Replay` is the actual active replay row-store in the current
baseline, the asymmetric producer/consumer smoke matrix was run there.

### `ln1Replay` producer-heavy `2/1`

Env:

- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT=1`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT=2`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT=1`

Result:

- pass
- mean latency: `16289.50 us`
- improvement vs baseline: `-141.62 us` (`-0.86%`)

### `ln1Replay` consumer-heavy `1/2`

Env:

- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT=1`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT=1`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT=2`

Result:

- pass
- mean latency: `16361.52 us`
- improvement vs baseline: `-69.60 us` (`-0.42%`)

### `ln1Replay` explicit symmetric `2/2`

Env:

- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT=1`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT=2`
- `ENCODER_LN1_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT=2`

Result:

- pass
- mean latency: `16333.74 us`
- improvement vs baseline: `-97.38 us` (`-0.59%`)

### MLIR validation

The generated source MLIR shows the side-specific attributes on `ln1Replay`:

```mlir
aie.memtile_row_store @ln1Replay(...) {
  buffer_count = 2 : i32,
  compute_produce_buffer_count = 2 : i32,
  compute_consume_buffer_count = 1 : i32,
  part_count = 8 : i32
}
```

## Current Practical Conclusion

After the real per-site isolation runs and the asymmetric follow-on runs:

1. O-proj accumulation deeper buffering is blocked by compute-tile L1
2. FFN-down accumulation deeper buffering is blocked by compute-tile L1
3. `ln1Replay` asymmetric tuning is safe, but only modestly helpful
4. producer-heavy `2/1` on `ln1Replay` is the best result so far on the
   current baseline

So the result of this tuning branch is:

- replay-site asymmetry helps a little and is already reflected in the current
  `ln1Replay` default
- accumulation row-store sites are limited by local memory, not by wrapper or
  lowering support
- the later localized FFN-down default on the `64q / 48e / 16pacc / <=2-branch`
  memtile case was measured and rejected because it was slightly slower than
  the explicit `1/1` baseline
- the main optimization work should now follow
  [design_optimization_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/design_optimization_plan.md)

## Step-2 Resident-Buffer Experiments

To test whether the blocker was specifically the weight-consumer footprint, the
two largest resident weight FIFO families were made shallow:

- `ENCODER_OW_FIFO_DEPTH=1`
- `ENCODER_FFN_WEIGHT_FIFO_DEPTH=1`

Those experiments used the same baseline topology:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`

Results:

1. `outOProjAccum 2/1` with `ENCODER_OW_FIFO_DEPTH=1`
   - pass
   - mean latency about `17219 us`

2. `outOProjAccum 1/2` with `ENCODER_OW_FIFO_DEPTH=1`
   - pass
   - mean latency about `17150 us`

3. `ffnDownAccum 2/1` with `ENCODER_FFN_WEIGHT_FIFO_DEPTH=1`
   - pass
   - mean latency about `17135 us`

4. `ffnDownAccum 1/2` with `ENCODER_FFN_WEIGHT_FIFO_DEPTH=1`
   - pass
   - mean latency about `17141 us`

Comparison point:

- current promoted default (`ln1Replay 2/1`) is `16289.50 us`

Interpretation:

- shallowing the dominant weight consumers is enough to make the deeper
  accumulation row-store shapes legal
- but it hurts steady-state performance enough that none of those four
  accumulation-side variants are competitive
- so the next useful optimization is not to keep the weight FIFOs shallow
- it is to reduce the weight-buffer footprint without paying the full
  depth-1 transport penalty

## O-Proj Init-Kernel Follow-Up

Grouped O-proj now uses a real init-style matmul path:

- remove the separate `o_proj_zero_scratch` tile
- start the first partial from zero accumulators directly in the kernel

Observed effect on the same `512seq / 4pheads / 4pffn / 8pacc / 4opg` controls:

Memtile control:
- default with real init kernel: about `16414 us`
- `outOProjAccum 2/1`: about `16374 us`
- `outOProjAccum 1/2`: about `16397 us`

DDR control:
- default with real init kernel: about `13522 us`
- `outOProjAccum 2/1`: about `13931 us`

Conclusion:

- the real init kernel is worth keeping
- it lowers the default grouped O-proj cost in both modes
- `outOProjAccum 2/1` is the best memtile variant tested so far
- but it regresses the DDR control, so it should not be promoted globally

## Activation-Depth Follow-On Experiments

To check whether the remaining blocker was the activation-side residency rather
than the weight-side residency, the active accumulation-side activation FIFOs
were also made shallow:

- `ENCODER_O_PROJ_INPUT_DEPTH=1`
- `ENCODER_FFN_UP_OUT_DEPTH=1`

Same baseline topology:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`

Results:

1. `outOProjAccum 2/1` with `ENCODER_O_PROJ_INPUT_DEPTH=1`
   - pass
   - mean latency about `16748 us`

2. `outOProjAccum 1/2` with `ENCODER_O_PROJ_INPUT_DEPTH=1`
   - pass
   - mean latency about `16723 us`

3. `ffnDownAccum 2/1` with `ENCODER_FFN_UP_OUT_DEPTH=1`
   - compile-time failure
   - `%tile_6_3` still exceeds L1

4. `ffnDownAccum 1/2` with `ENCODER_FFN_UP_OUT_DEPTH=1`
   - compile-time failure
   - `%tile_6_3` still exceeds L1

Interpretation:

- O-proj accumulation is sensitive to both weight and activation residency
- FFN-down accumulation is still dominated by `memBDown*_cons_buff`
- even the legal O-proj-only activation reductions are slower than the promoted
  `ln1Replay 2/1` default (`16289.50 us`)

So the next optimization should target:

- O-proj: the `memOW*_cons_buff` weight footprint or lifetime overlap
- FFN-down: the `memBDown*_cons_buff` weight footprint or lifetime overlap

## Localized Weight-Depth Experiments

Instead of forcing the entire stage to use shallow weight consumers, the next
experiment only made the accumulation sites shallow:

- `ENCODER_SHALLOW_OW_ON_O_PROJ_ACCUM=1`
- `ENCODER_SHALLOW_BDOWN_ON_FFN_ACCUM=1`

Same baseline topology:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`

Results:

1. `outOProjAccum 2/1` with localized shallow `W_O`
   - pass
   - mean latency about `17207 us`

2. `outOProjAccum 1/2` with localized shallow `W_O`
   - pass
   - mean latency about `16965 us`

3. `ffnDownAccum 2/1` with localized shallow `B_down`
   - pass
   - mean latency about `16721 us`

4. `ffnDownAccum 1/2` with localized shallow `B_down`
   - pass
   - mean latency about `16632 us`

Interpretation:

- localizing the shallow weight depth is better than flattening the entire
  stage to depth `1`
- but it is still not enough to beat the promoted default
  `ln1Replay 2/1` (`16289.50 us`)
- the best accumulation-side variant found so far is:
  - `ffnDownAccum 1/2` with localized shallow `B_down`
- even that remains about `2.1%` slower than the current promoted default

## Targeted Follow-Up Checks

The best localized accumulation-side variant was checked on two additional
representative cases:

Variant:

- `ENCODER_SHALLOW_BDOWN_ON_FFN_ACCUM=1`
- `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_BUFFER_COUNT=1`
- `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT=1`
- `ENCODER_FFN_DOWN_ACC_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT=2`

Results:

1. `lnstage_memtile-encoder_2048seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
   - baseline: about `87810 us`
   - localized FFN-down `1/2`: about `87823 us`
   - effectively flat

2. `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
   - baseline: about `4543 us`
   - localized FFN-down `1/2`: about `4517 us`
   - slight improvement

3. `lnstage_ddr-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
   - baseline: about `7600 us`
   - localized FFN-down `1/2`: about `7700 us`
   - slight regression

Interpretation:

- localized FFN-down `1/2` is not a good new global default
- it may be worth considering only as a topology-specific memtile high-`pacc`
  tuning, but not as a general optimization

## Default Promoted

The current design now promotes the best measured safe replay setting on the
main memtile baseline:

- `ln1Replay`
  - `compute_produce_buffer_count = 2`
  - `compute_consume_buffer_count = 1`

This keeps the global replay default unchanged while making `LN1` use the
better measured producer-heavy shape by default.

Status on a representative subset:

- `25 / 25` selected tests passed across:
  - `512seq` control in `memtile`
  - `512seq` control in `ddr`
  - `2048seq` control in `memtile`
  - `64q / 48e / 16pacc / 6pheads / 1pffn / 2opg` in both modes

The mixed-subset timing remained thermally noisy, so the isolated `LN1 2/1`
result is still the cleanest attribution for the promoted default:

- `16431.12 us -> 16289.50 us`
- improvement: `-141.62 us` (`-0.86%`)

## Working Hypothesis

The current row-store performance loss is not primarily caused by memtile row
banking. It is caused by shallow compute-side buffering.

Today, `encoder_pipeline` defaults still use:

- `replay_row_store_compute_buffer_count = 2`
- `accum_row_store_compute_buffer_count = 1`

That means:

- `ln1Replay` and `ln2Replay` already get a small 2-slot compute ring
- `outOProjAccum*` and `ffnDownAccum*` still collapse to a 1-slot compute
  interface

Those accumulation paths previously used FIFO depth proportional to
`proj_acc_depth`, so replacing them with `compute_buffer_count = 1` is a likely
throughput regression for high-`pacc` topologies.

## What Must Change In `iron`

The compiler change alone is not enough. The operator wrapper still blocks the
new tuning range.

There are now two separate compiler directions:

1. short-term diagnostic path:
   raise local `compute_buffer_count`
2. preferred long-term path:
   use asymmetric producer/consumer buffering so queue depth increases without
   doubling both local `src` and `dst` rings

This document is primarily for the short-term diagnostic path. If the benchmark
improves but local-L1 pressure becomes the next limiter, the next lowering to
test should be the asymmetric compute-buffer design described in
[MemTileRowStoreLowering.md](/home/agi-demo/mlir-aie/docs/MemTileRowStoreLowering.md).

### 1. Relax `row_store.py`

In [row_store.py](/home/agi-demo/iron/operators/encoder_pipeline/row_store.py):

- remove the `compute_buffer_count not in (1, 2)` restriction
- replace it with `compute_buffer_count >= 1`
- remove the `compute_buffer_count <= buffer_count` restriction

Reason:

- compute-side ring depth and memtile bank count are separate resources
- the new compiler support is specifically intended to allow
  `compute_buffer_count > buffer_count`

### 2. Make The Counts Easy To Override

In [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py):

This is now implemented.

Suggested env vars:

- `ENCODER_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT`
- `ENCODER_ACCUM_ROW_STORE_COMPUTE_BUFFER_COUNT`
- `ENCODER_REPLAY_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
- `ENCODER_REPLAY_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
- `ENCODER_ACCUM_ROW_STORE_COMPUTE_PRODUCE_BUFFER_COUNT`
- `ENCODER_ACCUM_ROW_STORE_COMPUTE_CONSUME_BUFFER_COUNT`
- plus the site-specific `LN1`, `LN2`, `OPROJ`, and `FFN_DOWN` variants

These now allow benchmarking without editing the design repeatedly.

## Compute-Tile Resource Cost

For a row-store with compute-visible tile size `tile_bytes`, compute-tile memory
cost is:

```text
(
  effective_produce_buffer_count +
  effective_consume_buffer_count
) * tile_bytes
```

plus one tiny `memref<2xi32>` index buffer when either side uses more than one
slot.

For the relevant encoder cases:

- `32 x 96 x bf16 = 6144` bytes
- `64 x 48 x bf16 = 6144` bytes

So per row-store site:

- `1/1` -> `12,288` bytes
- `2/1` or `1/2` -> `18,440` bytes
- `2/2` -> `24,584` bytes
- `4/4` -> `49,160` bytes

On NPU2 core tiles (`64 KiB` local memory), `4` is expensive enough that it may
become a new local-memory and lock-budget limiter. Start with `2`.

That is why symmetric deeper split-ring buffering should be treated as a
stopgap. If the benchmarks improve, the better end-state is asymmetric
producer/consumer buffering, not indefinitely increasing both sides together.

## Benchmark Matrix

Use the current measured topologies as the reference set.

### Baseline Cases

1. `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
   - current row-stores: `LN1`, `O-proj accumulation`, `FFN-down accumulation`
   - current mean latency: `16431.12 us`

2. `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_4pheads_4pffn_16pacc_4opg`
   - current row-stores: `LN1`, `O-proj accumulation`, `FFN-down accumulation`
   - current mean latency: `21095.98 us`

3. `lnstage_ddr-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
   - comparison topology
   - current row-stores: `LN1`, `O-proj accumulation`
   - current mean latency: `12106.08 us`

### First-Pass Tuning Matrix

Run the memtile cases above with:

1. current defaults
   - replay `= 2`
   - accum `= 1`

2. accumulation only
   - replay `= 2`
   - accum `= 2`

3. deeper replay only
   - replay `= 4`
   - accum `= 1`

4. both raised
   - replay `= 4`
   - accum `= 2`

If the `16pacc` case still regresses badly, stop before trying larger values.

### Follow-On Asymmetric Matrix

Once the `iron` wrapper exposes the side-specific controls, rerun the hot
row-store sites with:

1. producer side only
   - `compute_produce_buffer_count = 2`
   - `compute_consume_buffer_count = 1`

2. consumer side only
   - `compute_produce_buffer_count = 1`
   - `compute_consume_buffer_count = 2`

3. symmetric double buffering
   - `compute_produce_buffer_count = 2`
   - `compute_consume_buffer_count = 2`

This should show whether the stall is mostly on the producer or consumer side
before paying for the full symmetric `2/2` shape.

### Second-Pass Isolation

If the first pass shows improvement but not enough, isolate the biggest offender:

1. keep only `outOProjAccum` on raised compute depth
2. keep only `ffnDownAccum` on raised compute depth
3. keep only `ln2Replay` on raised compute depth

This identifies which row-store site benefits most from deeper compute-side
buffering.

## Expected Outcome

What should improve if the hypothesis is right:

- high-`pacc` memtile topologies should close part of the gap to the lower-row-store
  DDR contender
- the `16pacc` memtile case should benefit more than the `8pacc` memtile case
- the biggest gain should come from raising accumulation row-store depth from
  `1` to `2`

What should *not* change:

- memtile bank count remains `2`
- memtile BD shape remains compressed-bank for the double-buffered path
- shim drain behavior should be unchanged

## Pass / Fail Criteria

Call the experiment successful if all of these hold:

1. `accum = 2` improves latency on both memtile baselines
2. `16pacc` improves by a larger absolute amount than `8pacc`
3. no new compile-time failures appear from compute-tile local-memory or lock
   pressure
4. no new runtime hangs appear

Call it inconclusive if:

- one topology improves and another regresses
- the gain is within benchmark noise
- compile-time resource limits replace the throughput loss

## Suggested Commands

From `/home/agi-demo/iron`:

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
```

Baseline:

```bash
pytest operators/encoder_pipeline/test.py \
  -q \
  -k "lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg" \
  --iterations 5 \
  -s -x
```

Accumulation raised:

```bash
ENCODER_ACCUM_ROW_STORE_COMPUTE_BUFFER_COUNT=2 \
pytest operators/encoder_pipeline/test.py \
  -q \
  -k "lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg" \
  --iterations 5 \
  -s -x
```

Replay + accumulation raised:

```bash
ENCODER_REPLAY_ROW_STORE_COMPUTE_BUFFER_COUNT=4 \
ENCODER_ACCUM_ROW_STORE_COMPUTE_BUFFER_COUNT=2 \
pytest operators/encoder_pipeline/test.py \
  -q \
  -k "lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_4pheads_4pffn_16pacc_4opg" \
  --iterations 5 \
  -s -x
```

## What To Record

For each run, record:

- topology id
- `replay_row_store_compute_buffer_count`
- `accum_row_store_compute_buffer_count`
- mean latency
- pass/fail
- compile-time failure mode, if any
- runtime hang or mismatch count, if any

## Interpretation

If `accum = 2` helps substantially, the main issue was exactly what the current
compiler work was designed to fix: compute-side queue-depth collapse on the
same-core accumulation row-stores.

If `accum = 2` or `replay = 4` helps but local-memory pressure becomes the next
problem, that is the signal to switch from the symmetric split-ring experiment
to asymmetric producer/consumer buffering rather than pushing both sides'
`compute_buffer_count` higher.

If only `replay = 4` helps, then the replay-heavy LN paths still need more
compute-side depth than the current default.

If neither helps much, then the next bottleneck is probably elsewhere:

- compute-tile local memory pressure
- shim drain pressure
- non-row-store transport traffic
- topology differences unrelated to row-store
