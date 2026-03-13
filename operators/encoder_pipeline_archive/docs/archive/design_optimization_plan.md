# encoder_pipeline Optimization Plan

## Goal

Improve end-to-end latency on the passing `encoder_pipeline` topologies without
reopening the old memtile-BD/resource failures.

The current design already fixed the original memtile BD wall by moving the
same-core row staging paths onto `aie.memtile_row_store` where stable. The next
limits are:

1. accumulation-core L1 pressure
2. tail stream / transport pressure
3. AddNorm1 cost on the higher-`pacc` passing topologies
4. runtime overlap that is still conservative

## Current Bottlenecks

### 0. Full-vs-stage overlap gap

Recent controlled stage-profile runs show two different regimes:

- `memtile` control:
  `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
  - `full`: `16488.12 us`
  - max stage-only: `addnorm1 = 11739.34 us`
  - exposed gap: `4748.78 us`
  - `full / max_stage = 1.405`
- `ddr` fast case:
  `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
  - `full`: `12226.31 us`
  - max stage-only: `addnorm1 = 8984.99 us`
  - exposed gap: `3241.32 us`
  - `full / max_stage = 1.361`
- higher-`pacc` passing memtile case:
  `encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
  - `full`: about `4349 us`
  - max stage-only: `addnorm1 = 3514 us`
  - exposed gap: about `835 us`
  - `full / max_stage = 1.238`

Interpretation:

- There is still meaningful non-overlapped work on the `512seq` control and
  fastest `512seq` DDR topology. The end-to-end latency is roughly `36-41%`
  above the bottleneck isolated stage.
- The higher-`pacc` passing memtile topology is much closer to the bottleneck
  stage, so it is less dominated by conservative sequencing and more directly
  dominated by `AddNorm1`.
- Optimization work should therefore split into:
  1. reducing `AddNorm1` cost directly on the higher-`pacc` path
  2. improving overlap and reducing exposed transport/runtime work on the
     `512seq` control and fastest DDR topology

### 1. O-proj accumulation core L1

Observed on the main memtile control:

- `outOProjAccum` deeper compute-side buffering (`2/1`, `1/2`, `2/2`) runs into
  O-proj accumulation core L1 unless other resident buffers are reduced first.
- The biggest resident O-proj-side non-row-store buffer family is `memOW*_cons`.

Relevant code:

- O-proj weight streams and depths:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- O-proj accumulation row-store creation:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- O-proj BF16 init kernel:
  - [mm.cc](/home/agi-demo/iron/aie_kernels/aie2p/mm.cc)

### 2. FFN-down accumulation core L1

Observed on the same control:

- `ffnDownAccum` deeper compute-side buffering is still blocked by local L1.
- The dominant resident non-row-store buffer family there is `memBDown*_cons`.

Relevant code:

- FFN-down weight streams and depths:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- FFN-down accumulation row-store creation:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- FFN-down core contract:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)

### 3. Tail transport pressure

The remaining hard failures on the aggressive `64q / 48e / 16pacc` matrix are
not replay-site BD issues anymore. They are mostly tail-side output-channel
pressure.

Relevant code:

- LN2 replay / direct replay policy:
  - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- memtile / DDR mode placement hooks:
  - [design_ln1_memtile.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_memtile.py)
  - [design_ln1_ddr.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_ddr.py)

### 4. AddNorm1 cost on higher-`pacc` passing cases

Observed on the current high-`pacc` memtile case:

- `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`

Stage-profile results collected so far:

- `full`: about `4349 us`
- `self_attn`: about `1637 us`
- `mha_input`: about `1568 us`
- `residual`: about `1572 us`
- `ffn_up`: about `1590 us`
- `ffn_down`: about `1387 us`
- `addnorm2`: about `1063 us`
- `mha`: about `1068 us`
- `addnorm1`: about `3514 us`

Conclusion:

- on this passing higher-`pacc` topology, `addnorm1` is now the dominant stage
- the next meaningful structural optimization should target AddNorm1 replay /
  normalization cost rather than FFN-up/down

## Implementation Order

### Step 1. O-proj accumulation residency reduction

Target:

- make `outOProjAccum` `2/1` legal and profitable in `memtile` mode without
  globally forcing shallow `W_O`

Files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- [mm.cc](/home/agi-demo/iron/aie_kernels/aie2p/mm.cc)
- [op.py](/home/agi-demo/iron/operators/encoder_pipeline/op.py)

Work:

1. keep the real O-proj init kernel
2. promote a conservative memtile-only O-proj accumulation `2/1` default only
   if it does not regress DDR
3. if that regresses DDR, keep it as a topology/mode-specific tuning rather
   than a global default

Validation:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- `lnstage_ddr-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
- `lnstage_ddr-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`

Status:

- partially implemented

Measured result on the current tree:

- memtile control
  `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
  improves slightly with memtile-only O-proj `2/1`
- same-tree comparison:
  - memtile-only O-proj `2/1` default: about `16461 us`
  - explicit O-proj `1/1` override: about `16507 us`
  - delta: about `-46 us` (`-0.28%`)
- DDR is unchanged by construction because the new default is memtile-only
- broader memtile check still passes:
  - `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`

### Step 2. FFN-down accumulation residency reduction

Target:

- make `ffnDownAccum` asymmetric buffering legal without forcing shallow
  `B_down` globally

Files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)

Work:

1. reduce `memBDown*_cons` overlap only on the down stage/root branches
2. benchmark the best localized `ffnDownAccum 1/2` variant again after any
   residency reduction

Validation:

- same control pair as step 1
- plus
  `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`

Status:

- tuning branch exhausted

Measured result on the current tree:

- the targeted high-`pacc` memtile case
  `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`
  does not benefit from the localized FFN-down tuning default
- same-tree comparison:
  - tuned default: about `4347 us`
  - explicit FFN-down `1/1` baseline override: about `4336 us`
  - delta: about `+10 us` (`+0.24%`)
- conclusion:
  - do not promote the FFN-down tuning default
  - keep the localized knobs only for experiments
  - move the plan to structural/runtime work instead of more accumulation-site
    buffer-count tuning

### Step 3. Tail stream-count reduction

Target:

- reduce output-producing streams near LN2 for the still-failing aggressive
  topologies

Files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- [design_ln1_memtile.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_memtile.py)
- [design_ln1_ddr.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_ddr.py)

Work:

1. keep direct FFN-down replay where it really removes a tail stream
2. avoid reintroducing extra replay streams on already-pressured tail columns
3. only revisit this after step 1/2 gains are exhausted

Status:

- pending

### Step 4. Runtime overlap cleanup

Target:

- reduce conservative serialization that is no longer needed after the current
  row-store and tail changes

Files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)

Work:

1. compare `full` against max stage-only for each representative topology after
   every major design change
2. inspect B-weight fill ordering and waits on the `512seq` control and fastest
   DDR topology, where the exposed full-vs-stage gap is still large
3. inspect tail wait modes and host/drain synchronization
4. keep runtime changes secondary to the structural fixes above, but prioritize
   them earlier if the exposed overlap gap stays above about `30%`

Status:

- pending

### Step 5. Major design / compiler changes

Target:

- address the remaining limits that local row-store tuning cannot move:
  accumulation-core L1 pressure and tail transport pressure

Files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
- [design_ln1_memtile.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_memtile.py)
- [design_ln1_ddr.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_ddr.py)
- [row_store.py](/home/agi-demo/iron/operators/encoder_pipeline/row_store.py)

Compiler-side follow-on:

- row-store lowering / placement in `mlir-aie`
- tail drain coalescing / channel budgeting in `mlir-aie`

Work:

1. reduce tail stream count structurally rather than by memtile remaps alone
2. explore accumulation contracts that lower compute-side row-store footprint
3. reduce AddNorm1 cost on the higher-`pacc` passing cases, likely by moving
   more partial-stat work upstream or shortening replay traffic
4. revisit runtime overlap only after the stream-count, L1, and AddNorm1
   issues are reduced

Status:

- queued

## Current Best Defaults

Keep these unless a step above demonstrates a clear improvement:

- `ln1Replay`: producer-heavy `2/1`
- O-proj real init kernel
- accumulation row-stores:
  - O-proj: current default
  - FFN-down: current default

## Exit Criteria

A change is worth keeping only if:

1. the representative subset still passes
2. the `512seq` memtile control improves materially
3. DDR does not regress enough to erase the memtile gain
