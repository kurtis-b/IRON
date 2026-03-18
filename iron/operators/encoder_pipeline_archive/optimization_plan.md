# encoder_pipeline Optimization Plan

## Goal

Improve latency on the currently passing topologies without reopening the old
memtile-BD failures.

## Priority order

### 1. AddNorm1 cost

Why first:

- `AddNorm1` is the slowest isolated stage on the key passing cases
- on the higher-`pacc` passing memtile case it is the dominant bottleneck by a
  wide margin

Target directions:

- reduce replay traffic around AddNorm1
- keep the current LN1 replay row-store asymmetry:
  - memtile `2/1`
  - ddr `3/1`
- move more partial-stat work upstream where possible
- keep full-row layer-norm semantics unchanged

### 2. Runtime overlap

Why second:

- the `512seq` memtile and fast DDR cases still have a large gap between full
  runtime and the max isolated stage

Target directions:

- audit B-weight fill ordering and waits
- reduce conservative tail synchronization
- overlap data movement with compute more aggressively
- keep the current DDR-only residual-fill wait relaxation
- avoid naive wait removal on `K/V/W_O` head-block fills; that path was tested
  and broke correctness on the memtile control case

### 3. O-proj accumulation core residency

Why third:

- deeper accumulation-side buffering is blocked by O-proj core L1
- this is the next best candidate for exploiting row-store better

Target directions:

- shorten `memOW*_cons` lifetime overlap
- keep the real O-proj init kernel
- only promote deeper/asymmetric buffering if it wins in memtile mode without
  hurting DDR

### 4. FFN-down accumulation core residency

Why fourth:

- deeper accumulation-side buffering is also blocked by FFN-down core L1
- this path is currently less promising than O-proj

Target directions:

- shorten `memBDown*_cons` lifetime overlap on stage/root branches
- avoid global shallow-weight defaults

### 5. Tail stream-count reduction

Why fifth:

- the remaining hard failures on aggressive `64q / 48e / 16pacc` layouts are
  mostly tail-side output-channel pressure

Target directions:

- remove tail streams structurally instead of only remapping memtile columns
- keep direct FFN-down replay where it actually reduces tail load

## Explicit non-goals for now

- more generic row-store buffer-count tuning
- more memtile-column remapping by itself
- reviving unsupported uneven FFN partitions

Those directions have already been explored enough to show they are not the
highest-value path.

## Required regression set for every optimization pass

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- `lnstage_ddr-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
- `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`

## Exit criteria

This optimization pass should be considered successful only if:

1. full latency improves on one of the representative passing cases
2. the required regression set still passes
3. no old memtile-BD/resource failure is reintroduced
