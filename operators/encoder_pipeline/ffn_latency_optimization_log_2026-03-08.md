# encoder_pipeline FFN Latency Optimization Log

Date: 2026-03-08

## Scope

Focused options to reduce `ffn_up` and `ffn_down` stage latency in `encoder_pipeline`.

## Current stage bottleneck context

- Recent memtile stage profiling indicates:
  - `ffn_up` is the bottleneck stage.
  - `ffn_down` is the next largest stage.

## Candidate optimizations

1. Add down-proj init matmul and remove zero-priming pass.
   - Current down core zero-primes `of_new_acc` before compute:
     - `operators/encoder_pipeline/design.py` (`core_fn_ffn_down_proj`, around zero-priming loop).
   - Proposed change:
     - Introduce `ffn_matmul_init_*_down_proj` and use it on first group.
     - Use `ffn_matmul_with_acc_*_down_proj` for remaining groups.
   - Expected impact:
     - Lower `ffn_down` latency by removing one full priming pass and reducing memory traffic.

2. Enable FFN-up init matmul for all branch counts by default.
   - Current default:
     - `ffn_up_use_init_matmul_default = effective_ffn_branches <= 2`.
   - Proposed change:
     - Default to init-matmul path regardless of branch count.
   - Expected impact:
     - Reduce `ffn_up` per-group overhead from explicit zero-on-first-acc behavior.

3. Shift replay responsibility from FFN-down to LN2-side replay FIFO where feasible.
   - Current behavior:
     - Single-output topologies default `emit_ln2_replay_from_down=True`.
   - Proposed change:
     - Prefer LN2 replay FIFO path (`emit_ln2_replay_from_down=False`) in resource-feasible mappings.
   - Expected impact:
     - Reduce extra replay-pass pressure in final FFN-down branch.

4. Relax FFN weight fill wait policy in runtime scheduling.
   - Current behavior:
     - Tail IO defaults to strict waits.
   - Proposed change:
     - Use relaxed FFN-weight waits (`relax_ffn_weights`) for stable topologies.
   - Expected impact:
     - Improve overlap of compute and DMA; reduce stall time in `ffn_up/down`.

5. Reduce B-weight DMA fragmentation for low-pressure topologies.
   - Current behavior:
     - Group splitting caps DMA per transfer (`ENCODER_B_WEIGHT_FILL_MAX_GROUPS_PER_DMA` default uses `min(...,22)`).
   - Proposed change:
     - Increase cap in topologies where channel/BD pressure is not the limiter.
   - Expected impact:
     - Fewer DMA submissions and lower runtime scheduling overhead.

6. Note on LN1 broadcast repeats.
   - Current behavior:
     - LN1 broadcast groups already use `max(ffn_col_group_counts)`.
   - Conclusion:
     - Further broadcast-repeat reductions are limited unless branch/group mapping is changed.

## Implemented in this pass (1 + 2)

1. Implemented down-proj init kernel path and removed zero-priming pass.
   - Added kernel symbol:
     - `ffn_matmul_init_bf16_bf16_down_proj` in `aie_kernels/aie2p/encoder.cc`.
   - Wired new kernel in encoder design:
     - Added `ffn_matmul_init_kernel_down_proj`.
     - Updated `core_fn_ffn_down_proj`:
       - First group now uses `matmul_init`.
       - Remaining groups use `matmul_with_acc`.
       - Removed pre-loop zero seeding of `of_new_acc`.
   - Note:
     - Initial implementation used an `scf.if` per-group branch and triggered
       objectfifo acquire/release verification failure.
     - Final implementation uses explicit first-group + remaining-groups loops,
       which compiles and runs.

2. Enabled FFN-up init matmul by default for all branch counts.
   - Changed default from:
     - `effective_ffn_branches <= 2`
   - To:
     - `True`
   - Override knob remains available:
     - `ENCODER_FFN_UP_USE_INIT_MATMUL`

## Source pointers

- `operators/encoder_pipeline/design.py`
  - `core_fn_ffn_up_proj`
  - `core_fn_ffn_down_proj`
  - FFN-up worker default selection (`ffn_up_use_init_matmul_default`)
  - FFN-down worker kernel wiring (`ffn_matmul_init_kernel_down_proj`)
  - Replay mode toggle (`emit_ln2_replay_from_down`)
  - Runtime wait policy and B-weight fill splitting
- `aie_kernels/aie2p/encoder.cc`
  - FFN kernels:
    - `ffn_matmul_init_bf16_bf16_up_proj`
    - `ffn_matmul_init_bf16_bf16_down_proj` (added)
    - `ffn_matmul_bf16_bf16_up_proj`
    - `ffn_matmul_with_acc_bf16_bf16_down_proj`

## Status

- Code changes for items 1 and 2 are implemented.

## Implemented next (3)

3. Switched default replay source to LN2-side replay FIFO.
   - Changed default:
     - `emit_ln2_replay_from_down_default = False`
   - File:
     - `operators/encoder_pipeline/design.py`
   - Effect:
     - Single-output topologies now default to one FFN-down output pass and
       LN2-side replay buffering for AddNorm2 two-pass behavior.
     - Existing override remains available via `ENCODER_EMIT_LN2_REPLAY_FROM_DOWN`.

## Validation

Targeted regression checks (after clean build each run):

- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_1pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_ddr"`
  - Result: `5 passed`

Post-change stage profile (memtile, case:
`64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_1pheads_1pffn_8pacc_1opg`,
`warmup=1`, `timed=3`):

- `full`: `3528.61 us`
- `ffn_up`: `2321.10 us` (bottleneck)
- `ffn_down`: `1916.10 us`
- `addnorm2`: `1464.37 us`
- `mha`: `770.14 us`
- `addnorm1`: `1418.62 us`

Comparison vs prior recorded profile for same case:

- `ffn_up`: `2468.69 -> 2321.10 us` (`-5.98%`)
- `ffn_down`: `1947.09 -> 1916.10 us` (`-1.59%`)
- `full`: `3578.03 -> 3528.61 us` (`-1.38%`)

## Validation after optimization 3

Targeted regression checks (after clean build each run):

- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_1pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_ddr"`
  - Result: `5 passed`

Stage profile (memtile, same case, `warmup=1`, `timed=3`):

- `full`: `3495.84 us`
- `ffn_up`: `2461.65 us` (bottleneck)
- `ffn_down`: `1912.00 us`
- `addnorm2`: `1452.20 us`
- `mha`: `769.86 us`
- `addnorm1`: `1289.13 us`

Delta vs previous (after 1+2) profile:

- `full`: `3528.61 -> 3495.84 us` (`-0.93%`)
- `ffn_up`: `2321.10 -> 2461.65 us` (`+6.06%`)
- `ffn_down`: `1916.10 -> 1912.00 us` (`-0.21%`)

## Implemented next (4)

4. Relaxed FFN-weight wait policy in default runtime scheduling.
   - Updated default runtime policy:
     - `runtime_tail_wait_mode: "strict" -> "relax_ffn_weights"`
   - File:
     - `operators/encoder_pipeline/design.py`
   - Behavior:
     - FFN-weight fills are no longer strictly waited by default.
     - Residual/output waits remain serialized unless explicitly relaxed further.

## Validation after optimization 4

Targeted regression checks (after clean build each run):

- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_1pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_memtile"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "encoder_..._1pheads_2pffn_8pacc_1opg and lnstage_ddr"`
  - Result: `5 passed`

Stage profile (memtile, same case, `warmup=1`, `timed=3`):

- `full`: `3541.64 us`
- `ffn_up`: `2282.51 us` (bottleneck)
- `ffn_down`: `2065.99 us`
- `addnorm2`: `1467.18 us`
- `mha`: `802.29 us`
- `addnorm1`: `1277.89 us`

Delta vs previous (after 3) profile:

- `full`: `3495.84 -> 3541.64 us` (`+1.31%`)
- `ffn_up`: `2461.65 -> 2282.51 us` (`-7.28%`)
- `ffn_down`: `1912.00 -> 2065.99 us` (`+8.05%`)

Observation:

- This change improved `ffn_up` latency, but increased `ffn_down` and slightly
  regressed full latency in this measurement.

## Implemented next (5)

5. Reduced B-weight DMA fragmentation default for low-pressure topologies.
   - Updated runtime default split cap selection for B-weight fills:
     - Existing baseline cap: `min(max(ffn_col_group_counts), 22)`
     - New low-pressure default (non-split, `parallel_heads <= 2`, `effective_ffn_branches <= 2`):
       `max(ffn_col_group_counts)` (effectively no dim0 split)
   - File:
     - `operators/encoder_pipeline/design.py`
   - Behavior:
     - Keeps existing behavior for wider/high-pressure mappings.
     - Reduces `rt.fill` fragmentation overhead for small branch/head counts.

## Validation after optimization 5

Targeted regression checks (after clean build each run):

- `pytest operators/encoder_pipeline/test.py -q -k "test_encoder_pipeline and not stage_profile and lnstage_memtile and encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_1pheads_1pffn_8pacc_1opg"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "test_encoder_pipeline and not stage_profile and lnstage_memtile and 1pheads_2pffn_8pacc_1opg"`
  - Result: `5 passed`
- `pytest operators/encoder_pipeline/test.py -q -k "test_encoder_pipeline and not stage_profile and lnstage_ddr and 1pheads_2pffn_8pacc_1opg"`
  - Result: `5 passed`

Stage profile (memtile, same case as above, `warmup=1`, `timed=3`):

- `full`: `3672.71 us`
- `ffn_up`: `2303.46 us` (bottleneck)
- `ffn_down`: `1922.54 us`
- `addnorm2`: `1455.36 us`
- `mha`: `989.08 us`
- `addnorm1`: `1493.96 us`

Delta vs previous (after 4) profile:

- `full`: `3541.64 -> 3672.71 us` (`+3.70%`)
- `ffn_up`: `2282.51 -> 2303.46 us` (`+0.92%`)
- `ffn_down`: `2065.99 -> 1922.54 us` (`-6.94%`)
- `addnorm2`: `1467.18 -> 1455.36 us` (`-0.81%`)
- `mha`: `802.29 -> 989.08 us` (`+23.28%`)
- `addnorm1`: `1277.89 -> 1493.96 us` (`+16.91%`)

Observation:

- The change improved `ffn_down`, but end-to-end latency regressed in this run
  due to increased non-FFN stage times.
