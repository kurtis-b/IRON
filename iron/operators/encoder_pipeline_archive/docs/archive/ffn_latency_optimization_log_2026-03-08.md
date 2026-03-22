# encoder_pipeline FFN Latency Optimization Summary

Date: 2026-03-08

## Goal

- Reduce latency in FFN-up and FFN-down stages without changing correctness thresholds.

## Implemented changes

- FFN-down init-matmul path: first accumulation group uses init-matmul, remaining groups use with-acc matmul.
- Removed separate zero-priming pass in FFN-down core loop.
- FFN-up init-matmul path is enabled by default.
- Replay default prefers LN2-side replay buffering over FFN-down replay pass where supported.
- Runtime tail wait policy was tuned away from strict-all waits toward relaxed FFN-weight waits by default.
- B-weight fill fragmentation was reduced for low-pressure topologies.

## Outcome

- FFN-up remains the dominant stage in typical memtile profiling.
- Some stage latencies improved, but end-to-end impact is topology-dependent.
- Performance changes are topology-sensitive; regression checks are required per topology.

## Validation

- Always clean before profiling: `rm -rf ./build`
- Run stage profile: `pytest operators/encoder_pipeline/test.py -q -k "stage_profile and lnstage_memtile"`
- Or run debug sweep: `python operators/encoder_pipeline/profile_debug_modes.py --design memtile --clean-build`

## Practical takeaway

- Keep optimization changes that preserve correctness and improve target-topology latency.
- For new changes, re-check full latency plus `debug=3` (FFN-up), `debug=4` (FFN-down), `debug=5` (AddNorm2).
- Re-run pass/fail in both `memtile` and `ddr` staging modes.

## 2026-03-10 AddNorm1 follow-up

- Added two LN micro-optimizations in `aie_kernels/aie2p/encoder.cc`:
  - hoisted per-row `mean` / `variance` / `inv_std` out of inner loops
  - accumulated LN stats in registers before storing back to `sum` / `sumsq`
- Removed the extra LN1 norm scratch handoff in `design.py` so LN1 norm writes directly into replay or the next FIFO.
- Replaced the failed `memref.subview`-based packed-stats experiment with explicit whole-packet pack/unpack kernels in `aie_kernels/generic/convert_copy.cc`.

### Validation case

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`

### Measured Results For That Run

- Clean full run passes with `228` mismatches out of `1966` allowed.
- Full latency is now about `16065 us`.
- Filtered stage timings:
  - `mha`: `7442 us`
  - `ffn_up`: `7340 us`
  - `ffn_down`: `6647 us`
  - `addnorm2`: `7170 us`
  - `addnorm1`: `12207 us`
  - `addnorm1_stats`: `12797 us`
  - `addnorm1_post`: `7740 us`

### Conclusion

- `AddNorm1` remains the bottleneck.
- The dominant portion is still the LN1 stats path, not LN1 post.
- The packetized LN1-stats path preserves correctness and slightly improves end-to-end latency, but it does not reduce isolated `addnorm1_stats` time.
- The next useful optimization should avoid BF16 stats packet conversion overhead and instead reduce LN1 pass-1 work more directly.

### Later Baseline Check

- After removing the abandoned LN1 replay experiment, the same control case
  still passes.
- Latest clean control rerun:
  - about `16300-16400 us`
  - `229 / 1966` mismatches
