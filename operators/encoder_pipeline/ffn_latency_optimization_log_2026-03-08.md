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
