# Programmability And Debugging Notes

This document is the paper-section skeleton for the thesis discussion of engineering difficulty across the three transformer-layer NPU design patterns.

For the living engineering notes about pattern-specific constraints and
follow-on implementation considerations, see
[design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md).
This file should stay thesis-facing.

## Sources

Use these checked-in artifacts as the primary evidence sources:

- [study/programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv)
- manifest-run debug logs written via `debug_log_csv`
- bottleneck summaries from [analyze_design_pattern_bottlenecks.py](/home/cj/iron/iron/applications/transformer_layer/analyze_design_pattern_bottlenecks.py)
- parity summaries from [validate_npu_parity.py](/home/cj/iron/iron/applications/transformer_layer/validate_npu_parity.py)

## Suggested Thesis Table

| Pattern | Challenge | Symptom | Impact On Experiment | Mitigation | Status | Supporting Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| `encoder_pipeline` | Topology-coupled bringup and interpretation | The main study completed with a single dispatch per row, but the dominant stage is still the integrated pipeline itself; the current example row uses topology `seq32_kv64__ps1_ph1_pffn1` on 8 compute tiles at 0.25 utilization. | This pattern gives the cleanest integrated NPU execution path, but performance and support depend directly on topology choices rather than on isolated operator behavior alone. | Keep family-specific topology defaults in the wrapper, log topology metadata in every benchmark row, and interpret results together with dispatch/topology fields rather than latency alone. | Completed with documented topology dependence. | [design_patterns_main_bottlenecks.txt](/home/cj/iron/iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.txt) |
| `gemm_only` | Host-device orchestration remains part of steady-state cost | The study completed with 27 dispatches per row and `npu_gemm` dominates all rows, averaging 94.9% of total time. | The pattern is easier to scale and extend than a tightly integrated pipeline, but its results must be read as "GEMM-heavy offload plus orchestration" rather than as a fully device-native layer mapping. | Hold the shared `Q/K/V/R` boundary fixed, keep explicit per-stage timings, and use query-blocking/partitioning for long-sequence bringup instead of hiding host-side work. | Completed with documented orchestration overhead. | [design_patterns_main_bottlenecks.txt](/home/cj/iron/iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.txt) |
| `operator_runlist` | Runtime lifecycle and full-layer parity are still the hardest engineering surface | The retained study now runs cleanly only through isolated child-process execution, and full layer-local parity remains materially worse than the other patterns (`max_abs_diff=2.25` at `seq_len=64`, `2.484375` at `seq_len=128`). | This pattern demonstrates staged NPU offload without the same topology coupling as `encoder_pipeline`, but it currently carries higher runtime-management overhead and a weaker end-to-end parity story. | Keep benchmark and parity execution isolated in child processes, use component-boundary validation as the preferred correctness method, and treat the current full-layer parity gap as a documented limitation rather than a hidden success. | Working with a documented parity limitation. | [design_patterns_main_parity.csv](/home/cj/iron/iron/applications/transformer_layer/results/design_patterns_main_parity.csv) |

## Suggested Narrative Structure

### 1. Build And Bringup Friction

- compile failures
- unsupported topology or placement cases
- manifest/configuration mistakes that blocked full studies

### 2. Runtime Stability And Orchestration

- startup failures
- retries or reruns
- child-process or subprocess issues
- sequence-length-specific failures

### 3. Pattern-Specific Complexity

- topology selection and interpretation for `encoder_pipeline`
- host/device orchestration surface for `gemm_only`
- dispatch and binary-management overhead for `operator_runlist`

### 4. Mitigations That Became Methodology

- manifest-driven study control
- structured debug logging
- bottleneck summaries
- parity and roofline post-processing
