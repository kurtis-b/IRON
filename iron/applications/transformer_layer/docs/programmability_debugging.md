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
| `encoder_pipeline` | TODO | TODO | TODO | TODO | TODO | TODO |
| `gemm_only` | TODO | TODO | TODO | TODO | TODO | TODO |
| `operator_runlist` | TODO | TODO | TODO | TODO | TODO | TODO |

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
