# Benchmarking Methodology

## Scope

This app evaluates one encoder-style transformer layer at a time. The thesis question is not “which full model is fastest,” but “how does workload mapping affect achievable performance, efficiency, bottlenecks, and engineering overhead on Ryzen AI?”

The mainline study compares three NPU design patterns:

- `encoder_pipeline`
- `gemm_only`
- `operator_runlist`

The AMD GPU path is a separate end-of-study comparison against the best NPU result. It is not mixed into the primary NPU pattern sweep.

## Exact Workload Definition

The canonical workload contract is defined by [TransformerLayerSpec](/home/cj/iron/iron/applications/transformer_layer/src/layer_spec.py):

- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `batch_size`
- `seq_len`
- `dtype`
- `activation`
- `use_bias`
- `layer_norm_eps`
- `attention_mask_mode`
- `weights_source`
- `source_model_name`
- `source_layer_index`

Current branch assumptions:

- batch size is `1`
- `attention_mask_mode` is `none`
- `activation` is `gelu`
- the active study uses synthetic weights and synthetic post-projection `Q/K/V/R` inputs
- imported weights are represented in the schema but no import CLI is wired yet

Pattern-specific engineering tradeoffs are documented separately in
[design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md).
This methodology document is only about workload definition, measurement, and
analysis flow.

The default synthetic layer is BERT-shaped:

- `hidden_size=768`
- `intermediate_size=3072`
- `num_attention_heads=12`
- `dtype=bfloat16`
- `use_bias=false`

## Sequence-Length Sweep

The checked-in manifests define the intended sweep:

- [design_patterns_main.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_main.json)
- [design_patterns_sensitivity.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_sensitivity.json)

The normal sweep varies `seq_len` while keeping the rest of the layer spec fixed. That isolates sequence scaling from model-family differences.

## Measurement Protocol

For each case:

1. Build the layer spec for the requested `seq_len`.
2. Materialize deterministic synthetic weights and post-projection `Q/K/V/R` inputs from the configured seed.
3. Run warmup iterations.
4. Run timed iterations.
5. Write one schema-normalized suite row.

Important measurement rules:

- `warmup_runs` and `runs_per_sample` are recorded per row
- `avg_latency_ms` is computed from only the timed runs
- the row also stores per-pattern staged timings when available
- compile/setup time is tracked separately from steady-state timed latency
- execution is single-attempt; this app does not implement retry or recovery logic

The main CLI for one pattern is [npu_inference.py](/home/cj/iron/iron/applications/transformer_layer/npu_inference.py). The main study harness is [automated_benchmark.py](/home/cj/iron/iron/applications/transformer_layer/automated_benchmark.py).

There is no separate topology-cache warmup command in this app. Pattern compilation and runtime setup happen inside the pattern wrappers and harness.

## Result Schema

All pattern runners write the shared schema in [result_schema.py](/home/cj/iron/iron/applications/transformer_layer/src/result_schema.py).

The schema includes:

- workload metadata
- latency summary
- stage-level latency fields
- dispatch / topology metadata
- FLOP and byte estimates
- roofline annotation fields
- power and energy fields

That shared schema is the basis for:

- roofline annotation
- bottleneck analysis
- plotting
- NPU-vs-AMD-GPU comparison

## Power Collection

NPU pattern runs use the generic power monitor abstraction in [benchmark_power.py](/home/cj/iron/iron/applications/transformer_layer/benchmark_power.py). The AMD GPU path uses [gpu_power.py](/home/cj/iron/iron/applications/transformer_layer/gpu_power.py) and `rocm-smi`.

Power methodology rules:

- collect average and max power over the timed region
- derive `energy_j` from average power and timed duration
- leave power fields empty when no backend-specific monitor is active

The AMD GPU comparison should use `power_backend=rocm-smi` when energy or efficiency is part of the figure set.

## Roofline Method

The roofline path is layer-centric rather than model-centric.

Steps:

1. Create or update backend peak references with [calibrate_backend_peaks.py](/home/cj/iron/iron/applications/transformer_layer/calibrate_backend_peaks.py).
2. Run the suite.
3. Annotate suite rows with [annotate_roofline.py](/home/cj/iron/iron/applications/transformer_layer/annotate_roofline.py).

The annotation writes:

- `backend_peak_ops_per_sec`
- `roofline_bound_ops_per_sec`
- `backend_pct_of_peak`
- `roofline_pct`

Peak artifacts live in the multi-backend format defined by [peak_reference.py](/home/cj/iron/iron/applications/transformer_layer/peak_reference.py).

## Bottleneck Logging Method

Each NPU row may include pattern-specific stage timings such as:

- `avg_encoder_pipeline_latency_ms`
- `avg_npu_gemm_latency_ms`
- `avg_operator_runlist_latency_ms`
- `avg_host_preprocess_latency_ms`
- `avg_host_postprocess_latency_ms`
- `avg_device_sync_latency_ms`

Post-process the annotated suite with [analyze_design_pattern_bottlenecks.py](/home/cj/iron/iron/applications/transformer_layer/analyze_design_pattern_bottlenecks.py) to produce:

- row-level bottleneck CSV
- aggregate JSON summary
- human-readable text summary

These outputs support the thesis discussion of where each design pattern spends time and why.

`operator_runlist` also has a dedicated repeated-run stability check via
[validate_operator_runlist_stability.py](/home/cj/iron/iron/applications/transformer_layer/validate_operator_runlist_stability.py).
That validation is separate from the benchmark harness and is used to confirm
that the individual NPU operators used by the runlist complete repeatedly
without relying on retries. It also supports component-boundary checks, where
only the inputs and output of the selected operator are materialized for parity.

## Programmability and Debugging Log

Study automation can append structured events to a debug CSV via [debug_log.py](/home/cj/iron/iron/applications/transformer_layer/debug_log.py).

This log is used for thesis-facing engineering notes such as:

- manifest and path-resolution failures
- unsupported topology or placement cases
- instruction-binary reuse vs generation behavior
- benchmark-case completion and failure events

The curated thesis artifact lives at [programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv).

## Parity Validation

Parity is layer-local.

Steps:

1. Build a synthetic reference layer with [reference_layer.py](/home/cj/iron/iron/applications/transformer_layer/src/reference_layer.py).
2. Run the requested NPU pattern on the same synthetic weights and synthetic `Q/K/V/R` inputs.
3. Record `max_abs_diff` and `mean_abs_diff`.

Use [validate_npu_parity.py](/home/cj/iron/iron/applications/transformer_layer/validate_npu_parity.py) for this workflow.

Parity should be interpreted as:

- a correctness guard for pattern bringup
- not a benchmark result
- separate from the roofline and bottleneck analysis

Long-sequence validation constraints and follow-on engineering work, including
stage-level parity for `operator_runlist`, are tracked in
[design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md)
rather than in this methodology document.

## AMD GPU Comparison Methodology

The AMD GPU path is intentionally isolated:

- run [gpu_inference.py](/home/cj/iron/iron/applications/transformer_layer/gpu_inference.py) separately
- keep GPU rows separate from the main NPU suite
- compare the GPU only against the best NPU row per sequence length

This avoids turning the thesis app into a generic multi-backend benchmark surface. The dedicated manifest is [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json).

## Plotting Method

Final figures are generated by [plot_design_pattern_results.py](/home/cj/iron/iron/applications/transformer_layer/plot_design_pattern_results.py).

Inputs:

- annotated NPU suite CSV
- bottleneck summary CSV
- isolated AMD GPU comparison CSV

Outputs:

- latency, throughput, power, energy-efficiency, and roofline SVGs for the three NPU patterns
- bottleneck SVG for the NPU study
- best-NPU-vs-AMD-GPU SVGs
- an `index.html` bundle for quick browsing

The pure-Python validation target [paper_smoke_validation.py](/home/cj/iron/iron/applications/transformer_layer/paper_smoke_validation.py) exercises config parsing, schema writing, roofline annotation, bottleneck analysis, and plotting without hardware.
