# Benchmarking Methodology

## Scope

This app evaluates one encoder-style transformer layer at a time. The thesis question is not “which full model is fastest,” but “how does workload mapping affect achievable performance, efficiency, bottlenecks, and engineering overhead on Ryzen AI?”

The mainline study compares three NPU design patterns:

- `encoder_pipeline`
- `gemm_only`
- `operator_runlist`

The follow-on iGPU path is a separate end-of-study comparison against the best
completed NPU result per case. It is not mixed into the primary NPU pattern
sweep.

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

## Study Manifest Structure

The benchmark harness accepts either:

- a single `layer_spec`, for the existing one-family studies
- `study_cases`, for mixed-shape studies such as long-sequence or
  embedding-scale follow-ons

Each `study_case` carries:

- `case_id`
- `case_label`
- `layer_spec`
- optional local `seq_lens`
- optional local `execution_modes`

The checked-in manifests now include:

- [design_patterns_main.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_main.json)
- [design_patterns_sensitivity.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_sensitivity.json)
- [design_patterns_long_seq.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_long_seq.json)
- [design_patterns_embedding_scale.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_embedding_scale.json)

## Sweep Structure

The checked-in study set now has three sweep styles:

- the retained main sweep over `seq_len=64,128,256,512`
- the retained sensitivity sweep over `seq_len=64,128,256,512,1024,2048`
- the follow-on long-sequence sweep over `seq_len=64..16384` on the retained
  `768/3072/12` and `1024/4096/16` families
- the follow-on embedding-scale sweep over:
  - `baseline_768`
  - `baseline_1024`
  - `dense_4b_class`
  - `dense_8b_class`

The long-sequence study isolates sequence scaling while holding model-family
choice to the retained baseline families. The embedding-scale study changes the
layer family explicitly and records unsupported points instead of silently
dropping them.

## Measurement Protocol

For each case:

1. Build the layer spec for the requested `seq_len`.
2. Materialize deterministic synthetic weights and post-projection `Q/K/V/R` inputs from the configured seed.
3. Run warmup iterations.
4. Run timed iterations.
5. Write one schema-normalized suite row.

Important measurement rules:

- `warmup_runs` and `runs_per_sample` are recorded per row
- multi-case studies may override those with a per-`seq_len`
  `sampling_schedule`
- `avg_latency_ms` is computed from only the timed runs
- the row also stores per-pattern staged timings when available
- compile/setup time is tracked separately from steady-state timed latency
- execution is single-attempt; this app does not implement retry or recovery logic
- mixed-support studies may continue after a row-level failure and write an
  explicit `failed` or `unsupported` result row instead of aborting the whole
  study

The main CLI for one pattern is [npu_inference.py](/home/cj/iron/iron/applications/transformer_layer/npu_inference.py). The main study harness is [automated_benchmark.py](/home/cj/iron/iron/applications/transformer_layer/automated_benchmark.py).

There is no separate topology-cache warmup command in this app. Pattern compilation and runtime setup happen inside the pattern wrappers and harness.

## Result Schema

All pattern runners write the shared schema in [result_schema.py](/home/cj/iron/iron/applications/transformer_layer/src/result_schema.py).

The schema includes:

- workload metadata
- case metadata for mixed-shape studies
- latency summary
- stage-level latency fields
- dispatch / topology metadata
- FLOP and byte estimates
- roofline annotation fields
- power and energy fields
- run-status and failure metadata for explicit unsupported points

That shared schema is the basis for:

- roofline annotation
- bottleneck analysis
- plotting
- support-matrix summaries
- best-NPU-vs-iGPU comparison

## Power Collection

NPU pattern runs use the generic power monitor abstraction in [benchmark_power.py](/home/cj/iron/iron/applications/transformer_layer/benchmark_power.py). The AMD GPU path uses [gpu_power.py](/home/cj/iron/iron/applications/transformer_layer/gpu_power.py) and `rocm-smi`.

Power methodology rules:

- collect average and max power over the timed region
- derive `energy_j` from average power and timed duration
- leave power fields empty when no backend-specific monitor is active

The GPU/iGPU comparison should use `power_backend=rocm-smi` when energy or
efficiency is part of the figure set.

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
The checked-in config artifact currently serves as a study-local calibrated
reference derived from the benchmark corpus in `results/`. If thesis-final
roofline claims later use a separate hardware-calibration pass, preserve the
updated `source_note` provenance when replacing the checked-in values.

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

Long-sequence `operator_runlist` component studies can also be driven
manifest-first with
[operator_runlist_component_study.py](/home/cj/iron/iron/applications/transformer_layer/operator_runlist_component_study.py),
using
[operator_runlist_component_long_seq.json](/home/cj/iron/iron/applications/transformer_layer/study/operator_runlist_component_long_seq.json).

## Support-Matrix Reporting

Mixed-support studies write explicit support summaries when configured:

- `*_support_matrix.csv`
- `*_support_matrix.txt`

These artifacts are derived from the same benchmark rows as the latency suite,
but they preserve `completed`, `unsupported`, and `failed` outcomes instead of
filtering to performance-ready rows only. This is the mechanism that keeps
`encoder_pipeline` limitations visible in the embedding-scale study.

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

Long-sequence validation constraints and follow-on engineering work are tracked in
[design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md)
rather than in this methodology document. The retained runtime surfaces for
`encoder_pipeline`, `gemm_only`, and `operator_runlist` all now execute
through `seq_len=16384` on the supported `768/3072/12` and `1024/4096/16`
families. On the retained short surfaces, `operator_runlist` full-layer parity
is again in-family with the other patterns. Long-sequence correctness for
`operator_runlist` should still lean on component-boundary validation rather
than full host-materialized parity.

## iGPU Comparison Methodology

The iGPU path is intentionally isolated:

- keep iGPU rows separate from the main NPU suite
- compare the iGPU only against the best completed NPU row per
  `(study_case_id, seq_len)`
- use the embedding-scale NPU suite as the selection source

This avoids turning the thesis app into a generic multi-backend benchmark
surface. The retained follow-on compare driver is
[gpu_compare_best_npu.py](/home/cj/iron/iron/applications/transformer_layer/gpu_compare_best_npu.py),
configured by
[gpu_compare_embedding_igpu.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare_embedding_igpu.json).

## Plotting Method

Final figures are generated by [plot_design_pattern_results.py](/home/cj/iron/iron/applications/transformer_layer/plot_design_pattern_results.py).

Inputs:

- annotated NPU suite CSV
- bottleneck summary CSV
- optional iGPU comparison CSV

Outputs:

- latency, throughput, power, energy-efficiency, and roofline SVGs for the three NPU patterns
- bottleneck SVG for the NPU study
- best-NPU-vs-iGPU SVGs
- an `index.html` bundle for quick browsing

The plotter now supports both:

- `x_axis=seq_len` for the retained main and long-sequence studies
- `x_axis=hidden_size` for the embedding-scale study

The pure-Python validation target [paper_smoke_validation.py](/home/cj/iron/iron/applications/transformer_layer/paper_smoke_validation.py) exercises config parsing, schema writing, roofline annotation, bottleneck analysis, and plotting without hardware.
