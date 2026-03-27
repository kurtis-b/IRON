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
- the active studies use synthetic weights and one of two synthetic input boundaries:
  - post-projection `Q/K/V/R`
  - `hidden_states` plus derived residual `R`
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
- [design_patterns_long_seq_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_long_seq_with_projection.json)
- [design_patterns_embedding_scale_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_embedding_scale_with_projection.json)

## Sweep Structure

The checked-in study set now has three sweep styles:

- the retained main sweep over `seq_len=64,128,256,512`
- the retained sensitivity sweep over `seq_len=64,128,256,512,1024,2048`
- the follow-on long-sequence sweep over `seq_len=64..16384` on the retained
  `768/3072/12` and `1024/4096/16` families
- the follow-on embedding-scale sweep over:
  - `baseline_768`
  - `baseline_1024`
  - `dense_8b_class`

The long-sequence study isolates sequence scaling while holding model-family
choice to the retained baseline families. The embedding-scale study changes the
layer family explicitly and records unsupported points instead of silently
dropping them.

## Measurement Protocol

For each case:

1. Build the layer spec for the requested `seq_len`.
2. Materialize deterministic synthetic weights and either:
   - post-projection `Q/K/V/R` inputs, or
   - `hidden_states` inputs plus derived residual `R`,
   from the configured seed.
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
- input-boundary metadata
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

NPU pattern runs use the generic power monitor abstraction in [benchmark_power.py](/home/cj/iron/iron/applications/transformer_layer/benchmark_power.py). The AMD GPU/iGPU path uses [gpu_power.py](/home/cj/iron/iron/applications/transformer_layer/gpu_power.py) and `rocm-smi`.

Power methodology rules:

- for NPU rows, sample package power with `sudo -n turbostat --quiet --Summary --show PkgWatt`
- measure a quiescent package-power baseline immediately before each timed benchmark row
- report pseudo-NPU power as `max(package_power - quiescent_baseline, 0)`
- keep the configured sample interval as a ceiling, then adapt downward for short timed windows so the monitor still collects multiple samples without oversampling long runs
- perform NPU and iGPU power collection on a separate same-workload probe window so the latency loop remains minimally invasive
- allow the minimum probe window to drop below `0.5s`; the current retained floor is `0.25s`
- for iGPU rows, collect average and max power over the timed region with `rocm-smi`
- derive `energy_j` from average power and timed duration
- leave power fields empty only when no backend-specific monitor is active

The retained unattended full-study flow is [run_study_pipeline.py](/home/cj/iron/iron/applications/transformer_layer/run_study_pipeline.py), driven by [full_study_pipeline.json](/home/cj/iron/iron/applications/transformer_layer/study/full_study_pipeline.json). That runner also supports a smoke mode that skips power cycling but still measures NPU pseudo-power and iGPU power.

For unattended thermal reset, the pipeline power-cycles between benchmark-producing
steps only. It supports either one `power_cycle.command` or a structured
`power_cycle.off_command` / `power_cycle.on_command` sequence, then waits for
`k10temp` `Tctl` to recover to within `5%` of the initial captured package
temperature or until the configured timeout expires. The checked-in pipeline
config now points at a systemd service entrypoint:

- `sudo -n systemctl start transformer-layer-power-cycle.service`

The service template and env-file example live under
[systemd/](/home/cj/iron/iron/applications/transformer_layer/systemd).

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
- `avg_host_projection_latency_ms`
- `avg_npu_projection_latency_ms`
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
The same driver also supports the projection-inclusive long-sequence manifest at
[operator_runlist_component_long_seq_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/operator_runlist_component_long_seq_with_projection.json).

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

## Appendix: Result Columns

This appendix maps the shared result schema in
[result_schema.py](/home/cj/iron/iron/applications/transformer_layer/src/result_schema.py)
to the current benchmark implementation.

### Identity And Workload Columns

| Column | Meaning / Source |
| --- | --- |
| `study_id` | Study identifier from the manifest, for example `design_patterns_main` or `design_patterns_embedding_scale`. |
| `study_case_id` | Case identifier from `study_cases` in mixed-shape manifests. |
| `study_case_label` | Human-readable label from `study_cases`. |
| `backend` | `npu` for Ryzen AI pattern rows, `gpu` for the AMD iGPU comparison. |
| `execution_mode` | Pattern or backend mode, for example `encoder_pipeline`, `gemm_only`, `operator_runlist`, or `amd_igpu_reference`. |
| `pattern_label` | Plot-facing label; in the current app this usually matches `execution_mode`. |
| `input_boundary` | `post_projection` or `hidden_states`, from [TransformerLayerSpec](/home/cj/iron/iron/applications/transformer_layer/src/layer_spec.py). |
| `seq_len` | Sequence length from the layer spec. |
| `hidden_size` | Embedding width from the layer spec. |
| `intermediate_size` | FFN hidden width from the layer spec. |
| `num_attention_heads` | Attention head count from the layer spec. |
| `attention_head_size` | Derived from `hidden_size / num_attention_heads`. |
| `batch_size` | Batch size from the layer spec; retained studies use `1`. |
| `dtype` | Tensor dtype from the layer spec; retained studies use `bfloat16`. |
| `use_bias` | Bias toggle from the layer spec; retained studies use `false`. |
| `weights_source` | Weight provenance, currently `synthetic`. |
| `source_model_name` | Imported-model provenance when applicable; empty on the current synthetic studies. |
| `source_layer_index` | Imported-layer provenance when applicable; empty on the current synthetic studies. |

### Sampling And Latency Columns

| Column | Meaning / Computation |
| --- | --- |
| `warmup_runs` | Untimed warmup iterations from the manifest or `sampling_schedule`. |
| `runs_per_sample` | Timed iterations from the manifest or `sampling_schedule`. |
| `measured_inference_count` | Number of timed runs actually recorded for the row. |
| `timed_total_sec` | Sum of timed-run latencies only. Warmups are excluded. |
| `avg_latency_ms` | `(timed_total_sec / measured_inference_count) * 1000`. |
| `compile_setup_time_ms` | Pattern-reported setup / compile / runtime-preparation time. This is tracked separately from steady-state timed latency. |

### Stage-Timing Columns

These are averages over timed runs only. They are emitted when a pattern exposes
`forward_with_stage_timings()`.

| Column | Meaning / Source |
| --- | --- |
| `avg_encoder_pipeline_latency_ms` | Average of `encoder_pipeline_sec` from [pattern_encoder_pipeline.py](/home/cj/iron/iron/applications/transformer_layer/src/pattern_encoder_pipeline.py). |
| `avg_host_projection_latency_ms` | Average of `host_projection_sec`, used by projection-inclusive `encoder_pipeline`. |
| `avg_npu_projection_latency_ms` | Average of `npu_projection_sec`, used by projection-inclusive `gemm_only` and `operator_runlist`. |
| `avg_npu_gemm_latency_ms` | Average of `npu_gemm_sec`, used by `gemm_only`. |
| `avg_operator_runlist_latency_ms` | Average of `operator_runlist_sec`, used by `operator_runlist`. |
| `avg_host_preprocess_latency_ms` | Average of `host_preprocess_sec`, used by `gemm_only`. |
| `avg_host_postprocess_latency_ms` | Average of `host_postprocess_sec`, used by `gemm_only`. |
| `avg_device_sync_latency_ms` | Average of `device_sync_sec`, currently `0.0` for `gemm_only`. |

### NPU Dispatch And Topology Columns

| Column | Meaning / Source |
| --- | --- |
| `npu_dispatch_count` | Pattern-reported dispatch count. `encoder_pipeline` uses `len(runlist)`. `gemm_only` uses projection dispatches plus `5 * query_block_count`. `operator_runlist` uses `len(runlist)` plus projection GEMMs when present. |
| `npu_unique_instruction_binary_count` | Count of distinct `insts.bin` artifacts used by the row. |
| `npu_unique_xclbin_count` | Count of distinct runtime xclbins used by the row. |
| `topology_id` | Encoder-pipeline topology identifier from its topology registry. |
| `topology_family` | Encoder-pipeline family identifier from its topology registry. |
| `parallel_seq` | Encoder-pipeline sequence parallelism. |
| `parallel_heads` | Encoder-pipeline head parallelism. |
| `parallel_ffn` | Encoder-pipeline FFN parallelism. |
| `compute_tile_count` | Encoder-pipeline compute tile count. |
| `compute_tile_utilization_fraction` | Encoder-pipeline topology utilization fraction. |
| `process_model` | `in_process` for normal runs, `child_process` for isolated worker paths such as `operator_runlist`. |

### Status And Failure Columns

| Column | Meaning / Source |
| --- | --- |
| `run_status` | `completed`, `unsupported`, or `failed`. |
| `failure_component` | Normalized subsystem label from exception classification, for example `pattern_surface`, `npu_compile`, or `runtime_numeric`. |
| `failure_category` | Normalized failure bucket, for example `unsupported_pattern_surface` or `unsupported_dma_descriptor_limits`. |
| `failure_message` | Original failure symptom preserved in the result row. |

### Throughput And Roofline Columns

| Column | Meaning / Computation |
| --- | --- |
| `throughput_flops_per_sec` | `estimated_flops_per_inference * measured_inference_count / timed_total_sec`. |
| `estimated_flops_per_inference` | Layer-centric FLOP estimate from [roofline.py](/home/cj/iron/iron/applications/transformer_layer/roofline.py): attention scores GEMM + attention output GEMM + output projection + FFN up + FFN down, plus `3 * Q/K/V projection GEMMs` when `input_boundary=hidden_states`. |
| `estimated_bytes_per_inference` | Layer-centric byte estimate from [roofline.py](/home/cj/iron/iron/applications/transformer_layer/roofline.py). |
| `operational_intensity_flops_per_byte` | `estimated_flops_per_inference / estimated_bytes_per_inference`. |
| `backend_peak_ops_per_sec` | Peak backend throughput loaded from [peak_references.json](/home/cj/iron/iron/applications/transformer_layer/config/peak_references.json) during roofline annotation. |
| `roofline_bound_ops_per_sec` | `min(backend_peak_ops_per_sec, operational_intensity * peak_bytes_per_sec)`. |
| `backend_pct_of_peak` | `throughput_flops_per_sec / backend_peak_ops_per_sec`. |
| `roofline_pct` | `throughput_flops_per_sec / roofline_bound_ops_per_sec`. |

### Power And Energy Columns

| Column | Meaning / Computation |
| --- | --- |
| `power_backend` | Backend used to collect power, e.g. `turbostat_pkgwatt` or `rocm-smi`. |
| `raw_package_avg_power_w` | Raw average package power for NPU rows before subtracting the quiescent baseline. |
| `raw_package_max_power_w` | Raw max package power for NPU rows before subtracting the quiescent baseline. |
| `quiescent_package_power_w` | Per-row idle package-power baseline used for pseudo-NPU power. |
| `avg_power_w` | Average sampled power over the timed region. |
| `max_power_w` | Maximum sampled power over the timed region. |
| `energy_j` | `avg_power_w * timed_total_sec`. |
| `flops_per_joule` | `throughput_flops_per_sec / avg_power_w`. Equivalent to FLOPs per joule. |
| `gflops_per_joule` | `flops_per_joule / 1e9`. |
| `power_sample_count` | Number of power samples collected during the timed region. |

For the current app:

- NPU rows can populate them through the `turbostat_pkgwatt` pseudo-NPU path.
- GPU/iGPU rows populate them when the compare run uses `power_backend=rocm-smi`.

### Parity Output Columns

Parity runs write a separate CSV, not the main benchmark suite. The key metric
columns are:

| Column | Meaning / Computation |
| --- | --- |
| `max_abs_diff` | `max(abs(reference - candidate))` against [ReferenceTransformerLayer](/home/cj/iron/iron/applications/transformer_layer/src/reference_layer.py). |
| `mean_abs_diff` | `mean(abs(reference - candidate))` against the same reference. |

`operator_runlist` parity is measured through the isolated worker path in
[validate_npu_parity.py](/home/cj/iron/iron/applications/transformer_layer/validate_npu_parity.py)
and
[operator_runlist_worker.py](/home/cj/iron/iron/applications/transformer_layer/operator_runlist_worker.py).
