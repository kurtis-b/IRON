# Transformer Layer Design-Pattern Study

`transformer_layer` is a thesis-focused application for studying how a single encoder-style transformer layer maps onto Ryzen AI. The main study compares three NPU design patterns:

- `encoder_pipeline`
- `gemm_only`
- `operator_runlist`

The app is intentionally layer-centric rather than model-centric. It does not benchmark embeddings, tokenization, or full-sequence application behavior. The point is to isolate how workload mapping changes latency, efficiency, bottlenecks, and programmability on a resource-constrained NPU.

GPU comparisons are separate from the main NPU study. The retained follow-on
comparison is a best-NPU-vs-AMD-iGPU workflow for the embedding-scale study,
after the NPU study surface is already stable.

## Workload Contract

The default study uses one synthetic transformer layer with:

- batch size `1`
- no attention mask
- `bfloat16`
- synthetic inputs at either:
  - post-projection `Q/K/V/R`
  - `hidden_states` with derived residual `R`
- synthetic weights
- one layer-local CPU reference path for parity checks

The canonical layer spec lives in [layer_spec.py](/home/cj/iron/iron/applications/transformer_layer/src/layer_spec.py). Imported weights are represented in the schema, but this branch currently runs synthetic weights only. There is no `import_layer_weights.py` command in the app yet.

The retained baseline studies start from supplied `Q`, `K`, `V`, and residual
`R`. The projection-inclusive follow-on studies start from `hidden_states`:

- `encoder_pipeline` performs `Q/K/V` projection on the host
- `gemm_only` offloads `Q/K/V` projection GEMMs to the NPU
- `operator_runlist` offloads `Q/K/V` projection GEMM stages to the NPU

For the engineering tradeoffs that differ across `encoder_pipeline`,
`gemm_only`, and `operator_runlist`, see
[design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md).

## Environment

All commands below assume:

```bash
source /opt/xilinx/xrt/setup.sh
source ./ironenv/bin/activate
```

## Repro Workflow

Pure-Python smoke validation, no hardware required:

```bash
python -m iron.applications.transformer_layer.paper_smoke_validation \
  --output-dir iron/applications/transformer_layer/results/paper_smoke_validation
```

Pure-Python test suite:

```bash
python -m pytest -q iron/applications/transformer_layer
```

One-shot NPU smoke:

```bash
python iron/applications/transformer_layer/npu_inference.py \
  --execution-mode encoder_pipeline \
  --seq-len 64 \
  --input-boundary hidden_states \
  --warmup-runs 1 \
  --runs-per-sample 5 \
  --enable-measurement-log \
  --output-csv iron/applications/transformer_layer/results/npu_smoke_seq64.csv
```

Measurement-audit sidecar logging is available, but it is now opt-in.
If you pass `--enable-measurement-log`, the command above also writes:

- `iron/applications/transformer_layer/results/npu_smoke_seq64_measurements.jsonl`

When enabled, each result row carries:

- `measurement_log_path`
- `measurement_session_id`

That sidecar log records every warmup run, every timed run, every separate
power-probe run, the NPU quiescent-baseline measurement when active, and the
final summary row payload. Timing entries include explicit `start_time_utc`,
`end_time_utc`, and text describing the timer start and end points so the
placement can be checked by hand.

Main NPU sweep:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json
```

Long-sequence NPU sweep through `seq_len=16384`:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_long_seq.json
```

Projection-inclusive long-sequence NPU sweep:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_long_seq_with_projection.json
```

Embedding-scale NPU sweep through `dense_8b_class`:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_embedding_scale.json
```

Projection-inclusive embedding-scale NPU sweep:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_embedding_scale_with_projection.json
```

There is no separate topology-cache warmup step in this app. Each pattern owns its own compilation/runtime setup inside the study harness.

Parity validation:

```bash
python iron/applications/transformer_layer/validate_npu_parity.py \
  --execution-mode encoder_pipeline \
  --seq-lens 64,128 \
  --output-csv iron/applications/transformer_layer/results/parity_encoder_pipeline.csv
```

Operator-runlist stability validation:

```bash
python iron/applications/transformer_layer/validate_operator_runlist_stability.py \
  --seq-len 64 \
  --repeats 3 \
  --output-csv iron/applications/transformer_layer/results/operator_runlist_stability_seq64.csv
```

Component-boundary operator-runlist validation:

```bash
python iron/applications/transformer_layer/validate_operator_runlist_stability.py \
  --seq-len 64 \
  --targets attn_scores,attn_softmax,ln2 \
  --components-only \
  --output-csv iron/applications/transformer_layer/results/operator_runlist_components_seq64.csv
```

Manifest-driven long-sequence component-boundary validation:

```bash
python iron/applications/transformer_layer/operator_runlist_component_study.py \
  --config iron/applications/transformer_layer/study/operator_runlist_component_long_seq.json
```

Projection-inclusive long-sequence component-boundary validation:

```bash
python iron/applications/transformer_layer/operator_runlist_component_study.py \
  --config iron/applications/transformer_layer/study/operator_runlist_component_long_seq_with_projection.json
```

The retained runtime surface now executes through `seq_len=16384` on the
supported `768/3072/12` and `1024/4096/16` families for all three NPU
patterns. The implementation route is different in each case:

- `encoder_pipeline` uses family-specific topology defaults
- `gemm_only` uses long attention-score partitioning and query-blocked
  execution
- `operator_runlist` uses query-blocked execution with component-boundary
  validation as the preferred long-sequence correctness check

For those long-sequence cases, prefer component-boundary validation over full
host-materialized parity when checking `operator_runlist`.

Peak-reference artifact generation:

```bash
python iron/applications/transformer_layer/calibrate_backend_peaks.py \
  --backend npu \
  --peak-ops-per-sec 1.0 \
  --peak-bytes-per-sec 1.0 \
  --output iron/applications/transformer_layer/config/peak_references.json \
  --append
```

The checked-in [peak_references.json](/home/cj/iron/iron/applications/transformer_layer/config/peak_references.json)
now records study-local calibrated backend references derived from the current
checked-in benchmark artifacts. Those values are suitable for branch-local
roofline analysis; if a separate thesis-final hardware calibration pass is
performed later, preserve the updated `source_note` provenance when replacing
them.

Roofline annotation:

```bash
python -m iron.applications.transformer_layer.annotate_roofline \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main.csv \
  --peak-reference iron/applications/transformer_layer/config/peak_references.json \
  --output-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv
```

Bottleneck analysis:

```bash
python -m iron.applications.transformer_layer.analyze_design_pattern_bottlenecks \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv \
  --summary-csv iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.csv \
  --summary-json iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.json \
  --summary-text iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.txt
```

Programmability/debug logging during a study run:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json \
  --debug-log-csv iron/applications/transformer_layer/results/design_patterns_main_debug_log.csv
```

Measurement-audit logging is separate from `debug_log_csv`. The debug log tracks
study orchestration and failures. The measurement-audit sidecar tracks raw
per-run timing and power measurements.

Isolated ROCm GPU comparison:

```bash
python iron/applications/transformer_layer/gpu_inference.py \
  --seq-len 64 \
  --device cuda:0 \
  --power-backend rocm-smi \
  --output-csv iron/applications/transformer_layer/results/gpu_compare_amd.csv
```

Legacy single-case AMD GPU comparison:

```bash
python iron/applications/transformer_layer/gpu_inference.py \
  --seq-len 128 \
  --device cuda:0 \
  --power-backend rocm-smi \
  --output-csv iron/applications/transformer_layer/results/gpu_compare_amd_seq128.csv
```

The older single-case AMD GPU comparison manifest lives at
[gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json).
The retained follow-on study compare is the best-NPU-vs-iGPU workflow shown
below.

Best-NPU-vs-iGPU comparison for the embedding-scale study:

```bash
python iron/applications/transformer_layer/gpu_compare_best_npu.py \
  --config iron/applications/transformer_layer/study/gpu_compare_embedding_igpu.json
```

Full unattended study pipeline:

```bash
python iron/applications/transformer_layer/run_study_pipeline.py \
  --config iron/applications/transformer_layer/study/full_study_pipeline.json
```

The pipeline runner power-cycles only between benchmark-producing steps, not
between individual rows. The checked-in config supports either:

- one `power_cycle.command`
- or a structured `power_cycle.off_command` / `power_cycle.on_command` pair

The checked-in pipeline config is now wired for a systemd-backed helper:

- `power_cycle.command = ["sudo", "-n", "systemctl", "start", "transformer-layer-power-cycle.service"]`

It still leaves `power_cycle.enabled=false` until the service is installed and
the real environment-specific off/on commands are configured.

Quick end-to-end smoke run without power cycling:

```bash
python iron/applications/transformer_layer/run_study_pipeline.py \
  --config iron/applications/transformer_layer/study/full_study_pipeline.json \
  --smoke \
  --skip-power-cycle \
  --warmup-runs 0 \
  --runs-per-sample 1 \
  --output-root /tmp/transformer_layer_pipeline_smoke
```

NPU power sampling uses `turbostat` package power with per-row quiescent
baseline subtraction. The runtime chooses an adaptive sample interval: the
configured interval is treated as a ceiling, and short timed windows sample
faster down to a low floor so the run still produces multiple samples without
sampling aggressively during long runs. Power is measured on a separate
same-workload probe window so the latency loop stays minimally invasive. The
minimum power-probe duration is no longer pinned to `0.5s`; the current floor
is `0.25s`. The iGPU compare continues to use `rocm-smi`.

Install the power-cycle helper service:

```bash
sudo cp iron/applications/transformer_layer/systemd/transformer-layer-power-cycle.service /etc/systemd/system/
sudo cp iron/applications/transformer_layer/systemd/transformer-layer-power-cycle.env.example /etc/transformer-layer-power-cycle.env
sudoedit /etc/transformer-layer-power-cycle.env
sudo systemctl daemon-reload
sudo systemctl start transformer-layer-power-cycle.service
```

After the service works on the host, set `"power_cycle.enabled": true` in
[full_study_pipeline.json](/home/cj/iron/iron/applications/transformer_layer/study/full_study_pipeline.json).

Thesis plotting:

```bash
python -m iron.applications.transformer_layer.plot_design_pattern_results \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv \
  --bottleneck-csv iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.csv \
  --gpu-compare-csv iron/applications/transformer_layer/results/gpu_compare_amd.csv \
  --output-dir iron/applications/transformer_layer/results/plots/design_patterns_main
```

Embedding-scale plotting with `hidden_size` on the x-axis:

```bash
python -m iron.applications.transformer_layer.plot_design_pattern_results \
  --input-csv iron/applications/transformer_layer/results/design_patterns_embedding_scale_annotated.csv \
  --bottleneck-csv iron/applications/transformer_layer/results/design_patterns_embedding_scale_bottlenecks.csv \
  --gpu-compare-csv iron/applications/transformer_layer/results/gpu_compare_embedding_igpu.csv \
  --x-axis hidden_size \
  --output-dir iron/applications/transformer_layer/results/plots/design_patterns_embedding_scale
```

Unattended job command preview:

```bash
python iron/applications/transformer_layer/run_automated_benchmark_job.py \
  iron/applications/transformer_layer/systemd/benchmark_job.example.json \
  --print-command
```

## Study Artifacts

Checked-in manifests and scaffolds:

- [design_patterns_main.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_main.json)
- [design_patterns_sensitivity.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_sensitivity.json)
- [design_patterns_long_seq.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_long_seq.json)
- [design_patterns_embedding_scale.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_embedding_scale.json)
- [design_patterns_long_seq_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_long_seq_with_projection.json)
- [design_patterns_embedding_scale_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/design_patterns_embedding_scale_with_projection.json)
- [operator_runlist_component_long_seq.json](/home/cj/iron/iron/applications/transformer_layer/study/operator_runlist_component_long_seq.json)
- [operator_runlist_component_long_seq_with_projection.json](/home/cj/iron/iron/applications/transformer_layer/study/operator_runlist_component_long_seq_with_projection.json)
- [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json)
- [gpu_compare_embedding_igpu.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare_embedding_igpu.json)
- [full_study_pipeline.json](/home/cj/iron/iron/applications/transformer_layer/study/full_study_pipeline.json)
- [programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv)
- [design_pattern_considerations.md](/home/cj/iron/iron/applications/transformer_layer/docs/design_pattern_considerations.md)
- [programmability_debugging.md](/home/cj/iron/iron/applications/transformer_layer/docs/programmability_debugging.md)

Mixed-status study manifests may also emit support summaries under
`results/*_support_matrix.{csv,txt}` so unsupported pattern/case points remain
explicit in the study outputs instead of disappearing from the matrix.

Generated benchmark outputs belong under `iron/applications/transformer_layer/results/` and are intentionally ignored by git.
