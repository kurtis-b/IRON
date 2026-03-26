# Transformer Layer Design-Pattern Study

`transformer_layer` is a thesis-focused application for studying how a single encoder-style transformer layer maps onto Ryzen AI. The main study compares three NPU design patterns:

- `encoder_pipeline`
- `gemm_only`
- `operator_runlist`

The app is intentionally layer-centric rather than model-centric. It does not benchmark embeddings, tokenization, or full-sequence application behavior. The point is to isolate how workload mapping changes latency, efficiency, bottlenecks, and programmability on a resource-constrained NPU.

The AMD GPU comparison is separate from the main NPU study. It is only used to compare the best NPU result against a laptop-class AMD GPU baseline after the NPU workflow is already stable.

## Workload Contract

The default study uses one synthetic transformer layer with:

- batch size `1`
- no attention mask
- `bfloat16`
- synthetic hidden states
- synthetic weights
- one layer-local CPU reference path for parity checks

The canonical layer spec lives in [layer_spec.py](/home/cj/iron/iron/applications/transformer_layer/src/layer_spec.py). Imported weights are represented in the schema, but this branch currently runs synthetic weights only. There is no `import_layer_weights.py` command in the app yet.

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
  --warmup-runs 1 \
  --runs-per-sample 5 \
  --output-csv iron/applications/transformer_layer/results/npu_smoke_seq64.csv
```

Main NPU sweep:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json
```

There is no separate topology-cache warmup step in this app. Each pattern owns its own compilation/runtime setup inside the study harness.

Parity validation:

```bash
python iron/applications/transformer_layer/validate_npu_parity.py \
  --execution-mode encoder_pipeline \
  --seq-lens 64,128 \
  --output-csv iron/applications/transformer_layer/results/parity_encoder_pipeline.csv
```

Peak-reference artifact generation:

```bash
python iron/applications/transformer_layer/calibrate_backend_peaks.py \
  --backend npu \
  --peak-ops-per-sec 1.0 \
  --peak-bytes-per-sec 1.0 \
  --output iron/applications/transformer_layer/config/peak_references.json \
  --append
```

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

Isolated AMD GPU comparison:

```bash
python iron/applications/transformer_layer/gpu_inference.py \
  --seq-len 64 \
  --device cuda:0 \
  --power-backend rocm-smi \
  --output-csv iron/applications/transformer_layer/results/gpu_compare_amd.csv
```

Manifest-driven AMD GPU comparison:

```bash
python iron/applications/transformer_layer/gpu_inference.py \
  --seq-len 128 \
  --device cuda:0 \
  --power-backend rocm-smi \
  --output-csv iron/applications/transformer_layer/results/gpu_compare_amd_seq128.csv
```

The dedicated AMD GPU comparison manifest lives at [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json).

Thesis plotting:

```bash
python -m iron.applications.transformer_layer.plot_design_pattern_results \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv \
  --bottleneck-csv iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.csv \
  --gpu-compare-csv iron/applications/transformer_layer/results/gpu_compare_amd.csv \
  --output-dir iron/applications/transformer_layer/results/plots/design_patterns_main
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
- [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json)
- [programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv)
- [programmability_debugging.md](/home/cj/iron/iron/applications/transformer_layer/docs/programmability_debugging.md)

Generated benchmark outputs belong under `iron/applications/transformer_layer/results/` and are intentionally ignored by git.
