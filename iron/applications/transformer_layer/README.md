# Transformer Layer Thesis App

This app is the layer-centric thesis benchmark surface for comparing three Ryzen AI NPU design patterns on a single encoder-style transformer layer:

- `encoder_pipeline`
- `gemm_only`
- `operator_runlist`

The initial milestone in this branch is intentionally synthetic-first:

- one layer only
- no embeddings
- no tokenizer or whole-model wrapper
- NPU execution modes only
- `iron/applications/bert/` remains donor code temporarily and will be removed later

Use [npu_inference.py](/home/cj/iron/iron/applications/transformer_layer/npu_inference.py) to run the first synthetic layer benchmark path.

Typical local workflow:

```bash
source /opt/xilinx/xrt/setup.sh
source ./ironenv/bin/activate
python iron/applications/transformer_layer/npu_inference.py \
  --execution-mode encoder_pipeline \
  --seq-len 64 \
  --warmup-runs 1 \
  --runs-per-sample 5
```

For study-driven sweeps, use the checked-in manifests under [study](/home/cj/iron/iron/applications/transformer_layer/study):

```bash
source /opt/xilinx/xrt/setup.sh
source ./ironenv/bin/activate
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json
```

To annotate an existing suite with backend peak and roofline percentages:

```bash
python -m iron.applications.transformer_layer.annotate_roofline \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main.csv \
  --peak-reference iron/applications/transformer_layer/config/peak_references.json \
  --output-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv
```

To generate pattern-specific bottleneck summaries from a suite CSV:

```bash
python -m iron.applications.transformer_layer.analyze_design_pattern_bottlenecks \
  --input-csv iron/applications/transformer_layer/results/design_patterns_main_annotated.csv \
  --summary-csv iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.csv \
  --summary-json iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.json \
  --summary-text iron/applications/transformer_layer/results/design_patterns_main_bottlenecks.txt
```

To capture structured programmability/debug events during a manifest run:

```bash
python iron/applications/transformer_layer/automated_benchmark.py \
  --study-manifest iron/applications/transformer_layer/study/design_patterns_main.json \
  --debug-log-csv iron/applications/transformer_layer/results/design_patterns_main_debug_log.csv
```

The checked-in thesis-facing skeletons live at:

- [study/programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv)
- [docs/programmability_debugging.md](/home/cj/iron/iron/applications/transformer_layer/docs/programmability_debugging.md)
