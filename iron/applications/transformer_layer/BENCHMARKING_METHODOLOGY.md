# Benchmarking Methodology

The first milestone of `transformer_layer` uses a synthetic single-layer workload contract:

- one encoder-style transformer layer
- batch size `1`
- no attention mask
- synthetic weights and synthetic hidden states
- NPU-only comparison across `encoder_pipeline`, `gemm_only`, and `operator_runlist`

The methodology is intentionally staged:

1. Bring up a layer-local workload contract and reference layer.
2. Run synthetic NPU pattern comparisons with a shared CSV schema.
3. Drive repeatable studies from checked-in JSON manifests.
4. Add roofline annotation and bottleneck analysis.
5. Add imported checkpoint support only if needed.
6. Add the separate GPU comparison later.

Phase 4 roofline workflow:

1. Create or update [peak_reference.py](/home/cj/iron/iron/applications/transformer_layer/peak_reference.py) artifacts with [calibrate_backend_peaks.py](/home/cj/iron/iron/applications/transformer_layer/calibrate_backend_peaks.py).
2. Run a manifest-driven suite with [automated_benchmark.py](/home/cj/iron/iron/applications/transformer_layer/automated_benchmark.py).
3. Either let the manifest write an annotated CSV automatically, or post-process the suite with [annotate_roofline.py](/home/cj/iron/iron/applications/transformer_layer/annotate_roofline.py).

Phase 5 bottleneck workflow:

1. Run a normal suite so each row includes stage-level timing and pattern metadata.
2. Post-process the suite with [analyze_design_pattern_bottlenecks.py](/home/cj/iron/iron/applications/transformer_layer/analyze_design_pattern_bottlenecks.py).
3. Use the emitted CSV/JSON/text outputs as the basis for thesis bottleneck discussion.

Phase 6 programmability/debugging workflow:

1. Pass `debug_log_csv` in the study manifest or on the CLI when running [automated_benchmark.py](/home/cj/iron/iron/applications/transformer_layer/automated_benchmark.py).
2. Let the harness append structured study, benchmark-case, roofline, and parity events to the debug CSV.
3. Curate stable challenge/mitigation rows into [study/programmability_debug_log.csv](/home/cj/iron/iron/applications/transformer_layer/study/programmability_debug_log.csv).
4. Use [docs/programmability_debugging.md](/home/cj/iron/iron/applications/transformer_layer/docs/programmability_debugging.md) as the thesis-section skeleton.

Phase 7 isolated AMD GPU workflow:

1. Use [gpu_inference.py](/home/cj/iron/iron/applications/transformer_layer/gpu_inference.py) for the separate ROCm-based comparison path.
2. Keep the GPU rows separate from the main NPU suite; compare them only against the best NPU pattern.
3. Use `power_backend=rocm-smi` when AMD GPU power and energy collection is needed.
4. Use [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json) as the dedicated comparison manifest.

Phase 8 thesis plotting workflow:

1. Produce or update the annotated NPU suite CSV with [annotate_roofline.py](/home/cj/iron/iron/applications/transformer_layer/annotate_roofline.py).
2. Produce the bottleneck summary CSV with [analyze_design_pattern_bottlenecks.py](/home/cj/iron/iron/applications/transformer_layer/analyze_design_pattern_bottlenecks.py).
3. Run the isolated AMD GPU suite separately with [gpu_inference.py](/home/cj/iron/iron/applications/transformer_layer/gpu_inference.py) or [gpu_compare.json](/home/cj/iron/iron/applications/transformer_layer/study/gpu_compare.json).
4. Generate SVG and HTML figures with [plot_design_pattern_results.py](/home/cj/iron/iron/applications/transformer_layer/plot_design_pattern_results.py).
5. Treat the main NPU pattern plots and the best-NPU-vs-AMD-GPU plots as distinct figure groups in the thesis writeup.
