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
