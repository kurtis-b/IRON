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
3. Add roofline annotation and bottleneck analysis.
4. Add imported checkpoint support only if needed.
5. Add the separate GPU comparison later.
