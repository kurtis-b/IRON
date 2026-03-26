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
