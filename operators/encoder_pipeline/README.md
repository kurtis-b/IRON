# Encoder Pipeline

`operators/encoder_pipeline` is the minimal full-pipeline operator surface for the fused encoder block.

Files follow the same shape as `operators/mha` and `operators/ffn`:
- `design.py`
- `op.py`
- `reference.py`
- `test.py`

This module intentionally exposes only the full pipeline. It does not add debug or profiling modes in the new directory.

The operator surface, reference implementation, and design entrypoint are native to this directory and model only the full pipeline.

Current scope:
- non-sequence-parallel public path only
- objectfifo-based accumulation/replay on the live full-pipeline path
- hardcoded placement by supported topology instead of planner-style placement

Supported hardcoded topologies today:
- `(12, 64, 64, 32, 64, 96, 1, 1, 8, 1, 1, 3072)`
- `(1, 64, 64, 32, 64, 32, 1, 1, 2, 1, 1, 96)`
