# mha_to_an

Fused AIE operator for encoder-style MHA output projection plus Add+Norm.
It consumes packed Q/K/V and residual buffers, runs tiled kernels on NPU2, and writes normalized output.

## What This Operator Does

- Computes MHA core path: `QK^T -> softmax -> PV`
- Applies output projection `W_O`
- Applies fused Add+LayerNorm with residual input

Main files:
- `design.py`: AIE graph, FIFOs, workers, runtime sequencing
- `op.py`: host wrapper, artifact build, runtime buffers
- `reference.py`: golden model and host-side packing
- `test.py`: parametrized pytest coverage

## Buffer Contract

Tensor shapes use `embed_sz = num_heads * d`.

- `W_O`: `(embed_sz, embed_sz)`
- `QKV`: `(3 * seq_len, embed_sz)` packed as:
  - rows `[0:seq_len]` = `Q`
  - rows `[seq_len:2*seq_len]` = `K`
  - rows `[2*seq_len:3*seq_len]` = `V`
- `OR`: `(2 * seq_len, embed_sz)` packed as:
  - rows `[0:seq_len]` = output region (written by kernel)
  - rows `[seq_len:2*seq_len]` = residual `R` (read by Add+Norm)

In `op.py`, logical output `O` is aliased to the first half of `OR`.

## Execution Model

Pipeline is streamed with ObjectFifos and workers:

1. QK matmul workers
2. partial softmax workers
3. PV matmul + rescale workers
4. O-proj workers with configurable accumulation depth (`o_proj_acc_depth`)
5. Add+Norm worker

Add+Norm uses a two-pass protocol per q-block:
- pass 1: accumulate row `sum/sumsq`
- pass 2: apply fused layer norm + residual add

## Key Constraints

- Current implementation supports `d == 64`
- `heads % parallel_heads == 0`
- `seq_len % seq_tile == 0`
- `seq_len % kv_seq_tile == 0`
- `emb_tile * o_proj_acc_depth == embed_sz`

## Notes on Debug/Accuracy

- `test.py` defaults to `DEBUG_MODE=2`, which uses stress-style reference data.
- In this mode, output-projection numeric sensitivity can dominate mismatch counts; validate conclusions with `DEBUG_MODE=0` when checking functional correctness.
- `AIEMHAOutProj` also exposes `addnorm_debug_mode` (`-1`, `0`, `1`) that maps to `DEBUG_AIE_KERNELS` in `encoder.cc`, matching `ffn_addnorm` behavior:
  - `0`: AddNorm passes through its input tensor
  - `1`: AddNorm passes through residual
  - `-1`: normal AddNorm
