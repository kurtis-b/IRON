# encoder_pipeline

This directory contains the Python-side implementation and tests for the fused encoder pipeline operator:
MHA + AddNorm1 + FFN + AddNorm2.

## Python files

- `design.py`
  - Generates the AIE MLIR graph (`fused_mha`).
  - Defines object FIFOs, worker kernels, tile placement, tensor access patterns, and runtime fill/drain sequence.
  - Includes CLI entrypoint (`main`) to emit MLIR to `build/encoder_pipeline.mlir` (or a provided output path).

- `op.py`
  - Runtime-facing operator wrapper class: `AIEEncoderPipeline`.
  - Validates shape/depth constraints, builds compile artifacts, sets up runtime buffers, and runs the kernel.
  - Supports separate AddNorm-1 and AddNorm-2 compile-time LN weights (`ln1_weight`, `ln2_weight`), with `ln_weight` as backward-compatible fallback.
  - Converts tensors to/from host buffers (`QKV`, `W_O`, `OR`, `B_Up`, `B_Down`, `O`).

- `reference.py`
  - Golden reference model used by tests.
  - Mirrors debug/stage-isolation behavior expected by the hardware graph.
  - Produces reference inputs/outputs and packed host buffers.

- `test.py`
  - Parametrized pytest suite for `AIEEncoderPipeline`.
  - Generates golden data from `reference.py`, runs the operator, and checks output error counts within fixed tolerances.

## How files connect

1. `test.py` calls `reference.py` to generate input/output tensors.
2. `test.py` instantiates `AIEEncoderPipeline` from `op.py`.
3. `op.py` invokes `design.py` (via artifact generation) to produce MLIR/XCLBIN/insts.
4. Runtime executes the design and compares `O` against the reference output.

## Debug modes

`op.py` and `reference.py` accept a single top-level `debug` argument:

- `-1`: debug disabled (default full pipeline behavior).
- `0`: self-attention debug path; AddNorm stages pass through their input path.
- `1`: deterministic MHA-input debug path (linear QK/PV flow); AddNorm stages pass input.
- `2`: residual-path debug; AddNorm stages pass residual.
- `3`: FFN up-proj isolation (`ffn_stage_only=0`), deterministic MHA feed.
- `4`: FFN down-proj isolation (`ffn_stage_only=1`), deterministic MHA feed.
- `5`: FFN AddNorm2 isolation (`ffn_stage_only=2`), deterministic MHA feed.

Internally, `op.py` resolves `debug` into:
- MHA kernel debug value (`mha_debug`)
- FFN stage-only selector (`ffn_stage_only`)
- per-stage AddNorm pass-through modes (`addnorm1_debug_mode`, `addnorm2_debug_mode`)

## Dataflow mapping and pipelining

- Host-facing buffer packing:
  - `QKV` packs `[Q; K; V]` row-wise.
  - `OR` packs `[output_region; residual]` row-wise.
  - `B_Up` and `B_Down` are flat host buffers; `TensorAccessPattern`s in `design.py` define their logical 2D tile order.
  - Runtime aliases `O` to the first half of `OR`.

- Stage-to-stage flow on the array:
  - MHA path:
    - `Q/K -> QK matmul -> softmax -> PV matmul -> O-proj`
    - O-proj accumulates through mem-tile FIFOs (`outOProjAccum*`) and reduces across parallel-head workers (`outOPart*`) to `outO`.
  - AddNorm1 bridge:
    - `outO + R -> AddNorm1`
    - emits two streams:
      - `outLN` (FFN input)
      - `ffnROut -> ffnRIn` (residual stream for AddNorm2, routed through a mem tile for queue depth).
  - FFN path:
    - `outLN + B_Up -> FFN up (with GeLU) -> ffnUpOut`
    - `ffnUpOut + B_Down -> FFN down`, with accumulation queue in `ffnDownAccum`.
  - Final norm:
    - `ffnDownOut + ffnRIn -> AddNorm2 -> outLN2 -> memLN2 -> drain to OR/O`.

- Runtime pipelining model:
  - All workers are started once and run continuously.
  - Runtime iterates by `(q_block_idx, col_group)` and schedules fills/drain for one tile-group at a time:
    - fill `Q` for the q-block
    - for each head-group, fill `K`, `V`, `W_O`
    - fill residual `R`
    - fill FFN weights `B_Up`, `B_Down` for the same col-group
    - drain final output tile from `memLN2` to `OR`/`O`.

- Two-pass behavior used for AddNorm stages:
  - O-proj/AddNorm1:
    - pass 1 provides tiles for sum/sumsq accumulation.
    - pass 2 replays tiles for normalized output application.
  - FFN-down/AddNorm2 uses the same pattern:
    - FFN-down emits/replays accumulated tiles so AddNorm2 can do sum/sumsq then normalized apply.
