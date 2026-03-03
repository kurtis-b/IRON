# encoder_pipeline

This directory contains the Python-side implementation and tests for the fused encoder pipeline operator:
MHA + AddNorm1 + FFN + AddNorm2.

## Python files

- `constants.py`
  - Defines debug mode enums and compatibility constants.
  - Provides helper functions to resolve:
    - top-level debug mode into MHA debug + FFN stage-only mode
    - global/per-stage AddNorm debug modes.

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

Debug mode definitions are centralized in `constants.py` and used by `op.py`, `reference.py`, and `test.py`.

- Top-level `debug` (values in `DebugMode`):
  - `0` (`FULL`): run full pipeline.
  - `1` (`SELF_ATTN`): self-attention-focused MHA debug mode.
  - `2` (`O_PROJ`): output-projection-focused MHA debug mode.
  - `3` (`FFN_UP_ONLY`): FFN stage isolation for up projection.
  - `4` (`FFN_DOWN_ONLY`): FFN stage isolation for down projection.
  - `5` (`FFN_ADDNORM_ONLY`): FFN stage isolation for AddNorm2.

- FFN stage isolation mapping:
  - `debug=3 -> ffn_stage_only=0`
  - `debug=4 -> ffn_stage_only=1`
  - `debug=5 -> ffn_stage_only=2`
  - `debug in {0,1,2} -> ffn_stage_only=None` (all FFN stages enabled)

- MHA debug mapping:
  - `debug=0 -> mha_debug=0`
  - `debug=1 -> mha_debug=1`
  - `debug=2 -> mha_debug=2`
  - `debug in {3,4,5} -> mha_debug=2` (deterministic O-proj-style MHA feed for FFN isolation)

- AddNorm debug controls:
  - Global: `addnorm_debug_mode` in `{-1, 0, 1}`
  - Per-stage overrides: `addnorm1_debug_mode`, `addnorm2_debug_mode` in `{-1, 0, 1}`
  - Meaning:
    - `-1`: normal AddNorm (layer norm + residual add)
    - `0`: pass through AddNorm input
    - `1`: pass through residual
  - Precedence:
    - If per-stage mode is provided, it overrides global mode for that stage.
    - Otherwise the stage uses `addnorm_debug_mode`.

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
