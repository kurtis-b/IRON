# DDR Debug Mode Cleanup Plan

## Goal

Reduce the DDR debug-mode surface to the modes that are useful and maintainable:

- keep stage-only profile modes for all stages
- keep stage-correctness modes only for:
  - `O proj`
  - `Down proj`
  - `LN2`
- remove stage-correctness modes for:
  - `QK`
  - `softmax`
  - `PV`

## Why

- The early-stage verify path introduced a large amount of special-case runtime
  and FIFO code in `design.py`.
- Those paths are not needed for the current workflow.
- The early-stage direct verify modes are the unstable part of the implementation.

## File-Level Steps

1. `debug_modes.py`
- remove public verify IDs for `QK`, `softmax`, and `PV`
- keep stage IDs for profiling
- narrow `is_direct_verify_stage(...)` to the remaining direct-verify stages

2. `reference.py`
- stop generating `QK`, `softmax`, `PV`, `LN1`, and `Up proj` stage-native verify outputs
- keep verify outputs for `O proj`, `Down proj`, and `LN2`

3. `op.py`
- remove debug output-shape handling for `QK`, `softmax`, and `PV`

4. `debug_cases.py`
- run profile tests for all stages
- run verify tests only for the remaining supported correctness modes

5. `design.py`
- delete early-stage verify FIFOs and runtime branches:
  - `softmaxVerifyPart*`
  - `softmaxVerifyOut*`
  - `pvVerifyOut*`
  - `QK / softmax / PV` direct-verify runtime branch
- restore the normal `QK -> softmax -> PV` path for all remaining modes
- keep `LN1` and `Up proj` as profile-only modes in the live encoder graph

## Validation

Run:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_ddr/debug_cases.py -s -x
```

Then run the full DDR surface:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_ddr/cases.py -s -x
```
