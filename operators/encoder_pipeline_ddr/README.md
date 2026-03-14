# encoder_pipeline_ddr

Focused DDR-mode entrypoints for encoder pipeline experimentation.

- full mode-local design: [design.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/design.py)
- mode hooks: [hooks.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/hooks.py)
- fixed-mode operator: [op.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/op.py)
- focused pytest entrypoint: [cases.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/cases.py)
- debug pytest entrypoint: [debug_cases.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/debug_cases.py)
- debug cleanup plan: [debug_mode_cleanup_plan.md](/home/agi-demo/iron/operators/encoder_pipeline_ddr/debug_mode_cleanup_plan.md)

Supported debug surface:
- stage-only profile modes: `QK`, `softmax`, `PV`, `O proj`, `LN1`, `Up proj`, `Down proj`, `LN2`
- stage-correctness modes: `O proj`, `Down proj`, `LN2`
- `LN1` and `Up proj` remain profile-only in the live encoder graph

The mode-local design and test files are self-contained. They no longer import
implementation code from `operators/encoder_pipeline/*`.

Run:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_ddr/cases.py
```

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_ddr/debug_cases.py
```
