# encoder_pipeline_ddr

Focused DDR-mode entrypoints for encoder pipeline experimentation.

- full mode-local design: [design.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/design.py)
- mode hooks: [hooks.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/hooks.py)
- fixed-mode operator: [op.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/op.py)
- focused pytest entrypoint: [cases.py](/home/agi-demo/iron/operators/encoder_pipeline_ddr/cases.py)

The mode-local design and test files are self-contained. They no longer import
implementation code from `operators/encoder_pipeline/*`.

Run:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_ddr/cases.py
```
