# encoder_pipeline_memtile

Focused memtile-mode entrypoints for encoder pipeline experimentation.

- full mode-local design: [design.py](<repo_root>/operators/encoder_pipeline_memtile/design.py)
- mode hooks: [hooks.py](<repo_root>/operators/encoder_pipeline_memtile/hooks.py)
- fixed-mode operator: [op.py](<repo_root>/operators/encoder_pipeline_memtile/op.py)
- focused pytest entrypoint: [cases.py](<repo_root>/operators/encoder_pipeline_memtile/cases.py)

The mode-local design and test files are self-contained. They no longer import
implementation code from `operators/encoder_pipeline/*`.

Run:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_memtile/cases.py
```
