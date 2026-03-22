# encoder_pipeline_memtile

Focused memtile-mode entrypoints for encoder pipeline experimentation.

- full mode-local design: [design.py](/home/agi-demo/iron/operators/encoder_pipeline_memtile/design.py)
- mode hooks: [hooks.py](/home/agi-demo/iron/operators/encoder_pipeline_memtile/hooks.py)
- fixed-mode operator: [op.py](/home/agi-demo/iron/operators/encoder_pipeline_memtile/op.py)
- focused pytest entrypoint: [cases.py](/home/agi-demo/iron/operators/encoder_pipeline_memtile/cases.py)

The mode-local design and test files are self-contained. They no longer import
implementation code from `operators/encoder_pipeline/*`.

Run:

```bash
rm -rf ./build
pytest -q operators/encoder_pipeline_memtile/cases.py
```

Known issues after removing the row-store path:

- The direct row-store implementation has been removed from this mode. Replay
  and accumulation now use FIFO-only forwarding.
- Two representative numerical regressions are still present in the FIFO-only
  path:
  - `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_1pheads_1pffn_8pacc_1opg`
  - `encoder_64seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- One representative compile-time memtile DMA-pressure regression is also still
  present:
  - `encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_2pffn_16pacc_2opg`
