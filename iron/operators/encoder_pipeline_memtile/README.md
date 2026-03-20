# encoder_pipeline_memtile

`encoder_pipeline_memtile` is the maintained on-chip LN1 variant of
[`encoder_pipeline`](../encoder_pipeline/README.md).

It reuses the current `encoder_pipeline` design and placement logic. The only
intentional behavior difference is the LN1 post-activation path:

- `encoder_pipeline`: stages `LN1 -> FFN` through the host-visible `OR` buffer
- `encoder_pipeline_memtile`: keeps `LN1 -> FFN` and the LN2 residual replay
  on-chip

Current scope:

- non-sequence-parallel topologies
- sequence-parallel benchmark topologies:
  - `2ps`
  - `2ps_2ph`
  - `2ps_2pffn`
  - `2ps_2ph_2pffn` (using grouped O-proj accumulation)
- heavier sequence-parallel topologies are still in progress
- `4ps` is intentionally unsupported in memtile mode
  - a pure on-chip `4ps` design needs LN1 output to feed FFN-up, replay, and LN2
    residual simultaneously
  - on the current placement/tile budget, that exceeds the available DMA channel
    budget without either an extra worker tile or a DDR fallback
  - DDR fallback is explicitly not part of the memtile contract
- same correctness thresholds as `encoder_pipeline`
- default pytest slice covers the benchmark-style `64` and `512` sequence cases

The live entrypoints are:

- [op.py](./op.py): thin operator wrapper that forces `ln1_staging_design=memtile`
- [design.py](./design.py): thin MLIR entrypoint wrapper over `encoder_pipeline`
- [reference.py](./reference.py): golden reference wrapper over `encoder_pipeline`
- [cases.py](./cases.py): focused memtile validation slice for the passing
  benchmark topologies

Run the default benchmark-style slice with:

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source <repo_root>/ironenv/bin/activate
pytest -q iron/operators/encoder_pipeline_memtile/cases.py -m "not extensive" -s
```
