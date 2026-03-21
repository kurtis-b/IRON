# Encoder Pipeline

`encoder_pipeline` is the full encoder block operator:

- front: `MHA + O_PROJ + LN1`
- tail: `FFN + LN2`

The implementation is placement-driven. Supported runtime/topology combinations are the hardcoded keys in [placements.py](/home/agi-demo/iron/operators/encoder_pipeline/placements.py).

## Design

The operator entry point is [op.py](/home/agi-demo/iron/operators/encoder_pipeline/op.py). The design callback and runtime schedule live in [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py).

Current design points:

- `d=64` only
- separate tile sizes for:
  - embedding/output dimension: `emb_tile`
  - FFN intermediate dimension: `ffn_tile`
- non-sequence-parallel path (`parallel_seq=1`) now uses stats-first layer norm handling instead of LN replay
- multi-head non-seq (`2ph`, `4ph`) stages the final reduced `O_PROJ` result before `LN1`
- large non-seq multi-head cases chunk host `K`/`V` fills so long sequence lengths fit DMA-BD limits
- sequence-parallel uses hardcoded lane placements
- unified sequence-parallel is enabled through `parallel_seq=4`

Useful implementation files:

- [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py): worker graph, objectfifos, runtime schedule
- [op.py](/home/agi-demo/iron/operators/encoder_pipeline/op.py): public operator surface, artifact naming, compile/runtime integration
- [placements.py](/home/agi-demo/iron/operators/encoder_pipeline/placements.py): hardcoded supported topologies and placements
- [reference.py](/home/agi-demo/iron/operators/encoder_pipeline/reference.py): golden-reference generation
- [test.py](/home/agi-demo/iron/operators/encoder_pipeline/test.py): public pytest matrix

## Toolchain Note

`encoder_pipeline` should be used with the specific `mlir_aie` wheel currently validated in `ironenv`:

- `mlir_aie==0.0.1.2026031811+71fb44f147`

This operator is sensitive to `mlir_aie` lowering/allocation behavior. Different wheels can change:

- DMA-BD allocation outcomes
- compile viability of some topologies
- runtime behavior of the generated design

## Test Surface

The public test matrix in [test.py](/home/agi-demo/iron/operators/encoder_pipeline/test.py) currently uses:

- `seq_len`: powers of 2 from `64` through `16384`
- base topology:
  - `d=64`
  - `num_heads=12`
  - `ffn_intermediate_size=3072`
  - `seq_tile=32`
  - `kv_seq_tile=64`
  - `emb_tile=96`
  - `ffn_tile=64`

Runtime topology cases:

- baseline: `1ps 1ph 1pffn`
- sequence parallel: `2ps`, `4ps`
- head parallel: `2ph`, `4ph`
- FFN parallel: `2pffn`, `4pffn`
- mixed:
  - `2ph_2pffn`
  - `2ph_4pffn`

Constraints that affect which test cases are generated:

- `seq_len` must be divisible by `seq_tile`
- `(seq_len // seq_tile)` must be divisible by `parallel_seq`
- so `4ps` starts effectively at `128seq`, not `64seq`

## Running Tests

Set up the environment first:

```bash
source /opt/xilinx/xrt/setup.sh
source /home/agi-demo/iron/ironenv/bin/activate
```

Run the full public matrix:

```bash
pytest -q operators/encoder_pipeline/test.py
```

Run only sequence-parallel cases:

```bash
pytest -q operators/encoder_pipeline/test.py -k "_2ps or _4ps" -x
```

Run only multi-head non-seq cases:

```bash
pytest -q operators/encoder_pipeline/test.py -k "_2ph or _4ph" -x
```

Show generated cases without running hardware:

```bash
pytest -q operators/encoder_pipeline/test.py --collect-only
```

Notes:

- default pytest iterations come from [conftest.py](/home/agi-demo/iron/conftest.py)
- running pytest without `-s` writes metrics to `tests_latest.csv`
- the test prints latency and bandwidth for each case

## Artifacts

Generated MLIR, xclbin, and related build artifacts are emitted under `build/`.

The artifact stem includes the key tiling/runtime choices, for example:

- `..._{emb_tile}e_{ffn_tile}f_{parallel_seq}ps_{parallel_heads}ph_...`

That makes it easier to match generated artifacts back to a specific test case.
