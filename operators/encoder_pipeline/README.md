# encoder_pipeline

Fused encoder operator: `MHA + AddNorm1 + FFN + AddNorm2`.

## What matters

- LN1 and LN2 are both two-pass layer norm stages (full-row statistics are required).
- Same-tile full-row staging now uses `aie.memtile_row_store` with `buffer_count=2` where legal:
  - LN1 replay
  - LN2 replay in `memtile` mode only for the current stable envelope:
    `parallel_heads == 4` and `effective_ffn_branches <= 4`
  - O-proj accumulation staging for the current stable envelope:
    `parallel_heads <= 4`, with FIFO fallback when no safe memtile slot exists
  - FFN-down accumulation staging only for `effective_ffn_branches <= 4`, with
    DDR low-head layouts falling back to FIFO by default
- `ln1_staging_design="memtile"`: LN1 output is broadcast on-chip to FFN-up branches.
- `ln1_staging_design="ddr"`: LN1 output is staged through DDR once, then broadcast on-chip.
- Topology knobs in test IDs: `pheads`, `pffn`, `pacc`, `opg`.

## Placement overview

- MHA + O-proj: per-head compute columns (`rows 2..5`).
- AddNorm1: LN1 norm then LN1 mul/add with residual.
- FFN: up/down tiles selected from FFN layout; down reduction is a neighbor chain that ends near LN2.
- AddNorm2: LN2 consumes reduced FFN output + residual and drains final output.
- Exact coordinates are printed in runtime logs (`FFN mapped placement`, `down_reduction_edges`).

## Hard constraints

- `emb_tile * proj_acc_depth == embed_sz`
- `parallel_heads % o_proj_acc_group_size == 0`
- `o_proj_acc_group_size <= parallel_heads`
- Compute-tile, shim-channel, memtile-channel, and BD budgets are validated during generation/lowering.

## Entry points

- Regression: `operators/encoder_pipeline/test.py`
- Debug/bottleneck sweep: `operators/encoder_pipeline/profile_debug_modes.py`

## Environment setup

```bash
cd /home/agi-demo/iron
# Only if conda is initialized and active in your shell:
conda deactivate
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
```

## Required run rule

Delete `./build` before each test run:

```bash
rm -rf ./build
```

## Common commands

All encoder pipeline tests:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -q
```

Tracked staging-mode selections:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -q -k lnstage_memtile

rm -rf ./build
pytest operators/encoder_pipeline/test.py -q -k lnstage_ddr
```

One staging mode:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -q -k lnstage_memtile
```

One topology:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -q -k "4pheads_4pffn_8pacc_4opg and lnstage_memtile"
```

Verbose run:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -s -vv
```

## CSV and iteration options

Provided via pytest options:

- `--csv-output <path>`
- `--iterations <N>`

Example:

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py \
  -q \
  -k "(lnstage_ddr or lnstage_memtile) and not stage_profile" \
  --csv-output operators/encoder_pipeline/encoder_pipeline_results.csv \
  --iterations 1
```

## Stage-profile runs

`test_encoder_pipeline_stage_profile` is skipped unless `ENABLE_STAGE_PROFILE_TESTS=True` in `test.py`.

```bash
rm -rf ./build
pytest operators/encoder_pipeline/test.py -q -k stage_profile
```

## Debug-mode sweep helper

```bash
python operators/encoder_pipeline/profile_debug_modes.py --design memtile --clean-build
```

## Current validated state

- Targeted row-store control case (`memtile`) passes:
  - `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- Targeted row-store control case (`ddr`) passes:
  - `encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- Long-sequence complex subset passes after narrowing row-store usage to the stable envelope:
  - `105 passed, 190 deselected`
  - covers `512/1024/2048` for:
    - `6pheads_2pffn_8pacc_2opg`
    - `4pheads_4pffn_8pacc_4opg`
    - `4pheads_6pffn_6pacc_2opg`
    - `16heads_4096ffn_32qseqtile_64kvtile_128embtile_4pheads_4pffn_8pacc_4opg` (`ddr` only)
- Re-run broader selections after changing row-store channel maps or adding new row-store sites.
- The current `64embtile / 12pacc` experiments no longer fail on LN/O-proj/FFN-down row-store staging; the first exposed allocator limit is now the final `memLN2` shim drain.
