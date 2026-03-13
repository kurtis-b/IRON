# encoder_pipeline

Fused encoder operator:

- `MHA + AddNorm1 + FFN + AddNorm2`

This file is the entrypoint only. The active design state and forward work live
in:

- [current_status.md](/home/agi-demo/iron/operators/encoder_pipeline/current_status.md)
- [optimization_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/optimization_plan.md)
- [readability_cleanup_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/readability_cleanup_plan.md)

Historical notes and experiment logs live under:

- [docs/history.md](/home/agi-demo/iron/operators/encoder_pipeline/docs/history.md)

## Core facts

- LN1 and LN2 are both two-pass layer norm stages.
- `ln1_staging_design="memtile"` broadcasts LN1 output on-chip.
- `ln1_staging_design="ddr"` stages LN1 output through DDR once, then
  broadcasts on-chip.
- Same-tile full-row staging uses `aie.memtile_row_store` where the current
  runtime path is stable.

## Supported constraints

- `emb_tile * proj_acc_depth == embed_sz`
- `parallel_heads % o_proj_acc_group_size == 0`
- `o_proj_acc_group_size <= parallel_heads`
- only evenly partitioned FFN layouts are supported

## Main entry points

- regression: `operators/encoder_pipeline/test.py`
- debug sweep: `operators/encoder_pipeline/profile_debug_modes.py`

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

- Tracked staging-mode selections are currently green:
  - `lnstage_memtile`: `100 passed, 40 skipped`
  - `lnstage_ddr`: `115 passed, 40 skipped`
- See [current_status.md](/home/agi-demo/iron/operators/encoder_pipeline/current_status.md)
  for the current stable row-store envelope, representative cases, and the
  active bottlenecks.
