# encoder_pipeline

Fused encoder operator: `MHA + AddNorm1 + FFN + AddNorm2`.

## Design summary

- LN1 and LN2 stay two-pass because layer norm requires full-row statistics.
- LN1 staging modes:
- `ln1_staging_design="memtile"`: LN1 output is broadcast on-chip to FFN-up branches.
- `ln1_staging_design="ddr"`: LN1 output stages through DDR once, then is broadcast on-chip.
- FFN and O-proj parallelization are controlled by test topology parameters (`pheads`, `pffn`, `pacc`, `opg` in test IDs).

## Core constraints

- `emb_tile * proj_acc_depth == embed_sz`
- `o_proj_acc_group_size in {1, 2, 4}`
- `parallel_heads % o_proj_acc_group_size == 0`
- Compute-tile and DMA/BD budgets are validated during design construction/lowering.

## Test entry points

- Main regression suite: `operators/encoder_pipeline/test.py`
- Bottleneck/debug sweep helper: `operators/encoder_pipeline/profile_debug_modes.py`

## Environment setup for tests

```bash
cd /home/agi-demo/iron
# If conda is active in your shell:
conda deactivate
source /opt/xilinx/xrt/setup.sh
source ~/iron/ironenv/bin/activate
rm -rf ./build
```

## Pytest commands

Run all encoder_pipeline tests:

```bash
pytest operators/encoder_pipeline/test.py -q
```

Run one LN1 staging mode only:

```bash
pytest operators/encoder_pipeline/test.py -q -k lnstage_ddr
pytest operators/encoder_pipeline/test.py -q -k lnstage_memtile
```

Run both modes explicitly:

```bash
pytest operators/encoder_pipeline/test.py -q -k "(lnstage_ddr or lnstage_memtile)"
```

Run one topology/case by node-id substring:

```bash
pytest operators/encoder_pipeline/test.py -q -k "4pheads_4pffn_8pacc_4opg and lnstage_memtile"
```

Stop on first failure during triage:

```bash
pytest operators/encoder_pipeline/test.py -q -x
```

Print per-test logs/stdout (useful for mismatch/runtime diagnostics):

```bash
pytest operators/encoder_pipeline/test.py -s -vv
```

## CSV output and iteration control

`conftest.py` provides two pytest options used by this suite:

- `--csv-output <path>`: write metrics CSV to a chosen file.
- `--iterations <N>`: repeat each test `N` times and aggregate metrics.

Run both LN1 modes and write all results to one CSV:

```bash
pytest operators/encoder_pipeline/test.py \
  -q \
  -k "(lnstage_ddr or lnstage_memtile) and not stage_profile" \
  --csv-output operators/encoder_pipeline/encoder_pipeline_results.csv \
  --iterations 1
```

## Stage-profile tests

- Stage-profile test cases are in `test_encoder_pipeline_stage_profile`.
- They are skipped unless `ENABLE_STAGE_PROFILE_TESTS = True` in `test.py`.
- Once enabled, run only stage-profile tests:

```bash
pytest operators/encoder_pipeline/test.py -q -k stage_profile
```

Run stage profile for one mode:

```bash
pytest operators/encoder_pipeline/test.py -q -k "stage_profile and lnstage_ddr"
```

## Debug-mode sweep script

`profile_debug_modes.py` runs all debug modes for selected cases and reports bottleneck stages.

```bash
python operators/encoder_pipeline/profile_debug_modes.py --design ddr --clean-build
```

Useful script options:

- `--design {ddr,memtile}`
- `--warmup-iters <N>`
- `--timed-iters <N>`
- `--case-index 0,2,4`
- `--extensive`
- `--stop-on-fail`
- `--output-json <path>`
