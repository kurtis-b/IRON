# encoder_pipeline

Fused encoder operator: `MHA + AddNorm1 + FFN + AddNorm2`.

## File layout

- `design.py`: shared graph/core/runtime logic
- `design_ln1_ddr.py`: LN1 DDR-staging mode hooks
- `design_ln1_memtile.py`: LN1 memtile-staging mode hooks
- `op.py`: operator wrapper and compile/runtime entry
- `reference.py`: golden reference path
- `test.py`: pytest topologies and stage/profile tests

## Modes

- `ln1_staging_design="ddr"`: LN1 output staged to DDR before FFN consumption
- `ln1_staging_design="memtile"`: LN1 output broadcast on-chip to FFN-up consumers

Memtile mode uses LN1 broadcast fanout only (no LN1 route-split helper chain).

## Key parameters

- `parallel_heads`: number of parallel MHA/O-proj lanes
- `o_proj_acc_group_size` (`opg`): grouped O-proj accumulation factor (`1`, `2`, `4`)
- `nB_tiles_distributed`: requested FFN branch distribution
- `proj_acc_depth`: must satisfy `emb_tile * proj_acc_depth == embed_sz`

## Mapping summary

### Fixed MHA block

For lane `i`:
- QK core: `(i,2)`
- softmax core: `(i,3)`
- PV core: `(i,4)`
- O-proj core: `(i,5)`

Host ingress split:
- `Q/K/V/W_O` use memtile columns `0/1/2/3`

### O-proj grouped reduction (`opg > 1`)

1. Local in-group reduction to boundary cores (`opg-1`, `2*opg-1`, ...)
2. Boundary-core memtile accumulation
3. Cross-group reduction along neighbor chain
4. Final boundary core emits to LN1 input

### Tail mapping (LN1/FFN/LN2)

`mapping_validation.py` selects:
- `ln1_tile`
- FFN `up_tiles/down_tiles`
- `ln2_tile`
- down-core neighbor reduction chain

Reduction topology is constrained to:
- full chain through all down cores, or
- chain through all down cores minus one (dual LN2 FFN inputs)

## Core constraint

Layer norm requires full-row stats. LN1/LN2 therefore keep two-pass replay semantics:
- pass 1: sum/sumsq
- pass 2: normalize/output (with required replay fanout/staging)

## Resource constraints

Compute-tile budget check:

`parallel_heads*4 + 3 + 2*effective_ffn_branches <= 32`

Practical limits are usually from:
- memtile DMA channels
- memtile BD blocks
- per-core L1 live-buffer pressure

## Test/profiling entry points

Basic:

```bash
pytest operators/encoder_pipeline/test.py -q
```

Stage profile:

```bash
pytest operators/encoder_pipeline/test.py -q -k stage_profile
```

Debug-mode sweep:

```bash
python operators/encoder_pipeline/profile_debug_modes.py --clean-build
```

## Related docs

- `debug_findings_2026-03-04.md`: active root-cause and fix log
- `resource_utilization_2026-03-04.md`: current resource model and inspection guide
