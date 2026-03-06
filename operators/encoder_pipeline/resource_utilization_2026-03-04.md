# Encoder Pipeline Resource Utilization (Cleaned)

Last updated: 2026-03-06

## Data source
- Physical artifacts from `build/encoder_pipeline_*_lnstage.mlir.prj/{input_physical.mlir,input_with_addresses.mlir}`.
- Capacity assumptions:
  - Compute-tile L1: 64 KiB
  - Mem-tile L2: 512 KiB
  - Compute-tile DMA channels: 2 in / 2 out
  - Mem-tile DMA channels: 6 in / 6 out

## Topology-level summary

The previous table had repeated entries across sequence lengths. The utilization pattern is effectively topology-dependent, not `seq_len`-dependent, for the measured set.

| Topology Class | Covered designs | Compute tiles | Peak compute L1 | Peak mem L2 | Peak compute DMA | Peak mem DMA |
|---|---|---:|---|---|---|---|
| `12h, 1ph, 1nbdist, 6acc, 3072ffn` | `64s`, `128s`, `512s`, `2048s` | 9 | 57,364 B (87.5%) on `tile_1_5` | 114,688 B (21.9%) on `mem_tile_7_1` | 2/2 in (`tile_1_5`), 2/2 out (`tile_1_2`) | 4/6 in, 4/6 out (`mem_tile_7_1`) |
| `12h, 6ph, 3nbdist, 6acc, 3072ffn` | `64s`, `512s`, `1024s`, `2048s` | 29 | 57,364 B (87.5%) on `tile_6_5` | 147,456 B (28.1%) on `mem_tile_5_1` | 2/2 in (`tile_6_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 6/6 out (`mem_tile_1_1`) |
| `16h, 4ph, 3nbdist, 8acc, 4096ffn` | `512s`, `1024s`, `2048s` | 21 | 57,364 B (87.5%) on `tile_4_5` | 172,032 B (32.8%) on `mem_tile_6_1` | 2/2 in (`tile_4_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 5/6 out (`mem_tile_3_1`) |

## Block locations by topology

| Topology Class | MHA | AddNorm | FFN |
|---|---|---|---|
| `12h, 1ph, 1nbdist, 6acc` | `(0,2..5)` | `LN1-norm (1,3), LN1-post (1,2), LN2 (2,5)` | `up (1,4), down (1,5)` |
| `12h, 6ph, 3nbdist, 6acc` | `(0..5, 2..5)` | `LN1-norm (6,3), LN1-post (6,2), LN2 (7,5)` | `up (6,4), down (6,5)` |
| `16h, 4ph, 3nbdist, 8acc` | `(0..3, 2..5)` | `LN1-norm (4,3), LN1-post (6,2), LN2 (5,5)` | `up (4,4), down (4,5)` |

## Total allocated memory (from measured artifacts)

| Topology Class | Total L1 on used compute tiles | Total L2 on mem tiles |
|---|---:|---:|
| `12h, 1ph, 1nbdist, 6acc, 3072ffn` | 356,768 B | 360,448 B |
| `12h, 6ph, 3nbdist, 6acc, 3072ffn` | 1,016,408 B | 892,928 B |
| `16h, 4ph, 3nbdist, 8acc, 4096ffn` | 761,768 B | 794,624 B |

## Revalidation status (2026-03-06)
- Functional regression check after latest OR/LN1 staging-buffer layout update:
  - `pytest operators/encoder_pipeline/test.py -q --iterations 1`
  - result: `5 passed`
- Default resource shape for baseline path is unchanged by this host-buffer layout update.

## LN1 DDR staging caveat
- `ENCODER_STAGE_LN1_TO_DDR=1` introduces extra shim/memtile traffic for LN1->FFN handoff.
- Timeout on the targeted `1pheads_1pffn_6pacc` case was fixed via runtime synchronization/order updates.
- The mode is still not globally feasible for all tested topologies due compile-time resource limits (output DMA-channel and memtile-BD pressure), so it is not included in the stable utilization snapshot above.
