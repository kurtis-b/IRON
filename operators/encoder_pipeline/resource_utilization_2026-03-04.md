# Encoder Pipeline Resource Utilization (2026-03-04)

Source artifacts: `build/encoder_pipeline_*_lnstage.mlir.prj/{input_physical.mlir,input_with_addresses.mlir}`

Assumptions used for percentages:
- Compute tile L1 capacity: 64 KiB
- Mem tile L2 capacity: 512 KiB
- DMA channels per compute tile direction: 2
- DMA channels per mem tile direction: 6

## Summary Table

| Design | Compute Tiles Used | Peak L1 (tile) | Peak L2 (mem tile) | Peak Compute DMA In/Out | Peak Mem DMA In/Out |
|---|---:|---|---|---|---|
| `12h_1024s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 29 | 57,364 B (87.5%) on `tile_6_5` | 147,456 B (28.1%) on `mem_tile_5_1` | 2/2 in (`tile_6_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 6/6 out (`mem_tile_1_1`) |
| `12h_128s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 9 | 57,364 B (87.5%) on `tile_1_5` | 114,688 B (21.9%) on `mem_tile_7_1` | 2/2 in (`tile_1_5`), 2/2 out (`tile_1_2`) | 4/6 in (`mem_tile_7_1`), 4/6 out (`mem_tile_7_1`) |
| `12h_2048s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 9 | 57,364 B (87.5%) on `tile_1_5` | 114,688 B (21.9%) on `mem_tile_7_1` | 2/2 in (`tile_1_5`), 2/2 out (`tile_1_2`) | 4/6 in (`mem_tile_7_1`), 4/6 out (`mem_tile_7_1`) |
| `12h_2048s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 29 | 57,364 B (87.5%) on `tile_6_5` | 147,456 B (28.1%) on `mem_tile_5_1` | 2/2 in (`tile_6_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 6/6 out (`mem_tile_1_1`) |
| `12h_512s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 9 | 57,364 B (87.5%) on `tile_1_5` | 114,688 B (21.9%) on `mem_tile_7_1` | 2/2 in (`tile_1_5`), 2/2 out (`tile_1_2`) | 4/6 in (`mem_tile_7_1`), 4/6 out (`mem_tile_7_1`) |
| `12h_512s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 29 | 57,364 B (87.5%) on `tile_6_5` | 147,456 B (28.1%) on `mem_tile_5_1` | 2/2 in (`tile_6_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 6/6 out (`mem_tile_1_1`) |
| `12h_64s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 9 | 57,364 B (87.5%) on `tile_1_5` | 114,688 B (21.9%) on `mem_tile_7_1` | 2/2 in (`tile_1_5`), 2/2 out (`tile_1_2`) | 4/6 in (`mem_tile_7_1`), 4/6 out (`mem_tile_7_1`) |
| `12h_64s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 29 | 57,364 B (87.5%) on `tile_6_5` | 147,456 B (28.1%) on `mem_tile_5_1` | 2/2 in (`tile_6_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 6/6 out (`mem_tile_1_1`) |
| `16h_1024s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 21 | 57,364 B (87.5%) on `tile_4_5` | 172,032 B (32.8%) on `mem_tile_6_1` | 2/2 in (`tile_4_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 5/6 out (`mem_tile_3_1`) |
| `16h_2048s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 21 | 57,364 B (87.5%) on `tile_4_5` | 172,032 B (32.8%) on `mem_tile_6_1` | 2/2 in (`tile_4_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 5/6 out (`mem_tile_3_1`) |
| `16h_512s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 21 | 57,364 B (87.5%) on `tile_4_5` | 172,032 B (32.8%) on `mem_tile_6_1` | 2/2 in (`tile_4_5`), 2/2 out (`tile_6_2`) | 4/6 in (`mem_tile_6_1`), 5/6 out (`mem_tile_3_1`) |

## Block Locations (MHA / AddNorm / FFN)

| Design | MHA Tiles | AddNorm Tiles | FFN Tiles |
|---|---|---|---|
| `12h_1024s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5), (4,2), (4,3), (4,4), (4,5), (5,2), (5,3), (5,4), (5,5) | LN1-norm: (6,3); LN1-post: (6,2); LN2: (7,5) | up: (6,4); down: (6,5) |
| `12h_128s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5) | LN1-norm: (1,3); LN1-post: (1,2); LN2: (2,5) | up: (1,4); down: (1,5) |
| `12h_2048s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5) | LN1-norm: (1,3); LN1-post: (1,2); LN2: (2,5) | up: (1,4); down: (1,5) |
| `12h_2048s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5), (4,2), (4,3), (4,4), (4,5), (5,2), (5,3), (5,4), (5,5) | LN1-norm: (6,3); LN1-post: (6,2); LN2: (7,5) | up: (6,4); down: (6,5) |
| `12h_512s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5) | LN1-norm: (1,3); LN1-post: (1,2); LN2: (2,5) | up: (1,4); down: (1,5) |
| `12h_512s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5), (4,2), (4,3), (4,4), (4,5), (5,2), (5,3), (5,4), (5,5) | LN1-norm: (6,3); LN1-post: (6,2); LN2: (7,5) | up: (6,4); down: (6,5) |
| `12h_64s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5) | LN1-norm: (1,3); LN1-post: (1,2); LN2: (2,5) | up: (1,4); down: (1,5) |
| `12h_64s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5), (4,2), (4,3), (4,4), (4,5), (5,2), (5,3), (5,4), (5,5) | LN1-norm: (6,3); LN1-post: (6,2); LN2: (7,5) | up: (6,4); down: (6,5) |
| `16h_1024s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5) | LN1-norm: (4,3); LN1-post: (6,2); LN2: (5,5) | up: (4,4); down: (4,5) |
| `16h_2048s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5) | LN1-norm: (4,3); LN1-post: (6,2); LN2: (5,5) | up: (4,4); down: (4,5) |
| `16h_512s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3), (2,4), (2,5), (3,2), (3,3), (3,4), (3,5) | LN1-norm: (4,3); LN1-post: (6,2); LN2: (5,5) | up: (4,4); down: (4,5) |

## Totals (Allocated Buffers)

| Design | Total L1 on Used Compute Tiles | Total L2 on Mem Tiles |
|---|---:|---:|
| `12h_1024s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 1,016,408 B | 892,928 B |
| `12h_128s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 356,768 B | 360,448 B |
| `12h_2048s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 356,768 B | 360,448 B |
| `12h_2048s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 1,016,408 B | 892,928 B |
| `12h_512s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 356,768 B | 360,448 B |
| `12h_512s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 1,016,408 B | 892,928 B |
| `12h_64s_64d_32qt_64kvt_128e_1ph_6acc_1nbdist_3072ffn` | 356,768 B | 360,448 B |
| `12h_64s_64d_32qt_64kvt_128e_6ph_6acc_3nbdist_3072ffn` | 1,016,408 B | 892,928 B |
| `16h_1024s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 761,768 B | 794,624 B |
| `16h_2048s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 761,768 B | 794,624 B |
| `16h_512s_64d_32qt_64kvt_128e_4ph_8acc_3nbdist_4096ffn` | 761,768 B | 794,624 B |

## Notes
- The `seq_len` variants with the same `(heads, parallel_heads, proj_acc_depth, nB_tiles_distributed)` have identical placement/utilization, which matches expectations because tile graph shape is independent of sequence length here.
- Peak compute-tile L1 is consistently ~87.5%, so L1 headroom is limited but non-zero in all tested designs.
- Highest mem-tile DMA output pressure appears in `6pheads` designs (6/6 channels on at least one mem tile).
