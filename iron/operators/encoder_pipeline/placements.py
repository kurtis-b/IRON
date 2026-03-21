# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

COMMON_MEM_TILES = {
    "q": 0,
    "k": 1,
    "v": 2,
    "w_o": 3,
    "o_proj_acc": 4,
    "b_down": 4,
    "o_proj_stage": 5,
    "b_up": 5,
    "ffn_down_acc": 6,
    "ln1_stage": 6,
    "residual": 7,
    "ffn_residual": 7,
    "output": 7,
}

COMMON_SHIM_TILES = {
    "q": 0,
    "k": 1,
    "v": 2,
    "w_o": 3,
    "b_down": 4,
    "b_up": 5,
    "ln1_stage": 6,
    "residual": 7,
    "output": 7,
}

O_PROJ_ACC_BY_HEAD = {
    1: (4,),
    2: (4, 5),
    4: (4, 5, 7, 3),
}

SCALED_SEQ_LENS = tuple(1 << exp for exp in range(6, 15))
FFN_TILE_SIZES = (96, 64)


def _placement(
    *,
    mha_cols,
    ln1_tile,
    ffn_up_by_branch,
    ffn_down_by_branch,
    ln2_tile,
    o_proj_acc_by_head,
    ffn_down_acc_by_branch,
    b_up_by_branch,
    b_down_by_branch,
    o_proj_stage=5,
    sequence_parallel=None,
    shim_tiles=None,
):
    resolved_shim_tiles = dict(COMMON_SHIM_TILES)
    if shim_tiles is not None:
        resolved_shim_tiles.update(shim_tiles)
    return {
        "enabled": True,
        "mha_cols": mha_cols,
        "tail_tiles": {
            "ln1": ln1_tile,
            "ffn_up_by_branch": ffn_up_by_branch,
            "ffn_down_by_branch": ffn_down_by_branch,
            "ln2": ln2_tile,
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": o_proj_acc_by_head,
            "o_proj_stage": o_proj_stage,
            "ffn_down_acc_by_branch": ffn_down_acc_by_branch,
        },
        "weight_mem_tiles": {
            "b_up_by_branch": b_up_by_branch,
            "b_down_by_branch": b_down_by_branch,
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": resolved_shim_tiles,
        "sequence_parallel": sequence_parallel,
    }


LOW_HEAD_TAILS = {
    (1, 1): {
        "ln1_tile": (1, 5),
        "ffn_up_by_branch": ((5, 5),),
        "ffn_down_by_branch": ((6, 5),),
        "ln2_tile": (7, 5),
        "ffn_down_acc_by_branch": (6,),
        "b_up_by_branch": (5,),
        "b_down_by_branch": (4,),
    },
    (1, 2): {
        "ln1_tile": (1, 5),
        "ffn_up_by_branch": ((5, 4), (5, 5)),
        "ffn_down_by_branch": ((6, 4), (6, 5)),
        "ln2_tile": (7, 5),
        "ffn_down_acc_by_branch": (6, 2),
        "b_up_by_branch": (5, 5),
        "b_down_by_branch": (4, 4),
    },
    (1, 4): {
        "ln1_tile": (1, 5),
        "ffn_up_by_branch": ((5, 2), (6, 2), (5, 4), (5, 5)),
        "ffn_down_by_branch": ((5, 3), (6, 3), (6, 4), (6, 5)),
        "ln2_tile": (7, 5),
        "o_proj_stage": 1,
        "ffn_down_acc_by_branch": (6, 5, 4, 7),
        "b_up_by_branch": (5, 5, 4, 3),
        "b_down_by_branch": (2, 1, 0, 7),
    },
    (2, 1): {
        "ln1_tile": (2, 5),
        "ffn_up_by_branch": ((5, 5),),
        "ffn_down_by_branch": ((6, 5),),
        "ln2_tile": (7, 5),
        "ffn_down_acc_by_branch": (6,),
        "b_up_by_branch": (5,),
        "b_down_by_branch": (4,),
    },
    (2, 2): {
        "ln1_tile": (2, 5),
        "ffn_up_by_branch": ((5, 4), (5, 5)),
        "ffn_down_by_branch": ((6, 4), (6, 5)),
        "ln2_tile": (7, 5),
        "ffn_down_acc_by_branch": (6, 2),
        "b_up_by_branch": (5, 5),
        "b_down_by_branch": (4, 4),
    },
    (2, 4): {
        "ln1_tile": (2, 5),
        "ffn_up_by_branch": ((5, 2), (6, 2), (5, 4), (5, 5)),
        "ffn_down_by_branch": ((5, 3), (6, 3), (6, 4), (6, 5)),
        "ln2_tile": (7, 5),
        "o_proj_stage": 1,
        "ffn_down_acc_by_branch": (6, 5, 4, 7),
        "b_up_by_branch": (5, 5, 4, 3),
        "b_down_by_branch": (2, 1, 0, 7),
    },
    (4, 1): {
        "ln1_tile": (4, 5),
        "ffn_up_by_branch": ((6, 3),),
        "ffn_down_by_branch": ((7, 3),),
        "ln2_tile": (7, 2),
        "ffn_down_acc_by_branch": (6,),
        "b_up_by_branch": (5,),
        "b_down_by_branch": (4,),
    },
}


TOPOLOGY_PLACEMENTS = {
    (1, 64, 64, 32, 64, 32, 32, 1, 1, 2, 1, 1, 96): _placement(
        mha_cols=(0,),
        ln1_tile=(1, 5),
        ffn_up_by_branch=((5, 5),),
        ffn_down_by_branch=((6, 5),),
        ln2_tile=(7, 5),
        o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
        ffn_down_acc_by_branch=(6,),
        b_up_by_branch=(5,),
        b_down_by_branch=(4,),
    ),
}

for seq_len in SCALED_SEQ_LENS:
    for ffn_tile in FFN_TILE_SIZES:
        for parallel_heads in (1, 2, 4):
            for parallel_ffn in (1, 2, 4):
                tail = LOW_HEAD_TAILS.get((parallel_heads, parallel_ffn))
                if tail is None:
                    continue
                TOPOLOGY_PLACEMENTS[
                    (
                        12,
                        seq_len,
                        64,
                        32,
                        64,
                        96,
                        ffn_tile,
                        1,
                        parallel_heads,
                        8,
                        1,
                        parallel_ffn,
                        3072,
                    )
                ] = _placement(
                    mha_cols=tuple(range(parallel_heads)),
                    ln1_tile=tail["ln1_tile"],
                    ffn_up_by_branch=tail["ffn_up_by_branch"],
                    ffn_down_by_branch=tail["ffn_down_by_branch"],
                    ln2_tile=tail["ln2_tile"],
                    o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[parallel_heads],
                    ffn_down_acc_by_branch=tail["ffn_down_acc_by_branch"],
                    b_up_by_branch=tail["b_up_by_branch"],
                    b_down_by_branch=tail["b_down_by_branch"],
                    o_proj_stage=tail.get("o_proj_stage", 5),
                )

for seq_len in SCALED_SEQ_LENS:
    for ffn_tile in (64,):
        for parallel_heads in (1, 2, 4):
            for parallel_ffn in (1, 2, 4):
                tail = LOW_HEAD_TAILS.get((parallel_heads, parallel_ffn))
                if tail is None:
                    continue
                TOPOLOGY_PLACEMENTS[
                    (
                        16,
                        seq_len,
                        64,
                        32,
                        64,
                        128,
                        ffn_tile,
                        1,
                        parallel_heads,
                        8,
                        1,
                        parallel_ffn,
                        4096,
                    )
                ] = _placement(
                    mha_cols=tuple(range(parallel_heads)),
                    ln1_tile=tail["ln1_tile"],
                    ffn_up_by_branch=tail["ffn_up_by_branch"],
                    ffn_down_by_branch=tail["ffn_down_by_branch"],
                    ln2_tile=tail["ln2_tile"],
                    o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[parallel_heads],
                    ffn_down_acc_by_branch=tail["ffn_down_acc_by_branch"],
                    b_up_by_branch=tail["b_up_by_branch"],
                    b_down_by_branch=tail["b_down_by_branch"],
                    o_proj_stage=tail.get("o_proj_stage", 5),
                )

for seq_len in SCALED_SEQ_LENS:
    for ffn_tile in FFN_TILE_SIZES:
        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                2,
                1,
                8,
                1,
                1,
                3072,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((5, 5),),
            ffn_down_by_branch=((6, 5),),
            ln2_tile=(7, 5),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(6,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 3,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "lane_tiles": (
                    {
                        "qk": (1, 2),
                        "softmax": (1, 3),
                        "pv": (0, 3),
                        "o_proj": (0, 4),
                        "ln1": (0, 5),
                        "ffn_up": (1, 4),
                        "ffn_down": (2, 4),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (5, 2),
                        "softmax": (5, 3),
                        "pv": (4, 3),
                        "o_proj": (4, 4),
                        "ln1": (4, 5),
                        "ffn_up": (5, 5),
                        "ffn_down": (6, 5),
                        "ln2": (7, 5),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 3),
                "lane_tail_mem_cols": (7, 6),
            },
        )

for seq_len in SCALED_SEQ_LENS:
    if (seq_len // 32) % 2 != 0:
        continue
    for ffn_tile in (64,):
        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                2,
                1,
                8,
                1,
                2,
                3072,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((1, 2), (2, 2)),
            ffn_down_by_branch=((1, 3), (2, 3)),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(2, 3),
            b_up_by_branch=(5, 6),
            b_down_by_branch=(4, 5),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 2,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                        "weight_mem_cols": {
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": ((1, 2), (2, 2)),
                        "ffn_down": ((1, 3), (2, 3)),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": ((5, 2), (6, 2)),
                        "ffn_down": ((5, 3), (6, 3)),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 6),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((2, 3), (6, 7)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                2,
                1,
                8,
                1,
                4,
                3072,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((2, 2), (3, 2), (2, 4), (2, 5)),
            ffn_down_by_branch=((2, 3), (3, 3), (3, 4), (3, 5)),
            ln2_tile=(1, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(0, 1, 2, 3),
            b_up_by_branch=(2, 3, 4, 5),
            b_down_by_branch=(4, 5, 6, 7),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 3,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (2, 3, 4, 5),
                            "b_down": (4, 5, 6, 7),
                        },
                        "weight_mem_cols": {
                            "b_up": (2, 3, 4, 5),
                            "b_down": (4, 5, 6, 7),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": ((2, 2), (3, 2), (2, 4), (2, 5)),
                        "ffn_down": ((2, 3), (3, 3), (3, 4), (3, 5)),
                        "ln2": (1, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": ((6, 2), (7, 2), (6, 4), (6, 5)),
                        "ffn_down": ((6, 3), (7, 3), (7, 4), (7, 5)),
                        "ln2": (5, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 6),
                "lane_tail_mem_cols": (1, 5),
                "lane_ffn_down_acc_mem_cols": ((0, 1, 2, 3), (4, 5, 6, 7)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                2,
                2,
                8,
                1,
                1,
                3072,
            )
        ] = _placement(
            mha_cols=(0, 1),
            ln1_tile=(2, 5),
            ffn_up_by_branch=((2, 2),),
            ffn_down_by_branch=((2, 3),),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[2],
            ffn_down_acc_by_branch=(3,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "lane_tiles": (
                    {
                        "qk": ((0, 2), (1, 2)),
                        "softmax": ((0, 3), (1, 3)),
                        "pv": ((0, 4), (1, 4)),
                        "o_proj": ((0, 5), (1, 5)),
                        "ln1": (2, 5),
                        "ffn_up": ((2, 2),),
                        "ffn_down": ((2, 3),),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": ((4, 2), (5, 2)),
                        "softmax": ((4, 3), (5, 3)),
                        "pv": ((4, 4), (5, 4)),
                        "o_proj": ((4, 5), (5, 5)),
                        "ln1": (6, 5),
                        "ffn_up": ((6, 2),),
                        "ffn_down": ((6, 3),),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": ((2, 3), (6, 7)),
                "lane_o_proj_stage_mem_cols": (1, 5),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((3,), (7,)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                2,
                2,
                8,
                1,
                2,
                3072,
            )
        ] = _placement(
            mha_cols=(0, 1),
            ln1_tile=(2, 5),
            ffn_up_by_branch=((2, 2), (3, 2)),
            ffn_down_by_branch=((2, 3), (3, 3)),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[2],
            ffn_down_acc_by_branch=(2, 3),
            b_up_by_branch=(6, 7),
            b_down_by_branch=(4, 5),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 5,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 5,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                        "weight_mem_cols": {
                            "b_up": (6, 7),
                            "b_down": (4, 5),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": ((0, 2), (1, 2)),
                        "softmax": ((0, 3), (1, 3)),
                        "pv": ((0, 4), (1, 4)),
                        "o_proj": ((0, 5), (1, 5)),
                        "ln1": (2, 5),
                        "ffn_up": ((2, 2), (3, 2)),
                        "ffn_down": ((2, 3), (3, 3)),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": ((4, 2), (5, 2)),
                        "softmax": ((4, 3), (5, 3)),
                        "pv": ((4, 4), (5, 4)),
                        "o_proj": ((4, 5), (5, 5)),
                        "ln1": (6, 5),
                        "ffn_up": ((6, 2), (7, 2)),
                        "ffn_down": ((6, 3), (7, 3)),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": ((2, 3), (6, 7)),
                "lane_o_proj_stage_mem_cols": (1, 4),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((2, 3), (6, 7)),
            },
        )

for seq_len in SCALED_SEQ_LENS:
    if (seq_len // 32) % 2 != 0:
        continue
    for ffn_tile in (64,):
        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                2,
                1,
                8,
                1,
                1,
                4096,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((5, 5),),
            ffn_down_by_branch=((6, 5),),
            ln2_tile=(7, 5),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(6,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 3,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "lane_tiles": (
                    {
                        "qk": (1, 2),
                        "softmax": (1, 3),
                        "pv": (0, 3),
                        "o_proj": (0, 4),
                        "ln1": (0, 5),
                        "ffn_up": (1, 4),
                        "ffn_down": (2, 4),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (5, 2),
                        "softmax": (5, 3),
                        "pv": (4, 3),
                        "o_proj": (4, 4),
                        "ln1": (4, 5),
                        "ffn_up": (5, 5),
                        "ffn_down": (6, 5),
                        "ln2": (7, 5),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 3),
                "lane_tail_mem_cols": (7, 6),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                2,
                1,
                8,
                1,
                2,
                4096,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((1, 2), (2, 2)),
            ffn_down_by_branch=((1, 3), (2, 3)),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(2, 3),
            b_up_by_branch=(5, 6),
            b_down_by_branch=(4, 5),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 2,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                        "weight_mem_cols": {
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": ((1, 2), (2, 2)),
                        "ffn_down": ((1, 3), (2, 3)),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": ((5, 2), (6, 2)),
                        "ffn_down": ((5, 3), (6, 3)),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 6),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((2, 3), (6, 7)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                2,
                1,
                8,
                1,
                4,
                4096,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((2, 2), (3, 2), (2, 4), (2, 5)),
            ffn_down_by_branch=((2, 3), (3, 3), (3, 4), (3, 5)),
            ln2_tile=(1, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(0, 1, 2, 3),
            b_up_by_branch=(2, 3, 4, 5),
            b_down_by_branch=(4, 5, 6, 7),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 3,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (2, 3, 4, 5),
                            "b_down": (4, 5, 6, 7),
                        },
                        "weight_mem_cols": {
                            "b_up": (2, 3, 4, 5),
                            "b_down": (4, 5, 6, 7),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": ((2, 2), (3, 2), (2, 4), (2, 5)),
                        "ffn_down": ((2, 3), (3, 3), (3, 4), (3, 5)),
                        "ln2": (1, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": ((6, 2), (7, 2), (6, 4), (6, 5)),
                        "ffn_down": ((6, 3), (7, 3), (7, 4), (7, 5)),
                        "ln2": (5, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (2, 6),
                "lane_tail_mem_cols": (1, 5),
                "lane_ffn_down_acc_mem_cols": ((0, 1, 2, 3), (4, 5, 6, 7)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                2,
                2,
                8,
                1,
                1,
                4096,
            )
        ] = _placement(
            mha_cols=(0, 1),
            ln1_tile=(2, 5),
            ffn_up_by_branch=((2, 2),),
            ffn_down_by_branch=((2, 3),),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[2],
            ffn_down_acc_by_branch=(3,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 2,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "lane_tiles": (
                    {
                        "qk": ((0, 2), (1, 2)),
                        "softmax": ((0, 3), (1, 3)),
                        "pv": ((0, 4), (1, 4)),
                        "o_proj": ((0, 5), (1, 5)),
                        "ln1": (2, 5),
                        "ffn_up": ((2, 2),),
                        "ffn_down": ((2, 3),),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": ((4, 2), (5, 2)),
                        "softmax": ((4, 3), (5, 3)),
                        "pv": ((4, 4), (5, 4)),
                        "o_proj": ((4, 5), (5, 5)),
                        "ln1": (6, 5),
                        "ffn_up": ((6, 2),),
                        "ffn_down": ((6, 3),),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": ((2, 3), (6, 7)),
                "lane_o_proj_stage_mem_cols": (1, 5),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((3,), (7,)),
            },
        )

        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                2,
                2,
                8,
                1,
                2,
                4096,
            )
        ] = _placement(
            mha_cols=(0, 1),
            ln1_tile=(2, 5),
            ffn_up_by_branch=((2, 2), (3, 2)),
            ffn_down_by_branch=((2, 3), (3, 3)),
            ln2_tile=(3, 4),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[2],
            ffn_down_acc_by_branch=(2, 3),
            b_up_by_branch=(6, 7),
            b_down_by_branch=(4, 5),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 0,
                    "ln1_stage": 5,
                    "ln1_refill": 4,
                    "ffn_residual_refill": 5,
                    "output": 1,
                },
                "joined_or_shim_cols": {
                    "residual": 7,
                    "ln1_stage": 6,
                    "ln1_refill": 0,
                    "ffn_residual_refill": 1,
                    "output": 7,
                },
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 5,
                            "ln1_refill": 4,
                            "ffn_residual_refill": 5,
                            "output": 1,
                        },
                        "joined_or_shim_cols": {
                            "residual": 7,
                            "ln1_stage": 6,
                            "ln1_refill": 0,
                            "ffn_residual_refill": 1,
                            "output": 7,
                        },
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": (5, 6),
                            "b_down": (4, 5),
                        },
                        "weight_mem_cols": {
                            "b_up": (6, 7),
                            "b_down": (4, 5),
                        },
                    },
                ),
                "lane_tiles": (
                    {
                        "qk": ((0, 2), (1, 2)),
                        "softmax": ((0, 3), (1, 3)),
                        "pv": ((0, 4), (1, 4)),
                        "o_proj": ((0, 5), (1, 5)),
                        "ln1": (2, 5),
                        "ffn_up": ((2, 2), (3, 2)),
                        "ffn_down": ((2, 3), (3, 3)),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": ((4, 2), (5, 2)),
                        "softmax": ((4, 3), (5, 3)),
                        "pv": ((4, 4), (5, 4)),
                        "o_proj": ((4, 5), (5, 5)),
                        "ln1": (6, 5),
                        "ffn_up": ((6, 2), (7, 2)),
                        "ffn_down": ((6, 3), (7, 3)),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": ((2, 3), (6, 7)),
                "lane_o_proj_stage_mem_cols": (1, 4),
                "lane_tail_mem_cols": (3, 7),
                "lane_ffn_down_acc_mem_cols": ((2, 3), (6, 7)),
            },
        )

for seq_len in SCALED_SEQ_LENS:
    if (seq_len // 32) % 4 != 0:
        continue
    for ffn_tile in FFN_TILE_SIZES:
        TOPOLOGY_PLACEMENTS[
            (
                12,
                seq_len,
                64,
                32,
                64,
                96,
                ffn_tile,
                4,
                1,
                8,
                1,
                1,
                3072,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((5, 5),),
            ffn_down_by_branch=((6, 5),),
            ln2_tile=(7, 5),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(6,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 4,
                    "ln1_stage": 5,
                    "ln1_refill": 6,
                    "ffn_residual_refill": 7,
                    "output": 0,
                },
                "joined_or_shim_cols": {
                    "residual": 4,
                    "ln1_stage": 5,
                    "ln1_refill": 6,
                    "ffn_residual_refill": 7,
                    "output": 0,
                },
                "unified_qr_split": {
                    "q_mem_col": 0,
                    "q_shim_col": 0,
                    "residual_mem_col": 4,
                    "residual_shim_col": 4,
                },
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": (1, 2),
                        "ffn_down": (1, 3),
                        "ln2": (1, 4),
                    },
                    {
                        "qk": (2, 2),
                        "softmax": (2, 3),
                        "pv": (2, 4),
                        "o_proj": (2, 5),
                        "ln1": (3, 5),
                        "ffn_up": (3, 2),
                        "ffn_down": (3, 3),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": (5, 2),
                        "ffn_down": (5, 3),
                        "ln2": (5, 4),
                    },
                    {
                        "qk": (6, 2),
                        "softmax": (6, 3),
                        "pv": (6, 4),
                        "o_proj": (6, 5),
                        "ln1": (7, 5),
                        "ffn_up": (7, 2),
                        "ffn_down": (7, 3),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (1, 2, 5, 6),
                "lane_tail_mem_cols": (1, 3, 5, 7),
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": 4,
                            "b_down": 5,
                        },
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 3,
                            "ln1_refill": 2,
                            "ffn_residual_refill": 3,
                            "output": 0,
                        },
                        "joined_or_shim_cols": {
                            "residual": 4,
                            "ln1_stage": 5,
                            "ln1_refill": 6,
                            "ffn_residual_refill": 7,
                            "output": 4,
                        },
                        "weight_mem_cols": {"b_up": 3, "b_down": 2},
                    },
                    {
                        "lanes": (2, 3),
                        "joined_q_mem_col": 4,
                        "shared_ingress_cols": {"k": 5, "v": 6, "w_o": 7},
                        "shim_cols": {
                            "q": 4,
                            "k": 5,
                            "v": 6,
                            "w_o": 7,
                            "b_up": 0,
                            "b_down": 1,
                        },
                        "joined_or_mem_cols": {
                            "residual": 4,
                            "ln1_stage": 7,
                            "ln1_refill": 6,
                            "ffn_residual_refill": 7,
                            "output": 4,
                        },
                        "joined_or_shim_cols": {
                            "residual": 0,
                            "ln1_stage": 1,
                            "ln1_refill": 2,
                            "ffn_residual_refill": 3,
                            "output": 0,
                        },
                        "weight_mem_cols": {"b_up": 7, "b_down": 6},
                    },
                ),
            },
        )

for seq_len in SCALED_SEQ_LENS:
    if (seq_len // 32) % 4 != 0:
        continue
    for ffn_tile in (64,):
        TOPOLOGY_PLACEMENTS[
            (
                16,
                seq_len,
                64,
                32,
                64,
                128,
                ffn_tile,
                4,
                1,
                8,
                1,
                1,
                4096,
            )
        ] = _placement(
            mha_cols=(0,),
            ln1_tile=(1, 5),
            ffn_up_by_branch=((5, 5),),
            ffn_down_by_branch=((6, 5),),
            ln2_tile=(7, 5),
            o_proj_acc_by_head=O_PROJ_ACC_BY_HEAD[1],
            ffn_down_acc_by_branch=(6,),
            b_up_by_branch=(5,),
            b_down_by_branch=(4,),
            sequence_parallel={
                "joined_q_mem_col": 0,
                "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                "joined_or_mem_cols": {
                    "residual": 4,
                    "ln1_stage": 5,
                    "ln1_refill": 6,
                    "ffn_residual_refill": 7,
                    "output": 0,
                },
                "joined_or_shim_cols": {
                    "residual": 4,
                    "ln1_stage": 5,
                    "ln1_refill": 6,
                    "ffn_residual_refill": 7,
                    "output": 0,
                },
                "unified_qr_split": {
                    "q_mem_col": 0,
                    "q_shim_col": 0,
                    "residual_mem_col": 4,
                    "residual_shim_col": 4,
                },
                "lane_tiles": (
                    {
                        "qk": (0, 2),
                        "softmax": (0, 3),
                        "pv": (0, 4),
                        "o_proj": (0, 5),
                        "ln1": (1, 5),
                        "ffn_up": (1, 2),
                        "ffn_down": (1, 3),
                        "ln2": (1, 4),
                    },
                    {
                        "qk": (2, 2),
                        "softmax": (2, 3),
                        "pv": (2, 4),
                        "o_proj": (2, 5),
                        "ln1": (3, 5),
                        "ffn_up": (3, 2),
                        "ffn_down": (3, 3),
                        "ln2": (3, 4),
                    },
                    {
                        "qk": (4, 2),
                        "softmax": (4, 3),
                        "pv": (4, 4),
                        "o_proj": (4, 5),
                        "ln1": (5, 5),
                        "ffn_up": (5, 2),
                        "ffn_down": (5, 3),
                        "ln2": (5, 4),
                    },
                    {
                        "qk": (6, 2),
                        "softmax": (6, 3),
                        "pv": (6, 4),
                        "o_proj": (6, 5),
                        "ln1": (7, 5),
                        "ffn_up": (7, 2),
                        "ffn_down": (7, 3),
                        "ln2": (7, 4),
                    },
                ),
                "lane_o_proj_acc_mem_cols": (1, 2, 5, 6),
                "lane_tail_mem_cols": (1, 3, 5, 7),
                "transport_groups": (
                    {
                        "lanes": (0, 1),
                        "joined_q_mem_col": 0,
                        "shared_ingress_cols": {"k": 1, "v": 2, "w_o": 3},
                        "shim_cols": {
                            "q": 0,
                            "k": 1,
                            "v": 2,
                            "w_o": 3,
                            "b_up": 4,
                            "b_down": 5,
                        },
                        "joined_or_mem_cols": {
                            "residual": 0,
                            "ln1_stage": 3,
                            "ln1_refill": 2,
                            "ffn_residual_refill": 3,
                            "output": 0,
                        },
                        "joined_or_shim_cols": {
                            "residual": 4,
                            "ln1_stage": 5,
                            "ln1_refill": 6,
                            "ffn_residual_refill": 7,
                            "output": 4,
                        },
                        "weight_mem_cols": {"b_up": 3, "b_down": 2},
                    },
                    {
                        "lanes": (2, 3),
                        "joined_q_mem_col": 4,
                        "shared_ingress_cols": {"k": 5, "v": 6, "w_o": 7},
                        "shim_cols": {
                            "q": 4,
                            "k": 5,
                            "v": 6,
                            "w_o": 7,
                            "b_up": 0,
                            "b_down": 1,
                        },
                        "joined_or_mem_cols": {
                            "residual": 4,
                            "ln1_stage": 7,
                            "ln1_refill": 6,
                            "ffn_residual_refill": 7,
                            "output": 4,
                        },
                        "joined_or_shim_cols": {
                            "residual": 0,
                            "ln1_stage": 1,
                            "ln1_refill": 2,
                            "ffn_residual_refill": 3,
                            "output": 0,
                        },
                        "weight_mem_cols": {"b_up": 7, "b_down": 6},
                    },
                ),
            },
        )


def register_mirrored_family(
    *,
    source_seq_tile: int,
    source_kv_seq_tile: int,
    target_seq_tile: int,
    target_kv_seq_tile: int,
):
    existing_entries = sorted(TOPOLOGY_PLACEMENTS.items())
    for key, placement in existing_entries:
        (
            num_heads,
            seq_len,
            d,
            seq_tile,
            kv_seq_tile,
            emb_tile,
            ffn_tile,
            parallel_seq,
            parallel_heads,
            proj_acc_depth,
            o_proj_acc_group_size,
            parallel_ffn,
            ffn_intermediate_size,
        ) = key
        if num_heads not in (12, 16):
            continue
        if (seq_tile, kv_seq_tile) != (source_seq_tile, source_kv_seq_tile):
            continue
        if seq_len % target_seq_tile != 0 or seq_len % target_kv_seq_tile != 0:
            continue
        if (seq_len // target_seq_tile) % parallel_seq != 0:
            continue
        mirrored_key = (
            num_heads,
            seq_len,
            d,
            target_seq_tile,
            target_kv_seq_tile,
            emb_tile,
            ffn_tile,
            parallel_seq,
            parallel_heads,
            proj_acc_depth,
            o_proj_acc_group_size,
            parallel_ffn,
            ffn_intermediate_size,
        )
        if mirrored_key in TOPOLOGY_PLACEMENTS:
            continue
        TOPOLOGY_PLACEMENTS[mirrored_key] = deepcopy(placement)


register_mirrored_family(
    source_seq_tile=32,
    source_kv_seq_tile=64,
    target_seq_tile=64,
    target_kv_seq_tile=32,
)


SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES = frozenset(
    key for key, placement in TOPOLOGY_PLACEMENTS.items() if placement["enabled"]
)
