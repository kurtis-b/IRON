# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

COMMON_MEM_TILES = {
    "q": 0,
    "k": 1,
    "v": 2,
    "w_o": 3,
    "o_proj_acc": 4,
    "b_down": 4,
    "ln2_replay": 4,
    "ln1_replay": 5,
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
    ln1_replay=5,
    ln2_replay=4,
):
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
            "ln1_replay": ln1_replay,
            "ln2_replay": ln2_replay,
            "ffn_down_acc_by_branch": ffn_down_acc_by_branch,
        },
        "weight_mem_tiles": {
            "b_up_by_branch": b_up_by_branch,
            "b_down_by_branch": b_down_by_branch,
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
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
        "ln1_replay": 1,
        "ln2_replay": 3,
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
        "ln1_replay": 1,
        "ln2_replay": 3,
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
    (1, 64, 64, 32, 64, 32, 1, 1, 2, 1, 1, 96): _placement(
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

for seq_len in (64, 128, 256):
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
                ln1_replay=tail.get("ln1_replay", 5),
                ln2_replay=tail.get("ln2_replay", 4),
            )


SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES = frozenset(
    key for key, placement in TOPOLOGY_PLACEMENTS.items() if placement["enabled"]
)
