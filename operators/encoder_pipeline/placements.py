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

# Hardcoded placements derived from the non-sequence-parallel encoder_pipeline_ddr
# topology. For parallel_heads>1, the front half occupies columns [0..parallel_heads-1]
# on rows 2/3/4/5, and the tail tiles come from the DDR one-branch FFN layout.
TOPOLOGY_PLACEMENTS = {
    (12, 64, 64, 32, 64, 96, 1, 1, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0,),
        "tail_tiles": {
            "ln1": (1, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4,),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 128, 64, 32, 64, 96, 1, 1, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0,),
        "tail_tiles": {
            "ln1": (1, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4,),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 256, 64, 32, 64, 96, 1, 1, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0,),
        "tail_tiles": {
            "ln1": (1, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4,),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 64, 64, 32, 64, 96, 1, 2, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1),
        "tail_tiles": {
            "ln1": (2, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 128, 64, 32, 64, 96, 1, 2, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1),
        "tail_tiles": {
            "ln1": (2, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 256, 64, 32, 64, 96, 1, 2, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1),
        "tail_tiles": {
            "ln1": (2, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 64, 64, 32, 64, 96, 1, 4, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1, 2, 3),
        "tail_tiles": {
            "ln1": (4, 5),
            "ffn_up": (6, 3),
            "ffn_down": (7, 3),
            "ln2": (7, 2),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5, 7, 3),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 128, 64, 32, 64, 96, 1, 4, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1, 2, 3),
        "tail_tiles": {
            "ln1": (4, 5),
            "ffn_up": (6, 3),
            "ffn_down": (7, 3),
            "ln2": (7, 2),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5, 7, 3),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (12, 256, 64, 32, 64, 96, 1, 4, 8, 1, 1, 3072): {
        "enabled": True,
        "mha_cols": (0, 1, 2, 3),
        "tail_tiles": {
            "ln1": (4, 5),
            "ffn_up": (6, 3),
            "ffn_down": (7, 3),
            "ln2": (7, 2),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4, 5, 7, 3),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
    (1, 64, 64, 32, 64, 32, 1, 1, 2, 1, 1, 96): {
        "enabled": True,
        "mha_cols": (0,),
        "tail_tiles": {
            "ln1": (1, 5),
            "ffn_up": (5, 5),
            "ffn_down": (6, 5),
            "ln2": (7, 5),
        },
        "accumulation_mem_tiles": {
            "o_proj_acc_by_head": (4,),
            "ln1_replay": 5,
            "ln2_replay": 4,
            "ffn_down_acc_by_branch": (6,),
        },
        "mem_tiles": COMMON_MEM_TILES,
        "shim_tiles": COMMON_SHIM_TILES,
    },
}

SUPPORTED_ENCODER_PIPELINE_TOPOLOGIES = frozenset(
    key for key, placement in TOPOLOGY_PLACEMENTS.items() if placement["enabled"]
)
