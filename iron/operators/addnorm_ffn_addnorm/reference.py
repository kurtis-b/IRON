# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from iron.operators.ffn_addnorm.reference import (
    generate_golden_reference as generate_ffn_addnorm_reference,
)


def generate_golden_reference(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int = 42,
    debug_mode: int = -1,
) -> dict[str, object]:
    golden = generate_ffn_addnorm_reference(
        M=seq_len,
        K=hidden_size,
        N=intermediate_size,
        seed=seed,
        debug_mode=debug_mode,
    )
    return {
        "hidden_states": golden["input"],
        "residual": golden["input_residual"],
        "ffn_up_weight": golden["input_b_up"],
        "ffn_down_weight": golden["input_b_down"],
        "ln2_weight": golden["weight2"],
        "output": golden["output"],
    }
