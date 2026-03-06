#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from operators.encoder_pipeline.design import fused_mha as _fused_mha_base


def fused_mha(*args, **kwargs):
    kwargs["ln1_stage_mode"] = "ddr"
    return _fused_mha_base(*args, **kwargs)
