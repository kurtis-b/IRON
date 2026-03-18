#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

raise SystemExit(
    "iron/applications/bert/inference.py is a legacy entrypoint. "
    "Use iron/applications/bert/cpu_inference.py for the Hugging Face CPU baseline "
    "or iron/applications/bert/npu_inference.py for the encoder_pipeline NPU benchmark."
)
