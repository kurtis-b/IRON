#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from iron.applications.transformer_layer.src.pipeline.operator_runlist_worker import (
    benchmark_operator_runlist_request,
    main,
    parity_operator_runlist_request,
)

__all__ = [
    "benchmark_operator_runlist_request",
    "parity_operator_runlist_request",
]


if __name__ == "__main__":
    main()
