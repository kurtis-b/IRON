#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

rm -r build/
pytest operators/gemm/
pytest operators/transpose/
pytest operators/elementwise_mul/
pytest operators/elementwise_add/
pytest operators/softmax/
pytest operators/layer_norm/