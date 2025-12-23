#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

rm -r build/
pytest operators/gemm/ --csv-output operators_gemm_results.csv
pytest operators/transpose/ --csv-output operators_transpose_results.csv
pytest operators/elementwise_mul/ --csv-output operators_elementwise_mul_results.csv
pytest operators/elementwise_add/ --csv-output operators_elementwise_add_results.csv
pytest operators/softmax/ --csv-output operators_softmax_results.csv
pytest operators/layer_norm/ --csv-output operators_layer_norm_results.csv
pytest operators/gelu/ --csv-output operators_gelu_results.csv
cat operators*.csv > bert_operators_results.csv