#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

rm -r build/
pytest operators/gemm/ --csv-output operators_gemm_results.csv --iterations 1
pytest operators/transpose/ --csv-output operators_transpose_results.csv --iterations 1
pytest operators/elementwise_mul/ --csv-output operators_elementwise_mul_results.csv --iterations 1
pytest operators/elementwise_add/ --csv-output operators_elementwise_add_results.csv --iterations 1
pytest operators/softmax/ --csv-output operators_softmax_results.csv --iterations 1
pytest operators/layer_norm/ --csv-output operators_layer_norm_results.csv --iterations 1
pytest operators/gelu/ --csv-output operators_gelu_results.csv --iterations 1
pytest operators/encoder/ --csv-output operators_encoder_results.csv --iterations 1
pytest operators/ffn/ --csv-output operators_ffn_results.csv --iterations 1
pytest operators/add_and_norm/ --csv-output operators_add_and_norm_results.csv --iterations 1
pyteset operators/mha/ --csv-output operators_mha_results.csv --iterations 1
cat operators*.csv > bert_operators_results.csv