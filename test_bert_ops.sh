#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

rm -r build/
pytest operators/addnorm/ --csv-output operators_addnorm_results.csv --iterations 1
pytest operators/addnorm_with_layout_transform/ --csv-output operators_addnorm_with_layout_transform_results.csv --iterations 1
pytest operators/elementwise_add/ --csv-output operators_elementwise_add_results.csv --iterations 1
pytest operators/elementwise_mul/ --csv-output operators_elementwise_mul_results.csv --iterations 1
pytest operators/encoder/ --csv-output operators_encoder_results.csv --iterations 1
pytest operators/encoder_pipeline/ --csv-output operators_encoder_pipeline_results.csv --iterations 1
pytest operators/ffn/ --csv-output operators_ffn_results.csv --iterations 1
pytest operators/ffn_addnorm/ --csv-output operators_ffn_addnorm_results.csv --iterations 1
pytest operators/gelu/ --csv-output operators_gelu_results.csv --iterations 1
pytest operators/gemm/ --csv-output operators_gemm_results.csv --iterations 1
pytest operators/layer_norm/ --csv-output operators_layer_norm_results.csv --iterations 1
pytest operators/mha/ --csv-output operators_mha_results.csv --iterations 1
pytest operators/mha_out_proj/ --csv-output operators_mha_out_proj_results.csv --iterations 1
pytest operators/mha_to_an/ --csv-output operators_mha_to_an_results.csv --iterations 1
pytest operators/softmax/ --csv-output operators_softmax_results.csv --iterations 1
pytest operators/transpose/ --csv-output operators_transpose_results.csv --iterations 1
cat operators*.csv > bert_operators_results.csv
pytest applications/bert/ --csv-output bert_application_results.csv --iterations 1
