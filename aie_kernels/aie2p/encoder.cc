// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void fused_add_layer_norm(const T *restrict input,
                          const T *restrict residual,
                          const T *restrict weight,
                          T *restrict output,
                          int32_t cols,
                          int32_t rows_to_process)
{
    event0();
    constexpr float epsilon = 1e-5f;
    int vector_chunks = cols / N;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int row = 0; row < rows_to_process; row++) {

        ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
        ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();

        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input);
            sum_acc = ::aie::add(sum_acc, reg_a);
            ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
            sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
            input += N;
        }

        input -= cols; // reset for next calculations

        float sum_of_vals = ::aie::reduce_add(sum_acc);
        float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

        float mean = sum_of_vals / float(cols);
        float mean_sq = mean * mean;
        float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
        float inv_std = aie::invsqrt(variance + epsilon);

        ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
        ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

        for (int i = 0; i < vector_chunks; i++) {

            ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input);
            ::aie::vector<T, N> reg_weight = ::aie::load_v<N>(weight);
            ::aie::vector<T, N> reg_res = ::aie::load_v<N>(residual);
            ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
            ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
            ::aie::vector<T, N> scaled_v = aie::mul(norm_v, reg_weight);
            // ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
            ::aie::vector<T, N> out_v = ::aie::add(scaled_v, reg_res);
            ::aie::store_v(output, out_v);
            input += N;
            weight += N;
            residual += N;
            output += N;
        }
        weight -= cols;
    }
    event1();
}
template <typename T, int N>
void passThrough_aie(T *restrict in, T *restrict out, int32_t K, int32_t k, int32_t m, int32_t col_offset)
{
    event0();

    v64uint8 *restrict outPtr = (v64uint8 *)out;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned row = 0; row < m; row++) {

        v64uint8 *restrict inPtr = (v64uint8 *)(in + row * K + col_offset * k);

        for (int j = 0; j < k; j += N) { // Nx samples per loop

            *outPtr++ = *inPtr++;
        }
    }

    event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef ADD_NORM_LAYER

void fused_add_layer_norm(bfloat16 *input,
                          bfloat16 *residual,
                          bfloat16 *weights,
                          bfloat16 *output,
                          int32_t cols,
                          int32_t rows_to_process)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    fused_add_layer_norm<bfloat16, 32>(input, residual, weights, output, cols, rows_to_process);
}
void passThroughTile(int16_t *in,
                     int16_t *out,
                     int32_t cols,
                     int32_t cols_to_process,
                     int32_t rows_to_process,
                     int32_t col_offset)
{
    passThrough_aie<int16_t, 32>(in, out, cols, cols_to_process, rows_to_process, col_offset);
}
#endif
}
