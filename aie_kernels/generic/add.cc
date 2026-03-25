// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out> void eltwise_add(T_in *a, T_in *b, T_out *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

template <typename T_in, typename T_out> void eltwise_vadd(T_in *a, T_in *b, T_out *c, int size)
{

    constexpr int vec_factor = 16;
    event0();
    T_in *__restrict pA1 = a;
    T_in *__restrict pB1 = b;
    T_out *__restrict pC1 = c;
    const int F = size / vec_factor;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        pA1 += vec_factor;
        aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
        pB1 += vec_factor;
        aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
        aie::store_v(pC1, cout);
        pC1 += vec_factor;
    }
    event1();
}

extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_add<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

void eltwise_add_f32_vector(float *a_in, float *b_in, float *c_out, int size)
{
    eltwise_vadd<float, float>(a_in, b_in, c_out, size);
}

#ifndef EXCLUDE_TILE_BIAS_ADD_KERNELS
void eltwise_add_bf16_tile_bias_vector(
    bfloat16 *tile_in, bfloat16 *bias_in, bfloat16 *tile_out, int rows, int cols
)
{
    constexpr int vec_factor = 16;
    event0();
    for (int row = 0; row < rows; ++row) {
        bfloat16 *pIn = tile_in + row * cols;
        bfloat16 *pOut = tile_out + row * cols;
        bfloat16 *pBias = bias_in;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(4)
        for (int col = 0; col < cols; col += vec_factor) {
            aie::vector<bfloat16, vec_factor> in_v = aie::load_v<vec_factor>(pIn);
            aie::vector<bfloat16, vec_factor> bias_v = aie::load_v<vec_factor>(pBias);
            aie::store_v(pOut, aie::add(in_v, bias_v));
            pIn += vec_factor;
            pOut += vec_factor;
            pBias += vec_factor;
        }
    }
    event1();
}

void eltwise_add_bf16_tile_bias_matrix(
    bfloat16 *tile_in,
    bfloat16 *bias_in,
    bfloat16 *tile_out,
    int bias_row_idx,
    int rows,
    int cols
)
{
    eltwise_add_bf16_tile_bias_vector(
        tile_in, bias_in + bias_row_idx * cols, tile_out, rows, cols
    );
}
#endif

} // extern "C"
