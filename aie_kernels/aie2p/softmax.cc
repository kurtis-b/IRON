// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SM_VEC_LEN
#define SM_VEC_LEN 64
#endif
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

void softmax_simple_bf16(bfloat16 *restrict input_vector, bfloat16 *restrict output_vector, const int32_t vector_size)
{
    event0();

    auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
    auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
    auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
    aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

    float max_val = std::numeric_limits<float>::lowest();
    float accum_exp_val = 0;
    float running_max = std::numeric_limits<float>::lowest();
    bfloat16 col_sum_inv;
    const int elem_iters = vector_size / SM_VEC_LEN;

    exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

    log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);

    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_log_in++;
        scaled_accum = aie::mul(input_bf16, log2e_vec);
        running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (running_max > max_val) {
            max_val = running_max;
        }
    }
    max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_exp_in++;

        scaled_accum = aie::mul(input_bf16, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);

        *it_exp_out++ = exp_val;
    }

    aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);
    col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

    for (int c = 0; c < elem_iters; c++) {
        in_elems = *it_scale++;
        out_vals = aie::mul(in_elems, col_sum_inv);
        *it_soft_out++ = out_vals.to_vector<bfloat16>();
    }

    event1();
}

void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                bfloat16 *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx,
                                const int32_t num_rows,
                                const bfloat16 scale)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
    aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

    float max_val = std::numeric_limits<float>::lowest();
    float accum_exp_val = 0;
    float running_max = std::numeric_limits<float>::lowest();
    float col_sum_inv;
    const int elem_iters = vector_size / SM_VEC_LEN;

    exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

    log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)scale);

    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_log_in++;
        scaled_accum = aie::mul(input_bf16, log2e_vec);
        running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (running_max > max_val) {
            max_val = running_max;
        }
    }

    if (max_val > scale_buffer[row_idx]) {
        scale_buffer[num_rows + row_idx] = max_val;
    } else {
        scale_buffer[num_rows + row_idx] = scale_buffer[row_idx];
        max_val = scale_buffer[row_idx];
    }

    max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_exp_in++;

        scaled_accum = aie::mul(input_bf16, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);

        *it_exp_out++ = exp_val;
    }

    aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);

    scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

    event1();
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output, const int32_t input_size)
{
    softmax_simple_bf16(input, output, input_size);
}

void init_softmax_scale_buffer(bfloat16 *scale_buffer, const int32_t num_rows)
{
    for (int32_t row = 0; row < num_rows; ++row) {
        scale_buffer[row] = std::numeric_limits<bfloat16>::lowest();
        scale_buffer[num_rows + row] = (bfloat16)0.0f;
        scale_buffer[2 * num_rows + row] = (bfloat16)0.0f;
        scale_buffer[3 * num_rows + row] = (bfloat16)0.0f;
    }
}

void copy_softmax_scale_bf16(bfloat16 *restrict input, bfloat16 *restrict output, const int32_t num_elements)
{
    for (int32_t idx = 0; idx < num_elements; ++idx) {
        output[idx] = input[idx];
    }
}

void partial_softmax_rows_bf16(bfloat16 *restrict input,
                               bfloat16 *restrict output,
                               bfloat16 *restrict scale_buffer,
                               const int32_t row_width,
                               const int32_t num_rows)
{
    for (int32_t row = 0; row < num_rows; ++row) {
        partial_softmax_alias_bf16(
            input + row * row_width, output + row * row_width, scale_buffer, row_width, row, num_rows, (bfloat16)log2e);
    }

    for (int32_t row = 0; row < num_rows; ++row) {
        const float m_prev = (float)scale_buffer[row];
        const float m_cur = (float)scale_buffer[num_rows + row];
        const float l_prev = (float)scale_buffer[2 * num_rows + row];
        const float accum_exp_val = (float)scale_buffer[3 * num_rows + row];
        const bfloat16 max_diff_exp =
            aie::reduce_max(aie::exp2<bfloat16>(aie::broadcast<float, SM_VEC_LEN>(m_prev - m_cur)));
        scale_buffer[3 * num_rows + row] = (bfloat16)max_diff_exp;
        scale_buffer[2 * num_rows + row] = (bfloat16)((float)max_diff_exp * l_prev + accum_exp_val);
        scale_buffer[row] = scale_buffer[num_rows + row];
    }
}

void normalize_softmax_rows_bf16(bfloat16 *restrict input,
                                 bfloat16 *restrict scale_buffer,
                                 bfloat16 *restrict output,
                                 const int32_t row_width,
                                 const int32_t num_rows)
{
    for (int32_t row = 0; row < num_rows; ++row) {
        const bfloat16 inv_sum = (bfloat16)aie::inv((float)scale_buffer[2 * num_rows + row]);
        auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input + row * row_width);
        auto it_out = aie::begin_restrict_vector<SM_VEC_LEN>(output + row * row_width);
        const int32_t elem_iters = row_width / SM_VEC_LEN;
        for (int32_t i = 0; i < elem_iters; ++i) {
            aie::vector<bfloat16, SM_VEC_LEN> in_vec = *it_in++;
            auto out_acc = aie::mul(in_vec, aie::broadcast<bfloat16, SM_VEC_LEN>(inv_sum));
            *it_out++ = out_acc.to_vector<bfloat16>();
        }
    }
}

void partial_softmax_bf16(bfloat16 *restrict input,
                          bfloat16 *restrict output,
                          bfloat16 *restrict scale_buffer,
                          const int32_t input_size,
                          const int32_t row_idx,
                          const int32_t num_rows,
                          const bfloat16 scale)
{
    partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx, num_rows, scale);
}

} // extern "C"
