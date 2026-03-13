// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#ifndef VECTOR_LENGTH
#define VECTOR_LENGTH 64
#endif

#define ROUNDING_MODE aie::rounding_mode::conv_even

extern "C" {

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef IS_CAUSAL
#define IS_CAUSAL 1
#endif

static inline void
scale_O_tile_rows(bfloat16 *out, const bfloat16 *scale_buffer, const int32_t scale_offset, const int32_t B_q)
{
    // The PV row-major microkernel emits an output tile laid out as contiguous 8x64 row groups.
    // Walk the number of emitted 8-row groups from B_q instead of assuming a fixed 32-row tile.
    constexpr int32_t rows_per_group = 8;
    constexpr int32_t cols = 64;
    constexpr int32_t tile_elems = rows_per_group * rows_per_group;
    constexpr int32_t col_groups = cols / rows_per_group;
    using Vec8bf16 = aie::vector<bfloat16, rows_per_group>;

    for (int32_t l = 0; l < B_q / rows_per_group; l++) {
        Vec8bf16 scale_row = aie::load_v<rows_per_group>(scale_buffer + scale_offset + l * rows_per_group);
        const int32_t row_group_base = l * rows_per_group * cols;
        for (int32_t k = 0; k < rows_per_group; k++) {
            Vec8bf16 scale_vec = aie::broadcast<bfloat16, rows_per_group>(scale_row[k]);
            for (int32_t j = 0; j < col_groups; j++) {
                const int32_t offset = row_group_base + j * tile_elems + k * rows_per_group;
                Vec8bf16 o_vec = aie::load_v<rows_per_group>(out + offset);
                o_vec = aie::mul(o_vec, scale_vec);
                aie::store_v(out + offset, o_vec);
            }
        }
    }
}

static inline void copy_O_tile_rows(bfloat16 *out, const int32_t B_q)
{
    constexpr int32_t rows_per_group = 8;
    constexpr int32_t cols = 64;
    constexpr int32_t tile_elems = rows_per_group * rows_per_group;
    constexpr int32_t col_groups = cols / rows_per_group;
    using Vec8bf16 = aie::vector<bfloat16, rows_per_group>;

    for (int32_t l = 0; l < B_q / rows_per_group; l++) {
        const int32_t row_group_base = l * rows_per_group * cols;
        for (int32_t k = 0; k < rows_per_group; k++) {
            for (int32_t j = 0; j < col_groups; j++) {
                const int32_t offset = row_group_base + j * tile_elems + k * rows_per_group;
                Vec8bf16 o_vec = aie::load_v<rows_per_group>(out + offset);
                aie::store_v(out + offset, o_vec);
            }
        }
    }
}

void matmul_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
void matmul_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
void partial_softmax_bf16(bfloat16 *input,
                          bfloat16 *output,
                          bfloat16 *scale_buffer,
                          const int32_t input_size,
                          const int32_t row_idx,
                          const int32_t row_size,
                          const bfloat16 scale);
void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth);
void zero_bf16(bfloat16 *buffer);

void matmul_bf16_bf16_wrapper(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t *idx_buffer)
{

    ::aie::set_rounding(ROUNDING_MODE);

#if IS_CAUSAL
    if (idx_buffer[0] > idx_buffer[1]) {
        return;
    }
#endif

    matmul_bf16_bf16(a_in, b_in, c_out);
}

void matmul_bf16_bf16_wrapper_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{

    ::aie::set_rounding(ROUNDING_MODE);
    matmul_scalar_bf16_bf16(a_in, b_in, c_out);
}

void matmul_PV(bfloat16 *Q,
               bfloat16 *K,
               bfloat16 *out,
               bfloat16 *scale_buffer,
               const int32_t B_q,
               int32_t first_iter,
               int32_t *idx_buffer)
{

    ::aie::set_rounding(ROUNDING_MODE);

#if IS_CAUSAL
    if (idx_buffer[0] > idx_buffer[1]) {
        return;
    }
#endif

#if DEBUG == 0 || DEBUG == 1
    // VJUNG: Scale O_{i-1} by 1/exp(m_{i-1} - m_{i}) store in scale_buffer[3*B_q:3*B_q + B_q]
    // VJUNG: Skip this for the first iteration as 1/exp(m_{i-1} - m_{i}) degenerates to inf due to m intizalized to
    // -inf
    if (first_iter != 0) {
        scale_O_tile_rows(out, scale_buffer, 3 * B_q, B_q);
    }
#endif

    matmul_bf16_bf16_rowmaj(Q, K, out);
}

void rescale_O(bfloat16 *O, bfloat16 *scale_buffer, int32_t B_q, int32_t *idx_buffer)
{

    ::aie::set_rounding(ROUNDING_MODE);

#if DEBUG == 0 || DEBUG == 1
    for (int32_t i = 0; i < B_q; i += VECTOR_LENGTH) {
        using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
        Vec64bf16 l_vec = aie::load_v<VECTOR_LENGTH>(scale_buffer + 2 * B_q + i);
        l_vec = aie::inv(l_vec);
        aie::store_v(scale_buffer + 2 * B_q + i, l_vec);
    }

    // VJUNG: Only after all KV are processed
    // VJUNG: Scale O_{i} by 1/l_{i}
    scale_O_tile_rows(O, scale_buffer, 2 * B_q, B_q);
#else
    // In debug mode, just copy input to output
    copy_O_tile_rows(O, B_q);
#endif
}

void partial_softmax(bfloat16 *A,
                     bfloat16 *P,
                     bfloat16 *scale_buffer,
                     int32_t *idx_buffer,
                     bfloat16 inv_scale,
                     int32_t B_q,
                     int32_t B_kv,
                     int32_t S_q_eff,
                     int32_t S_kv_eff)
{

    ::aie::set_rounding(ROUNDING_MODE);

    // Block indices
    int32_t q_block_idx = idx_buffer[1];
    int32_t kv_block_idx = idx_buffer[0];

    // Causal full mask: skip blocks strictly above diagonal
#if IS_CAUSAL
    if (kv_block_idx > q_block_idx) {
        zero_bf16(P);
        return;
    }
#endif

    // Compute valid extents within this block for padded tails
    int32_t valid_q_rows = S_q_eff - q_block_idx * B_q;
    if (valid_q_rows < 0)
        valid_q_rows = 0;
    if (valid_q_rows > B_q)
        valid_q_rows = B_q;

    int32_t valid_kv_cols = S_kv_eff - kv_block_idx * B_kv;
    if (valid_kv_cols < 0)
        valid_kv_cols = 0;
    if (valid_kv_cols > B_kv)
        valid_kv_cols = B_kv;

    // Fully padded block: contributes nothing
    if (valid_q_rows == 0 || valid_kv_cols == 0) {
        zero_bf16(P);
        return;
    }

#if DEBUG == 0 || DEBUG == 1
    // Tail mask: invalidate padded Q rows
    if (valid_q_rows < B_q) {
        using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
        Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(std::numeric_limits<bfloat16>::lowest());

        for (int32_t i = valid_q_rows; i < B_q; i++) {
            for (int32_t j = 0; j < B_kv; j += VECTOR_LENGTH) {
                aie::store_v(A + i * B_kv + j, lowest_vec);
            }
        }
    }
    // Tail mask: invalidate padded KV cols for valid rows
    if (valid_kv_cols < B_kv) {
        using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
        Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(std::numeric_limits<bfloat16>::lowest());

        for (int32_t i = 0; i < valid_q_rows; i++) {
            int32_t j = valid_kv_cols;
            for (; j + VECTOR_LENGTH <= B_kv; j += VECTOR_LENGTH) {
                aie::store_v(A + i * B_kv + j, lowest_vec);
            }
            // Remainder loop
            for (; j < B_kv; j++) {
                A[i * B_kv + j] = std::numeric_limits<bfloat16>::lowest();
            }
        }
    }

#if IS_CAUSAL
    // Diagonal small causal mask only within valid region (vectorized)
    if (kv_block_idx == q_block_idx) {
        using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
        Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(std::numeric_limits<bfloat16>::lowest());
        for (int32_t i = 0; i < valid_q_rows; i++) {
            int32_t j = i + 1;
            if (j < valid_kv_cols) {
                // Vectorized stores for upper triangle within valid_kv_cols
                for (; j + VECTOR_LENGTH <= valid_kv_cols; j += VECTOR_LENGTH) {
                    aie::store_v(A + i * B_kv + j, lowest_vec);
                }
                // Remainder
                for (; j < valid_kv_cols; ++j) {
                    A[i * B_kv + j] = std::numeric_limits<bfloat16>::lowest();
                }
            }
        }
    }
#endif

    using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
    int32_t i = 0;
    for (; i + 4 <= valid_q_rows; i += 4) {
        partial_softmax_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv, i, B_q, inv_scale);
        partial_softmax_bf16(A + B_kv * (i + 1), P + B_kv * (i + 1), scale_buffer, B_kv, i + 1, B_q, inv_scale);
        partial_softmax_bf16(A + B_kv * (i + 2), P + B_kv * (i + 2), scale_buffer, B_kv, i + 2, B_q, inv_scale);
        partial_softmax_bf16(A + B_kv * (i + 3), P + B_kv * (i + 3), scale_buffer, B_kv, i + 3, B_q, inv_scale);
    }
    for (; i < valid_q_rows; i++) {
        partial_softmax_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv, i, B_q, inv_scale);
    }
    // Zero out P rows corresponding to padded Q rows
    if (valid_q_rows < B_q) {
        using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
        Vec64bf16 zeros_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(0.0f);

        for (int32_t i = valid_q_rows; i < B_q; i++) {
            for (int32_t j = 0; j < B_kv; j += VECTOR_LENGTH) {
                aie::store_v(P + i * B_kv + j, zeros_vec);
            }
        }
    }

    for (int32_t i = 0; i < B_q; i += VECTOR_LENGTH) {

        Vec64bf16 m_i_minus_1 = aie::load_v<VECTOR_LENGTH>(scale_buffer + i);
        Vec64bf16 m_i = aie::load_v<VECTOR_LENGTH>(scale_buffer + B_q + i);
        Vec64bf16 l_i_minus_1 = aie::load_v<VECTOR_LENGTH>(scale_buffer + 2 * B_q + i);
        Vec64bf16 accum_exp_val = aie::load_v<VECTOR_LENGTH>(scale_buffer + 3 * B_q + i);

        aie::accum<accfloat, VECTOR_LENGTH> l_i_accum = aie::zeros<accfloat, VECTOR_LENGTH>();

        aie::accum<accfloat, VECTOR_LENGTH> diff = aie::accum<accfloat, VECTOR_LENGTH>(aie::sub(m_i_minus_1, m_i));
        l_i_accum = aie::exp2<bfloat16>(diff.to_vector<float>());
        Vec64bf16 max_diff_exp = l_i_accum.to_vector<bfloat16>();

        aie::store_v(scale_buffer + 3 * B_q + i, max_diff_exp);
        aie::accum<accfloat, VECTOR_LENGTH> l_i = aie::add(aie::mul(max_diff_exp, l_i_minus_1), accum_exp_val);
        aie::store_v(scale_buffer + 2 * B_q + i, l_i.to_vector<bfloat16>());
        aie::store_v(scale_buffer + i, m_i);
    }
#else
    // In debug mode, just copy input to output
    using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
    int32_t i = 0;
    for (; i + 4 <= valid_q_rows; i += 4) {
        for (int j = 0; j < B_kv; j += VECTOR_LENGTH) {
            Vec64bf16 a_vec0 = aie::load_v<VECTOR_LENGTH>(A + (i + 0) * B_kv + j);
            Vec64bf16 a_vec1 = aie::load_v<VECTOR_LENGTH>(A + (i + 1) * B_kv + j);
            Vec64bf16 a_vec2 = aie::load_v<VECTOR_LENGTH>(A + (i + 2) * B_kv + j);
            Vec64bf16 a_vec3 = aie::load_v<VECTOR_LENGTH>(A + (i + 3) * B_kv + j);
            aie::store_v(P + (i + 0) * B_kv + j, a_vec0);
            aie::store_v(P + (i + 1) * B_kv + j, a_vec1);
            aie::store_v(P + (i + 2) * B_kv + j, a_vec2);
            aie::store_v(P + (i + 3) * B_kv + j, a_vec3);
        }
    }
    for (; i < valid_q_rows; i++) {
        for (int j = 0; j < B_kv; j += VECTOR_LENGTH) {
            Vec64bf16 a_vec = aie::load_v<VECTOR_LENGTH>(A + i * B_kv + j);
            aie::store_v(P + i * B_kv + j, a_vec);
        }
    }
#endif
}

void init_scale_buffer(bfloat16 *scale_buffer, int32_t size)
{
    ::aie::set_rounding(ROUNDING_MODE);

    using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
    Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(std::numeric_limits<bfloat16>::lowest());
    Vec64bf16 zeros_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(0.0f);

    for (int32_t i = 0; i < size; i += VECTOR_LENGTH) {
        // VJUNG: m_{i-1} vector
        aie::store_v(scale_buffer + i, lowest_vec);
        // VJUNG: m_{i} vector
        aie::store_v(scale_buffer + size + i, zeros_vec);
        // VJUNG: l_{i} vector
        aie::store_v(scale_buffer + 2 * size + i, zeros_vec);
    }
}
}
